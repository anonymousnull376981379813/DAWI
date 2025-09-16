# Metric computation code from Open-Unlearning src/eval/memorization.py
# Certain metrics like tokenwise vocab logprobs, dict_transpose, run_batchwise_evals from Open-Unlearning src/eval/utils.py
# Minor changes made for running properly outside of the library

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import transformers
from typing import Sequence, Dict, Union, List, Any
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
from sklearn.metrics import roc_auc_score
from enum import Enum

IGNORE_INDEX = -100

def tokenwise_logprobs(model, batch, grad=False, return_labels=False):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            print(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch
def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts

def load_hf_dataset(path, **kwargs):
    
    dataset = datasets.load_dataset(path, **kwargs)
    print(path, kwargs)
    return dataset

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column("index", indexing)
    return dataset

def preprocess_chat_instance(
    tokenizer,
    template_config: Dict[str, Any],
    prompt_msgs: Union[List[str], str],
    response_msgs: Union[List[str], str],
    max_length: int,
    predict_with_generate: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocesses a chat instance for training or generation.
    When in training, both the returned `input_ids` and `labels` cover the entire conversation.
    `input_ids` has no padding, and `labels` assign `IGNORE_INDEX` to tokens where loss is not computed (i.e. all tokens except the final response message).
    When in generation, `input_ids` are returned only up to the last user prompt, excluding the assistant's response. The `labels` returned are the same as during training.
    `attention_mask` is always 1 over the full `input_ids` token sequence.

    `prompt_msgs` and `response_msgs` are lists where, except for the last pair, all
    corresponding pairs are in-context examples. When they are a string and not
    a list, there are no in-context examples.

    Args:
        tokenizer: Tokenizer to apply on text
        template_config (Dict[str, Any]): Configuration for the chat template (comes from model-specific config).
        prompt_msgs (Union[List[str], str]): List of prompt messages or a single prompt message string.
        response_msgs (Union[List[str], str]): List of response messages or a single response message string.
        max_length (int): Maximum sequence length after tokenization.
        predict_with_generate (bool, optional): Whether to prepare inputs for generation.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 'labels', and 'attention_mask' tensors for model input.
    """
    assert len(prompt_msgs) == len(response_msgs)
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    if template_config["apply_chat_template"]:
        chat = []
        system_prompt = template_config.get("system_prompt", None)
        if system_prompt:
            chat += [{"role": "system", "content": system_prompt}]
        for prompt, response in zip(prompt_msgs, response_msgs):
            chat += [{"role": "user", "content": prompt}]
            chat += [{"role": "assistant", "content": response}]
        date_str = template_config.get("date_string", None)
        date_info = {"date_string": date_str} if date_str is not None else {}
        chat_ids = tokenizer.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=False, **date_info
        )
        # all except last response are in-context examples
        wrapped_prompt = tokenizer.apply_chat_template(
            chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
        )
        prompt_ids = tokenizer.apply_chat_template(
            chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
        )
    else:
        wrapped_prompt = ""
        system_prompt_with_special_tokens = template_config.get(
            "system_prompt_with_special_tokens", None
        )
        if system_prompt_with_special_tokens:
            wrapped_prompt += system_prompt_with_special_tokens
        # add in-context examples
        n_few_shot = len(prompt_msgs) - 1
        for i in range(n_few_shot):
            fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
            wrapped_prompt += (
                template_config["user_start_tag"]
                + fs_prompt
                + template_config["user_end_tag"]
                + template_config["asst_start_tag"]
                + fs_response
                + template_config["asst_end_tag"]
            )

        # add actual example
        final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + final_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
        )
        chat_ids = tokenizer(
            wrapped_prompt + final_response,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

        prompt_ids = tokenizer(
            wrapped_prompt,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)

    item = {}
    if predict_with_generate:
        item["input_ids"] = prompt_ids
        labels = chat_ids  # contains the entire conversation
    else:
        item["input_ids"] = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]
        if len(prompt_ids) == len(chat_ids):
            # Rarely, tokenization can result in this condition being entered.
            # Say a input prompt is ABC and target output is D, tokenizer(ABCD)
            # can be [AB, CD] and tokenizer(ABC) can be [AB, C]. In this case,
            # we ignore loss on all indices in the labels. So, there is no way
            # to use this for next token prediction. Be careful while
            # interpreting results of such instances.
            print(
                "Tokenization mismatch: no valid target tokens for loss computation"
            )

    item["labels"] = labels
    item["attention_mask"] = [1] * len(item["input_ids"])
    for attr in item:
        item[attr] = torch.tensor(item[attr])
    return item


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
 
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        print(item)
        exit()
        return item

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        padding_side: str = "right",
        index: str = None,
    ):
        self.tokenizer = tokenizer
        self.padding_side = padding_side
        self.index = index

    def get_instances_from_key(self, instances: Sequence[Dict], key: str):
        ret_instances = [instance[key] for instance in instances]
        return ret_instances

    def _pad_tokens(self, input_ids, padding_value):
        if self.padding_side == "right":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=padding_value
            )
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.flip(i, dims=[0]) for i in input_ids],
                batch_first=True,
                padding_value=padding_value,
            ).flip(dims=[1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert isinstance(instances[0], dict), type(instances[0])
        return_dct = {}
        if "input_ids" not in instances[0]:
            for key in instances[0].keys():
                key_instances = self.get_instances_from_key(
                    instances=instances, key=key
                )
                return_dct[key] = self(key_instances)
        else:
            input_ids = [instance["input_ids"] for instance in instances]
            input_ids = self._pad_tokens(input_ids, self.tokenizer.pad_token_id)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
            return_dct.update({"input_ids": input_ids})
            return_dct.update({"attention_mask": attention_mask})
            if "labels" in instances[0]:
                labels = [instance["labels"] for instance in instances]
                labels = self._pad_tokens(labels, IGNORE_INDEX)
                return_dct.update({"labels": labels})
            if self.index:
                if self.index in instances[0]:
                    return_dct.update(
                        {
                            self.index: torch.tensor(
                                [example[self.index] for example in instances]
                            )
                        }
                    )
                else:
                    raise Warning(f"{self.index} not found in dataset")
        return return_dct


def dict_transpose(evals):
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


def tokenwise_vocab_logprobs(model, batch, grad=False, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            print(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))

def run_batchwise_evals(model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    evals = defaultdict(dict)
    for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):

        # if data arrives in normal format we convert the batch to multiple answer-style
        # like in tofu_perturbed by adding a fake intra_item_index
        if "input_ids" in batch:
            batch = {"0": batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]},
        #                    "1": {"input_ids": [[]]..., "index": [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for intra_item_idx, mini_batch in batch.items():
            data_indices = (
                mini_batch.pop("index").cpu().float().numpy().tolist()
            )  # data item indices
            batch_evals = batch_eval_fn(
                model=model, batch=mini_batch, **batch_eval_fn_args
            )
            indexwise_batch_evals = dict(zip(data_indices, batch_evals))
            assert not (
                evals[intra_item_idx].keys() & indexwise_batch_evals.keys()
            ), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    if len(evals) == 1:  # normal single answer dataset, no need for list
        evals = next(iter(evals.values()))
    else:
        # for each index return a dict with all intra_item_idx values in list
        # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
        evals = dict_transpose(evals)
    print("Evaluated", len(evals), "examples")
    return evals

def extraction_strength(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
    def _extraction_strength(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                print(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        return es_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _extraction_strength, fun_args, "Calculating ES"
    )
    es_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    es_values = aggregate_to_1D(es_values)
    return {"agg_value": np.mean(es_values), "value_by_index": scores_by_index}

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
class ForgetDataset(Dataset):
        def __init__(self):
            self.ds = datasets.load_dataset("Anonymous0192830/MUD-Qwen2.5-Math-1.5B-Instruct")["train"]
            
        def __len__(self):
            return len(self.ds)
        
        def __getitem__(self, index) -> Any:
            
            data = self.ds[index]
            forget_data = data["forget"]
            question = forget_data["question"]
            answer = forget_data["answer"]
            prompt_length = get_prompt_len(tokenizer, question)
            messages = [{"role": "assistant", "content": question}, {"role": "user", "content": answer}]
            tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            labels = tokens["input_ids"].clone()
            labels[:, :prompt_length] = -100
            tokens["index"] = index
            tokens["labels"] = labels
            tokens["labels"] = tokens["labels"].squeeze()
            tokens["attention_mask"] = tokens["attention_mask"].squeeze()
            tokens["input_ids"] = tokens["input_ids"].squeeze()
            return dict(tokens)

class Evaluator():
    def __init__(self, tokenizer, dataset_name):
        self.data = ForgetDataset()

    #data = QADataset({'name': 'forget10_perturbed', 'split': 'train', 'path': 'locuslab/TOFU'}, {'apply_chat_template': True, 'system_prompt': 'Please reason step by step, and put your final answer within \\boxed{}.', 'system_prompt_with_special_tokens': '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>', 'user_start_tag': '<|im_start|>user\n', 'user_end_tag': '<|im_end|>\n', 'asst_start_tag': '<|im_start|>assistant\n', 'asst_end_tag': '<|im_end|>\n', 'date_string': '10 Apr 2025'}, tokenizer)
        self.collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
    
    def eval(self, model):
        with torch.no_grad():
            return extraction_strength(model, **{"data": self.data, "collators": self.collator, "batch_size": 8})["agg_value"]
        
def get_prompt_len(tokenizer, question):
    messages = [
        {"role": "user", "content": question},
    ]
    length = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )["input_ids"].shape[-1]
    return length

def evaluate_probability(model, batch):
    """Evaluate model probabilities and average token-level loss for a given batch."""
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    # agg loss across tokens
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().float().numpy().tolist()
    normalized_probs = normalized_probs.cpu().float().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]

        
        
def probability(model, **kwargs):
    """Compute the probabilities by data points and report aggregated average"""
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, evaluate_probability, fun_args, "Calculating loss"
    )
    prob_values = np.array(
        [
            evals["prob"]
            for evals in scores_by_index.values()
            if evals["prob"] is not None
        ]
    )
    prob_values = aggregate_to_1D(prob_values)
    return {"agg_value": np.mean(prob_values), "value_by_index": scores_by_index}


def exact_memorization(model, **kwargs):
    data = kwargs["data"]
    collator = kwargs["collators"]
    batch_size = kwargs["batch_size"]
    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)

    def _exact_memorization(model, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(
            model, batch, grad=False, return_labels=True
        )
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                print(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                em_batch.append({"score": None})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})
        return em_batch

    fun_args = {}
    scores_by_index = run_batchwise_evals(
        model, dataloader, _exact_memorization, fun_args, "Calculating EM"
    )
    em_values = np.array(
        [
            evals["score"]
            for evals in scores_by_index.values()
            if evals["score"] is not None
        ]
    )
    em_values = aggregate_to_1D(em_values)
    return {"agg_value": np.mean(em_values), "value_by_index": scores_by_index}

def truth_ratio(model, **kwargs):
    """Compute the truth ratio, aggregating false/true scores, and
    return the aggregated value."""

    # Forget data: It is better if false and true are equally likely,
    # i.e., tr=false/true is closest to 1.
    def closer_to_1_better(arr):
        return np.mean(np.minimum(arr, 1 / (arr + 1e-10)))

    # Non-forget data: It is better if tr=false/true is lower, i.e.,
    # 1-tr is higher.
    def true_better(arr):
        return np.mean(np.maximum(0, 1 - arr))

    if kwargs["aggregator"] == "closer_to_1_better":
        aggregator = closer_to_1_better
    elif kwargs["aggregator"] == "true_better":
        aggregator = true_better
    else:
        raise ValueError(f"Invalid truth ratio aggregator: {kwargs['aggregator']}")

    correct_answer_results = kwargs["pre_compute"]["correct"]["value_by_index"]
    wrong_answer_results = kwargs["pre_compute"]["wrong"]["value_by_index"]

    correct_indices = list(correct_answer_results.keys())
    wrong_indices = list(wrong_answer_results.keys())
    assert correct_indices == wrong_indices

    # Filter out None values from both correct and wrong answers
    filtered_indices = [
        idx
        for idx in correct_indices
        if correct_answer_results[idx] is not None
        and wrong_answer_results[idx] is not None
    ]
    correct_avg_losses = [
        correct_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]
    wrong_avg_losses = [
        wrong_answer_results[idx]["avg_loss"] for idx in filtered_indices
    ]

    correct_avg_losses = aggregate_to_1D(np.array(correct_avg_losses))
    wrong_avg_losses = aggregate_to_1D(np.array(wrong_avg_losses))

    correct_prob = np.exp(-correct_avg_losses)
    wrong_prob = np.exp(-wrong_avg_losses)

    truth_ratios = wrong_prob / (correct_prob + 1e-10)
    value_by_index = dict(
        zip(correct_indices, [{"score": val} for val in truth_ratios])
    )
    truth_ratio_stats = np.array([evals["score"] for evals in value_by_index.values()])
    forget_tr_avg = aggregator(truth_ratio_stats)
    return {"agg_value": forget_tr_avg, "value_by_index": value_by_index}

### PRIVACY METRICS

def mia_auc(attack_cls, model, data, collator, batch_size, **kwargs):
    """
    Compute the MIA AUC and accuracy.

    Parameters:
      - attack_cls: the attack class to use.
      - model: the target model.
      - data: a dict with keys "forget" and "holdout".
      - collator: data collator.
      - batch_size: batch size.
      - kwargs: additional optional parameters (e.g. k, p, tokenizer, reference_model).

    Returns a dict containing the attack outputs, including "acc" and "auc".

    Note on convention: auc is 1 when the forget data is much more likely than the holdout data
    """
    # Build attack arguments from common parameters and any extras.
    attack_args = {
        "model": model,
        "collator": collator,
        "batch_size": batch_size,
    }
    attack_args.update(kwargs)

    output = {
        "forget": attack_cls(data=data["forget"], **attack_args).attack(),
        "holdout": attack_cls(data=data["holdout"], **attack_args).attack(),
    }
    forget_scores = [
        elem["score"] for elem in output["forget"]["value_by_index"].values()
    ]
    holdout_scores = [
        elem["score"] for elem in output["holdout"]["value_by_index"].values()
    ]
    scores = np.array(forget_scores + holdout_scores)
    labels = np.array(
        [0] * len(forget_scores) + [1] * len(holdout_scores)
    )  # see note above
    auc_value = roc_auc_score(labels, scores)
    output["auc"], output["agg_value"] = auc_value, auc_value
    return output

class AllAttacks(str, Enum):
    LOSS = "loss"
    REFERENCE_BASED = "ref"
    ZLIB = "zlib"
    MIN_K = "min_k"
    MIN_K_PLUS_PLUS = "min_k++"
    GRADNORM = "gradnorm"
    RECALL = "recall"


# Base attack class
class Attack:
    def __init__(self, model, data, collator, batch_size, **kwargs):
        """Initialize attack with model and create dataloader."""
        self.model = model
        self.dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collator)
        self.setup(**kwargs)

    def setup(self, **kwargs):
        """Setup attack-specific parameters."""
        pass

    def compute_batch_values(self, batch):
        """Process a batch through model to get needed statistics."""
        raise NotImplementedError

    def compute_score(self, sample_stats):
        """Compute MIA score for a single sample."""
        raise NotImplementedError

    def attack(self):
        """Run full MIA attack."""
        all_scores = []
        all_indices = []

        for batch in tqdm(self.dataloader, total=len(self.dataloader)):
            indices = batch.pop("index").cpu().float().numpy().tolist()
            batch_values = self.compute_batch_values(batch)
            scores = [self.compute_score(values) for values in batch_values]

            all_scores.extend(scores)
            all_indices.extend(indices)

        scores_by_index = {
            str(idx): {"score": float(score)}
            for idx, score in zip(all_indices, all_scores)
        }

        return {
            "agg_value": float(np.mean(all_scores)),
            "value_by_index": scores_by_index,
        }
        
class LOSSAttack(Attack):
    def compute_batch_values(self, batch):
        """Compute probabilities and losses for the batch."""
        return evaluate_probability(self.model, batch)

    def compute_score(self, sample_stats):
        """Return the average loss for the sample."""
        return sample_stats["avg_loss"]
def mia_loss(model, **kwargs):
    return mia_auc(
        LOSSAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
    )
    

class MinKProbAttack(Attack):
    def setup(self, k=0.2, **kwargs):
        self.k = k

    def compute_batch_values(self, batch):
        """Get token-wise log probabilities for the batch."""
        return tokenwise_logprobs(self.model, batch, grad=False)

    def compute_score(self, sample_stats):
        """Score single sample using min-k negative log probs scores attack."""
        lp = sample_stats.cpu().float().numpy()
        if lp.size == 0:
            return 0

        num_k = max(1, int(len(lp) * self.k))
        sorted_vals = np.sort(lp)
        return -np.mean(sorted_vals[:num_k])
def mia_min_k(model, **kwargs):
    return mia_auc(
        MinKProbAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )
    

class MinKPlusPlusAttack(MinKProbAttack):
    def compute_batch_values(self, batch):
        """Get both token-wise and vocab-wise log probabilities for the batch."""
        vocab_log_probs = tokenwise_vocab_logprobs(self.model, batch, grad=False)
        token_log_probs = tokenwise_logprobs(self.model, batch, grad=False)
        return [
            {"vocab_log_probs": vlp, "token_log_probs": tlp}
            for vlp, tlp in zip(vocab_log_probs, token_log_probs)
        ]

    def compute_score(self, sample_stats):
        """Score using min-k negative log probs scores with vocab-wise normalization."""
        all_probs = sample_stats["vocab_log_probs"]
        target_prob = sample_stats["token_log_probs"]

        if len(target_prob) == 0:
            return 0

        # Compute normalized scores using vocab distribution
        mu = (torch.exp(all_probs) * all_probs).sum(-1)
        sigma = (torch.exp(all_probs) * torch.square(all_probs)).sum(-1) - torch.square(
            mu
        )

        # Handle numerical stability
        sigma = torch.clamp(sigma, min=1e-6)
        scores = (target_prob.cpu().float().numpy() - mu.cpu().float().numpy()) / torch.sqrt(
            sigma
        ).cpu().float().numpy()

        # Take bottom k% as the attack score
        num_k = max(1, int(len(scores) * self.k))
        return -np.mean(sorted(scores)[:num_k])
def mia_min_k_plus_plus(model, **kwargs):
    return mia_auc(
        MinKPlusPlusAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        k=kwargs["k"],
    )
import zlib
class ZLIBAttack(Attack):
    def setup(self, tokenizer=None, **kwargs):
        """Setup tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct") # CHECK

    def compute_batch_values(self, batch):
        """Get loss and text for batch."""
        eval_results = evaluate_probability(self.model, batch)
        texts = extract_target_texts_from_processed_data(self.tokenizer, batch)
        return [{"loss": r["avg_loss"], "text": t} for r, t in zip(eval_results, texts)]

    def compute_score(self, sample_stats):
        """Score using loss normalized by compressed text length."""
        text = sample_stats["text"]
        zlib_entropy = len(zlib.compress(text.encode("utf-8")))
        return sample_stats["loss"] / zlib_entropy

    
def mia_zlib(model, **kwargs):
    return mia_auc(
        ZLIBAttack,
        model,
        data=kwargs["data"],
        collator=kwargs["collators"],
        batch_size=kwargs["batch_size"],
        tokenizer=kwargs.get("tokenizer"),
    )

class HoldoutDataset(Dataset):
    def __init__(self):
        self.ds = datasets.load_dataset("Anonymous0192830/MUD20k-holdout")["train"]
        
    def __len__(self):
        return len(self.ds) // 2
    
    def __getitem__(self, index) -> Any:
        data = self.ds[index]
        forget_data = data
        question = forget_data["question"]
        answer = forget_data["answer"]
        prompt_length = get_prompt_len(tokenizer, question)
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[:, :prompt_length] = -100
        tokens["index"] = index
        tokens["labels"] = labels
        tokens["labels"] = tokens["labels"].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"].squeeze()
        tokens["input_ids"] = tokens["input_ids"].squeeze()
        return dict(tokens)

class ForgetDataset(Dataset):
    def __init__(self):
        self.ds = datasets.load_dataset("Anonymous0192830/MUD20k")["train"]
        
    def __len__(self):
        return len(self.ds) // 2
    
    def __getitem__(self, index) -> Any:
        index = index * 2
        data = self.ds[index]
        forget_data = data["forget"]
        question = forget_data["question"]
        answer = forget_data["answer"]
        prompt_length = get_prompt_len(tokenizer, question)
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[:, :prompt_length] = -100
        tokens["index"] = index
        tokens["labels"] = labels
        tokens["labels"] = tokens["labels"].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"].squeeze()
        tokens["input_ids"] = tokens["input_ids"].squeeze()
        return dict(tokens)

class RetainDataset(Dataset):
    def __init__(self):
        self.ds = datasets.load_dataset("Anonymous0192830/MUD-Qwen2.5-Math-1.5B-Instruct")["train"]
        
    def __len__(self):
        return len(self.ds) // 2
    
    def __getitem__(self, index) -> Any:
        index = index * 2 + 1
        data = self.ds[index]
        forget_data = data["forget"]
        question = forget_data["question"]
        answer = forget_data["answer"]
        prompt_length = get_prompt_len(tokenizer, question)
        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        tokens = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        labels = tokens["input_ids"].clone()
        labels[:, :prompt_length] = -100
        tokens["index"] = index
        tokens["labels"] = labels
        tokens["labels"] = tokens["labels"].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"].squeeze()
        tokens["input_ids"] = tokens["input_ids"].squeeze()
        return dict(tokens)

data = ForgetDataset()
retain_data = RetainDataset()
holdout_data = HoldoutDataset()
collator = DataCollatorForSupervisedDataset(tokenizer, index="index")
priv_kwargs = {"data": {"forget": data, "holdout": holdout_data}, "collators": collator, "batch_size": 32, "k": 0.4}
mem_kwargs = {"data": data, "collators": collator, "batch_size": 32}
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
    '''for element in data:
        print(tokenizer.decode(element["input_ids"]))
    exit()'''
    kwargs = priv_kwargs
    device = "cuda"
    
    
    tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hello2"}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )["input_ids"]
    
    

    

    model_path = "/mnt/t9/mud_saves/SMALL_STEP_DAWI10"
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype = torch.bfloat16, device_map = device)
    print(model_path)
    model = model.to(torch.float32)
    model.eval()
    #retain_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", torch_dtype = torch.bfloat16, device_map = device)

    
    
    retain_score = extraction_strength(model, **{"data": retain_data, "collators": collator, "batch_size": 32})["agg_value"]
    print("Retain Score", retain_score)
    #data = QADataset({'name': 'forget10_perturbed', 'split': 'train', 'path': 'locuslab/TOFU'}, {'apply_chat_template': True, 'system_prompt': 'Please reason step by step, and put your final answer within \\boxed{}.', 'system_prompt_with_special_tokens': '<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>', 'user_start_tag': '<|im_start|>user\n', 'user_end_tag': '<|im_end|>\n', 'asst_start_tag': '<|im_start|>assistant\n', 'asst_end_tag': '<|im_end|>\n', 'date_string': '10 Apr 2025'}, tokenizer)

    
    s_LOSS = 1 - 2 * abs(mia_loss(model, **kwargs)["agg_value"] - 0.5)
    print(s_LOSS)
    s_ZLib = 1 - 2 * abs(mia_zlib(model, **kwargs)["agg_value"] - 0.5)
    print(s_ZLib)
    s_Min_k = 1 - 2 * abs(mia_min_k(model, **kwargs)["agg_value"] - 0.5)
    print(s_Min_k)
    s_Min_k_plus_plus = 1 - 2 * abs(mia_min_k_plus_plus(model, **kwargs)["agg_value"] - 0.5)
    print(s_Min_k_plus_plus)
    priv_score = 4 / (1/s_LOSS + 1/s_ZLib + 1/s_Min_k + 1/s_Min_k_plus_plus)
    print(priv_score)
     
     
     
     
    
    
    
    #TR = 1 - truth_ratio(model, **{"data": data, "collators": collator, "batch_size": 2, 'aggregator': 'closer_to_1_better'})["agg_value"]
    ES = 1 - extraction_strength(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"]
    print(ES)
    EM = 1 - exact_memorization(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"]
    
    Prob = 1 - probability(model, **{"data": data, "collators": collator, "batch_size": 32})["agg_value"]
    memorization = 3 / (1/EM + 1/ES + 1/Prob)
    print(memorization)
    
    
    