import torch
from transformers import AutoModelForCausalLM,  AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import calculate_sequence_prob
from datasets import load_dataset
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

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

overtrained_model = AutoModelForCausalLM.from_pretrained("Anonymous0192830/MUD20k-finetune", torch_dtype = torch.float32, device_map = device)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype = torch.float32, device_map = device)
ds = load_dataset("Anonymous0192830/MUD-Qwen2.5-Math-1.5B-Instruct")["train"]

save_dataset = []
with torch.no_grad():
    for i in range(0, len(ds), 2):
        retain = ds[i + 1]["forget"]
        
        retain_padded = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": retain["question"]}, {"role": "user", "content" : retain["answer"]}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)["input_ids"]
        forget = ds[i]["forget"]
        forget_padded = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": forget["question"]}, {"role": "user", "content" : forget["answer"]}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)["input_ids"]
        retain_cutoffs = get_prompt_len(tokenizer, retain["question"])
        forget_cutoffs = get_prompt_len(tokenizer, forget["question"])

        forget_logits = base_model.forward(forget_padded).logits
        forget_logits = forget_logits[:, :-1]
        forget_probs = F.softmax(forget_logits, dim = -1)
        vocab_size = forget_logits.shape[-1]
        forget_oh = F.one_hot(forget_padded[:, 1:], num_classes=vocab_size).to(forget_probs.dtype)
                #log_probs = torch.log(probs + 1e-12)
        forget_prob_vec  = (forget_oh * forget_probs).sum(dim=-1) # in shape of [1, seq] 
        
        retain_logits = overtrained_model.forward(retain_padded).logits[:, :-1]
        retain_probs = F.softmax(retain_logits, dim = -1)
        retain_oh =  F.one_hot(retain_padded[:, 1:], num_classes=vocab_size).to(forget_probs.dtype)

        retain_prob_vec  = (retain_oh * retain_probs).sum(dim=-1) # in shape of [1, seq] 
        #print(retain_prob_vec)
        
        print(torch.prod(retain_prob_vec[:, retain_cutoffs-1:]).item(), calculate_sequence_prob(overtrained_model, retain_padded, retain_cutoffs))
        print(torch.prod(forget_prob_vec[:, retain_cutoffs-1:]).item(), calculate_sequence_prob(base_model, forget_padded, forget_cutoffs))

        data =  {"forget": {"content": forget_padded.squeeze(), "cutoff": forget_cutoffs, "basemodel_lower_bound_probs": forget_prob_vec}, 
                "retain": {"content": retain_padded.squeeze(), "cutoff": retain_cutoffs, "overtrained_upper_bound_probs": retain_prob_vec}}
        save_dataset.append(data)
    torch.save(save_dataset, "dataset_changed.pt")    
    