import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from typing import Tuple
from dataclasses import dataclass
from typing import Tuple, List
import heapq
import torch
from mud import BetterDataset
import time
import sys
import wandb
from typing import Dict
from torch.nn.utils.rnn import pad_sequence
from utils import calculate_sequence_prob
from metrics import Evaluator
from forget_eval import mia_loss, priv_kwargs, extraction_strength, mem_kwargs

LOSS_IMPLEMENTATION = "default"
hyperparams = {
    "beta": 2.5,
    "gamma": 0.0,
    "lambda": 0.15,
}

from torch.utils.data import DataLoader




def compute_diff(A, B, diff, grad):
    mask = diff * grad > 0 # If true, a += 1, so basemodel votes += 1
    alpha = A/(A+B)
    divisor = A + B + 1
    dAlpha = torch.where(mask,
                (A + 1) / divisor - alpha,
                - (alpha - (B + 1) / divisor)) # This is for convenience, since dLoss always > 0 if we don't have this negative sign
    # The negative lets us know whether to increment A or B
    dLoss = dAlpha * grad * diff
    #print("HERE2", dLoss, grad)
    return dLoss


def global_topk(
    model,
    base_parameters: List[torch.Tensor],
    overtrained_parameters: List[torch.Tensor],
    ratios: List[dict],
    *,
    num_replacements: int = 10_000,
    device: torch.device | str = "cuda",
):
    candidates_abs_vals:   List[torch.Tensor] = []

    k_local = num_replacements

    with torch.no_grad():
        for layer_id, (param, base, overtrained, votes) in enumerate(
            zip(model.parameters(), base_parameters, overtrained_parameters, ratios)
        ):
            if param.grad is None or param.ndim != 2 or layer_id == 0:
                continue

            A     = votes["A"].float().to(device)
            B     = votes["B"].float().to(device)
            diff  = base.to(device) - overtrained.to(device)
            dLoss = compute_diff(A, B, diff, param.grad)

            flat = dLoss.reshape(-1)
            k = min(k_local, flat.numel())

            abs_vals, _ = torch.topk(flat.abs(), k, largest=True, sorted=False)

            candidates_abs_vals.append(abs_vals)

        all_vals    = torch.cat(candidates_abs_vals)
        
        k_global = min(num_replacements, all_vals.numel())
        vals, idx = torch.topk(all_vals.abs(), k_global, largest=True, sorted=True)
        max_val = vals[0]
        smallest = vals[-1]
        return max_val, smallest
num_replacements = 1_000_000 # We reduce this number for the cutoff because the model has too many parameters to compute 10 million
# To compensate, we divide by 10 instead of 5

def subseq_prob(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, return_log = False):
    cumsum = torch.cumsum(A, dim=1)
    start = B[:, 0]
    end   = B[:, 1] - 1
    
    batch_idx = torch.arange(A.size(0), device=A.device)

    log_end   = cumsum[batch_idx, end]

    log_start = torch.zeros_like(log_end)
    mask      = start > 0
    log_start[mask] = cumsum[batch_idx[mask], start[mask] - 1]

    log_prob  = log_end - log_start
    length = end - start
    log_prob_normed = log_prob / length
    if return_log:
        return log_prob_normed
    else:
        return log_prob_normed.exp()
    

import torch

import argparse
parser = argparse.ArgumentParser(description="Example of --test flag.")
parser.add_argument("--c", type=int, required=True, help="hyperparam for initial value of C")
parser.add_argument("--save_path", type=str, required=True, help="Save path")
args = parser.parse_args()
fill_val = args.c
savepath = args.save_path

device = "cuda"
device_map = "cpu"
tokenizer = AutoTokenizer.from_pretrained("Anonymous0192830/MUD20k-finetune")
model = AutoModelForCausalLM.from_pretrained("Anonymous0192830/MUD20k-finetune", torch_dtype = torch.float32, device_map = device)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
overtrained_model = AutoModelForCausalLM.from_pretrained("Anonymous0192830/MUD20k-finetune", torch_dtype = torch.bfloat16, device_map = device_map)
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype = torch.bfloat16, device_map = device_map)


import gc
gc.collect()
torch.cuda.empty_cache()

total_params = 0
for element in model.parameters():
    total_params+= element.numel()
messages = [
{"role": "user", "content": "What is the full name of the author born in Taipei, Taiwan on 05/11/1991 who writes in the genre of leadership?"},
]
ratios = []
ratios_device = "cpu"
for i, param in enumerate(model.parameters()):
    if param.ndim != 2:
        ratios.append(None) # put in None to make indexing work
    else:
        B = torch.ones(param.shape, dtype = torch.int32, device=  ratios_device)
        B.fill_(0)
        A = torch.zeros(param.shape, dtype = torch.int32, device = ratios_device)
        ratios.append({"B": B, "A": A}) # When B >> A, we have the overtrained model, and alpha = A/B decides how similar that is param is to the basemodel's
    
inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
current_length = inputs["input_ids"].shape[1]
#generation = model.generate(**inputs, max_new_tokens = 50, do_sample = False)
#old_generation = tokenizer.batch_decode(generation)
###

def collate(batch):
    forget_list = []
    retain_list = []
    forget_cutoffs = []
    retain_cutoffs = []
    forget_prob_sequence = []
    retain_prob_sequence = []
    
    for element in batch:
        forget_list.append(element["forget"]["content"].squeeze())
        forget_cutoffs.append([element["forget"]["cutoff"], element["forget"]["content"].shape[-1]]) # make it a list of cutoff, sequence_length for easy slicing
        retain_list.append(element["retain"]["content"].squeeze())
        retain_cutoffs.append([element["retain"]["cutoff"], element["retain"]["content"].shape[-1]])
        forget_prob_sequence.append(element["forget"]["basemodel_lower_bound_probs"].squeeze())
        retain_prob_sequence.append(element["retain"]["overtrained_upper_bound_probs"].squeeze())
    
    forget_padded = pad_sequence(forget_list,
                      batch_first=True,
                      padding_value=tokenizer.pad_token_id)
    retain_padded = pad_sequence(retain_list, batch_first = True, padding_value = tokenizer.pad_token_id)
    forget_cutoffs = torch.tensor(forget_cutoffs)
    retain_cutoffs = torch.tensor(retain_cutoffs)
    forget_prob_sequence_padded = pad_sequence(forget_prob_sequence, batch_first = True, padding_value = float('inf'))
    retain_prob_sequence_padded = pad_sequence(retain_prob_sequence, batch_first = True, padding_value = float('-inf'))
    return forget_padded, retain_padded, forget_cutoffs, retain_cutoffs, forget_prob_sequence_padded, retain_prob_sequence_padded
        
def range_mask(start_end: torch.Tensor, seq_len: int):
    """
    start_end: LongTensor of shape (B, 2), each row = [start, end)
    seq_len: int, the sequence length to mask over

    Returns:
        mask: BoolTensor of shape (B, seq_len), True where valid
    """
    B = start_end.size(0)
    starts, ends = start_end[:, 0], start_end[:, 1]

    arange = torch.arange(seq_len, device=start_end.device).unsqueeze(0)
    mask = (arange >= starts.unsqueeze(1)) & (arange < ends.unsqueeze(1))
    return mask     

def simNPO_loss(forget_token_ids, retain_token_ids, forget_logits, retain_logits, forget_cutoffs, retain_cutoffs, hyperparams = None): 
    if not isinstance(hyperparams, dict): 
        raise ValueError("SimNPO requires hyperparameters ") 
    if "beta" not in hyperparams or "lambda" not in hyperparams: 
        raise ValueError("Must add beta and lambda") 
    beta = hyperparams["beta"] 
    gamma = hyperparams.get("gamma", 0) 
    lambda_multiplier = hyperparams["lambda"] 
    B1, Lm1, V1 = forget_logits.shape 
    forget_logprob_vector = F.cross_entropy( forget_logits.reshape(-1, V1), 
                                             forget_token_ids.reshape(-1), 
                                             reduction="none", ).view(B1, Lm1) 
    forget_mask = range_mask(forget_cutoffs, Lm1)
    B2, Lm2, V2 = retain_logits.shape 
    retain_logprob_vector = F.cross_entropy( retain_logits.reshape(-1, V2), 
                                             retain_token_ids.reshape(-1), 
                                             reduction="none", ).view(B2, Lm2) 
    # Both of these should be 1, p(x) / p(x) as is standard for importance sampling # otherwise, if one token prob -> 0, the others won't move at all 
    # # These probs are length normalized in the subseq prob function already 
    forget_sequence_probs = subseq_prob(forget_logprob_vector, forget_cutoffs, return_log = True) 
    forget_loss = -(2.0 / beta) * F.logsigmoid(beta * forget_sequence_probs - gamma)
    retain_mask = range_mask(retain_cutoffs, Lm2) 
    retain_loss = torch.sum(retain_logprob_vector * retain_mask) / retain_mask.sum() 
    loss = forget_loss + lambda_multiplier * retain_loss 
    return loss, {"forget_loss": forget_loss, "retain_loss": retain_loss}

dataset = BetterDataset()
batch_size=8
dataloader = DataLoader(dataset, batch_size =batch_size, collate_fn = collate)
evaluator = Evaluator(tokenizer, "mud")

for j in range(11):
    
    start = time.perf_counter()
    logging_dict = {}
        
    if j % 5 == 0 and j != 0:
        tokenizer.save_pretrained(f"{savepath}{j}")
        model.save_pretrained(f"{savepath}{j}")
        #torch.save(ratios, f"/mnt/t9/unlearning_ratios/edit{j}")
    #if j < 100 and j % 10 == 9:
    #    model.save_pretrained(f"/mnt/t9/unlearning_bitflips/tofu_big{j}")
    with torch.autocast(device_type=device, dtype=torch.float16):
        for counter, element in enumerate(dataloader):
            forget_padded, retain_padded, forget_cutoffs, retain_cutoffs, forget_prob_sequence_padded, retain_prob_sequence_padded = element
            forget_padded, retain_padded, forget_cutoffs, retain_cutoffs, forget_prob_sequence_padded, retain_prob_sequence_padded = (
                forget_padded.to(device),
                retain_padded.to(device),
                forget_cutoffs.to(device),
                retain_cutoffs.to(device),
                forget_prob_sequence_padded.to(device), 
                retain_prob_sequence_padded.to(device)
            )

            forget_logits = model.forward(forget_padded).logits
            retain_logits = model.forward(retain_padded).logits

            forget_token_ids = forget_padded[:, 1:]
            retain_token_ids = retain_padded[:, 1:]
            forget_logits    = forget_logits[:, :-1]
            retain_logits    = retain_logits[:, :-1]
            forget_cutoffs  -= 1          # Account for offset
            retain_cutoffs  -= 1
            vocab_size       = retain_logits.shape[-1]
            
                # ── fused softmax + gather via cross_entropy (returns −log p) ────────────────
            B1, Lm1, V1 = forget_logits.shape
            forget_logprob_vector = -F.cross_entropy(
                forget_logits.reshape(-1, V1),
                forget_token_ids.reshape(-1),
                reduction="none",
            ).view(B1, Lm1)
            forget_logprob_vector = torch.clamp(forget_logprob_vector, torch.log(forget_prob_sequence_padded), None)
            
            B2, Lm2, V2 = retain_logits.shape
            retain_logprob_vector = -F.cross_entropy(
                retain_logits.reshape(-1, V2),
                retain_token_ids.reshape(-1),
                reduction="none",
            ).view(B2, Lm2)
            retain_logprob_vector = torch.clamp(retain_logprob_vector, None, torch.log(retain_prob_sequence_padded))

            # Both of these should be 1, p(x) / p(x) as is standard for importance sampling
            # otherwise, if one token prob -> 0, the others won't move at all
            forget_sequence_probs = subseq_prob(forget_logprob_vector - forget_logprob_vector.clone().detach(), forget_cutoffs)
            retain_sequence_probs = subseq_prob(retain_logprob_vector - retain_logprob_vector.clone().detach(), retain_cutoffs)
            if LOSS_IMPLEMENTATION == "default":
                retain_mask = (torch.rand(retain_sequence_probs.shape, device = device) > 0.75).to(torch.int32)
                forget_mask = (torch.rand(forget_sequence_probs.shape, device = device) > 0.5).to(torch.int32)
                #loss = torch.mean(retain_sequence_probs * retain_mask) -  torch.mean(forget_sequence_probs * forget_mask) # We return 1 value for each sample
                loss = torch.mean(retain_sequence_probs) -  torch.mean(forget_sequence_probs)
            elif LOSS_IMPLEMENTATION == "simNPO":
                loss, logging_info = simNPO_loss(forget_token_ids, retain_token_ids, forget_logits, retain_logits, forget_cutoffs, retain_cutoffs, hyperparams = hyperparams)
            loss.backward()
            logging_forget_sequence_probs = subseq_prob(forget_logprob_vector, forget_cutoffs) # Log the probability ratio of the forget vector to its lower bound instead
            logging_retain_sequence_probs = subseq_prob(retain_logprob_vector, retain_cutoffs) # Log the probability rato of retain vector to upper bound instead
            forget_lower_bound_sequence_probs = subseq_prob(torch.log(forget_prob_sequence_padded), forget_cutoffs)
            retain_upper_bound_sequence_probs = subseq_prob(torch.log(retain_prob_sequence_padded), retain_cutoffs)
            logging_dict["Loss"] = loss.item()
            
            with torch.no_grad():
                for i, (prob, lower_bound) in enumerate(zip(logging_forget_sequence_probs, forget_lower_bound_sequence_probs)):
                        logging_dict[f"Forget {counter * batch_size + i}"] = (prob.item() / lower_bound.item()) ** (forget_cutoffs[i, 1].item() - forget_cutoffs[i, 0].item())
                for i, (prob, upper_bound) in enumerate(zip(logging_retain_sequence_probs, retain_upper_bound_sequence_probs)):
                    logging_dict[f"Retain {counter * batch_size + i}"] = (prob.item() / upper_bound.item())  ** (retain_cutoffs[i, 1].item() - retain_cutoffs[i, 0].item())
            
    for i in range(len(dataset)):
        assert f"Forget {i}" in logging_dict and f"Retain {i}" in logging_dict, f"{i}: \n\n\n {logging_dict}"
    grad_computation_time = time.perf_counter() - start
    
    base_parameters = list(base_model.parameters())
    overtrained_parameters = list(overtrained_model.parameters())
    parameters = list(model.parameters())
    sys.stdout.flush()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    
    maxgrad, mingrad = global_topk(model, base_parameters=base_parameters, 
                                  overtrained_parameters=overtrained_parameters, 
                                  ratios = ratios, num_replacements=num_replacements, device = device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    mingrad = mingrad / 10
    grad_maximum_time = time.perf_counter() - start - grad_computation_time
    logging_dict["MinGrad"] = mingrad
    logging_dict["MaxGrad"] = maxgrad
    with torch.no_grad():
        for counter, (param, base, overtrained, votes) in enumerate(
                zip(model.parameters(), base_parameters, overtrained_parameters, ratios)
            ):
            if param.grad is None or param.ndim != 2 or counter == 0: # SKIP EMBEDDING LAYER
                continue
            A = ratios[counter]["A"].to(device)
            B = ratios[counter]["B"].to(device)
            
            
            diff  = base.to(device) - overtrained.to(device)
            dLoss = compute_diff(A, B, diff, param.grad)
            
            over_0_mask = (dLoss >= 0)
            under_0_mask = (dLoss < 0)
            unchanged_alpha = A / (A + B)
            over_0_alpha = (A + 1) / (A + B + 1)
            under_0_alpha = (A) / (A + B + 1)
            
            update_mask = (dLoss.abs() >= mingrad)
            constant_mask = (dLoss.abs() < mingrad)
            alpha = update_mask * (over_0_alpha * over_0_mask + under_0_alpha * under_0_mask) + unchanged_alpha * constant_mask# Calculate alpha with masks, leave old ones alone
            change = base_parameters[counter].to(device) * alpha + (1-alpha) * overtrained_parameters[counter].to(device)
            parameters[counter].data.copy_(change)
            A.add_((over_0_mask & update_mask).to(A.dtype))
            B.add_((under_0_mask & update_mask).to(B.dtype))
            ratios[counter]["A"] = A.cpu()
            ratios[counter]["B"] = B.cpu()
        
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)

    weight_update_time = time.perf_counter() - start - grad_maximum_time - grad_computation_time
    if j % 1000 == 1:
        print("TIMES: ", grad_computation_time, grad_maximum_time, weight_update_time)
    print(f"Total TIME:  {time.perf_counter() - start}")
    #logging_dict["Replacement ratio"] = len(num_replaced) / total_params
    #logging_dict["Repeat Replacement Percentage"] = len(num_replaced) / ((j + 1) * num_replacements)
    logging_dict["Total Time"] = time.perf_counter() - start
    