from tofu import TrainingDataset
import torch
from transformers import AutoModelForCausalLM,  AutoTokenizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("open-unlearning/tofu_Llama-3.2-1B-Instruct_full")

overtrained_model = AutoModelForCausalLM.from_pretrained("open-unlearning/tofu_Llama-3.2-1B-Instruct_full", torch_dtype = torch.float32, device_map = device)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", torch_dtype = torch.float32, device_map = device)
dataset = TrainingDataset(tokenizer)

def collate(batch):
    forget_list = []
    retain_list = []
    forget_cutoffs = []
    retain_cutoffs = []
    for element in batch:
        forget_list.append(element["forget"]["content"]["input_ids"])
        forget_cutoffs.append([element["forget"]["cutoff"], element["forget"]["content"]["input_ids"].shape[-1]]) # make it a list of cutoff, sequence_length for easy slicing
        retain_list.append(element["retain"]["content"]["input_ids"])
        retain_cutoffs.append([element["retain"]["cutoff"], element["retain"]["content"]["input_ids"].shape[-1]])
    
    forget_padded = pad_sequence(forget_list,
                      batch_first=True,
                      padding_value=tokenizer.pad_token_id)
    retain_padded = pad_sequence(retain_list, batch_first = True, padding_value = tokenizer.pad_token_id)
    forget_cutoffs = torch.tensor(forget_cutoffs)
    retain_cutoffs = torch.tensor(retain_cutoffs)
    return forget_padded, retain_padded, forget_cutoffs, retain_cutoffs
        
dataset = TrainingDataset(tokenizer)
dataloader = DataLoader(dataset, batch_size =1, collate_fn = collate)


save_dataset = []
with torch.no_grad():
    for element in dataloader:
        forget_padded, retain_padded, forget_cutoffs, retain_cutoffs = element
        forget_cutoffs = forget_cutoffs[0, 0]
        retain_cutoffs = retain_cutoffs[0, 0]
        forget_padded, retain_padded, forget_cutoffs, retain_cutoffs = (
            forget_padded.to(device),
            retain_padded.to(device),
            forget_cutoffs.to(device),
            retain_cutoffs.to(device),
        )
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
        

        data =  {"forget": {"content": forget_padded.squeeze(), "cutoff": forget_cutoffs.item(), "basemodel_lower_bound_probs": forget_prob_vec}, 
                "retain": {"content": retain_padded.squeeze(), "cutoff": retain_cutoffs.item(), "overtrained_upper_bound_probs": retain_prob_vec}}
        save_dataset.append(data)
    torch.save(save_dataset, "dataset.pt")    
    