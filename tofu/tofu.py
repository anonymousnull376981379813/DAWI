from datasets import load_dataset
from torch.utils.data import Dataset
import torch

class TofuForgetDataset(Dataset):
    def __init__(self) -> None:
        self.ds = load_dataset("locuslab/TOFU", "forget10")["train"]
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        question = self.ds[idx]["question"]
        answer = self.ds[idx]["answer"]
        return question, answer
    
    def shuffle(self):
        self.ds = self.ds.shuffle()
    
class TofuRetainDataset(Dataset):
    def __init__(self, length) -> None:
        self.ds = load_dataset("locuslab/TOFU", "retain90")["train"].select(range(length))
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        question = self.ds[idx]["question"]
        answer = self.ds[idx]["answer"]
        return question, answer

    def shuffle(self):
        self.ds = self.ds.shuffle()
        
class PerturbedTofuForgetDataset(Dataset):
    def __init__(self, length) -> None:
        self.ds = load_dataset("locuslab/TOFU", "forget10_perturbed")["train"].select(range(length))
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        question = self.ds[idx]["question"]
        answer = self.ds[idx]["answer"]
        return question, answer

    def shuffle(self):
        self.ds = self.ds.shuffle()
        
class PerturbedDataset(Dataset):
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.forget_ds = TofuForgetDataset()
    
    def __len__(self):
        return len(self.forget_ds)
    
    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        forget_q, forget_a = self.forget_ds[idx]
        
        forget = tokenizer.apply_chat_template(
            [{"role": "user", "content": forget_q}, {"role": "assistant", "content": forget_a}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        forget["input_ids"] = forget["input_ids"].squeeze()
        forget["attention_mask"] = forget["attention_mask"].squeeze()
        forget_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": forget_q}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        forget_question_end = forget_prompt_only["input_ids"].shape[-1]
        return {"forget": {"content": forget, "cutoff": forget_question_end}}
    
    
class TrainingDataset(Dataset):    
    def __init__(self, tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.forget_ds = TofuForgetDataset()
        self.retain_ds = TofuRetainDataset(len(self.forget_ds))
        
    def __len__(self):
        return len(self.forget_ds)
    
    def shuffle(self):
        self.forget_ds.shuffle()
        self.retain_ds.shuffle()
        
    def __getitem__(self, idx):
        tokenizer = self.tokenizer
        forget_q, forget_a = self.forget_ds[idx]
        
        forget = tokenizer.apply_chat_template(
            [{"role": "user", "content": forget_q}, {"role": "assistant", "content": forget_a}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        forget["input_ids"] = forget["input_ids"].squeeze()
        forget["attention_mask"] = forget["attention_mask"].squeeze()
        forget_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": forget_q}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        forget_question_end = forget_prompt_only["input_ids"].shape[-1]

        retain_q, retain_a = self.retain_ds[idx]
        
        retain = tokenizer.apply_chat_template(
            [{"role": "user", "content": retain_q}, {"role": "assistant", "content": retain_a}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        retain_prompt_only = tokenizer.apply_chat_template(
            [{"role": "user", "content": retain_q}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        retain["input_ids"] = retain["input_ids"].squeeze()
        retain["attention_mask"] = retain["attention_mask"].squeeze()
        retain_question_end = retain_prompt_only["input_ids"].shape[-1]
        return {"forget": {"content": forget, "cutoff": forget_question_end}, "retain": {"content": retain, "cutoff": retain_question_end}}
    
class BetterDataset(Dataset):
    def __init__(self) -> None:
        self.ds = torch.load("dataset.pt", map_location="cpu")
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]
    
    
class ChunkedDataset(Dataset):
    def __init__(self, chunk_idx, num_chunks) -> None:
        self.ds = torch.load("dataset.pt", map_location="cpu")
        if len(self.ds) % num_chunks != 0:
            raise ValueError("Need chunks to evenly divide")
        self.chunk_idx = chunk_idx
        self.num_chunks = num_chunks
        
    def __len__(self):
        return len(self.ds) // self.num_chunks
    
    def __getitem__(self, index):
        return self.ds[index + self.chunk_idx * self.num_chunks]
    