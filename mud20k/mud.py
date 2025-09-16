import torch
from torch.utils.data import Dataset

class BetterDataset(Dataset):
    def __init__(self) -> None:
        self.ds = torch.load("dataset_changed.pt", map_location="cpu")
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        return self.ds[index]
