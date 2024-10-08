from torch.utils.data import Dataset
from abc import ABC, abstractmethod

class CustomImageDataset(Dataset):
    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass