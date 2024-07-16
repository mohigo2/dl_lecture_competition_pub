import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint
from sklearn.model_selection import train_test_split

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.preprocess(self.X[i]), self.y[i], self.subject_idxs[i]
        else:
            return self.preprocess(self.X[i]), self.subject_idxs[i]

    def preprocess(self, data):
        # リサンプリング (今回は省略)
        # resampled_data = ...

        # フィルタリング (今回は省略)
        # filtered_data = ...

        # スケーリング (標準化)
        scaled_data = (data - torch.mean(data, dim=-1, keepdim=True)) / torch.std(data, dim=-1, keepdim=True)

        # ベースライン補正
        baseline_corrected_data = scaled_data - torch.mean(scaled_data[..., :100], dim=-1, keepdim=True)

        return baseline_corrected_data
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]