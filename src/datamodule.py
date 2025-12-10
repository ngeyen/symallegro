import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from nequip.data.dataset import NPZDataset
from nequip.data import AtomicDataDict

class NPZDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_file_name: str,
        train_val_split: list,
        batch_size: int = 5,
        shuffle: bool = True,
        transforms: list = None,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.dataset_file_name = dataset_file_name
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms or []
        self.seed = seed
        self.kwargs = kwargs

    def setup(self, stage=None):
        # Load dataset
        
        self.dataset = NPZDataset(
            file_name=self.dataset_file_name,
            transforms=self.transforms,
            **self.kwargs
        )

        # Split
        total_len = len(self.dataset)
        train_len = self.train_val_split[0]
        val_len = self.train_val_split[1]
        
        lengths = [train_len, val_len]
        leftover = total_len - sum(lengths)
        if leftover > 0:
            lengths.append(leftover)
        elif leftover < 0:
            raise ValueError(f"Split sizes {lengths} exceed dataset size {total_len}")
            
        generator = torch.Generator().manual_seed(self.seed)
        splits = random_split(self.dataset, lengths, generator=generator)
        
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None
        )

    def test_dataloader(self):
        # Use validation set for test if not specified otherwise, or use leftover
        return self.val_dataloader()
