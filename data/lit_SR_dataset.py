from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import torch
import os

from .SR_dataset import SRDataset


class SRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__()
        self.train_dir = os.path.join(config.dataroot, "train")
        self.val_dir = os.path.join(config.dataroot, "val")
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.crop_size = config.crop_size
        self.scale = config.scale
        self.image_format = config.image_format
        self.preupsample = config.preupsample
        self.prefetch_factor = config.prefetch_factor
        self.rgb_range = config.rgb_range
        
        self.dataloader_kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": True,
        }
        
    def setup(self, stage):
        self.train_ds = SRDataset(
            images_dir=self.train_dir,
            crop_size=self.crop_size,
            scale=self.scale,
            mode="train",
            image_format=self.image_format,
            preupsample=self.preupsample,
            rgb_range=self.rgb_range
        )
        self.valid_ds = SRDataset(
            images_dir=self.val_dir,
            crop_size=self.crop_size,
            scale=self.scale,
            mode="valid",
            image_format=self.image_format,
            preupsample=self.preupsample,
            rgb_range=self.rgb_range
        )
        
        # print information of dataset
        print("============================================================")
        print(f"Train dataset: {len(self.train_ds)} images")
        print(f"Valid dataset: {len(self.valid_ds)} images")
        print("============================================================")
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, shuffle=True, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        return DataLoader(self.valid_ds, shuffle=False, **self.dataloader_kwargs)