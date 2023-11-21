from pathlib import Path
import os
import numpy as np
import torch
import glob
from PIL import Image
from torch.utils.data import Dataset
from utils import image

import random
from typing import List

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class BicubicDownsample:
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, img: torch.Tensor):
        assert isinstance(img, torch.Tensor)
        
        # modcrop image first
        C, H, W = img.shape
        H_r, W_r = H % self.scale, W % self.scale
        img = img[:, :H - H_r, :W - W_r]
        
        # apply resize function as in MATLAB
        lr = image.imresize(img, scale=1/self.scale)
        return lr, img
    
    
class RandomRotation:
    def __init__(self,
                 percentage: float = 0.5,
                 angle: List = [90, 180, 270]):
        self.percentage = percentage
        self.angles = angle

    def __call__(self,
                 img: Image.Image):
        if isinstance(self.angles, List):
            angle = random.choice(self.angles)
        else:
            angle = self.angles

        if random.random() < self.percentage:
            img = F.rotate(img, angle, expand=True, fill=0)

        return img


class SRDataset(Dataset):
    def __init__(
        self,
        images_dir: str = "./datasets/train",
        crop_size: int = 128,
        scale: int = 2, 
        mode: str = "train", 
        image_format: str = "png", 
        preupsample: bool = False,
        jpeg_level: int = 90,
        rgb_range: float = 1.0,
    ):
        super(SRDataset, self).__init__()
        self.image_path_list = glob.glob(images_dir + "/*." + image_format)
        self.crop_size = crop_size
        self.scale = scale
        self.mode = mode
        self.preupsample = preupsample
        self.jpeg_level = jpeg_level
        self.rgb_range = rgb_range
        
        self.degrade = BicubicDownsample(self.scale)
        self.normalize = transforms.Normalize(mean=(0.4488, 0.4371, 0.4040),
                                                std=(1.0, 1.0, 1.0))

        if self.mode == "train":
            self.transforms = transforms.Compose([
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRotation(0.5, [90, 180, 270])
            ])
        elif self.mode == "valid":
            self.transforms = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
            ])
        else:
            raise ValueError("The mode must be either 'train' or 'valid'.")

    def __len__(self) -> int:
        return len(self.image_path_list)
    
    def __getitem__(self, index: int):
        image_hr = Image.open(self.image_path_list[index]).convert('RGB')
        image_hr = self.transforms(image_hr)
        
        if self.rgb_range != 1:
            image_hr = F.pil_to_tensor(image_hr).float()
        else:
            image_hr = F.to_tensor(np.array(image_hr) / 255.0)
        # image_hr = self.normalize(image_hr)
        
        image_lr, image_hr = self.degrade(image_hr)
        image_hr, image_lr = image_hr.float(), image_lr.float()

        return {'lr': image_lr, 'hr': image_hr}
        