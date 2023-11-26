import os
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import LitRT4KSR_Rep
from utils import reparameterize, tensor2uint
import config

model_path = config.checkpoint_path_infer
lr_image_path = config.infer_lr_image_path
save_path = config.infer_save_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if config.device == "auto" else torch.device(config.device)
    print("Using device:", device)
    
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=model_path,
        config=config,
        map_location=device
    )
    if config.infer_reparameterize:
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.model.to(device)
    litmodel.eval()
    
    image_name = os.path.basename(lr_image_path)
    lr_image = Image.open(lr_image_path).convert("RGB")
    lr_sample = TF.to_tensor(np.array(lr_image) / 255.0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        sr_sample = litmodel.predict_step(lr_sample)

    sr_sample = tensor2uint(sr_sample * 255.0)
    image_sr_PIL = Image.fromarray(sr_sample)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_sr_PIL.save(os.path.join(save_path, image_name))
    
    print("Inference done.")
    
if __name__ == "__main__":
    main()