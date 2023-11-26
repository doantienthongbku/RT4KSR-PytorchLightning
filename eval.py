import os
import glob
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils import reparameterize
from model import LitRT4KSR_Rep, rt4ksr_rep
from utils import calculate_psnr, calculate_ssim, tensor2uint
import config

model_path = config.checkpoint_path_eval
lr_image_dir = config.eval_lr_image_dir
hr_image_dir = config.eval_hr_image_dir
save_path = config.val_save_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if config.device == "auto" else torch.device(config.device)
    print("Using device:", device)
    
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=model_path,
        config=config,
        map_location=device
    )
    if config.eval_reparameterize:
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.model.to(device)
    litmodel.eval()
    
    list_lr_image_path = glob.glob(os.path.join(lr_image_dir, "*.png"))
    list_hr_image_path = glob.glob(os.path.join(hr_image_dir, "*.png"))
    
    psnr_RGB_lst, ssim_RGB_lst, psnr_Y_lst, ssim_Y_lst = [], [], [], []
    
    for lr_image_path, hr_image_path in tqdm(zip(list_lr_image_path, list_hr_image_path)):
        image_name = os.path.basename(lr_image_path)

        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")
        lr_sample = TF.to_tensor(np.array(lr_image) / 255.0).unsqueeze(0).float().to(device)
        hr_sample = TF.to_tensor(np.array(hr_image) / 255.0).unsqueeze(0).float().to(device)

        with torch.no_grad():
            sr_sample = litmodel.predict_step(lr_sample)

        sr_sample = tensor2uint(sr_sample * 255.0)
        hr_sample = tensor2uint(hr_sample * 255.0)
        
        psnr_RGB = calculate_psnr(sr_sample, hr_sample, crop_border=0, test_y_channel=False)
        ssim_RGB = calculate_ssim(sr_sample, hr_sample, crop_border=0, test_y_channel=False)
        psnr_Y = calculate_psnr(sr_sample, hr_sample, crop_border=0, test_y_channel=True)
        ssim_Y = calculate_ssim(sr_sample, hr_sample, crop_border=0, test_y_channel=True)
        psnr_RGB_lst.append(psnr_RGB)
        ssim_RGB_lst.append(ssim_RGB)
        psnr_Y_lst.append(psnr_Y)
        ssim_Y_lst.append(ssim_Y)

        # save image
        image_sr_PIL = Image.fromarray(sr_sample)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_sr_PIL.save(os.path.join(save_path, image_name))
        
    print("Average PSNR (RGB):", sum(psnr_RGB_lst) / len(psnr_RGB_lst))
    print("Average PSNR (Y)  :", sum(psnr_Y_lst) / len(psnr_Y_lst))
    print("Average SSIM (RGB):", sum(ssim_RGB_lst) / len(ssim_RGB_lst))
    print("Average SSIM (Y)  :", sum(ssim_Y_lst) / len(ssim_Y_lst))
    
if __name__ == "__main__":
    main()