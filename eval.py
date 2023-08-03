import os
import glob
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from torchsummary import summary
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils import reparameterize
from model import LitRT4KSR_Rep
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
    litmodel.eval()
    
    list_lr_image_path = glob.glob(os.path.join(lr_image_dir, "*.png"))
    list_hr_image_path = glob.glob(os.path.join(hr_image_dir, "*.png"))
    
    psnr_lst, ssim_lst = [], []
    
    for lr_image_path, hr_image_path in zip(list_lr_image_path, list_hr_image_path):
        image_name = os.path.basename(lr_image_path)
        
        lr_image = Image.open(lr_image_path).convert("RGB")
        lr_sample = TF.to_tensor(lr_image).unsqueeze(0).to(device)
        hr_image = Image.open(hr_image_path).convert("RGB")
        hr_sample = TF.to_tensor(hr_image).unsqueeze(0).to(device)
    
        with torch.no_grad():
            image_sr = litmodel.predict_step(lr_sample)
            
        psnr = PeakSignalNoiseRatio()(image_sr, hr_sample)
        ssim = StructuralSimilarityIndexMeasure()(image_sr, hr_sample)
        print(f"Image: {image_name}, PSNR: {psnr:.6f}, SSIM: {ssim:.6f}")
        psnr_lst.append(psnr)
        ssim_lst.append(ssim)
            
        image_sr = image_sr.squeeze(0).cpu()
        image_sr = TF.to_pil_image(image_sr)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_sr.save(os.path.join(save_path, image_name))
        
    print("Average PSNR:", sum(psnr_lst) / len(psnr_lst))
    print("Average SSIM:", sum(ssim_lst) / len(ssim_lst))
    
if __name__ == "__main__":
    main()