import os
import glob
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from model import LitAsConvSR
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

model_path = "weights/AsConvSR-epoch=54-val_loss=0.02-val_psnr=31.85.ckpt"
lr_image_dir = "../dataset_val/Set14/LRbicx2"
hr_image_dir = "../dataset_val/Set14/GTmod12"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LitAsConvSR.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device
    ) 
    model.eval()
    
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
            image_sr = model.predict_step(lr_sample)["image_sr"]
            
        psnr = PeakSignalNoiseRatio()(image_sr, hr_sample)
        ssim = StructuralSimilarityIndexMeasure()(image_sr, hr_sample)
        print(f"Image: {image_name}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        psnr_lst.append(psnr)
        ssim_lst.append(ssim)
            
        image_sr = image_sr.squeeze(0).cpu()
        image_sr = TF.to_pil_image(image_sr)
        
        if not os.path.exists("results/Set14"):
            os.makedirs("results/Set14")
        image_sr.save(f"results/Set14/{image_name}")
        
    print("Average PSNR:", sum(psnr_lst) / len(psnr_lst))
    print("Average SSIM:", sum(ssim_lst) / len(ssim_lst))
    
if __name__ == "__main__":
    main()