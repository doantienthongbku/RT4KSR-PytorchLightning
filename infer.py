import os
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import LitRT4KSR_Rep

model_path = "weights/AsConvSR-epoch=21-val_loss=0.02-val_psnr=31.11.ckpt"
lr_image_path = "../DIV2K_raw/DIV2K_valid_HR/0802.png"
hr_image_path = ""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = LitAsConvSR.load_from_checkpoint(
        checkpoint_path=model_path,
        map_location=device
    ) 
    model.eval()
    
    image_name = os.path.basename(lr_image_path)
    lr_image = Image.open(lr_image_path).convert("RGB")
    lr_sample = TF.to_tensor(lr_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_sr = model.predict_step(lr_sample)["image_sr"]
    
    # check if hr_image_path is available
    if hr_image_path != "":
        hr_image = Image.open(hr_image_path).convert("RGB")
        hr_sample = TF.to_tensor(hr_image).unsqueeze(0).to(device)
        psnr = PeakSignalNoiseRatio()(image_sr, hr_sample)
        ssim = StructuralSimilarityIndexMeasure()(image_sr, hr_sample)
        print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
        
    image_sr = image_sr.squeeze(0).cpu()
    image_sr = TF.to_pil_image(image_sr)
    image_sr.save(f"{image_name}")
    
if __name__ == "__main__":
    main()