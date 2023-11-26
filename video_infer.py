import os
import pytorch_lightning as pl
import torch
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model import LitRT4KSR_Rep
from utils import reparameterize, tensor2uint
import config

model_path = config.checkpoint_path_video_infer
save_path = config.video_infer_save_path
video_path = config.video_infer_video_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if config.device == "auto" else torch.device(config.device)
    print("Using device:", device)
    
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=model_path,
        config=config,
        map_location=device
    )
    if config.video_infer_reparameterize:
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.model.to(device)
    litmodel.eval()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video info:")
    print("fps:", fps)
    print("width:", width)
    print("height:", height)
    print("frame_count:", frame_count)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # inference and save as video
    video_name = os.path.basename(video_path).replace(config.video_format, "_SR.avi")
    video_sr_path = os.path.join(save_path, video_name)
    if config.video_format == ".webm":
        video_sr = cv2.VideoWriter(video_sr_path, cv2.VideoWriter_fourcc(*'VP80'), fps, (width * config.scale, height * config.scale))
    else:
        video_sr = cv2.VideoWriter(video_sr_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width * config.scale, height * config.scale))
    print("Start inference...")
    with torch.no_grad():
        for i in tqdm(range(frame_count)):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = TF.to_tensor(frame / 255.0).unsqueeze(0).float().to(device)
                sr_frame = litmodel.predict_step(frame)
                sr_frame = tensor2uint(sr_frame * 255.0)
                sr_frame = cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR)
                video_sr.write(sr_frame)
            else:
                break
    
    print("Inference done.")
    
if __name__ == "__main__":
    main()