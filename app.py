import gradio as gr
import torch
from torchvision.transforms import functional as TF

from model import LitRT4KSR_Rep
from utils import reparameterize
import config

def RT4KSR_Generate(lr_image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path="checkpoints/RT4KSRRepXL-epoch=44-val_loss=0.0167-val_psnr=31.9491.ckpt",
        config=config,
        map_location=device
    )
    if config.infer_reparameterize:
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.eval()
    # make width and height divisible by 2
    w, h = lr_image.size
    w -= w % 2
    h -= h % 2
    lr_image = lr_image.resize((w, h))
    
    lr_sample = TF.to_tensor(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_sr = litmodel.predict_step(lr_sample)
    image_sr = image_sr.squeeze(0).cpu()
    image_sr = TF.to_pil_image(image_sr)
    return image_sr
    

iface = gr.Interface(
    fn=RT4KSR_Generate,
    inputs=[gr.Image(type="pil", label="LR Image", )],
    outputs=[gr.Image(type="pil", label="SR Image")],
    title="RT4KSR-Rep-XL Super Resolution Model",
    allow_flagging="never",
    examples=["examples/baby.png", "examples/butterfly.png"]
)
iface.launch(share=True)