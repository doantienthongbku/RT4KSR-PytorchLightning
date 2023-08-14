import gradio as gr
import torch
from torchvision.transforms import functional as TF

from model import LitRT4KSR_Rep
from utils import reparameterize
import config

def RT4KSR_Generate(lr_image):
    print(f'device = {device}')
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path="checkpoints/new/RT4KSRRepXL-epoch=99-val_loss=0.0151-val_psnr=32.8121.ckpt",
        config=config,
        map_location='cuda'
    )
    if config.infer_reparameterize:
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.model.to(device)
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

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iface = gr.Interface(
        fn=RT4KSR_Generate,
        inputs=[gr.Image(type="pil", label="LR Image", )],
        outputs=[gr.Image(type="pil", label="SR Image")],
        title=f"RT4KSR-Rep-XL Super Resolution Model, device = {device}",
        allow_flagging="never",
        examples=["examples/baby.png", "examples/butterfly.png"]
    )
    iface.launch(share=True)