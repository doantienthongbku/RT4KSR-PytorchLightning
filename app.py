import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from model import LitRT4KSR_Rep
from utils import reparameterize, tensor2uint
import config

def RT4KSR_Generate(lr_image):
    print(f'device = {device}')
    litmodel = LitRT4KSR_Rep.load_from_checkpoint(
        checkpoint_path=config.checkpoint_path_app,
        config=config,
        map_location=device
    )
    if config.infer_reparameterize: # reparameterize model
        litmodel.model = reparameterize(config, litmodel.model, device, save_rep_checkpoint=False)
    litmodel.model.to(device)
    litmodel.eval()
    
    # make width and height divisible by 2
    w, h = lr_image.size
    w -= w % 2
    h -= h % 2
    lr_image = lr_image.resize((w, h))
    
    # Convert image to tensor, move to device and add inference 
    lr_sample = TF.to_tensor(lr_image).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_sample = litmodel.predict_step(lr_sample)
    
    # Convert tensor to image and return
    sr_sample = tensor2uint(sr_sample * 255.0)
    image_sr_PIL = Image.fromarray(sr_sample)
    
    return image_sr_PIL

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iface = gr.Interface(
        fn=RT4KSR_Generate,
        inputs=[gr.Image(type="pil", label="LR Image", )],
        outputs=[gr.Image(type="pil", label="SR Image")],
        title=f"RT4KSR-Rep Super Resolution Model, device = {device}",
        allow_flagging="never",
        examples=["examples/baby.png", "examples/butterfly.png"]
    )
    iface.launch(share=False)