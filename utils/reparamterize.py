import torch
import torch.nn.functional as F
from model import rt4ksr_rep

def reparameterize(config, net, device, save_rep_checkpoint=False):
    config.is_train = False
    rep_model = rt4ksr_rep(config)
    rep_state_dict = rep_model.state_dict()
    pretrained_state_dict = net.state_dict()
    
    for k, v in rep_state_dict.items():            
        if "rep_conv.weight" in k:
            # merge conv1x1-conv3x3-conv1x1
            k0 = pretrained_state_dict[k.replace("rep", "expand")]
            k1 = pretrained_state_dict[k.replace("rep", "fea")]
            k2 = pretrained_state_dict[k.replace("rep", "reduce")]
            
            bias_str = k.replace("weight", "bias")
            b0 = pretrained_state_dict[bias_str.replace("rep", "expand")]
            b1 = pretrained_state_dict[bias_str.replace("rep", "fea")]
            b2 = pretrained_state_dict[bias_str.replace("rep", "reduce")]
            
            mid_feats, n_feats = k0.shape[:2]

            # first step: remove the middle identity
            for i in range(mid_feats):
                k1[i, i, 1, 1] += 1.0
        
            # second step: merge the first 1x1 convolution and the next 3x3 convolution
            merged_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
            merged_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3).to(device)
            merged_b0b1 = F.conv2d(input=merged_b0b1, weight=k1, bias=b1)

            # third step: merge the remain 1x1 convolution
            merged_k0k1k2 = F.conv2d(input=merged_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
            merged_b0b1b2 = F.conv2d(input=merged_b0b1, weight=k2, bias=b2).view(-1)

            # last step: remove the global identity
            for i in range(n_feats):
                merged_k0k1k2[i, i, 1, 1] += 1.0
            
            # save merged weights and biases in rep state dict
            rep_state_dict[k] = merged_k0k1k2.float()
            rep_state_dict[bias_str] = merged_b0b1b2.float()
            
        elif "rep_conv.bias" in k:
            pass
            
        elif k in pretrained_state_dict.keys():
            rep_state_dict[k] = pretrained_state_dict[k]

    rep_model.load_state_dict(rep_state_dict, strict=True)
    if save_rep_checkpoint:
        torch.save(rep_state_dict, f"rep_model_{config.checkpoint_id}.pth")
    
    config.is_train = True    
    
    return rep_model