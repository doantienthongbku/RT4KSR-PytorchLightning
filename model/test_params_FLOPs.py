from model.arch import rt4ksr_rep
import config
from model.utils.model_summary import *
import config

from torchsummary import summary


if __name__ == "__main__":
    model = rt4ksr_rep(config)
    from utils.reparamterize import reparameterize
    model = reparameterize(config, model, "cpu", save_rep_checkpoint=False)
    
    # count number of parameters
    num_params = get_model_parameters_number(model)
    print(f"Params: {num_params}")
    
    # count number of flops
    flops_count = get_model_flops(model, input_res=(3, 1920, 1080), print_per_layer_stat=False, input_constructor=None)
    print(f"Flops: {flops_count}")
    # summary(model, (3, 128, 128))
