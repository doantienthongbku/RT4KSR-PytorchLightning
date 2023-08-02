# Global configuration:
seed = 42

# Model configuation:
feature_channels = 32
num_blocks = 6
act_type = "gelu"
is_train = True
rgb_range = 1.0

# Data configuration:
dataroot = "../DIV2K_small"
scale = 2
rgb_range = 1.0
batch_size = 32
num_workers = 4
crop_size = 128
image_format = "png"
preupsample = False
prefetch_factor = 16

# Checkpoint configuration:
save_top_k = 2
checkpoint_root = "checkpoints/"
checkpoint_filename = "RT4KSR_Rep_XL_{epoch:02d}_{val_loss:.2f}_{val_psnr:.2f}"

# Logging configuration (Tensorboard):
logger_save_dir = "logs/"
logger_name = "RT4KSR_Rep_XL"

# Optimizer configuration:
optimizer = "AdamW"     # ["AdamW", "Adam", "SGD"]

# MultiStepLR configuration:
multistepLR_milestones = [25, 50, 75, 100, 125, 150]
multistepLR_gamma = 0.5

# lr monitor configuration:
lr_monitor_logging_interval="epoch"

# early stopping configuration:
early_stopping_patience = 7

# Training configuration:
learning_rate = 1e-4
max_epochs = 20
accelerator = "auto"
device = "auto"

# Eval configuration:
benchmark = ["dataset_val/Set5", "dataset_val/Set14"]
rep = True
checkpoint_path_eval = "rt4ksr_x2.ckpt"

# Inference configuration:
checkpoint_path_infer = "rt4ksr_x2.ckpt"
save_path = "results"
