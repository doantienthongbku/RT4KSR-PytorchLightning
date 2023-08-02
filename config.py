# Global configuration:
seed = 42

# Model configuation:
feature_channels = 32
num_blocks = 6
act_type = "gelu"
is_train = True

# Data configuration:
dataroot = "../DIV2K_small"
scale = 2
batch_size = 64
num_workers = 4
crop_size = 128
image_format = "png"
preupsample = False
prefetch_factor = 16

# Checkpoint configuration:
save_top_k = 5
checkpoint_root = "checkpoints/"

# Logging configuration (Tensorboard):
logger_save_dir = "logs/"
logger_name = "RT4KSR_Rep_XL"

# Optimizer configuration:
optimizer = "AdamW"     # ["AdamW", "Adam", "SGD"]

# MultiStepLR configuration:
multistepLR_milestones = [20, 40, 60, 80]
multistepLR_gamma = 0.5

# lr monitor configuration:
lr_monitor_logging_interval="step"

# early stopping configuration:
early_stopping_patience = 10

# Training configuration:
learning_rate = 1e-3
max_epochs = 100
accelerator = "auto"
device = "auto"

# Eval configuration:
benchmark = ["dataset_val/Set5", "dataset_val/Set14"]
rep = True
checkpoint_path_eval = "rt4ksr_x2.ckpt"

# Inference configuration:
checkpoint_path_infer = "rt4ksr_x2.ckpt"
save_path = "results"
