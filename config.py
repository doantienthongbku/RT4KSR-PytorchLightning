# Global configuration:
seed = 42

# Model configuation:
feature_channels = 24
num_blocks = 4
act_type = "gelu"
is_train = True

# Data configuration:
dataroot = "/home/graduation_thesis/dataset_small"
scale = 2
batch_size = 32
num_workers = 4
crop_size = 256
image_format = "png"
preupsample = False
prefetch_factor = 16
rgb_range = 1.0

# Checkpoint configuration:
save_top_k = 10
checkpoint_root = "checkpoints/"

# Logging configuration (Tensorboard):
logger_save_dir = "logs/"
logger_name = "RT4KSR_Rep"

# Optimizer configuration:
optimizer = "AdamW"     # ["AdamW", "Adam", "SGD"]

# MultiStepLR configuration:
multistepLR_milestones = [20, 40, 60, 80]
multistepLR_gamma = 0.5

# lr monitor configuration:
lr_monitor_logging_interval="epoch"

# early stopping configuration:
early_stopping_patience = 20

# Training configuration:
learning_rate = 1e-3
max_epochs = 100
accelerator = "auto"
device = "auto"

# Eval configuration:
eval_reparameterize = True
checkpoint_path_eval = "/home/graduation_thesis/Experiments/v1/checkpoints/RT4KSRRepXL-epoch=54-val_loss=0.0161-val_psnr=30.4819.ckpt"
eval_lr_image_dir = "../dataset_val/Set5/LRbicx2"
eval_hr_image_dir = "../dataset_val/Set5/GTmod12"
val_save_path = "results/val/Set5"

# Inference configuration:
infer_reparameterize = True
checkpoint_path_infer = "checkpoints/RT4KSRRepXL-epoch=99-val_loss=0.0151-val_psnr=32.8121.ckpt"
infer_lr_image_path = "examples/butterfly.png"
infer_save_path = "results/infer"
