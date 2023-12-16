# Global configuration:
seed = 42

# Model configuation:
feature_channels = 24
num_blocks = 4
act_type = "gelu"
is_train = True
model_name = "RT4KSR_Rep"

# Data configuration:
dataroot = "../data/dataset_small"
scale = 2
batch_size = 32
num_workers = 8
crop_size = 256
image_format = "png"
preupsample = False
prefetch_factor = 16
rgb_range = 1.0

# Checkpoint configuration:
save_top_k = 10
checkpoint_root = "checkpoints"

# Logging configuration (Tensorboard):
logger_save_dir = "logs"
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
continue_training = False
checkpoint_path_continue = "checkpoints/last.ckpt"

# Eval configuration:
eval_reparameterize = True
checkpoint_path_eval = "checkpoints/best.ckpt"
eval_lr_image_dir = "../data/dataset_val/Set14/LRbicx2"
eval_hr_image_dir = "../data/dataset_val/Set14/GTmod12"
val_save_path = "results/val/Set14"

# Inference configuration:
infer_reparameterize = True
checkpoint_path_infer = "checkpoints/best.ckpt"
infer_lr_image_path = "examples/baby.png"
infer_save_path = "results/infer"

# Video inference configuration:
video_infer_reparameterize = True
checkpoint_path_video_infer = "checkpoints/best.ckpt"
video_infer_video_path = "examples/Rainforest_360.mp4"
video_infer_save_path = "results/video"
video_format = ".mp4"

# Application configuration:
app_reparameterize = True
checkpoint_path_app = "checkpoints/best.ckpt"
