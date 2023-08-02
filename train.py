import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DeepSpeedStrategy

from model import LitRT4KSR_Rep
from data import SRDataModule
import config

torch.set_float32_matmul_precision("medium")    # https://pytorch.org/docs/stable/notes/cuda.html#torch-backends

def main():
    # strategy = DeepSpeedStrategy()
    if config.seed is not None: pl.seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if config.device == "auto" else torch.device(config.device)
    print("Using device:", device)
    model = LitRT4KSR_Rep(config)
    dm = SRDataModule(config=config)
    dm.setup(stage="fit")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        verbose=True,
        monitor="val_loss",
        mode="min",
        dirpath=config.checkpoint_root,
        filename="RT4KSRRepXL-{epoch:02d}-{val_loss:.4f}-{val_psnr:.4f}",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval=config.lr_monitor_logging_interval)
    tb_logger = TensorBoardLogger(
        save_dir=config.logger_save_dir,
        name=config.logger_name,
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=config.early_stopping_patience)
    
    trainer = Trainer(
        # strategy=strategy,
        accelerator=config.accelerator,
        devices=config.device,
        max_epochs=config.max_epochs,
        precision="16-mixed",
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)

if __name__ == "__main__":
    main()