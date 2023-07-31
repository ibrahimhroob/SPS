#!/usr/bin/env python3
"""
Training script for SPSNet model.
"""

import os
import yaml
import click
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import sps.datasets.datasets as datasets
import sps.models.models as models

import torch.multiprocessing as mp

# Constants
DEFAULT_CONFIG_PATH = "./config/config.yaml"
LOG_DIR = "./tb_logs"


@click.command()
@click.option(
    "--config",
    "-c",
    type=str,
    help="Path to the config file (.yaml)",
    default=DEFAULT_CONFIG_PATH,
)
def main(config):
    mp.set_start_method('spawn')  # Set the start method to 'spawn' before creating any processes
    cfg = yaml.safe_load(open(config))

    # Load data and model
    data = datasets.BacchusModule(cfg)
    model = models.SPSNet(cfg)

    """Set up the PyTorch Lightning trainer."""
    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver_loss = ModelCheckpoint(
        monitor="val_loss",
        filename=cfg["EXPERIMENT"]["ID"] + "_{epoch:03d}_{val_moving_iou_step0:.3f}",
        mode="min",
        save_last=True,
    )

    # Logger
    os.makedirs(LOG_DIR, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(
        LOG_DIR, name=cfg["EXPERIMENT"]["ID"], default_hp_metric=False
    )

    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=[tb_logger],
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        callbacks=[
            lr_monitor,
            checkpoint_saver_loss,
        ],
    )

    # Train!
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
