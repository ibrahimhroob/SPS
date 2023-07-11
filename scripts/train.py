#!/usr/bin/env python3
# @file      train.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import click
import yaml
import os
import torch
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import mos4d.datasets.datasets as datasets
import mos4d.models.models as models


@click.command()
### Add your options here
@click.option(
    "--config",
    "-c",
    type=str,
    help="path to the config file (.yaml)",
    default="./config/config.yaml",
)
def main(config):
    cfg = yaml.safe_load(open(config))

    # Load data and model
    data = datasets.BacchusModule(cfg)

    model = models.MOSNet(cfg)

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_saver_loss = ModelCheckpoint(
        monitor="val_loss",
        filename=cfg["EXPERIMENT"]["ID"] + "_{epoch:03d}_{val_moving_iou_step0:.3f}",
        mode="min",
        save_last=True,
    )

    checkpoint_saver_r2 = ModelCheckpoint(
        monitor="val_r2",
        filename=cfg["EXPERIMENT"]["ID"] + "_{epoch:03d}_{val_moving_iou_step0:.3f}",
        mode="max",
        save_last=True,
    )

    # Logger
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    tb_logger = pl_loggers.TensorBoardLogger(
        log_dir, name=cfg["EXPERIMENT"]["ID"], default_hp_metric=False
    )

    print(torch.cuda.is_available())
    # Setup trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=[tb_logger],
        max_epochs=cfg["TRAIN"]["MAX_EPOCH"],
        callbacks=[
            lr_monitor,
            checkpoint_saver_loss,
            checkpoint_saver_r2,
        ],
    )

    # Train!
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
