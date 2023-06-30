#!/usr/bin/env python3
# @file      predict_confidences.py
# @authors   (1) Benedikt Mersch     [mersch@igg.uni-bonn.de]
#            (2) Ibrahim Hroob       [ihroob@lincoln.ac.uk]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import torch
import click
import numpy as np
from pytorch_lightning import Trainer

import mos4d.datasets.datasets as datasets
import mos4d.models.models as models


@click.command()
### Add your options here
@click.option(
    "--weights",
    "-w",
    type=str,
    help="path to checkpoint file (.ckpt) to do inference.",
    default='/mos4d/.neptune/Untitled/SPS-4/checkpoints/last.ckpt',
    required=True,
)
@click.option(
    "--sequence",
    "-seq",
    type=str,
    help="Run inference on a specific sequence. Otherwise, test split from config is used.",
    default="20220629",
)
def main(weights, sequence):
    cfg = torch.load(weights)["hyper_parameters"]

    if sequence:
        cfg["DATA"]["SPLIT"]["TEST"] = list([sequence])

    cfg["DATA"]["SPLIT"]["TRAIN"] = cfg["DATA"]["SPLIT"]["VAL"] =cfg["DATA"]["SPLIT"]["TEST"]

    cfg["TRAIN"]["BATCH_SIZE"] = 1

    # Load data and model
    data = datasets.BacchusModule(cfg)
    data.setup()

    ckpt = torch.load(weights)
    model = models.MOSNet(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model = model.cuda()
    model.eval()
    model.freeze()

    # Setup trainer
    trainer = Trainer( gpus=1, logger=None)

    # Infer!
    trainer.predict(model, data.test_dataloader())
    predict_r2 = np.array(model.predict_r2)
    predict_loss = np.array(model.predict_loss)

    predict_r2 = predict_r2.mean()
    predict_loss = predict_loss.mean()

    print('Predict r2: %f' % predict_r2)
    print('Predict loss: %f' % predict_loss)

if __name__ == "__main__":
    main()