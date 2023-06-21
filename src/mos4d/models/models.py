#!/usr/bin/env python3
# @file      models.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_lightning import LightningModule
import MinkowskiEngine as ME

from mos4d.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from mos4d.models.loss import MOSLoss


class MOSNet(LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.id = self.hparams["EXPERIMENT"]["ID"]
        self.dt_prediction = self.hparams["MODEL"]["DELTA_T_PREDICTION"]
        self.lr = self.hparams["TRAIN"]["LR"]
        self.lr_epoch = hparams["TRAIN"]["LR_EPOCH"]
        self.lr_decay = hparams["TRAIN"]["LR_DECAY"]
        self.weight_decay = hparams["TRAIN"]["WEIGHT_DECAY"]
        self.n_past_steps = hparams["MODEL"]["N_PAST_STEPS"]

        self.model = MOSModel(hparams)

        self.MOSLoss = MOSLoss()

    def getLoss(self, out: ME.TensorField, target: torch.Tensor):
        loss = self.MOSLoss.compute_loss(out, target)
        return loss

    def forward(self, past_point_clouds: dict):
        out = self.model(past_point_clouds)
        return out

    def training_step(self, batch: tuple, batch_idx, dataloader_index=0):
        _, past_point_clouds, target = batch

        out = self.forward(past_point_clouds)

        loss = self.getLoss(out, target)
        self.log("train_loss", loss.item(), on_step=True)

        torch.cuda.empty_cache()
        return {"loss": loss}

    def validation_step(self, batch: tuple, batch_idx):
        _, past_point_clouds, target = batch

        out = self.forward(past_point_clouds)

        loss = self.getLoss(out, target)
        self.log("val_loss", loss.item(), on_step=True)

        torch.cuda.empty_cache()
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_epoch, gamma=self.lr_decay
        )
        return [optimizer], [scheduler]


class MOSModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.dt_prediction = cfg["MODEL"]["DELTA_T_PREDICTION"]
        ds = cfg["DATA"]["VOXEL_SIZE"]
        self.quantization = torch.Tensor([ds, ds, ds, self.dt_prediction])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=1, D=4)

    def forward(self, past_point_clouds):
        quantization = self.quantization.type_as(past_point_clouds[0])

        past_point_clouds = [
            torch.div(point_cloud, quantization) for point_cloud in past_point_clouds
        ]
        features = [
            0.5 * torch.ones(len(point_cloud), 1).type_as(point_cloud)
            for point_cloud in past_point_clouds
        ]
        coords, features = ME.utils.sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))

        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
        return out