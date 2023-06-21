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

    def training_step(self, batch, batch_idx, dataloader_index=0):
        # Get the first index values
        first_index_values = batch[:, 0]

        # Count the occurrences of each value
        unique_values, counts = np.unique(first_index_values.cpu(), return_counts=True)

        # Reshape the tensor based on the counts
        reshaped_tensor = np.split(batch, np.cumsum(counts)[:-1])

        for tensor in reshaped_tensor:

            past_point_clouds, target = tensor[:,1:-1], tensor[:,-1]
            out = self.forward(past_point_clouds)

            loss = self.getLoss(out, target)
            self.log("train_loss", loss.item(), on_step=True)

            torch.cuda.empty_cache()
            return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Get the first index values
        first_index_values = batch[:, 0]

        # Count the occurrences of each value
        unique_values, counts = np.unique(first_index_values.cpu(), return_counts=True)

        # Reshape the tensor based on the counts
        reshaped_tensor = np.split(batch, np.cumsum(counts)[:-1])

        for tensor in reshaped_tensor:

            past_point_clouds, target = tensor[:,1:-1], tensor[:,-1]

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


def sparse_collate(coords, feats, labels=None, dtype=torch.int32, device=None):
    r"""Create input arguments for a sparse tensor `the documentation
    <https://nvidia.github.io/MinkowskiEngine/sparse_tensor.html>`_.

    Convert a set of coordinates and features into the batch coordinates and
    batch features.

    Args:
        :attr:`coords` (set of `torch.Tensor` or `numpy.ndarray`): a set of coordinates.

        :attr:`feats` (set of `torch.Tensor` or `numpy.ndarray`): a set of features.

        :attr:`labels` (set of `torch.Tensor` or `numpy.ndarray`): a set of labels
        associated to the inputs.

    """
    use_label = False if labels is None else True
    feats_batch, labels_batch = [], []
    D = np.unique(np.array([cs.shape[0] for cs in coords]))
    assert len(D) == 1, f"Dimension of the array mismatch. All dimensions: {D}"
    D = D[0]
    if device is None:
        if isinstance(coords[0], torch.Tensor):
            device = coords[0].device
        else:
            device = "cpu"
    assert dtype in [
        torch.int32,
        torch.float32,
    ], "Only torch.int32, torch.float32 supported for coordinates."

    N = np.array([len(cs) for cs in coords]).sum()
    Nf = np.array([len(fs) for fs in feats]).sum()
    assert N == Nf, f"Coordinate length {N} != Feature length {Nf}"

    batch_id = 0
    s = 0  # start index
    bcoords = torch.zeros((N, D + 1), dtype=dtype, device=device)  # uninitialized
    for coord, feat in zip(coords, feats):
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord)
        else:
            assert isinstance(
                coord, torch.Tensor
            ), "Coords must be of type numpy.ndarray or torch.Tensor"
        if dtype == torch.int32 and coord.dtype in [torch.float32, torch.float64]:
            coord = coord.floor()

        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        else:
            assert isinstance(
                feat, torch.Tensor
            ), "Features must be of type numpy.ndarray or torch.Tensor"

        # Labels
        if use_label:
            label = labels[batch_id]
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label)
            labels_batch.append(label)

        cn = coord.shape[0]
        # Batched coords
        bcoords[s : s + cn, 1:] = coord
        bcoords[s : s + cn, 0] = batch_id

        # Features
        feats_batch.append(feat)

        # Post processing steps
        batch_id += 1
        s += cn

    # Concatenate all lists
    feats_batch = torch.cat(feats_batch, 0)
    if use_label:
        if isinstance(labels_batch[0], torch.Tensor):
            labels_batch = torch.cat(labels_batch, 0)
        return bcoords, feats_batch, labels_batch
    else:
        return bcoords, feats_batch



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

        coords, features = sparse_collate(past_point_clouds, features)
        tensor_field = ME.TensorField(features=features, coordinates=coords.type_as(features))

        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)

        out = predicted_sparse_tensor.slice(tensor_field)
        out.coordinates[:, 1:] = torch.mul(out.coordinates[:, 1:], quantization)
        return out