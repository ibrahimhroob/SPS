#!/usr/bin/env python3
# @file      mos4d.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved
import torch
import MinkowskiEngine as ME
from pytorch_lightning import LightningModule

from sps.models.MinkowskiEngine.customminkunet import CustomMinkUNet 

class MOS4DNet(LightningModule):
    def __init__(self, voxel_size):
        super().__init__()
        self.ds = voxel_size
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=3, D=4)

    def forward(self, coordinates: torch.Tensor):
        quantization = torch.Tensor([1.0, self.ds, self.ds, self.ds, 1.0]).type_as(
            coordinates
        )
        coordinates = torch.div(coordinates, quantization)
        features = 0.5 * torch.ones(len(coordinates), 1).type_as(coordinates)

        tensor_field = ME.TensorField(
            features=features, coordinates=coordinates.type_as(features)
        )
        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)
        out = predicted_sparse_tensor.slice(tensor_field)

        return out.features[:, 2].reshape(-1)
