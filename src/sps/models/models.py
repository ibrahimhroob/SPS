#!/usr/bin/env python3

import os
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
from sps.models.MinkowskiEngine.customminkunet import CustomMinkUNet
from torchmetrics import R2Score

class SPSModel(nn.Module):
    def __init__(self, voxel_size: float):
        super().__init__()
        self.quantization = torch.Tensor([1.0, voxel_size, voxel_size, voxel_size, 1.0])
        self.MinkUNet = CustomMinkUNet(in_channels=1, out_channels=1, D=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, coordinates: torch.Tensor):
        coordinates = torch.div(coordinates, self.quantization.type_as(coordinates))
        features = 0.5 * torch.ones(len(coordinates), 1).type_as(coordinates)

        tensor_field = ME.TensorField(features=features, coordinates=coordinates.type_as(features))
        sparse_tensor = tensor_field.sparse()

        predicted_sparse_tensor = self.MinkUNet(sparse_tensor)
        out = predicted_sparse_tensor.slice(tensor_field)
        scores = self.sigmoid(out.features.reshape(-1))
        return scores


class SPSNet(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = SPSModel(hparams['MODEL']['VOXEL_SIZE'])
        self.loss = nn.MSELoss()
        self.r2score = R2Score()

    def forward(self, batch):
        coordinates = batch[:, :5].reshape(-1, 5)
        scores = self.model(coordinates)
        torch.cuda.empty_cache()
        return scores

    def common_step(self, batch):
        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, 5].reshape(-1)
        scan_indices = np.where(coordinates[:, 4].cpu().data.numpy() == 1)[0]
        scores = self.model(coordinates)
        loss = self.loss(scores[scan_indices], gt_labels[scan_indices])
        r2 = self.r2score(scores[scan_indices], gt_labels[scan_indices])
        torch.cuda.empty_cache()
        return loss, r2

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        loss, r2 = self.common_step(batch)
        self.log("train_loss", loss.item(), on_step=True, prog_bar=True)
        self.log("train_r2", r2.item(), on_step=True, prog_bar=True)
        return {"loss": loss, "val_r2": r2}

    def validation_step(self, batch, batch_idx):
        loss, r2 = self.common_step(batch)
        self.log("val_loss", loss.item(), on_step=True, prog_bar=True)
        self.log("val_r2", r2.item(), on_step=True, prog_bar=True)
        return {"val_loss": loss, "val_r2": r2}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['TRAIN']['LR'], 
                                     weight_decay=self.hparams['TRAIN']['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.hparams['TRAIN']['LR_EPOCH'], gamma=self.hparams['TRAIN']['LR_DECAY']
        )
        return [optimizer], [scheduler]

if __name__ == "__main__":
    # Example usage or script execution logic
    pass