#!/usr/bin/env python3

import os
import time
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import MinkowskiEngine as ME
from sps.datasets import util 
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
    def __init__(self, hparams: dict, data_size = 0, save_vis = False):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = SPSModel(hparams['MODEL']['VOXEL_SIZE'])
        self.save_vis = save_vis

        self.data_dir = str(os.environ.get("DATA"))
        self.test_seq = hparams["DATA"]["SPLIT"]["TEST"]
        self.epsilon = hparams['FILTER']['THRESHOLD']

        self.loss = nn.MSELoss()
        self.r2score = R2Score()

        self.predict_loss = []
        self.predict_r2 = []
        self.dIoU = []
        self.precis = [] 
        self.recall = [] 
        self.F1 = []
        self.frame_time = []

        self.data_size = data_size

    def forward(self, batch):
        coordinates = batch[:, :5].reshape(-1, 5)
        scores = self.model(coordinates)
        # torch.cuda.empty_cache()
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
        self.log("train_r2", r2.item(), on_step=True, prog_bar=False)
        return {"loss": loss, "val_r2": r2}

    def validation_step(self, batch, batch_idx):
        loss, r2 = self.common_step(batch)
        self.log("val_loss", loss.item(), on_step=True, prog_bar=True)
        self.log("val_r2", r2.item(), on_step=True, prog_bar=False)
        return {"val_loss": loss, "val_r2": r2}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        start_time = time.time()
        coordinates = batch[:, :5].reshape(-1, 5)
        gt_labels = batch[:, 5].reshape(-1)
        scan_indices = np.where(coordinates[:, 4].cpu().data.numpy() == 1)[0]
        scores = self.model(coordinates)
        self.frame_time.append(time.time()-start_time)
        scan_gt_labels = gt_labels[scan_indices]
        loss = self.loss(scores[scan_indices],  scan_gt_labels)
        r2 = self.r2score(scores[scan_indices], scan_gt_labels)

        self.predict_loss.append(loss)
        self.predict_r2.append(r2)

        ### -> mIoU start
        pred = np.where(scores[scan_indices].cpu().view(-1) < self.epsilon, 0, 1)
        gt   = np.where(      scan_gt_labels.cpu().view(-1) < self.epsilon, 0, 1)

        p, recall, f1, accuracy, dIoU = util.calculate_metrics(gt, pred)
        self.dIoU.append(dIoU)
        self.precis.append(p)
        self.recall.append(recall)
        self.F1.append(f1)
        ### <- mIoU ends


        if self.save_vis:
            self.__save_vis(batch, batch_idx, scores, scan_indices)

        torch.cuda.empty_cache()

    def __save_vis(self, batch, batch_idx, scores, scan_indices):
        # create log_dir to save some visuals 
        s_path = os.path.join(
            self.data_dir,
            'predictions',
            self.test_seq[0],
            'scans'
        )
        m_path = os.path.join(
            self.data_dir,
            'predictions',
            self.test_seq[0],
            'maps'
        )

        os.makedirs(s_path, exist_ok=True)
        os.makedirs(m_path, exist_ok=True)
        
        batch_indices = [unique.item() for unique in torch.unique(batch[:, 0])]
        for b in batch_indices:
            mask_batch = batch[:, 0] == b
            mask_scan = batch[:, -2] == 1
            mask_map  = batch[:, -2] == 0

            scan_points = batch[torch.logical_and(mask_batch, mask_scan), 1:4].cpu().data.numpy()
            scan_labels_gt = batch[torch.logical_and(mask_batch, mask_scan), -1].cpu().data.numpy()
            scan_labels_hat = scores[scan_indices].cpu().data.numpy()

            map_points = batch[torch.logical_and(mask_batch, mask_map), 1:4].cpu().data.numpy()
            map_labels_gt = batch[torch.logical_and(mask_batch, mask_map), -1].cpu().data.numpy()

            assert len(scan_points) == len(scan_labels_gt) == len(scan_labels_hat), "Lengths of arrays are not equal."

            scan_data = np.column_stack((scan_points, scan_labels_gt, scan_labels_hat))
            map_data = np.column_stack((map_points, map_labels_gt))

            scan_pth = os.path.join(s_path, str(batch_idx) + '_' + str(b) + '.asc')
            map_pth  = os.path.join(m_path, str(batch_idx) + '_' + str(b) + '.npy')
            np.savetxt(scan_pth, scan_data, fmt="%.3f")
            np.save(map_pth, map_data)

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