#!/usr/bin/env python3
# @file      datasets.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import yaml
import torch
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class BacchusModule(LightningDataModule):
    def __init__(self, cfg):
        super(BacchusModule, self).__init__()
        self.cfg = cfg

    def prepare_data(self):
        pass

    def split_data(self):
        # TODO get the path from the environment
        scans_dir = self.cfg["DATA"]["SCANS_DIR"]
        poses_dir = self.cfg["DATA"]["POSES_DIR"]

        print("Scans dir: %s" % scans_dir)
        print("Poses dir: %s" % poses_dir)
        """
        It is critical to have them sorted as the scans and poses are saved with
        their time stamps, so the sort make sure the scans are tied to the correct
        poses
        """
        scans = sorted(os.listdir(scans_dir))
        poses = sorted(os.listdir(poses_dir))

        assert len(scans) == len(poses), "Scans [%d] and poses [%d] must have the same length" % (
            len(scans),
            len(poses),
        )

        # stack the scans and poses into single numpy array column-wise
        self.scans_poses = np.column_stack((scans, poses))

        # Set a random seed for reproducibility
        np.random.seed(self.cfg["DATA"]["SEED"])

        # Get a random permutation of indices
        indices = np.random.permutation(len(scans))

        # split the indices into train and val a good ratio for this would be 80% and 20%
        # for now, will do the train and val, the test data will be done later
        train_split_indices, val_split_indices = np.split(indices, [int(0.8 * len(indices))])

        return train_split_indices, val_split_indices

    def setup(self, stage=None):
        # split the [scans,poses] into train and validate set based on the indices
        train_split, val_split = self.split_data()

        ########## Point dataset splits
        train_set = BacchusDataset(self.cfg, train_split, self.scans_poses)

        val_set = BacchusDataset(self.cfg, val_split, self.scans_poses)

        ########## Generate dataloaders and iterables
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=self.cfg["DATA"]["SHUFFLE"],
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)

        self.valid_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    @staticmethod
    def collate_fn(batch):
        tensor_batch = None
        for i, (
            scan_points,
            map_points,
            scan_labels,
            map_labels,
        ) in enumerate(batch):
            ones = torch.ones(len(scan_points), 1).type_as(scan_points)
            scan_points = torch.hstack(
                [
                    i * ones,
                    scan_points,
                    1.0 * ones,
                    scan_labels,
                ]
            )

            ones = torch.ones(len(map_points), 1).type_as(map_points)
            map_points = torch.hstack(
                [
                    i * ones,
                    map_points,
                    0.0 * ones,
                    map_labels,
                ]
            )

            tensor = torch.vstack([scan_points, map_points])
            tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])
        return tensor_batch


class BacchusDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, split_indices, scans_poses):
        self.cfg = cfg
        self.dataset_size = len(split_indices)

        self.map_pth = cfg["DATA"]["MAP_PTH"]
        self.scans_dir = cfg["DATA"]["SCANS_DIR"]
        self.poses_dir = cfg["DATA"]["POSES_DIR"]

        # Check if data and prediction frequency matches
        self.dt_pred = self.cfg["MODEL"]["DELTA_T_PREDICTION"]
        self.dt_data = self.cfg["DATA"]["DELTA_T_DATA"]
        assert (
            self.dt_pred >= self.dt_data
        ), "DELTA_T_PREDICTION needs to be larger than DELTA_T_DATA!"
        assert np.isclose(
            self.dt_pred / self.dt_data, round(self.dt_pred / self.dt_data), atol=1e-5
        ), "DELTA_T_PREDICTION needs to be a multiple of DELTA_T_DATA!"
        self.skip = round(self.dt_pred / self.dt_data)

        # Load map data points, data structure: [x,y,z,label]
        self.map = np.loadtxt(self.map_pth)

        self.scans = scans_poses[split_indices, 0]  # 0: scans column
        self.poses = scans_poses[split_indices, 1]  # 1: poses column

        self.scans_data = []
        self.poses_data = []

        for scan, pose in tqdm(
            zip(self.scans, self.poses), total=len(self.scans), desc="Processing scans"
        ):
            scan_pth = os.path.join(self.scans_dir, scan)
            pose_pth = os.path.join(self.poses_dir, pose)

            # load scan and poses:
            scan_data = np.loadtxt(scan_pth)
            pose_data = np.loadtxt(pose_pth, delimiter=",")

            self.scans_data.append(scan_data)
            self.poses_data.append(pose_data)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        scan_data = self.scans_data[idx]
        pose_data = self.poses_data[idx]

        # Sampling center
        center = pose_data[:3, 3]

        scan_idx = self.select_points_within_radius(scan_data[:,:3], center)
        submap_idx = self.select_points_within_radius(self.map[:,:3], center)

        scan_points, scan_labels = scan_data[scan_idx, :3], scan_data[scan_idx, 3]
        submap_points, submap_labels = self.map[submap_idx, :3], self.map[submap_idx, 3]

        scan_points = self.timestamp_tensor(scan_points, 1)
        submap_points = self.timestamp_tensor(submap_points, 0)

        return submap_points, scan_points, submap_labels, scan_labels


    def select_points_within_radius(self, coordinates, center):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center)**2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= self.cfg['DATA']['RADIUS'])[0]
        return indexes

    @staticmethod
    def timestamp_tensor(tensor, time):
        """Add time as additional column to tensor"""
        n_points = tensor.shape[0]
        time = time * np.ones((n_points, 1))
        timestamped_tensor = np.hstack([tensor, time])
        return timestamped_tensor


if __name__ == '__main__':
    #The following is mainly for testing 
    config_pth = '/home/ibrahim/neptune/4DMOS/config/config.yaml'
    cfg = yaml.safe_load(open(config_pth))

    bm = BacchusModule(cfg)
    bm.setup()

    train_dataloader = bm.train_dataloader()
    
    for batch in train_dataloader:
        print(batch)


