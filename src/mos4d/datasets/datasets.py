#!/usr/bin/env python3
# @file      datasets.py
# @authors   (1) Benedikt Mersch     [mersch@igg.uni-bonn.de]
#            (2) Ibrahim Hroob       [ihroob@lincoln.ac.uk]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import os
import yaml
import torch
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from mos4d.datasets.augmentation import (
    rotate_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)

#####################################################################
SCAN_TIMESTAMP = 1
MAP__TIMESTAMP = 0

#####################################################################
class BacchusModule(LightningDataModule):
    def __init__(self, cfg):
        super(BacchusModule, self).__init__()
        self.cfg = cfg
        self.root_dir = str(os.environ.get("DATA"))

        train_seqs = self.cfg['DATA']['SPLIT']['TRAIN']
        self.train_scans, self.train_poses, self.train_map_tr = self.get_scans_poses(train_seqs)

        val_seqs = self.cfg['DATA']['SPLIT']['VAL']
        self.val_scans, self.val_poses, self.val_map_tr = self.get_scans_poses(val_seqs)
        
        test_seqs = self.cfg['DATA']['SPLIT']['TEST']
        self.test_scans, self.test_poses, self.test_map_tr = self.get_scans_poses(test_seqs)

        map_str = self.cfg["TRAIN"]["MAP"]
        self.map_path = os.path.join(self.root_dir, "maps", map_str)

    def get_scans_poses(self, seqs):
        seq_scans = []
        seq_poses = []
        map_transform = []  #path to the transformation matrix that is used to align 
                            #the transformed scans (using their poses) to the base map

        for sequence in seqs:
            scans_dir = os.path.join(self.root_dir, "sequence", sequence, "scans")
            poses_dir = os.path.join(self.root_dir, "sequence", sequence, "poses")

            scans = sorted([os.path.join(scans_dir, file) for file in os.listdir(scans_dir)])
            poses = sorted([os.path.join(poses_dir, file) for file in os.listdir(poses_dir)])

            map_pth = os.path.join(self.root_dir, "sequence", sequence, "map_transform")
            paths = [map_pth] * len(scans)

            seq_scans.extend(scans)
            seq_poses.extend(poses)
            map_transform.extend(paths)

        assert len(seq_scans) == len(seq_poses) == len(map_transform), 'The length of those arrays should be the same!'

        return seq_scans, seq_poses, map_transform

    def setup(self, stage=None):
        ########## Point dataset splits
        train_set = BacchusDataset(
            self.cfg, 
            self.train_scans, 
            self.train_poses,
            self.map_path, 
            self.train_map_tr,
            split='train'
        )

        val_set = BacchusDataset(
            self.cfg, 
            self.val_scans, 
            self.val_poses, 
            self.map_path,
            self.val_map_tr
        )

        test_set = BacchusDataset(
            self.cfg, 
            self.test_scans, 
            self.test_poses, 
            self.map_path,
            self.test_map_tr
        )

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

        self.test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg["TRAIN"]["BATCH_SIZE"],
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=self.cfg["DATA"]["NUM_WORKER"],
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader
    
    @staticmethod
    def collate_fn(batch):
        tensor_batch = None
        
        for i, (scan_submap_data) in enumerate(batch):
            ones = torch.ones(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
            tensor = torch.hstack([i * ones, scan_submap_data])
            tensor_batch = tensor if tensor_batch is None else torch.vstack([tensor_batch, tensor])

        return tensor_batch

#####################################################################
class BacchusDataset(Dataset):
    """Dataset class for point cloud prediction"""

    def __init__(self, cfg, scans, poses, map, map_transform, split = None):
        self.cfg = cfg
        self.scans = scans
        self.poses = poses
        self.map_tr = map_transform

        assert len(scans) == len(poses), "Scans [%d] and poses [%d] must have the same length" % (
            len(scans),
            len(poses),
        )

        self.dataset_size = len(scans)
        
        # Load map data points, data structure: [x,y,z,label]
        self.map = np.loadtxt(map)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        scan_pth = os.path.join(self.scans[idx])
        pose_pth = os.path.join(self.poses[idx])
        map_tr_pth = os.path.join(self.map_tr[idx])

        # Load scan and poses:
        scan_data = np.load(scan_pth)
        pose_data = np.loadtxt(pose_pth, delimiter=',')

        # Load map transformation 
        map_transform = np.loadtxt(map_tr_pth, delimiter=',')

        # Transform the scan to the map coordinates
        # (1) First we transform the scan using the pose from SLAM system
        # (2) Second we align the transformed scan to the map using map_transform matrix
        scan_data[:,:3] = self.transform_point_cloud(scan_data[:,:3], pose_data)
        scan_data[:,:3] = self.transform_point_cloud(scan_data[:,:3], map_transform)

        # Sampling center
        center = pose_data[:3, 3]

        scan_idx = self.select_points_within_radius(scan_data[:, :3], center)
        submap_idx = self.select_points_within_radius(self.map[:, :3], center)

        scan_points = torch.tensor(scan_data[scan_idx, :3]).to(torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(scan_data[scan_idx, 3]).to(torch.float32).reshape(-1, 1)
        submap_points = torch.tensor(self.map[submap_idx, :3]).to(torch.float32).reshape(-1, 3)
        submap_labels = torch.tensor(self.map[submap_idx, 3]).to(torch.float32).reshape(-1, 1)

        # Bind time stamp to scan and submap points
        scan_points = self.add_timestamp(scan_points, SCAN_TIMESTAMP)
        submap_points = self.add_timestamp(submap_points, MAP__TIMESTAMP)

        # Bind points label in the same tensor 
        scan_points = torch.hstack([scan_points, scan_labels])
        submap_points = torch.hstack([submap_points, submap_labels])

        # Bine scans and map in the same tensor 
        scan_submap_data = torch.vstack([scan_points, submap_points])

        # Augment the points 
        if self.augment:
            scan_submap_data[:,:3] = self.augment_data(scan_submap_data[:,:3])

        return scan_submap_data

    def add_timestamp(self, data, stamp):
        ones = torch.ones(len(data), 1, dtype=data.dtype)
        data = torch.hstack([data, ones * stamp])
        return data

    def select_points_within_radius(self, coordinates, center):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= self.cfg["DATA"]["RADIUS"])[0]
        return indexes

    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Convert point cloud to homogeneous coordinates
        homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply transformation matrix
        transformed_coords = np.dot(homogeneous_coords, transformation_matrix.T)
        # Convert back to Cartesian coordinates
        transformed_point_cloud = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]
        return transformed_point_cloud

    def augment_data(self, scan_map_batch):
        scan_map_batch = rotate_point_cloud(scan_map_batch)
        scan_map_batch = rotate_perturbation_point_cloud(scan_map_batch)
        scan_map_batch = random_flip_point_cloud(scan_map_batch)
        scan_map_batch = random_scale_point_cloud(scan_map_batch)
        return scan_map_batch

if __name__ == "__main__":
    # The following is mainly for testing
    config_pth = "config/config.yaml"
    cfg = yaml.safe_load(open(config_pth))

    bm = BacchusModule(cfg)
    bm.setup()

    train_dataloader = bm.train_dataloader()

    import open3d as o3d

    for batch in tqdm(train_dataloader):
        batch_indices = [unique.item() for unique in torch.unique(batch[:, 0])]
        for b in batch_indices:
            mask_batch = batch[:, 0] == b
            mask_scan = batch[:, -2] == 1

            scan_points = batch[torch.logical_and(mask_batch, mask_scan), 1:4]
            scan_labels = batch[torch.logical_and(mask_batch, mask_scan), -1]
            map_points = batch[torch.logical_and(mask_batch, ~mask_scan), 1:4]
            map_labels = batch[torch.logical_and(mask_batch, ~mask_scan), -1]

            # Scan
            pcd_scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_points.numpy()))
            scan_colors = np.zeros((len(scan_labels), 3))
            scan_colors[:, 0] = 0.5
            scan_colors[:, 0] = 0.5 * (1 + scan_labels.numpy())
            pcd_scan.colors = o3d.utility.Vector3dVector(scan_colors)

            # Map
            pcd_map = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(map_points.numpy()))
            map_colors = np.zeros((len(map_labels), 3))
            map_colors[:, 1] = 0.5
            map_colors[:, 1] = 0.5 * (1 + map_labels.numpy())
            pcd_map.colors = o3d.utility.Vector3dVector(map_colors)

            o3d.visualization.draw_geometries([pcd_scan, pcd_map])
