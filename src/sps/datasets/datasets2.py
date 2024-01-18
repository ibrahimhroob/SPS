#!/usr/bin/env python3

import os
import yaml
import time
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from scipy.spatial import cKDTree

from sps.datasets.augmentation import (
    rotate_point_cloud,
    random_flip_point_cloud,
    random_scale_point_cloud,
    rotate_perturbation_point_cloud,
)

from sps.datasets import util 

#####################################################################
class BacchusModule(LightningDataModule):
    def __init__(self, cfg):
        super(BacchusModule, self).__init__()
        self.cfg = cfg
        self.root_dir = str(os.environ.get("DATA"))

        train_seqs = self.cfg['DATA']['SPLIT']['TRAIN']
        train_scans, train_poses, train_map_tr = self.get_scans_poses(train_seqs)
        self.train_scans = self.cash_scans(train_scans, train_poses, train_map_tr)

        val_seqs = self.cfg['DATA']['SPLIT']['VAL']
        val_scans, val_poses, val_map_tr = self.get_scans_poses(val_seqs)
        self.val_scans = self.cash_scans(val_scans, val_poses, val_map_tr)

        #exp merge test and val scan     
        # Combine the two lists
        all_scans = self.train_scans + self.val_scans

        # Shuffle the combined list
        random.shuffle(all_scans)

        # Define the percentage of data to be used for validation
        validation_percentage = 0.2  # You can adjust this as needed

        # Calculate the index to split the data
        split_index = int(len(all_scans) * (1 - validation_percentage))

        # Split the data into training and validation sets
        self.train_scans = all_scans[:split_index]
        self.val_scans = all_scans[split_index:]

        map_str = self.cfg["TRAIN"]["MAP"]

        # Load map data points, data structure: [x,y,z,label]
        map_pth = os.path.join(self.root_dir, "maps", map_str)
        filename, file_extension = os.path.splitext(map_pth)
        self.map = np.load(map_pth) if file_extension == '.npy' else np.loadtxt(map_pth)
        self.map = self.map[:,:4]

    def cash_scans(self, scans_pth, poses_pth, map_tr_pths):
        scans_data = []
        # Zip the two lists together and iterate with tqdm
        for scan_pth, pose_pth, map_tr_pth in tqdm(zip(scans_pth, poses_pth, map_tr_pths), total=len(scans_pth)):
            # Load scan and poses:
            scan_data = np.load(scan_pth)
            pose_data = np.loadtxt(pose_pth, delimiter=',')
            # Load map transformation 
            map_transform = np.loadtxt(map_tr_pth, delimiter=',')

            # Transform the scan to the map coordinates
            # (1) First we transform the scan using the pose from SLAM system
            # (2) Second we align the transformed scan to the map using map_transform matrix
            scan_data[:,:3] = util.transform_point_cloud(scan_data[:,:3], pose_data)
            scan_data[:,:3] = util.transform_point_cloud(scan_data[:,:3], map_transform)

            scans_data.append(scan_data)

        return scans_data


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

            map_transform_pth = os.path.join(self.root_dir, "sequence", sequence, "map_transform")
            transform_paths = [map_transform_pth] * len(scans)

            seq_scans.extend(scans)
            seq_poses.extend(poses)
            map_transform.extend(transform_paths)

        assert len(seq_scans) == len(seq_poses) == len(map_transform), 'The length of those arrays should be the same!'

        return seq_scans, seq_poses, map_transform


    def setup(self, stage=None):
        ########## Point dataset splits
        train_set = BacchusDataset(
            self.cfg, 
            self.train_scans, 
            self.map, 
            split='train'
        )

        val_set = BacchusDataset(
            self.cfg, 
            self.val_scans, 
            self.map,
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

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader
    
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

    def __init__(self, cfg, scans, pc_map, split = None):
        self.cfg = cfg
        self.scans = scans

        self.dataset_size = len(scans)
        
        self.map = pc_map

        self.kd_tree_tar = cKDTree(self.map[:,:3])

        if self.cfg['DATA']['ME_PRUNE']:
            self.map = torch.tensor(self.map[:, :4]).to(torch.float32).reshape(-1, 4)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"
        self.discrepancy = self.cfg["TRAIN"]["DISCREPANCY"] 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __len__(self):
        return self.dataset_size


    def __getitem__(self, idx):
        # Load scan and poses:
        scan_data = self.scans[idx]

        scan_points = torch.tensor(scan_data[:, :3]).to(torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(scan_data[:, 3]).to(torch.float32).reshape(-1, 1)
        # Bind time stamp to scan points
        scan_points = self.add_timestamp(scan_points, util.SCAN_TIMESTAMP)
        # Bind points label in the same tensor 
        scan_points = torch.hstack([scan_points, scan_labels])

        scan_submap_data = scan_points

        if self.discrepancy:
            kd_tree_scan = cKDTree(scan_data[:,:3])
            submap_idx = self.select_closest_points(kd_tree_scan, self.kd_tree_tar)
            submap_points = torch.tensor(self.map[submap_idx, :3]).to(torch.float32).reshape(-1, 3)

            # Submap points labels are not important, so we just create a tensor of ones
            submap_labels = torch.ones(submap_points.shape[0], 1)

            # Bind time stamp to submap points
            submap_points = self.add_timestamp(submap_points, util.MAP_TIMESTAMP)

            # Bind points label in the same tensor 
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

    def select_closest_points(self, kd_tree_ref, kd_tree_tar):
        start_time = time.time()

        indexes = kd_tree_ref.query_ball_tree(kd_tree_tar, self.cfg["MODEL"]["VOXEL_SIZE"])
        
        # Merge the indexes into one array using numpy.concatenate and list comprehension
        merged_indexes = np.concatenate([idx_list for idx_list in indexes])

        # Convert the merged_indexes to int data type
        merged_indexes = merged_indexes.astype(int)

        elapsed_time = time.time() - start_time
        # print('Closest Points time: %.3f s' % elapsed_time)
        
        return merged_indexes

        

    def augment_data(self, scan_map_batch):
        scan_map_batch = rotate_point_cloud(scan_map_batch)
        scan_map_batch = rotate_perturbation_point_cloud(scan_map_batch)
        scan_map_batch = random_flip_point_cloud(scan_map_batch)
        scan_map_batch = random_scale_point_cloud(scan_map_batch)
        return scan_map_batch

if __name__ == "__main__":
    pass
