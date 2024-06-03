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
    def __init__(self, cfg, test=False):
        super(BacchusModule, self).__init__()
        self.cfg = cfg
        self.test = test
        self.root_dir = str(os.environ.get("DATA"))
       

        if self.test:
            print('Loading testing data ...')
            test_seqs = self.cfg['DATA']['SPLIT']['TEST']
            test_scans = self.get_scans(test_seqs)
            self.test_scans = self.cash_scans(test_scans)
        else:
            print('Loading training data ...')
            train_seqs = self.cfg['DATA']['SPLIT']['TRAIN']
            train_scans = self.get_scans(train_seqs)
            self.train_scans = self.cash_scans(train_scans)

            print('Loading validating data ...')
            val_seqs = self.cfg['DATA']['SPLIT']['VAL']
            val_scans = self.get_scans(val_seqs)
            self.val_scans = self.cash_scans(val_scans)


    def cash_scans(self, scans_pth):
        scans_data = []
        # Zip the two lists together and iterate with tqdm
        for scan_pth in tqdm(scans_pth, total=len(scans_pth)):
            # Load scan and poses:
            scan_data = np.load(scan_pth)

            scans_data.append(scan_data)

        return scans_data


    def get_scans(self, seqs):
        seq_scans = []

        for sequence in seqs:
            scans_dir = os.path.join(self.root_dir, "sequence", sequence, "scans")

            scans = sorted([os.path.join(scans_dir, file) for file in os.listdir(scans_dir)])

            seq_scans.extend(scans)
        return seq_scans


    def setup(self, stage=None):
        ########## Point dataset splits
        if self.test:
            test_set = BacchusDataset(
                self.cfg, 
                self.test_scans, 
            )
        else:
            train_set = BacchusDataset(
                self.cfg, 
                self.train_scans, 
                split='train'
            )

            val_set = BacchusDataset(
                self.cfg, 
                self.val_scans, 
            )

        ########## Generate dataloaders and iterables
        if self.test:
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
        else:
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

    def __init__(self, cfg, scans, split = None):
        self.cfg = cfg
        self.scans = scans

        self.dataset_size = len(scans)

        self.augment = self.cfg["TRAIN"]["AUGMENTATION"] and split == "train"

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

        # Augment the points 
        if self.augment:
            scan_points[:,:3] = self.augment_data(scan_points[:,:3])

        return scan_points

    def add_timestamp(self, data, stamp):
        ones = torch.ones(len(data), 1, dtype=data.dtype)
        data = torch.hstack([data, ones * stamp])
        return data

    def augment_data(self, scan_map_batch):
        scan_map_batch = rotate_point_cloud(scan_map_batch)
        scan_map_batch = rotate_perturbation_point_cloud(scan_map_batch)
        scan_map_batch = random_flip_point_cloud(scan_map_batch)
        scan_map_batch = random_scale_point_cloud(scan_map_batch)
        return scan_map_batch

if __name__ == "__main__":
    pass
