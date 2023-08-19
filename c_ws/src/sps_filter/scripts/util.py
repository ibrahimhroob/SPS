#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rospy
import ros_numpy

from nav_msgs.msg import Odometry	
from std_msgs.msg import Float32
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2, PointField

import sps.models.models as models

from torchmetrics import R2Score

import MinkowskiEngine as ME


def load_model(cfg, weights_pth):
    state_dict = {
        k.replace("model.MinkUNet.", ""): v
        for k, v in torch.load(weights_pth)["state_dict"].items()
    }
    state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
    model = models.SPSNet(cfg)
    model.model.MinkUNet.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    model.freeze()

    rospy.loginfo("Model loaded successfully!")

    return model, cfg

def load_point_cloud_map(cfg):
    map_id = cfg["TRAIN"]["MAP"]
    map_pth = os.path.join(str(os.environ.get("DATA")), "maps", map_id)
    __, file_extension = os.path.splitext(map_pth)
    rospy.loginfo('Loading point cloud map, pth: %s' % (map_pth))
    try:
        point_cloud_map = np.load(map_pth) if file_extension == '.npy' else np.loadtxt(map_pth, dtype=np.float32)
        point_cloud_map = torch.tensor(point_cloud_map[:, :4]).to(torch.float32).reshape(-1, 4)
        rospy.loginfo('Point cloud map loaded successfully!')
    except:
        rospy.logerr('Failed to load point cloud map from %s', map_pth)
        sys.exit()

    return point_cloud_map


def prune_map_points(ds, scan_data, point_cloud_map, device):
    start_time = time.time()
    quantization = torch.Tensor([ds, ds, ds]).to(device)

    '''we are only intrested in the coordinates of the points, thus we are keeping only the xyz columns'''
    scan = scan_data[:,:3]
    pc_map = point_cloud_map[:,:3]

    scan_coords = torch.div(scan, quantization.type_as(scan)).int().to(device)
    scan_features = torch.zeros(scan.shape[0], 2).to(device)
    scan_features[:, 0] = 1

    map_coord = torch.div(pc_map, quantization.type_as(pc_map)).int().to(device)
    map_features = torch.zeros(pc_map.shape[0], 2).to(device)
    map_features[:, 1] = 1

    map_sparse = ME.SparseTensor(
        features=map_features, 
        coordinates=map_coord,
        )

    scan_sparse = ME.SparseTensor(
        features=scan_features, 
        coordinates=scan_coords,
        coordinate_manager = map_sparse.coordinate_manager
        )

    # Merge the two sparse tensors
    union = ME.MinkowskiUnion()
    output = union(scan_sparse, map_sparse)

    # Only keep map points that lies in the same voxel as scan points
    mask = (output.F[:,0] * output.F[:,1]) == 1

    # Prune the merged tensors 
    pruning = ME.MinkowskiPruning()
    output = pruning(output, mask)

    # Retrieve the original coordinates
    submap_points = output.coordinates

    # dequantize the points
    submap_points = submap_points * ds
    
    elapsed_time = time.time() - start_time

    return  submap_points.cpu(), elapsed_time, len(scan_sparse)
