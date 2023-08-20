#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rospy
import ros_numpy

from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2, PointField

import sps.models.models as models

import MinkowskiEngine as ME

''' Constants '''
SCAN_TIMESTAMP = 1
MAP_TIMESTAMP = 0

#assert submap_points.size(-1) == 3, f"Expected 3 columns, but the submap tensor has {submap_points.size(-1)} columns."

def load_model(cfg=None, weights_pth=None):
    assert cfg != None, "cfg is None!"
    assert weights_pth != None, "weights_pth is None!"

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

    return model

def load_point_cloud_map(cfg):
    assert cfg != None, "cfg is None!"

    map_id = cfg["TRAIN"]["MAP"]
    map_pth = os.path.join(str(os.environ.get("DATA")), "maps", map_id)
    __, file_extension = os.path.splitext(map_pth)
    rospy.loginfo('Loading point cloud map, pth: %s' % (map_pth))
    try:
        point_cloud_map = np.load(map_pth) if file_extension == '.npy' else np.loadtxt(map_pth, dtype=np.float32)
        point_cloud_map = torch.tensor(point_cloud_map[:, :4]).to(torch.float32).reshape(-1, 4)
        rospy.loginfo('Point cloud map loaded successfully with %d points', len(point_cloud_map))
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

    ''' Merge the two sparse tensors '''
    union = ME.MinkowskiUnion()
    output = union(scan_sparse, map_sparse)

    ''' Only keep map points that lies in the same voxel as scan points '''
    mask = (output.F[:,0] * output.F[:,1]) == 1

    ''' Prune the merged tensors '''
    pruning = ME.MinkowskiPruning()
    output = pruning(output, mask)

    ''' Retrieve the original coordinates '''
    submap_points = output.coordinates

    ''' dequantize the points '''
    submap_points = submap_points * ds
    
    elapsed_time = time.time() - start_time

    return  submap_points.cpu(), elapsed_time, len(scan_sparse)


def to_rosmsg(data, header, frame_id=None):
    cloud = PointCloud2()
    cloud.header = header
    if frame_id:
        cloud.header.frame_id = frame_id

    ''' Define the fields for the filtered point cloud '''
    filtered_fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)
    ]
    cloud.fields = filtered_fields
    cloud.is_bigendian = False
    cloud.point_step = 16
    cloud.row_step = cloud.point_step * len(data)
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.width = len(data)
    cloud.height = 1

    ''' Create a single array to hold all the point data '''
    point_data = np.array(data, dtype=np.float32)
    cloud.data = point_data.tobytes()

    return cloud


def to_numpy(pointcloud_msg):
    pc = ros_numpy.numpify(pointcloud_msg)
    height, width = pc.shape[0], pc.shape[1] if len(pc.shape) > 1 else 1

    scan = np.zeros((height * width, len(pc.dtype.names)), dtype=np.float32)
    for i, attr in enumerate(pc.dtype.names):
        scan[:, i] = np.resize(pc[attr], height * width)
    return scan


def add_timestamp(data, stamp):
    ones = torch.ones(len(data), 1, dtype=data.dtype)
    data = torch.hstack([data, ones * stamp])
    return data


def infer(scan_points, submap_points, model):
    assert scan_points.size(-1) == 3, f"Expected 3 columns, but the scan tensor has {scan_points.size(-1)} columns."
    assert submap_points.size(-1) == 3, f"Expected 3 columns, but the submap tensor has {submap_points.size(-1)} columns."

    ''' Bind time stamp to scan and submap points '''
    scan_points = add_timestamp(scan_points, SCAN_TIMESTAMP)
    submap_points = add_timestamp(submap_points, MAP_TIMESTAMP)

    ''' Combine scans and map into the same tensor '''
    scan_submap_data = torch.vstack([scan_points, submap_points])

    batch = torch.zeros(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
    tensor = torch.hstack([batch, scan_submap_data]).reshape(-1, 5)

    start_time = time.time()
    with torch.no_grad():
        ''' Move the data tensor to the GPU manually when using the model directly without PyTorch Lightning '''       
        scores = model.forward(tensor.cuda())  
        scores = scores.cpu()

    end_time = time.time()
    elapsed_time = end_time - start_time

    ''' Get the scan scores '''
    scan_scores = scores[:len(scan_points)]

    return scan_scores, elapsed_time


def transform_point_cloud(point_cloud, transformation_matrix):
    # Convert point cloud to homogeneous coordinates
    homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    # Apply transformation matrix
    transformed_coords = np.dot(homogeneous_coords, transformation_matrix.T)
    # Convert back to Cartesian coordinates
    transformed_point_cloud = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]
    return transformed_point_cloud


def inverse_transform_point_cloud(transformed_point_cloud, transformation_matrix):
    # Invert the transformation matrix
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)
    # Convert transformed point cloud to homogeneous coordinates
    homogeneous_coords = np.hstack((transformed_point_cloud, np.ones((transformed_point_cloud.shape[0], 1))))
    # Apply inverse transformation matrix
    original_coords = np.dot(homogeneous_coords, inv_transformation_matrix.T)
    # Convert back to Cartesian coordinates
    original_point_cloud = original_coords[:, :3] / original_coords[:, 3][:, np.newaxis]
    return original_point_cloud


def to_tr_matrix(odom_msg):
    # Extract the translation and orientation from the Odometry message
    x = odom_msg.pose.pose.position.x
    y = odom_msg.pose.pose.position.y
    z = odom_msg.pose.pose.position.z

    qx = odom_msg.pose.pose.orientation.x
    qy = odom_msg.pose.pose.orientation.y
    qz = odom_msg.pose.pose.orientation.z
    qw = odom_msg.pose.pose.orientation.w

    # Create the translation matrix
    translation_matrix = np.array( [[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z],
                                    [0, 0, 0, 1]] )

    # Create the rotation matrix
    rotation_matrix = quaternion_matrix([qx, qy, qz, qw])

    # Compute the transformation matrix
    transformation_matrix = np.dot(translation_matrix, rotation_matrix)

    return transformation_matrix