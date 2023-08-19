#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rospy
import ros_numpy

from nav_msgs.msg import Odometry	
from std_msgs.msg import Float32, String
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2, PointField

import sps.models.models as models

from torchmetrics import R2Score

import MinkowskiEngine as ME

''' Constants '''
SCAN_TIMESTAMP = 1
MAP_TIMESTAMP = 0


class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation')

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/odometry_node/frame_estimated")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered_kiss")
        self.weights_pth = rospy.get_param('~model_weights_pth', "/sps/tb_logs/SPS_ME_Union/version_39/checkpoints/last.ckpt")
        self.threshold_dynamic = rospy.get_param('~epsilon', 0.90)

        ''' Subscribe to ROS topics '''
        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback_points)

        rospy.Subscriber("/odometry_node/local_map", PointCloud2, self.global_map)

        ''' Initialize the publisher '''
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.submap_pub = rospy.Publisher('/cloud_submap_kiss', PointCloud2, queue_size=10)
        self.loss_pub = rospy.Publisher('model_loss_kiss', Float32, queue_size=10)
        self.r2_pub = rospy.Publisher('model_r2_kiss', Float32, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('cloud_filtered: %s', filtered_cloud_topic)
        rospy.loginfo('cloud_submap: %s', '/cloud_submap')
        rospy.loginfo('Upper threshold: %f', self.threshold_dynamic)

        ''' Load the model and the associated configs '''
        self.model, self.cfg = self.load_model()

        self.ds = self.cfg["MODEL"]["VOXEL_SIZE"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.MSELoss()
        self.r2score = R2Score()
        self.odom_msg_stamp = 0	
        self.transformation = None

        ''' Load point cloud map '''
        # self.point_cloud_map = self.load_point_cloud_map()
        self.point_cloud_map = None

        rospy.spin()


    def load_model(self):
        cfg = torch.load(self.weights_pth)["hyper_parameters"]

        state_dict = {
            k.replace("model.MinkUNet.", ""): v
            for k, v in torch.load(self.weights_pth)["state_dict"].items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model = models.SPSNet(cfg)
        model.model.MinkUNet.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        model.freeze()

        rospy.loginfo("Model loaded successfully!")

        return model, cfg


    def load_point_cloud_map(self):
        map_id = self.cfg["TRAIN"]["MAP"]
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


    def prune_map_points(self, scan_data):
        start_time = time.time()
        quantization = torch.Tensor([self.ds, self.ds, self.ds]).to(self.device)

        '''we are only intrested in the coordinates of the points, thus we are keeping only the xyz columns'''
        scan = scan_data[:,:3]
        pc_map = self.point_cloud_map[:,:3]

        scan_coords = torch.div(scan, quantization.type_as(scan)).int().to(self.device)
        scan_features = torch.zeros(scan.shape[0], 2).to(self.device)
        scan_features[:, 0] = 1

        map_coord = torch.div(pc_map, quantization.type_as(pc_map)).int().to(self.device)
        map_features = torch.zeros(pc_map.shape[0], 2).to(self.device)
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
        submap_points = submap_points * self.ds
        
        elapsed_time = time.time() - start_time

        return  submap_points.cpu(), elapsed_time, len(scan_sparse)


    def to_rosmsg(self, data, header, frame_id=None):
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


    def add_timestamp(self, data, stamp):
        ones = torch.ones(len(data), 1, dtype=data.dtype)
        data = torch.hstack([data, ones * stamp])
        return data


    def infer(self, scan_points, submap_points):
        assert scan_points.size(-1) == 3, f"Expected 3 columns, but the scan tensor has {scan_points.size(-1)} columns."
        assert submap_points.size(-1) == 3, f"Expected 3 columns, but the submap tensor has {submap_points.size(-1)} columns."

        ''' Bind time stamp to scan and submap points '''
        scan_points = self.add_timestamp(scan_points, SCAN_TIMESTAMP)
        submap_points = self.add_timestamp(submap_points, MAP_TIMESTAMP)

        ''' Combine scans and map into the same tensor '''
        scan_submap_data = torch.vstack([scan_points, submap_points])

        batch = torch.zeros(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
        tensor = torch.hstack([batch, scan_submap_data]).reshape(-1, 5)

        start_time = time.time()
        with torch.no_grad():
            ''' Move the data tensor to the GPU manually when using the model directly without PyTorch Lightning '''       
            scores = self.model.forward(tensor.cuda())  
            scores = scores.cpu()

        end_time = time.time()
        elapsed_time = end_time - start_time

        ''' Get the scan scores '''
        scan_scores = scores[:len(scan_points)]

        return scan_scores, elapsed_time


    def global_map(self, pointcloud_msg):
        if self.point_cloud_map == None:
            pc = ros_numpy.numpify(pointcloud_msg)
            height = pc.shape[0]
            width = pc.shape[1] if len(pc.shape) > 1 else 1
            names = pc.dtype.names
            gmap = np.zeros((height * width, len(names)), dtype=np.float32)
            for i, attr in enumerate(names):
                gmap[:, i] = np.resize(pc[attr], height * width)

            ''' Only interested in xyz coordinates, thus we only taking the first 3 axis'''
            self.point_cloud_map = torch.tensor(gmap[:, :3]).to(torch.float32).reshape(-1, 3)

            rospy.loginfo('Global map revieved with %d points' % (len(self.point_cloud_map)))

    def callback_points(self, pointcloud_msg):
        start_time = time.time()
        pc = ros_numpy.numpify(pointcloud_msg)
        height, width = pc.shape[0], pc.shape[1] if len(pc.shape) > 1 else 1

        scan = np.zeros((height * width, len(pc.dtype.names)), dtype=np.float32)
        for i, attr in enumerate(pc.dtype.names):
            scan[:, i] = np.resize(pc[attr], height * width)
        scan_len_before = len(scan)

        ''' convert scan points to torch tensor '''
        scan_points = torch.tensor(scan[:, :3], dtype=torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(scan[:, 3], dtype=torch.float32).reshape(-1, 1)

        ''' Prune map points by keeping only the points that in the same voxel as of the scan points'''
        submap_points, prune_time, len_scan_coord = self.prune_map_points(scan_points)

        ''' Infere the stability labels '''
        # predicted_scan_labels, infer_time = self.infer(scan_points, submap_points)
        # predicted_scan_labels = predicted_scan_labels.numpy()
        predicted_scan_labels, infer_time = scan_labels, 0.001
        predicted_scan_labels = predicted_scan_labels.reshape(-1)

        ''' Calculate loss and r2 '''
        loss = self.loss(predicted_scan_labels.view(-1), scan_labels.view(-1))
        r2 = self.r2score(predicted_scan_labels.view(-1), scan_labels.view(-1))
        self.loss_pub.publish(loss)
        self.r2_pub.publish(r2)

        ''' Filter the scan points based on the threshold'''
        assert len(predicted_scan_labels) == len(scan), f"Predicted scans labels len ({len(predicted_scan_labels)}) does not equal scan len ({len(scan)})"
        filtered_scan = scan[(predicted_scan_labels < self.threshold_dynamic)]
        self.scan_pub.publish(self.to_rosmsg(filtered_scan, pointcloud_msg.header))

        ''' Publish the submap points for debugging '''
        submap_labels = torch.ones(submap_points.shape[0], 1)
        submap = torch.hstack([submap_points, submap_labels])
        self.submap_pub.publish(self.to_rosmsg(submap.numpy(), pointcloud_msg.header))

        scan_len_after = len(filtered_scan)

        ''' Print log message '''
        end_time = time.time()
        elapsed_time = end_time - start_time
        hz = lambda t: 1 / t if t else 0

        log_message = (
            f"T: {elapsed_time:.3f} sec [{hz(elapsed_time):.2f} Hz]. "
            f"P: {prune_time:.3f} sec [{hz(prune_time):.2f} Hz]. "
            f"I: {infer_time:.3f} sec [{hz(infer_time):.2f} Hz]. "
            f"L: {loss:.3f}, r2: {r2:.3f}. "
            f"N: {scan_len_before:d}, n: {scan_len_after:d}, "
            f"S: {len_scan_coord:d}, "
            f"M: {len(submap_labels):d}"
        )
        rospy.loginfo(log_message)


if __name__ == '__main__':
    SPS_node = SPS()
    


