#!/usr/bin/env python3

import time
import torch
import numpy as np

import rospy

from nav_msgs.msg import Odometry	
from sensor_msgs.msg import PointCloud2

from torchmetrics import R2Score

import util


class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation_hdl')

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        self.weights_pth = rospy.get_param('~model_weights_pth', "/sps/tb_logs/SPS_ME_Union/version_39/checkpoints/last.ckpt")
        self.threshold_dynamic = rospy.get_param('~epsilon', 0.85)
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/ndt/predicted/odom")

        ''' Subscribe to ROS topics '''
        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback_points)
        rospy.Subscriber(predicted_pose_topic, Odometry, self.callback_odom)

        ''' Initialize the publisher '''
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.cloud_tr_pub = rospy.Publisher('/cloud_tr_debug', PointCloud2, queue_size=10)
        self.submap_pub = rospy.Publisher('/cloud_submap', PointCloud2, queue_size=10)
        # self.submap_base_pub = rospy.Publisher('/cloud_submap_base', PointCloud2, queue_size=10)
        # self.loss_pub = rospy.Publisher('model_loss', Float32, queue_size=10)
        # self.r2_pub = rospy.Publisher('model_r2', Float32, queue_size=10)
        # self.scan_filter_ratio = rospy.Publisher('scan_filter_ratio', Float32, queue_size=10)
        # self.map_to_scan_ratio = rospy.Publisher('map_to_scan_ratio', Float32, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('cloud_filtered: %s', filtered_cloud_topic)
        rospy.loginfo('cloud_submap: %s', '/cloud_submap')
        rospy.loginfo('Upper threshold: %f', self.threshold_dynamic)
        rospy.loginfo('Predicted pose: %s', predicted_pose_topic)

        ''' Load the model and the associated configs '''
        self.cfg = torch.load(self.weights_pth)["hyper_parameters"]
        self.model = util.load_model(self.cfg, self.weights_pth)

        ''' Load point cloud map '''
        self.point_cloud_map = util.load_point_cloud_map(self.cfg)

        self.ds = self.cfg["MODEL"]["VOXEL_SIZE"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.loss = torch.nn.MSELoss()
        self.r2score = R2Score()
        
        ''' Variables to hold odom stamp and transformation'''
        self.odom_msg_stamp = 0	
        self.transformation = None

        ''' Variable to hold point cloud frame '''
        self.scan = None
        self.scan_msg_header = 0

        rospy.spin()


    def callback_points(self, pointcloud_msg):
        self.scan = util.to_numpy(pointcloud_msg)
        self.scan_msg_header = pointcloud_msg.header


    def callback_odom(self, odom_msg):
        start_time = time.time()
        self.odom_msg_stamp = odom_msg.header.stamp.to_sec()

        transformation_matrix = util.to_tr_matrix(odom_msg)

        ''' Step 0: make sure the time stamp for odom and scan are the same!'''
        df = self.odom_msg_stamp - self.scan_msg_header.stamp.to_sec()
        assert df == 0, f"Odom and scan time stamp is out of sync! time difference is {df} sec."

        ''' Step 1: Transform the scan '''
        scan_tr = util.transform_point_cloud(self.scan[:,:3], transformation_matrix)

        ''' Step 2: convert scan points to torch tensor '''
        scan_points = torch.tensor(scan_tr[:,:3], dtype=torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(self.scan[:, 3], dtype=torch.float32).reshape(-1, 1)

        ''' Step 3: Prune map points by keeping only the points that in the same voxel as of the scan points'''
        submap_points, prune_time, len_scan_coord = util.prune_map_points(self.ds, scan_points, self.point_cloud_map, self.device)
        
        ''' Step 4: Infer the stability labels'''
        predicted_scan_labels, infer_time = util.infer(scan_points, submap_points, self.model)
        predicted_scan_labels = predicted_scan_labels.reshape(-1)
        
        ''' The following line is for debugging mainly to test the model with the true lables!! '''
        # predicted_scan_labels, infer_time = scan_labels.view(-1), 0.001

        ''' Step 5: Calculate loss and r2 '''
        loss = self.loss(predicted_scan_labels, scan_labels.view(-1))
        r2 = self.r2score(predicted_scan_labels, scan_labels.view(-1))
        # self.loss_pub.publish(loss)
        # self.r2_pub.publish(r2)

        ''' Step 6: Filter the scan points based on the threshold'''
        # self.scan[:, 3] = predicted_scan_labels = predicted_scan_labels.numpy()
        assert len(predicted_scan_labels) == len(self.scan), f"Predicted scans labels len ({len(predicted_scan_labels)}) does not equal scan len ({len(self.scan)})"
        filtered_scan = self.scan[(predicted_scan_labels < self.threshold_dynamic)]
        self.scan_pub.publish(self.to_rosmsg(filtered_scan, self.scan_msg_header))


        ''' Publish the transformed point cloud for debugging '''
        # scan_tr = np.hstack([scan_tr[:,:3], predicted_scan_labels.numpy()[0]])
        # self.cloud_tr_pub.publish(self.to_rosmsg(scan_tr, self.scan_msg_header, 'map'))

        ''' Publish the submap points for debugging '''
        submap_labels = torch.ones(submap_points.shape[0], 1)
        submap = torch.hstack([submap_points, submap_labels])
        self.submap_pub.publish(self.to_rosmsg(submap.numpy(), self.scan_msg_header, 'map'))

        ''' Publish the submap points to the original coordinates '''
        # submap = submap.numpy()
        # submap[:,:3] = self.inverse_transform_point_cloud(submap[:,:3], self.transformation)
        # self.submap_base_pub.publish(self.to_rosmsg(submap, pointcloud_msg.header, 'os_sensor'))

        '''Publish scan filter ration and map to scan ratio'''
        scan_len_before = len(self.scan)
        scan_len_after = len(filtered_scan)
        scan_filter_ratio = (scan_len_after/scan_len_before) * 100.0
        map_to_scan_ratio = (len(submap_labels) / len_scan_coord) * 100.0
        # self.scan_filter_ratio.publish(scan_filter_ratio)
        # self.map_to_scan_ratio.publish(map_to_scan_ratio)

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
    



