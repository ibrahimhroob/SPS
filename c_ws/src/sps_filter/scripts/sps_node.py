#!/usr/bin/env python3

import time
import torch
import numpy as np

import rospy

import message_filters
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry	
from sensor_msgs.msg import PointCloud2

from torchmetrics import R2Score

from sps.datasets import util

class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation_node')

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic      = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/odometry_node/odometry_estimate")

        weights_pth = rospy.get_param('~model_weights_pth', "/sps/tb_logs/SPS_ME_Union/version_39/checkpoints/last.ckpt")

        self.odom_frame    = rospy.get_param('~odom_frame', "odom")
        self.epsilon       = rospy.get_param('~epsilon', 0.84)
        self.use_gt_labels = rospy.get_param('~use_gt_labels', False)
        self.pub_submap    = rospy.get_param('~pub_submap', True)
        self.pub_cloud_tr  = rospy.get_param('~pub_cloud_tr', True)

        ''' Subscribe to ROS topics '''
        odom_sub = message_filters.Subscriber(predicted_pose_topic, Odometry)
        scan_sub = message_filters.Subscriber(raw_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([odom_sub, scan_sub], 10)
        ts.registerCallback(self.callback)

        ''' Initialize the publisher '''
        self.scan_pub     = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.submap_pub   = rospy.Publisher('debug/cloud_submap', PointCloud2, queue_size=10)
        self.cloud_tr_pub = rospy.Publisher('debug/raw_cloud_tr', PointCloud2, queue_size=10)
        self.loss_pub     = rospy.Publisher('debug/model_loss', Float32, queue_size=10)
        self.r2_pub       = rospy.Publisher('debug/model_r2', Float32, queue_size=10)

        rospy.loginfo('raw_cloud:      %s', raw_cloud_topic)
        rospy.loginfo('cloud_filtered: %s', filtered_cloud_topic)
        rospy.loginfo('predicted_pose: %s', predicted_pose_topic)
        rospy.loginfo('epsilon:        %f', self.epsilon)

        ''' Load configs '''
        cfg = torch.load(weights_pth)["hyper_parameters"]
        cfg['DATA']['NUM_WORKER'] = 12
        rospy.loginfo(cfg)

        ''' Load the model '''
        self.model = util.load_model(cfg, weights_pth)

        ''' Get VOXEL_SIZE for quantization '''
        self.ds = cfg["MODEL"]["VOXEL_SIZE"]

        ''' Get available device '''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ''' Load point cloud map '''
        cfg["TRAIN"]["MAP"] = 'base_map.asc.npy'
        pc_map_tensor = util.load_point_cloud_map(cfg)
        self.pc_map_coord_feat = util.to_coords_features(pc_map_tensor,
                                                         feature_type='map',
                                                         ds=self.ds, 
                                                         device=self.device)
        
        ''' Define loss and r2score for model evaluation '''
        self.loss = torch.nn.MSELoss().to(self.device)
        self.r2score = R2Score().to(self.device)
        
        ''' Variable to hold point cloud frame '''
        self.scan = None
        self.scan_msg_header = None
        self.scan_received = False

        ''' Create a lock '''
        # self.lock = threading.Lock()

        rospy.spin()


    def callback(self, odom_msg, scan_msg):
        self.scan = util.to_numpy(scan_msg)
        self.scan_msg_header = scan_msg.header

        start_time = time.time()
        odom_msg_stamp = odom_msg.header.stamp.to_sec()

        transformation_matrix = util.to_tr_matrix(odom_msg)

        ''' Step 0: make sure the time stamp for odom and scan are the same!'''
        df = odom_msg_stamp - self.scan_msg_header.stamp.to_sec()
        if df != 0:
            rospy.logwarn(f"Odom and scan time stamp is out of sync! time difference is {df} sec.")

        ''' Step 1: Transform the scan '''
        scan_tr = util.transform_point_cloud(self.scan[:,:3], transformation_matrix)

        ''' Step 2: convert scan points to torch tensor '''
        scan_points = torch.tensor(scan_tr[:,:3], dtype=torch.float32).reshape(-1, 3).to(self.device)
        scan_labels = torch.tensor(self.scan[:, 3], dtype=torch.float32).reshape(-1, 1).to(self.device)

        ''' Step 3: Prune map points by keeping only the points that in the same voxel as of the scan points'''
        start_time = time.time()
        scan_coord_feat = util.to_coords_features(scan_points,
                                                    feature_type='scan',
                                                    ds=self.ds,
                                                    device=self.device)
        submap_points, len_scan_coord = util.prune(self.pc_map_coord_feat, scan_coord_feat, self.ds) 

        prune_time = time.time() - start_time

        ''' Step 4: Infer the stability labels'''
        if self.use_gt_labels:
            rospy.logwarn("The model inference is disabled, and ground truth labels are being used.")
            predicted_scan_labels, infer_time = scan_labels.view(-1), 0.001
        else:
            predicted_scan_labels, infer_time = util.infer(scan_points, submap_points, self.model)

        ''' Step 5: Calculate loss and r2 '''
        loss = self.loss(predicted_scan_labels.view(-1), scan_labels.view(-1))
        r2 = self.r2score(predicted_scan_labels.view(-1), scan_labels.view(-1))
        self.loss_pub.publish(loss)
        self.r2_pub.publish(r2)

        ''' Step 6: Filter the scan points based on the threshold'''
        assert len(predicted_scan_labels) == len(self.scan), f"Predicted scans labels len ({len(predicted_scan_labels)}) does not equal scan len ({len(self.scan)})"
        filtered_scan = self.scan[(predicted_scan_labels.cpu() <= self.epsilon)]
        # filtered_scan = self.scan[(predicted_scan_labels.cpu() <= self.epsilon) and (predicted_scan_labels.cpu() >= 0.05)]
        self.scan_pub.publish(util.to_rosmsg(filtered_scan, self.scan_msg_header))

        ''' Publish the transformed point cloud for debugging '''
        if self.pub_cloud_tr:
            psl = predicted_scan_labels.cpu().data.numpy().reshape(-1,1)
            scan_tr = np.hstack([scan_tr[:,:3], psl])
            self.cloud_tr_pub.publish(util.to_rosmsg(scan_tr, self.scan_msg_header, self.odom_frame))

        ''' Publish the submap points for debugging '''
        submap_points = submap_points.cpu()
        submap_labels = torch.ones(submap_points.shape[0], 1)
        if self.pub_submap:
            submap = torch.hstack([submap_points, submap_labels])
            self.submap_pub.publish(util.to_rosmsg(submap.numpy(), self.scan_msg_header, self.odom_frame))

        ''' Print log message '''
        end_time = time.time()
        elapsed_time = end_time - start_time
        hz = lambda t: 1 / t if t else 0

        log_message = (
            f"T: {elapsed_time:.3f} [{hz(elapsed_time):.2f} Hz] "
            f"P: {prune_time:.3f} [{hz(prune_time):.2f} Hz] "
            f"I: {infer_time:.3f} [{hz(infer_time):.2f} Hz] "
            f"L: {loss:.3f} r2: {r2:.3f} "
            f"N: {len(self.scan):d} n: {len(filtered_scan):d} "
            f"S: {len_scan_coord:d} M: {len(submap_labels):d}"
        )
        rospy.loginfo(log_message)
        

if __name__ == '__main__':
    SPS_node = SPS()
    



