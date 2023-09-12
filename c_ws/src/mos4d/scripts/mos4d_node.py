#!/usr/bin/env python3

import re
import sys
import time
import torch
import numpy as np
from mos4d import MOS4DNet

import rospy

import message_filters
from nav_msgs.msg import Odometry	
from sensor_msgs.msg import PointCloud2

from sps.datasets import util

class MOS4D():
    def __init__(self):
        rospy.init_node('mos4d_node')

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic      = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/odometry_node/odometry_estimate")

        weights_pth = rospy.get_param('~model_weights_pth', "/sps/c_ws/src/mos4d/checkpoints/10_scans.ckpt")

        self.filter = rospy.get_param('~filter', True)

        # Use regular expressions to find the integer
        self.buffer_size = re.search(r'(\d+)_scans\.ckpt', weights_pth)

        if self.buffer_size:
            self.buffer_size = int(self.buffer_size.group(1))
            rospy.loginfo('Scan buffer size: %d' % self.buffer_size)
        else:
            rospy.logerr("Buffer size not found in the path.")
            sys.exit()

        ''' Subscribe to ROS topics '''
        odom_sub = message_filters.Subscriber(predicted_pose_topic, Odometry)
        scan_sub = message_filters.Subscriber(raw_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([scan_sub, odom_sub], 10)
        ts.registerCallback(self.callback)

        ''' Initialize the publisher '''
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.mos4d_pub = rospy.Publisher('/debug/mos4d', PointCloud2, queue_size=10)

        self.lidar_buffer = []

        ''' Load the model '''
        self.model = self.load_model(weights_pth)

        self.scan_index = 0

        rospy.spin()


    def load_model(self, path, voxel_size=0.1):
        # Load model
        state_dict = {
            k.replace("model.MinkUNet.", ""): v
            for k, v in torch.load(path)["state_dict"].items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model = MOS4DNet(voxel_size)
        model.MinkUNet.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        model.freeze()

        rospy.loginfo("Model loaded successfully!")

        return model
    
    def callback(self, scan_msg, odom_msg):
        self.scan = util.to_numpy(scan_msg)
        self.scan_msg_header = scan_msg.header

        start_time = time.time()
        odom_msg_stamp = odom_msg.header.stamp.to_sec()

        transformation_matrix = util.to_tr_matrix(odom_msg)

        ''' Make sure the time stamp for odom and scan are the same!'''
        df = odom_msg_stamp - self.scan_msg_header.stamp.to_sec()
        if df != 0:
            rospy.logwarn(f"Odom and scan time stamp is out of sync! time difference is {df} sec.")

        ''' Transform the scan '''
        scan_tr = util.transform_point_cloud(self.scan[:,:3], transformation_matrix)

        ''' Append scan to the buffer '''
        scan_tr = np.hstack(
            [
                scan_tr, 
                np.ones(len(scan_tr)).reshape(-1, 1)*self.scan_index
            ]
        )
        self.scan_index += 1
        self.lidar_buffer.append(scan_tr)

        if len(self.lidar_buffer) > self.buffer_size:
            self.lidar_buffer.pop(0)

        # Merge the scans in the buffer into a single tensor
        merged_scans = np.vstack(self.lidar_buffer)
        merged_scans = torch.from_numpy(merged_scans).squeeze().to(torch.float32).cuda()

        # Add batch index and pass through the model
        coordinates = torch.hstack([torch.zeros(len(merged_scans)).reshape(-1, 1).type_as(merged_scans), merged_scans])
        predicted_logits = self.model.forward(coordinates) if self.filter else torch.zeros(len(merged_scans))

        end_time = time.time()
        elapsed_time = end_time - start_time

        hz = lambda t: 1 / t if t else 0
        rospy.loginfo(f"T: {elapsed_time:.3f} [{hz(elapsed_time):.2f} Hz]")

        predicted_logits = (predicted_logits > 0).int().cpu().data.numpy()
        scan_len = len(self.scan)
        scan_labels = predicted_logits[-scan_len:]
        self.scan = np.hstack((self.scan[:,:3], scan_labels.reshape(-1, 1)))
        filtered_scan = self.scan[(scan_labels == 0)] if self.filter else self.scan
        self.scan_pub.publish(util.to_rosmsg(filtered_scan, self.scan_msg_header, 'odom'))

        self.mos4d_pub.publish(util.to_rosmsg(np.hstack((scan_tr[:,:3], scan_labels.reshape(-1, 1))), self.scan_msg_header, 'odom'))



if __name__ == "__main__":
    MOS4D()
