#!/usr/bin/env python3

import time
import torch
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from sps.datasets import util
from mos4d import MOS4DNet

class MOS4D():
    def __init__(self):
        rospy.init_node('mos4d_node')

        # Retrieve parameters from ROS parameter server
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/odometry_node/frame")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/4dmos")
        weights_pth = rospy.get_param('~model_weights_pth', "/sps/c_ws/src/mos4d/checkpoints/10_scans.ckpt")

        # Subscribe to ROS topics
        scan_sub = message_filters.Subscriber(raw_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([scan_sub], 10)
        ts.registerCallback(self.callback)

        # Initialize the publisher
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.lidar_buffer = []

        # Load the model
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

    def callback(self, scan_msg):
        self.scan = util.to_numpy(scan_msg)
        self.scan_msg_header = scan_msg.header
        start_time = time.time()

        # Append scan to the buffer
        scan_index_array = np.ones(len(self.scan)).reshape(-1, 1) * self.scan_index
        self.scan = np.hstack([self.scan, scan_index_array])
        self.scan_index += 1
        self.lidar_buffer.append(self.scan)

        if len(self.lidar_buffer) > 10:
            self.lidar_buffer.pop(0)

        # Merge the scans in the buffer into a single tensor
        merged_scans = np.vstack(self.lidar_buffer)
        merged_scans = torch.from_numpy(merged_scans).squeeze().to(torch.float32).cuda()

        # Add batch index and pass through the model
        coordinates = torch.hstack([torch.zeros(len(merged_scans)).reshape(-1, 1).type_as(merged_scans), merged_scans])
        predicted_logits = self.model.forward(coordinates)

        end_time = time.time()
        elapsed_time = end_time - start_time

        hz = lambda t: 1 / t if t else 0
        rospy.loginfo(f"T: {elapsed_time:.3f} [{hz(elapsed_time):.2f} Hz]")

        predicted_logits = (predicted_logits > 0).int().cpu().data.numpy()
        scan_len = len(self.scan)
        scan_labels = predicted_logits[-scan_len:]
        self.scan = np.hstack((self.scan[:,:3], scan_labels.reshape(-1, 1)))
        self.scan_pub.publish(util.to_rosmsg(self.scan, self.scan_msg_header, 'odom'))

if __name__ == "__main__":
    MOS4D()
