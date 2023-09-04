#!/usr/bin/env python3

import time
import torch
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from sps.datasets import util
from mapmos import MapMOSNet

class MapMos():
    def __init__(self):
        rospy.init_node('mapmos_node')

        # Retrieve parameters from ROS parameter server
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/odometry_node/frame")
        map_cloud_topic = rospy.get_param('~local_map', "/odometry_node/local_map")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/debug/mapmos")
        weights_pth = rospy.get_param('~model_weights_pth', "/sps/c_ws/src/mapmos/checkpoints/mapmos.ckpt")

        # Subscribe to ROS topics
        scan_sub = message_filters.Subscriber(raw_cloud_topic, PointCloud2)
        map_sub = message_filters.Subscriber(map_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([scan_sub, map_sub], 10)
        ts.registerCallback(self.callback)

        # Initialize the publisher
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)

        # Load the model
        self.model = self.load_model(weights_pth)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rospy.spin()

    def load_model(self, path, voxel_size=0.1):
        # Load model
        state_dict = torch.load(path)["state_dict"].items()
        state_dict = {
            k.replace("mos.MinkUNet.", ""): v
            for k, v in torch.load(path)["state_dict"].items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model = MapMOSNet(voxel_size)
        model.MinkUNet.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        model.freeze()
        rospy.loginfo("Model loaded successfully!")
        return model

    def callback(self, scan_msg, map_msg):
        scan = util.to_numpy(scan_msg)
        pc_map = util.to_numpy(map_msg)
        scan_msg_header = scan_msg.header
        start_time = time.time()

        scan_points = torch.tensor(scan[:,:3], dtype=torch.float32).reshape(-1, 3).to(self.device)
        map_points = torch.tensor(pc_map[:,:3], dtype=torch.float32).reshape(-1, 3).to(self.device)

        scan_indices = torch.ones(len(scan_points), 1).type_as(scan_points).to(self.device)
        map_indices = torch.zeros(len(map_points), 1).type_as(map_points).to(self.device)

        logits_scan, logits_map = self.model.predict(scan_points, map_points, scan_indices, map_indices)

        scan_lablels = self.model.to_label(logits_scan)

        scan_lablels = scan_lablels.cpu().data.numpy()

        scan = np.hstack((scan[:,:3], scan_lablels.reshape(-1,1)))

        self.scan_pub.publish(util.to_rosmsg(scan, scan_msg_header))

        end_time = time.time()
        elapsed_time = end_time - start_time

        hz = lambda t: 1 / t if t else 0
        rospy.loginfo(f"T: {elapsed_time:.3f} [{hz(elapsed_time):.2f} Hz], map {len(pc_map):d}, scan {len(scan):d}")


if __name__ == "__main__":
    MapMos()
