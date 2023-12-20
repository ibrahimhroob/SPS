#!/usr/bin/env python3

import time
import torch
import numpy as np
import rospy
import message_filters
from nav_msgs.msg import Odometry	
from sensor_msgs.msg import PointCloud2
from sps.datasets import util
from mapmos import MapMOSNet

class MapMos():
    def __init__(self):
        rospy.init_node('mapmos_node')

        # Retrieve parameters from ROS parameter server
        self.odom_frame = rospy.get_param('~odom_frame', "odom")
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/odometry_node/odometry_estimate")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        weights_pth = rospy.get_param('~model_weights_pth', "/sps/c_ws/src/mapmos/checkpoints/mapmos.ckpt")

        # Subscribe to ROS topics
        odom_sub = message_filters.Subscriber(predicted_pose_topic, Odometry)
        scan_sub = message_filters.Subscriber(raw_cloud_topic, PointCloud2)

        ts = message_filters.TimeSynchronizer([odom_sub, scan_sub], 10)
        ts.registerCallback(self.callback)

        # Initialize the publisher
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.mapmos_pub = rospy.Publisher('/debug/raw_cloud_tr', PointCloud2, queue_size=10)
        self.mapmos_map_pub = rospy.Publisher('/debug/mos_map', PointCloud2, queue_size=10)

        # Load the model
        self.model = self.load_model(weights_pth)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pc_map = np.load('/sps/data/maps/base_map.asc.npy')

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


    def select_points_within_radius(self, coordinates, center, radius=30):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= radius)[0]
        return indexes

    def callback(self, odom_msg, scan_msg):
        start_time = time.time()

        scan = util.to_numpy(scan_msg)
        scan_msg_header = scan_msg.header

        odom_msg_stamp = odom_msg.header.stamp.to_sec()
        transformation_matrix = util.to_tr_matrix(odom_msg)

        origin = transformation_matrix[:3, 3]
        pc_map = self.pc_map[self.select_points_within_radius(self.pc_map[:,:3], origin)] 

        ''' Make sure the time stamp for odom and scan are the same!'''
        df = odom_msg_stamp - scan_msg_header.stamp.to_sec()
        if df != 0:
            rospy.logwarn(f"Odom and scan time stamp is out of sync! time difference is {df} sec.")

        ''' Transform the scan '''
        scan_tr = util.transform_point_cloud(scan[:,:3], transformation_matrix)

        scan_points = torch.tensor(scan_tr[:,:3], dtype=torch.float32).reshape(-1, 3).to(self.device)
        map_points = torch.tensor(pc_map[:,:3], dtype=torch.float32).reshape(-1, 3).to(self.device)

        scan_indices = torch.ones(len(scan_points), 1).type_as(scan_points).to(self.device)
        map_indices = torch.zeros(len(map_points), 1).type_as(map_points).to(self.device)

        logits_scan, logits_map = self.model.predict(scan_points, map_points, scan_indices, map_indices)

        scan_lablels = self.model.to_label(logits_scan)

        scan_lablels = scan_lablels.cpu().data.numpy()

        scan = np.hstack((scan[:,:3], scan_lablels.reshape(-1,1)))
        filtered_scan = scan[(scan_lablels == 0)]

        self.scan_pub.publish(util.to_rosmsg(filtered_scan, scan_msg_header))
        self.mapmos_pub.publish(util.to_rosmsg(np.hstack((scan_tr[:,:3], scan_lablels.reshape(-1,1))), scan_msg_header, self.odom_frame))
        self.mapmos_map_pub.publish(util.to_rosmsg(pc_map, scan_msg_header, self.odom_frame))


        end_time = time.time()
        elapsed_time = end_time - start_time

        hz = lambda t: 1 / t if t else 0
        rospy.loginfo(f"T: {elapsed_time:.3f} [{hz(elapsed_time):.2f} Hz], map {len(pc_map):d}, scan {len(scan):d}, filtered {len(filtered_scan):d}")


if __name__ == "__main__":
    MapMos()
