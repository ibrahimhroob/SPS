#!/usr/bin/env python3

import os
import sys
import time
import torch
import struct
import importlib
import numpy as np

import rospy
import ros_numpy
import sensor_msgs.point_cloud2 as pc2

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

import mos4d.models.models as models

#####################################################################
SCAN_TIMESTAMP = 1
MAP__TIMESTAMP = 0

#####################################################################
class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation')

        self.odom_msg_stamp = 0
        self.center = [0, 0, 0]
        self.cfg = None

        raw_cloud_topic = rospy.get_param('~raw_cloud')
        filtered_cloud_topic = rospy.get_param('~filtered_cloud')
        self.weights_pth = rospy.get_param('~model_weights_pth')
        predicted_pose_topic = rospy.get_param('~predicted_pose')
        self.threshold_dynamic = rospy.get_param('~epsilon')

        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback_points)
        rospy.Subscriber(predicted_pose_topic, Odometry, self.callback_odom)

        # Initialize the publisher
        self.pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('filtered_cloud: %s', filtered_cloud_topic)
        rospy.loginfo('Upper threshold: %f', self.threshold_dynamic)
        rospy.loginfo('Predicted pose: %s', predicted_pose_topic)

        # The model need to be loaded before the map in order to init the self.cfg var 
        self.model = self.load_model()

        # Map path
        map_str = self.cfg["TRAIN"]["MAP"]
        map_pth = os.path.join(str(os.environ.get("DATA")), "maps", map_str)

        # Load point cloud map
        try:
            self.point_cloud_map = np.loadtxt(map_pth)
            rospy.loginfo('Point cloud map loaded successfuly')
        except:
            rospy.logerr('Faild to load point cloud map form %s', map_pth)
            sys.exit()

        rospy.spin()

    def callback_odom(self, msg):
        self.odom_msg_stamp = msg.header.stamp

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z

        self.center = [x, y, z]

    def callback_points(self, pointcloud_msg):
        assert pointcloud_msg.header.stamp == self.odom_msg_stamp, 'Pose and point cloud messeges should have the same time stamp!!'
        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        try:
            width = pc.shape[1]
        except:
            width = 1
        scan = np.zeros((height * width, 4), dtype=float)
        scan[:, 0] = np.resize(pc['x'], height * width)
        scan[:, 1] = np.resize(pc['y'], height * width)
        scan[:, 2] = np.resize(pc['z'], height * width)
        scan[:, 3] = np.resize(pc['intensity'], height * width)

        # Get submap points
        submap_indices = self.select_points_within_radius(self.point_cloud_map[:,:3], self.center)
        submap = self.point_cloud_map[submap_indices, :3]
        submap_labels = self.point_cloud_map[submap_indices, 3]

        scan_indices = self.select_points_within_radius(scan[:,:3], self.center)
        scan = scan[scan_indices, :3]
        scan_labels = scan[scan_indices, 3]

        # Infere the stability labels
        scan = self.infer(scan, submap, scan_labels, submap_labels)

        filtered_cloud = self.to_rosmsg(scan, pointcloud_msg.header)

        self.pub.publish(filtered_cloud)

    def load_model(self):
        self.cfg = torch.load(self.weights_pth)["hyper_parameters"]

        ckpt = torch.load(self.weights_pth)
        model = models.MOSNet(self.cfg)
        model.load_state_dict(ckpt["state_dict"])
        model = model.cuda()
        model.eval()
        model.freeze()

        rospy.loginfo("Model loaded successfully!")

        return model
    
    def select_points_within_radius(self, coordinates, center):
        # Calculate the Euclidean distance from each point to the center
        distances = np.sqrt(np.sum((coordinates - center) ** 2, axis=1))
        # Select the indexes of points within the radius
        indexes = np.where(distances <= self.cfg["DATA"]["RADIUS"])[0]
        return indexes
    
    def to_rosmsg(self, data, header):
        filtered_cloud = PointCloud2()
        filtered_cloud.header = header

        # Define the fields for the filtered point cloud
        filtered_fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)]

        filtered_cloud.fields = filtered_fields
        filtered_cloud.is_bigendian = False
        filtered_cloud.point_step = 16
        filtered_cloud.row_step = filtered_cloud.point_step * len(data)
        filtered_cloud.is_bigendian = False
        filtered_cloud.is_dense = True
        filtered_cloud.width = len(data)
        filtered_cloud.height = 1


        # Filter the point cloud based on intensity
        for point in data:
            filtered_cloud.data += struct.pack('ffff', point[0], point[1], point[2], point[3])

        return filtered_cloud

    def add_timestamp(self, data, stamp):
        ones = torch.ones(len(data), 1, dtype=data.dtype)
        data = torch.hstack([data, ones * stamp])
        return data
    
    def infer(self, scan, submap, scan_labels, submap_labels):
        start_time = time.time()

        # Prepare the data for the model, it need to be like this [Batch, X, Y, Z, time, Labels]
        scan_points = torch.tensor(scan).to(torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(scan_labels).to(torch.float32).reshape(-1, 1)
        submap_points = torch.tensor(submap).to(torch.float32).reshape(-1, 3)
        submap_labels = torch.tensor(submap_labels).to(torch.float32).reshape(-1, 1)

        # Bind time stamp to scan and submap points
        scan_points = self.add_timestamp(scan_points, SCAN_TIMESTAMP)
        submap_points = self.add_timestamp(submap_points, MAP__TIMESTAMP)

        # Bind points label in the same tensor 
        scan_points = torch.hstack([scan_points, scan_labels])
        submap_points = torch.hstack([submap_points, submap_labels])

        # Bind scans and map in the same tensor 
        scan_submap_data = torch.vstack([scan_points, submap_points])

        batch = torch.zeros(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
        tensor = torch.hstack([batch, scan_submap_data]).reshape(-1, 5)

        scores = self.model(tensor)

        scan_scores = scores[:len(scan_labels)]

        # Find the indecies where the score is less than the threshold
        scan_points = scan_points[(scan_scores <= self.threshold_dynamic)]

        end_time = time.time()
        elapsed_time = end_time - start_time
        rospy.loginfo("Frame inference and filter processing time: {:.4f} seconds [{:.2f} Hz]".format(elapsed_time, 1/elapsed_time))

        # Convert the torch tensor back to numpy array
        scan_points = scan_points.numpy()

        return scan_points


if __name__ == '__main__':
    SPS_node = SPS()
    



