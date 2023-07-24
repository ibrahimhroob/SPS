#!/usr/bin/env python3

import os
import sys
import time
import tqdm
import numpy as np

import rospy

from sensor_msgs.msg import PointCloud2, PointField

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Quaternion
from tf.transformations import quaternion_from_matrix
'''
Note, this code is mainly for debugging in order to debug the filter node, so here we
publish the true labelled scan and the true pose, this will help dubugging the submap
generation and the filter health overall. 

'''

class PubScans:
    def __init__(self):
        rospy.init_node('Labelled_scans_publisher')

        ''' Retrieve parameters from ROS parameter server '''
        cloud_topic = rospy.get_param('~raw_cloud', "/ndt/predicted/aligned_points")
        scans_pth   = rospy.get_param('~scans_pth', "/sps/data/sequence/20220420/scans")
        poses_pth   = rospy.get_param('~poses_pth', "/sps/data/sequence/20220420/poses")
        pub_rate    = rospy.get_param('~rate', 5)
        map_tr_pth  = rospy.get_param('~map_tr_pth', "/sps/data/sequence/20220420/map_transform")

        self.pub = rospy.Publisher(cloud_topic, PointCloud2, queue_size=10)
        self.pose_pub = rospy.Publisher('/ndt/predicted/odom', Odometry, queue_size=10)

        self.rate = rospy.Rate(pub_rate)  # Set the publishing rate 
        self.scans_pth = scans_pth
        self.poses_pth = poses_pth

        self.timer = rospy.Timer(rospy.Duration(1.0 / pub_rate), self.timer_callback)

        self.scans = self.load_scans()
        self.poses = self.load_poses()

        self.map_transform = np.loadtxt(map_tr_pth, delimiter=',')

        self.total_scans = len(self.scans)

        assert len(self.scans) == len(self.poses), 'Must have the same length!!'

        self.index = 0

        rospy.spin()


    def timer_callback(self, event):
        scan = self.scans[self.index]
        pose = self.poses[self.index]
        timestamp_str = os.path.splitext(scan)[0]
        scan_data = np.load(os.path.join(self.scans_pth, scan))
        pose_data = np.loadtxt(os.path.join(self.poses_pth, pose), delimiter=',')

        '''Transform the scan_data using the transformed pose'''
        scan_data[:,:3] = self.transform_point_cloud(scan_data[:,:3], pose_data)
        scan_data[:,:3] = self.transform_point_cloud(scan_data[:,:3], self.map_transform)

        msg = self.to_pointmsg(scan_data, timestamp_str)
        print('Publish: %s\t\t\r' % (timestamp_str), end='')
        self.pub.publish(msg)
        self.pose_pub.publish(self.to_posemsg(pose_data, timestamp_str))
        
        #update index
        if(self.index < self.total_scans):
            self.index += 1

        if(self.index >= self.total_scans):
            rospy.signal_shutdown("No more scans to publish.")

    def load_scans(self):
        scans = sorted(os.listdir(self.scans_pth))
        return scans

    def load_poses(self):
        poses = sorted(os.listdir(self.poses_pth))
        return poses

    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Convert point cloud to homogeneous coordinates
        homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply transformation matrix
        transformed_coords = np.dot(homogeneous_coords, transformation_matrix.T)
        # Convert back to Cartesian coordinates
        transformed_point_cloud = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]
        return transformed_point_cloud

    def to_pointmsg(self, data, timestamp_str):
        cloud = PointCloud2()
        timestamp = float(timestamp_str)
        scan_time = rospy.Time.from_sec(timestamp)
        cloud.header.stamp = scan_time
        cloud.header.frame_id = 'map'

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

        point_data = np.array(data, dtype=np.float32)
        cloud.data = point_data.tobytes()

        return cloud

    def to_posemsg(self, pose_data, timestamp_str):
        odom_msg = Odometry()
        timestamp = float(timestamp_str)
        pose_time = rospy.Time.from_sec(timestamp)
        odom_msg.header.stamp = pose_time
        odom_msg.header.frame_id = 'map'
        odom_msg.child_frame_id = 'base_link'

        # Extract position (translation) from the 4x4 transformation matrix
        pose_msg = Pose()
        pose_msg.position.x = pose_data[0, 3]
        pose_msg.position.y = pose_data[1, 3]
        pose_msg.position.z = pose_data[2, 3]

        # Extract orientation (quaternion) from the 4x4 transformation matrix
        quaternion = quaternion_from_matrix(pose_data)
        pose_msg.orientation = Quaternion(*quaternion)

        odom_msg.pose.pose = pose_msg

        # Fill in the covariance matrix if available (optional)
        # odom_msg.pose.covariance = [0.0] * 36
        
        return odom_msg

if __name__ == '__main__':
    pub_scans_node = PubScans()