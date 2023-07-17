#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rospy
import ros_numpy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField

import mos4d.models.models as models

''' Constants '''
SCAN_TIMESTAMP = 1
MAP_TIMESTAMP = 0
CLOUD_ODOM_MAX_TIME_DIFF = 0.05 #seconds 

''' Define color thresholds '''
GREEN_THRESHOLD = 0.1  # Adjust as needed
RED_THRESHOLD = 0.2  # Adjust as needed

class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation')

        self.odom_msg_stamp = 0
        self.center = [0, 0, 0]
        self.cfg = None

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/ndt/predicted/aligned_points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        self.weights_pth = rospy.get_param('~model_weights_pth', "/mos4d/best_models/last.ckpt")
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/ndt/predicted/odom")
        self.threshold_dynamic = rospy.get_param('~epsilon', 0.85)

        ''' Subscribe to ROS topics '''
        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback_points)
        rospy.Subscriber(predicted_pose_topic, Odometry, self.callback_odom)

        ''' Initialize the publisher '''
        self.pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('filtered_cloud: %s', filtered_cloud_topic)
        rospy.loginfo('Upper threshold: %f', self.threshold_dynamic)
        rospy.loginfo('Predicted pose: %s', predicted_pose_topic)

        ''' The model need to be loaded before the map in order to init the self.cfg var '''
        self.model = self.load_model()

        ''' Load point cloud map '''
        map_str = self.cfg["TRAIN"]["MAP"]
        map_pth = os.path.join(str(os.environ.get("DATA")), "maps", map_str)
        rospy.loginfo('Loading point cloud map, pth: %s' % (map_pth))
        try:
            self.point_cloud_map = np.loadtxt(map_pth, dtype=np.float32)
            rospy.loginfo('Point cloud map loaded successfully!')
        except:
            rospy.logerr('Failed to load point cloud map from %s', map_pth)
            sys.exit()

        rospy.spin()


    def callback_odom(self, msg):
        self.odom_msg_stamp = msg.header.stamp.to_sec()
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.center = [x, y, z]


    def callback_points(self, pointcloud_msg):
        cloud_msg_stamp = pointcloud_msg.header.stamp.to_sec()
        time_diff = abs(cloud_msg_stamp - self.odom_msg_stamp)

        if time_diff >= CLOUD_ODOM_MAX_TIME_DIFF:
            raise AssertionError('Pose and point cloud messages should have the same timestamp!! Diff: %f' % time_diff)

        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        width = pc.shape[1] if len(pc.shape) > 1 else 1

        scan = np.zeros((height * width, 4), dtype=np.float32)
        scan[:, 0] = np.resize(pc['x'], height * width)
        scan[:, 1] = np.resize(pc['y'], height * width)
        scan[:, 2] = np.resize(pc['z'], height * width)
        scan[:, 3] = np.resize(pc['intensity'], height * width)

        ''' Get submap points '''
        submap_indices = self.select_points_within_radius(self.point_cloud_map[:,:3], self.center)
        submap_points = self.point_cloud_map[submap_indices, :3]
        # submap_labels = self.point_cloud_map[submap_indices, 3]

        scan_indices = self.select_points_within_radius(scan[:,:3], self.center)
        scan_points = scan[scan_indices, :3]
        # scan_labels = scan[scan_indices, 3]

        ''' Infere the stability labels '''
        scan = self.infer(scan_points, submap_points)

        self.pub.publish(self.to_rosmsg(scan, pointcloud_msg.header))


    def load_model(self):
        self.cfg = torch.load(self.weights_pth)["hyper_parameters"]

        state_dict = {
            k.replace("model.MinkUNet.", ""): v
            for k, v in torch.load(self.weights_pth)["state_dict"].items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model = models.MOSNet(self.cfg)
        model.model.MinkUNet.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        model.freeze()

        rospy.loginfo("Model loaded successfully!")

        return model
    

    def select_points_within_radius(self, coordinates, center):
        squared_distances = np.sum((coordinates - center) ** 2, axis=1)
        indexes = np.where(squared_distances <= self.cfg["DATA"]["RADIUS"] ** 2)[0]
        return indexes
    

    def to_rosmsg(self, data, header):
        cloud = PointCloud2()
        cloud.header = header

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
    

    def print_infer_time(self, elapsed_time, frame_size):
        ''' Determine color code based on elapsed_time value '''
        if elapsed_time < GREEN_THRESHOLD:
            color_code = "\033[32m"  # Green
        elif elapsed_time > RED_THRESHOLD:
            color_code = "\033[31m"  # Red
        else:
            color_code = "\033[33m"  # Yellow

        ''' Reset text color code '''
        reset_code = "\033[0m"

        ''' Format the log message with color codes '''
        log_message = f"Frame inference: {color_code}{elapsed_time:.4f}{reset_code} sec [{color_code}{1/elapsed_time:.2f}{reset_code} Hz]. Size: {frame_size}"
        rospy.loginfo(log_message)


    def infer(self, scan, submap):
        # start_time = time.time()

        ''' Prepare the data for the model, it need to be like this [Batch, X, Y, Z, time] '''
        scan_points = torch.tensor(scan, dtype=torch.float32).reshape(-1, 3)
        submap_points = torch.tensor(submap, dtype=torch.float32).reshape(-1, 3)

        ''' Bind time stamp to scan and submap points '''
        scan_points = self.add_timestamp(scan_points, SCAN_TIMESTAMP)
        submap_points = self.add_timestamp(submap_points, MAP_TIMESTAMP)

        ''' Combine scans and map into the same tensor '''
        scan_submap_data = torch.vstack([scan_points, submap_points])

        batch = torch.zeros(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
        tensor = torch.hstack([batch, scan_submap_data]) #.reshape(-1, 5)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # rospy.loginfo("Data preperation time: {:.4f} seconds [{:.2f} Hz]".format(elapsed_time, 1/elapsed_time))

        start_time = time.time()
        with torch.no_grad():
            ''' Move the data tensor to the GPU manually when using the model directly without PyTorch Lightning '''       
            scores = self.model.forward(tensor.cuda())  
            scores = scores.cpu().data.numpy()

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.print_infer_time(elapsed_time, len(scores))

        # start_time = time.time()
        ''' Find the indecies where the score is less than the threshold, i.e. filter out dynamic points '''
        scan_scores = scores[:len(scan)]
        scan_points = scan_points[(scan_scores <= self.threshold_dynamic)]

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # rospy.loginfo("Filter processing time: {:.4f} seconds [{:.2f} Hz]".format(elapsed_time, 1/elapsed_time))

        ''' Convert the torch tensor back to a NumPy array '''
        scan_points = scan_points.numpy()

        return scan_points



if __name__ == '__main__':
    SPS_node = SPS()
    



