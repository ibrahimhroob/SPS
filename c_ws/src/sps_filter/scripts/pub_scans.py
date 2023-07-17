#!/usr/bin/env python3

import os
import sys
import time
import tqdm
import numpy as np

import rospy

from sensor_msgs.msg import PointCloud2, PointField

class PubScans:
    def __init__(self):
        rospy.init_node('Labelled_scans_publisher')

        ''' Retrieve parameters from ROS parameter server '''
        cloud_topic = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        scans_pth   = rospy.get_param('~scans_pth', "/mos4d/data/sequence/20220420/scans")
        pub_rate    = rospy.get_param('~rate', 5)

        self.pub = rospy.Publisher(cloud_topic, PointCloud2, queue_size=10)
        self.rate = rospy.Rate(pub_rate)  # Set the publishing rate 
        self.scans_pth = scans_pth

        self.timer = rospy.Timer(rospy.Duration(1.0 / pub_rate), self.timer_callback)

        self.scans = self.load_scans()

        self.total_scans = len(self.scans)

        self.index = 0

        rospy.spin()


    def timer_callback(self, event):
        scan = self.scans[self.index]
        timestamp_str = os.path.splitext(scan)[0]
        scan_data = np.load(os.path.join(self.scans_pth, scan))
        msg = self.to_rosmsg(scan_data, timestamp_str)
        print('Publish: %s\t\t\r' % (timestamp_str), end='')
        self.pub.publish(msg)
        
        #update index
        if(self.index < self.total_scans):
            self.index += 1

        if(self.index >= self.total_scans):
            rospy.signal_shutdown("No more scans to publish.")

    def load_scans(self):
        scans = sorted(os.listdir(self.scans_pth))
        return scans

    def to_rosmsg(self, data, timestamp_str):
        cloud = PointCloud2()
        timestamp = float(timestamp_str)
        scan_time = rospy.Time.from_sec(timestamp)
        cloud.header.stamp = scan_time
        cloud.header.frame_id = 'os_sensor'

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


if __name__ == '__main__':
    pub_scans_node = PubScans()