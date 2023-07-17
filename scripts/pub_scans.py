#!/usr/bin/env python3

import os
import sys
import time
import tqdm
import click
import numpy as np

import rospy

from sensor_msgs.msg import PointCloud2, PointField

class PubScans:
    def __init__(self, scans_pth, pub_rate):
        rospy.init_node('Labelled_scans_publisher')
        self.pub = rospy.Publisher('/os_cloud_node/points', PointCloud2, queue_size=10)
        self.rate = rospy.Rate(pub_rate)  # Set the publishing rate 
        self.scans_pth = scans_pth

    def run(self):
        scans = self.load_scans()

        for scan in scans: #tqdm.tqdm(scans, desc='Publishing scans', unit='scan'):
            timestamp_str = os.path.splitext(scan)[0]
            scan_data = np.load(os.path.join(self.scans_pth, scan))
            msg = self.to_rosmsg(scan_data, timestamp_str)
            print('Publish: %s\r' % (timestamp_str), end='')
            self.pub.publish(msg)
            self.rate.sleep()

        print('\n\nDone!!')
        rospy.signal_shutdown("Labelled_scans_publisher node has finished its task")  # Self-terminate the node

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


@click.command()
@click.option(
    "--scans_pth",
    "-c",
    type=str,
    help="path to the labelled scans",
    default="/mos4d/data/sequence/20220420/scans"
)
@click.option(
    "--pub_rate",
    "-r",
    type=int,
    help="Scans publish rate",
    default=5
)
def main(scans_pth, pub_rate):
    try:
        pub_scans_node = PubScans(scans_pth, pub_rate)
        pub_scans_node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()