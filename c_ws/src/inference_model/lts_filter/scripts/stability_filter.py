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
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from loader import Loader

# from sklearn.metrics import r2_score
from torchmetrics import R2Score

from sps.datasets import util

# Additionally, some operations on a GPU are implemented stochastic for efficiency
# We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class Stability():
    def __init__(self):
        rospy.init_node('pointcloud_stability_inference')

        raw_cloud_topic = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        epsilon_0 = rospy.get_param('~epsilon_0', 0.0)
        epsilon_1 = rospy.get_param('~epsilon_1', 0.84)
        self.lidar = rospy.get_param('~lidar', 'hdl-32')

        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback)

        # Initialize the publisher
        self.pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.loss_pub = rospy.Publisher('debug/model_loss', Float32, queue_size=0)
        self.r2_pub   = rospy.Publisher('debug/model_r2', Float32, queue_size=0)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('filtered_cloud: %s', filtered_cloud_topic)
        rospy.loginfo('Bottom threshold: %f', epsilon_0)
        rospy.loginfo('Upper threshold: %f', epsilon_1)
        rospy.loginfo('Lidar type: %s', self.lidar)

        self.threshold_ground = epsilon_0
        self.threshold_dynamic = epsilon_1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(self.device)

        ''' Define loss and r2score for model evaluation '''
        self.loss = torch.nn.MSELoss() #.to(self.device)
        self.r2score = R2Score() #.to(self.device)
        
        rospy.spin()

    def callback(self, pointcloud_msg):
        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        try:
            width = pc.shape[1]
        except:
            width = 1
        data = np.zeros((height * width, 4), dtype=np.float32)
        data[:, 0] = np.resize(pc['x'], height * width)
        data[:, 1] = np.resize(pc['y'], height * width)
        data[:, 2] = np.resize(pc['z'], height * width)
        data[:, 3] = np.resize(pc['intensity'], height * width)

        # Infere the stability labels
        data = self.infer(data)

        filtered_cloud = self.to_rosmsg(data, pointcloud_msg.header)

        self.pub.publish(filtered_cloud)


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


    def load_model(self, device):
        current_pth = os.path.dirname(os.path.realpath(__file__))
        parent_pth = os.path.dirname(current_pth)
        rospy.loginfo('rosnode pth: %s', parent_pth)
        model_path = os.path.join(parent_pth, 'model')
        sys.path.append(model_path)

        '''HYPER PARAMETER'''
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        MODEL = importlib.import_module('transformer')

        model = MODEL.SPCTReg()
        model.to(device)

        checkpoint = torch.load( os.path.join(model_path, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.eval()   

        rospy.loginfo("Model loaded successfully!")

        return model

    def infer(self, pointcloud):
        
        start_time = time.time()
        FRAME_DATASET = Loader(pointcloud, self.lidar)
        batch_size = FRAME_DATASET.num_windows

        points, labels = FRAME_DATASET[0]

        points = torch.from_numpy(points)
        points = points.unsqueeze(0)

        labels = torch.from_numpy(labels)
        labels = labels.unsqueeze(0)

        for i in range(1, len(FRAME_DATASET)):
            p, l = FRAME_DATASET[i]
            p = torch.from_numpy(p)
            p = p.unsqueeze(0)
            l = torch.from_numpy(l)
            l = l.unsqueeze(0)
            points = torch.vstack((points, p))
            labels = torch.vstack((labels, l))

        points = points.float().to(self.device)
        points = points.transpose(2, 1)
        labels_pred = self.model(points)
        labels_pred = labels_pred.permute(0,2,1).cpu()

        # ''' Step 5: Calculate loss and r2 '''
        loss = self.loss(labels_pred.view(-1), labels.view(-1))
        r2 = self.r2score(labels_pred.view(-1), labels.view(-1))
        self.loss_pub.publish(loss)
        self.r2_pub.publish(r2)

        points = points.permute(0,2,1).cpu().data.numpy().reshape((-1, 3))
        labels_pred = labels_pred.data.numpy().reshape((-1, ))

        data = np.column_stack((points, labels_pred))

        ### -> mIoU start
        pred = np.where(labels_pred < self.threshold_dynamic, 0, 1)
        gt   = np.where(labels.cpu().view(-1) < self.threshold_dynamic, 0, 1)

        precision, recall, f1, accuracy, dIoU = util.calculate_metrics(gt, pred)

        log_message = (
            f"dIoU: {dIoU:.3f} "
            f"accuracy: {accuracy:.3f} "
            f"precision: {precision:.3f} "
            f"recall: {recall:.3f} "
            f"f1: {f1:.3f} "
        )
        rospy.loginfo(log_message)
        ### <- mIoU ends

        original_len = len(data)

        data = data[(data[:,3] <= self.threshold_dynamic)]

        filtered_len = len(data)

        end_time = time.time()
        elapsed_time = end_time - start_time

        rospy.loginfo("T: {:.4f} sec [{:.2f} Hz], L: {:.4f}, R2: {:.4f}, N: {:d}, n: {:d}".format(elapsed_time, 1/elapsed_time, loss, r2, original_len, filtered_len))

        return data


if __name__ == '__main__':
    stability_node = Stability()
    
