#!/usr/bin/env python3

import os
import sys
import time
import torch
import numpy as np

import rospy
import ros_numpy

from nav_msgs.msg import Odometry	
from std_msgs.msg import Float32
from tf.transformations import quaternion_matrix
from sensor_msgs.msg import PointCloud2, PointField

import sps.models.models as models

from torchmetrics import R2Score

import MinkowskiEngine as ME

''' Constants '''
SCAN_TIMESTAMP = 1
MAP_TIMESTAMP = 0


class SPS():
    def __init__(self):
        rospy.init_node('Stable_Points_Segmentation_hdl')

        ''' Retrieve parameters from ROS parameter server '''
        raw_cloud_topic = rospy.get_param('~raw_cloud', "/os_cloud_node/points")
        filtered_cloud_topic = rospy.get_param('~filtered_cloud', "/cloud_filtered")
        self.weights_pth = rospy.get_param('~model_weights_pth', "/sps/tb_logs/SPS_ME_Union/version_39/checkpoints/last.ckpt")
        self.threshold_dynamic = rospy.get_param('~epsilon', 0.85)
        predicted_pose_topic = rospy.get_param('~predicted_pose', "/ndt/predicted/odom")

        ''' Subscribe to ROS topics '''
        rospy.Subscriber(raw_cloud_topic, PointCloud2, self.callback_points)
        rospy.Subscriber(predicted_pose_topic, Odometry, self.callback_odom)

        ''' Initialize the publisher '''
        self.scan_pub = rospy.Publisher(filtered_cloud_topic, PointCloud2, queue_size=10)
        self.cloud_tr_pub = rospy.Publisher('/cloud_tr_debug', PointCloud2, queue_size=10)
        self.submap_pub = rospy.Publisher('/cloud_submap', PointCloud2, queue_size=10)
        self.submap_base_pub = rospy.Publisher('/cloud_submap_base', PointCloud2, queue_size=10)
        self.loss_pub = rospy.Publisher('model_loss', Float32, queue_size=10)
        self.r2_pub = rospy.Publisher('model_r2', Float32, queue_size=10)
        self.scan_filter_ratio = rospy.Publisher('scan_filter_ratio', Float32, queue_size=10)
        self.map_to_scan_ratio = rospy.Publisher('map_to_scan_ratio', Float32, queue_size=10)

        rospy.loginfo('raw_cloud: %s', raw_cloud_topic)
        rospy.loginfo('cloud_filtered: %s', filtered_cloud_topic)
        rospy.loginfo('cloud_submap: %s', '/cloud_submap')
        rospy.loginfo('Upper threshold: %f', self.threshold_dynamic)
        rospy.loginfo('Predicted pose: %s', predicted_pose_topic)

        ''' Load the model and the associated configs '''
        self.model, self.cfg = self.load_model()

        ''' Load point cloud map '''
        self.point_cloud_map = self.load_point_cloud_map()

        self.ds = self.cfg["MODEL"]["VOXEL_SIZE"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = torch.nn.MSELoss()
        self.r2score = R2Score()
        self.odom_msg_stamp = 0	
        self.scan_msg_header = 0	
        self.transformation = None

        ''' Debug buffers '''
        self.loss_buffer = []
        self.r2_buffer = []
        self.scan_filter_buffer = []
        self.map_filter_buffer = []

        ''' Variable to hold point cloud frame '''
        self.scan = None

        rospy.spin()

    def load_model(self):
        cfg = torch.load(self.weights_pth)["hyper_parameters"]

        state_dict = {
            k.replace("model.MinkUNet.", ""): v
            for k, v in torch.load(self.weights_pth)["state_dict"].items()
        }
        state_dict = {k: v for k, v in state_dict.items() if "MOSLoss" not in k}
        model = models.SPSNet(cfg)
        model.model.MinkUNet.load_state_dict(state_dict)
        model = model.cuda()
        model.eval()
        model.freeze()

        rospy.loginfo("Model loaded successfully!")

        return model, cfg


    def load_point_cloud_map(self):
        map_id = self.cfg["TRAIN"]["MAP"]
        map_pth = os.path.join(str(os.environ.get("DATA")), "maps", map_id)
        __, file_extension = os.path.splitext(map_pth)
        rospy.loginfo('Loading point cloud map, pth: %s' % (map_pth))
        try:
            point_cloud_map = np.load(map_pth) if file_extension == '.npy' else np.loadtxt(map_pth, dtype=np.float32)
            point_cloud_map = torch.tensor(point_cloud_map[:, :4]).to(torch.float32).reshape(-1, 4)
            rospy.loginfo('Point cloud map loaded successfully!')
        except:
            rospy.logerr('Failed to load point cloud map from %s', map_pth)
            sys.exit()

        return point_cloud_map


    def transform_point_cloud(self, point_cloud, transformation_matrix):
        # Convert point cloud to homogeneous coordinates
        homogeneous_coords = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
        # Apply transformation matrix
        transformed_coords = np.dot(homogeneous_coords, transformation_matrix.T)
        # Convert back to Cartesian coordinates
        transformed_point_cloud = transformed_coords[:, :3] / transformed_coords[:, 3][:, np.newaxis]
        return transformed_point_cloud


    def inverse_transform_point_cloud(self, transformed_point_cloud, transformation_matrix):
        # Invert the transformation matrix
        inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        # Convert transformed point cloud to homogeneous coordinates
        homogeneous_coords = np.hstack((transformed_point_cloud, np.ones((transformed_point_cloud.shape[0], 1))))
        # Apply inverse transformation matrix
        original_coords = np.dot(homogeneous_coords, inv_transformation_matrix.T)
        # Convert back to Cartesian coordinates
        original_point_cloud = original_coords[:, :3] / original_coords[:, 3][:, np.newaxis]
        return original_point_cloud


    def prune_map_points(self, scan_data, pc_map_data):
        start_time = time.time()
        quantization = torch.Tensor([self.ds, self.ds, self.ds]).to(self.device)

        '''we are only intrested in the coordinates of the points, thus we are keeping only the xyz columns'''
        scan = scan_data[:,:3]
        pc_map = pc_map_data[:,:3]

        scan_coords = torch.div(scan, quantization.type_as(scan)).int().to(self.device)
        scan_features = torch.zeros(scan.shape[0], 2).to(self.device)
        scan_features[:, 0] = 1

        map_coord = torch.div(pc_map, quantization.type_as(pc_map)).int().to(self.device)
        map_features = torch.zeros(pc_map.shape[0], 2).to(self.device)
        map_features[:, 1] = 1

        map_sparse = ME.SparseTensor(
            features=map_features, 
            coordinates=map_coord,
            )

        scan_sparse = ME.SparseTensor(
            features=scan_features, 
            coordinates=scan_coords,
            coordinate_manager = map_sparse.coordinate_manager
            )

        # Merge the two sparse tensors
        union = ME.MinkowskiUnion()
        output = union(scan_sparse, map_sparse)

        # Only keep map points that lies in the same voxel as scan points
        mask = (output.F[:,0] * output.F[:,1]) == 1

        # Prune the merged tensors 
        pruning = ME.MinkowskiPruning()
        output = pruning(output, mask)

        # Retrieve the original coordinates
        submap_points = output.coordinates

        # dequantize the points
        submap_points = submap_points * self.ds
        
        elapsed_time = time.time() - start_time

        return  submap_points.cpu(), elapsed_time, len(scan_sparse)


    def to_rosmsg(self, data, header, frame_id=None):
        cloud = PointCloud2()
        cloud.header = header
        if frame_id:
            cloud.header.frame_id = frame_id

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


    def infer(self, scan_points, submap_points):
        assert scan_points.size(-1) == 3, f"Expected 3 columns, but the scan tensor has {scan_points.size(-1)} columns."
        assert submap_points.size(-1) == 3, f"Expected 3 columns, but the submap tensor has {submap_points.size(-1)} columns."

        ''' Bind time stamp to scan and submap points '''
        scan_points = self.add_timestamp(scan_points, SCAN_TIMESTAMP)
        submap_points = self.add_timestamp(submap_points, MAP_TIMESTAMP)

        ''' Combine scans and map into the same tensor '''
        scan_submap_data = torch.vstack([scan_points, submap_points])

        batch = torch.zeros(len(scan_submap_data), 1, dtype=scan_submap_data.dtype)
        tensor = torch.hstack([batch, scan_submap_data]) #.reshape(-1, 5)

        start_time = time.time()
        with torch.no_grad():
            ''' Move the data tensor to the GPU manually when using the model directly without PyTorch Lightning '''       
            scores = self.model.forward(tensor.cuda())  
            scores = scores.cpu()

        end_time = time.time()
        elapsed_time = end_time - start_time

        ''' Find the indecies where the score is less than the threshold, i.e. filter out dynamic points '''
        scan_scores = scores[:len(scan_points)]

        return scan_scores, elapsed_time
    
    def to_tr_matrix(self, odom_msg):
        # Extract the translation and orientation from the Odometry message
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        z = odom_msg.pose.pose.position.z

        qx = odom_msg.pose.pose.orientation.x
        qy = odom_msg.pose.pose.orientation.y
        qz = odom_msg.pose.pose.orientation.z
        qw = odom_msg.pose.pose.orientation.w

        # Create the translation matrix
        translation_matrix = np.array([[1, 0, 0, x],
                                    [0, 1, 0, y],
                                    [0, 0, 1, z],
                                    [0, 0, 0, 1]])

        # Create the rotation matrix
        rotation_matrix = quaternion_matrix([qx, qy, qz, qw])

        # Compute the transformation matrix
        transformation_matrix = np.dot(translation_matrix, rotation_matrix)

        return transformation_matrix


    def callback_odom(self, odom_msg):
        start_time = time.time()
        self.odom_msg_stamp = odom_msg.header.stamp.to_sec()

        transformation_matrix = self.to_tr_matrix(odom_msg)

        ''' Step 0: make sure the time stamp for odom and scan are the same!'''
        df = self.odom_msg_stamp - self.scan_msg_header.stamp.to_sec()
        assert df == 0, f"Odom and scan time stamp is out of sync! time difference is {df} sec."

        ''' Step 1: Transform the scan '''
        scan_tr = self.transform_point_cloud(self.scan[:,:3], transformation_matrix)

        ''' Step 2: convert scan points to torch tensor '''
        scan_points = torch.tensor(scan_tr[:,:3], dtype=torch.float32).reshape(-1, 3)
        scan_labels = torch.tensor(self.scan[:, 3], dtype=torch.float32).reshape(-1, 1)

        ''' Step 3: Prune map points by keeping only the points that in the same voxel as of the scan points'''
        submap_points, prune_time, len_scan_coord = self.prune_map_points(scan_points, self.point_cloud_map)
        
        ''' Step 4: Infer the stability labels'''
        predicted_scan_labels, infer_time = self.infer(scan_points, submap_points)
        predicted_scan_labels = predicted_scan_labels.reshape(-1)
        
        ''' The following line is for debugging mainly to test the model with the true lables!! '''
        # predicted_scan_labels, infer_time = scan_labels.view(-1), 0.001

        ''' Step 5: Calculate loss and r2 '''
        loss = self.loss(predicted_scan_labels, scan_labels.view(-1))
        r2 = self.r2score(predicted_scan_labels, scan_labels.view(-1))
        self.loss_pub.publish(loss)
        self.r2_pub.publish(r2)
        self.loss_buffer.append(loss)
        self.r2_buffer.append(r2)

        ''' Step 6: Filter the scan points based on the threshold'''
        # self.scan[:, 3] = predicted_scan_labels = predicted_scan_labels.numpy()
        # condition = predicted_scan_labels <= self.threshold_dynamic
        # indexes = np.where(condition)[0]
        # filtered_scan = self.scan[indexes, :]
        # self.scan_pub.publish(self.to_rosmsg(filtered_scan, self.scan_msg_header))

        assert len(predicted_scan_labels) == len(self.scan), f"Predicted scans labels len ({len(predicted_scan_labels)}) does not equal scan len ({len(self.scan)})"
        filtered_scan = self.scan[(predicted_scan_labels < self.threshold_dynamic)]
        self.scan_pub.publish(self.to_rosmsg(filtered_scan, self.scan_msg_header))



        ''' Publish the transformed point cloud for debugging '''
        # scan_tr = np.hstack([scan_tr[:,:3], predicted_scan_labels.numpy()[0]])
        # self.cloud_tr_pub.publish(self.to_rosmsg(scan_tr, self.scan_msg_header, 'map'))

        ''' Publish the submap points for debugging '''
        submap_labels = torch.ones(submap_points.shape[0], 1)
        # submap = torch.hstack([submap_points, submap_labels])
        # self.submap_pub.publish(self.to_rosmsg(submap.numpy(), self.scan_msg_header, 'map'))

        ''' Publish the submap points to the original coordinates '''
        # submap = submap.numpy()
        # submap[:,:3] = self.inverse_transform_point_cloud(submap[:,:3], self.transformation)
        # self.submap_base_pub.publish(self.to_rosmsg(submap, pointcloud_msg.header, 'os_sensor'))

        scan_len_before = len(self.scan)
        scan_len_after = len(filtered_scan)

        '''Publish scan filter ration and map to scan ratio'''
        scan_filter_ratio = (scan_len_after/scan_len_before) * 100.0
        map_to_scan_ratio = (len(submap_labels) / len_scan_coord) * 100.0
        self.scan_filter_ratio.publish(scan_filter_ratio)
        self.map_to_scan_ratio.publish(map_to_scan_ratio)
        self.scan_filter_buffer.append(scan_filter_ratio)
        self.map_filter_buffer.append(map_to_scan_ratio)


        ''' Print log message '''
        end_time = time.time()
        elapsed_time = end_time - start_time
        hz = lambda t: 1 / t if t else 0

        log_message = (
            f"T: {elapsed_time:.3f} sec [{hz(elapsed_time):.2f} Hz]. "
            f"P: {prune_time:.3f} sec [{hz(prune_time):.2f} Hz]. "
            f"I: {infer_time:.3f} sec [{hz(infer_time):.2f} Hz]. "
            f"L: {loss:.3f}, r2: {r2:.3f}. "
            f"N: {scan_len_before:d}, n: {scan_len_after:d}, "
            f"S: {len_scan_coord:d}, "
            f"M: {len(submap_labels):d}"
        )
        rospy.loginfo(log_message)


    def callback_points(self, pointcloud_msg):
        pc = ros_numpy.numpify(pointcloud_msg)
        height = pc.shape[0]
        width = pc.shape[1] if len(pc.shape) > 1 else 1

        scan = np.zeros((height * width, 4), dtype=np.float32)
        scan[:, 0] = np.resize(pc['x'], height * width)
        scan[:, 1] = np.resize(pc['y'], height * width)
        scan[:, 2] = np.resize(pc['z'], height * width)
        scan[:, 3] = np.resize(pc['intensity'], height * width)
        scan_len_before = len(scan)

        self.scan = scan
        self.scan_msg_header = pointcloud_msg.header



if __name__ == '__main__':
    SPS_node = SPS()
    



