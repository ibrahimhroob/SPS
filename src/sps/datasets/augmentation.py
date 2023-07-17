#!/usr/bin/env python3
# @file      augmentation.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved
import torch


def rotate_point_cloud(points):
    """Randomly rotate the point clouds to augment the dataset"""
    rotation_angle = torch.rand(1) * 2 * torch.pi
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix = torch.Tensor([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    points = points @ rotation_matrix.type_as(points)
    return points


def rotate_perturbation_point_cloud(points, angle_sigma=0.2, angle_clip=0.5):
    """Randomly perturb the point clouds by small rotations"""
    angles = torch.clip(angle_sigma * torch.randn(3), -angle_clip, angle_clip)
    Rx = torch.Tensor(
        [
            [1, 0, 0],
            [0, torch.cos(angles[0]), -torch.sin(angles[0])],
            [0, torch.sin(angles[0]), torch.cos(angles[0])],
        ]
    )
    Ry = torch.Tensor(
        [
            [torch.cos(angles[1]), 0, torch.sin(angles[1])],
            [0, 1, 0],
            [-torch.sin(angles[1]), 0, torch.cos(angles[1])],
        ]
    )
    Rz = torch.Tensor(
        [
            [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
            [torch.sin(angles[2]), torch.cos(angles[2]), 0],
            [0, 0, 1],
        ]
    )
    rotation_matrix = Rz @ Ry @ Rx
    points = points @ rotation_matrix.type_as(points)
    return points


def random_flip_point_cloud(points):
    """Randomly flip the point cloud. Flip is per points."""
    if torch.rand(1).item() > 0.5:
        points = torch.multiply(points, torch.tensor([-1, 1, 1]).type_as(points))
    if torch.rand(1).item() > 0.5:
        points = torch.multiply(points, torch.tensor([1, -1, 1]).type_as(points))
    return points


def random_scale_point_cloud(points, scale_low=0.8, scale_high=1.2):
    """Randomly scale the points."""
    scales = (scale_low - scale_high) * torch.rand(1, 3) + scale_high
    points = torch.multiply(points, scales.type_as(points))
    return points