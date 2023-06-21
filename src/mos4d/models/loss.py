#!/usr/bin/env python3
# @file      loss.py
# @author    Benedikt Mersch     [mersch@igg.uni-bonn.de]
# Copyright (c) 2022 Benedikt Mersch, all rights reserved

import torch
import torch.nn as nn

import MinkowskiEngine as ME


class MOSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.loss = nn.MSELoss()

    def compute_loss(self, out: ME.TensorField, past_labels: list):
        # Get raw point wise scores
        logits = out.features

        assert False, 'TODO'

        logits = self.tanh(logits)

        # softmax = self.softmax(logits)
        # log_softmax = torch.log(softmax.clamp(min=1e-8))

        # Prepare ground truth labels
        gt_labels = torch.cat(past_labels, dim=0)[:, 0]

        loss = self.loss(logits, gt_labels.long())
        return loss