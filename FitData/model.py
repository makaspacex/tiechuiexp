#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/12
# @fileName model.py
# Copyright 2017 izhangxm@gmail.com. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from torch import nn
from config import DATA_INFO
import torch

def human_p(data_list:torch.tensor):
    return [ float(f'{ele:.4f}') for ele in data_list.tolist()]


class RMSE(nn.Module):

    def __init__(self):
        super(RMSE, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred, y):
        eps = 1e-8
        dis = torch.sqrt(torch.square(pred-y))
        rmse = dis.mean(dim=0)

        return rmse, torch.sqrt(self.criterion(pred, y) + eps)

class NeuralNetwork(nn.Module):
    def __init__(self, data_loc=(0, 1, 2, 3, 4), label_loc=(5, 6), re_norm=False):
        super(NeuralNetwork, self).__init__()

        c1_out, c2_out, c3_out = 48, 32, 16
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, c1_out, kernel_size=(3,), padding=1),
            nn.BatchNorm1d(c1_out),
            nn.PReLU(),
            nn.Conv1d(c1_out, c2_out, kernel_size=(3,), padding=1),
            nn.BatchNorm1d(c2_out),
            nn.PReLU(),
            nn.Conv1d(c2_out, c3_out, kernel_size=(3,), padding=1),
            nn.BatchNorm1d(c3_out),
            nn.PReLU(),
        )

        self.input_dim = len(data_loc)
        self.output_dim = len(label_loc)
        self.data_loc, self.label_loc = list(data_loc), list(label_loc)

        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(c3_out * self.input_dim, 32 * self.input_dim),
            nn.Linear(32 * self.input_dim, 32),
            nn.Linear(32, self.output_dim)
        )
        self.re_norm = re_norm

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        if self.re_norm:
            for _, loc in enumerate(self.label_loc):
                _min, _max = DATA_INFO[loc]
                i = loc - 5
                logits[:, i] = logits[:, i] * (_max - _min) + _min

        return logits
