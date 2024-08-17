#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/12
# @fileName dataset.py
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
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

from torchvision.transforms import ToTensor
import pandas as pd
from random import shuffle
import math
import numpy as np
from sklearn import preprocessing
import json
from config import DATA_INFO

def z_norm(x):
    return (x - x.mean()) / x.std()


def mean_norm(df_input):
    return df_input.apply(z_norm, axis=0)


class MyDataset(Dataset):

    def __init__(self, file_path, data_type, data_loc=(0, 1, 2, 3, 4), label_loc=(5, 6), test_ratio=0.2,
                 train_idxes=None, force_update_des=False):
        self.df = pd.read_csv(file_path)
        self.df = self.df.drop(['Number'], axis=1)
        self.data_l = self.df.to_numpy(dtype=np.float32)

        self.indexes = list(range(len(self.data_l)))
        shuffle(self.indexes)
        self.data_loc, self.label_loc = list(data_loc), list(label_loc)

        self.ori_data_label = self.data_l.copy()
        self.data_label = self.data_l.copy()

        if test_ratio <= 1 and test_ratio >= 0:
            test_len = math.ceil(len(self.indexes) * test_ratio)
        elif test_ratio < 0:
            test_len = len(self.indexes)
        else:
            test_len = test_ratio

        data_dir = os.path.dirname(file_path)
        json_path = os.path.join(data_dir, f"_des.json")

        if data_type == 'train':
            if (not os.path.exists(json_path)) or force_update_des:
                des = []
                for i, (mean, std) in enumerate(zip(self.data_label.mean(axis=0), self.data_label.std(axis=0))):
                    des.append([float(mean), float(std)])
                    self.data_label[:, i] = (self.data_label[:, i] - mean) / std
                json.dump(des, open(json_path, 'w+', encoding='utf-8'))
            if test_len == 0:
                self.idxes = self.indexes
            else:
                self.idxes = self.indexes[:-test_len]
        else:
            if (not os.path.exists(json_path)):
                raise Exception(f'{json_path}文件不存在')
            mean_std = json.load(open(json_path, 'r'))
            mean_std = np.array(mean_std)
            for i, (m, s) in enumerate(mean_std):
                self.data_label[:, i] = (self.data_label[:, i] - m) / s

            if train_idxes is not None:
                self.indexes = train_idxes

            if test_len == 0:
                self.idxes = self.indexes
            else:
                self.idxes = self.indexes[-test_len:]
        self.test_len = test_len

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        idx = self.idxes[idx]
        ori_data = torch.tensor(self.ori_data_label[idx, self.data_loc], dtype=torch.float32)
        label = torch.tensor(self.ori_data_label[idx, self.label_loc], dtype=torch.float32)

        data = torch.tensor(self.data_label[idx, self.data_loc], dtype=torch.float32)

        return ori_data, data, label


class MyDataset_Normal(Dataset):

    def __init__(self, file_path, data_type, data_loc=(0, 1, 2, 3, 4), label_loc=(5, 6), test_ratio=0.2,
                 train_idxes=None, force_update_des=False):
        self.datainfo = DATA_INFO
        self.df = pd.read_csv(file_path)
        self.df = self.df.drop(['Number'], axis=1)
        self.data_l = self.df.to_numpy(dtype=np.float32)

        self.indexes = list(range(len(self.data_l)))
        shuffle(self.indexes)
        self.data_loc, self.label_loc = list(data_loc), list(label_loc)

        self.ori_data_label = self.data_l.copy()
        self.data_label = self.data_l.copy()

        if test_ratio <= 1 and test_ratio >= 0:
            test_len = math.ceil(len(self.indexes) * test_ratio)
        elif test_ratio < 0:
            test_len = len(self.indexes)
        else:
            test_len = test_ratio

        for i, (_min, _max) in enumerate(self.datainfo):
            self.data_label[:, i] -= _min
            self.data_label[:, i] /= (_max - _min)

        if data_type == 'train':
            if test_len == 0:
                self.idxes = self.indexes
            else:
                self.idxes = self.indexes[:-test_len]
        else:
            if train_idxes is not None:
                self.indexes = train_idxes
            if test_len == 0:
                self.idxes = self.indexes
            else:
                self.idxes = self.indexes[-test_len:]
        self.test_len = test_len

    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        idx = self.idxes[idx]
        ori_data = torch.tensor(self.ori_data_label[idx, self.data_loc], dtype=torch.float32)
        label = torch.tensor(self.ori_data_label[idx, self.label_loc], dtype=torch.float32)

        data = torch.tensor(self.data_label[idx, self.data_loc], dtype=torch.float32)

        return ori_data, data, label


if __name__ == '__main__':
    test_ratio = 0.2
    data_path = 'dataset/v2/data.csv'

    train_data = MyDataset(data_path, 'train', test_ratio=test_ratio, force_update_des=True)
    test_data = MyDataset(data_path, 'test', test_ratio=test_ratio, train_idxes=train_data.indexes)

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    for _, data, labels in train_dataloader:
        print(data)
        print(labels)
        print('*' * 20)
