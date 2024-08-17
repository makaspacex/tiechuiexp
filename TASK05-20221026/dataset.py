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
import shutil


def load_des(des_path):
    json_path = des_path
    if (not os.path.exists(json_path)):
        raise Exception(f'{json_path}文件不存在')
    mean_std = json.load(open(json_path, 'r'))
    mean_std = np.array(mean_std)
    return mean_std

def z_norm(x):
    return (x - x.mean()) / x.std()

def mean_norm(df_input):
    return df_input.apply(z_norm, axis=0)

class MyDataset(Dataset):

    def __init__(self, file_path, des_path=None, time_loc=(0,), data_loc=(1, 2, 3, 4,5,6), label_loc=(7,), mode='train'):
        
        self.df = pd.read_csv(file_path)
        
        self.data_l = self.df.values
        
        self.time_loc, self.data_loc, self.label_loc = list(time_loc), list(data_loc), list(label_loc)

        self.idxes = list(range(len(self.data_l)))
        
        self.ori_data_label = self.data_l.copy()
        self.data_label = self.data_l.copy()

        data_dir = os.path.dirname(file_path)
        
        if des_path is None:
            des_path = os.path.join(data_dir, f"des.json")
        mean_std = load_des(des_path=des_path)
        
        for i, (mean, std, _, _) in zip(self.data_loc+self.label_loc, mean_std):
            
            self.data_label[:, i] = (self.ori_data_label[:, i] - mean) / std
        
    def __len__(self):
        return len(self.idxes)

    def __getitem__(self, idx):
        idx = self.idxes[idx]
        
        time_ix_ = np.ix_([idx], self.time_loc)
        data_ix_ = np.ix_([idx], self.data_loc)
        label_ix_ = np.ix_([idx], self.label_loc)
        
        
        ori_data = torch.tensor(self.ori_data_label[data_ix_].astype(np.float32).squeeze(), dtype=torch.float32)
        ori_label = torch.tensor(self.ori_data_label[label_ix_].astype(np.float32).squeeze(), dtype=torch.float32).unsqueeze(0)

        data = torch.tensor(self.data_label[data_ix_].astype(np.float32).squeeze(), dtype=torch.float32)
        label = torch.tensor(self.data_label[label_ix_].astype(np.float32).squeeze(), dtype=torch.float32).unsqueeze(0)
        
        _time = [self.ori_data_label[time_ix_].squeeze().tolist()]

        return _time, ori_data, data, ori_label, label


def split_dataset(dataset_csv, dst_dir_name="data", test_ratio=0.2):
    df = pd.read_csv(dataset_csv)
    
    data_l = df.values
    
    dataset_idxes = list(range(len(data_l)))
    shuffle(dataset_idxes)
    
    test_len = math.ceil(len(dataset_idxes) * test_ratio)

    train_idx = sorted(dataset_idxes[:-test_len])
    test_idx = sorted(dataset_idxes[-test_len:])
    
    tran_data = data_l[train_idx]
    test_data = data_l[test_idx]
    
    titles = list(df.columns)
    
    dst_full_dir = os.path.join(os.path.dirname(dataset_csv), dst_dir_name)
    os.makedirs(dst_full_dir, exist_ok=True)
    
    with open(os.path.join(dst_full_dir, "train.csv"), 'w+', encoding="utf8") as f:
        title_line = ",".join(titles)
        f.write(f"{title_line}\n")
        for line in tran_data:
            f.write(f"{','.join([str(x) for x in line])}\n")
    
    with open(os.path.join(dst_full_dir, "test.csv"), 'w+', encoding="utf8") as f:
        title_line = ",".join(titles)
        f.write(f"{title_line}\n")
        for line in test_data:
             f.write(f"{','.join([str(x) for x in line])}\n")
    
    json_path = os.path.join(dst_full_dir, f"des.json")
    des = []
    
    df = df.drop(['date-UTC'], axis=1)
    data_l = df.to_numpy(dtype=np.float32)
    for i, (mean, std, _min, _max) in enumerate(zip(data_l.mean(axis=0), data_l.std(axis=0), data_l.min(axis=0),data_l.max(axis=0))):
        des.append([float(mean), float(std), float(_min), float(_max)])
    json.dump(des, open(json_path, 'w+', encoding='utf-8'))
    
    shutil.copy(dataset_csv, os.path.join(dst_full_dir, os.path.basename(dataset_csv)))
    
    print("ok")

    
if __name__ == '__main__':
    test_ratio = 0.20
    data_path = 'dataset/data2014_7-10/2014.7-10 5min-without par_full.csv'
    split_dataset(data_path, dst_dir_name="data_2014_7-10_full")
    
    exit(0)
    
    train_data = MyDataset("dataset/data_v1/train.csv")
    test_data = MyDataset("dataset/data_v1/test.csv")

    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

    for _time, ori_data, data, ori_label, label in train_dataloader:
        print(_time)
        print(ori_data)
        print(data)
        print(ori_label)
        print(label)
        print('*' * 20)
