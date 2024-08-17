#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/13
# @fileName predict.py
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
import os
from model import NeuralNetwork
from dataset import MyDataset
from torch.utils.data import DataLoader
import argparse
import numpy as np

mean_std = [[0.027584388888888884, 0.032773262495322336], [17.376962500000005, 11.413249904902978], [1738.328206697222, 492.7634408545658], [1645.2416666666663, 446.7021528458788], [107.16944444444442, 24.244201420951825], [10.78318250738889, 21.218910956909316], [317.1474411111111, 367.61722120859247]]
mean_std = np.array(mean_std)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='input data')
    parser.add_argument('--model_path', required=True, help='model path')
    opt = parser.parse_args()


    # 预处理
    data = np.array([float(x) for x in opt.data.split(',')])
    mean = mean_std[1:5, 0]
    std = mean_std[1:5, 1]
    if len(data) == 5:
        mean = mean_std[0:5, 0]
        std = mean_std[0:5, 1]
    data = torch.tensor((data - mean)/std, dtype=torch.float32)
    data = torch.unsqueeze(data, dim=0)

    saved_model_path = opt.model_path
    model = NeuralNetwork(input_dim=len(data))


    if os.path.exists(saved_model_path):
        ckpt = torch.load(saved_model_path)
        model.load_state_dict(ckpt)
    else:
        raise Exception('模型不存在')

    with torch.no_grad():
        pred = model(data)
        print(pred.tolist()[0])


if __name__ == '__main__':
    main()


