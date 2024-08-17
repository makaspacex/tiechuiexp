#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/13
# @fileName test.py
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

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for i, (ori, X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss

            # 优化输出
            ori = ['%010.4f' % _x for _x in ori.tolist()[0]]
            y = ['%010.4f' % _x for _x in y.tolist()[0]]
            pred = ['%010.4f' % _x for _x in pred.tolist()[0]]

            print(f"X:{ori}, Y:{y}, pred: {pred}, loss:{loss}")

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

def main():
    saved_model_path = 'run/best_4dim_v2.pth'
    data_path = 'dataset/v2/data.csv'
    data_loc = (1, 2, 3, 4)
    label_loc = (5, 6)

    model = NeuralNetwork(input_dim=len(data_loc))

    if os.path.exists(saved_model_path):
        print('loading model....')
        ckpt = torch.load(saved_model_path)
        model.load_state_dict(ckpt)
    else:
        raise Exception('模型不存在')

    data = MyDataset(data_path, 'test', test_ratio=-1, data_loc=data_loc, label_loc=label_loc)
    dataloader = DataLoader(data, batch_size=1, shuffle=True)
    loss_fn = torch.nn.MSELoss()

    print(f'testing {len(data)} examples ...')
    test_loop(dataloader, model, loss_fn)


if __name__ == '__main__':
    main()


