#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/12
# @fileName train.py
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
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MyDataset, MyDataset_Normal
from model import NeuralNetwork, RMSE
from tqdm import tqdm, trange
import argparse
from argparse import Namespace
from model import human_p

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    rmse_total = None
    for batch, (_, X, Y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)

        rmse, loss = loss_fn(pred, Y)

        if rmse_total is None:
            rmse_total = rmse
        else:
            rmse_total += rmse

        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= num_batches
    rmse_total /= num_batches
    return rmse_total, train_loss


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss, rmse_loss = 0, None

    with torch.no_grad():
        for _, X, y in dataloader:
            pred = model(X)
            rmse, loss = loss_fn(pred, y)
            test_loss += loss.item()
            if rmse_loss is None:
                rmse_loss = rmse
            else:
                rmse_loss += rmse

    test_loss /= num_batches
    rmse_loss /= num_batches
    return rmse_loss, test_loss


def main(opt):
    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--hekou_name', type=str, required=True)
        parser.add_argument('--epoch', type=int, default=1000 * 4)
        parser.add_argument('--use_norm_data', type=bool, default=False)
        opt = parser.parse_args()

    hekou_name = opt.hekou_name

    train_path = f'dataset/v2/train_{hekou_name}.csv'
    test_path = f'dataset/v2/test_{hekou_name}.csv'
    save_model_path = f'run/best_5dim_{hekou_name}.pth'  # 这个地方设置最优模型存贮的位置，就是参数文件，这个后来再说

    #  TrainLoss:94.6828 TestLoss:383.8854 BestLoss:68.4577
    # 分别是最新训练数据MSE，最新测试数据MSE，最好的测试数据MSE，都是指的整体所有的数据 over！

    data_loc = (0, 1, 2, 3, 4)
    # label_loc = (5,)
    # label_loc = (6,)
    label_loc = (5, 6)

    HekouDataset = MyDataset
    if opt.use_norm_data:
        HekouDataset = MyDataset_Normal

    train_data = HekouDataset(train_path, 'train', test_ratio=0, data_loc=data_loc, label_loc=label_loc,
                              force_update_des=False)
    test_data = HekouDataset(test_path, 'test', test_ratio=-1, train_idxes=None, data_loc=data_loc, label_loc=label_loc)

    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)

    loss_rmse_fn = RMSE()


    model = NeuralNetwork(data_loc=data_loc, label_loc=label_loc, re_norm=opt.use_norm_data)

    # 随机数据 不能恢复模型 模型已经见过测试数据了
    # if os.path.exists(save_model_path):
    #     print('loading model....')
    #     ckpt = torch.load(save_model_path)
    #     model.load_state_dict(ckpt)

    learning_rate = 0.001
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5)


    epochs = opt.epoch  # 10K
    test_margin = 20
    min_loss = np.Inf
    test_min_rmse = None
    _loss = np.Inf
    print(f'start training for {hekou_name}')
    print(f'train data:{train_path}, test data:{test_path}')
    print(f"training on {len(train_data)} examples, testing on {len(test_data)} examples")
    pbar = tqdm(total=epochs, ascii=True, ncols=200)
    for t in range(epochs):
        rmse, tran_loss = train_loop(train_dataloader, model, loss_rmse_fn, optimizer)
        if t == epochs - 1 or t % test_margin == 0:
            rmse_loss, _loss = test_loop(test_dataloader, model, loss_rmse_fn)
            if _loss < min_loss:
                # print('updating best model ......')
                torch.save(model.state_dict(), save_model_path)
                min_loss = _loss
                test_min_rmse = rmse_loss

        pbar.set_description(
            f"[Epoch:{t + 1}/{epochs} TrainLoss:{tran_loss:.4f} {human_p(rmse)} TestLoss:{_loss:.4f} {human_p(rmse_loss)} BestLoss:{min_loss:.4f} {human_p(test_min_rmse)} ]")
        pbar.update(1)
    pbar.close()

    print(f"training for {hekou_name}  has done! Best loss on testing dataset is {min_loss} and model params saved.")


if __name__ == '__main__':
    import glob

    test_list = glob.glob('dataset/v2/test_*.csv')
    hekou_names = []

    for ele in test_list:
        name = os.path.basename(ele).split('.')[0].split('_')[-1]
        opt = Namespace(epoch=4000, hekou_name=name, use_norm_data=False)
        main(opt)
        print('*' * 80 + '\n\n')




