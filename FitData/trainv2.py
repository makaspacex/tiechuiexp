#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2021/12/14
# @fileName trainv2.py
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
# 基础
import os.path
import os
import numpy as np
import pandas as pd
import time
import pickle

# 绘图
import seaborn as sns
import matplotlib.pyplot as plt

# 模型
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from mlxtend.regressor import StackingCVRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

# 模型相关
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.stats import skew, kurtosis, norm
import argparse
from argparse import Namespace
# 忽略警告
import warnings


def ignore_warn(*args, **kwargs):
    pass


warnings.warn = ignore_warn

def RMSE(y, y_preds):
    y = y.to_numpy()

    dis = np.abs(y - y_preds)
    rmse_loss = np.mean(dis, axis=0)


    return rmse_loss, np.sqrt(mean_squared_error(y, y_preds))

def load_data(train_path, test_path, data_loc, label_loc):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print('The shape of training data:', train.shape)
    print('The shape of testing data:', test.shape)

    columns = train.columns
    train = train.drop([columns[0]], axis=1)
    test = test.drop([columns[0]], axis=1)

    columns = train.columns

    data_loc, label_loc = list(data_loc), list(label_loc)

    target_columns = columns[label_loc]
    data_columns = columns[data_loc]



    test_y_real = test[target_columns]

    # y = train[['H+', 'pCO2']]
    y = train[target_columns]
    print('Skewness of target:', y.skew())
    print('kurtosis of target:', y.kurtosis())
    # sns.distplot(y, fit=norm)
    # plt.show()

    y_log = np.log1p(y)
    print('Skewness of target:', y_log.skew())
    print('kurtosis of target:', y_log.kurtosis())
    # sns.distplot(y_log, fit=norm)
    # plt.show()



    train = train[data_columns]
    test = test[data_columns]


    # 检查训练集与测试集的维度是否一致
    print('The shape of training data:', train.shape)
    print('The length of y:', len(y_log))
    print('The shape of testing data:', test.shape)

    return train, y_log, test, test_y_real


def train_process(train, y_log):

    # Lasso
    lasso_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    lasso = make_pipeline(RobustScaler(), LassoCV(alphas=lasso_alpha, random_state=2))

    # ElasticNet
    enet_beta = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
    enet_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    ENet = make_pipeline(RobustScaler(), ElasticNetCV(l1_ratio=enet_beta, alphas=enet_alpha, random_state=12))

    # Ridge
    rid_alpha = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    rid = make_pipeline(RobustScaler(), RidgeCV(alphas=rid_alpha))

    # Gradient Boosting
    gbr_params = {'loss': 'huber',
                  'criterion': 'mse',
                  'learning_rate': 0.1,
                  'n_estimators': 600,
                  'max_depth': 4,
                  'subsample': 0.6,
                  'min_samples_split': 20,
                  'min_samples_leaf': 5,
                  'max_features': 0.6,
                  'random_state': 32,
                  'alpha': 0.5}
    gbr = GradientBoostingRegressor(**gbr_params)

    # LightGBM
    lgbr_params = {'learning_rate': 0.01,
                   'n_estimators': 1850,
                   'max_depth': 4,
                   'num_leaves': 20,
                   'subsample': 0.6,
                   'colsample_bytree': 0.6,
                   'min_child_weight': 0.001,
                   'min_child_samples': 21,
                   'random_state': 42,
                   'reg_alpha': 0,
                   'reg_lambda': 0.05}
    lgbr = lgb.LGBMRegressor(**lgbr_params)


    # XGBoost
    xgbr_params = {'learning_rate': 0.01,
                   'n_estimators': 3000,
                   'max_depth': 5,
                   'subsample': 0.6,
                   'colsample_bytree': 0.7,
                   'min_child_weight': 3,
                   'seed': 52,
                   'gamma': 0,
                   'reg_alpha': 0,
                   'reg_lambda': 1}
    xgbr = xgb.XGBRegressor(**xgbr_params)

    models_name = ['Lasso', 'ElasticNet', 'Ridge', 'Gradient Boosting', 'LightGBM', 'XGBoost']
    models_base = [lasso, ENet, rid, gbr, lgbr, xgbr]

    models = []
    for model in models_base:
        models.append(MultiOutputRegressor(model))

    # for i, model in enumerate(models):
    #   score = rmse_cv(model)
    #   print('{} score: {}({})'.format(models_name[i], score.mean(), score.std()))

    # 6、设置Stacking模型参数
    # stack_model = StackingCVRegressor(regressors=(lasso, ENet, rid, gbr, lgbr, xgbr), meta_regressor=lasso, use_features_in_secondary=True)
    # models.append(stack_model)
    # models_name.append('Stacking_model')

    # 7、在整个训练集上训练各模型

    #
    # #Lasso
    # lasso_trained = lasso.fit(np.array(train), np.array(y))
    #
    # #ElasticNet
    # ENet_trained = ENet.fit(np.array(train), np.array(y))
    #
    # #Ridge
    # rid_trained = rid.fit(np.array(train), np.array(y))
    # #Gradient Boosting
    # gbr_trained = gbr.fit(np.array(train), np.array(y))
    #
    # #LightGBM
    # lgbr_trained = lgbr.fit(np.array(train), np.array(y))
    #
    # #XGBoost
    # xgbr_trained = xgbr.fit(np.array(train), np.array(y))
    #
    # #Stacking
    # stack_model_trained = stack_model.fit(np.array(train), np.array(y))




    trained_models = []

    for name, model in zip(models_name, models):
        print(f'training {name}')
        trained_m = model.fit(np.array(train), np.array(y_log))
        trained_models.append(trained_m)

    return models_name, trained_models


def main(opt=None):

    if opt is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--hekou_name', type=str, required=True)
        parser.add_argument('--epoch', type=int, default=1000 * 4)
        parser.add_argument('--use_norm_data', type=bool, default=False)
        opt = parser.parse_args()

    hekou_name = opt.hekou_name

    train_path = f'dataset/v3/train_{hekou_name}.csv'
    test_path = f'dataset/v3/test_{hekou_name}.csv'
    save_model_path = f'run/v3/{hekou_name}.pkl'  # 这个地方设置最优模型存贮的位置，就是参数文件，这个后来再说

    save_dir = os.path.dirname(save_model_path)
    os.makedirs(save_dir, exist_ok=True)


    # H+ out total scale,salinity,TA um,DIC um,DOC um,  H+,pCO2
    # 这个地方可以设置用到的特征
    data_loc = (0, 1, 2, 3, 4)
    # label_loc = (5,)
    # label_loc = (6,)
    # 这个地方表示需要预测指标的索引
    label_loc = (5, 6,)
    # 暂时先不用改 作用不大

    train, y_log, test, test_y_real = load_data(train_path, test_path, data_loc, label_loc)

    # if os.path.exists(save_model_path):
    #     with open(save_model_path, 'rb') as f:
    #         models_name, trained_models = pickle.load(f)
    # else:
    #     models_name, trained_models = train_process(train, y_log)

    models_name, trained_models = train_process(train, y_log)

    #评估各个模型在完整训练集上的表现

    for name, model in zip(models_name, trained_models):
        y_preds = model.predict(np.array(train))
        y_real = np.expm1(y_log)
        y_preds_real = np.expm1(y_preds)
        rmse_loss, model_score = RMSE(y_real, y_preds_real)
        print(f'RMSE of {name} on training data for {hekou_name} : {model_score} {rmse_loss}')

    print("-" * 80)

    for name, model in zip(models_name, trained_models):
        y_preds = model.predict(np.array(test))
        y_preds_real = np.expm1(y_preds)

        rmse_loss, model_score = RMSE(test_y_real, y_preds_real)
        print(f'RMSE of {name} on testing data for {hekou_name} : {model_score} {rmse_loss}')

    # save
    with open(save_model_path, 'wb') as f:
        pickle.dump([models_name, trained_models], f)

    print("done")


if __name__ == '__main__':
    import glob
    test_list = glob.glob('dataset/v3/test_*.csv')
    hekou_names = []

    for ele in test_list:
        name = os.path.basename(ele).split('.')[0].split('_')[-1]
        opt = Namespace(epoch=4000, hekou_name=name, use_norm_data=False)
        main(opt)
        print('*' * 80 + '\n\n')

