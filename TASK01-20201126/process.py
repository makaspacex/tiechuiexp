#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
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
import pandas as pd
import copy
import glob
import traceback
import numpy as np

# DataFrame from pandas
def processSheetDF(sheet_df, margin_minutes, utc_column_name='date-UTC', TAG=None):
    print("预处理数据,%s" % TAG)
    standard_time_column_name = 'STANDARD_TIME'

    # 得到标准时间列（忽略秒）
    standard_time_colmn = []
    date_list = copy.deepcopy(sheet_df[utc_column_name])
    need_to_drop_indexs = []
    for index, timeStamp in enumerate(date_list):
        try:
            if isinstance(timeStamp, pd._libs.tslibs.nattype.NaTType):
                need_to_drop_indexs.append(index)
                continue
            standard_time_colmn.append(pd.Timestamp(timeStamp.to_pydatetime().strftime('%Y-%m-%d %H:%M') + ':00'))
        except Exception as e:
            print("%s: %d行的时间+%s+出现错误:%s" % (TAG, index, timeStamp, e))
            need_to_drop_indexs.append(index)

            # sheet_df = sheet_df.drop(index=index)
    sheet_df = sheet_df.drop(need_to_drop_indexs)
    sheet_df.insert(0, standard_time_column_name, standard_time_colmn)

    # 开始时间与结束时间
    standard_time_colmn.sort()
    start_Timestamp = standard_time_colmn[0]
    end_Timestamp = standard_time_colmn[-1]

    # 间隔时间
    margin_timedelta = pd.Timedelta(minutes=margin_minutes)

    # 准备空白行
    row_blank = sheet_df.iloc[0].values

    for i in range(len(row_blank)):
        row_blank[i] = None

    next_time = start_Timestamp + margin_timedelta

    print('开始查找缺失数据')
    n = 0

    # 为了加快速度 使用np的形式
    sheet_df_value = sheet_df.values

    while True:

        if next_time > end_Timestamp:
            break

        if next_time not in standard_time_colmn:
            _row = row_blank.copy()
            row_blank[0] = next_time
            sheet_df_value = np.insert(sheet_df_value, 0, row_blank, axis=0)

            # sheet_df.loc[next_time] = row_blank
            n += 1
            print("%d:%s的%s内发现缺失数据，时间：%s" % (n, TAG, sheet_name, next_time))
        next_time += margin_timedelta

    columns = sheet_df.columns.to_list()
    sheet_df = pd.DataFrame(sheet_df_value, columns=columns)
    sheet_df = sheet_df.set_index(standard_time_column_name)
    sheet_df = sheet_df.sort_index()
    return sheet_df


def processFile(file_path, sheet_name, margin_minutes, save_path=None, utc_column_name='date-UTC'):
    pass


def getExcelWriter(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    mode = 'w'
    if os.path.exists(save_path):
        mode = 'a'
    writer = pd.ExcelWriter(save_path, mode=mode, engine="openpyxl")
    return writer


def writeToExcel(sheet_df, writer, sheet_name, overwrite=True):
    if overwrite:
        wb = writer.book
        if sheet_name in wb.sheetnames:
            wb.remove(wb[sheet_name])
    sheet_df.to_excel(writer, sheet_name, index=True)


if __name__ == '__main__':

    # file_list = glob.glob("data_ready_for_process/*.xls")

    # file_list = [
    #     "data_ready_for_process/xls/July with all data.xls",
    #     "data_ready_for_process/xls/August with all data.xls",
    #     "data_ready_for_process/xls/September with all data.xls",
    #     "data_ready_for_process/xls/October with all data.xls",
    #     "data_ready_for_process/xls/November with all data.xls",
    #     "data_ready_for_process/xls/December with all data.xls",
    # ]

    file_list = [
        "data_ready_for_process/xlsx/July with all data.xlsx",
        # "data_ready_for_process/xlsx/August with all data.xlsx",
        # "data_ready_for_process/xlsx/September with all data.xlsx",
        # "data_ready_for_process/xlsx/October with all data.xlsx",
        # "data_ready_for_process/xlsx/November with all data.xlsx",
        # "data_ready_for_process/xlsx/December with all data.xlsx",
    ]
    # file_list = glob.glob("data_ready_for_process/*.xls")

    configures = [
        {"file_path": "", "sheet_name": "2014", "margin_minutes": 5, "save_path": None, "utc_column_name": 'date-UTC'},
        {"file_path": "", "sheet_name": "2015", "margin_minutes": 5, "save_path": None, "utc_column_name": 'date-UTC'},
        {"file_path": "", "sheet_name": "2016", "margin_minutes": 30, "save_path": None, "utc_column_name": 'date-UTC'},
    ]

    # file_list = [
    #     "data_ready_for_process/add/July with all data-2.xlsx",
    #     "data_ready_for_process/add/September with all data-2.xlsx",
    # ]

    for file_path in file_list:
        # 打开文件和写入文件
        print('读取文件:%s' % (file_path))
        excel_ = pd.read_excel(file_path, sheet_name=None)

        file_name = os.path.basename(file_path)
        save_path = os.path.join('aaa', file_name)
        print('打开保存的文件:%s' % (save_path))
        writer = getExcelWriter(save_path=save_path)

        sheet_name = None
        for index, conf in enumerate(configures):
            try:
                sheet_name = conf['sheet_name']
                sheet_df = excel_[sheet_name]
                margin_minutes = conf['margin_minutes']
                utc_column_name = conf['utc_column_name']
                TAG = "%s:%s" % (file_name, sheet_name)
                sheet_df = processSheetDF(sheet_df, margin_minutes, utc_column_name=utc_column_name, TAG=TAG)

                print("正在保存文件%s" % save_path)
                writeToExcel(sheet_df, writer=writer, sheet_name=sheet_name)

            except Exception as e:
                print("errorSheet:%s, errorFile:%s" % (sheet_name, file_name))
                print(traceback.format_exc())

        # 持久化到磁盘
        writer.save()
        writer.close()

    print('完成！')
