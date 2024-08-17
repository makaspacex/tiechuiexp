#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2022/11/28
# @fileName main.py
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
import numpy as np
import netCDF4 as nc
from pathlib import Path
import os
import glob
import argparse
import pandas as pd


def main(f_path, lon_error=0.01, lat_error=0.02):
    f_path = Path(f_path)
    df = pd.read_excel(f_path)

    f = open(f'{f_path.stem}_filt_lon_error{lon_error}_lat_error{lat_error}.csv', 'w+', encoding='utf8')
    f.write(f"lon,lat,{','.join([ f'{x}' for x in df.columns[4:]])}\n")


    for index, row in df.iterrows():
        if index % 100 == 0:
            print(index)

        lon, lat = row['lon'], row['lat']
        df_fl = df[(abs(df['lon-1'] - lon) <= lon_error) & (abs(df['lat-1'] - lat) <= lat_error)]
        mean_s = df_fl.mean()
        values = [lon, lat] + mean_s.iloc[4:].values.tolist()
        line = ",".join([f"{'' if np.isnan(ele) else ele}" for ele in values])
        f.write(f"{line}\n")
    f.close()
    print("ok")


if __name__ == '__main__':
    f_path = Path("./data/2011-2016 monthlyT and sality.xlsx")
    lon_error = 0.01
    lat_error = 0.02
    main(f_path=f_path, lon_error=lon_error, lat_error=lat_error)
