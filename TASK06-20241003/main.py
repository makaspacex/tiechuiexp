#! /usr/bin/python
# -*- coding: utf-8 -*-
# @author izhangxm
# @date 2022/5/21
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

parser = argparse.ArgumentParser()
parser.add_argument('--lon', )


def save_data(dataset, lon_range, lat_range, data_name, out_put_dir='output'):
    lat_s, lat_e = [float(ele) for ele in lat_range.split('-')]
    lon_s, lon_e = [float(ele) for ele in lon_range.split('-')]

    data = dataset.variables
    lat = data['lat'][:].data
    lon = data['lon'][:].data

    lat_mask = np.where((lat >= lat_s) & (lat <= lat_e), np.ones_like(lat), np.zeros_like(lat))
    lon_mask = np.where((lon >= lon_s) & (lon <= lon_e), np.ones_like(lon), np.zeros_like(lon))

    lat_i = np.where(lat_mask > 0)[0]
    lon_i = np.where(lon_mask > 0)[0]

    wind_speed_AW = data[data_name][:].data
    wind_speed_data = wind_speed_AW[:, lat_i, :][:, :, lon_i]

    out_put_dir = os.path.join(out_put_dir, f"{dataset.time_coverage_start}")

    out_file_path = f"{dataset.time_coverage_start} {data_name} lat {lat_range} lon {lon_range}.csv"
    out_file_path = os.path.join(out_put_dir, out_file_path)

    os.makedirs(out_put_dir, exist_ok=True)

    with open(out_file_path, 'w', encoding='utf8') as f:
        title_line0 = '"",'
        for li in lon_i:
            title_line0 += f'"{lon[li]}",'

        f.write(f"{title_line0}\n")
        lat_len, lon_len = wind_speed_data.shape[1:]
        for r in range(lat_len):
            row_str = f'"{lat[lat_i[r]]}",'
            for c in range(lon_len):
                _pass = [str(ele) for ele in wind_speed_data[:, r, c]]
                _tmp = ",".join(_pass)
                row_str += f'"{_tmp}",'
            f.write(f"{row_str}\n")

    print(out_file_path)


def main(f_path, lon_range, lat_range, data_names, out_put_dir='output'):
    dataset = nc.Dataset(f_path)
    for name in data_names:
        save_data(dataset, lat_range=lat_range, lon_range=lon_range, data_name=name, out_put_dir=out_put_dir)


if __name__ == '__main__':
    # f_path = Path("./data/RSS_AMSR2_ocean_L3_daily_2020-08-01_v08.2.nc")
    # f_path = Path("./data/RSS_AMSR2_ocean_L3_daily_2018-07-11_v08.2.nc")
    lat_range = "28-33"
    lon_range = "120-124.5"
    data_names = ['SST', 'wind_speed_LF', 'wind_speed_MF', 'wind_speed_AW', 'water_vapor'
        , 'cloud_liquid_water', 'rain_rate', 'land_mask', 'sea_ice_mask', 'coast_mask', 'noobs_mask']
    # main(f_path, lon_range, lat_range, data_names)
    out_put_dir = "output/2018 july wind"
    files = glob.glob('2018 july wind/*.nc')
    for f_path in files:
        main(f_path, lon_range, lat_range, data_names,out_put_dir)
