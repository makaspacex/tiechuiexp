import os
import pandas as pd
import numpy as np
import math
from pandas import Timestamp, Timedelta


def get_indexes(dataframe, days, step_null=True):
    date_ = dataframe['date-EST']
    co2 = dataframe['dissolved co2/30min']
    wl = dataframe['water depth (m)/30min']

    indexs_night = []
    indexs_day = []
    indexs_all = []
    for i, time_s in enumerate(date_):
        h = time_s.hour
        if time_s.dayofyear not in days:
            continue

        _c = co2[i]
        _wl = wl[i]

        try:
            if step_null and (pd.isna(_c) or pd.isna(_wl)):
                continue
        except Exception:
            continue

        if h >= 22 or h < 9:
            indexs_night.append(i)
        else:
            indexs_day.append(i)
        indexs_all.append(i)

    return indexs_day, indexs_night, indexs_all


def get_wl_describe(index_info, dataframe):
    index_day, index_night, index_all = index_info
    WL = dataframe['water depth (m)/30min']
    co2 = dataframe['dissolved co2/30min']

    try:
        des_day = WL[index_day].describe()
        des_night = WL[index_night].describe()
        des_all = WL[index_all].describe()

        wl_day_min = des_day['min'] if not pd.isna(des_day['min']) else 0
        wl_day_max = des_day['max'] if not pd.isna(des_day['max']) else 0
        wl_night_min = des_night['min'] if not pd.isna(des_night['min']) else 0
        wl_night_min_idx = WL[index_night].idxmin() if len(WL[index_night]) > 0 else None
        wl_night_max = des_night['max'] if not pd.isna(des_night['max']) else 0
        wl_night_max_idx = WL[index_night].idxmax() if len(WL[index_night]) > 0 else None
        wl_all_min = des_all['min'] if not pd.isna(des_all['min']) else 0

        wl_all_max = des_all['max'] if not pd.isna(des_all['max']) else 0
        wl_allmin_sub_nightmin = wl_all_min - wl_night_min
        wl_allmax_sub_nightmax = wl_all_max - wl_night_max

        wl_night_min_crespd_co2 = co2[wl_night_min_idx] if wl_night_min_idx is not None else None
        wl_night_max_crespd_co2 = co2[wl_night_max_idx] if wl_night_max_idx is not None else None

    except Exception as e:
        raise e

    wl_describe = wl_day_min, wl_day_max, wl_night_min, wl_night_max, wl_all_min, wl_all_max, wl_allmin_sub_nightmin, wl_allmax_sub_nightmax

    co2_crespd = wl_night_min_crespd_co2, wl_night_max_crespd_co2

    return wl_describe, co2_crespd


def plotfit(index_info, dataframe):
    try:
        index_day, index_night, index_all = index_info

        co2 = dataframe['dissolved co2/30min']
        WL = dataframe['water depth (m)/30min']

        co2_night = co2[index_night].values
        WL_night = WL[index_night].values
        z = np.polyfit(WL_night, co2_night, 1)

        co2_pred_night = WL_night * z[0] + z[1]
        co2_night = co2[index_night].values
        co2_night_bar = np.mean(co2_night)
        SST = np.sum(np.square(co2_night - co2_night_bar))
        SSE = np.sum(np.square(co2_night - co2_pred_night))
        r2 = 1 - SSE / SST

        return z, r2
    except Exception as e:
        return (None, None), None


def eval_fit(index_info_with_null, dataframe, z):
    index_day, index_night, index_all = index_info_with_null
    co2 = dataframe['dissolved co2/30min']
    WL = dataframe['water depth (m)/30min']

    WL_all = WL[index_all]

    co2_pred_all = []
    for wl in WL_all:
        if pd.isna(wl) or z[0] is None or z[1] is None:
            co2_pred_all.append(-1)
            continue
        _co2 = wl * z[0] + z[1]
        co2_pred_all.append(_co2)

    key_data = cal_key_data(dataframe, index_all, co2_pred_all)

    return WL_all, co2_pred_all, key_data


def cal_kkk(_t, _sal, co2):
    if _t is None or _sal is None or co2 is None:
        return -1, -1, -1

    T = _t + 273.15
    S = _sal
    ln_k = 93.4517 * (100 / T) - 60.2409 + 23.3585 * math.log(T / 100) + \
           S * (0.023517 - 0.023656 * (T / 100) + 0.0047036 * (T / 100) ** 2)
    fugFac = math.exp((-1636.75 + 12.0408 * T - 0.0327957 * T ** 2 + 3.16528 * 0.00001 * T ** 3 +
                       2 * (57.7 - 0.118 * T)) * 1.01325 / (83.1451 * T))
    pCO2 = co2 / math.exp(ln_k) / fugFac

    return ln_k, fugFac, pCO2


def cal_key_data(datafame, index_all, co2_pred):
    """
    ln(K)=93.4517(100/T)−60.2409+23.3585ln(T/100)+S*(0.023517-0.023656*(T/100)+0.0047036*(T/100)^2))
    FugFac= exp(( -1636.75 + 12.0408.*T - 0.0327957.*T.^2 + 3.16528.*0.00001.*T.^3+2*(57.7 - 0.118.*T))*1.01325/(83.1451*T)
    T=t+273.15
    S=sal
    """
    t = datafame['t-30min']
    sal = datafame['sal-30min']

    lnks, fugfacs, p_co2s = [], [], []

    for i, index in enumerate(index_all):
        _t = t[index]
        _sal = sal[index]
        if pd.isna(_t) or pd.isna(_sal) or pd.isna(co2_pred[i]) or co2_pred[i] == -1:
            ln_k = -1
            fugFac = -1
            pCO2 = -1
        else:
            ln_k, fugFac, pCO2 = cal_kkk(_t, _sal, co2_pred[i])
        lnks.append(ln_k)
        fugfacs.append(fugFac)
        p_co2s.append(pCO2)

    return lnks, fugfacs, p_co2s, t[index_all], sal[index_all]


def write_data(fp, dataframe, indexes, wl_all, co2_pred_all, key_data, WL_all2, co2_pred_all2, key_data2):
    time_stamp = dataframe['date-EST']
    ln_ks, fugfacs, p_co2s, t, sal = key_data
    ln_ks2, fugfacs2, p_co2s2, t2, sal2 = key_data2

    wl_all = wl_all.fillna(0)
    t = t.fillna(0)
    sal = sal.fillna(0)

    for index, wl, co2, ln_k, fugfac, p_co2, _t, _sal, wl2, co22, ln_k2, fugfac2, p_co22, _t2, _sal2 in \
            zip(indexes, wl_all, co2_pred_all, ln_ks, fugfacs, p_co2s, t, sal, WL_all2, co2_pred_all2, ln_ks2, fugfacs2,
                p_co2s2, t2, sal2):
        _ts = time_stamp[index]
        data = f'"{_ts}","{wl}","{co2}","{ln_k}","{fugfac}","{p_co2}","{_t}","{_sal}","{co22}","{p_co22}"\n'
        fp.write(data)


def get_co2_info(index_info, dataframe):
    #  找出白天wl最大值对应的CO2值，将白天所有的CO2值减去这个CO2值，输出这些数据，然后求这些数据的平均值和标准偏差
    index_day, index_night, index_all = index_info

    co2 = dataframe['dissolved co2/30min']
    WL = dataframe['water depth (m)/30min']
    t = dataframe['t-30min']
    sal = dataframe['sal-30min']

    co2_day = co2[index_day]
    co2_night = co2[index_night]
    wl_day = WL[index_day].dropna()
    wl_night = WL[index_day].dropna()

    wl_day_idx_max = wl_day.idxmax() if len(wl_day) > 0 else None
    wl_night_idx_max = wl_night.idxmax() if len(wl_night) > 0 else None

    co2_day_corsp_wl_max = co2[wl_day_idx_max] if wl_day_idx_max is not None else None
    co2_night_corsp_wl_max = co2[wl_night_idx_max] if wl_night_idx_max is not None else None

    co2_day_sub_corsp_co2 = co2_day - co2_day_corsp_wl_max if co2_day_corsp_wl_max is not None else None
    co2_night_sub_corsp_co2 = co2_night - co2_night_corsp_wl_max if co2_night_corsp_wl_max is not None else None

    co2_daysub_mean = np.mean(co2_day_sub_corsp_co2) if co2_day_sub_corsp_co2 is not None else None
    co2_nightsub_mean = np.mean(co2_night_sub_corsp_co2) if co2_night_sub_corsp_co2 is not None else None
    co2_daysub_std = np.std(co2_day_sub_corsp_co2) if co2_day_sub_corsp_co2 is not None else None
    co2_nightsub_std = np.std(co2_night_sub_corsp_co2) if co2_night_sub_corsp_co2 is not None else None

    co2_day_min = co2[index_day].min() if len(co2[index_day]) > 0 else None
    co2_night_min = co2[index_night].min() if len(co2[index_night]) > 0 else None

    t_day_mean = t[index_day].mean() if len(t[index_day]) > 0 else None
    t_night_mean = t[index_night].mean() if len(t[index_night]) > 0 else None

    sal_day_mean = sal[index_day].mean() if len(sal[index_day]) > 0 else None
    sal_night_mean = sal[index_night].mean() if len(sal[index_night]) > 0 else None

    co2_des = co2_daysub_mean, co2_daysub_std, co2_nightsub_mean, co2_nightsub_std, co2_day_min, co2_night_min
    t_s_des = t_day_mean, t_night_mean, sal_day_mean, sal_night_mean

    return co2_day_sub_corsp_co2, co2_night_sub_corsp_co2, co2_des, t_s_des


def write_co2_info(index_info, dataframe, fp_co2_sub_info, co2_day_sub_corsp_co2, co2_night_sub_corsp_co2):
    index_day, index_night, index_all = index_info
    co2 = dataframe['dissolved co2/30min']
    WL = dataframe['water depth (m)/30min']
    date_ = dataframe['date-EST']

    try:
        all_ = pd.concat([co2_day_sub_corsp_co2, co2_night_sub_corsp_co2]).sort_index()
        for idx in all_.index:
            co2_sub = all_[idx]
            fp_co2_sub_info.write(f'"{idx}","{date_[idx]}","{co2[idx]}","{co2_sub}"\n')
    except Exception as e:
        pass


def write_co2_other_info(index_info, t_s, dataframe, fp_co2_other_info, co2_des, t_s_des, key_day, key_night):
    try:
        index_day, index_night, index_all = index_info
        co2_daysub_mean, co2_daysub_std, co2_nightsub_mean, co2_nightsub_std, co2_day_min, co2_night_min = co2_des
        t_day_mean, t_night_mean, sal_day_mean, sal_night_mean = t_s_des

        ln_k_day, fugFac_day, pCO2_day = key_day
        ln_k_night, fugFac_night, pCO2_night = key_night

        data = f'"{t_s}","{co2_daysub_mean}","{co2_daysub_std}","{co2_nightsub_mean}",' \
               f'"{co2_nightsub_std}","{co2_day_min}","{co2_night_min}","{t_day_mean}",' \
               f'"{t_night_mean}","{sal_day_mean}","{sal_night_mean}","{ln_k_day}","{fugFac_day}",' \
               f'"{pCO2_day}","{ln_k_night}","{fugFac_night}","{pCO2_night}"\n'
        fp_co2_other_info.write(data)
    except Exception as e:
        pass


def process(excel_dfs, sheet_name, margin_day):
    os.makedirs('data', exist_ok=True)
    result_file_name = f"data/result_{sheet_name}_{margin_day}_days_v2.csv"
    pred_file_name = f"data/pred_{sheet_name}_{margin_day}_days_v2.csv"
    co2_sub_file_name = f"data/co2_sub_{sheet_name}_{margin_day}_days_v2.csv"
    co2_other_info_file_name = f"data/co2_other_info_{sheet_name}_{margin_day}_days_v2.csv"

    dataframe = excel_dfs[sheet_name]
    date_ = dataframe['date-EST']

    fp = open(result_file_name, 'w+', encoding='utf-8')
    fp.write("time,k,b,r2, wl_day_min, wl_night_min, wl_all_min, wl_day_max,wl_night_max," +
             "wl_all_max,wl_allmin_sub_nightmin,wl_allmax_sub_nightmax\n")

    fp_pred = open(pred_file_name, 'w+', encoding='utf-8')
    fp_pred.write(
        f'"time","water depth (m)/30min","co2_pred","ln_k","fugfac","pco2_pred","t-30min","sal-30min","co2_pred2","pco2_pred2"\n')

    fp_co2_sub_info = open(co2_sub_file_name, 'w+', encoding='utf-8')
    fp_co2_sub_info.write(f'"index","time","co2", "co2_sub",""\n')

    fp_co2_other_info = open(co2_other_info_file_name, 'w+', encoding='utf-8')
    line = f'"time","co2_daysub_mean","co2_daysub_std","co2_nightsub_mean",' \
           f'"co2_nightsub_std","co2_day_min","co2_night_min","t_day_mean",' \
           f'"t_night_mean","sal_day_mean","sal_night_mean","ln_k_day","fugFac_day",' \
           f'"pCO2_day","ln_k_night","fugFac_night","pCO2_night"\n'
    fp_co2_other_info.write(line)

    start_dayofyear = date_.min().dayofyear
    end_dayofyear = date_.max().dayofyear
    start_timestamp = date_.min()

    print(f"{'*'*30}{date_.min()}-{date_.max()}{'*'*30}")

    for i, _d in enumerate(range(start_dayofyear, margin_day + end_dayofyear, margin_day)):

        _td = Timedelta(days=_d - start_dayofyear)
        t_s = start_timestamp + _td

        ds = [_d + xx for xx in range(margin_day)]
        try:
            index_info = get_indexes(dataframe, ds, step_null=True)
            wl_describe, co2_crespd = get_wl_describe(index_info, dataframe)

            z, r2 = plotfit(index_info, dataframe=dataframe)
            wl_night_min_crespd_co2, wl_night_max_crespd_co2 = co2_crespd

            # ==================== 我是华丽的分割线 ====================================================================
            index_info_with_null = get_indexes(dataframe, ds, step_null=False)
            #  找出白天wl最大值对应的CO2值，将白天所有的CO2值减去这个CO2值，输出这些数据，然后求这些数据的平均值和标准偏差
            co2_day_sub_corsp_co2, co2_night_sub_corsp_co2, co2_des, t_s_des = get_co2_info(index_info_with_null,
                                                                                            dataframe)
            co2_daysub_mean, co2_daysub_std, co2_nightsub_mean, co2_nightsub_std, co2_day_min, co2_night_min = co2_des
            t_day_mean, t_night_mean, sal_day_mean, sal_night_mean = t_s_des
            write_co2_info(index_info, dataframe, fp_co2_sub_info, co2_day_sub_corsp_co2, co2_night_sub_corsp_co2)

            key_day = cal_kkk(t_day_mean, sal_day_mean, co2_daysub_mean)
            key_night = cal_kkk(t_night_mean, sal_night_mean, co2_nightsub_mean)
            write_co2_other_info(index_info, t_s, dataframe, fp_co2_other_info, co2_des, t_s_des, key_day, key_night)

            WL_all, co2_pred_all, key_data = eval_fit(index_info_with_null, dataframe, z)
            wl_day_min, wl_day_max, wl_night_min, wl_night_max, \
            wl_min, wl_max, wl_allmin_sub_nightmin, wl_allmax_sub_nightmax = wl_describe

            X = [wl_night_min, wl_night_max]
            Y = [wl_night_min_crespd_co2, wl_night_max_crespd_co2]
            try:
                z_wl_minmax = np.polyfit(X, Y, 1)
            except Exception:
                z_wl_minmax = [None, None]

            WL_all2, co2_pred_all2, key_data2 = eval_fit(index_info_with_null, dataframe, z_wl_minmax)

            print(f"{_d},{i},{t_s},z:{z},r2:{r2}")
            data = f'"{t_s}","{z[0]}","{z[1]}","{r2}","{wl_day_min}",' \
                   f'"{wl_night_min}","{wl_min}","{wl_day_max}","{wl_night_max}","{wl_max}",' \
                   f'"{wl_allmin_sub_nightmin}","{wl_allmax_sub_nightmax}"\n'
            fp.write(data)

            write_data(fp_pred, dataframe, index_info_with_null[2], WL_all, co2_pred_all, key_data, WL_all2,
                       co2_pred_all2, key_data2)

        except Exception as e:
            print(f"{_d},{t_s} error:{e}")
            continue
    fp.close()
    fp_pred.close()
    fp_co2_sub_info.close()
    fp_co2_other_info.close()


def main():
    excel_path = '6.25 dissolved co2 for three years.xlsx'

    margin_day = 3  # 比如是2天为一组
    sheet_names = ["2014 DATA WITH HALF HOUR", "2015 DATA WITH HALF HOUR", "2016 DATA WITH HALF HOUR"]

    print(f"reading excel:{excel_path}")
    excel_dfs = pd.read_excel(excel_path, sheet_name=None)
    print(f"done. start process sheets...")

    for s_name in sheet_names:
        print(f"{s_name},margin_day:{margin_day}")
        process(excel_dfs, s_name, margin_day=margin_day)


if __name__ == '__main__':
    main()
