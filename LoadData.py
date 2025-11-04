import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
import xarray as xr
import random
from scipy.interpolate import NearestNDInterpolator
from collections import OrderedDict,defaultdict
import pandas as pd

def compute_mean_std(vars_concat):
    """
    计算输入数据中每个变量的均值和标准差，用于归一化。

    参数:
        vars_concat (np.ndarray): 拼接后的变量数组，形状为 (N, C, H, W)

    返回:
        means (np.ndarray): 每个变量的均值，形状为 (C,)
        stds (np.ndarray): 每个变量的标准差，形状为 (C,)
    """
    # 沿着样本和空间维度计算每个通道的均值和标准差
    means = np.nanmean(vars_concat, axis=(0, 2, 3))
    stds = np.nanstd(vars_concat, axis=(0, 2, 3))

    return means, stds

def fillmiss_2d(x):
    """
    利用最近邻插值填充数据的nan值
    :param x: 含nan值的二维数组
    :return: 最近邻插值后的二维数组
    """
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(x[mask])
    interp0 = NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

class MakeDatasetRNN(Dataset):
    """
    Dataset implementation for RNN training
    """

    def __init__(self, mypara, rnn2configs, dataset_kind, norm= False):
        self.mypara = mypara
        self.rnn2configs = rnn2configs
        if dataset_kind == 'trainer':
            path = mypara.path_train
        elif dataset_kind == 'eval':
            path = mypara.path_eval
        elif dataset_kind == 'test':
            path = mypara.path_test
        else:
            raise ValueError("Please enter the correct dataset_kind('trainer','eval', 'test')")
        self.dataset_kind = dataset_kind

        lat_min, lat_max = mypara.lat_range
        lon_min, lon_max = mypara.lon_range
        data_in = xr.open_mfdataset(path, concat_dim="time", parallel=False, combine="nested",
                                    engine="netcdf4")

        print(f"dataset_kind: {dataset_kind}\n")
        # print("------original data:\n", data_in)

        data_in = data_in.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        data_in = data_in.astype(np.float32)

        print(f"lon: {data_in['longitude'][0].values} E to {data_in['longitude'][-1].values}\n "
              f"lat: {data_in['latitude'][0].values} N to {data_in['latitude'][-1].values}\n ")

        self.lat = data_in["latitude"].values
        self.lon = data_in["longitude"].values

        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.field_data = data_in['adt'].values[:, None, :, :]
        self.mean = np.load(mypara.path_mean)
        self.std = np.load(mypara.path_std)
        if norm:
            self.field_data = (self.field_data - self.mean[0]) / self.std[0]

        if mypara.need_uv:
            uo = data_in['ugos'].values[:, None, :, :]
            vo = data_in['vgos'].values[:, None, :, :]
            self.field_data = np.concatenate((self.field_data, uo, vo), axis=1)

        del data_in

        self.field_data[abs(self.field_data) > 100] = np.nan
        mask_nan = np.isnan(self.field_data[0, 0, :, :])
        T, C, H, W = self.field_data.shape

        if self.mypara.need_mask:
            mask_nan = np.broadcast_to(mask_nan[None, None, :, :], (T, 1, H, W))
            self.field_data = np.concatenate((self.field_data, mask_nan), axis=1)
            C = C + 1

        p = self.rnn2configs.patch_size

        # 先 reshape 出补丁维度
        self.field_data = self.field_data.reshape(T, C, H // p, p, W // p, p)
        # 交换维度使得每个 patch 内的像素连续
        self.field_data = self.field_data.transpose(0, 1, 3, 5, 2, 4)
        # 再 reshape 合并 patch 内像素到通道维度
        self.field_data = self.field_data.reshape(T, C * p * p, H // p, W // p)
        print(f"data shape:{self.field_data.shape}")

        # 预计算索引范围
        self.st_min = self.rnn2configs.total_length
        self.ed_max = self.field_data.shape[0]
        self.valid_indices = list(range(self.st_min, self.ed_max))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        indice = self.valid_indices[idx]
        dataX = np.nan_to_num(self.field_data[indice - self.rnn2configs.total_length: indice])
        dataX = torch.from_numpy(dataX)
        return dataX

    def get_grids(self):
        return self.lon, self.lat

    def get_info(self):
        return self.mean, self.std

class MakeIterDataset(IterableDataset):
    """
    online reading dataset
    """
    def __init__(self, config, dataset_kind, norm=False):
        self.config = config
        if dataset_kind == 'train':
            path = config.path_train

        elif dataset_kind == 'eval':
            path = config.path_eval

        elif dataset_kind == 'test':
            path = config.path_test

        else:
            print("Please enter the correct dataset_kind('train','eval', 'test')")
        if config.need_wind:
            path_wind = config.path_wind
        self.dataset_kind = dataset_kind

        lat_min,lat_max = config.lat_range
        lon_min,lon_max = config.lon_range

        data_in = xr.open_mfdataset(path, concat_dim="time", parallel=False, combine="nested",
                    engine="netcdf4" )

        print(f"dataset_kind: {dataset_kind}\n")


        data_in = data_in.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        data_in = data_in.astype(np.float32)

        print(f"lon: {data_in['longitude'][0].values} E to {data_in['longitude'][-1].values}\n "
              f"lat: {data_in['latitude'][0].values} N to {data_in['latitude'][-1].values}\n "
              )

        self.lat = data_in["latitude"].values
        self.lon = data_in["longitude"].values
        np.save(config.path_lon_lat, [self.lon, self.lat])

        self.input_length = config.input_length
        self.output_length = config.output_length
        self.field_data = data_in['adt'].values[:, None, :, :]

        dates = data_in['time']
        self.dates = [pd.Timestamp(d) for d in dates.values]
        print(f"dates: {self.dates[0]} to {self.dates[-1]}")

        print(f'length of dates: {len(dates)}\n')

        if config.need_uv:
            uo = data_in['ugos'].values[:,None,:,:]
            vo = data_in['vgos'].values[:,None,:,:]
            self.field_data = np.concatenate((self.field_data ,uo, vo), axis=1)

        del data_in

        self.mean = np.load(config.path_mean)
        self.std = np.load(config.path_std)
        if norm:
            for i in range(self.field_data.shape[1]):
                self.field_data[:, i, ...] = (self.field_data[:, i, ...] - self.mean[i]) / self.std[i]  # todo

        if config.need_wind:
            print("loading wind...")
            data_in = xr.open_dataset(path_wind)
            if dataset_kind == 'train':
                data_in = data_in.sel(time=slice('1993-01-01', '2021-12-31'))
            elif dataset_kind == 'eval':
                data_in = data_in.sel(time=slice('2022-01-01', '2022-12-31'))
            elif dataset_kind == 'test':
                data_in = data_in.sel(time=slice('2023-01-01', '2024-06-14'))
            # data_in = data_in.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            data_in = data_in.astype(np.float32)
            print(f"u10shape{data_in['u10'].shape}")

            self.field_data = np.concatenate(
                (self.field_data, data_in['u10'].values[:, None, ...], data_in['v10'].values[:, None, ...]),
                axis=1
            )

        self.field_data[abs(self.field_data) > 100] = np.nan

        mask_nan = np.isnan(self.field_data[0, 0, :, :])
        if self.config.need_mask:
            T, C, H, W = self.field_data.shape
            mask_nan = np.broadcast_to(mask_nan[ None, None, :, :],(T, 1, H, W))
            self.field_data = np.concatenate((self.field_data, mask_nan), axis=1)  # todo

        # # 初始化一个和 self.field_data 同样形状的数组来存放插值后的数据
        # self.field_data_x = np.empty_like(self.field_data)

        # 遍历time和channel两个维度，对每个二维(lat, lon)切片进行填充处理
        # for t in range(self.field_data.shape[0]):
        #     for c in range(self.field_data.shape[1]):
        #         self.field_data_x[t, c, :, :] = fillmiss_2d(self.field_data[t, c, :, :])

        # print(f"data shape:{self.field_data.shape}")

        st_min = self.input_length
        ed_max = self.field_data.shape[0] - self.output_length
        self.time_indices = list(range(st_min, ed_max+1))
        self.length = len(self.time_indices)

        if self.config.winter_only or  self.config.summer_only:
            # 1. 用一个临时 dict 收集原始的跨年段
            raw = defaultdict(list)
            for idx, dt in enumerate(self.dates):
                m, y = dt.month, dt.year
                if not self.config.summer_only and not self.config.winter_only:
                    raw[y].append(idx)
                elif self.config.summer_only:
                    if 4 <= m <= 9:
                        raw[y].append(idx)
                else:  # winter_only
                    if m >= 10:
                        raw[y].append(idx)
                    elif m <= 3:
                        raw[y - 1].append(idx)

            # 2. 过滤掉「不完整」的跨年冬季段
            #    只保留那些同时有 (year,10–12) 和 (year+1,1–3) 两部分的 season_year
            self.segments = OrderedDict()
            if dataset_kind == 'train' and config.winter_only:
                self.segments[1992] = raw[1992]

            for season_year in sorted(raw):
                idxs = raw[season_year]
                if self.config.winter_only:
                    # 拆两部分看有没有数据
                    oct_dec = [i for i in idxs if (
                            self.dates[i].year == season_year and 10 <= self.dates[i].month <= 12)]
                    jan_mar = [i for i in idxs if (
                            self.dates[i].year == season_year + 1 and 1 <= self.dates[i].month <= 3)]
                    if not (oct_dec and jan_mar):
                        continue  # 不完整，剔除
                    combined = sorted(oct_dec + jan_mar)
                    self.segments[season_year] = combined

                elif self.config.summer_only:
                     self.segments[season_year] = sorted(idxs)

                else:
                    # 全年模式下直接按年保留
                    self.segments[season_year] = sorted(idxs)

            # 3. 统计总长度（实际样本数要减去前后 input/output 长度）
            self.length = sum(
                max(0, len(idxs) - self.input_length - self.output_length)
                for idxs in self.segments.values()
            )

    def __iter__(self):
        if self.config.winter_only or  self.config.summer_only:
            years = list(self.segments.keys())
            # 如果是训练集，打乱年序
            if self.dataset_kind == 'train':
                random.shuffle(years)
            for season_year in years:
                indices = self.segments[season_year]
                indices = indices[self.input_length:len(indices)-self.output_length]
                if self.dataset_kind == 'train':
                    random.shuffle(indices)
                for indice in indices:
                    dataX = torch.from_numpy(np.nan_to_num(self.field_data[indice - self.input_length  : indice]))
                    dataY = torch.from_numpy(self.field_data[indice  : indice + self.output_length,0:self.config.output_channel,...])

                    yield dataX, dataY
        else:

            if self.dataset_kind == 'train':
                random.shuffle(self.time_indices)
            for indice in self.time_indices:
                dataX = torch.from_numpy(np.nan_to_num(self.field_data[indice - self.input_length: indice]))
                dataY = torch.from_numpy(self.field_data[indice: indice + self.output_length,0:self.config.output_channel, ...])
                yield dataX, dataY

    def __len__(self):
        return self.length
    def get_grids(self):
        return  self.lon, self.lat

    def get_info(self):
        return self.mean, self.std
    def get_times(self):
        return self.dates[self.time_indices[0]: self.time_indices[-1]+1]


if __name__ == '__main__':
    from configs import mypara
    evalset = MakeIterDataset(mypara,'test')
    print(f'len {len(evalset)}')
    dates = evalset.get_times()
    print(len( dates))
    print(dates[0])
    print(dates[-1])
    n=0
    for dataX, dataY in evalset:
        n+=1
    print(f'{n} n')

