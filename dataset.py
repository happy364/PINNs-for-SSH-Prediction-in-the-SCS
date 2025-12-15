import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr

class MvDataset(Dataset):
    def __init__(self, args, mode='train', norm=False):
        self.args = args
        self.data = xr.open_mfdataset(self.args.datapath, combine='nested', concat_dim='time')

        if mode == 'train':
            time_span = slice(args.start_time_train, args.end_time_train)
        elif mode == 'eval':
            time_span = slice(args.start_time_val, args.end_time_val)
        elif mode == 'test':
            time_span = slice(args.start_time_test, args.end_time_test)
        else:
            raise ValueError('mode must be train, eval or test')

        self.data = self.data.sel(time=time_span)
        self.lat = self.data.latitude.values
        self.lon = self.data.longitude.values

        self.field = None

        if args.need_ssh:
            self.adt = np.expand_dims(self.data.adt.values.astype(np.float32), axis=1)
            if self.field is None:
                self.field = self.adt
            else:
                self.field = np.concatenate([self.field, self.adt], axis=1)
            # del self.adt
        if args.need_uv:
            self.uo = self.data.ugos.values.astype(np.float32)
            self.vo = self.data.vgos.values.astype(np.float32)
            self.field = np.concatenate([self.uo, self.vo], axis=1)
            # del self.uo, self.vo

        # ssh data and ugos, vgos were not been  normalized
        self.mean = np.load(args.path_means)
        self.std = np.load(args.path_stds)
        if norm:
            for i in range(self.field_data.shape[1]):
                self.field_data[:, i, ...] = (self.field_data[:, i, ...] - self.mean[i]) / self.std[i]

        if args.need_wind:
            self.wind_data = xr.open_dataset(args.wind_path).sel(time=time_span)
            self.u10 = np.expand_dims(self.wind_data.u10.values.astype(np.float32), axis=1)
            self.v10 = np.expand_dims(self.wind_data.v10.values.astype(np.float32), axis=1)
            self.field = np.concatenate([self.field, self.u10, self.v10], axis=1)
            del self.u10, self.v10

        if args.need_mask:
            mask = np.load(args.path_land_mask)
            mask = mask[None,None,...]
            mask = np.tile(mask,(self.field.shape[0],1,1,1))
            self.field = np.concatenate([self.field, mask], axis=1)
            del mask

        self.field[abs(self.field) > 100] = np.nan

        # mask_ssh = np.isnan(self.adt[0, 0])
        # np.save('mask.npy', mask_ssh)

        self.input_length = args.input_length
        self.field = np.nan_to_num(self.field)

        self.output_length = args.output_length
        if args.model_name == 'predrnn':
            p = self.args.model_config['patch_size']
            T = self.field.shape[0]
            C = self.field.shape[1]
            H, W = self.field.shape[2:]
            # 先 reshape 出补丁维度
            self.field = self.field.reshape(T, C, H // p, p, W // p, p)
            # 交换维度使得每个 patch 内的像素连续
            self.field = self.field.transpose(0, 1, 3, 5, 2, 4)
            # 再 reshape 合并 patch 内像素到通道维度
            self.field = self.field.reshape(T, C * p * p, H // p, W // p)

    def __len__(self):
        return len(self.data.time)-self.input_length-self.output_length+1

    def __getitem__(self, idx):
        if self.args.model_name == 'predrnn':
            datax = self.field[idx:idx+self.input_length+self.output_length]
            return torch.from_numpy(datax)
        else:
            datax = np.nan_to_num(self.field[idx:idx+self.input_length])

            datay = self.field[idx+self.input_length:idx+self.input_length+self.output_length, :self.args.output_channels]

            return torch.from_numpy(datax), torch.from_numpy(datay)

if __name__ == '__main__':
    from configs import parse_args
    args = parse_args()
    dataset = MvDataset(args, mode='test')
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)






