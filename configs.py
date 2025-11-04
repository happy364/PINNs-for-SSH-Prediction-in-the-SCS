import argparse
import torch
from pathlib import Path
import re

def get_args():
    parser = argparse.ArgumentParser(description="Full config for SSH experiments")

    # 环境与训练超参数
    parser.add_argument('--code_env', type=str, default='windows')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_eval', type=int, default=16)
    parser.add_argument('--is_acc', action='store_false', help='Enable gradient accumulation')  # todo
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--early_stopping', action='store_false')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_norm', type=float, default=2.0)
    parser.add_argument('--loss_weighting', type=str, default=None)

    # 随机种子
    parser.add_argument('--SEED', type=int, default=42)

    # 数据选项
    parser.add_argument('--summer_only', action='store_true')
    parser.add_argument('--winter_only', action='store_true')
    parser.add_argument('--loss_mask', action='store_true')

    # 输入特征选项
    parser.add_argument('--need_ssh', action='store_false')
    parser.add_argument('--need_uv', action='store_true')
    parser.add_argument('--need_wind', action='store_true')
    parser.add_argument('--need_mask', action='store_true')
    parser.add_argument('--no_weight', action='store_true')

    # 模式与PINN设置
    parser.add_argument('--is_finetune', action='store_true')
    parser.add_argument('--is_pinn', action='store_true')
    parser.add_argument('--pinn_lambda', type=float, default=0.)
    parser.add_argument('--finetune_path', type=str, default='')

    # 输入输出设置
    parser.add_argument('--input_length', type=int, default=10)
    parser.add_argument('--output_length', type=int, default=10)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=1)

    # 空间范围
    parser.add_argument('--extent', nargs=4, type=float, default=[104, 124, 2, 22])

    return parser.parse_args()


class Mypara:
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)

        self.device = torch.device(self.device)
        a, b, c, d = self.extent
        self.lat_range = [c, d]
        self.lon_range = [a, b]

        # 数据路径
        if self.code_env == 'windows':
            base = Path('')
            self.path_mask = r"" # path of mask, land: 1; water: 0
            self.path_mean = r""
            self.path_std = r""
            self.model_savepath = r""
            self.depth_path = r""
            self.finetune_path = r''
            self.path_wind = r''
            self.input_length = 10
            self.path_lon_lat = r""

        paths = sorted(base.glob("*.nc"), key=lambda x: int(re.search(r'\d+', x.stem).group()))
        self.path_train = paths[:29]
        self.path_eval = paths[29]
        self.path_test = paths[30:32]

        if self.winter_only:
            self.path_train = paths[:30]
            self.path_eval = paths[29:31]
            self.path_test = paths[30:32]

        # 通道逻辑
        if self.need_ssh:
            self.file_name = 'ssh'
        if self.need_uv:
            self.input_channel += 2
            self.file_name = 'ssh_uv'
        if self.need_wind:
            self.input_channel += 2
        if self.need_mask:
            self.input_channel += 1
            self.file_name = self.file_name + f'_mask'

        if self.summer_only:
            self.file_name = self.file_name + f'_summer'
        if self.winter_only:
            self.file_name = self.file_name + f'_winter'

        if self.is_pinn:
            self.file_name = self.file_name + f'_pinn_{self.pinn_lambda:.3f}'

        self.file_name = self.file_name + f'_B{self.batch_size_train}'

        if self.no_weight:
            self.file_name = self.file_name + f'_no_weight'

        self.simvp_gsta_config = {
            "in_shape": [self.input_length, self.input_channel, (self.lon_range[1] - self.lon_range[0]) * 8,
                         (self.lat_range[1] - self.lat_range[0]) * 8],
            "C_out": self.output_channel,
            "hid_S": 16,
            "hid_T": 128,
            "N_S": 2,
            "N_T": 4,
            "model_type": 'gSTA',
            "drop": 0.3,
            "drop_path": 0.3,
            "spatio_kernel_enc": 3,
            "spatio_kernel_dec": 3,
            "mlp_ratio": 8.0,
        }

        self.scheduler_kwargs = {
            "max_lr": 0.0001,
            "total_steps": int(self.num_epochs),
            "pct_start": 0.05,
            "div_factor": 1e6,
            "final_div_factor": 1e8,
            "anneal_strategy": "linear",
            "cycle_momentum": True,
            "base_momentum": 0.85,
            "max_momentum": 0.95,
        }


args = get_args()

mypara = Mypara(args)


class Config:
    def __init__(self, mypara=None):
        self.in_shape = (mypara.input_length, mypara.input_channel, 160, 160)  # (时间步数, 通道数, 高度, 宽度)
        self.input_channel = mypara.input_channel
        self.output_channel = mypara.output_channel
        self.img_width = 160
        self.img_height = 160
        self.img_channel = 1
        self.input_length = mypara.input_length
        self.patch_size = 4
        self.filter_size = 3
        self.stride = 1
        self.layer_norm = True
        self.reverse_schedule = True
        self.total_length = 2 * mypara.input_length  # 预测总长度
        self.decouple_beta = 0.3  # 解耦损失权重
        self.need_mask = mypara.need_mask
        self.device = mypara.device
        self.r_sampling_step_1 = 20  # 100
        self.r_sampling_step_2 = 60  # 200
        self.r_exp_alpha = int((self.r_sampling_step_2 - self.r_sampling_step_1) / 2)

rnn2configs = Config(mypara)

if __name__ == '__main__':
    args = get_args()
    args.need_ssh = True
    mypara = Mypara(args)
    print("Mypara parameters loaded:")
    for k, v in vars(mypara).items():
        if not k.startswith("__"):
            print(f"{k}: {v}")

