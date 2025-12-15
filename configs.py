import argparse
from pathlib import Path
import re
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')

    # Data
    base = Path('0.125deg')

    parser.add_argument('--datapath', type=tuple, default=
    (sorted(base.glob("*.nc"), key=lambda x: int(re.search(r'\d+', x.stem).group()))),
                        help='Paths to the dataset')
    parser.add_argument('--path_means', type=str, default=r"0.125deg_mean.npy",
                        help='path to ssh_mean, uo_mean,vo_std,et al.')
    parser.add_argument('--path_stds', type=str, default=r"0.125deg_std.npy",
                        help='path to ssh_std, uo_std,vo_std,et al.')

    parser.add_argument('--need_mask', action='store_true', default=False,
                        help='Whether to use mask')
    parser.add_argument('--path_land_mask', type=str, default=r"0.125deg_mask.npy",)

    parser.add_argument('--need_ssh', action='store_true', default=False,
                        help='Whether to include ssh data')

    parser.add_argument('--need_uv', action='store_true', default=False,
                        help='Whether to include uo, vo data')

    parser.add_argument('--need_wind', action='store_true', default=False,
                        help='Whether to include wind data')
    parser.add_argument('--wind_path', type=str,
                        default='era5_wind_1_12_deg_normalized/era5_wind_1_12_deg_all_years_normalized.nc',
                        help='Path to wind data (required if need_wind is True)')
    parser.add_argument('--wind_stats_path', type=str,
                        default='era5_wind_1_12_deg_normalized/wind_stats.nc',
                        help='Path to wind u10_mean,v10_std,et al., required if need_wind is True')

    # Time range
    parser.add_argument('--start_time_train', type=str, default= '1993-01-01', 
                        help='Start time for training data')
    parser.add_argument('--end_time_train', type=str, default= '2021-12-31', 
                        help='End time for training data')
    parser.add_argument('--start_time_val', type=str, default= '2022-01-01', 
                        help='Start time for validation data')
    parser.add_argument('--end_time_val', type=str, default= '2022-12-31', 
                        help='End time for validation data')
    parser.add_argument('--start_time_test', type=str, default= '2023-01-01', 
                        help='Start time for test data')
    parser.add_argument('--end_time_test', type=str, default= '2024-06-14', 
                        help='End time for test data')

    # Data shape
    parser.add_argument('--input_length', type=int, default= 10, 
                        help='Length of input sequence')
    parser.add_argument('--output_length', type=int, default= 10, 
                        help='Length of output sequence')
    parser.add_argument('--height', type=int, default=160,
                        help='Image height')
    parser.add_argument('--width', type=int, default=160,
                        help='Image width')
    parser.add_argument('--input_channels', type=int, default=0,
                        help='Number of input channels')
    parser.add_argument('--output_channels', type=int, default=0,
                        help='Number of output channels')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size')

    parser.add_argument('--gated', action='store_true', default=False,
                        help='apply gated conv or not in the models predrnn and simvp')

    # Train
    parser.add_argument('--model_name', type=str, default='predformer', )
    parser.add_argument('--device', type=str, default='cuda:0',)
    parser.add_argument('--batch_size_train', type=int, default=4)
    parser.add_argument('--batch_size_eval', type=int, default=4)
    parser.add_argument('--is_acc', action='store_false', help='Enable gradient accumulation')
    parser.add_argument('--acc_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--early_stopping', action='store_false', default=True,)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gradient_clip', action='store_true',default=False,)
    parser.add_argument('--gradient_clip_value', type=float, default=0.5)
    parser.add_argument('--gradient_clip_type', type=str, default='norm',
                        help='Gradient clip type:norm or value')

    parser.add_argument('--is_pinn', action='store_true')
    parser.add_argument('--pinn_lambda', type=float, default=0.)

    parser.add_argument('--model_savepath', type=str, default=r"D:/Data/sej/model_paras/")

    # 随机种子
    parser.add_argument('--SEED', type=int, default=42)

    return parser.parse_args()


def get_simvp_gsta_config(args):

    simvp_gsta_config = {
        "in_shape": [args.input_length, args.input_channels, args.height, args.width],
        "C_out": args.output_channels,
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
        "gated": args.gated
    }
    return simvp_gsta_config

def get_rnn_config(args):
    rnn_config = {
        'num_layers':4,
        'hidden_dim': 64,
        'in_shape': (args.input_length, args.input_channels, args.height, args.width),  # (时间步数, 通道数, 高度, 宽度)
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'img_width': args.width,
        'img_height': args.height,
        'input_length': args.input_length,
        'patch_size': args.patch_size,
        'filter_size': 3,
        'stride': 1,
        'layer_norm': True,
        'reverse_schedule': True,
        'total_length': 2 * args.input_length,  # 预测总长度
        'decouple_beta': 0.3,  # 解耦损失权重
        'need_mask': args.need_mask,
        'device': args.device,
        'r_sampling_step_1': 20,
        'r_sampling_step_2': 60,
        'r_exp_alpha': int((30 - 10) / 2),
        'gated': args.gated,
    }
    return rnn_config

def get_predformer_config(args):

    model_config = {
        'height': args.height,
        'width': args.width,
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'input_length': args.input_length,
        'output_length': args.output_length,
        'patch_size': args.patch_size,
        'dim': 256,
        'heads': 4,
        'dropout': 0.5,
        'attn_dropout': 0.5,
        'drop_path': 0.5,
        'scale_dim': 4,
        'depth': 2,
        'Ndepth': 3
    }
    return model_config

def get_my_config():
    args = parse_args()
    args.file_name = 'var'

    if args.need_ssh:
        args.input_channels += 1
        args.output_channels += 1
        args.file_name += '_ssh'

    if args.need_wind:
        args.input_channels += 2
        args.file_name += '_wind'

    if args.need_mask:
        args.input_channels += 1
        args.file_name += '_mask'

    args.model_name = args.model_name.lower()
    if args.model_name == 'simvp':
        model_config = get_simvp_gsta_config(args)
    elif args.model_name == 'predformer':
        model_config = get_predformer_config(args)
    elif args.model_name == 'predrnn':
        model_config = get_rnn_config(args)
    else:
        raise ValueError('Invalid model name')
    args.model_config = model_config
    return args

