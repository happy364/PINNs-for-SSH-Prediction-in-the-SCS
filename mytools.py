import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import random
import os
import math
from configs import get_my_config
config = get_my_config()
class ConfigObject:
    """将字典转换为对象的包装类"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
def convert_configs(configs):
    """检查configs是否为字典，如果是则转换为对象"""
    if isinstance(configs, dict):
        return ConfigObject(configs)
    return configs

def set_all_seeds(seed):
    """
    设置所有随机种子以确保结果可复现
    """
    # 系统库
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时

    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unpatchify_with_batch(patched_tensor, patch_size, original_channels):
    """
    对带 Batch 的 patchified tensor 进行还原，恢复为原始 (B, T, C, H, W) 格式。

    参数:
        patched_tensor: torch.Tensor，形状为 (B, T, C * p * p, H', W')
        patch_size: int，patch 大小 p
        original_channels: int，原始通道数 C

    返回:
        torch.Tensor，形状为 (B, T, C, H, W)
    """
    B, T, Cp2, H_, W_ = patched_tensor.shape
    p = patch_size
    C = original_channels

    assert Cp2 == C * p * p, f"通道数不匹配：{Cp2} != {C} * {p} * {p}"

    # 恢复 patch 通道维度为 patch 格式
    x = patched_tensor.reshape(B, T, C, p, p, H_, W_)

    # 交换维度：将 patch 中像素恢复到空间位置
    x = x.permute(0, 1, 2, 5, 3, 6, 4)  # (B, T, C, H', p, W', p)

    # 合并 patch 像素还原空间维度
    x = x.reshape(B, T, C, H_ * p, W_ * p)

    return x

def masked_pearson_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算 pred 和 target 之间的整体 Pearson 相关系数，忽略 target 中为 NaN 的位置。

    参数:
        pred (torch.Tensor): 预测值张量，形状为 [M, N]
        target (torch.Tensor): 目标张量，形状为 [M, N]

    返回:
        torch.Tensor: 一个标量，表示整体相关系数
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")

    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # 掩膜：忽略 target 中为 NaN 的位置
    mask = ~torch.isnan(target_flat)
    pred_masked = pred_flat[mask]
    target_masked = target_flat[mask]

    if pred_masked.numel() == 0:
        return torch.tensor(float('nan'))  # 如果有效点为 0，则返回 NaN

    # 计算 Pearson 相关
    pred_mean = pred_masked.mean()
    target_mean = target_masked.mean()

    pred_centered = pred_masked - pred_mean
    target_centered = target_masked - target_mean

    numerator = torch.sum(pred_centered * target_centered)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))

    epsilon = 1e-8  # 防止除以0
    corr = numerator / (denominator + epsilon)
    return corr


class MSELossIgnoreNaN(nn.Module):
    def __init__(self):
        super(MSELossIgnoreNaN, self).__init__()
        self.mse_func = nn.MSELoss(reduction="sum")

    def forward(self, pred, target):

        valid_mask = ~(torch.isnan(target) | torch.isinf(target))
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(valid_mask, target, pred)
        loss = self.mse_func(pred, target) / valid_count

        return loss

class MSELossIgnoreNaNv2(nn.Module):
    def __init__(self, mask_valid, model_configs=None,config=config,  patched=False):
        """
        :param mask_valid: valid: True, invalid: False
        :param rnn2configs: model config
        :param patched: if the pred and target are patched
        """
        super().__init__()
        self.mse_func = nn.MSELoss(reduction="sum")
        self.mask_valid = mask_valid
        H, W = self.mask_valid.shape
        C = config.output_channels
        self.mask_valid = torch.from_numpy(self.mask_valid[None,None,None,:,:]).to(device=config.device)

        if patched:
            p = model_configs.patch_size
            self.mask_valid = self.mask_valid.expand(1,1,C,H,W)
            print(f"after 1: {self.mask_valid.shape}")
            self.mask_valid = self.mask_valid.reshape(C, H // p, p, W // p, p)
            print(f"after 2: {self.mask_valid.shape}")
            self.mask_valid = self.mask_valid.permute(0,2,4,1,3).reshape(1,1,C * p * p, H // p, W // p)
            print(f"after 3: {self.mask_valid.shape}")

    def forward(self, pred, target):
        mask_valid = self.mask_valid.expand_as(pred)
        valid_count = mask_valid.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(mask_valid, target, pred)
        loss = self.mse_func(pred, target) / valid_count

        return loss


def compute_gradients_sobel(sla, lon, lat, R_E=6371000.0, is_real_gradient=True):
    """
    使用Sobel算子计算梯度，输入sla尺寸为 [B, T, C, H, W]，
    利用二维经纬度数据（lon, lat均为一维提取自二维数据）计算每行dx和全局dy。
    返回:
      grad_x_phys, grad_y_phys: 分别为沿x和y方向的物理梯度, 尺寸均为 [B, T, C, H, W]
    """
    if lon.ndim == 2 and np.all(lon == lon[0, :][None, :]) and np.all(lat == lat[:, 0][:, None]):
        lon = lon[0,:]
        lat = lat[:,0]
    B, T, C, H, W = sla.shape
    # 合并B和T维度进行卷积计算
    x = sla.reshape(B * T, C, H, W)  # [B*T, C, H, W]

    # 定义Sobel卷积核（Sobel算子）
    kernel_sobel_x = torch.tensor([[-1, 0, 1],
                                   [-2, 0, 2],
                                   [-1, 0, 1]], dtype=x.dtype, device=x.device) / 8.0
    kernel_sobel_y = torch.tensor([[-1, -2, -1],
                                   [0, 0, 0],
                                   [1, 2, 1]], dtype=x.dtype, device=x.device) / 8.0
    # 为每个通道创建独立的卷积核，使用grouped convolution实现通道分离
    kernel_sobel_x = kernel_sobel_x.view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    kernel_sobel_y = kernel_sobel_y.view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # 使用replicate模式的padding保证尺寸不变（对3×3核，padding=1）
    x_padded = F.pad(x, pad=(1, 1, 1, 1), mode='replicate')
    # 使用分组卷积，每个通道独立计算
    grad_x = F.conv2d(x_padded, kernel_sobel_x, groups=C)
    grad_y = F.conv2d(x_padded, kernel_sobel_y, groups=C)
    if is_real_gradient:
        # 经纬度差
        # 假设lon, lat为从二维经纬度数据中提取的一维数组
        delta_lat = lat[1] - lat[0]  # 单位：度
        delta_lon = lon[1] - lon[0]  # 单位：度
        dy = delta_lat * (np.pi / 180) * R_E  # m
        lat_rad = np.deg2rad(lat)  # [H]
        dx_per_row = delta_lon * (np.pi / 180) * R_E * np.cos(lat_rad)  # [H] todo:改成常数是否会变差？
        dx_tensor = torch.tensor(dx_per_row, dtype=x.dtype, device=x.device).view(1, 1, H, 1)

        grad_x_phys = grad_x / dx_tensor
        grad_y_phys = grad_y / dy

        # 恢复原始尺寸 [B, T, C, H, W]
        grad_x_phys = grad_x_phys.view(B, T, C, H, W)
        grad_y_phys = grad_y_phys.view(B, T, C, H, W)

        return grad_x_phys, grad_y_phys
    else:
        return grad_x, grad_y



def compute_f_and_sigmoid_weight(lat,k=2,phi0=5, if_solid_f=False):
    """
    基于纬度计算科氏参数 f 和 sigmoid 型的 f_weight

    参数:
        lat: numpy array, shape [H, W]，网格纬度

    返回:
        f: torch.Tensor, shape [1,1,1,H,W]，科氏参数
        f_weight: torch.Tensor, shape [1,1,1,H,W]，sigmoid 权重
    """
    Omega = 7.2921e-5  # 地球自转角速度
    lat_t = torch.from_numpy(lat).float()  # [H, W]
    phi = torch.abs(lat_t)

    if if_solid_f:
        f = 2 * Omega * torch.sin(torch.deg2rad(torch.mean(lat_t)))  # [H, W]
        f_weight = 1.0
        return f, f_weight

    # 空间变化的科氏参数
    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))  # [H, W]
    f = f[None, None, None, :, :]  # [1,1,1,H,W]

    f_weight = 1 / (1.0 + torch.exp(-k * (phi - phi0)))  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight

def compute_f_and_gaussian_weight(lat,theta=2.2):
    """

    参数:
        lat: numpy array, shape [H, W]，网格纬度

    返回:
        f: torch.Tensor, shape [1,1,1,H,W]，科氏参数
        f_weight: torch.Tensor, shape [1,1,1,H,W]，sigmoid 权重
    """
    Omega = 7.2921e-5  # 地球自转角速度
    lat_t = torch.from_numpy(lat).float()  # [H, W]
    phi = torch.abs(lat_t)

    # 空间变化的科氏参数
    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))  # [H, W]
    f = f[None, None, None, :, :]  # [1,1,1,H,W]

    f_weight = 1-torch.exp(-(phi/theta)**2)  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight

def compute_geostrophic_current(pred,  lon, lat, if_solid_f=False):
    """
    基于空间 f 计算地转流速度（物理单位 m/s）。
    """
    g = 9.81  # 重力加速度

    f, f_weight = compute_f_and_sigmoid_weight(lat, if_solid_f)
    f = f.to(pred.device)

    grad_x, grad_y = compute_gradients_sobel(pred, lon, lat, R_E=6.371e6)

    # 地转流分量
    u_geo = - (g / f) * grad_y
    v_geo = (g / f) * grad_x

    return u_geo, v_geo, f_weight

class EvalModel:
    def __init__(self, model, config):
        self.mymodel = model
        self.device = config.device
        self.loss_var = MSELossIgnoreNaN()

    def test_model(self, dataloader):
        self.mymodel.eval()
        mse = []
        num_samples = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                with autocast('cuda'):
                    out_var = self.mymodel(
                        inputs.float().to(self.device),
                    )
                    B = out_var.shape[0]
                    num_samples += B
                    mse.append(self.loss_var(out_var, targets.float().to(
                        self.device)).item() * B)
        return torch.sqrt(torch.tensor(mse).sum() / num_samples)

class EvalModel_RNN:
    def __init__(self, model, loss, config, rnn2configs):
        self.mymodel = model
        self.device = config.device
        self.mypara = config
        self.input_length = rnn2configs["input_length"]
        self.patch_size = rnn2configs["patch_size"]
        self.loss_var = loss

    def test_model(self, dataloader, mask_true):
        self.mymodel.eval()
        print("begin testing......")
        mse = 0
        num_samples = 0
        with torch.no_grad():
            for input_var in dataloader:
                # print("testing...\n")
                B = input_var.shape[0]
                if B != self.mypara.batch_size_train:
                    mask_true_batch = mask_true[:B]
                else:
                    mask_true_batch = mask_true
                num_samples += B
                with autocast('cuda'):
                    out_var, loss = self.mymodel(input_var.to(self.device), mask_true_batch)
                    mse += loss.item() * input_var.shape[0]
        return math.sqrt(mse / num_samples)

    def test_mask_steps(self, dataloader,mask_true, is_persistence=False):
        self.mymodel.eval()
        mse_list = []
        corr_list = []
        num_samples = 0
        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                B = inputs.shape[0]
                if B != self.mypara.batch_size_train:
                    mask_true_batch = mask_true[:B]
                else:
                    mask_true_batch = mask_true
                inputs = inputs.float().to(self.device)
                pred_y,loss = self.mymodel(inputs,mask_true_batch)

                targets = unpatchify_with_batch(inputs, self.patch_size, self.mypara.input_channel)[:,
                         self.input_length:, 0:self.mypara.output_channel].to(self.device)
                pred_y = unpatchify_with_batch(pred_y, self.patch_size, self.mypara.output_channel)

                B, T, C, H, W = targets.shape

                if is_persistence:
                    pred_y = inputs[:, -1].unsqueeze(1).expand(-1, T, -1, -1, -1)

                num_samples += B
                # print(f"predy : {pred_y.shape}\n target: {targets.shape}")

                for t in range(T):
                    mse = self.loss_var(pred_y[:, t:t+1], targets[:, t:t+1])
                    corr = 0
                    for j in range(B):
                        corr += masked_pearson_corrcoef(pred_y[j, t, 0], targets[j, t, 0])

                    if len(mse_list) <= t:
                        mse_list.append(0)
                        corr_list.append(0)
                    mse_list[t] += mse * B
                    corr_list[t] += corr

        rmse_avg = [torch.sqrt(mse_step / num_samples) for mse_step in mse_list]
        corr_avg = [corr_step / num_samples for corr_step in corr_list]
        return rmse_avg, corr_avg  # shape = [T]

def reverse_schedule_sampling(itr,  total_length, input_length, img_shape, args, reverse=True, mode='train'):
    """
    PyTorch版本 - 支持训练和测试的reverse schedule sampling

    Args:
        itr: 当前迭代步数
        total_length: 总序列长度
        input_length: 输入序列长度
        img_shape: 图像形状 (C, H, W)
        args: 参数对象
        mode: 模式 'train' 或 'test'
        :param reverse:
    """
    C, H, W = img_shape

    # 测试模式：完全使用真实帧（不进行mask）
    if mode == 'test':
        # 创建全为True的mask，表示所有帧都使用真实值
        real_input_flag = torch.ones(( total_length - 2, C, H, W),device=args.device)
        real_input_flag[input_length-1:] = 0

        return real_input_flag

    # 训练模式：根据调度策略生成mask
    elif mode == 'train':
        # 1. 计算调度概率
        r_eta_st, eta_st = 0.5, 0.5
        if reverse:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")

            elif itr < args.r_sampling_step_2:

                r_eta =r_eta_st + (1-r_eta_st)*(1.0 -  math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha))
                eta = eta_st * (1 - ((itr - args.r_sampling_step_1) / (args.r_sampling_step_2 - args.r_sampling_step_1)))
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
        else:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")
            elif itr < args.r_sampling_step_2:
                r_eta = 1.0
                eta = eta_st - eta_st*(1 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (itr - args.r_sampling_step_1)
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
                print(f"r_eta: {r_eta}, eta: {eta}")

        # 2. 使用torch.bernoulli生成mask
        # 输入序列内部的mask (t=1 到 input_length-1)
        r_mask = torch.bernoulli(
            torch.full(( input_length - 1, C, H, W), r_eta, device=args.device)
        )

        # 预测序列的mask (t=input_length 到 total_length-1)
        pred_mask = torch.bernoulli(
            torch.full(( total_length - input_length -1, C, H, W), eta, device=args.device)
        )

        # 3. 合并mask
        real_input_flag = torch.cat([r_mask, pred_mask], dim=0)

        return real_input_flag

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'test'.")