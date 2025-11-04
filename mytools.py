import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from pathlib import Path
import re
from configs import rnn2configs, mypara
import math
import random
import os


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


def extract_model_info(parent_dir):
    """
    从父目录中提取所有 model_paras.pkl 文件的路径及其对应的简化文件夹名
    """
    parent_path = Path(parent_dir)
    pattern = re.compile(r'^(.*?)(?:_\d{8}_\d{4})?$')
    results = []

    for folder in parent_path.iterdir():
        if folder.is_dir():
            pkl_file = folder / "model_paras.pkl"
            if pkl_file.exists():
                name = folder.name
                match = pattern.match(name)
                simple_name = match.group(1) if match else name
                results.append((str(pkl_file), simple_name))
    return results


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
    return corr.item()


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


class MSELossIgnoreNaN_RNN(nn.Module):
    def __init__(self, patched=True):
        super().__init__()
        self.mse_func = nn.MSELoss(reduction="sum")
        self.mask_valid = ~np.load(mypara.path_mask)
        H, W = self.mask_valid.shape
        C = rnn2configs.output_channel
        T = rnn2configs.input_length
        p = rnn2configs.patch_size
        print(f"raw: {self.mask_valid.shape}")
        self.mask_valid = np.tile(self.mask_valid.reshape(1, 1, H, W), (1, C, 1, 1))
        if patched:
            print(f"after 1: {self.mask_valid.shape}")
            self.mask_valid = self.mask_valid.reshape(1, C, H // p, p, W // p, p)
            print(f"after 2: {self.mask_valid.shape}")
            self.mask_valid = self.mask_valid.transpose(0, 1, 3, 5, 2, 4).reshape(1, 1, C * p * p, H // p, W // p)
            print(f"after 3: {self.mask_valid.shape}")

    def forward(self, pred, target):
        B, T, C, H, W = pred.shape
        mask_valid = torch.from_numpy(np.tile(self.mask_valid, (B, T, 1, 1, 1))).to(rnn2configs.device)
        valid_count = mask_valid.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(mask_valid, target, pred)
        loss = self.mse_func(pred, target) / valid_count
        return loss


class MAELossIgnoreNaN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae_func = nn.L1Loss(reduction="sum")

    def forward(self, pred, target):
        valid_mask = ~(torch.isnan(target) | torch.isinf(target))
        valid_count = valid_mask.sum()
        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        target = torch.where(valid_mask, target, pred)
        loss = self.mae_func(pred, target) / valid_count
        return loss


class EvalModel:
    def __init__(self, model, config):
        self.mymodel = model
        self.device = config.device
        self.loss_var = MSELossIgnoreNaN()
        self.mask_valid = torch.from_numpy(np.load(config.path_mask)).float()
        self.need_mask = config.need_mask

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
                    mse.append(self.loss_var(out_var, targets[:, :, :1, :, :].float().to(
                        self.device)).item() * B)
        return torch.sqrt(torch.tensor(mse).sum() / num_samples)


class EvalModelMask(EvalModel):
    def __init__(self, model, config, mask):
        super().__init__(model, config)
        self.mask = mask
        self.need_mask = config.need_mask
        self.loss_var_mae = MAELossIgnoreNaN()

    def test_mask(self, dataloader):
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
                    mask_l = self.mask.repeat(out_var.shape[0], out_var.shape[1], out_var.shape[2], 1, 1)
                    mse.append(self.loss_var(out_var[mask_l], targets[:, :, :1, :, :][mask_l].float().to(
                        self.device)).item() * B)  #
        return torch.sqrt(torch.tensor(mse).sum() / num_samples)

    def test_mask_v2(self, dataloader):
        self.mymodel.eval()
        mse = 0
        mae = 0
        corr = 0

        num_samples = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                with autocast('cuda'):
                    inputs = inputs.float().to(self.device)

                    out_var = self.mymodel(
                        inputs,
                    )
                    B = out_var.shape[0]
                    num_samples += B
                    mask_l = self.mask.repeat(out_var.shape[0], out_var.shape[1], out_var.shape[2], 1, 1)
                    mse += self.loss_var(out_var[mask_l], targets[:, :, :1, :, :][mask_l].float().to(
                        self.device)).item() * B  #
                    mae += self.loss_var_mae(out_var[mask_l], targets[:, :, :1, :, :][mask_l].float().to(
                        self.device)).item() * B

                    for j in range(B):
                        corr += masked_pearson_corrcoef(out_var[j][mask_l[0]],
                                                        targets[j, :, :1, ...][mask_l[0]].float().to(
                                                            self.device))

        return np.sqrt(mse / num_samples), mae / num_samples, corr / num_samples

    def test_mask_steps(self, dataloader, is_persistence=False):
        self.mymodel.eval()
        mse_list = []
        corr_list = []
        num_samples = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.float().to(self.device)
                targets = targets.to(self.device)
                B, T, C, H, W = targets.shape

                pred_y = self.mymodel(inputs)
                if is_persistence:
                    pred_y = inputs[:, -1].unsqueeze(1).expand(-1, T, -1, -1, -1)

                mask = self.mask.repeat(B, C, 1, 1)  # broadcast shape
                num_samples += B

                for t in range(T):
                    mse = self.loss_var(pred_y[:, t][mask], targets[:, t, :1, ...][mask])
                    corr = 0
                    for j in range(B):
                        corr += masked_pearson_corrcoef(pred_y[j, t][mask[0]], targets[j, t, :1, ...][mask[0]])

                    if len(mse_list) <= t:
                        mse_list.append(0)
                        corr_list.append(0)
                    mse_list[t] += mse * B
                    corr_list[t] += corr

        rmse_avg = [torch.sqrt(mse_step / num_samples) for mse_step in mse_list]
        corr_avg = [corr_step / num_samples for corr_step in corr_list]
        return rmse_avg, corr_avg  # shape = [T]


class EvalModel_RNN:
    def __init__(self, model, mypara, rnn2configs):
        self.mymodel = model
        self.device = mypara.device
        self.mypara = mypara
        self.configs = rnn2configs
        self.loss_var = MSELossIgnoreNaN_RNN(patched=False)

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

    def test_mask_steps(self, dataloader, mask_true, is_persistence=False):
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
                pred_y, loss = self.mymodel(inputs, mask_true_batch)

                targets = unpatchify_with_batch(inputs, self.configs.patch_size, self.mypara.input_channel)[:,
                          self.configs.input_length:, 0:self.mypara.output_channel].to(self.device)
                pred_y = unpatchify_with_batch(pred_y, self.configs.patch_size, self.mypara.output_channel)

                B, T, C, H, W = targets.shape

                if is_persistence:
                    pred_y = inputs[:, -1].unsqueeze(1).expand(-1, T, -1, -1, -1)

                num_samples += B
                # print(f"predy : {pred_y.shape}\n target: {targets.shape}")

                for t in range(T):
                    mse = self.loss_var(pred_y[:, t:t + 1], targets[:, t:t + 1])
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


def compute_gradients_sobel(sla, lon, lat, R_E=6371000.0):
    """
    使用Sobel算子计算标准化后的SLA梯度，输入sla尺寸为 [B, T, C, H, W]，
    其中数据为归一化后数据，计算梯度时乘回sla_std以还原物理单位。
    利用二维经纬度数据（lon, lat均为一维提取自二维数据）计算每行dx和全局dy。

    返回:
      grad_x_phys, grad_y_phys: 分别为沿x和y方向的物理梯度, 尺寸均为 [B, T, C, H, W]
    """
    if lon.ndim == 2 and np.all(lon == lon[0, :][None, :]) and np.all(lat == lat[:, 0][:, None]):
        lon = lon[0, :]
        lat = lat[:, 0]
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

    # 经纬度差
    # 假设lon, lat为从二维经纬度数据中提取的一维数组
    delta_lat = lat[1] - lat[0]  # 单位：度
    delta_lon = lon[1] - lon[0]  # 单位：度
    dy = delta_lat * (np.pi / 180) * R_E  # m
    lat_rad = np.deg2rad(lat)  # [H]
    dx_per_row = delta_lon * (np.pi / 180) * R_E * np.cos(lat_rad)  # [H]
    dx_tensor = torch.tensor(dx_per_row, dtype=x.dtype, device=x.device).view(1, 1, H, 1)

    grad_x_phys = grad_x / dx_tensor
    grad_y_phys = grad_y / dy

    # 恢复原始尺寸 [B, T, C, H, W]
    grad_x_phys = grad_x_phys.view(B, T, C, H, W)
    grad_y_phys = grad_y_phys.view(B, T, C, H, W)

    return grad_x_phys, grad_y_phys


def compute_f_and_sigmoid_weight(lat, k=2, phi0=5):
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

    # 空间变化的科氏参数
    f = 2 * Omega * torch.sin(torch.deg2rad(lat_t))  # [H, W]
    f = f[None, None, None, :, :]  # [1,1,1,H,W]

    # Sigmoid 权重：在 φ=7° 处快速过渡
    f_weight = 1 / (1.0 + torch.exp(-k * (phi - phi0)))  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight


def compute_f_and_gaussian_weight(lat, theta=2.2):
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

    f_weight = 1 - torch.exp(-(phi / theta) ** 2)  # [H, W]
    f_weight = f_weight[None, None, None, :, :]  # [1,1,1,H,W]

    return f, f_weight


def compute_geostrophic_current(pred, lon, lat):
    """
    基于空间 f 计算地转流速度（物理单位 m/s）。
    """
    g = 9.81  # 重力加速度
    # 获得 f 和权重
    f, f_weight = compute_f_and_sigmoid_weight(lat)
    f = f.to(pred.device)

    grad_x, grad_y = compute_gradients_sobel(pred, lon, lat, R_E=6.371e6)

    # 地转流分量
    u_geo = - (g / f) * grad_y
    v_geo = (g / f) * grad_x

    return u_geo, v_geo, f_weight


def compute_geostrophic_current_solid_f(var_pred, sla_std, lon, lat, center_lat=13.0):
    """
    计算并标准化地转流速度（u, v 分量）

    参数:
        var_pred: 预测值张量，形状为 [B, T, C, H, W]，其中 C 的第0维为 SLA
        means: 包含 sla, u, v 的均值 (sla_mean, u_mean, v_mean)
        stds: 包含 sla, u, v 的标准差 (sla_std, u_std, v_std)
        lon, lat: 网格经纬度
        center_lat: 中心纬度（默认 13.0）

    返回:
        u_geo_norm, v_geo_norm: 标准化后的地转流分量
    """
    g = 9.81  # 重力加速度 (m/s²)
    Omega = 7.2921e-5  # 地球自转角速度 (s^-1)
    # center_lat = np.nanmean(lat)
    f = 2 * Omega * np.sin(np.deg2rad(center_lat))  # 科氏参数

    # 提取 SLA 预测值
    sla_pred = var_pred[:, :, :1, :, :]  # 形状 [B, T, 1, H, W]

    # 计算梯度（单位 m/m）
    grad_x_phys, grad_y_phys = compute_gradients_sobel(sla_pred, lon, lat, R_E=6371000.0)

    # 计算地转流（物理单位 m/s）
    u_geo_phys = - (g / f) * grad_y_phys * sla_std
    v_geo_phys = (g / f) * grad_x_phys * sla_std

    return u_geo_phys, v_geo_phys


def fill_nan_extrapolate(da, method='linear'):
    """
    对于xarray.DataArray数据，先使用前向填充和后向填充填补外围NaN，
    然后对内部NaN进行指定方法的插值，适用于大量连续NaN且位于有效数据外围的情况。

    参数：
        da: xarray.DataArray，输入数据，要求为二维
        method: 插值方法，如'linear'（线性插值）、'cubic'（三次插值）等

    返回：
        填充后的DataArray
    """
    assert da.ndim == 2, "输入数据必须为二维"

    # 假设前两维分别为y和x
    dim_y, dim_x = da.dims[:2]

    # 先填充边界：前向填充和后向填充确保外围NaN被填补
    da_filled = da.ffill(dim=dim_y).bfill(dim=dim_y).ffill(dim=dim_x).bfill(dim=dim_x)

    # 对内部缺失值做平滑插值
    da_filled = da_filled.interpolate_na(dim=dim_y, method=method).interpolate_na(dim=dim_x, method=method)

    return da_filled


def _ffill_1d(x):
    """对 1D 向量做前向填充（forward fill）"""
    # x: 1D Tensor
    for i in range(1, x.size(0)):
        mask = torch.isnan(x[i])
        # 如果当前位置是 nan，就用前一个位置的值
        x[i] = torch.where(mask, x[i - 1], x[i])
    return x


def _bfill_1d(x):
    """对 1D 向量做后向填充（backward fill）"""
    for i in range(x.size(0) - 2, -1, -1):
        mask = torch.isnan(x[i])
        x[i] = torch.where(mask, x[i + 1], x[i])
    return x


def _linear_interp_1d(x):
    """
    对 1D 向量中的内部 NaN 做线性插值。
    要求两端已被填充（即最左、最右不含 NaN）。
    """
    isnan = torch.isnan(x)
    if not isnan.any():
        return x

    idx = torch.arange(x.size(0), device=x.device, dtype=torch.long)
    valid_idx = idx[~isnan]
    valid_val = x[~isnan]

    # 至少需要两个已知点才能插值
    if valid_idx.numel() < 2:
        return x

    # 遍历每一段区间做线性插值
    for k in range(valid_idx.numel() - 1):
        i, j = valid_idx[k].item(), valid_idx[k + 1].item()
        vi, vj = valid_val[k].item(), valid_val[k + 1].item()
        if j > i + 1:
            # 插值坐标
            t = torch.arange(1, j - i, device=x.device, dtype=x.dtype) / (j - i)
            x[i + 1:j] = vi + (vj - vi) * t

    return x


def fill_nan_extrapolate_torch(tensor: torch.Tensor, method: str = 'linear') -> torch.Tensor:
    """
    对最后两维进行 NaN 边界填充 + 插值。

    支持输入维度 3 或 4：
      - 3D: (C, H, W) 或 (N, H, W)
      - 4D: (N, C, H, W)

    填充流程：
      1. 沿宽度和高度做前向 + 后向填充，确保边缘 NaN 被填掉；
      2. 沿两轴分别做线性插值，填补内部 NaN。

    参数:
      tensor (torch.Tensor): 输入，有 NaN 的地方将被填充。
      method (str): 目前仅支持 'linear'。

    返回:
      torch.Tensor: 同形状，NaN 被填满。
    """
    if method.lower() != 'linear':
        raise ValueError("目前仅支持 'linear' 插值方式")

    if tensor.ndim not in (3, 4):
        raise ValueError("输入张量必须是 3D 或 4D")

    # 将前面任意维度先展平到 batch 维
    *lead, H, W = tensor.shape
    batch = int(torch.prod(torch.tensor(lead))) if lead else 1
    x = tensor.reshape(batch, H, W).clone()

    # 对每个 [H×W] 做处理
    for b in range(batch):
        mat = x[b]

        # —— 1. 边界填充 ——#
        # 横向（宽度方向）
        for i in range(H):
            row = mat[i, :]
            row = _ffill_1d(row)
            row = _bfill_1d(row)
            mat[i, :] = row
        # 纵向（高度方向）
        for j in range(W):
            col = mat[:, j]
            col = _ffill_1d(col)
            col = _bfill_1d(col)
            mat[:, j] = col

        # —— 2. 线性插值 ——#
        # 横向
        for i in range(H):
            mat[i, :] = _linear_interp_1d(mat[i, :])
        # 纵向
        for j in range(W):
            mat[:, j] = _linear_interp_1d(mat[:, j])

        x[b] = mat

    # 恢复原始形状
    return x.reshape(*lead, H, W)



def reverse_schedule_sampling(itr, batch_size, total_length, input_length, img_shape, args, reverse=True, mode='train'):
    """
    PyTorch版本 - 支持训练和测试的reverse schedule sampling

    Args:
        itr: 当前迭代步数
        batch_size: 批次大小
        total_length: 总序列长度
        input_length: 输入序列长度
        img_shape: 图像形状 (C, H, W)
        args: 参数对象
        mode: 模式 'train' 或 'test'
    """
    C, H, W = img_shape

    # 测试模式：完全使用真实帧（不进行mask）
    if mode == 'test':
        # 创建全为True的mask，表示所有帧都使用真实值
        real_input_flag = torch.ones((batch_size, total_length - 2, C, H, W),
                                     device=args.device)
        real_input_flag[:, input_length - 1:] = 0

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

                r_eta = r_eta_st + (1 - r_eta_st) * (
                            1.0 - math.exp(-float(itr - args.r_sampling_step_1) / args.r_exp_alpha))
                eta = eta_st * (
                            1 - ((itr - args.r_sampling_step_1) / (args.r_sampling_step_2 - args.r_sampling_step_1)))
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
        else:
            if itr < args.r_sampling_step_1:
                r_eta, eta = r_eta_st, eta_st
                print(f"r_eta: {r_eta}, eta: {eta}")
            elif itr < args.r_sampling_step_2:
                r_eta = 1.0
                eta = eta_st - eta_st * (1 / (args.r_sampling_step_2 - args.r_sampling_step_1)) * (
                            itr - args.r_sampling_step_1)
                print(f"r_eta: {r_eta}, eta: {eta}")

            else:
                r_eta, eta = 1.0, 0.0
                print(f"r_eta: {r_eta}, eta: {eta}")

        # 2. 使用torch.bernoulli生成mask
        # 输入序列内部的mask (t=1 到 input_length-1)
        r_mask = torch.bernoulli(
            torch.full((batch_size, input_length - 1, C, H, W), r_eta, device=args.device)
        )

        # 预测序列的mask (t=input_length 到 total_length-1)
        pred_mask = torch.bernoulli(
            torch.full((batch_size, total_length - input_length - 1, C, H, W), eta, device=args.device)
        )

        # 3. 合并mask
        real_input_flag = torch.cat([r_mask, pred_mask], dim=1)

        return real_input_flag

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'train' or 'test'.")