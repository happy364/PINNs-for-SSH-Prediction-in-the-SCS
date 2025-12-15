import numpy as np
from scipy.ndimage import distance_transform_edt
from mytools import  MSELossIgnoreNaNv2, set_all_seeds
from configs import parse_args, get_my_config
from dataset import MvDataset
import torch
from torch.utils.data import DataLoader
from tools.trainers import Trainer, TrainerMask
from models import PredFormer_Model, Mask_PredFormer_Model, SimVP_Model, RNN


args = get_my_config()
set_all_seeds(args.SEED)

dataset = MvDataset(args, mode='test')
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
# ----------------------------
# 输入：
# pred, gt: (H, W) 数组
# land_mask: (H, W)，1=陆，0=海
# lat: (H,) 每行对应的纬度（单位：度）
# ----------------------------

land_mask = np.load('/data/hjj/MVPfore/Datas/processed_data/mask.npy')
lat = dataset.lat          # shape (H,)

buffer_km = 20     # 想要海岸多少 km 范围内

# ===== Step 1: 计算海/陆距离（像素单位） =====
dist_to_land = distance_transform_edt(1 - land_mask)   # 对海像素表示距离陆地几像素

H, W = land_mask.shape

# ===== Step 2: 经纬度分辨率换算（每行单独 km/像素）=====
# 1°纬度大约是 111.0 km
# 经度方向距离随纬度乘 cos(lat)
pixel_km_per_row = 111.0 * 0.083 * np.cos(np.deg2rad(lat))   # shape (H,)

# 为每一行计算 buffer 需要的像素数
buffer_pixels_per_row = buffer_km / pixel_km_per_row          # shape (H,)

# 广播成 (H, W)
buffer_pixels = buffer_pixels_per_row[:, None]                # shape (H,W)

# ===== Step 3: 判断是否在海岸 buffer 内（仅保留海侧像素）=====
coastal_mask = (dist_to_land <= buffer_pixels) & (land_mask == 0)
print('coastal_mask shape:', coastal_mask.shape)
print('coastal_mask sum:', np.sum(coastal_mask))

model = SimVP_Model(**args.model_config).to(args.device)
model.load_state_dict(torch.load('/data/hjj/MVPfore/model_paras/SimVP_Model_seed42/var_ssh_mask_20251207_0500/model_paras.pkl'))

mse_ig_nan = MSELossIgnoreNaNv2(coastal_mask)

trainer = Trainer(model,mse_ig_nan, '', args, None, None,None, mode='test' )

rmse = trainer.test_model(dataloader)
print(f'RMSE: {rmse}')
