import torch
from dask.array import nan_to_num
from torch import nn, device
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np
import os
# from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.predFormer_Modules import Attention, PreNorm, FeedForward
import math


class SwiGLU(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def init_weights(self):
        nn.init.ones_(self.fc1_g.bias)
        nn.init.normal_(self.fc1_g.weight, std=1e-6)

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GatedTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, SwiGLU(dim, mlp_dim, drop=dropout)),
                DropPath(drop_path) if drop_path > 0. else nn.Identity(),
                DropPath(drop_path) if drop_path > 0. else nn.Identity()
            ]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for attn, ff, drop_path1, drop_path2 in self.layers:
            x = x + drop_path1(attn(x))
            x = x + drop_path2(ff(x))
        return self.norm(x)


class PredFormerLayer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., attn_dropout=0., drop_path=0.1):
        super(PredFormerLayer, self).__init__()

        self.ts_temporal_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                        mlp_dim, dropout, attn_dropout, drop_path)
        self.ts_space_transformer = GatedTransformer(dim, depth, heads, dim_head,
                                                     mlp_dim, dropout, attn_dropout, drop_path)

    def forward(self, x):
        b, t, n, _ = x.shape
        x_ts, x_ori = x, x

        # ts-t branch
        x_ts = rearrange(x_ts, 'b t n d -> b n t d')
        x_ts = rearrange(x_ts, 'b n t d -> (b n) t d')
        x_ts = self.ts_temporal_transformer(x_ts)

        # ts-s branch
        x_ts = rearrange(x_ts, '(b n) t d -> b n t d', b=b)
        x_ts = rearrange(x_ts, 'b n t d -> b t n d')
        x_ts = rearrange(x_ts, 'b t n d -> (b t) n d')
        x_ts = self.ts_space_transformer(x_ts)

        # ts output branch
        x_ts = rearrange(x_ts, '(b t) n d -> b t n d', b=b)

        return x_ts


def sinusoidal_embedding(n_channels, dim):
    pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                            for p in range(n_channels)])
    pe[:, 0::2] = torch.sin(pe[:, 0::2])
    pe[:, 1::2] = torch.cos(pe[:, 1::2])
    return rearrange(pe, '... -> 1 ...')


class Mask_PredFormer_Model(nn.Module):
    def __init__(self, model_config, mask):
        super().__init__()

        # ----- Basic configs -----
        self.H = model_config['height']
        self.W = model_config['width']
        self.P = model_config['patch_size']
        self.Tin = model_config['input_length']
        self.Cin = model_config['input_channels']
        self.Cout = model_config['output_channels']

        self.num_ph = self.H // self.P
        self.num_pw = self.W // self.P
        self.num_p = self.num_ph * self.num_pw

        self.dim = model_config['dim']
        self.heads = model_config['heads']
        self.dim_head = self.dim // self.heads
        self.scale_dim = model_config['scale_dim']
        self.Ndepth = model_config['Ndepth']
        self.depth = model_config['depth']

        # ----- Patch dims -----
        self.patch_dim_in = self.Cin * self.P * self.P
        self.patch_dim_out = self.Cout * self.P * self.P

        # =============================================================
        # 1) ---- Precompute mask & valid patches ----
        # =============================================================
        mask = torch.from_numpy(mask)          # (H, W)
        mask = mask.unsqueeze(0).repeat(self.Cin, 1, 1)  # (Cin, H, W)

        # mask → patch flatten
        mask_patches = rearrange(
            mask, "c (h p1) (w p2) -> (h w) (p1 p2 c)",
            p1=self.P, p2=self.P
        )

        # valid patch (=至少一个像素非缺失)
        valid_patch = (~mask_patches.bool()).any(dim=-1)   # (num_p,)
        valid_idx = torch.where(valid_patch)[0]            # (num_valid,)

        self.num_valid = valid_idx.shape[0]

        # 存成 buffer，避免 forward 时移动设备
        self.register_buffer("valid_patch_idx", valid_idx)    # (num_valid,)

        # For gather input (T, num_valid, patch_dim_in)
        idx_in = valid_idx.view(1, 1, -1, 1).expand(1, self.Tin, -1, self.patch_dim_in)
        self.register_buffer("gather_idx_in", idx_in)

        # For scatter output
        idx_out = valid_idx.view(1, 1, -1, 1).expand(1, self.Tin, -1, self.patch_dim_out)
        self.register_buffer("scatter_idx_out", idx_out)

        # Blank restore buffer
        restore_shape = (1, self.Tin, self.num_p, self.patch_dim_out)
        self.register_buffer("restore_template", torch.zeros(restore_shape))

        # =============================================================
        # 2) ---- Model components ----
        # =============================================================
        self.to_patch_embedding = nn.Linear(self.patch_dim_in, self.dim)

        pos = sinusoidal_embedding(self.Tin * self.num_p, self.dim)
        pos = pos.view(1, self.Tin, self.num_p, self.dim)
        pos = pos[:, :, valid_idx, :]   # select valid patches only
        self.register_buffer("pos_embedding", pos)

        # PredFormer blocks
        self.blocks = nn.ModuleList([
            PredFormerLayer(
                self.dim, self.depth, self.heads, self.dim_head,
                self.dim * self.scale_dim,
                model_config['dropout'],
                model_config['attn_dropout'],
                model_config['drop_path']
            )
            for _ in range(self.Ndepth)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.patch_dim_out)
        )

    # =======================================================================
    # Forward
    # =======================================================================
    def forward(self, x):
        B, T, C, H, W = x.shape

        # ---- 1) Unfold into patches ----
        x = rearrange(
            x, "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
            p1=self.P, p2=self.P
        )

        # ---- 2) Keep only valid patches ----
        # gather_idx_in shape: (1, T, num_valid, patch_dim_in)
        x = torch.gather(
            x,
            dim=2,
            index=self.gather_idx_in.expand(B, -1, -1, -1)
        )

        # ---- 3) Patch embedding + Pos emb ----
        x = self.to_patch_embedding(x)
        x = x + self.pos_embedding

        # ---- 4) PredFormer ----
        for blk in self.blocks:
            x = blk(x)

        # ---- 5) MLP head → patch reconstruction ----
        x = self.mlp_head(x.reshape(-1, self.dim))
        x = x.view(B, T, self.num_valid, self.patch_dim_out)

        # ---- 6) Scatter back to full patch grid ----
        restore = self.restore_template.expand(B, -1, -1, -1).clone().to(dtype=x.dtype)
        restore.scatter_(2, self.scatter_idx_out.expand(B, -1, -1, -1), x)

        # ---- 7) Fold back to image ----
        restore = restore.view(
            B, T, self.num_ph, self.num_pw,
            self.Cout, self.P, self.P
        )

        img = restore.permute(0,1,4,2,5,3,6).reshape(B, T, self.Cout, H, W)
        return img


if __name__ == '__main__':
    from configs import parse_args
    from dataset import MvDataset

    model_config = {
        # image h w c
        'height': 288,
        'width': 288,
        'input_channels': 2,
        'output_channels': 2,
        # video length in and out
        'input_length': 1,
        'output_length': 1,
        # patch size
        'patch_size': 8,
        'dim': 64,
        'heads': 8,
        # dropout
        'dropout': 0.0,
        'attn_dropout': 0.0,
        'drop_path': 0.0,
        'scale_dim': 4,
        # depth
        'depth': 1,
        'Ndepth': 1
    }
    args = parse_args()
    mask = np.load(r'D:\mask.npy')  # (H,W) 1: invalid, 0: valid

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B, T, C, H, W = 4, 1, 2, 288, 288
    mask = torch.zeros((H, W))
    data = torch.randn(B, T, C, H, W)

    # 设置一些无效区域
    mask[0:8, 0:8] = 1  # 第一个patch完全无效
    mask[4:12, 4:12] = 1  # 中间某个patch完全无效
    mask[280:288, 280:288] = 1  # 最后一个patch完全无效
    # 其他部分随机设置一些无效点
    mask[35, 4] = 1  # 部分有效
    mask[56, 4:6] = 1  # 部分有效

    # 扩展mask到与data相同的形状 (B, T, C, H, W)
    mask_land = mask
    mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 先扩展为 (1, 1, 1, H, W)
    mask = mask.repeat(B, T, C, 1, 1)  # 再扩展为 (B, T, C, H, W)

    # 将无效区域设置为NaN
    data[mask == 1] = torch.nan
    print("原始输入x中的NaN数量:", torch.isnan(data).sum().item())

    x = torch.nan_to_num(data)
    x = x.to(device)

    y = data.to(device)
    mask_land= mask_land.bool().squeeze()

    model = Mask_PredFormer_Model(model_config, mask_land).to(device)
    output = model(x)
    print(output.shape)  # [B, T, C, H, W]
    print(y.shape)

    # 方法1：使用 torch.equal 直接比较
    nan_positions_equal = torch.equal(torch.isnan(y), torch.isnan(output))
    if nan_positions_equal:
        print("位置相同")
    else:
        print("位置不同")

    # 方法2：分别获取 NaN 位置并比较
    nan_y = torch.isnan(y)
    nan_output = torch.isnan(output)
    positions_match = (nan_y == nan_output).all()
    if positions_match:
        print("2位置相同")
    else:
        print("2位置不同")

    # 方法3：计算差异
    diff = torch.logical_xor(nan_y, nan_output)
    no_difference = not diff.any()  # 如果没有任何差异，则位置相同
    if no_difference:
        print("d位置相同")
    else:
        print("d位置不同")



