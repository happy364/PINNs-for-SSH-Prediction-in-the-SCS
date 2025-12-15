
from tools.base_method import BaseMethod
import torch
import numpy as np
from torch.amp import autocast
from mytools import MSELossIgnoreNaNv2, reverse_schedule_sampling,convert_configs, MSELossIgnoreNaN, compute_geostrophic_current, unpatchify_with_batch
import torch.nn as nn

class Trainer(BaseMethod):
    def __init__(self,model, loss_func, loss_name, config, log_dir, optimizer, scheduler, mode='train'):
        super().__init__(model, config, loss_name, log_dir, optimizer, scheduler, mode=mode)
        self.mask_land = np.load(config.path_land_mask)
        self.loss_ignore_nan = MSELossIgnoreNaNv2(~self.mask_land, self.model_config, patched= False)
        self.loss = self.loss_ignore_nan
        self.lon, self.lat = lon, lat
        self.stds = stds
        self.config = config

    def _compute_loss(self, train_data, step, mask=None, test=False):
        inputs, targets = train_data

        B = inputs.shape[0]
        with autocast('cuda'):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            preds = self.model(inputs)
            loss = self.loss(preds, targets) if not test else self.loss_ignore_nan(preds, targets)
            if not test and self.config.is_pinn:
                pinn_loss = self._compute_pinn_loss_sigmoid_weight(preds, targets)
                loss = loss + self.config.pinn_lambda * pinn_loss
        return loss, B

    def _create_mask(self, epoch):
        return None

    def _compute_pinn_loss_sigmoid_weight(self,preds, targets):
        """
        计算带 Sigmoid 加权的地转 PINN 损失。
        """
        # 计算预测和真实的地转流以及权重
        targets = targets.to(self.config.device)
        ssh_concate = torch.cat([targets[:, :, :1],  preds[:, :, :1]], dim=2)
        u, v, w = compute_geostrophic_current(ssh_concate, self.lon, self.lat, if_solid_f= False)

        u_true, u_pred = u[:, :, 0], u[:, :, 1]
        v_true, v_pred = v[:, :, 0], v[:, :, 1]

        # 标准化
        ssh_std, u_std, v_std = self.stds
        u_norm_pred = (u_pred * ssh_std / u_std)
        v_norm_pred = (v_pred * ssh_std / v_std)
        u_norm_true = (u_true * ssh_std / u_std)
        v_norm_true = (v_true * ssh_std / v_std)


        w = w.to(self.config.device)

        # 加权 MSE
        loss_u = self.loss(u_norm_pred * torch.sqrt(w), u_norm_true * torch.sqrt(w))
        loss_v = self.loss(v_norm_pred * torch.sqrt(w), v_norm_true * torch.sqrt(w))
        return loss_u + loss_v

class TrainerMask(BaseMethod):
    def __init__(self,model, loss_func, loss_name, config, log_dir, optimizer, scheduler, mode='train'):
        super().__init__(model, config,'mask_mse',  log_dir, optimizer, scheduler, mode=mode)
        self.mask_land = np.load(config.path_land_mask)
        self.model_config = convert_configs(self.model_config)
        self.loss_ignore_nan = MSELossIgnoreNaNv2(~self.mask_land, model_configs=self.model_config, patched=True)
        self.loss = self.loss_ignore_nan
        self.loss_unpatch = MSELossIgnoreNaNv2(~self.mask_land, model_configs=self.model_config, patched=False)

        self.lon, self.lat = lon, lat
        self.stds = stds

        self.img_shape = (self.config.output_channels * self.model_config.patch_size ** 2, self.config.height // self.model_config.patch_size,
                     self.config.width // self.model_config.patch_size)
        self.mask_true_test = reverse_schedule_sampling(0,  self.model_config.total_length,
                                                   self.model_config.input_length, self.img_shape, self.model_config, mode='test')
        print(f"mask test: {self.mask_true_test.shape} {'-'*30}")
        self.config = config

    def _compute_loss(self, train_data, step, mask=None, test=False):
        B = train_data.shape[0]
        with autocast('cuda'):
            train_data = train_data.to(self.device)
            var_pred, loss = self.model(train_data, mask, loss_func = self.loss_ignore_nan if test else self.loss)
            if not test and self.config.is_pinn:
                # target = unpatchify_with_batch(train_data, self.model_config.patch_size, self.config.input_channel)[:,
                #          self.config.input_length:, 0:self.config.output_channel]
                # pred = unpatchify_with_batch(var_pred, self.model_config.patch_size, self.config.output_channel)[:,
                #            self.config.input_length - 1:, 0:self.config.output_channel]
                target = unpatchify_with_batch(train_data, self.model_config.patch_size, self.config.input_channels)[:,
                         1:, 0:self.config.output_channels]
                pred = unpatchify_with_batch(var_pred, self.model_config.patch_size, self.config.output_channels)[:,
                       :, 0:self.config.output_channels]
                pinn_loss = self._compute_pinn_loss_sigmoid_weight(pred, target)
                loss = loss + self.config.pinn_lambda * pinn_loss
        return loss, B

    def _compute_pinn_loss_sigmoid_weight(self,preds, targets):
        """
        计算带 Sigmoid 加权的地转 PINN 损失。
        """
        # 计算预测和真实的地转流以及权重
        targets = targets.to(self.config.device)
        ssh_concate = torch.cat([targets[:, :, :1],  preds[:, :, :1]], dim=2)
        u, v, w = compute_geostrophic_current(ssh_concate, self.lon, self.lat, if_solid_f= False)

        u_true, u_pred = u[:, :, 0], u[:, :, 1]
        v_true, v_pred = v[:, :, 0], v[:, :, 1]

        # 标准化
        ssh_std, u_std, v_std = self.stds
        u_norm_pred = (u_pred * ssh_std / u_std)
        v_norm_pred = (v_pred * ssh_std / v_std)
        u_norm_true = (u_true * ssh_std / u_std)
        v_norm_true = (v_true * ssh_std / v_std)


        w = w.to(self.config.device)

        # 加权 MSE
        loss_u = self.loss_unpatch(u_norm_pred * torch.sqrt(w), u_norm_true * torch.sqrt(w))
        loss_v = self.loss_unpatch(v_norm_pred * torch.sqrt(w), v_norm_true * torch.sqrt(w))
        return loss_u + loss_v

    def _create_mask(self, epoch):
        mask_true_train = reverse_schedule_sampling(epoch + 1,  self.model_config.total_length,
                                                    self.config.input_length, self.img_shape, self.model_config,
                                                    reverse=self.model_config.reverse_schedule)
        print(f"mask: {mask_true_train.shape}")

        return mask_true_train

if __name__ == '__main__':
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from configs import get_my_config
    from models import PredFormer_Model, Mask_PredFormer_Model, SimVP_Model, RNN
    from dataset import MvDataset
    import time
    import os
    from mytools import set_all_seeds

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = get_my_config()

    set_all_seeds(args.SEED)
    mask = np.load(args.path_land_mask)  # (H,W) 1: invalid, 0: valid
    if args.model_name == 'predformer':
        if args.mask_predformer:
            model = Mask_PredFormer_Model(args.model_config, mask).to(args.device)
        else:
            model = PredFormer_Model(args.model_config).to(args.device)
    elif args.model_name == 'simvp':
        model = SimVP_Model(**args.model_config).to(args.device)
    elif args.model_name == 'predrnn':
        model = RNN(args.model_config, mask_valid=~mask).to(args.device)

    train_dataset = MvDataset(args, mode='train')
    eval_dataset = MvDataset(args, mode='eval')
    test_dataset = MvDataset(args, mode='test')

    lon, lat = eval_dataset.lon, eval_dataset.lat
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    stds = np.load(args.path_stds)

    log_dir = rf"{args.model_savepath}/{model.__class__.__name__}_seed{args.SEED}/{args.file_name}_{time.strftime('%Y%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)

    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )

    if args.model_name == 'predrnn':
        loss = nn.MSELoss()
        loss_name = loss.__class__.__name__
        trainer = TrainerMask(model, loss, loss_name, args, log_dir, optimizer, scheduler,mode='train')
    else:
        loss = nn.MSELoss()
        loss_name = loss.__class__.__name__
        trainer = Trainer(model, loss, loss_name, args, log_dir, optimizer, scheduler,mode='train')

    trainer.train_model(train_dataset, eval_dataset, test_dataset)