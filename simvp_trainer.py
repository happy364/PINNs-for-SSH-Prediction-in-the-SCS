import numpy as np
from simvpv2 import SimVP_Model
from configs import mypara
import torch
from torch.utils.data import DataLoader
import math
from LoadData import MakeIterDataset
from torch.amp import autocast, GradScaler
import time
import os
from mytools import MSELossIgnoreNaN, EvalModel, EvalModelMask, compute_geostrophic_current, set_all_seeds

set_all_seeds(mypara.SEED)


class modelTrainer:
    def __init__(self, model, mask, config, log_file, optimizer, scheduler):
        self.config = config
        self.device = config.device
        self.log_file = log_file
        self.mymodel = model.to(config.device)
        self.scheduler = scheduler
        if hasattr(config, 'loss_weighting') and config.loss_weighting == 'uncertainty':
            print("Initializing uncertainty weighting parameters...")
            self.mymodel.log_var_main = torch.nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
            self.mymodel.log_var_pinn = torch.nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)

            # 重新创建优化器以确保包含所有参数
            all_params = list(self.mymodel.parameters())
            self.optimizer = torch.optim.AdamW(all_params, lr=optimizer.param_groups[0]['lr'],
                                               weight_decay=optimizer.param_groups[0]['weight_decay'])
        else:
            self.optimizer = optimizer

        self.mask_extent = torch.from_numpy(mask).bool()
        self.mask_nan  = torch.from_numpy(np.load(config.path_mask)).float()
        self.mask_valid = torch.from_numpy(~np.load(config.path_mask)).float()

        # 梯度累积配置
        self._configure_gradient_accumulation()

        self.loss_var = MSELossIgnoreNaN()
        self.test_model = EvalModel(self.mymodel, config).test_model
        self.model_savepath = f"{os.path.dirname(log_file)}/model_paras.pkl"

    def _train_epoch(self, dataset):
        """单epoch训练逻辑"""
        self.mymodel.train()
        total_loss = 0
        total_samples = 0
        epoch_time_start = time.time()
        scaler = GradScaler()
        dataloader = self._create_dataloader(dataset, is_train=True)
        for step, (inputs, targets) in enumerate(dataloader, 1):
            # 前向计算
            loss, loss_dict = self._compute_loss(inputs, targets, training=True)
            # 梯度累积
            loss = loss / self.accumulation_steps
            scaler.scale(loss).backward()

            # 参数更新
            if step % self.accumulation_steps == 0:
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            # 统计指标
            total_loss += loss.item() * self.accumulation_steps * inputs.size(0)
            total_samples += inputs.size(0)
        epoch_time = time.time() - epoch_time_start
        print(f'{total_samples/epoch_time:.2f} samples/s')
        if self.config.loss_weighting :
            print(f"Main weight: {loss_dict['main_weight'].item():.4f}, Pinn weight: {loss_dict['pinn_weight'].item():.4f}")

        return math.sqrt(total_loss / total_samples) if total_samples > 0 else 0


    def train_model(self, train_dataset, eval_dataset):
        """完整的训练流程"""
        self.lon, self.lat = eval_dataset.get_grids()
        if self.lon.ndim == 1 and self.lat.ndim == 1:
            self.lon, self.lat = np.meshgrid(self.lon, self.lat)

        self.stds = np.load(self.config.path_std)

        eval_loader = self._create_dataloader(eval_dataset, is_train=False)

        best_loss = math.inf
        early_stop_counter = 0

        start_time = time.time()  # 记录训练开始时间

        for epoch in range(self.config.num_epochs):
            print(f"\n{'-' * 20} Epoch {epoch + 1} {'-' * 20}")

            train_loss = self._train_epoch(train_dataset)
            eval_loss = self.test_model(dataloader=eval_loader)

            self.scheduler.step(eval_loss)
            self._log_metrics(epoch, train_loss, eval_loss)

            # 早停检查
            if eval_loss < best_loss:
                best_loss = eval_loss
                early_stop_counter = 0
                torch.save(self.mymodel.state_dict(), self.model_savepath)
                print(f"Model saved (New best loss: {best_loss:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        total_time  = time.time() - start_time
        print(f"Total training time: {total_time // 3600:.2f}h {total_time % 3600 // 60:.2f}m {total_time % 60:.2f}s")

    def _configure_gradient_accumulation(self):
        """配置梯度累积参数"""
        if getattr(self.config, 'is_acc', False):
            self.accumulation_steps = max(1, self.config.acc_steps)
            self.batch_size_train = self.config.batch_size_train // self.accumulation_steps
            print(f"[Grad Accum] Effective batch: {self.config.batch_size_train} = "
                  f"{self.batch_size_train} * {self.accumulation_steps}")
            self.batch_size_eval = 2* self.batch_size_train
        else:
            self.accumulation_steps = 1
            self.batch_size_train = self.config.batch_size_train
            self.batch_size_eval = 2 * self.batch_size_train

    def _create_dataloader(self, dataset, is_train=True):
        """创建单个DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size_train if is_train else self.config.batch_size_eval,
            shuffle=False,
            pin_memory=True
        )

    # def _compute_loss(self, inputs, targets, training=False):
    #     """计算损失（包含PINN等特殊逻辑）"""
    #     B, T, C, H, W = targets.shape
    #     mask_extent = self.mask_extent[None, None, None, :, :].expand(B, T, 1, H, W).to(self.device)
    #
    #     with autocast('cuda'):
    #         preds = self.mymodel(inputs.to(self.device))
    #         loss = self.loss_var(preds[mask_extent], targets[:, :, :1].to(self.device)[mask_extent])
    #
    #         if training and self.config.is_pinn:
    #             ssh_std, u_std, v_std = self.stds
    #             pinn_loss = self._compute_pinn_loss_sigmoid_weight(preds, targets, ssh_std, u_std, v_std)
    #             loss += self.config.pinn_lambda * pinn_loss
    #     return loss

    def _compute_loss(self, inputs, targets, training=False):
        """计算损失（包含PINN等特殊逻辑，支持不确定性加权和损失归一化）"""
        B, T, C, H, W = targets.shape
        mask_extent = self.mask_extent[None, None, None, :, :].expand(B, T, 1, H, W).to(self.device)

        with autocast('cuda'):
            preds = self.mymodel(inputs.to(self.device))

            # 主损失计算
            main_loss = self.loss_var(preds[mask_extent], targets[:, :, :1].to(self.device)[mask_extent])

            # PINN损失计算
            pinn_loss = 0.0
            if training and self.config.is_pinn:
                ssh_std, u_std, v_std = self.stds
                pinn_loss = self._compute_pinn_loss_sigmoid_weight(preds, targets, ssh_std, u_std, v_std)


            # 根据配置选择不同的损失加权策略
            if self.config.loss_weighting  and self.config.loss_weighting == 'uncertainty':
                # 不确定性加权方法
                total_loss, loss_dict = self._apply_uncertainty_weighting(main_loss, pinn_loss, training)

            elif self.config.loss_weighting  and self.config.loss_weighting == 'normalization':
                # 损失归一化方法
                total_loss, loss_dict = self._apply_loss_normalization(main_loss, pinn_loss, training)

            else:
                # 默认简单相加（原始逻辑）
                total_loss = main_loss + self.config.pinn_lambda * pinn_loss
                loss_dict = None

            return total_loss, loss_dict

    def _apply_uncertainty_weighting(self, main_loss, pinn_loss, training):
        """应用不确定性加权"""

        if training and pinn_loss > 0:
            # 不确定性加权公式: L_total = 1/(2*σ²)*L + log(σ)

            sigma_main_sq = torch.exp(self.mymodel.log_var_main)  # σ₁²
            sigma_pinn_sq = torch.exp(self.mymodel.log_var_pinn)  # σ₂²

            weight_main = 1.0 / (2 * sigma_main_sq)
            weight_pinn = 1.0 / (2 * sigma_pinn_sq)
            # weight_pinn = sigma_pinn_sq

            weighted_main_loss =  weight_main * main_loss
            weighted_pinn_loss = weight_pinn * pinn_loss
            regularization_term = 1/2 * (self.mymodel.log_var_pinn **2 + self.mymodel.log_var_main **2)

            total_loss = weighted_main_loss + weighted_pinn_loss + regularization_term

            loss_dict = {
                'total_loss': total_loss,
                'main_loss': main_loss,
                'pinn_loss': pinn_loss,
                'main_weight': weight_main.detach(),
                'pinn_weight': weight_pinn.detach(),
                'log_var_main': self.mymodel.log_var_main.detach(),
                'log_var_pinn': self.mymodel.log_var_pinn.detach()
            }
        else:
            total_loss = main_loss
            loss_dict = None


        return total_loss, loss_dict

    def _apply_loss_normalization(self, main_loss, pinn_loss, training):
        """应用损失归一化"""
        # 初始化损失统计信息
        if not hasattr(self, 'main_loss_history'):
            self.main_loss_history = []
            self.pinn_loss_history = []

        if training and pinn_loss > 0:
            # 收集损失历史（用于归一化）
            self.main_loss_history.append(main_loss.detach())
            self.pinn_loss_history.append(pinn_loss.detach())

            # 保持历史记录长度
            if len(self.main_loss_history) > 100:  # 可配置的窗口大小
                self.main_loss_history.pop(0)
                self.pinn_loss_history.pop(0)

            if len(self.main_loss_history) > 1:
                # 方法1: 基于损失比例的归一化
                main_mean = torch.stack(self.main_loss_history).mean()
                pinn_mean = torch.stack(self.pinn_loss_history).mean()

                total_mean = main_mean + pinn_mean
                weight_main = main_mean / total_mean
                weight_pinn = pinn_mean / total_mean

                # 方法2: 基于初始损失的归一化（备选）
                # if not hasattr(self, 'initial_main_loss'):
                #     self.initial_main_loss = main_loss.detach()
                #     self.initial_pinn_loss = pinn_loss.detach()
                # weight_main = main_loss / self.initial_main_loss
                # weight_pinn = pinn_loss / self.initial_pinn_loss

                total_loss = weight_main * main_loss + weight_pinn * pinn_loss

                loss_dict = {
                    'total_loss': total_loss,
                    'main_loss': main_loss,
                    'pinn_loss': pinn_loss,
                    'main_weight': weight_main,
                    'pinn_weight': weight_pinn
                }
            else:
                # 历史数据不足时使用简单相加
                total_loss = main_loss + pinn_loss
                loss_dict = {
                    'total_loss': total_loss,
                    'main_loss': main_loss,
                    'pinn_loss': pinn_loss,
                    'main_weight': 1.0,
                    'pinn_weight': 1.0
                }
        else:
            # 没有PINN损失时，只使用主损失
            total_loss = main_loss
            loss_dict = {
                'total_loss': total_loss,
                'main_loss': main_loss,
                'pinn_loss': pinn_loss,
                'main_weight': 1.0,
                'pinn_weight': 0.0
            }

        return total_loss, loss_dict

    def _compute_pinn_loss_sigmoid_weight(self,preds, targets, ssh_std, u_std, v_std):
        """
        计算带 Sigmoid 加权的地转 PINN 损失。
        """
        # 计算预测和真实的地转流以及权重
        targets = targets.to(self.config.device)
        ssh_concate = torch.cat([targets[:, :, :1],  preds[:, :, :1]], dim=2)
        u, v, w = compute_geostrophic_current(ssh_concate, lon, lat)

        u_true, u_pred = u[:, :, 0], u[:, :, 1]
        v_true, v_pred = v[:, :, 0], v[:, :, 1]

        # 标准化
        ssh_std, u_std, v_std = self.stds
        u_norm_pred = (u_pred * ssh_std / u_std)
        v_norm_pred = (v_pred * ssh_std / v_std)
        u_norm_true = (u_true * ssh_std / u_std)
        v_norm_true = (v_true * ssh_std / v_std)

        if self.config.no_weight:
            w = torch.tensor(1.0)
        w = w.to(self.config.device)

        # 加权 MSE
        loss_u = self.loss_var(u_norm_pred * torch.sqrt(w), u_norm_true * torch.sqrt(w))
        loss_v = self.loss_var(v_norm_pred * torch.sqrt(w), v_norm_true * torch.sqrt(w))
        return loss_u + loss_v

    def _compute_pinn_loss(self,preds, targets, ssh_std, u_std, v_std):
        """
        计算带 Sigmoid 加权的地转 PINN 损失。
        """
        u_geo_pred, v_geo_pred = compute_geostrophic_current(preds, self.lon, self.lat,mask_valid=self.mask_valid.bool())
        u_true, v_true = compute_geostrophic_current(targets, self.lon, self.lat,mask_valid=self.mask_valid.bool())

        u_geo_pred = (u_geo_pred * ssh_std / u_std)
        v_geo_pred = (v_geo_pred * ssh_std / v_std)
        u_geo_true = (u_true * ssh_std / u_std).to(self.config.device)
        v_geo_true = (v_true * ssh_std / v_std).to(self.config.device)
        return self.loss_var(u_geo_pred, u_geo_true) + self.loss_var(v_geo_pred, v_geo_true)

    def _log_metrics(self, epoch, train_loss, eval_loss):
        """记录训练指标"""
        current_lr = self.scheduler.get_last_lr()[0]
        # ssh_std, u_std, v_std = self.stds
        # train_loss *= ssh_std
        # eval_loss *= ssh_std
        log_str = (f"Epoch {epoch + 1}: "
                   f"Train Loss={train_loss:.6f}, "
                   f"Eval Loss={eval_loss:.6f}, "
                   f"LR={current_lr:.2e}")
        print(log_str)

        with open(self.log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{eval_loss:.6f},{current_lr}\n")



if __name__ == "__main__":
    import csv
    from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
    from configs import mypara
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


    model = SimVP_Model(**mypara.simvp_gsta_config)

    train_dataset = MakeIterDataset(mypara, 'train',norm=False)
    eval_dataset = MakeIterDataset(mypara, 'eval',norm=False)

    log_file = rf"{mypara.model_savepath}/{model.__class__.__name__}_seed{mypara.SEED}/{mypara.file_name}_{time.strftime('%Y%m%d_%H%M')}/trainlog{time.strftime('%Y%m%d_%H%M')}.csv"  # todo
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保目录存在

    def writeDict2csv(data_dict, file_object):

        writer = csv.writer(file_object)
        # 确保键值顺序一致
        items = list(data_dict.items())
        writer.writerow([k for k, v in items])
        writer.writerow([str(v) for k, v in items])

    weight_decay = 0.0001

    # scheduler_kwargs = {
    #     'lr': 1e-5,
    #     'scheduler_type': 'MultiStepLR',
    #     'milestones': [100, 200, 300, 500],
    #     'gamma': 0.1,
    # }
    scheduler_kwargs = {
        'lr': 1e-4,
        'scheduler_type': 'ReduceLROnPlateau',
        'gamma': 0.1,
        'patience': 5,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=scheduler_kwargs['lr'], weight_decay=weight_decay)
    # scheduler = MultiStepLR(optimizer, milestones=scheduler_kwargs['milestones'], gamma=scheduler_kwargs['gamma'])
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
    )

    if mypara.is_finetune:
        scheduler_kwargs = {
            'lr': 1e-6,
            'scheduler_type': 'MultiStepLR',
            'milestones': [50, 100, 200, 300],
            'gamma': 0.1,
        }
        if mypara.is_pinn:
            model.load_state_dict(torch.load(mypara.finetune_pinn_path))
        else:
            model.load_state_dict(torch.load(mypara.finetune_path))  # todo

        optimizer = torch.optim.AdamW(model.parameters(), lr=scheduler_kwargs['lr'], weight_decay=weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=scheduler_kwargs['milestones'], gamma=scheduler_kwargs['gamma'])

    with open(log_file, 'w', newline='') as f:  # 注意添加 newline='' 避免空行
        f.write(f"# New Training Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        writeDict2csv(scheduler_kwargs, f)
        writeDict2csv(mypara.simvp_gsta_config, f)
        f.write(
            f"tauxy:{mypara.need_wind},ssh:{mypara.need_ssh},uv:{mypara.need_uv},need mask:{mypara.need_mask},pinn:{mypara.is_pinn},weight_decay:{weight_decay}\n")
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    lon, lat = eval_dataset.get_grids()
    mean, std = eval_dataset.get_info()

    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)

    extent = mypara.extent
    lon_min, lon_max, lat_min, lat_max = extent
    mask = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)

    if mypara.loss_mask:
        mask_l = mask
    else:
        mask_l = np.ones_like(lon, dtype=bool)

    trainer = modelTrainer(model, mask_l, mypara, log_file, optimizer, scheduler)
    trainer.train_model(train_dataset,eval_dataset)

    model.eval()
    model.load_state_dict(torch.load(f"{os.path.dirname(log_file)}/model_paras.pkl"))
    test_dataset = MakeIterDataset(mypara, 'test',norm=False)

    data_loader = DataLoader(test_dataset, batch_size=mypara.batch_size_eval, shuffle=False, pin_memory=True)
    modelEvaler = EvalModelMask(model, mypara, torch.from_numpy(mask))
    rmse = modelEvaler.test_mask(data_loader)
    print(f"test loss:{rmse} rmse)")
    with open(log_file, 'a') as f:
        f.write(f"test loss:,{rmse} rmse\n")



