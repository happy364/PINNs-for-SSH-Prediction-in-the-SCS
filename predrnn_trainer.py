import numpy as np
from LoadData import MakeDatasetRNN
from predrnnv2 import RNN
from configs import mypara, rnn2configs
import torch
from torch.utils.data import DataLoader
import math
from torch.amp import autocast, GradScaler
import time
import os
from mytools import (MSELossIgnoreNaN_RNN, EvalModel_RNN, compute_geostrophic_current,
                     reverse_schedule_sampling, set_all_seeds, unpatchify_with_batch)

set_all_seeds(mypara.SEED)


class modelTrainer:
    def __init__(self, model, mypara, rnn2configs, log_file, optimizer, scheduler):
        self.mypara = mypara
        self.configs = rnn2configs
        self.device = mypara.device
        self.log_file = log_file
        self.mymodel = model.to(mypara.device)
        self.optimizer = optimizer
        self.mask_land = np.load(mypara.path_mask)
        self.criterion = MSELossIgnoreNaN_RNN(patched=False)
        self.warmup_epochs = rnn2configs.r_sampling_step_2

        self.scheduler = scheduler

        self.test_model = EvalModel_RNN(self.mymodel, mypara, rnn2configs).test_model
        self.model_savepath = f"{os.path.dirname(log_file)}/model_paras.pkl"

        self._configure_gradient_accumulation()

    def _train_epoch(self, dataset, mask_true):
        """单epoch训练逻辑"""
        self.mymodel.train()
        total_loss = 0
        total_samples = 0
        epoch_time_start = time.time()
        scaler = GradScaler()
        dataloader = self._create_dataloader(dataset, is_train=True)

        for step, input_var in enumerate(dataloader, 1):

            # 前向计算
            B = input_var.shape[0]
            if B != self.mypara.batch_size_train:
                mask_true_batch = mask_true[:B]
            else:
                mask_true_batch = mask_true
            loss = self._compute_loss(input_var, mask_true_batch)
            # 梯度累积
            loss = loss / self.accumulation_steps
            scaler.scale(loss).backward()

            # 参数更新
            if step % self.accumulation_steps == 0:
                # **unscale 将 grad 从 "grad * scale" 恢复为真实 grad**
                scaler.unscale_(self.optimizer)

                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()

            # 统计指标
            total_loss += loss.item() * self.accumulation_steps * B
            total_samples += B

        epoch_time = time.time() - epoch_time_start
        print(f'{total_samples / epoch_time:.2f} samples/s')

        return math.sqrt(total_loss / total_samples) if total_samples > 0 else 0

    def _compute_loss(self, input_var, mask_true):
        """计算损失（包含PredRNN特定逻辑和PINN等特殊逻辑）"""
        with autocast('cuda'):
            input_var = input_var.to(self.device)
            var_pred, loss = self.mymodel(input_var, mask_true)

            if self.mypara.is_pinn:
                # 获取网格经纬度
                target = unpatchify_with_batch(input_var, rnn2configs.patch_size, mypara.input_channel)[:,
                         self.configs.input_length:, 0:self.mypara.output_channel]
                var_pred = unpatchify_with_batch(var_pred, rnn2configs.patch_size, mypara.output_channel)[:,
                           self.configs.input_length - 1:, 0:self.mypara.output_channel]

                lon, lat = self.lon, self.lat

                ssh_concate = torch.concat([target, var_pred], dim=2)

                u, v, w = compute_geostrophic_current(ssh_concate, lon, lat)

                u_true, u_pred = u[:, :, 0], u[:, :, 1]
                v_true, v_pred = v[:, :, 0], v[:, :, 1]

                # 标准化
                ssh_std, u_std, v_std = self.stds
                u_norm_pred = (u_pred * ssh_std / u_std)
                v_norm_pred = (v_pred * ssh_std / v_std)
                u_norm_true = (u_true * ssh_std / u_std)
                v_norm_true = (v_true * ssh_std / v_std)

                if self.mypara.no_weight:
                    w = torch.tensor(1.0)
                w = w.to(self.device)

                # B,T,H,W = u_norm_pred.shape
                # mask_land = np.tile(self.mask_land.reshape(1,1,H,W), (B,T,1,1))
                # mask_land = torch.from_numpy(mask_land).to(self.device)
                # u_norm_true[mask_land] = torch.nan
                # v_norm_true[mask_land] = torch.nan

                # 加权 MSE
                loss_u = self.criterion(u_norm_pred * torch.sqrt(w), u_norm_true * torch.sqrt(w))
                loss_v = self.criterion(v_norm_pred * torch.sqrt(w), v_norm_true * torch.sqrt(w))
                pinn_loss = loss_u + loss_v

                loss_var = loss + self.mypara.pinn_lambda * pinn_loss
            else:
                loss_var = loss
        return loss_var

    def train_model(self, train_dataset, eval_dataset):
        """完整的训练流程"""
        self.lon, self.lat = eval_dataset.get_grids()
        if self.lon.ndim == 1 and self.lat.ndim == 1:
            self.lon, self.lat = np.meshgrid(self.lon, self.lat)

        self.stds = np.load(self.mypara.path_std)

        eval_loader = self._create_dataloader(eval_dataset, is_train=False)

        best_loss = math.inf
        early_stop_counter = 0

        start_time = time.time()  # 记录训练开始时间

        for epoch in range(self.mypara.num_epochs):
            self.current_epoch = epoch
            print(f"\n{'-' * 20} Epoch {epoch + 1} {'-' * 20}")

            img_shape = (self.mypara.output_channel * rnn2configs.patch_size ** 2, 160 // rnn2configs.patch_size,
                         160 // rnn2configs.patch_size)
            mask_true_train = reverse_schedule_sampling(epoch + 1, self.mypara.batch_size_train,
                                                        rnn2configs.total_length,
                                                        rnn2configs.input_length, img_shape, rnn2configs,
                                                        reverse=self.configs.reverse_schedule)
            mask_true_eval = reverse_schedule_sampling(epoch, self.mypara.batch_size_eval, rnn2configs.total_length,
                                                       rnn2configs.input_length, img_shape, rnn2configs, mode='test')

            train_loss = self._train_epoch(train_dataset, mask_true_train)
            eval_loss = self.test_model(dataloader=eval_loader, mask_true=mask_true_eval)

            self._log_metrics(epoch, train_loss, eval_loss)

            # 早停检查
            if epoch < self.warmup_epochs + 1:
                continue

            self.scheduler.step(eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                early_stop_counter = 0
                torch.save(self.mymodel.state_dict(), self.model_savepath)
                print(f"Model saved (New best loss: {best_loss:.6f})")
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.mypara.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        total_time = time.time() - start_time
        print(f"Total training time: {total_time // 3600:.2f}h {total_time % 3600 // 60:.2f}m {total_time % 60:.2f}s")

    def _configure_gradient_accumulation(self):
        """配置梯度累积参数"""
        if getattr(self.mypara, 'is_acc', False):
            self.accumulation_steps = max(1, self.mypara.acc_steps)
            self.batch_size_train = self.mypara.batch_size_train // self.accumulation_steps
            print(f"[Grad Accum] Effective batch: {self.mypara.batch_size_train} = "
                  f"{self.batch_size_train} * {self.accumulation_steps}")
            self.batch_size_eval = 2 * self.batch_size_train
        else:
            self.accumulation_steps = 1
            self.batch_size_train = self.mypara.batch_size_train
            self.batch_size_eval = 2 * self.batch_size_train

    def _create_dataloader(self, dataset, is_train=True):
        """创建单个DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size_train if is_train else self.mypara.batch_size_eval,
            shuffle=True if is_train else False,
            pin_memory=True,
        )

    def _log_metrics(self, epoch, train_loss, eval_loss):
        """记录训练指标"""
        current_lr = self.scheduler.get_last_lr()[0]
        log_str = (f"Epoch {epoch + 1}: "
                   f"Train Loss={train_loss:.6f}, "
                   f"Eval Loss={eval_loss:.6f}, "
                   f"LR={current_lr:.2e},")
        print(log_str)

        with open(self.log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{eval_loss:.6f},{current_lr},\n")


if __name__ == "__main__":
    import csv
    from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau

    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    num_layers = 4
    hidden_size = 64

    model = RNN(num_layers, [hidden_size] * num_layers, rnn2configs)

    traindataset = MakeDatasetRNN(mypara, rnn2configs, 'trainer', norm=True)
    evaldataset = MakeDatasetRNN(mypara, rnn2configs, 'eval', norm=True)
    test_dataset = MakeDatasetRNN(mypara, rnn2configs, 'test', norm=True)

    log_file = rf"{mypara.model_savepath}/{model.__class__.__name__}/{mypara.file_name}_{time.strftime('%Y%m%d_%H%M%S')}/trainlog{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # 确保目录存在


    def writeDict2csv(data_dict, file_object):
        """将字典写入CSV文件，键和值分两行
        Args:
            data_dict: 要写入的字典
            file_object: 已经打开的文件对象
        """
        writer = csv.writer(file_object)
        # 确保键值顺序一致
        items = list(data_dict.items())
        writer.writerow([k for k, v in items])
        writer.writerow([str(v) for k, v in items])

    weight_decay = 0.0001

    scheduler_kwargs = {
        'lr': 1e-5,
        'scheduler_type': 'ReduceLROnPlateau',
        'gamma': 0.5,
        'patience': 5,
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=scheduler_kwargs['lr'], weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控损失应该最小化
        factor=0.1,
        patience=5,  # 5个epoch没有改善就衰减
    )
    # scheduler_kwargs = {
    #     'lr': 1e-4,
    #     'scheduler_type': 'MultiStepLR',
    #     'milestones': [mypara.warmup_epochs, 50, 200,250, 300],
    #     'gamma': 0.1,
    # }
    # scheduler = MultiStepLR(optimizer, milestones=scheduler_kwargs['milestones'], gamma=scheduler_kwargs['gamma'])

    with open(log_file, 'w', newline='') as f:  # 注意添加 newline='' 避免空行
        # 写入头部信息（作为注释行）
        f.write(f"# New Training Session - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        writeDict2csv(scheduler_kwargs, f)
        # 写入配置参数
        rnn2configs_dict = rnn2configs.__dict__.copy()
        writeDict2csv(rnn2configs_dict, f)
        f.write(f"num_layers:,{num_layers},,hidden_size:,{hidden_size}")
        f.write(f"predrnn2")
        f.write(
            f"ssh:{mypara.need_ssh},uv:{mypara.need_uv},weight_decay:{weight_decay},reverse_schedule:{rnn2configs.reverse_schedule}"
            f"step1:{rnn2configs.r_sampling_step_1},step2:{rnn2configs.r_sampling_step_2}\n")
        # 写入CSV表头
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "lr"])

    trainer = modelTrainer(model, mypara, rnn2configs, log_file, optimizer, scheduler)
    trainer.train_model(
        train_dataset=traindataset,
        eval_dataset=evaldataset,
    )
    model.eval()
    model.load_state_dict(torch.load(f"{os.path.dirname(log_file)}/model_paras.pkl"))

    data_loader = DataLoader(test_dataset, batch_size=mypara.batch_size_eval, shuffle=False, pin_memory=True)
    modelEvaler = EvalModel_RNN(model, mypara, rnn2configs)
    img_shape = (mypara.output_channel * rnn2configs.patch_size ** 2, 160 // rnn2configs.patch_size,
                 160 // rnn2configs.patch_size)

    mask_true_test = reverse_schedule_sampling(None, mypara.batch_size_eval, rnn2configs.total_length,
                                               rnn2configs.input_length, img_shape, rnn2configs, mode='test')

    rmse = modelEvaler.test_model(data_loader, mask_true_test)
    print(f"test loss:{rmse} rmse)")
    with open(log_file, 'a') as f:
        f.write(f"test loss:,{rmse} rmse\n")