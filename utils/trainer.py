import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from pathlib import Path
import time
from collections import defaultdict

from .masking import DynamicMaskGenerator
from .losses import GANLoss, AdaptiveWeightScheduler, MultiScaleSpectralLoss

class Trainer:
    def __init__(self, config, generator, discriminator, train_loader, val_loader, device):
        self.config = config
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics = {}
        self.best_val_loss = float('inf')
        
        # 🆕 添加动态掩码生成器
        self.mask_generator = DynamicMaskGenerator(config)
        
        # 添加自适应权重调度器
        self.weight_scheduler = AdaptiveWeightScheduler({
            'rec': config.lambda_rec,
            'spec': config.lambda_spec,
            'adv': config.lambda_adv,
            'gp': config.lambda_gp,
            'rel': config.lambda_rel
        }, beta=config.weight_beta)
        
        # 初始化损失函数
        self.criterion = GANLoss(
            lambda_rec=config.lambda_rec,
            lambda_spec=config.lambda_spec,
            lambda_gp=config.lambda_gp
        ).to(device)
        
        # 使用更好的优化器和学习率调度器
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config.lr_g,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config.lr_d,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
        
        # 使用 ReduceLROnPlateau 调度器
        self.g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.g_optimizer,
            mode='min',
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=config.min_lr
        )
        
        self.d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.d_optimizer,
            mode='min',
            factor=0.5,
            patience=config.scheduler_patience,
            min_lr=config.min_lr
        )
        
        self.multi_scale_spec_loss = MultiScaleSpectralLoss()
        
        # 添加时间记录相关属性
        self.train_start_time = None
        self.train_end_time = None
        self.total_training_time = None
        self.inference_times = []
        
        # 设置日志
        self.setup_logging()
    
    # 🔄 修改create_dynamic_mask方法
    def create_dynamic_mask(self, data, dynamic_mode=True):
        """
        创建动态缺失掩码 - 简化版本，直接调用DynamicMaskGenerator
        """
        if dynamic_mode and self.config.dynamic_masking:
            return self.mask_generator.create_dynamic_mask(data)
        else:
            # 如果不使用动态模式，返回全1掩码（无缺失）
            return torch.ones_like(data)
    
    def calculate_d_loss(self, real_data, mask=None):
        """计算判别器的损失"""
        batch_size = real_data.size(0)
        
        # 生成假数据
        with torch.no_grad():
            fake_data = self.generator(real_data, mask)
        
        # 真实数据的判别结果
        real_pred = self.discriminator(real_data)
        
        # 生成数据的判别结果
        fake_pred = self.discriminator(fake_data.detach())
        
        # Wasserstein损失
        d_loss = -torch.mean(real_pred) + torch.mean(fake_pred)
        
        # 计算梯度惩罚
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        alpha = alpha.expand_as(real_data)
        
        # 生成插值样本
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # 计算插值样本的判别结果
        d_interpolated = self.discriminator(interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 修改这里：使用reshape而不是view，并计算正确的维度
        gradients = gradients.reshape(batch_size, -1)  # 将所有维度展平为一维
        gradient_norm = gradients.norm(2, dim=1)  # 计算L2范数
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        # 总损失
        d_loss = d_loss + self.config.lambda_gp * gradient_penalty
        
        # 返回损失和详细信息
        return d_loss, {
            'd_loss_real': -torch.mean(real_pred).item(),
            'd_loss_fake': torch.mean(fake_pred).item(),
            'gradient_penalty': gradient_penalty.item(),
            'd_loss': d_loss.item()
        }
    
    def calculate_g_loss(self, real_data, fake_data, mask=None):
        # 计算各个损失
        fake_pred = self.discriminator(fake_data)
        g_loss_adv = -torch.mean(fake_pred)
        
        if mask is not None:
            g_loss_rec = F.mse_loss(
                fake_data * (1 - mask),
                real_data * (1 - mask)
            )
        else:
            g_loss_rec = F.mse_loss(fake_data, real_data)
        
        spec_loss = self.multi_scale_spec_loss(real_data, fake_data)
        
        # 获取当前的自适应权重
        current_weights = self.weight_scheduler.get_weights()
        
        # 计算总损失
        g_loss = (
            current_weights['adv'] * g_loss_adv +
            current_weights['rec'] * g_loss_rec +
            current_weights['spec'] * spec_loss
        )
        
        # 准备当前损失字典用于更新权重
        current_losses = {
            'adv': g_loss_adv.item(),
            'rec': g_loss_rec.item(),
            'spec': spec_loss.item()
        }
        
        # 更新权重
        self.weight_scheduler.update(current_losses)
        
        return g_loss, {
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_rec': g_loss_rec.item(),
            'spec_loss': spec_loss.item(),
            'weights': current_weights
        }
    
    def evaluate_metrics(self, real_data, generated_data, mask):
        """计算评估指标"""
        with torch.no_grad():
            # 只计算缺失部分的指标
            missing_mask = 1 - mask  # 转换为1表示缺失
            
            # 提取缺失部分的真实数据和生成数据
            real_missing = real_data * missing_mask
            gen_missing = generated_data * missing_mask
            
            # 计算基本指标
            mse = F.mse_loss(gen_missing, real_missing).item()
            mae = F.l1_loss(gen_missing, real_missing).item()
            rmse = torch.sqrt(torch.tensor(mse)).item()
            
            # 计算相对误差 (避免除以0)
            denominator = torch.abs(real_missing) + 1e-6
            relative_error = torch.mean(
                torch.abs(gen_missing - real_missing) / denominator
            ).item()
            
            # 计算频谱误差
            real_fft = torch.fft.fft(real_data, dim=-1)
            gen_fft = torch.fft.fft(generated_data, dim=-1)
            spec_error = F.mse_loss(
                torch.abs(gen_fft), 
                torch.abs(real_fft)
            ).item()
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'relative_error': relative_error,
                'spec_error': spec_error
            }
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch):
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        epoch_g_losses = defaultdict(float)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, data in enumerate(pbar):
            # 处理数据加载器返回的数据
            if isinstance(data, (list, tuple)):
                real_data = data[0]
            else:
                real_data = data
                
            # 确保数据是张量格式
            if not isinstance(real_data, torch.Tensor):
                real_data = torch.FloatTensor(real_data)
                
            real_data = real_data.to(self.device)
            
            # 创建动态掩码
            mask = self.create_dynamic_mask(real_data, dynamic_mode=True)
            mask = mask.to(self.device)
            
            # 应用掩码到输入数据
            masked_data = real_data * mask
            
            batch_size = real_data.size(0)
            
            # 训练判别器
            for _ in range(self.config.n_critic):
                self.d_optimizer.zero_grad()
                d_loss, d_loss_dict = self.calculate_d_loss(real_data, mask)
                d_loss.backward()
                # 梯度裁剪
                d_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(), 
                    self.config.gradient_clip
                )
                
                # 检查梯度是否正常
                if torch.isnan(d_grad_norm) or torch.isinf(d_grad_norm):
                    self.logger.warning(f"Epoch {epoch}, Batch {batch_idx}: D gradient norm is {d_grad_norm}")
                    continue
                    
                self.d_optimizer.step()
            
            # 生成假数据 - 使用掩码数据作为输入
            fake_data = self.generator(masked_data, mask)
            
            # 训练生成器
            self.g_optimizer.zero_grad()
            g_loss, g_loss_dict = self.calculate_g_loss(real_data, fake_data, mask)
            g_loss.backward()
            # 梯度裁剪
            g_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(), 
                self.config.gradient_clip
            )
            
            # 梯度检查
            if torch.isnan(g_grad_norm) or torch.isinf(g_grad_norm):
                self.logger.warning(f"Epoch {epoch}, Batch {batch_idx}: G gradient norm is {g_grad_norm}")
                continue
            self.g_optimizer.step()
            
            # 更新损失统计
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            # 更新各损失项统计
            for key, value in g_loss_dict.items():
                if key != 'weights' and isinstance(value, (float, int)):
                    epoch_g_losses[key] += value
                    
            # 获取当前权重
            current_weights = g_loss_dict['weights']
            # 计算平均损失
            avg_g_loss = total_g_loss / (batch_idx + 1)
            avg_d_loss = total_d_loss / (batch_idx + 1)
            # 更新进度条显示
            pbar.set_postfix({
                'D_loss': f'{avg_d_loss:.4f}',
                'G_loss': f'{avg_g_loss:.4f}',
                'D_grad': f'{d_grad_norm:.2f}',
                'G_grad': f'{g_grad_norm:.2f}',
                'rec_w': f'{current_weights["rec"]:.2f}',
                'spec_w': f'{current_weights["spec"]:.2f}',
                'adv_w': f'{current_weights["adv"]:.2f}'
            })
        
        # 计算epoch级别的平均损失
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        
        # 计算各个损失项的平均值
        for key in epoch_g_losses:
            epoch_g_losses[key] /= len(self.train_loader)
        
        # 验证
        val_loss, metrics = self.validate()
        
        # 更新学习率调度器
        self.g_scheduler.step(val_loss)
        self.d_scheduler.step(val_loss)
        
        return avg_g_loss, avg_d_loss, val_loss, metrics
    
    @torch.no_grad()
    def validate(self):
        self.generator.eval()
        self.discriminator.eval()
        
        total_val_loss = 0
        all_metrics = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for data in self.val_loader:
                if isinstance(data, (list, tuple)):
                    real_data = data[0]
                else:
                    real_data = data
                
                real_data = real_data.to(self.device)
                
                # 创建动态掩码进行验证
                mask = self.create_dynamic_mask(real_data, dynamic_mode=True)
                mask = mask.to(self.device)
                
                # 应用掩码到输入
                masked_data = real_data * mask
                
                # 生成恢复数据
                fake_data = self.generator(masked_data, mask)
                
                # 只评估被掩码的部分
                val_loss = F.mse_loss(fake_data * (1 - mask), real_data * (1 - mask))
                total_val_loss += val_loss.item()
                
                # 计算其他指标
                batch_metrics = self.evaluate_metrics(real_data, fake_data, mask)
                for key, value in batch_metrics.items():
                    all_metrics[key] += value
                
                num_batches += 1
        
        # 计算平均损失和指标
        avg_val_loss = total_val_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
        
        self.generator.train()
        self.discriminator.train()
        
        return avg_val_loss, avg_metrics
    
    def train(self):
        """训练方法"""
        self.train_start_time = time.time()  # 记录训练开始时间
        for epoch in range(self.config.num_epochs):
            # 获取训练结果
            train_g_loss, train_d_loss, val_loss, metrics = self.train_epoch(epoch)
            
            # 更新学习率调度器
            self.g_scheduler.step(val_loss)
            self.d_scheduler.step(val_loss)
            
            # 记录当前epoch的指标
            current_metrics = {
                'train_g_loss': train_g_loss,
                'train_d_loss': train_d_loss,
                'val_loss': val_loss
            }
            # 合并其他指标
            if metrics:
                current_metrics.update(metrics)
            
            # 保存到self.metrics
            self.metrics[epoch] = current_metrics
            
            # 格式化日志信息
            try:
                # 基础训练信息
                log_msg = [
                    f'Epoch {epoch}',
                    f'G_loss={train_g_loss:.4f}',
                    f'D_loss={train_d_loss:.4f}',
                    f'Val_loss={val_loss:.4f}'
                ]
                
                # 添加其他指标
                metric_items = []
                for k, v in current_metrics.items():
                    if k not in ['train_g_loss', 'train_d_loss', 'val_loss']:
                        if isinstance(v, (float, int)):
                            metric_items.append(f'{k}={v:.4f}')
                        else:
                            metric_items.append(f'{k}={v}')
                
                if metric_items:
                    log_msg.append('Metrics: ' + ', '.join(metric_items))
                
                # 合并所有日志信息
                log_str = ' | '.join(log_msg)
                self.logger.info(log_str)
                
            except Exception as e:
                self.logger.error(f"Error formatting log message: {str(e)}")
                self.logger.error(f"Current metrics: {current_metrics}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"New best model saved with val_loss={val_loss:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
                self.logger.info(f"Saved checkpoint for epoch {epoch}")
        
        self.train_end_time = time.time()  # 记录训练结束时间
        self.total_training_time = self.train_end_time - self.train_start_time
        
        # 记录训练时间信息
        hours, remainder = divmod(self.total_training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.logger.info(f"总训练时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        return self.metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': self.metrics
        }
        
        # 创建检查点目录
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存常规检查点
        if not is_best:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_model_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_model_path)