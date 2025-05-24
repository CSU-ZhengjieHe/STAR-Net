import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):
    def __init__(self, lambda_rec, lambda_spec, lambda_gp):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_spec = lambda_spec
        self.lambda_gp = lambda_gp
        self.l1_loss = nn.L1Loss()
    
    def wasserstein_loss(self, output, real=True):
        """Wasserstein损失"""
        return output.mean() if real else -output.mean()
    
    def reconstruction_loss(self, generated_data, true_data, mask):
        """改进的重构损失"""
        # MSE损失
        mse_loss = F.mse_loss(
            generated_data * (1 - mask),
            true_data * (1 - mask)
        )
        # L1损失
        l1_loss = self.l1_loss(
            generated_data * (1 - mask),
            true_data * (1 - mask)
        )
        # 组合损失
        return 0.7 * mse_loss + 0.3 * l1_loss
    
    def spectral_loss(self, generated, real):
        """改进的频谱损失"""
        gen_freq = torch.fft.fft(generated, dim=-1)
        real_freq = torch.fft.fft(real, dim=-1)
        
        # 幅度谱损失
        magnitude_loss = F.mse_loss(
            torch.abs(gen_freq),
            torch.abs(real_freq)
        )
        
        # 相位谱损失
        phase_loss = F.mse_loss(
            torch.angle(gen_freq),
            torch.angle(real_freq)
        )
        
        return magnitude_loss + 0.5 * phase_loss
    
    def gradient_penalty(self, discriminator, real_data, fake_data):
        """梯度惩罚"""
        batch_size = real_data.size(0)
        # 创建随机插值系数
        alpha = torch.rand(batch_size, 1, 1, device=real_data.device)
        alpha = alpha.expand_as(real_data)
        
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        d_interpolated = discriminator(interpolated)
        grad_outputs = torch.ones_like(d_interpolated)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.reshape(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class AdaptiveWeightScheduler:
    def __init__(self, init_weights, beta=0.99):
        """
        init_weights: 初始权重字典，例如 {'rec': 50.0, 'spec': 10.0, 'adv': 1.0}
        beta: 平滑因子
        """
        self.weights = init_weights
        self.loss_ema = {k: 0.0 for k in init_weights.keys()}
        self.beta = beta
        # 为不同损失类型设置特定的权重范围
        self.weight_ranges = {
            'rec': {'min': 50.0, 'max': 500.0},    # 重建损失是核心
            'spec': {'min': 0.005, 'max': 0.02},    # 频谱损失范围收窄
            'adv': {'min': 15.0, 'max': 25.0},      # 对抗损失保持适度范围
            'gp': {'min': 8.0, 'max': 12.0},        # 梯度惩罚小范围浮动
            'rel': {'min': 0.2, 'max': 0.8}         # 相对误差损失小范围浮动
        }
        # 调整权重变化率
        self.increase_rate = 1.02    # 更温和的增长率
        self.decrease_rate = 0.98    # 更温和的衰减率
    
    def update(self, current_losses):
        # 更新损失的EMA
        for k, v in current_losses.items():
            if k in self.loss_ema:
                self.loss_ema[k] = self.beta * self.loss_ema[k] + (1 - self.beta) * abs(v)
        
        # 计算损失的相对大小
        total_loss = sum(self.loss_ema.values())
        if total_loss > 0:
            loss_ratios = {k: v/total_loss for k, v in self.loss_ema.items()}
            
            # 动态调整权重
            for k in self.weights.keys():
                weight_range = self.weight_ranges[k]
                if loss_ratios[k] > 0.5:  # 损失占比过大，降低权重
                    self.weights[k] = max(
                        weight_range['min'],
                        min(weight_range['max'],
                            self.weights[k] * self.decrease_rate)
                    )
                elif loss_ratios[k] < 0.1:  # 损失占比过小，提高权重
                    self.weights[k] = max(
                        weight_range['min'],
                        min(weight_range['max'],
                            self.weights[k] * self.increase_rate)
                    )
        
        return self.weights
    
    def get_weights(self):
        return self.weights

class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, scales=[64, 128, 256, 512]):
        super().__init__()
        self.scales = scales
    
    def forward(self, real, fake):
        total_loss = 0
        batch_size, channels, length = real.shape
        
        for scale in self.scales:
            # 计算FFT
            real_fft = torch.fft.rfft(real, n=scale, dim=2)
            fake_fft = torch.fft.rfft(fake, n=scale, dim=2)
            
            # 计算幅度谱
            real_mag = torch.abs(real_fft)
            fake_mag = torch.abs(fake_fft)
            
            # 频谱损失
            spec_loss = F.l1_loss(real_mag, fake_mag)
            total_loss += spec_loss
            
        return total_loss / len(self.scales)