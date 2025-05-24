import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class FeatureVisualizer:
    def __init__(self, generator, device):
        self.generator = generator
        self.device = device
        self.generator.eval()
    
    def _ensure_save_dir(self, save_path):
        """确保保存目录存在"""
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
    
    def save_results(self, real_data, predictions, save_dir='matlab_results'):
        """保存完整的结果数据"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # 确保数据转换为numpy数组
            real_data_np = real_data.cpu().numpy() if torch.is_tensor(real_data) else real_data
            predictions_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
            
            # 1. 基础数据
            results_dict = {
                'real_data': real_data_np,
                'predictions': predictions_np,
                'errors': predictions_np - real_data_np,
                'time_steps': np.arange(real_data_np.shape[-1])
            }
            
            # 2. 频域分析数据
            for i in range(real_data_np.shape[1]):  # 对每个通道
                # FFT 分析
                real_fft = np.fft.fft(real_data_np[:, i, :], axis=1)
                pred_fft = np.fft.fft(predictions_np[:, i, :], axis=1)
                freq = np.fft.fftfreq(real_data_np.shape[-1])
                
                results_dict.update({
                    f'channel_{i}_real_fft': real_fft,
                    f'channel_{i}_pred_fft': pred_fft,
                    f'channel_{i}_freq': freq
                })
            
            # 3. 统计指标
            for i in range(real_data_np.shape[1]):
                channel_metrics = {
                    f'channel_{i}_mse': mean_squared_error(real_data_np[:, i, :].flatten(), 
                                                        predictions_np[:, i, :].flatten()),
                    f'channel_{i}_mae': mean_absolute_error(real_data_np[:, i, :].flatten(), 
                                                        predictions_np[:, i, :].flatten()),
                    f'channel_{i}_rmse': np.sqrt(mean_squared_error(real_data_np[:, i, :].flatten(), 
                                                                predictions_np[:, i, :].flatten()))
                }
                results_dict.update(channel_metrics)
            
            # 保存文件
            sio.savemat(os.path.join(save_dir, 'gan_results.mat'), results_dict)
            
            print(f"数据已保存到: {save_dir}/gan_results.mat")
            return results_dict
            
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
            raise
    
    def generate_and_compare(self, test_loader):
        """生成预测结果并与真实数据对比"""
        all_predictions = []
        all_real = []
        
        with torch.no_grad():
            for batch in test_loader:
                real_data = batch[0].to(self.device)
                predictions = self.generator(real_data)
                all_predictions.append(predictions.cpu())
                all_real.append(real_data.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        real_data = torch.cat(all_real, dim=0)
        
        return predictions, real_data
    
    def visualize_attention_map(self, input_data, sample_idx=0, layer_idx=None, head_idx=None, save_path=None, save_mat=True):
        """可视化注意力图"""
        self.generator.eval()
        
        # 确保输入是张量并在正确的设备上
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.FloatTensor(input_data).to(self.device)
        else:
            input_data = input_data.to(self.device)
        
        # 选择单个样本
        if input_data.dim() == 3:
            single_sample = input_data[sample_idx:sample_idx+1]
        else:
            single_sample = input_data
            
        # 重置并获取注意力权重
        self.generator.return_attention = True
        self.generator.attention_weights = []
        
        with torch.no_grad():
            output = self.generator(single_sample, return_attention=True)
            attention_weights = self.generator.attention_weights
            
            # 添加调试信息
            print(f"收集到的注意力层数: {len(attention_weights)}")
            if len(attention_weights) > 0:
                print(f"第一层形状: {attention_weights[0].shape}")
                print(f"最大值: {attention_weights[0].max().item():.6f}, 最小值: {attention_weights[0].min().item():.6f}")
            else:
                print("警告: 未收集到任何注意力权重!")
                return {}
        
        # 准备要保存的数据
        attention_data = {}
        
        # 可视化注意力图
        if layer_idx is None:
            layers_to_plot = range(len(attention_weights))
        else:
            layers_to_plot = [layer_idx]
        
        plt.figure(figsize=(15, 10))
        
        for i, layer_i in enumerate(layers_to_plot):
            layer_attention = attention_weights[layer_i]  # [batch, num_heads, seq_len, seq_len]
            
            # 保存原始注意力数据
            attention_data[f'layer_{layer_i}_attention'] = layer_attention.cpu().numpy()
            
            if head_idx is None:
                heads_to_plot = range(layer_attention.size(1))
            else:
                heads_to_plot = [head_idx]
            
            num_heads = len(heads_to_plot)
            fig_cols = min(4, num_heads)
            fig_rows = (num_heads + fig_cols - 1) // fig_cols
            
            for j, head_j in enumerate(heads_to_plot):
                plt.subplot(fig_rows, fig_cols, j + 1)
                
                # 提取该头的注意力权重
                attn = layer_attention[0, head_j].cpu().numpy()  # [seq_len, seq_len]
                
                # 绘制热力图
                im = plt.imshow(attn, cmap='viridis')
                plt.title(f'层 {layer_i+1}, 头 {head_j+1}')
                plt.xlabel('序列位置 (目标)')
                plt.ylabel('序列位置 (查询)')
                plt.colorbar(im)
        
        plt.tight_layout()
        
        if save_path:
            # 确保保存目录存在
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_path}_attention_maps.png", dpi=300, bbox_inches='tight')
        
        # 保存.mat文件
        if save_mat:
            mat_dir = os.path.dirname(save_path) if save_path else 'results'
            os.makedirs(mat_dir, exist_ok=True)
            mat_path = f"{save_path if save_path else os.path.join(mat_dir, 'attention_data')}.mat"
            
            # 处理数据以确保可以保存为.mat格式
            processed_data = {}
            for key, value in attention_data.items():
                processed_data[key] = value
            
            sio.savemat(mat_path, processed_data)
            print(f"注意力数据已保存到: {mat_path}")
        
        plt.show()
        return attention_data
    
    def calculate_metrics(self, real_data, predictions):
        """计算评估指标"""
        mse = mean_squared_error(real_data.numpy().reshape(-1), predictions.numpy().reshape(-1))
        mae = mean_absolute_error(real_data.numpy().reshape(-1), predictions.numpy().reshape(-1))
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def plot_comparison(self, real_data, predictions, sample_idx=0, save_path=None):
        """绘制时间序列对比图"""
        plt.figure(figsize=(15, 5))
        
        # 1. 时间序列对比
        plt.subplot(131)
        plt.plot(real_data[sample_idx, 0, :].numpy(), label='真实值', linewidth=2, alpha=0.7)
        plt.plot(predictions[sample_idx, 0, :].numpy(), label='生成值', linewidth=2, alpha=0.7)
        plt.title('时间序列对比', fontsize=14)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('幅值', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        # 2. 误差分布
        plt.subplot(132)
        errors = (predictions - real_data).numpy().reshape(-1)
        plt.hist(errors, bins=50, density=True, alpha=0.7)
        plt.title('误差分布', fontsize=14)
        plt.xlabel('误差', fontsize=12)
        plt.ylabel('密度', fontsize=12)
        plt.grid(True)
        
        # 3. 真实值vs预测值散点图
        plt.subplot(133)
        plt.scatter(real_data.numpy().reshape(-1), 
                   predictions.numpy().reshape(-1), 
                   alpha=0.1, s=1)
        plt.plot([-1, 1], [-1, 1], 'r--', label='理想线')
        plt.title('预测值 vs 真实值', fontsize=14)
        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()