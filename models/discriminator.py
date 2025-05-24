import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import PositionalEncoding

class FrequencyDiscriminator(nn.Module):
    def __init__(self, sequence_length=512, hidden_dim=256):
        super().__init__()
        
        # 自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, sequence_length)
        
        # 特征提取层
        self.feature_proj = nn.Linear(5, hidden_dim)
        
        # 频域特征提取
        self.freq_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 时域特征提取
        self.time_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: [batch_size, 5, sequence_length]
        batch_size = x.size(0)
        
        # 转换维度并进行特征投影
        x = x.transpose(1, 2)  # [batch_size, sequence_length, 5]
        x = self.feature_proj(x)  # [batch_size, sequence_length, hidden_dim]
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # 自注意力机制
        attn_output, _ = self.self_attention(x, x, x)
        
        # 转回原始维度顺序用于卷积
        x = attn_output.transpose(1, 2)  # [batch_size, hidden_dim, sequence_length]
        
        # 频域特征
        fft = torch.fft.fft(x, dim=-1).real
        freq_features = self.freq_conv(fft)
        freq_features = F.adaptive_avg_pool1d(freq_features, 1).squeeze(-1)
        
        # 时域特征
        time_features = self.time_conv(x)
        time_features = F.adaptive_avg_pool1d(time_features, 1).squeeze(-1)
        
        # 特征融合
        combined = torch.cat([freq_features, time_features], dim=1)
        output = self.fusion(combined)
        return output