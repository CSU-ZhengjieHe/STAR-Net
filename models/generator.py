import torch
import torch.nn as nn
from .components import PositionalEncoding

class TransformerGenerator(nn.Module):
    def __init__(self, sequence_length=512, input_dim=5, d_model=128, num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # 将特征映射到d_model维度
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 添加掩码嵌入层
        self.mask_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, sequence_length, dropout)
        
        # 添加一个标志来跟踪注意力权重
        self.return_attention = False
        self.attention_weights = []
        
        # Transformer编码器层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=5*d_model,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers+1)
        ])
        
        # Transformer解码器层
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=5*d_model,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers+1)
        ])
        
        # 输出映射
        self.output_projection = nn.Linear(d_model, input_dim)
    
    def forward(self, x, mask=None, return_attention=False):
        # x shape: [batch_size, features, sequence_length]
        # 转换维度顺序以适应Transformer
        x = x.transpose(1, 2)  # [batch_size, sequence_length, features]
        
        # 特征映射
        x = self.input_projection(x)  # [batch_size, sequence_length, d_model]
        
        # 如果提供了掩码，创建掩码嵌入
        if mask is not None:
            # 转置掩码与x保持一致的维度
            mask_t = mask.transpose(1, 2)  # [batch_size, sequence_length, features]
            
            # 创建掩码指示器嵌入 (0表示缺失，1表示有效)
            mask_embedding = self.mask_embedding(mask_t)
            
            # 结合特征和掩码信息
            x = x + mask_embedding
        
        # 位置编码
        x = self.positional_encoding(x)
        
        # 重置注意力权重列表
        self.attention_weights = []
        
        # 编码器
        encoder_output = x
        for i, encoder_layer in enumerate(self.encoder_layers):
            # 分别获取self_attn的参数以直接调用
            if return_attention:
                # 直接调用多头注意力的forward来获取注意力权重
                src = encoder_layer.norm1(encoder_output)
                q = k = v = src
                
                # 直接调用self_attn并获取注意力权重
                attn_output, attn_weights = encoder_layer.self_attn(
                    q, k, v, 
                    attn_mask=None,
                    key_padding_mask=None,
                    need_weights=True,
                    average_attn_weights=False
                )
                
                # 保存注意力权重
                self.attention_weights.append(attn_weights.detach())
                
                # 继续处理
                encoder_output = encoder_output + encoder_layer.dropout1(attn_output)
                encoder_output = encoder_layer.norm2(encoder_output)
                
                # FFN部分
                ff_output = encoder_layer.linear2(
                    encoder_layer.dropout(
                        encoder_layer.activation(
                            encoder_layer.linear1(encoder_output)
                        )
                    )
                )
                encoder_output = encoder_output + encoder_layer.dropout2(ff_output)
            else:
                # 正常forward
                encoder_output = encoder_layer(encoder_output)
        
        # 解码器
        decoder_output = x
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output,
                encoder_output,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=mask if mask is not None else None,
                memory_key_padding_mask=mask if mask is not None else None
            )
        
        # 输出映射
        output = self.output_projection(decoder_output)  # [batch_size, sequence_length, features]
        
        # 转换回原始维度顺序
        output = output.transpose(1, 2)  # [batch_size, features, sequence_length]
        
        return output
        
    def load_best_model(self, model_path, device=None):
        """加载最佳模型权重"""
        try:
            if device is None:
                device = next(self.parameters()).device
                
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'generator_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['generator_state_dict'])
            else:
                self.load_state_dict(checkpoint)
                
            self.eval()
            
            print(f"成功加载模型权重从: {model_path}")
            
            if 'epoch' in checkpoint:
                print(f"模型保存时的训练轮次: {checkpoint['epoch']}")
            if 'best_val_loss' in checkpoint:
                print(f"最佳验证损失: {checkpoint['best_val_loss']:.6f}")
                
            return self
            
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            raise