import torch
import numpy as np

class DynamicMaskGenerator:
    """动态掩码生成器 - 用于训练"""
    
    def __init__(self, config):
        self.config = config
    
    def create_dynamic_mask(self, data):
        """
        创建完全动态的缺失掩码
        
        Args:
            data: 输入数据，形状为 [batch_size, n_features, seq_length]
        
        Returns:
            mask: 缺失掩码，1表示保留，0表示缺失
        """
        batch_size, n_features, seq_length = data.shape
        mask = torch.ones_like(data)
        
        # 为每个批次样本动态选择缺失模式
        for i in range(batch_size):
            sample_mask = torch.ones(n_features, seq_length, device=data.device)
            
            # 随机决定使用哪些缺失模式的组合
            missing_types = []
            
            # 1. 随机缺失 (概率50%)
            if torch.rand(1).item() > 0.5:
                missing_types.append('random')
                
            # 2. 连续缺失 (概率40%) 
            if torch.rand(1).item() > 0.6:
                missing_types.append('continuous')
                
            # 3. 通道缺失 (概率30%)
            if torch.rand(1).item() > 0.7:
                missing_types.append('channel')
            
            # 如果没有选中任何类型，至少使用随机缺失
            if not missing_types:
                missing_types = ['random']
            
            # 应用选中的缺失类型
            for missing_type in missing_types:
                if missing_type == 'random':
                    sample_mask = self._apply_random_missing(sample_mask, seq_length)
                elif missing_type == 'continuous':
                    sample_mask = self._apply_continuous_missing(sample_mask, seq_length)
                elif missing_type == 'channel':
                    sample_mask = self._apply_channel_missing(sample_mask, n_features, seq_length)
            
            mask[i] = sample_mask
        
        return mask
    
    def _apply_random_missing(self, mask, seq_length):
        """应用随机缺失"""
        min_ratio, max_ratio = self.config.random_missing_range
        random_ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
        random_mask = torch.bernoulli(torch.ones_like(mask) * (1 - random_ratio))
        return mask * random_mask
    
    def _apply_continuous_missing(self, mask, seq_length):
        """应用连续缺失"""
        min_ratio, max_ratio = self.config.continuous_missing_range  
        continuous_ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
        continuous_length = int(seq_length * continuous_ratio)
        
        if continuous_length > 0:
            start_idx = torch.randint(0, seq_length - continuous_length + 1, (1,)).item()
            mask[:, start_idx:start_idx + continuous_length] = 0
        
        return mask
    
    def _apply_channel_missing(self, mask, n_features, seq_length):
        """应用通道缺失"""
        max_missing = min(self.config.max_missing_channels, n_features - 1)
        num_channels_missing = torch.randint(1, max_missing + 1, (1,)).item()
        missing_channels = torch.randperm(n_features)[:num_channels_missing]
        
        for channel in missing_channels:
            # 随机决定是整个通道缺失还是部分缺失
            if torch.rand(1).item() > 0.5:
                # 整个通道缺失
                mask[channel, :] = 0
            else:
                # 部分通道缺失
                partial_ratio = torch.rand(1).item() * 0.6
                partial_mask = torch.bernoulli(
                    torch.ones(seq_length, device=mask.device) * (1 - partial_ratio)
                )
                mask[channel, :] *= partial_mask
        
        return mask

class TestScenarioMasks:
    """固定测试场景 - 仅用于评估"""
    
    @staticmethod
    def create_scenario_mask(data, scenario_config):
        """根据场景配置创建固定掩码"""
        batch_size, n_features, seq_length = data.shape
        mask = torch.ones_like(data)
        
        if scenario_config['type'] == 'random':
            ratio = scenario_config['ratio']
            for i in range(batch_size):
                random_mask = torch.bernoulli(torch.ones(n_features, seq_length) * (1 - ratio))
                mask[i] = random_mask
                
        elif scenario_config['type'] == 'continuous':
            ratio = scenario_config['ratio']
            continuous_length = int(seq_length * ratio)
            for i in range(batch_size):
                start_idx = torch.randint(0, seq_length - continuous_length + 1, (1,)).item()
                mask[i, :, start_idx:start_idx + continuous_length] = 0
                
        elif scenario_config['type'] == 'channel':
            channels = scenario_config['channels']
            for i in range(batch_size):
                for channel in channels:
                    if channel < n_features:
                        mask[i, channel, :] = 0
                        
        elif scenario_config['type'] == 'mixed':
            # 实现混合缺失模式
            for i in range(batch_size):
                sample_mask = torch.ones(n_features, seq_length)
                
                # 应用随机缺失
                random_ratio = scenario_config.get('random_ratio', 0.1)
                random_mask = torch.bernoulli(torch.ones_like(sample_mask) * (1 - random_ratio))
                sample_mask *= random_mask
                
                # 应用连续缺失
                continuous_ratio = scenario_config.get('continuous_ratio', 0.05)
                continuous_length = int(seq_length * continuous_ratio)
                if continuous_length > 0:
                    start_idx = torch.randint(0, seq_length - continuous_length + 1, (1,)).item()
                    sample_mask[:, start_idx:start_idx + continuous_length] = 0
                
                # 应用通道缺失
                channels = scenario_config.get('channels', [])
                for channel in channels:
                    if channel < n_features:
                        sample_mask[channel, :] = 0
                
                mask[i] = sample_mask
        
        return mask