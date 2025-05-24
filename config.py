class Config:
    def __init__(self):
        # 数据配置
        self.sequence_length = 512
        self.batch_size = 32
        self.num_workers = 4
        
        # 模型配置
        self.d_model = 256
        self.num_heads = 4
        self.num_layers = 3
        self.dropout = 0.25
        self.input_dim = 5
        
        # 训练配置
        self.n_critic = 5            # 修改为5，符合论文描述
        self.lambda_rec = 50.0       # 重构损失权重
        self.lambda_spec = 5.0       # 频谱损失权重
        self.lambda_gp = 10.0        # 梯度惩罚权重
        self.lambda_adv = 20         # 对抗损失权重
        self.lambda_rel = 0.5        # 相对误差损失权重
        self.lr_g = 5e-5            # 生成器学习率
        self.lr_d = 5e-5            # 判别器学习率
        self.beta1 = 0.5            # Adam优化器参数
        self.beta2 = 0.999
        self.weight_decay = 1e-5
        self.num_epochs = 300        # 训练轮数
        
        # 自适应权重参数
        self.weight_beta = 0.99  # EMA平滑因子
        
        # 🆕 动态掩码配置 - 替换原来的固定工况
        self.dynamic_masking = True  # 启用动态掩码
        
        # 动态掩码参数范围
        self.random_missing_range = (0.1, 0.3)      # 随机缺失率范围 10%-30%
        self.continuous_missing_range = (0.2, 0.4)  # 连续缺失率范围 20%-40%
        self.max_missing_channels = 2               # 最大缺失通道数
        self.mixed_missing_prob = 0.4               # 混合缺失概率
        self.channel_missing_prob = 0.3             # 通道缺失概率
        
        # 🆕 测试场景配置 - 仅用于评估
        self.test_scenarios = {
            'A1': {'type': 'random', 'ratio': 0.1, 'description': '10%随机缺失'},
            'A2': {'type': 'random', 'ratio': 0.2, 'description': '20%随机缺失'},
            'A3': {'type': 'random', 'ratio': 0.3, 'description': '30%随机缺失'},
            'B1': {'type': 'continuous', 'ratio': 0.2, 'description': '20%连续缺失'},
            'B2': {'type': 'continuous', 'ratio': 0.3, 'description': '30%连续缺失'},
            'B3': {'type': 'continuous', 'ratio': 0.4, 'description': '40%连续缺失'},
            'C1': {'type': 'channel', 'channels': [1], 'description': '单通道失效'},
            'C2': {'type': 'channel', 'channels': [1, 2], 'description': '双通道失效'},
            'D1': {'type': 'mixed', 'random_ratio': 0.1, 'continuous_ratio': 0.05, 
                   'channels': [1], 'description': '混合缺失模式1'},
            'D2': {'type': 'mixed', 'random_ratio': 0.2, 'continuous_ratio': 0.1, 
                   'channels': [1, 2], 'description': '混合缺失模式2'}
        }
        
        # 其他配置保持不变...
        self.lr_decay_start = 10
        self.lr_decay_every = 5
        self.lr_decay_rate = 0.5
        self.save_interval = 10
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        self.warmup_epochs = 5
        self.label_smoothing = 0.1
        self.gradient_clip = 1.0
        self.scheduler_patience = 5
        self.min_lr = 1e-6 