import torch
from torch.utils.data import DataLoader, TensorDataset
from joblib import load

def load_data(config):
    """加载训练和验证数据"""
    
    # 加载数据
    train_features = load('processed_data/train_feature.joblib')
    val_features = load('processed_data/val_feature.joblib')
    
    print("训练数据维度:", train_features.shape)
    print("验证数据维度:", val_features.shape)
    
    # 转换为PyTorch张量
    train_features = torch.from_numpy(train_features).float()
    val_features = torch.from_numpy(val_features).float()
    
    # 转换维度顺序：[batch, sequence_length, features] -> [batch, features, sequence_length]
    train_features = train_features.permute(0, 2, 1)
    val_features = val_features.permute(0, 2, 1)
    
    # 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(train_features),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TensorDataset(val_features),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def load_test_data(config):
    """加载测试数据"""
    test_features = load('processed_data/test_features.joblib')
    test_features = torch.from_numpy(test_features).float()
    test_features = test_features.permute(0, 2, 1)
    
    test_loader = DataLoader(
        TensorDataset(test_features),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return test_loader