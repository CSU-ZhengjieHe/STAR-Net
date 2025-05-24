import torch
import torch.manual_seed as manual_seed

from config import Config
from models import TransformerGenerator, FrequencyDiscriminator
from data.dataloader import load_data
from utils.trainer import Trainer

def main():
    # 设置随机种子
    manual_seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载配置
    config = Config()
    print("配置加载完成")
    print(f"动态掩码训练: {config.dynamic_masking}")
    print(f"判别器更新频率: {config.n_critic}")
    
    # 加载数据
    train_loader, val_loader = load_data(config)
    print("数据加载完成")
    
    # 初始化模型
    generator = TransformerGenerator(
        sequence_length=config.sequence_length,
        input_dim=config.input_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    discriminator = FrequencyDiscriminator(
        sequence_length=config.sequence_length,
        hidden_dim=config.d_model
    ).to(device)
    
    print("模型初始化完成")
    print(f"生成器参数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数量: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 初始化训练器
    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    print("训练器初始化完成")
    
    # 开始训练
    print("开始训练...")
    print("="*50)
    print("使用动态掩码机制进行训练")
    print("训练过程中将随机生成各种缺失模式组合")
    print("="*50)
    
    metrics = trainer.train()
    
    print("\n训练完成!")
    print(f"最佳验证损失: {trainer.best_val_loss:.6f}")
    print(f"总训练时间: {trainer.total_training_time:.2f} 秒")
    
    print("\n运行 'python experiments/evaluate.py' 进行模型评估")

if __name__ == '__main__':
    main()