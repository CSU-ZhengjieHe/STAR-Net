import torch
import os
from utils.masking import TestScenarioMasks
from config import Config
from models import TransformerGenerator
from data.dataloader import load_test_data
from utils.visualizer import FeatureVisualizer

def evaluate_on_test_scenarios(model, test_loader, config, device):
    """在所有测试场景上评估模型"""
    model.eval()
    results = {}
    
    with torch.no_grad():
        for scenario_name, scenario_config in config.test_scenarios.items():
            print(f"评估场景 {scenario_name}: {scenario_config['description']}")
            
            scenario_metrics = []
            for batch in test_loader:
                real_data = batch[0].to(device)
                
                # 创建固定场景掩码
                mask = TestScenarioMasks.create_scenario_mask(real_data, scenario_config)
                mask = mask.to(device)
                
                # 应用掩码
                masked_data = real_data * mask
                
                # 生成恢复数据
                generated_data = model(masked_data, mask)
                
                # 计算指标（只在缺失部分）
                missing_mask = 1 - mask
                mse = torch.mean((generated_data - real_data) ** 2 * missing_mask).item()
                mae = torch.mean(torch.abs(generated_data - real_data) * missing_mask).item()
                
                scenario_metrics.append({'mse': mse, 'mae': mae})
            
            # 计算平均指标
            avg_mse = sum(m['mse'] for m in scenario_metrics) / len(scenario_metrics)
            avg_mae = sum(m['mae'] for m in scenario_metrics) / len(scenario_metrics)
            
            results[scenario_name] = {
                'mse': avg_mse,
                'mae': avg_mae,
                'description': scenario_config['description']
            }
            
            print(f"  MSE: {avg_mse:.6f}, MAE: {avg_mae:.6f}")
    
    return results

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载配置
    config = Config()
    
    # 加载最佳模型
    best_model_path = 'checkpoints/best_model.pt'
    
    generator = TransformerGenerator(
        sequence_length=config.sequence_length,
        input_dim=5,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    generator.load_best_model(best_model_path)
    
    # 加载测试数据
    test_loader = load_test_data(config)
    
    # 创建可视化器实例
    visualizer = FeatureVisualizer(generator, device)
    
    # 生成预测结果
    predictions, real_data = visualizer.generate_and_compare(test_loader)
    
    # 基础评估指标
    metrics = visualizer.calculate_metrics(real_data, predictions)
    print("\n模型评估指标:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")
    
    # 在10种测试场景上评估
    print("\n在测试场景上评估:")
    scenario_results = evaluate_on_test_scenarios(generator, test_loader, config, device)
    
    # 可视化分析
    print("\n生成可视化结果...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存结果
    visualizer.save_results(real_data, predictions, save_dir=results_dir)
    
    # 生成各种图表
    visualizer.plot_comparison(real_data, predictions, save_path=f'{results_dir}/comparison.png')
    
    # 可视化注意力图
    sample_data = real_data[:5]
    visualizer.visualize_attention_map(
        sample_data,
        sample_idx=0,
        save_path=f"{results_dir}/attention_map"
    )
    
    print(f"\n评估完成！结果已保存到: {os.path.abspath(results_dir)}")

if __name__ == '__main__':
    main()