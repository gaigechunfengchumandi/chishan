import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/xingyulu/Public/physionet')
sys.path.append('/Users/xingyulu/Public/physionet/utils/fsst_convert')
from models.seg_model_cnn_lstm import VFSegmentationModel

from utils.fsst_convert.time2fsst_cls import time2fsst_without_label 

def load_signal_data(data_dir):
    """加载信号数据"""
    signals = []
    file_names = []
    
    # 获取所有npy文件
    files = sorted(Path(data_dir).glob('*.npy'))
    
    for file_path in files:
        signal = np.load(file_path)
        signals.append(signal)
        file_names.append(file_path.stem)
    
    return signals, file_names

def predict_signal(model, signal, device='cuda', window_size=2500):
    """对单个信号进行预测"""
    model.eval()
    
    # 转换为张量
    signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device) # shape: (2500,) or (40,2500)
    
    with torch.no_grad():
        # 添加批次维度
        signal_tensor = signal_tensor.unsqueeze(0)
        # 前向传播
        output = model(signal_tensor).squeeze(-1)
        # 获取预测结果
        predictions = (output > 0.5).float().cpu().numpy()
    
    return predictions[0]

def visualize_and_save(signal, prediction, file_name, save_dir):
    """可视化信号和预测结果并保存"""
    plt.figure(figsize=(15, 8))
    
    # 绘制原始信号
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 绘制预测结果
    plt.subplot(2, 1, 2)
    plt.plot(prediction)
    plt.title('VF Prediction (1 = VF, 0 = Non-VF)')
    plt.xlabel('Sample')
    plt.ylabel('Prediction')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{file_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 配置参数
    model_path = '/Users/xingyulu/Public/physionet/models/saved/vf_segmentation_best_fsst.pth'
    data_dir = '/Users/xingyulu/Public/监护心电预警/监护部门提供数据/室颤/86_10s/processed_data'
    output_dir = '/Users/xingyulu/Public/监护心电预警/监护部门提供数据/室颤/86_10s/fsst_inference_results'
    data_mode = 'fsst'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = VFSegmentationModel(mode=data_mode, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 加载数据
    signals, file_names = load_signal_data(data_dir)
    print(f'加载了 {len(signals)} 个信号文件')
    
    # 对每个信号进行预测和可视化
    for signal, file_name in zip(signals, file_names):
        print(f'处理文件: {file_name}')

        # 转换为FSST特征
        if data_mode == 'fsst':
            fsst_signal = time2fsst_without_label(signal) # shape: (40, 2500)
            prediction = predict_signal(model, fsst_signal, device)
        else:
            prediction = predict_signal(model, signal, device)
        
        # 可视化和保存结果
        visualize_and_save(signal, prediction, file_name, output_dir)
        
        # 保存预测结果
        # np.save(os.path.join(output_dir, f'{file_name}_pred.npy'), prediction)
    
    print('推理完成！')
    print(f'结果保存在: {output_dir}')

if __name__ == '__main__':
    main()