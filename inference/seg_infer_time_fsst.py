import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/xingyulu/Public/physionet')
sys.path.append('/Users/xingyulu/Public/physionet/utils/fsst_convert')
from models.seg_model_cnn_lstm import AFSegmentationModel

from utils.fsst_convert.time2fsst_cls import time2fsst_without_label 


# 房颤的推理代码和室颤的代码不同，室颤的数据事没有标签的，而目前所有房颤的代码都有标签


def predict_signal(model, signal, device='cuda', window_size=2500):
    """对单个信号进行预测"""
    model.eval()

    # 标准化处理（与训练时保持一致）
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    signal = np.clip(signal, -5.0, 5.0) / 5.0  # 归一化到[-1, 1]
    
    # 转换为张量并调整维度
    signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)
    signal_tensor = signal_tensor.unsqueeze(0)  # 添加batch和channel维度 [1, 2500]
    # 验证一下形状
    # print(f'signal_tensor shape: {signal_tensor.shape}', '正确的shape应该是[1, 2500]')
    
    with torch.no_grad():
        # 前向传播
        output = model(signal_tensor) # shape: [1, 2500, 5]
        # 获取预测结果
        _, predictions = torch.max(output, dim=-1)  # predictions shape: [1, 2500]
        predictions = predictions.cpu().numpy()
    
    return predictions[0]

# 定义一个函数，可视化信号数据和标签并保存图像
def visualize_and_save_signal(signal, prediction, file_name, picture_path):
    """
    可视化信号数据和预测结果并保存图像
    
    参数:
        signal: 信号数据
        prediction: 预测结果
        file_name: 文件名
        picture_path: 图片保存路径
    """
    try:
        plt.figure(figsize=(20, 6))
        
        # 绘制信号数据
        plt.plot(signal, 'b-', alpha=0.7, label='Signal')
        
        # 定义标签对应的颜色
        label_colors = {
            0: 'lightgreen',   # N类型
            1: 'skyblue',      # 新增类型
            2: 'lightcoral',   # V类型
            3: 'moccasin',     # 其他类型
            4: 'plum'          # af类型
        }
        
        # 为每个样本点上色（使用透明度较低的背景色）
        for i in range(len(prediction)):
            label = int(prediction[i])
            color = label_colors.get(label, 'gray')
            plt.axvspan(i, i+1, alpha=0.3, color=color)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.5, label='N Type (0)'),
            Patch(facecolor='skyblue', alpha=0.5, label='Type (1)'),
            Patch(facecolor='lightcoral', alpha=0.5, label='V Type (2)'),
            Patch(facecolor='moccasin', alpha=0.5, label='Other Type (3)'),
            Patch(facecolor='plum', alpha=0.5, label='AF Type (4)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f"ECG Signal with Prediction: {file_name}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(picture_path, f"{file_name}.png")
        plt.savefig(save_path)
        print(f"图像已保存到: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")


def main():
    # 配置参数
    model_path = '/Users/xingyulu/Public/physionet/models/saved/af_segmentation_best_time1.pth'
    data_dir = '/Users/xingyulu/Public/afafaf/用于尝试训练/test'
    output_picture_dir = '/Users/xingyulu/Public/afafaf/推理尝试/picture'
    data_mode = 'time' 
    
    # 创建输出目录
    os.makedirs(output_picture_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = AFSegmentationModel(mode=data_mode, hidden_size=64, num_layers=2, dropout=0.3)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 加载数据
    files = sorted(Path(data_dir).glob('*.npy'))
    print(f'找到 {len(files)} 个信号文件')
    
    # 对每个信号文件进行预测和可视化
    for file_path in files:
        file_name = file_path.stem
        print(f'处理文件: {file_name}')
        try:
            # 加载单个文件
            data = np.load(file_path)
            signal = data[:, 0]  # 取第一列作为信号数据
            
            signal = np.squeeze(signal)  # 确保信号是1D的
            # 转换为FSST特征
            if data_mode == 'fsst':
                fsst_signal = time2fsst_without_label(signal)  # shape: (40, 2500)
                prediction = predict_signal(model, fsst_signal, device)
            else:
                prediction = predict_signal(model, signal, device) #prediction（2500，）
            
            # 可视化和保存结果
            visualize_and_save_signal(signal, prediction, file_name, output_picture_dir)
            
            # 保存预测结果
            # np.save(os.path.join(output_dir, f'{file_name}_pred.npy'), prediction)
        
            # 手动清理
            del signal, prediction
            gc.collect()  # 强制垃圾回收
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            continue
    print('推理完成！')
    print(f'结果保存在: {output_picture_dir}')

if __name__ == '__main__':
    main()