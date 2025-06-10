import torch
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/devel/code/monitor/physionet')
sys.path.append('/home/devel/code/monitor/physionet/utils/fsst_convert')
from models.seg_transformer128_wav2vec import TransformerSegmentation128

import torch.nn.functional as F

# from utils.fsst_convert.time2fsst_cls import time2fsst_without_label 


# 房颤的推理代码和室颤的代码不同，室颤的数据事没有标签的，而目前所有房颤的代码都有标签


def predict_signal(model, signal, device='cuda'):
    """对单个信号进行预测"""
    model.eval()

    # 标准化处理（与训练时保持一致）
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    signal = np.clip(signal, -5.0, 5.0) / 5.0  # 归一化到[-1, 1]
    
    # 转换为张量并调整维度
    signal_tensor = torch.tensor(signal, dtype=torch.float32).to(device)
    signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # 添加batch和channel维度 [1, 1, 1280] inorder to match the Unet's input shape
    # 验证一下形状
    # print(f'signal_tensor shape: {signal_tensor.shape}', '正确的shape应该是[1, 1, 1280]')
    
    with torch.no_grad():
        # 前向传播
        output = model(signal_tensor)
        output = output.squeeze(0) #output shape [5, 10]
    return output.cpu().numpy()

# 定义一个函数，可视化信号数据和标签并保存图像
def visualize_and_save_signal(signal, prediction, feature, file_name, picture_path):
    
    matplotlib.use('Agg')  # 使用更快的后端
    plt.ioff()  # 关闭交互模式
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 10))
    ax1.plot(signal, 'b-', alpha=0.7, label='Signal')
    label_colors = {
        0: 'lightgreen',
        1: 'skyblue',
        2: 'lightcoral',
        3: 'moccasin',
        4: 'plum',
        5: 'purple'
    }
    # 合并连续相同标签区间，减少axvspan调用
    last_label = int(prediction[0])
    start_idx = 0
    for i in range(1, len(prediction)):
        curr_label = int(prediction[i])
        if curr_label != last_label:
            color = label_colors.get(last_label, 'gray')
            ax1.axvspan(start_idx, i, alpha=0.3, color=color)
            start_idx = i
            last_label = curr_label
    color = label_colors.get(last_label, 'gray')
    ax1.axvspan(start_idx, len(prediction), alpha=0.3, color=color)
    # legend只添加一次
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.5, label='N Type (0)'),
        Patch(facecolor='skyblue', alpha=0.5, label='S Type (1)'),
        Patch(facecolor='lightcoral', alpha=0.5, label='V Type (2)'),
        Patch(facecolor='moccasin', alpha=0.5, label='X Type (3)'),
        Patch(facecolor='plum', alpha=0.5, label='AF Type (4)'),
        Patch(facecolor='purple', alpha=0.5, label='VF Type (5)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=14, frameon=False)
    # 竖线减少alpha和linewidth
    for x in range(0, len(signal), 128):
        ax1.axvline(x=x, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.set_title(f"ECG Signal with Prediction: {file_name}")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(False)
    if feature.shape[0] == 1:
        feature = feature[0]
    channels = feature.shape[0]
    prob_feature = F.softmax(torch.tensor(feature), dim=0).numpy()
    img = ax2.imshow(prob_feature, cmap='Reds', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
    ax2.grid(False)
    plt.xticks([])
    plt.yticks(np.arange(0.5, channels+0.5, 1),[f'Ch{i}' for i in range(channels)])
    plt.colorbar(img, ax=ax2, label='Probability')
    save_path = os.path.join(picture_path, f"{file_name}.png")
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def calculate_heartbeat_metrics(prediction, labels):
    # 初始化真正例计数
    true_positives = 0
    # 初始化假正例计数
    false_positives = 0
    # 初始化假负例计数
    false_negatives = 0

    for pred_beat, true_beat in zip(prediction, labels):
        # 房颤检出率计算（真实为4）
        if true_beat == 4:
            if pred_beat == 4:
                true_positives += 1
            else:
                false_negatives += 1
        # 房颤检准率计算（预测为4）
        if pred_beat == 4:
            if true_beat != 4:
                false_positives += 1

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, f1_score


def main():
    # 配置参数
    model_path = '/home/devel/code/monitor/physionet/models/saved/clinicDB_995.pth'
    data_dir = '/home/devel/监护共享数据/01-成人数据20250304/ClinicdatDB/ClinicDB/af_ONLY/numpy'
    output_picture_dir = '/home/devel/监护共享数据/01-成人数据20250304/ClinicdatDB/ClinicDB/af_ONLY/picture_pred'

    
    # 创建输出目录
    os.makedirs(output_picture_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载模型
    model = TransformerSegmentation128(input_size=1280)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 加载数据
    files = sorted(Path(data_dir).glob('*.npy'))
    print(f'找到 {len(files)} 个信号文件')

    total_recall = 0
    total_precision = 0
    total_f1 = 0
    file_count = 0

    total_pred_4 = 0
    total_label_4 = 0

    all_predictions = []
    all_labels = []
    # 对每个信号文件进行预测和可视化
    for file_path in files:
        file_name = file_path.stem
        print(f'处理文件: {file_name}')
        try:
            # 加载单个文件
            data = np.load(file_path)
            signal = data[:, 0]  # 取第一列作为信号数据
            label = data[:, 1] # shape [1280,]
            # 转换为10个心拍的标签
            label_10 = label[::128]  # shape [10,]
            
            
            signal = np.squeeze(signal)  # 确保信号是1D的

            output = predict_signal(model, signal, device) # prediction shape 

            prediction_beats = np.argmax(feature, axis=0)
            num_pred_4 = np.sum(prediction_beats == 4)
            num_label_4 = np.sum(label == 4)
            total_pred_4 += num_pred_4
            total_label_4 += num_label_4
            print(f"prediction_beats中4的数量: {num_pred_4}")
            print(f"labels中4的数量: {num_label_4}")
            print("Sample preds:", prediction_beats)
            print("Sample labels:", labels)

            # 可视化和保存结果
            feature = output
            prediction_point = np.repeat(prediction_beats, 128, axis=-1) # 把预测标签转成128个点的标签

            all_predictions.extend(prediction_beats.tolist())
            all_labels.extend(label.tolist())

            # 可视化和保存结果
            visualize_and_save_signal(signal, prediction_point, feature, file_name, output_picture_dir)


            
            # 保存预测结果
            # np.save(os.path.join(output_dir, f'{file_name}_pred.npy'), prediction)
        
            # 手动清理
            del signal, prediction
        
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")
            continue


    recall, precision, f1_score = calculate_heartbeat_metrics(all_predictions, all_labels)
    print(f'全局检出率（recall）：{recall:.4f}')
    print(f'全局检准率（precision）：{precision:.4f}')
    print(f'全局F1分数（f1_score）：{f1_score:.4f}') 
    
    print(f"整个文件夹 prediction_beats 中4的总数: {total_pred_4}")
    print(f"整个文件夹 labels 中4的总数: {total_label_4}")

    print('推理完成！')
    print(f'结果保存在: {output_picture_dir}')

if __name__ == '__main__':
    main()