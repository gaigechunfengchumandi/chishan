import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import signal

data = np.load('/Users/xingyulu/Public/afafaf/Holter_Data_3例/01-01-20200806-104604_raw_0_10929280.npy')#形状: (12193, 1301, 12)
save_path = '/Users/xingyulu/Public/afafaf/Holter_Data_3例/picture'
# 把第二导联作为ECG信号
ecg_signal_label = data[:, :, 1] # ECG信号形状: (12193, 1301) （总的养样本数， 样本序列长度）
def segment_beats(ecg_signal, RR_Interval):
    """
    根据RR间期将ECG信号分割成心拍片段
    
    参数:
    ecg_signal: ECG信号数据，一维数组
    RR_Interval: RR间期数据，包含10个心拍的间隔
    
    返回:
    beat_segments: 分割后的心拍片段列表，每个元素是一个心拍信号片段
    """
    R_location = 0  # R波位置标记
    start = 0       # 心拍起始位置
    end = 0         # 心拍结束位置
    beat_segments = []  # 存储分割后的心拍
    
    for k in range(10):  # 处理10个心拍
        # 第一个心拍的特殊处理
        if k == 0:
            start = 0  # 第一个心拍从信号开始
            R_location = int(0.6 * RR_Interval[k])  # 计算R波位置(假设在60%处)
            end = int(0.6 * RR_Interval[k]) + int(0.5 * RR_Interval[k + 1])  # 结束位置到下一个心拍的50%
        
        # 中间心拍的处理(第2-8个)
        if k != 0 and k != 9:
            start = end  # 当前心拍开始于上一个心拍的结束
            R_location = int(R_location) + int(RR_Interval[k])  # 更新R波位置
            end = int(R_location) + int(0.5 * RR_Interval[k + 1])  # 结束到下一个心拍的50%
        
        # 最后一个心拍的特殊处理
        if k == 9:
            start = end  # 开始于上一个心拍的结束
            R_location = int(R_location) + int(RR_Interval[k])  # 更新R波位置
            end = int(R_location) + int(0.5 * RR_Interval[k])  # 结束到当前RR间期的50%
        
        # 截取心拍片段并保存
        Beat_seg = ecg_signal[start: end]
        if len(Beat_seg) > 0:  # 只处理非空片段
            Beat_seg = signal.resample(Beat_seg, 128)
            beat_segments.append(Beat_seg)
        else:
            # 如果片段为空，添加一个全零数组
            beat_segments.append(np.zeros(128))
    
    return beat_segments

def plot_beat_sequence(beat_segments, beat_type, sample_idx=None, title='ECG Beat Signal Sequence', save_path=None):
    """
    绘制心拍信号序列图，根据心拍类别使用不同颜色
    
    参数:
    beat_segments: 心拍片段列表
    beat_type: 心拍类别数组 (1-5)
    sample_idx: 样本索引
    title: 图表标题
    save_path: 图片保存路径
    """
    plt.figure(figsize=(15, 5))
    
    # 定义5种心拍类别的颜色和标签
    type_colors = {
        0: 'green',    # 正常心拍
        1: 'blue',    # S
        2: 'green',   # V
        3: 'purple',     # X
        4: 'red',  # AF
    }
    
    type_labels = {
        0: 'N',
        1: 'S',
        2: 'V',
        3: 'X',
        4: 'AF'
    }
    
    current_position = 0
    legend_handles = []

    for i, (beat, b_type) in enumerate(zip(beat_segments, beat_type)):  # 遍历每个心拍片段和对应的类型
        b_type = int(b_type) if not np.isnan(b_type) else 0  # 处理NaN值，将其转换为0
        color = type_colors.get(b_type, 'gray')  # 根据心拍类型获取对应的颜色，默认为灰色
        x_values = np.arange(current_position, current_position + len(beat))  # 生成x轴坐标值
        line = plt.plot(x_values, beat, color=color)  # 绘制心拍波形
        
        # 只为每种类型添加一次图例
        if b_type not in [h.get_label() for h in legend_handles]:  # 检查该类型是否已添加到图例中
            line[0].set_label(type_labels[b_type])  # 设置图例标签
            legend_handles.append(line[0])  # 将该类型添加到图例句柄列表中
        
        current_position += len(beat)  # 更新下一个心拍的起始位置

    plt.xlabel('Sample Points')  # 设置x轴标签
    plt.ylabel('Amplitude')  # 设置y轴标签
    plt.title(title)  # 设置图表标题
    plt.legend(handles=legend_handles)  # 添加图例
    plt.grid(True)  # 显示网格线
    
    # 保存图片
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 使用样本索引创建唯一文件名
        if sample_idx is not None:
            filename = os.path.join(save_path, f"sample_{sample_idx}_{title.replace(' ', '_')}.png")
        else:
            filename = os.path.join(save_path, f"{title.replace(' ', '_')}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {filename}")

# 在循环中调用时传入样本索引
processed_count = 0
for i in range(12193):
    if i >= len(ecg_signal_label):  # 检查索引是否超出范围
        break
    if ecg_signal_label[i, 0] == 0:
        ecg_signal_label = np.delete(ecg_signal_label, i, axis=0)
        continue  # 改为continue跳过当前样本而不是break
    # 取出第i个样本的第1290至1301的值，这11个值代表了这10个心拍的RR间期
    RR_Interval = ecg_signal_label[i, 1290:1301]  # eg:[80. 80. 80. 80. 79. 80. 79. 79. 80. 48.]

    # 打印第一个样本的第1280至1290的值，这10个值代表了这10个心拍的类型
    beat_type = ecg_signal_label[i, 1280:1290]  # eg:[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

    ecg_signal = ecg_signal_label[i, :] / 200

    # 调用函数获取心拍片段
    beat_segments = segment_beats(ecg_signal, RR_Interval)

    # 调用函数显示心拍序列并保存图片
    plot_beat_sequence(beat_segments, beat_type, sample_idx=i, save_path=save_path)
    processed_count += 1
    if processed_count % 100 == 0:  # 每处理100个样本打印一次进度
        print(f"已处理 {processed_count} 个样本")

print(f"总共处理了 {processed_count} 个样本")




