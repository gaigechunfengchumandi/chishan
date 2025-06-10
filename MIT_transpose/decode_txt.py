import os
import wfdb
import numpy as np
from pathlib import Path

from scipy.signal import resample
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

def convert_wfdb_to_training_data(input_dir, output_dir):
    """
    将WFDB格式的心电数据转换为适合训练的格式
    
    参数:
        input_dir: 输入目录，包含WFDB格式的数据文件(.dat和.hea)
        output_dir: 输出目录，用于保存转换后的数据

    要功能包括：

    1. 心拍分组(每10个一组)
    2. 动态计算窗长(RR间期)
    3. 波形提取和重采样
    4. 数据拼接(波形+标签+RR间期)
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有.dat文件
    dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
    
    for dat_file in dat_files:
        try:
            record_name = dat_file[:-4]
            
            # 读取WFDB注释文件（保留）
            annotations = wfdb.rdann(os.path.join(input_dir, record_name), 'atr', return_label_elements=['label_store'])
            
            # 从txt读取信号数据
            txt_path = os.path.join(input_dir, f"{record_name}.txt")
            signals = np.loadtxt(txt_path)
            # 若signals是一维，需reshape为二维
            if signals.ndim == 1:
                signals = signals.reshape(-1, 1)

            # 处理所有导联中的NaN值
            for lead in range(signals.shape[1]):
                nan_indices = np.where(np.isnan(signals[:,lead]))[0]
                for idx in nan_indices:
                    prev_val = signals[idx-1,lead] if idx > 0 else signals[idx+1,lead]
                    next_val = signals[idx+1,lead] if idx < len(signals)-1 else signals[idx-1,lead]
                    signals[idx,lead] = (prev_val + next_val) / 2
            
            # 处理心拍分组和动态窗长
            groups = process_ecg_data(signals, annotations)
            
            # 为每组数据单独保存npy文件
            for i, group_data in enumerate(groups):
                output_path = os.path.join(output_dir, f"{record_name}_group_{i}.npy")
                np.save(output_path, group_data)
                        
            print(f"成功转换: {record_name}")
            
        except Exception as e:
            print(f"处理{record_name}时出错: {str(e)}")

# 标签转换函数, 这个函数是根据监护心电数据库心拍映射表格转换的，每个类别都是多对一的映射，最后可以转换为0-5共6个类
def label_transform_edan(ecg_labels):
    label_out = np.zeros(len(ecg_labels))
    for i in range(len(ecg_labels)):
        if ecg_labels[i] == 1 or ecg_labels[i] == 2 or ecg_labels[i] == 3 or \
                ecg_labels[i] == 11 or ecg_labels[i] == 12 or ecg_labels[i] == 13 or \
                ecg_labels[i] == 25 or ecg_labels[i] == 30 or ecg_labels[i] == 34 or \
                ecg_labels[i] == 35 or ecg_labels[i] == 38 or ecg_labels[i] == 42 or \
                ecg_labels[i] == 43 or ecg_labels[i] == 44:
            label_out[i] = 0 # N

        elif ecg_labels[i] == 4 or ecg_labels[i] == 7 or ecg_labels[i] == 8 or ecg_labels[i] == 9:
            label_out[i] = 1 # S 

        elif ecg_labels[i] == 5 or ecg_labels[i] == 6 or ecg_labels[i] == 10 or \
                ecg_labels[i] == 15 or ecg_labels[i] == 17 or ecg_labels[i] == 41:
            label_out[i] = 2 # V

        elif ecg_labels[i] == 45 or ecg_labels[i] == 46:
            label_out[i] = 4 # AF

        elif ecg_labels[i] == 33 :
            label_out[i] = 5 # VF

        else:
            label_out[i] = 3 # else

    return label_out

def process_ecg_data(signals, annotations):
    """
    处理心电数据，实现动态窗长分割和重采样
    
    参数:
        signals: 心电信号数据 (采样点数, 导联数)
        annotations: WFDB注释对象
        
    返回:
        处理后的数据数组 (n_samples, 1301)
    """


    # 获取注释信息
    ann_len = len(annotations.sample) # 注释长度
    label_store = annotations.label_store # 心拍类型
    samples = annotations.sample # 心拍位置
    rr_intervals = np.diff(samples) # 所有的rr间期
    
    # 步骤1：分组心拍(每10个一组)
    groups = []
    for i in range(1, ann_len - ann_len % 10, 10): 
        # 每10个一组，把所有的心拍类别，位值和rr间期都取好。
        group_samples = samples[i:i+10]
        group_labels = label_store[i:i+10]
        group_rr_intervals = rr_intervals[i-1:i+10] # 往前取一个，用来做最开始的心拍的前面预留位置

        # 步骤2：标签转换
        group_labels = label_transform_edan(group_labels)
        groups.append((group_samples, group_labels, group_rr_intervals))
    
    processed_data = []

    # 步骤3： 用之前整理的变量截取
    for group_samples, group_symbols, group_rr_intervals in groups:
        # 添加边界检查，确保start不小于0
        start = int(group_samples[0] - 0.5*group_rr_intervals[0])
        start = max(0, start)  # 确保start不小于0
        end = int(group_samples[-1] + 0.9*group_rr_intervals[-1])
        count_10_beat = signals[start:end]  # 直接切片，无需逗号
        # 补0得到a矩阵
        a = np.pad(
            count_10_beat,
            (0, 10000 - count_10_beat.shape[0]),  # 只需一维补零
            mode='constant',
            constant_values=0
        )
        # 步骤4：拼接标签
        if len(group_symbols) != 10:
            continue
        group_symbols = np.array(group_symbols).reshape(10, 1)  # (10,1)
        b = np.concatenate([a.reshape(-1, 1), group_symbols], axis=0)  # (10010,1)
        # 步骤5：拼接RR间期
        if len(group_rr_intervals) != 11:
            continue
        group_rr_intervals = group_rr_intervals.reshape(11, 1)  # (11,1)
        c = np.concatenate([b, group_rr_intervals], axis=0)  # (10021,1)
        processed_data.append(c)  # c的形状是(10021,1)
    
    return processed_data  # 返回列表而不是合并后的数组


def plot_ecg_with_annotations(txt_path, annotations, lead=0, max_points=5000):
    """
    画出txt读取的心电信号，并在图中标出annotation.sample的位置
    参数：
        txt_path: txt文件路径
        annotations: wfdb.rdann返回的注释对象
        lead: 选择画哪一导联（默认0）
        max_points: 最多显示多少采样点（防止太长）
    """
    signals = np.loadtxt(txt_path)
    if signals.ndim == 1:
        signals = signals.reshape(-1, 1)
    signal = signals[:max_points, lead]
    plt.figure(figsize=(15, 4))
    plt.plot(signal, label=f'Lead {lead}')
    # 标注annotation.sample
    ann_samples = [s for s in annotations.sample if s < max_points]
    plt.scatter(ann_samples, signal[ann_samples], color='red', marker='o', label='Annotations')
    plt.title('ECG Signal with Annotations')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = "/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/心拍断掉/ori"
    output_directory = "/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/心拍断掉/train"
    
    # 执行转换
    convert_wfdb_to_training_data(input_directory, output_directory)
