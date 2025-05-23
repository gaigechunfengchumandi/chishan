

import os
import wfdb
import numpy as np
from pathlib import Path
from scipy import signal

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
            
            # 读取WFDB记录和注释文件
            record = wfdb.rdrecord(os.path.join(input_dir, record_name))
            annotations = wfdb.rdann(os.path.join(input_dir, record_name), 'atr')
            
            # 获取信号数据和注释信息
            signals = record.p_signal

            # 处理心拍分组和动态窗长
            groups = process_ecg_data(signals, annotations)
            
            # 为每组数据单独保存npy文件
            for i, group_data in enumerate(groups):
                output_path = os.path.join(output_dir, f"{record_name}_group_{i}.npy")
                np.save(output_path, group_data)
            
            # 保存元数据
            meta_path = os.path.join(output_dir, f"{record_name}_meta.txt")
            with open(meta_path, 'w') as f:
                f.write(f"采样率: {fs}\n")
                f.write(f"信号名称: {', '.join(sig_names)}\n")
                
            print(f"成功转换: {record_name}")
            
        except Exception as e:
            print(f"处理{record_name}时出错: {str(e)}")

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
    ann_len = len(annotations.sample)
    symbols = annotations.symbol
    samples = annotations.sample
    
    # 步骤1：分组心拍(每10个一组)
    groups = []
    for i in range(0, ann_len - ann_len % 10, 10):
        group_samples = samples[i:i+10]
        group_symbols = symbols[i:i+10]
        # 增加一个标签转换函数 如果是N就为0，如果是S就为1，如果是V就为2，其他为3
        group_symbols = [0 if sym == 'N' else 1 if sym == 'S' else 2 if sym == 'V' else 3 for sym in group_symbols]
        groups.append((group_samples, group_symbols))
    
    processed_data = []
    
    for group_samples, group_symbols in groups:
        # 步骤2：计算RR间期
        rr_intervals = np.diff(group_samples)
        left_pad = int(rr_intervals[0] / 2)  # Convert to integer
        right_pad = int(rr_intervals[-1] / 2)  # Convert to integer
        rr_intervals = np.concatenate([[left_pad], rr_intervals, [right_pad]])
        
        # 步骤3：简化版：直接截取10个心拍之间的信号并重采样到1280
        start = int(group_samples[0])
        end = int(group_samples[-1])
        beat = signals[start:end, :]  # shape (n, 4)
        beat_resampled = resample_beat(beat, 1280)  # (1280, 4)
        beat_waveforms = [beat_resampled]
        
        # 修改拼接方式，保持4导联结构
        a = np.concatenate(beat_waveforms, axis=0)  # 形状变为(1280,4)
        
        # 步骤4：拼接标签
        # 将标签转换为(10,1)形状并复制4次以匹配导联维度
        group_symbols = np.tile(np.array(group_symbols).reshape(10,1), (1,4))
        b = np.concatenate([a, group_symbols], axis=0)  # 形状变为(1290,4)
        
        # 步骤5：拼接RR间期
        # 将RR间期转换为(11,1)形状并复制4次以匹配导联维度
        rr_intervals = np.tile(rr_intervals.reshape(11,1), (1,4))
        c = np.concatenate([b, rr_intervals], axis=0)  # 最终形状(1301,4)
        # 添加打印最后20行数据的代码

        # print("最后20位数据:")
        # print(c[-20:])
        processed_data.append(c)  # c的形状是(1301,4)
    
    return processed_data  # 返回列表而不是合并后的数组

def resample_beat(beat, target_length):
    """
    将心拍波形重采样到目标长度
    
    参数:
        beat: 原始心拍波形 (n_samples, n_leads)
        target_length: 目标长度
        
    返回:
        重采样后的波形 (target_length, n_leads)
    """
    # 线性插值实现重采样
    original_length = beat.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    
    beat_resampled = np.zeros((target_length, beat.shape[1]))
    for lead in range(beat.shape[1]):
        beat_resampled[:, lead] = np.interp(x_new, x_original, beat[:, lead])
    
    return beat_resampled

if __name__ == "__main__":
    # 设置输入输出目录
    input_directory = "/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/ori_data"
    output_directory = "/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/train"
    
    # 执行转换
    convert_wfdb_to_training_data(input_directory, output_directory)