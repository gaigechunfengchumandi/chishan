

import wfdb
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt


def highpass_filter(data, cutoff_freq, fs, order=4):
    """应用高通滤波器，这是产科那边给的，不要改了""" 
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype = 'high', analog= False)
    y = signal.filtfilt(b, a, data, axis = 0)
    return y


def lowpass_filter(data, cutoff_freq, fs, order=4):
    """应用低通滤波器"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype = 'low', analog= False)
    y = signal.filtfilt(b, a, data, axis = 0)
    return y


def process_data(input_file, output_dir, window_size_sec=10):
    """处理WFDB文件并保存切分后的数据"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取WFDB文件
    base_path = input_file.rsplit('.', 1)[0]
    record = wfdb.rdrecord(base_path)
    signals, fs = record.p_signal, record.fs  # signals shape eg:(1105264, 4) 有4个导联
    


    # 选取第一个导联
    signals = signals[:, 1]
    # 高通滤波,去除基线漂移
    signals = highpass_filter(signals, cutoff_freq=0.67, fs=fs)
    # 低通滤波,去除高频噪声
    signals = lowpass_filter(signals, cutoff_freq=40, fs=fs)


    # 切分数据
    window_size = window_size_sec * fs
    num_windows = len(signals) // window_size
    
    segments = []
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        window_data = signals[start_idx:end_idx]
        # 归一化窗口
        # window_data = (window_data - np.mean(window_data)) / np.std(window_data)
        # 应用高通滤波器
        # window_data = highpass_filter(window_data, cutoff_freq=0.67, fs=fs)
        
        output_file = os.path.join(output_dir, f'segment_{i:04d}.npy')
        np.save(output_file, window_data)
        segments.append(window_data)
    
    return segments, fs, num_windows

def load_data(data_dir):
    """从指定目录加载所有数据段"""
    segments = []
    fs = 250  # 采样率固定为250Hz
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    for file in files:
        file_path = os.path.join(data_dir, file)
        segment = np.load(file_path)
        segments.append(segment)
    
    return segments, fs

def plot_data(segments, fs, picture_dir):
    """将数据段绘制成图片并保存"""
    os.makedirs(picture_dir, exist_ok=True)
    
    total_segments = len(segments)
    for i, data in enumerate(segments):
        plt.figure(figsize=(15, 5))
        time = np.arange(len(data)) / fs
        plt.plot(time, data)
        plt.title(f'Segment {i:04d}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        plt.savefig(os.path.join(picture_dir, f'segment_{i:04d}.png'))
        plt.close()
        
        # 打印进度
        print(f'正在处理图片: {i+1}/{total_segments}, 完成度: {((i+1)/total_segments)*100:.2f}%')

def main():

    # 配置路径
    input_file = "/Users/xingyulu/Public/监护心电预警/监护部门提供数据/室颤/原始数据/141.dat"
    output_dir = "/Users/xingyulu/Public/监护心电预警/监护部门提供数据/室颤/141_10s/processed_data"
    picture_dir = "/Users/xingyulu/Public/监护心电预警/监护部门提供数据/室颤/141_10s/picture"
    
    # 选择操作模式
    mode = input("选择操作模式 (1: 处理数据, 2: 生成图片, 3: 全部执行): ")
    
    if mode in ['1', '3']:
        segments, fs, num_windows = process_data(input_file, output_dir)
        print(f"数据处理完成，总共切分出 {num_windows} 个样本")
        print(f"数据已保存到: {output_dir}")
    
    if mode in ['2', '3']:
        segments, fs = load_data(output_dir)
        plot_data(segments, fs, picture_dir)
        print(f"图片已保存到: {picture_dir}")

if __name__ == "__main__":
    main()