"""
CU Database VF Segment Processing and Visualization Script
CU数据库室颤片段处理与可视化脚本

This script combines the functionality of segment extraction and visualization:
本脚本结合了片段提取和可视化的功能：

1. Processes the CU Ventricular Tachyarrhythmia Database:
   处理CU室性心动过速数据库：
   - Extracts VF and non-VF segments
     提取室颤和非室颤片段
   - Saves segments as numpy arrays
     将片段保存为numpy数组

2. Visualizes the extracted segments:
   可视化提取的片段：
   - Plots ECG signals in 10-second windows
     以10秒为窗口绘制心电图信号
   - Saves plots as PNG files
     将图形保存为PNG文件
"""

import pandas as pd
import wfdb
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ECGProcessor:
    def __init__(self, config: Dict):
        """
        初始化ECG处理器
        
        Args:
            config: 配置字典，包含所有必要的参数和路径
        """
        self.config = config
        self.setup_directories()
        self.setup_plot_style()
        
    def setup_directories(self):
        """设置所有必要的目录"""
        # 数据处理相关目录
        self.source_dir = Path(self.config['data_dir'])
        self.base_output_dir = Path(self.config['segment_output_dir'])
        self.vf_dir = self.base_output_dir / 'vf_segments'
        self.non_vf_dir = self.base_output_dir / 'non_vf_segments'
        
        # 图形输出目录
        self.plot_base_dir = Path(self.config['plot_output_dir'])
        self.vf_plot_dir = self.plot_base_dir / 'vf_plots'
        self.non_vf_plot_dir = self.plot_base_dir / 'non_vf_plots'
        
        # 创建所有必要的目录
        for directory in [self.vf_dir, self.non_vf_dir, self.vf_plot_dir, self.non_vf_plot_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def setup_plot_style(self):
        """设置绘图样式"""
        sns.set_style(self.config['plot_style'])
        self.figure_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        
    def extract_segments(self):
        """提取室颤和非室颤片段"""
        annotations_df = pd.read_excel(self.config['annotations_file'])
        grouped = annotations_df.groupby('Record')
        
        for record_name, group in grouped:
            try:
                for _, row in group.iterrows():
                    time_info = row['Symbol & Time']
                    if not pd.isna(time_info):
                        self._process_single_record(record_name, time_info)
            except Exception as e:
                print(f"Error processing record {record_name}: {str(e)}")
                
    def plot_segments(self):
        """绘制所有提取的片段"""
        # 处理室颤片段
        self._plot_directory_segments(self.vf_dir, "VF")
        # 处理非室颤片段
        self._plot_directory_segments(self.non_vf_dir, "Non-VF")
        
    def _plot_directory_segments(self, directory: Path, segment_type: str):
        """处理指定目录中的所有片段"""
        npy_files = list(directory.glob('*.npy'))
        for npy_file in npy_files:
            try:
                self._plot_single_file(npy_file, segment_type)
            except Exception as e:
                print(f"Error plotting {npy_file.name}: {str(e)}")
                plt.close()
                
    def _plot_single_file(self, npy_file: Path, segment_type: str):
        """绘制单个文件的信号"""
        # 从npy文件加载信号数据
        signal = np.load(npy_file)
        # 从配置中获取采样率
        fs = self.config['sampling_rate']
        # 计算每个窗口的采样点数（窗口时长 * 采样率）
        window_size = self.config['window_seconds'] * fs
        # 计算需要的总窗口数，使用向上取整确保覆盖所有数据
        total_windows = int(np.ceil(signal.shape[0] / window_size))
        
        # 遍历每个窗口进行绘图
        for window in range(total_windows):
            # 调用绘图函数处理当前窗口的数据
            self._plot_window(signal, window, total_windows, npy_file.stem, fs, segment_type)
            
    def _plot_window(self, signal, window_idx, total_windows, filename, fs, segment_type):
        """绘制单个时间窗口的数据"""
        window_size = self.config['window_seconds'] * fs
        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, signal.shape[0])
        
        plt.figure(figsize=self.figure_size)
        
        for lead_idx in range(signal.shape[1]):
            window_data = signal[start_idx:end_idx, lead_idx]
            time_axis = np.arange(len(window_data)) / fs
            plt.plot(time_axis, window_data, label=f'Lead {lead_idx+1}')
            
        plt.title(f'{segment_type} Signal: {filename}\n(Window {window_idx+1}/{total_windows})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.xlim(0, self.config['window_seconds'])
        plt.legend()
        plt.grid(True)
        
        # 根据片段类型选择输出目录
        plot_dir = self.vf_plot_dir if segment_type == "VF" else self.non_vf_plot_dir
        output_path = plot_dir / f'{filename}_window{window_idx+1}.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _process_single_record(self, record_name: str, time_info: str):
        """处理单条记录"""
        record_path = self.source_dir / record_name
        record = wfdb.rdrecord(str(record_path))
        
        # 处理室颤片段
        bracket_periods = self._extract_cu_segment(time_info)
        for start_time, end_time in bracket_periods:
            self._save_segment(record_path, record.fs, start_time, end_time, record_name, is_vf=True)
            
        # 处理非室颤片段
        remaining_segments = self._get_remaining_segments(record.sig_len, bracket_periods, record.fs)
        for start_time, end_time in remaining_segments:
            self._save_segment(record_path, record.fs, start_time, end_time, record_name, is_vf=False)
            
    def _save_segment(self, record_path, fs, start_time, end_time, record_name, is_vf):
        """保存信号片段"""
        try:
            segment = wfdb.rdrecord(
                str(record_path),
                sampfrom=int(start_time * fs),
                sampto=int(end_time * fs)
            )
            
            output_dir = self.vf_dir if is_vf else self.non_vf_dir
            segment_type = "vf" if is_vf else "non_vf"
            output_filename = f"{record_name}_{segment_type}_t{start_time}_{end_time}.npy"
            output_path = output_dir / output_filename
            
            np.save(str(output_path), segment.p_signal)
            print(f"Successfully saved {segment_type.upper()} segment {output_filename}")
            
        except Exception as e:
            segment_type = "VF" if is_vf else "non-VF"
            print(f"Error processing {segment_type} segment {start_time}-{end_time} "
                  f"for record {record_name}: {str(e)}")
            
    @staticmethod
    def _extract_cu_segment(time_str: str) -> List[Tuple[float, float]]:
        """从时间字符串中提取被方括号标记的时间段"""
        times = []  # 初始化一个空列表用于存储时间区间
        start_time = None  # 初始化起始时间为空
        
        for line in time_str.split('\n'):  # 按行分割时间字符串
            try:
                time_part = line.split(' at ')[-1]  # 提取时间部分
                minutes, seconds = map(int, time_part.split(':'))  # 将时间字符串分割为分钟和秒
                total_seconds = minutes * 60 + seconds  # 将时间转换为总秒数
                
                if '[' in line:  # 如果行中包含左括号，表示室颤开始
                    start_time = total_seconds  # 记录开始时间
                elif ']' in line and start_time is not None:  # 如果行中包含右括号且有开始时间
                    times.append((start_time, total_seconds))  # 添加时间区间到列表
                    start_time = None  # 重置开始时间
                    
            except Exception as e:
                print(f"Warning: Could not parse time string '{line}': {str(e)}")
                
        return times
    
    @staticmethod
    def _get_remaining_segments(record_length: int, bracket_periods: List[Tuple[float, float]], 
                              fs: float) -> List[Tuple[float, float]]:
        """获取所有非室颤片段的时间区间"""
        remaining_segments = []  # 初始化一个空列表用于存储非室颤片段的时间区间
        current_time = 0  # 初始化当前时间为0
        
        for start_time, end_time in sorted(bracket_periods):  # 遍历排序后的室颤时间区间
            if current_time < start_time:  # 如果当前时间小于室颤开始时间
                remaining_segments.append((current_time, start_time))  # 将非室颤区间添加到列表中
            current_time = end_time  # 更新当前时间为室颤结束时间
            
        if current_time < record_length / fs:  # 如果当前时间小于记录总长度
            remaining_segments.append((current_time, record_length / fs))  # 添加最后一个非室颤区间
            
        return remaining_segments  # 返回所有非室颤时间区间列表

def main():
    """主程序入口"""
    config = {
        'data_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/cu-ventricular-tachyarrhythmia-database-1.0.0',
        'segment_output_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/extracted_segments',
        'plot_output_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/plots',
        'annotations_file': 'cu-annotations.xlsx',
        'sampling_rate': 250,
        'window_seconds': 10,
        'plot_style': 'darkgrid',
        'figure_size': (15, 5),
        'dpi': 300
    }
    
    processor = ECGProcessor(config)
    
    # 步骤1：提取片段
    print("开始提取信号片段...")
    processor.extract_segments()
    print("片段提取完成！")
    
    # 步骤2：绘制图形
    print("开始绘制信号图形...")
    processor.plot_segments()
    print("图形绘制完成！")

if __name__ == "__main__":
    main()