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
        self.figure_size = self.config['figure_size']
        self.dpi = self.config['dpi'] 
        
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
        annotations_df = pd.read_csv(self.config['annotations_file'])  # 从Excel文件读取标注数据
        grouped = annotations_df.groupby('Record')  # 按记录名称对数据进行分组
        
        for record_name, group in grouped:  # 遍历每个记录组
            try:
                for _, row in group.iterrows():  # 遍历组内的每一行数据
                    time_info = row['Symbol & Time']  # 获取时间标记信息
                    if not pd.isna(time_info):  # 如果时间信息不为空
                        self._process_single_record(record_name, time_info)  # 处理单条记录
            except Exception as e:
                print(f"Error processing record {record_name}: {str(e)}")  # 打印错误信息
                
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
                self._process_and_plot_single_file(npy_file, segment_type)
            except Exception as e:
                print(f"Error plotting {npy_file.name}: {str(e)}")
                plt.close()
                
    def _process_and_plot_single_file(self, npy_file: Path, segment_type: str):
        """处理并绘制单个文件的信号，使用内存映射和优化的绘图方法"""
        # 使用内存映射加载信号数据，减少内存使用
        signal = np.load(npy_file, mmap_mode='r')
        # 获取文件名（不带路径和扩展名）
        filename = npy_file.stem
        # 从配置中获取采样率
        fs = self.config['sampling_rate']
        # 计算每个窗口的采样点数（窗口时长 * 采样率）
        window_size = int(self.config['window_seconds'] * fs)
        # 计算需要的总窗口数，使用向上取整确保覆盖所有数据
        total_windows = int(np.ceil(signal.shape[0] / window_size))
        
        # 创建窗口数据保存目录，分别为VF和非VF创建不同的子目录
        if segment_type == "VF":
            window_data_dir = self.vf_dir.parent / 'window_data' / 'vf_windows'
        else:
            window_data_dir = self.non_vf_dir.parent / 'window_data' / 'non_vf_windows'
        window_data_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据片段类型选择输出目录
        plot_dir = self.vf_plot_dir if segment_type == "VF" else self.non_vf_plot_dir
        
        # 遍历每个窗口进行保存和绘图
        for window_idx in range(total_windows):
            # 计算窗口起始索引
            start_idx = window_idx * window_size
            # 计算窗口结束索引，确保不超过信号长度
            end_idx = min((window_idx + 1) * window_size, signal.shape[0])
            
            # 直接引用原始数据的视图，而不是复制数据
            window_data = signal[start_idx:end_idx]
            
            # 保存窗口数据（如果需要）
            save_path = window_data_dir / f'{filename}_window{window_idx+1}.npy'
            np.save(str(save_path), window_data)
            
            # 使用更高效的绘图方法
            self._plot_window_optimized(window_data, window_idx, total_windows, filename, fs, segment_type, plot_dir)
            
    def _plot_window_optimized(self, window_data, window_idx, total_windows, filename, fs, segment_type, plot_dir):
        """使用优化的方法绘制单个时间窗口的数据"""
        # 使用面向对象的接口创建图形和轴，提高性能
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # 生成时间轴数据（只生成一次）
        time_axis = np.linspace(0, len(window_data) / fs, num=len(window_data), endpoint=False)
        
        # 一次性绘制所有导联数据
        for lead_idx in range(window_data.shape[1]):
            ax.plot(time_axis, window_data[:, lead_idx], label=f'Lead {lead_idx+1}')
        
        # 设置图形属性
        ax.set_title(f'{segment_type} Signal: {filename}\n(Window {window_idx+1}/{total_windows})')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_xlim(0, self.config['window_seconds'])
        ax.grid(True)
        ax.legend()
        
        # 构建输出路径
        output_path = plot_dir / f'{filename}_window{window_idx+1}.png'
        
        # 保存图形并立即关闭，释放内存
        fig.savefig(str(output_path), dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
    def _process_single_record(self, record_name: str, time_info: str):
        """处理单条记录"""
        record_path = self.source_dir / record_name  # 构建记录文件的完整路径
        record = wfdb.rdrecord(str(record_path))  # 读取WFDB格式的记录文件
        
        # 处理室颤片段
        bracket_periods = self._extract_cu_segment(time_info)  # 从时间信息中提取室颤片段的时间区间
        for start_time, end_time in bracket_periods:  # 遍历每个室颤片段的时间区间
            self._save_segment(record_path, record.fs, start_time, end_time, record_name, is_vf=True)  # 保存室颤片段
            
        # 处理非室颤片段
        remaining_segments = self._get_remaining_segments(record.sig_len, bracket_periods, record.fs, time_info)  # 获取非室颤片段的时间区间
        for start_time, end_time in remaining_segments:  # 遍历每个非室颤片段的时间区间
            self._save_segment(record_path, record.fs, start_time, end_time, record_name, is_vf=False)  # 保存非室颤片段
            
    def _save_segment(self, record_path, fs, start_time, end_time, record_name, is_vf):
        """保存信号片段"""
        try:
            # 读取指定时间范围内的信号片段
            segment = wfdb.rdrecord(
                str(record_path),
                sampfrom=int(start_time * fs),
                sampto=int(end_time * fs)
            )
            
            # 根据是否为室颤确定输出目录和片段类型
            output_dir = self.vf_dir if is_vf else self.non_vf_dir
            segment_type = "vf" if is_vf else "non_vf"
            
            # 构建输出文件名和路径
            output_filename = f"{record_name}_{segment_type}_t{start_time}_{end_time}.npy"
            output_path = output_dir / output_filename
            
            # 将信号数据保存为numpy数组
            np.save(str(output_path), segment.p_signal)
            print(f"Successfully saved {segment_type.upper()} segment {output_filename}")
            
        except Exception as e:
            # 发生错误时输出错误信息
            segment_type = "VF" if is_vf else "non-VF"
            print(f"Error processing {segment_type} segment {start_time}-{end_time} "
                  f"for record {record_name}: {str(e)}")
            
    @staticmethod
    def _extract_cu_segment(time_str: str) -> List[Tuple[float, float]]:
        """从时间字符串中提取被方括号标记的时间段，并过滤掉噪声区域"""
        times = []  # 初始化一个空列表用于存储时间区间
        start_time = None  # 初始化起始时间为空
        noise_times = []  # 初始化噪声时间点列表
        
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
                elif '~' in line:  # 如果行中包含波浪号，表示噪声
                    noise_times.append(total_seconds)  # 记录噪声时间点
                    
            except Exception as e:
                print(f"Warning: Could not parse time string '{line}': {str(e)}")
        
        # 过滤掉噪声区域
        if noise_times:
            filtered_times = []
            for start, end in times:
                valid_segments = ECGProcessor._filter_noise(start, end, noise_times, 5)
                filtered_times.extend(valid_segments)
            return filtered_times
        
        return times
    
    @staticmethod
    def _filter_noise(start: float, end: float, noise_times: List[float], margin: float) -> List[Tuple[float, float]]:
        """
        过滤掉包含噪声的时间段
        
        Args:
            start: 片段开始时间
            end: 片段结束时间
            noise_times: 噪声时间点列表
            margin: 噪声前后需要排除的时间边界（秒）
            
        Returns:
            过滤后的有效时间段列表
        """
        if not noise_times:
            return [(start, end)]
            
        # 计算所有噪声区间
        noise_intervals = []
        for noise_time in noise_times:
            noise_start = max(0, noise_time - margin)
            noise_end = noise_time + margin
            noise_intervals.append((noise_start, noise_end))
            
        # 合并重叠的噪声区间
        noise_intervals.sort()
        merged_noise = []
        current_start, current_end = noise_intervals[0]
        
        for noise_start, noise_end in noise_intervals[1:]:
            if noise_start <= current_end:
                # 区间重叠，合并
                current_end = max(current_end, noise_end)
            else:
                # 区间不重叠，添加当前区间并开始新区间
                merged_noise.append((current_start, current_end))
                current_start, current_end = noise_start, noise_end
                
        merged_noise.append((current_start, current_end))
        
        # 计算有效区间
        valid_segments = []
        current_pos = start
        
        for noise_start, noise_end in merged_noise:
            if noise_start > current_pos and noise_start < end:
                # 添加噪声前的有效区间
                valid_segments.append((current_pos, noise_start))
            current_pos = max(current_pos, noise_end)
            
        # 添加最后一个噪声后的有效区间
        if current_pos < end:
            valid_segments.append((current_pos, end))
            
        return valid_segments
    
    @staticmethod
    def _get_remaining_segments(record_length: int, bracket_periods: List[Tuple[float, float]], 
                              fs: float, time_str: str = None) -> List[Tuple[float, float]]:
        """获取所有非室颤片段的时间区间，并过滤掉噪声区域"""
        remaining_segments = []  # 初始化一个空列表用于存储非室颤片段的时间区间
        current_time = 0  # 初始化当前时间为0
        noise_times = []  # 初始化噪声时间点列表
        
        # 如果提供了时间字符串，提取噪声时间点
        if time_str:
            for line in time_str.split('\n'):
                if '~' in line:
                    try:
                        time_part = line.split(' at ')[-1]
                        minutes, seconds = map(int, time_part.split(':'))
                        total_seconds = minutes * 60 + seconds
                        noise_times.append(total_seconds)
                    except Exception as e:
                        print(f"Warning: Could not parse noise time string '{line}': {str(e)}")
        
        for start_time, end_time in sorted(bracket_periods):  # 遍历排序后的室颤时间区间
            if current_time < start_time:  # 如果当前时间小于室颤开始时间
                remaining_segments.append((current_time, start_time))  # 将非室颤区间添加到列表中
            current_time = end_time  # 更新当前时间为室颤结束时间
            
        if current_time < record_length / fs:  # 如果当前时间小于记录总长度
            remaining_segments.append((current_time, record_length / fs))  # 添加最后一个非室颤区间
        
        # 如果有噪声时间点，过滤掉噪声区域
        if noise_times:
            filtered_segments = []
            for start, end in remaining_segments:
                valid_segments = ECGProcessor._filter_noise(start, end, noise_times, 5)
                filtered_segments.extend(valid_segments)
            return filtered_segments
            
        return remaining_segments  # 返回所有非室颤时间区间列表

def main():
    """主程序入口"""
    config = {
        'data_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/cu-ventricular-tachyarrhythmia-database-1.0.0',
        'segment_output_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/extracted_segments',
        'plot_output_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/plots',
        'annotations_file': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/annotations.csv',
        'sampling_rate': 250,
        'window_seconds': 10,
        'plot_style': 'darkgrid',
        'figure_size': (15, 5),
        'dpi': 300,
        'min_file_size_kb': 20  # 添加最小文件大小要求
    }
    
    processor = ECGProcessor(config)
    
    # 步骤1：提取片段
    print("开始提取信号片段...")
    # processor.extract_segments()
    print("片段提取完成！")
    
    # 步骤2：绘制图形
    print("开始绘制信号图形...")
    # processor.plot_segments()
    print("图形绘制完成！")
    


if __name__ == "__main__":
    main()