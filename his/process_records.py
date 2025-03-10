"""
CU Database VF Segment Extraction Script
CU数据库室颤片段提取脚本

This script processes the CU Ventricular Tachyarrhythmia Database to extract and save 
ventricular fibrillation (VF) and non-VF segments from ECG recordings.
该脚本处理CU室性心动过速数据库，从心电图记录中提取并保存室颤(VF)和非室颤片段。

Key Features 主要功能:
- Reads annotations from Excel file containing VF episode timestamps
  从Excel文件中读取包含室颤发作时间戳的标注
- Extracts VF segments based on bracketed time periods
  基于方括号标记的时间段提取室颤片段
- Extracts non-VF segments from the remaining periods
  从剩余时间段中提取非室颤片段
- Saves segments as numpy arrays in separate directories
  将片段以numpy数组格式保存在不同目录中

"""

import pandas as pd
import wfdb
import os
import numpy as np
from pathlib import Path

def extract_cu_segment(time_str):
    """
    Extract time segments marked with brackets from the time string.
    从时间字符串中提取被方括号标记的时间段。

    Args:
        time_str (str): String containing time information with brackets marking VF segments
                       包含时间信息的字符串，其中方括号标记室颤片段

    Returns:
        list: List of tuples (start_time, end_time) in seconds
              包含(开始时间,结束时间)的元组列表，单位为秒
    """
    times = []
    start_time = None
    
    # 按换行符分割多个时间点
    for line in time_str.split('\n'):
        try:
            # 提取时间部分
            time_part = line.split(' at ')[-1]
            # 分割分钟和秒
            minutes, seconds = map(int, time_part.split(':'))
            # 转换为总秒数
            total_seconds = minutes * 60 + seconds
            
            # 检查是否是开始或结束标记
            if '[' in line:
                start_time = total_seconds
            elif ']' in line and start_time is not None:
                times.append((start_time, total_seconds))
                start_time = None
                
        except Exception as e:
            print(f"Warning: Could not parse time string '{line}': {str(e)}")
            continue
    return times

def get_remaining_segments(record_length, bracket_periods, fs):
    """
    Get all segments that are not within the bracketed periods (non-VF segments).
    获取所有不在方括号时间段内的片段（非室颤片段）。

    Args:
        record_length (int): Total length of the record in samples
                           记录的总采样点数
        bracket_periods (list): List of tuples containing VF segments (start_time, end_time)
                              包含室颤片段的元组列表 (开始时间,结束时间)
        fs (float): Sampling frequency in Hz
                   采样频率，单位赫兹

    Returns:
        list: List of tuples (start_time, end_time) representing non-VF segments
              包含非室颤片段的元组列表 (开始时间,结束时间)
    """
    remaining_segments = []
    current_time = 0
    
    # 按时间顺序处理每个括号时间段
    for start_time, end_time in sorted(bracket_periods):
        # 如果当前时间到开始时间之间有间隔，添加这个间隔
        if current_time < start_time:
            remaining_segments.append((current_time, start_time))
        # 更新当前时间为结束时间
        current_time = end_time
    
    # 处理最后一个括号时间段之后的时间
    if current_time < record_length / fs:
        remaining_segments.append((current_time, record_length / fs))
        
    return remaining_segments

class SegmentProcessor:
    def __init__(self, config):
        """
        初始化处理器
        
        Args:
            config: 包含所有必要路径的配置字典
        """
        self.config = config
        self.setup_directories()
        
    def setup_directories(self):
        """设置必要的目录结构"""
        self.base_output_dir = Path(self.config['output_dir'])
        self.vf_dir = self.base_output_dir / 'vf_segments'
        self.non_vf_dir = self.base_output_dir / 'non_vf_segments'
        self.source_dir = Path(self.config['data_dir'])
        
        # 创建输出目录
        self.vf_dir.mkdir(parents=True, exist_ok=True)
        self.non_vf_dir.mkdir(parents=True, exist_ok=True)
    
    def save_segment(self, segment, record_name, start_time, end_time, is_vf=True):
        """保存单个片段"""
        output_dir = self.vf_dir if is_vf else self.non_vf_dir
        segment_type = "vf" if is_vf else "non_vf"
        output_filename = f"{record_name}_{segment_type}_t{start_time}_{end_time}.npy"
        output_path = output_dir / output_filename
        
        np.save(str(output_path), segment.p_signal)
        print(f"Successfully processed {segment_type.upper()} segment {output_filename}")
    
    def process_single_record(self, record_name, time_info):
        """处理单条记录"""
        record_path = self.source_dir / record_name
        record = wfdb.rdrecord(str(record_path))
        
        # 处理室颤片段
        bracket_periods = extract_cu_segment(time_info)
        for start_time, end_time in bracket_periods:
            try:
                segment = wfdb.rdrecord(
                    str(record_path),
                    sampfrom=int(start_time * record.fs),
                    sampto=int(end_time * record.fs)
                )
                self.save_segment(segment, record_name, start_time, end_time, is_vf=True)
            except Exception as e:
                print(f"Error processing VF segment {start_time}-{end_time} for record {record_name}: {str(e)}")
        
        # 处理非室颤片段
        remaining_segments = get_remaining_segments(record.sig_len, bracket_periods, record.fs)
        for start_time, end_time in remaining_segments:
            try:
                segment = wfdb.rdrecord(
                    str(record_path),
                    sampfrom=int(start_time * record.fs),
                    sampto=int(end_time * record.fs)
                )
                self.save_segment(segment, record_name, start_time, end_time, is_vf=False)
            except Exception as e:
                print(f"Error processing non-VF segment {start_time}-{end_time} for record {record_name}: {str(e)}")
    
    def process_all_records(self, annotations_file):
        """处理所有记录"""
        annotations_df = pd.read_excel(annotations_file)
        grouped = annotations_df.groupby('Record')
        
        for record_name, group in grouped:
            try:
                for _, row in group.iterrows():
                    time_info = row['Symbol & Time']
                    if not pd.isna(time_info):
                        self.process_single_record(record_name, time_info)
            except Exception as e:
                print(f"Error loading record {record_name}: {str(e)}")

def main():
    """主程序入口"""
    config = {
        'data_dir': '/Users/xingyulu/Downloads/心电预警/公开数据/cu-ventricular-tachyarrhythmia-database-1.0.0',
        'output_dir': '/Users/xingyulu/Downloads/心电预警/公开数据/extracted_segments',
        'annotations_file': 'cu-annotations.xlsx'
    }
    
    processor = SegmentProcessor(config)
    processor.process_all_records(config['annotations_file'])

if __name__ == "__main__":
    main()