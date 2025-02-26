import pandas as pd
import wfdb
import os
from pathlib import Path
import pdb

def process_records():
    # 读取Excel文件
    annotations_df = pd.read_excel('cu-annotations.xlsx')
    
    # 创建输出目录
    output_dir = Path('extracted_segments')
    output_dir.mkdir(exist_ok=True)
    
    # 原始数据目录
    source_dir = Path('cu-ventricular-tachyarrhythmia-database-1.0.0')
    pdb.set_trace()
    
    # 遍历每一行标注
    for index, row in annotations_df.iterrows():
        record_name = row['Record']
        time_info = row['Symbol & Time']
        
        # 确保时间信息存在
        if pd.isna(time_info):
            continue
            
        try:
            # 读取原始记录
            record_path = source_dir / record_name
            record = wfdb.rdrecord(str(record_path))
            
            # 解析时间信息（假设格式为"秒数"）
            time_seconds = float(time_info)
            
            # 计算采样点位置（假设采样率为250Hz）
            sample_point = int(time_seconds * record.fs)
            
            # 截取前后5秒的数据
            start_sample = max(0, sample_point - 5 * record.fs)
            end_sample = min(record.sig_len, sample_point + 5 * record.fs)
            
            # 读取指定范围的数据
            segment = wfdb.rdrecord(
                str(record_path), 
                sampfrom=start_sample,
                sampto=end_sample
            )
            
            # 生成新的文件名
            output_filename = f"{record_name}_t{time_seconds:.1f}"
            output_path = output_dir / output_filename
            
            # 保存截取的片段
            wfdb.wrsamp(
                str(output_path),
                segment.p_signal,
                fs=segment.fs,
                sig_name=segment.sig_name,
                units=segment.units,
                fmt=['16'] * len(segment.sig_name)
            )
            
            print(f"Successfully processed {output_filename}")
            
        except Exception as e:
            print(f"Error processing record {record_name}: {str(e)}")

if __name__ == "__main__":
    process_records() 