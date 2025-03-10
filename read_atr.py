"""
这个脚本用于读取CU Ventricular Tachyarrhythmia Database的注释文件(.atr)，
并将非正常心拍(非'N'类型)的注释信息导出到CSV文件中。

主要功能：
1. 遍历cu01到cu35的记录文件
2. 读取每个记录的注释信息
3. 提取非正常心拍的符号和时间点
4. 将数据保存为CSV格式，包含记录ID和对应的异常事件信息

输出文件格式：
- CSV文件包含两列：'Record'和'Symbol & Time'
- 时间以"分钟:秒"的格式显示
"""

import wfdb
import csv
import os
import re
from typing import Dict, List, Tuple, Set

# 配置参数
CONFIG = {
    'data_dir': '/Users/xingyulu/Public/心电预警/公开数据/',
    'database_dir': 'cu-ventricular-tachyarrhythmia-database-1.0.0',
    'output_file': 'annotations.csv'
}

def get_available_record_numbers(directory: str) -> List[int]:
    """
    扫描目录获取所有可用的记录编号
    
    Args:
        directory: 数据文件所在目录
        
    Returns:
        List[int]: 排序后的可用记录编号列表
    """
    record_numbers = set()
    
    # 扫描目录中的所有文件
    for filename in os.listdir(directory):
        # 使用正则表达式匹配除了后缀之外的全部名字
        match = re.match(r'(.+)\.(dat|hea|atr)$', filename)
        if match:
            record_numbers.add(match.group(1))
    
    # 转换为列表并排序
    return sorted(list(record_numbers))

def convert_sample_to_time(sample: int, fs: float) -> Tuple[int, int]:
    """
    将采样点转换为分钟和秒
    
    Args:
        sample: 采样点
        fs: 采样频率
    
    Returns:
        Tuple[int, int]: (分钟, 秒)
    """
    time_in_minutes = sample / fs / 60
    minutes = int(time_in_minutes)
    seconds = int((time_in_minutes - minutes) * 60)
    return minutes, seconds

def process_record(record_name: str) -> Dict[str, List[str]]:
    """
    处理单个记录文件，提取非正常心拍注释
    
    Args:
        record_name: 记录文件的完整路径（不包含扩展名）
    
    Returns:
        Dict[str, List[str]]: 包含异常事件的字典和采样率
    """
    data_dict = {record_name: {'events': [], 'fs': None}}
    
    try:
        annotation = wfdb.rdann(record_name, 'atr')
        record = wfdb.rdrecord(record_name)
        data_dict[record_name]['fs'] = record.fs
        
        for j, symbol in enumerate(annotation.symbol):
            if symbol != 'N':
                minutes, seconds = convert_sample_to_time(annotation.sample[j], annotation.fs)
                data_dict[record_name]['events'].append(f"{symbol} at {minutes}:{seconds:02d}")
                
    except Exception:
        print(f"记录 {record_name} 没有标注文件，只处理信号")
        record = wfdb.rdrecord(record_name)  # 验证信号文件是否存在
        data_dict[record_name]['fs'] = record.fs
        
    return data_dict

def write_annotations_to_csv(data: Dict[str, List[str]], record_id: str, writer: csv.writer) -> None:
    """
    将注释数据写入CSV文件
    
    Args:
        data: 包含异常事件和采样率的字典
        record_id: 记录ID
        writer: CSV writer对象
    """
    if data:  # 确保有数据再写入
        record_data = next(iter(data.values()))
        events_str = '\n'.join(record_data['events'])
        writer.writerow([record_id, events_str, record_data['fs']])

def main():
    """主程序入口"""
    data_path = os.path.join(CONFIG['data_dir'], CONFIG['database_dir'])
    output_path = os.path.join(CONFIG['data_dir'], CONFIG['output_file'])
    
    # 获取可用的记录编号
    record_numbers = get_available_record_numbers(data_path)
    
    if not record_numbers:
        print("未找到任何记录文件")
        return
        
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Record', 'Symbol & Time', 'Sampling Rate'])  # 添加采样率列
        
        for record_num in record_numbers:
            record_id = f'{record_num}' 
            record_path = os.path.join(data_path, str(record_num))
            
            print(f"处理记录 {record_id}")
            data_dict = process_record(record_path)
            if data_dict:
                write_annotations_to_csv(data_dict, record_id, writer)

if __name__ == '__main__':
    main()



