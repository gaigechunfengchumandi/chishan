"""
这个脚本用于读取指定目录下的所有.rhythm文件，
并检查文件中是否包含"CFE"字符，将符合条件的文件名记录到CSV文件中。

主要功能：
1. 遍历指定目录下的所有.rhythm文件
2. 读取每个文件的注释信息
3. 检查是否包含"CFE"字符
4. 将符合条件的文件名保存为CSV格式
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
    'output_file': 'cfe_annotations.csv'
}

def get_rhythm_files(directory: str) -> List[str]:
    """
    扫描目录获取所有.rhythm文件
    
    Args:
        directory: 数据文件所在目录
        
    Returns:
        List[str]: 排序后的.rhythm文件列表(不含扩展名)
    """
    rhythm_files = set() #创建一个无序且不重复的集合
    
    # 扫描目录中的所有文件
    for filename in os.listdir(directory):
        # 匹配.rhythm文件
        if filename.endswith('.rhythm'):
            rhythm_files.add(filename[:-6])  # 去掉.rhythm后缀
    
    return sorted(list(rhythm_files))

def process_rhythm_file(record_name: str) -> bool:
    """
    处理单个.rhythm文件，检查是否包含"CFE"字符
    
    Args:
        record_name: 记录文件的完整路径（不包含扩展名）
    
    Returns:
        bool: 是否包含"CFE"字符
    """
    try:
        annotation = wfdb.rdann(record_name, 'rhythm')
        # 检查aux_note字段中是否包含"CFE"
        return any('CFE' in note for note in annotation.aux_note) if annotation.aux_note else False
    except Exception:
        print(f"记录 {record_name} 没有有效的.rhythm文件")
        return False

def write_to_csv(record_id: str, writer: csv.writer) -> None:
    """
    将记录ID写入CSV文件
    
    Args:
        record_id: 记录ID
        writer: CSV writer对象
    """
    writer.writerow([record_id])

def main():
    """主程序入口"""
    data_path = os.path.join(CONFIG['data_dir'], CONFIG['database_dir'])
    output_path = os.path.join(CONFIG['data_dir'], CONFIG['output_file'])
    
    # 获取可用的.rhythm
    rhythm_files = get_rhythm_files(data_path)
    
    if not rhythm_files:
        print("未找到任何.rhythm文件")
        return
        
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Record'])  # 添加列头
        
        for record_num in rhythm_files:
            record_id = f'{record_num}'
            record_path = os.path.join(data_path, str(record_num))
            
            print(f"处理记录 {record_id}")
            if process_rhythm_file(record_path):
                write_to_csv(record_id, writer)

if __name__ == '__main__':
    main()