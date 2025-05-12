import os
import csv
from pathlib import Path
import re

def natural_sort_key(s):
    # 实现自然排序的关键函数
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', str(s))]

def process_files(input_dir, output_csv):
    # 获取A文件夹下所有文件并按自然顺序排序
    files = sorted(Path(input_dir).glob('*'), key=natural_sort_key)
    
    # 提取文件名前缀（去掉扩展名）
    prefixes = [f.stem for f in files]
    
    # 创建并写入CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入每行数据
        for prefix in prefixes:
            # 第一列是文件名前缀，第2-11列是数字4
            row = [prefix] + [4] * 10
            writer.writerow(row)

if __name__ == "__main__":
    input_dir = "/Users/xingyulu/Public/afafaf/Holter_Data_3例/Check_AF_N"  # 输入文件夹路径
    output_csv = "/Users/xingyulu/Public/afafaf/Holter_Data_3例/output.csv"  # 输出CSV文件路径
    process_files(input_dir, output_csv)