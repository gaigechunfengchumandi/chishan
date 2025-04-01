import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(segments_path, output_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    将数据分割成训练集、验证集和测试集
    
    参数:
        segments_path: 原始数据路径
        output_dir: 输出目录路径
        test_size: 测试集比例
        val_size: 验证集比例(占训练集的比例)
        random_state: 随机种子
    """
    # 创建输出目录
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 获取所有npy文件
    all_files = [f for f in os.listdir(segments_path) if f.endswith('.npy')]
    
    # 第一次分割：分出测试集
    train_val_files, test_files = train_test_split(
        all_files, test_size=test_size, random_state=random_state)
    
    # 第二次分割：从剩余数据中分出验证集
    train_files, val_files = train_test_split(
        train_val_files, test_size=val_size, random_state=random_state)
    
    # 复制文件到对应目录
    def copy_files(files, dest_dir):
        for f in files:
            src = os.path.join(segments_path, f)
            dst = os.path.join(dest_dir, f)
            shutil.copy2(src, dst)
    
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)
    
    print(f"数据集分割完成:")
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    print(f"测试集: {len(test_files)} 个文件")

if __name__ == "__main__":
    segments_path = '/Users/xingyulu/Public/afafaf/处理第一例/segments'
    output_dir = '/Users/xingyulu/Public/afafaf/处理第一例'
    
    split_dataset(segments_path, output_dir)