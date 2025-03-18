import os
import numpy as np
import shutil
from pathlib import Path

def check_nans_in_npy(directory, nan_files_dir):
    """
    遍历目录检查所有npy文件中的nan值，并将包含nan的文件移动到指定目录
    
    Args:
        directory: 要检查的目录
        nan_files_dir: 存放含nan文件的目录
    """
    # 存储含有nan的文件路径和列信息
    files_with_nans = []
    
    # 确保目标目录存在
    Path(nan_files_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                try:
                    # 加载npy文件
                    data = np.load(file_path)
                    has_nan = False
                    
                    # 检查每一列是否包含nan
                    for col in range(data.shape[1]):
                        if np.isnan(data[:, col]).any():
                            has_nan = True
                            nan_rows = np.where(np.isnan(data[:, col]))[0]
                            files_with_nans.append((file_path, col))
                            print(f"发现nan值: {file}, 出现在第{col}列, 第{nan_rows.tolist()}行")
                    
                    # 如果文件包含nan，移动到指定目录
                    if has_nan:
                        dest_path = os.path.join(nan_files_dir, file)
                        shutil.move(file_path, dest_path)
                        print(f"已移动文件: {file} -> {dest_path}")
                        
                except Exception as e:
                    print(f"处理文件时出错 {file}: {str(e)}")

    return files_with_nans

if __name__ == "__main__":
    # 设置要检查的目录路径
    directory_to_check = "/Users/xingyulu/Public/监护心电预警/公开数据/室颤/分割任务/transition_segments"
    # 设置存放含nan文件的目录
    nan_files_directory = "/Users/xingyulu/Public/监护心电预警/公开数据/室颤/分割任务/nan_files"
    
    print("开始检查npy文件中的nan值...")
    files_with_nans = check_nans_in_npy(directory_to_check, nan_files_directory)
    
    if files_with_nans:
        unique_files = len(set(file_path for file_path, _ in files_with_nans))
        print(f"\n共有{unique_files}个文件包含nan值，已移动到 {nan_files_directory}")
    else:
        print("\n未发现包含nan值的文件")
