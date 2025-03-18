import os
import glob

# 设置A和B文件夹的路径
folder_a = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/data_proccess/window_picture/vf_picture'
folder_b = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/data_proccess/window_data/vf_windows'
# 定义目标文件夹路径
target_folder = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/data_proccess/舍弃的数据/npy'


# 获取A文件夹中所有文件的前缀
a_prefixes = set()
for file_path in glob.glob(os.path.join(folder_a, '*')):
    filename = os.path.basename(file_path)
    # 假设前缀是文件名的第一部分（可以根据需要调整）
    prefix = os.path.splitext(filename)[0]
    a_prefixes.add(prefix)

# 检查B文件夹中的文件，删除那些前缀不在A中的文件
deleted_count = 0
total_files = 0

for file_path in glob.glob(os.path.join(folder_b, '*')):
    total_files += 1
    filename = os.path.basename(file_path)
    prefix = os.path.splitext(filename)[0]
    
    if prefix not in a_prefixes:
        # 确保目标文件夹存在
        os.makedirs(target_folder, exist_ok=True)
        # 获取目标文件路径
        target_path = os.path.join(target_folder, os.path.basename(file_path))
        
        print(f"移动文件: {file_path} -> {target_path}")
        os.rename(file_path, target_path)
        deleted_count += 1

print(f"处理完成！共检查了 {total_files} 个文件，删除了 {deleted_count} 个不匹配的文件。")
