import os
import numpy as np
import csv

def count_4_in_last10_per_subfolder(folder):
    results = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            sub_count = 0
            for file in os.listdir(subfolder_path):
                if file.endswith('.npy'):
                    npy_path = os.path.join(subfolder_path, file)
                    try:
                        arr = np.load(npy_path)
                        last10 = arr.flatten()[-10:]
                        count_4 = np.sum(last10 == 4)
                        sub_count += count_4
                    except Exception as e:
                        print(f"读取{npy_path}出错: {e}")
            print(f"{subfolder}: 房颤心拍的数量 = {sub_count}")
            results.append([subfolder, sub_count])
    # 写入CSV文件
    with open('count_result.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['子文件夹', '房颤心拍数量'])
        writer.writerows(results)

if __name__ == "__main__":
    folder = "你的大文件夹路径"
    count_4_in_last10_per_subfolder(folder)