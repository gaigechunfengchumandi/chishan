# 这个文件是对10s数据进行降噪处理，然后保存为npy格式，用来降噪监护的数据，1导联



import sys
sys.path.append('/Users/xingyulu/Public/physionet/utils')

import os   
import numpy as np
from denoise500hz.Test_ECG_500hz_12lead_denoise_openvino import initialize_variable as init_denoise500hz, predict_ecg_data as denoise_500hz



def down_sample_to(array_ori, target_points):
    sampling_interval = array_ori.shape[0] / target_points
    sampled_indices = np.arange(0, array_ori.shape[0], sampling_interval).astype(int)
    downsampled_points = array_ori[sampled_indices, :]
    return downsampled_points

def main():
    # 读取命令行参数
    input_dir = '/Users/xingyulu/Public/afafaf/推理尝试/data-现在直接用训练尝试文件夹里的test'
    output_dir = '/Users/xingyulu/Public/afafaf/推理尝试/data-现在直接用训练尝试文件夹里的test'

    # 加载模型
    init_denoise500hz()

    for file_name in sorted(os.listdir(input_dir)):
        if file_name == '.DS_Store':
            continue

        # 构建输入文件的完整路径
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.npy') # 
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 读取数据预测
        data = np.load(input_file)[:,0] # (2500,)
        
        # 将数据从2500点重采样到5000点
        data_resampled = np.interp(np.linspace(0, 1, 5000), np.linspace(0, 1, len(data)), data)
        
        # 将数据reshape为(5000, 12)
        array_ori = np.tile(data_resampled[:, np.newaxis], (1, 12)) # (5000, 12)
        array_500hz = down_sample_to(array_ori, 5000) # Downsample the array to 500Hz（5000，12）
        # label_12 = segment_2s(array_500hz, array_500hz) # Segment the array
        # 降噪处理
        array_denoise = denoise_500hz(array_500hz, 500)
        
        # 将降噪后的数据从(5000, 12)裁剪回(5000, 1)
        array_denoise = array_denoise[:, 0].reshape(5000, 1)
        
        # 将数据从5000点降采样到2500点
        array_denoise = array_denoise[::2, :]
        
        np.save(output_file, array_denoise)
        print(f"Processed file: {file_name}")



if __name__ == "__main__":
    main()