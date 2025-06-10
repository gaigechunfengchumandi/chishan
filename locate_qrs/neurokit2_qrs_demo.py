import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置数据文件夹和输出文件夹
input_folder = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/分割任务/data/val/'  # 源数据文件夹
output_folder = '/Users/xingyulu/Downloads/neurokit2_verify/z型态类/室颤/'  # 目标保存文件夹


# input_folder = '/Users/xingyulu/Desktop/验证病人/形态类/降噪后数据/前间壁心肌梗死_txt_segment/'  # 源数据文件夹
# output_folder = '/Users/xingyulu/Downloads/neurokit2_verify/z型态类/前间壁心肌梗死_txt_segment/'  # 目标保存文件夹
os.makedirs(output_folder, exist_ok=True)

# 室颤用这个
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        file_path = os.path.join(input_folder, filename)
        data_lead = np.load(file_path)
        signal = data_lead[:, 0]
        _, rpeaks_info = nk.ecg_peaks(signal, sampling_rate=250)
        rpeaks = rpeaks_info['ECG_R_Peaks']  # 从返回的字典中获取R波峰值索引
        plt.figure(figsize=(18, 4)) 
        plt.plot(signal, label='ECG')
        plt.scatter(rpeaks, signal[rpeaks], color='red', label='R peaks')
        plt.legend()
        plt.title(filename)
        save_path = os.path.join(output_folder, filename.replace('.npy', '.png'))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

# 其他用这个
# for filename in os.listdir(input_folder):
#     if filename.endswith('.txt'):
#         file_path = os.path.join(input_folder, filename)
#         data_12_lead = np.loadtxt(file_path)
#         signal = data_12_lead[:, 1]
#         signals, info = nk.ecg_process(signal, sampling_rate=500)
#         rpeaks = info['ECG_R_Peaks']
#         plt.figure(figsize=(18, 4)) 
#         plt.plot(signal, label='ECG')
#         plt.scatter(rpeaks, signal[rpeaks], color='red', label='R peaks')
#         plt.legend()
#         plt.title(filename)
#         save_path = os.path.join(output_folder, filename.replace('.txt', '.png'))
#         plt.savefig(save_path, bbox_inches='tight')
#         plt.close()



