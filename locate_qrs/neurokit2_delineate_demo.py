import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

data_12_lead = np.loadtxt('/Users/xingyulu/Desktop/验证病人/节律类/原数据/室速_txt_segment/ca0eb4c9-17ae-4e40-b917-bb4e7ffbf2e8_00000.txt')
ecg_signal = data_12_lead[:, 1]
SAMPLING_RATE = 500
# 生成模拟ECG信号
# signal = nk.ecg_simulate(duration=10, sampling_rate=500)

# QRS波检测
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=500)
# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'], ecg_signal)
# Zooming into the first 5 R-peaks
plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:5], ecg_signal[:5 * SAMPLING_RATE])
plt.show()
# P、QRS、T波起止点检测


# Delineate the ECG signal and visualizing all peaks of ECG complexes
# nk.ecg_delineate 这个函数本身在参数 show=True 时，会自动调用 matplotlib 的绘图功能，把图画出来。
# _, waves_peak = nk.ecg_delineate(ecg_signal, 
#                                  rpeaks, 
#                                  sampling_rate=500, 
#                                  method="peak", 
#                                  show=True, 
#                                  show_type='peaks')
# plt.show()


# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
# _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=500, method="peak")
# plot = nk.events_plot([
#     waves_peak['ECG_T_Peaks'][:3],   # 取前3个T波波峰索引
#     waves_peak['ECG_P_Peaks'][:3],   # 取前3个P波波峰索引
#     waves_peak['ECG_Q_Peaks'][:3],   # 取前3个Q波波峰索引
#     waves_peak['ECG_S_Peaks'][:3]    # 取前3个S波波峰索引
# ], ecg_signal[:5 * 500])             # 只显示前5秒（5*500采样点）的ECG信号
# plt.show()








