import numpy as np
from scipy.signal import get_window
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ssqueezepy._ssq_stft import ssq_stft
from scipy.signal import stft, windows

# 整个文件夹一起转换
def extract_features_fsst(input_cell, fs):
    sigs = input_cell  # list:990
    signals_fsst = [None] * len(sigs)
    mean_value = [None] * len(sigs)
    std_value = [None] * len(sigs)
    win = windows.kaiser(250, beta=10)

    # Compute time-frequency maps
    # 遍历所有输入信号
    for idx, sig in enumerate(sigs):
        # 对每个信号进行同步压缩短时傅里叶变换(FSST)
        # s: 变换结果, f: 频率轴
        s, _, _, f = ssq_stft(x=sig, fs=fs, window=win, n_fft=250)  # 原本的输出名 Tx, Sx, ssq_freqs, Sfs
        # (126,5000)(126,5000)(126,)(126,)
        
        # 提取FSST结果的实部和虚部
        Zxx_real = np.real(s)
        Zxx_imag = np.imag(s)

        # 选择0.5Hz到41Hz之间的频率分量
        f_indices = (f > 0.5) & (f < 41)
        # 将实部和虚部在频率维度上堆叠，并保存结果
        signals_fsst[idx] = np.vstack((Zxx_real[f_indices, :], Zxx_imag[f_indices, :]))  # signals_fsst list:990

        # 计算每个信号在时间维度上的均值和标准差
        mean_value[idx] = np.mean(signals_fsst[idx], axis=1)  # mean_value list:990
        std_value[idx] = np.std(signals_fsst[idx], axis=1)
        print(idx)

    # 标准化时频图
    # 计算所有信号的均值的平均值
    mean_value_combined = np.mean(np.column_stack(mean_value), axis=1)  # mean_value_combined ndarray:(40,)
    # 计算所有信号的标准差的平均值
    std_value_combined = np.mean(np.column_stack(std_value), axis=1)  # std_value_combined ndarray:(40,)
    # 定义标准化函数：减去均值后除以标准差
    standardize_fun = lambda x: (x - mean_value_combined[:, None]) / std_value_combined[:, None]

    # 对所有信号进行标准化处理
    signals_fsst = [standardize_fun(signal) for signal in signals_fsst]

    return signals_fsst

# 单个文件转换
def extract_features_fsst_1(input_cell, fs):
    sig = input_cell  # ndarray:(5000,)
    signals_fsst = [None] * 1
    signals_fsst_ = [None] * 1
    win = windows.kaiser(250, beta=10)

    # Compute time-frequency maps

    s, _, _, f = ssq_stft(x=sig, fs=fs, window=win, n_fft=250)  # 原本的输出名 Tx, Sx, ssq_freqs, Sfs
    # (126,5000)(126,5000)(126,)(126,)
    Zxx_real = np.real(s)
    Zxx_imag = np.imag(s)

    f_indices = (f > 0.5) & (f < 21) # 这里要调整到合适的数字让实部虚部的数量加起来是40 之前这个地方是(f > 0.5) & (f < 41)
    signals_fsst[0] = np.vstack(
        (Zxx_real[f_indices, :], Zxx_imag[f_indices, :]))  # len(signals_fsst)= 1 signals_fsst[0].shape=(40,5000)
    # abs_signals_fsst = np.abs(signals_fsst[0])  # abs_signals_fsst.shape=(40,5000)
    #
    mean_value = np.mean(signals_fsst[0], axis=1)  # mean_value.shape = (40,)
    std_value = np.std(signals_fsst[0], axis=1)  # std_value.shape = (40,)
    normalised_signals = (signals_fsst[0] - np.reshape(mean_value, (40, 1))) / (np.reshape(std_value, (40, 1)))
    signals_fsst_[0] = normalised_signals
    return signals_fsst_  # 以list的形式输出，即便里面只有一个元素,这样外面的调用代码就不需要改 signals_fsst_ list:1  signals_fsst_[0]ndarray:(40,5000)


