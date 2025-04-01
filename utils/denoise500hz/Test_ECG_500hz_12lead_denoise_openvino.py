import numpy as np
from openvino.inference_engine import IECore
from scipy import signal
import os
import gc
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

readmodel = None

def SNR_1lead(clean_signal, noise_signal):
    try:
        power_clean_signal = np.sum((clean_signal) * (clean_signal))
        power_noise_signal = np.sum((clean_signal - noise_signal) * (clean_signal - noise_signal))
        snr = 10 * math.log((power_clean_signal / (power_noise_signal + 1e-16)), 10)
    except:
        snr = 0
    return snr


def initialize_variable():

    global readmodel
    global net
    global exec_net

    if readmodel == None:
        ie = IECore()
        abs_file = __file__
        folder = os.path.split(abs_file)[0]
        model_xml = "denoise500hz/DenoiseModel_500hz_12lead_1s.xml"
        model_bin = "denoise500hz/DenoiseModel_500hz_12lead_1s.bin"
        net = ie.read_network(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, device_name='CPU')


def predict_ecg_data(ecg_data, sample_rate):

    global readmodel
    global net
    global exec_net
    readmodel = 1

########################################################################################################################

    if (len(ecg_data)) % (sample_rate)==0:
        ecg_data_cut_segment = np.zeros((int(len(ecg_data)/sample_rate), sample_rate, 12))
    else:
        ecg_data_cut_segment = np.zeros((int(len(ecg_data)/sample_rate)+1, sample_rate, 12))

    for i in range(len(ecg_data_cut_segment)):
        if (i+1) * sample_rate <= len(ecg_data):
            ecg_data_cut_segment[i, :, :] = ecg_data[i*sample_rate:((i+1)*sample_rate), :]
        else:
            ecg_data_cut_segment[i, 0:(len(ecg_data)-(i*sample_rate)), :] = ecg_data[(i*sample_rate):len(ecg_data), :]

    ecg_data_re = np.zeros((len(ecg_data_cut_segment), 500, 12))

    if len(ecg_data_cut_segment[0]) == 500:
        ecg_data_re = ecg_data_cut_segment
    else:
        for i in range(len(ecg_data_cut_segment)):
            ecg_data_re[i] = signal.resample(ecg_data_cut_segment[i], 500 * 1)

    del ecg_data_cut_segment
    gc.collect()

########################################################################################################################


    ecg_data_re_noise_segment = ecg_data_re[:, :, :]

    window_length = 125
    fft_num = 256
    win = signal.hamming(window_length, sym=False)
    Overlap = round(0.75 * window_length)
    n_lead_1 = 12
    n_time_1 = 1
    feature_1 = np.zeros([len(ecg_data_re_noise_segment), 2, 129, 18, 12])
    ecg_istft_1 = np.zeros([len(ecg_data_re_noise_segment), 500, 12])
    # ecg_istft_1_temp = np.zeros([sample_rate * n_time_1, 12])

    # print('step1')
    for k in range(len(ecg_data_re_noise_segment)):
        for t in range(n_lead_1):
            f_ecg, t_ecg, Z_ecg = signal.stft(ecg_data_re_noise_segment[k, :, t], fs=500, window=win,
                                              nperseg=window_length, noverlap=Overlap,
                                              nfft=fft_num, return_onesided=True)
            feature_1[k, 0, :, :, t] = np.real(Z_ecg).astype('float64')
            feature_1[k, 1, :, :, t] = np.imag(Z_ecg).astype('float64')

        # feature = np.zeros([1, 2, 33, 9, 8])
        feature = np.zeros([1, 2, 129, 18, 12])

        # feature[0, 0, :, :, :] = feature_1[k, :, :, :, 0]
        # feature[0, 1, :, :, :] = feature_1[k, :, :, :, 1]

        feature[0, :, :, :, :] = feature_1[k, :, :, :, :]

        input_blob = next(iter(net.input_info))
        output_info = net.outputs
        output_blob = next(iter(output_info))

        feature_pred_ori = exec_net.infer(inputs={input_blob: feature})

        feature_pred_1 = feature_pred_ori[output_blob]
        # print('step2')
        for t in range(n_lead_1):
            feature_pred_real = feature_pred_1[0, 0, :, :, t]
            feature_pred_imag = feature_pred_1[0, 1, :, :, t]
            re_spectrum = feature_pred_real + 1.0j * feature_pred_imag
            # 进行istft变换
            t_istft, ecg_istft_data = signal.istft(re_spectrum, fs=500, window=win,
                                                       nperseg=window_length, noverlap=Overlap, nfft=fft_num)
            ecg_istft_1[k, :, t] = ecg_istft_data[:500]

        SNR_Ilead = SNR_1lead(ecg_istft_1[k, :, 0], ecg_data_re_noise_segment[k, :, 0])
        SNR_IIlead = SNR_1lead(ecg_istft_1[k, :, 1], ecg_data_re_noise_segment[k, :, 1])
        SNR_IIIlead = SNR_1lead(ecg_istft_1[k, :, 2], ecg_data_re_noise_segment[k, :, 2])

        if np.all(ecg_data_re_noise_segment[k, :, 0] == 0) or np.all(ecg_data_re_noise_segment[k, :, 1] == 0) or np.all(ecg_data_re_noise_segment[k, :, 2] == 0):
            leadI = ecg_istft_1[k, :, 0]
            leadII = ecg_istft_1[k, :, 1]
            leadIII = ecg_istft_1[k, :, 2]
        else:
            if SNR_Ilead < SNR_IIlead and SNR_Ilead < SNR_IIIlead:
                leadI = ecg_istft_1[k, :, 1] - ecg_istft_1[k, :, 2]
                ecg_istft_1[k, :, 0] = leadI
            elif SNR_IIlead < SNR_Ilead and SNR_IIlead < SNR_IIIlead:
                leadII = ecg_istft_1[k, :, 0] + ecg_istft_1[k, :, 2]
                ecg_istft_1[k, :, 1] = leadII
            elif SNR_IIIlead < SNR_Ilead and SNR_IIIlead < SNR_IIlead:
                leadIII = ecg_istft_1[k, :, 1] - ecg_istft_1[k, :, 0]
                ecg_istft_1[k, :, 2] = leadIII

        dataAVR = -(ecg_istft_1[k, :, 1] + ecg_istft_1[k, :, 0]) / 2
        dataAVL = ecg_istft_1[k, :, 0] - ((ecg_istft_1[k, :, 1]) / 2)
        dataAVF = ecg_istft_1[k, :, 1] - ((ecg_istft_1[k, :, 0]) / 2)

        ecg_istft_1[k, :, 3] = dataAVR
        ecg_istft_1[k, :, 4] = dataAVL
        ecg_istft_1[k, :, 5] = dataAVF

    ecg_istft_denoise = np.zeros([len(ecg_data_re_noise_segment), sample_rate*n_time_1, 12])

    for k in range(len(ecg_data_re_noise_segment)):
        if sample_rate != 500:
            ecg_istft_denoise[k] = signal.resample(ecg_istft_1[k], sample_rate)
        else:
            ecg_istft_denoise[k] = ecg_istft_1[k]

    for k in range(len(ecg_data_re_noise_segment)):
        if (k+1)*sample_rate <= len(ecg_data):
            ecg_data[(k*sample_rate):((k+1)*sample_rate), :] = ecg_istft_denoise[k]
        else:
            ecg_data[(k*sample_rate):len(ecg_data), :] = ecg_istft_denoise[k, 0:(len(ecg_data)-(k*sample_rate)), :]

########################################################################################################################

    del feature_1
    del feature_pred_1
    del ecg_istft_1
    gc.collect()
    return ecg_data


if __name__ == "__main__":

    data = pd.read_table('/home/devel/data', names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12'])
    ECGdata1 = np.stack([data.ch1[2:], data.ch2[2:], data.ch3[2:], data.ch4[2:], data.ch5[2:], data.ch6[2:], data.ch7[2:], data.ch8[2:], data.ch9[2:], data.ch10[2:], data.ch11[2:], data.ch12[2:]])
    ECGdata1 = ECGdata1.transpose()
    ecg_data = ECGdata1.astype('float64')
    ecg_data = ecg_data/200
    # ecg_data = np.zeros([709120, 12])

    plt.plot(ecg_data[:, 0])
    plt.show()
    sample_rate = 128
    # del data
    # del ECGdata1
    gc.collect()

    T1 =time.time()
    initialize_variable()

    denoise_ecg_data = predict_ecg_data(ecg_data, sample_rate)


    plt.plot(denoise_ecg_data[:, 0])
    plt.show()

    T2 = time.time()

    print(T2 -T1)

    print('finish Analyze')

