import numpy as np
import os
import pandas as pd
import time
from multiprocessing import Pool as ProcessPool

tic = time.time()
ecgClassSet = ['N', 'S', 'V', 'O', 'AF']
SIG_LEN = 1280

ECG_Data = []
Rlocation = []
RR_intervals = []
Rclass = []


def label_transform_edan(ecg_labels):
    label_out = np.zeros(len(ecg_labels))
    for i in range(len(ecg_labels)):
        if ecg_labels[i] == 1 or ecg_labels[i] == 2 or ecg_labels[i] == 3 or \
                ecg_labels[i] == 11 or ecg_labels[i] == 12 or ecg_labels[i] == 13 or \
                ecg_labels[i] == 25 or ecg_labels[i] == 30 or ecg_labels[i] == 34 or \
                ecg_labels[i] == 35 or ecg_labels[i] == 38 or ecg_labels[i] == 42 or \
                ecg_labels[i] == 43 or ecg_labels[i] == 44:
            label_out[i] = 0 # N

        elif ecg_labels[i] == 4 or ecg_labels[i] == 7 or ecg_labels[i] == 8 or ecg_labels[i] == 9:
            label_out[i] = 1 # S 

        elif ecg_labels[i] == 5 or ecg_labels[i] == 6 or ecg_labels[i] == 10 or \
                ecg_labels[i] == 15 or ecg_labels[i] == 17 or ecg_labels[i] == 41:
            label_out[i] = 2 # V

        elif ecg_labels[i] == 45 or ecg_labels[i] == 46:
            label_out[i] = 4 # AF

        elif ecg_labels[i] == 33 :
            label_out[i] = 5 # VF

        else:
            label_out[i] = 3 # else

    return label_out


def find_file_list(file_folder, key_word):
    filenames = os.listdir(file_folder)
    filename_list = list()
    for filename in filenames:
        if (filename[0] != '.'): # 防止隐藏文件名字包含‘raw’，程序报错
            if (filename.find(key_word) != -1):
                filename_list.append(filename)
    return filename_list


def ecg_merge(data, label, rr, r_first, ecg_len):
    label_all = np.zeros(len(label)+len(rr)+1)
    label_all[:len(label)] = label
    label_all[len(label):len(label)+len(rr)] = rr
    label_all[len(label)+len(rr)] = r_first

    len_ecg_data = data.shape[0]
    dim_ecg_data = data.shape[1]

    ecg_data = np.zeros([ecg_len, dim_ecg_data])
    if len_ecg_data < ecg_len:
        ecg_data[:len_ecg_data] = data
    elif len_ecg_data == ecg_len:
        ecg_data = data
    else:
        ecg_data = data[:ecg_len]
    labels = np.zeros([len(label_all), dim_ecg_data])
    for i in range(dim_ecg_data):
        labels[:, i] = label_all
    out = np.concatenate((ecg_data, labels), axis=0)
    out = np.expand_dims(out, axis=0)
    return out


def cut(j):
    # print(np.sum(ECG_Data[0].astype(int)))
    sig_data = None
    tmp_beat_10 = ECG_Data[int(Rlocation[j]) - int(0.6 * RR_intervals[j - 1]): \
                           int(Rlocation[j + 9]) + int(0.6 * RR_intervals[j + 9])]
    tmp_beat_10 = np.array(tmp_beat_10).astype('int')
    tmp_r_class = Rclass[j:j + 10]  # 原标签，未做转换
    tmp_edan_labels = label_transform_edan(tmp_r_class)
    tmp_rr_10 = RR_intervals[j - 1:j + 9]
    tmp_r_first = int(0.6 * RR_intervals[j - 1])
    sig_data = ecg_merge(tmp_beat_10, tmp_edan_labels, tmp_rr_10, tmp_r_first, SIG_LEN)
    return sig_data


def load_file(file_name, save_path):
    sFileName = file_name
    data = pd.read_table(sFileName,
                         names=['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10', 'ch11', 'ch12'])
    file_start = data.ch1[1]
    if isinstance(file_start, str):
        print(file_start, isinstance(file_start, str))
        ECGdata1 = np.stack([data.ch1[2:], data.ch2[2:], data.ch3[2:], data.ch4[2:], data.ch5[2:], data.ch6[2:],
                            data.ch7[2:], data.ch8[2:], data.ch9[2:], data.ch10[2:], data.ch11[2:], data.ch12[2:]])
    else:
        print(file_start, 'this is num')
        ECGdata1 = np.stack([data.ch1[0:], data.ch2[0:], data.ch3[0:], data.ch4[0:], data.ch5[0:], data.ch6[0:],
                            data.ch7[0:], data.ch8[0:], data.ch9[0:], data.ch10[0:], data.ch11[0:], data.ch12[0:]])
    ECGdata1 = ECGdata1.transpose()
    ECG_Data1 = ECGdata1.astype('float64')
    print('finished read data!!')

    global ECG_Data
    ECG_Data = ECG_Data1

    RInfoName = sFileName.replace('raw', 'ref')
    # 读取有表头的txt数据
    label_txt = pd.read_table(RInfoName, names=['a', 'b', 'c'])
    # len_RInfo = len(label_txt)
    # half_len_RInfo = len_RInfo // 2
    RInfo1 = np.stack([label_txt.a[1:], label_txt.b[1:], label_txt.c[1:]])
    # RInfo1 = np.stack([label_txt.a[1:half_len_RInfo], label_txt.b[1:half_len_RInfo], label_txt.c[1:half_len_RInfo]])
    # RInfo1 = np.stack([label_txt.a[half_len_RInfo:len_RInfo], label_txt.b[half_len_RInfo:len_RInfo], label_txt.c[half_len_RInfo:len_RInfo]])

    RInfo1 = RInfo1.transpose()
    RInfo1.astype('float64')
    print('finished read reference!!')

    global RInfo
    RInfo = RInfo1

    Rlocation1 = np.array(RInfo[:, 0]).astype('float64')
    Rclass1 = RInfo[:, 1]
    RR_intervals1 = np.diff(Rlocation1)

    # 得到R波位置
    global Rlocation
    Rlocation = Rlocation1
    # 得到心拍类型
    global Rclass
    Rclass = Rclass1
    # 计算RR间期
    global RR_intervals
    RR_intervals = RR_intervals1
    # 去掉前后的不稳定数据
    start = 1
    end = 0
    j = start
    Rnum = len(Rclass) - end
    # 截取心拍
    items = []
    record_data_use = []
    for i in range(start, Rnum, 10):
        items.append(i)
    items_use = items[:len(items)-1]

    pool = ProcessPool(processes=8)
    record_data_use = pool.map(cut, items_use)
    pool.close()
    pool.join()
    record_data_use = np.array(record_data_use).reshape((-1, 1301, 12))

    np.save(save_path, record_data_use)
    # print('processing done! delete: ', sFileName)
    # os.remove(data_path + '/' + file)

if __name__ == "__main__":
    # data_folder = '/media/ouyangzhuoran/6E5AF5AA4435F930/动态数据/所有数据整合'
    # save_folder = '/media/ouyangzhuoran/12B26D87B26D7061/文件备份/edan_train'

    # data_folder = '/media/ouyangzhuoran/12B26D87B26D70611/理邦数据库(新更新)/txt'
    # save_folder = '/media/ouyangzhuoran/12B26D87B26D70611/理邦数据库(新更新)/npy'

    # data_folder = '/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第一批数据/数据35例txt'
    # save_folder = '/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第一批数据/数据35例npy'

    # data_folder = '/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第二批数据/110例数据txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/宝安人民医院第二批数据/110例npy'

    # data_folder = '/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第三批数据/80例txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/宝安人民医院第三批数据/80例npy'

    # data_folder = '/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第四批数据/71例txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/宝安人民医院第四批数据/71例npy'

    # data_folder = '/media/ouyangzhuoran/EAGET忆捷/多源性室早txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/多源性室早npy'

    # data_folder = '/media/ouyangzhuoran/EAGET忆捷/深大总医院txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/深大总医院npy'

    # data_folder = '/media/ouyangzhuoran/Data/backup/Baoan128_txt'
    # save_folder = '/media/ouyangzhuoran/Data/backup/Baoan128_npy'

    # data_folder = '/media/ouyangzhuoran/Data/徐州测试txt_104/128hz_txt'
    # save_folder = '/media/ouyangzhuoran/Data/徐州测试txt_104/128hz_npy'

    # data_folder = '/media/ouyangzhuoran/A/宝安测试集2txt'
    # save_folder = '/media/ouyangzhuoran/A/宝安测试集2npy'

    # data_folder = '/media/ouyangzhuoran/A/宝安测试集3txt'
    # save_folder = '/media/ouyangzhuoran/A/宝安测试集3npy'

    # data_folder = '/media/ouyangzhuoran/A/txt/128hztxt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/天津航医-徐州测试数据/128hz'

    # data_folder = '/media/ouyangzhuoran/A/房颤筛选txt128hz'
    # save_folder = '/media/ouyangzhuoran/12B26D87B26D70611/房颤筛选npy128hz'

    # data_folder = '/media/ouyangzhuoran/Data/深大华南医院_128hz_txt'
    # save_folder = '/media/ouyangzhuoran/Data/深大华南医院_128hz_npy'

    # data_folder = '/media/ouyangzhuoran/新加卷/深大华南医院100例txt/128hz_txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/深大华南100例npy/128hz'

    # data_folder = '/media/ouyangzhuoran/新加卷/天津徐州医院数据txt/128hz_txt'
    # save_folder = '/media/ouyangzhuoran/EAGET忆捷/天津航医徐州测试数据/128hz'


    # data_folder = '/mnt/ntfs/数据/202410_Baoan_AF_128hz/4'
    # save_folder = '/media/ouyangzhuoran/KINGSTON/202410_Baoan_AF_128hz_npy'

    data_folder = '/mnt/ntfs/数据/DD49_ClinicalData_128hz'
    save_folder = '/media/ouyangzhuoran/KINGSTON/DD49_ClinicalData_128hz_npy'


    # filenames = find_file_list(data_folder, 'raw')
    # test_name_list = os.listdir('/media/ouyangzhuoran/6E5AF5AA4435F930/EDAN/1280_10RR_processed_ALL/EDAN_train')
    # test_name_list = os.listdir('/media/ouyangzhuoran/12B26D87B26D70611/理邦数据库(新更新)/list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第一批数据/数据35例list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第二批数据/110例数据list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第三批数据/80例list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/12B26D87B26D70611/宝安人民医院第四批数据/71例list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/EAGET忆捷/多源性室早list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/EAGET忆捷/深大总医院list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/Data/徐州测试txt_104/128hz_list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/A/宝安测试集2list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/A/宝安测试集3list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/A/txt/128hzlist')
    # test_name_list = os.listdir('/media/ouyangzhuoran/A/房颤筛选list128hz')
    # test_name_list = os.listdir('/media/ouyangzhuoran/Data/深大华南医院_128hz_list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/新加卷/深大华南医院100例txt/128hz_list')
    # test_name_list = os.listdir('/media/ouyangzhuoran/新加卷/天津徐州医院数据txt/128hz_list')
    # test_name_list = os.listdir('/mnt/ntfs/数据/202410_Baoan_AF_128hz_list/4')
    test_name_list = os.listdir('/mnt/ntfs/数据/DD49_ClinicalData_128hz_list')

    filenames = []
    for i in range(len(test_name_list)):
        filenames.append(test_name_list[i].replace('ref', 'raw'))
    count = 1

    processed_list = os.listdir(save_folder)
    for i in range(len(processed_list)):
        processed_list[i] = processed_list[i].replace('.npy', '.txt')
    filenames = [i for i in filenames if i not in processed_list]

    for filename in filenames:
        data_path = data_folder + os.sep + filename
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = save_folder + os.sep + filename.replace('.txt', '.npy')
        print('processing in: ', filename, count)
        load_file(data_path, save_path)
        count += 1


    toc = time.time()
    print('totally cost:', toc-tic)

'''
test 1 files 
totally cost: 14.152034282684326
'''
# 钟瑞琼
# 陈如花
# 陈守辉