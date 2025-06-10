import numpy as np
from openvino.inference_engine import IECore
from scipy import signal
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import math


# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

readmodel = None

# region 不在segment_2s里面被调用的代码
# 这个函数在doctor.py里调用，初始化的时候就初始化模型
def initialize_variable_s():

    global readmodel
    global net
    global exec_net

    if readmodel == None:
        ie = IECore()
        abs_file = __file__
        folder = os.path.split(abs_file)[0]
        model_xml = 'segment_500hz/dualpath_singlelead_2s.xml'#
        model_bin = 'segment_500hz/dualpath_singlelead_2s.bin'#
        net = ie.read_network(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, device_name='CPU')


# endregion

# region ---------------------------------------------------------------- 分割的代码
        
# region 模型处理代码
#直接对openvino模型的调用， 输入的形状必须是(9,1008,1)，输出的形状是(9.1008,4) 该函数被go_throught_model调用 (9)是片段数
def openvino_predict(ecg_data):

    global readmodel
    global net
    global exec_net
    readmodel = 1

    # Assign the input ECG data to the feature variable
    feature = ecg_data #(9,1008,1)

    # Get the input and output blobs of the network
    input_blob = next(iter(net.input_info))
    output_info = net.outputs
    output_blob = next(iter(output_info))

    # Perform inference on the input data
    feature_pred_ori = exec_net.infer(inputs={input_blob: feature})

    # Extract the output predictions
    feature_pred_1 = feature_pred_ori[output_blob]

    # Clean up to free memory
    del feature_pred_ori
    gc.collect()
    
    return feature_pred_1
#以10s，12导联的样本为单位先做滑窗处理然后调用openvino_predict函数，得到带重叠的片段输出， in (5008,12)，out (12,9,1008)
def go_throught_model(unlabel_data):
    slice_data =[] # 用于存放滑窗分割出来的2s片段
    pred_12_lead = [] # 用于存放预测结果，12导联分别的到9个预测结果，一个个地往这个列表添加
    step = 500 # 滑窗步长
    max_segment_num = int((5008-1008)//step)
    for i in range(max_segment_num+1):
        win_start_idx = i*step
        win_end_idx = win_start_idx + 1008
        slice_data.append(unlabel_data[win_start_idx:win_end_idx])
    slice_data = np.array(slice_data)# 这里的形状是（9,1008, 12）(9)是片段数，(12)是12导联
    for lead_idx in range(12):# 把导联一个一个取出来，推理完之后再合回来
        data_clip2s_single_lead = slice_data[:,:,lead_idx:lead_idx+1] #lead_idx:lead_idx+1可以保持切片后仍人保持3维的（9,1008,12）->(9,1008,1)
        pred_single_lead = openvino_predict(data_clip2s_single_lead) #ndarray(9,1008,4) 4是4个类的概率值
        # pred_single_lead = np.argmax(pred_single_lead,axis=2)#ndarray(9,1008)把每格采样点的预测概率最大的索引取到
        pred_12_lead.append(pred_single_lead)
    pred_12_lead = np.array(pred_12_lead)# (12,9,1008,4) 对应（12导联，9片段，1008个点, 4个类的概率值）9个片段会在外面使用extract_wave函数提取并拼接
    return pred_12_lead
# endregion

# region 整理导联，滑窗，频率
# 把有重叠的滑窗片段拼接好 in (9,1008,4) out (1,5008,4)
def concatenate_windows(model_out):
    discard_length = 100 #窗口末尾舍弃的长度
    step = 500 # 滑窗步长
    combine_modelout = []
    # 使用一个滑窗策略，把原本有重叠的n个片滑窗预测结果掐头去尾后合并起来，无重叠地拼接成一个长度5008的标签：combine_modelout
    combine_modelout.extend(model_out[0][:-discard_length])#先把第一段的除了discard_length的地方取到
    for i in range(1,len(model_out)-1): # 后面的n个段，负责把前一个片段末尾的discard_length个点补回去，然后新预测一段区域接在后面
        combine_modelout.extend(model_out[i][1008-discard_length-step:1008-discard_length])#起点：窗口右侧端点-舍弃长度-步长 终点：窗口右侧端点-舍弃长度
    combine_modelout.extend(model_out[len(model_out)-1][1008-discard_length-step:1008])#最后一个片段不舍弃，全部要
    return combine_modelout

# 遍历12个导联，每导联调用一次concatenate_windows函数 in (12,9,1008,4)      out (12,5000,4)
def splice_windows(pred_12_lead):
    combine_result_12_lead = []
    for lead_idx in range(12):
        model_out = pred_12_lead[lead_idx]
        combine_result = concatenate_windows(model_out)
        combine_result_12_lead.append(combine_result)# （n,5008)
    combine_result_12_lead = np.array(combine_result_12_lead)# (12,5008,4)12个导联的分割结果

    # 去掉每一行的之前多加的8个点
    trimmed_label = combine_result_12_lead[:, :-8, :]# (12,5000,4)
    return trimmed_label

# endregion

# region 利用信噪比抑制噪声过大的心拍
#  计算信噪比的工具函数
def SNR_tool(clean_signal, noise_signal):
    try:
        power_clean_signal = np.sum((clean_signal) * (clean_signal))
        power_noise_signal = np.sum((clean_signal - noise_signal) * (clean_signal - noise_signal))
        snr = 10 * math.log((power_clean_signal / (power_noise_signal + 1e-16)), 10)
    except:
        snr = 0
    # 保留两位小数
    snr = round(snr, 2)
    return snr

# 以QRS波的起点开始划分波形区域，输出n个段和每个段的起止点索引
def split_by_twos_with_indices(lst):
    segments = []
    start_indices = []

    # 找到所有 "2" 段的起始位置
    for i in range(len(lst) - 1):
        if lst[i] == 2 and (i == 0 or lst[i - 1] != 2):  # 当前元素为2，且前一个元素不为2
            start_indices.append(i)

    # 根据起始位置分割列表并记录索引范围
    for i in range(len(start_indices)):
        start = start_indices[i]
        end = start_indices[i + 1] if i + 1 < len(start_indices) else len(lst)
        segment = lst[start:end]
        segments.append((segment, start, end - 1))  # 添加段内容及其起始和结束索引

    return segments

# 把原始信号和降噪后信号按上面的n段划分，然后一一计算他们的SNR，再定位到snr值小于0的心拍的位置，返回一个字典
def beat_snr(data_12_clean, data_12_ori, label_12):
    snr_matrix = []
    record_for_supress = {}
    for lead_idx in range(12):
        label_1 = label_12[lead_idx]
        data_1_clean = data_12_clean[:, lead_idx]
        data_1_ori = data_12_ori[:, lead_idx]
        beat_segments = split_by_twos_with_indices(label_1)
        # 把data_1_clean和data_1_ori分成n个段，每个段是一个心搏的数据，然后计算每个段的信噪比
        snr_lead = []
        for beat, (segment, start, end) in enumerate(beat_segments):
            snr = SNR_tool(data_1_clean[start:end + 1], data_1_ori[start:end + 1])
            snr_lead.append(snr)
            if snr < 0:
                record_for_supress.setdefault(lead_idx, []).append((start, end))# 如果信噪比小于0的，记录下来他的起始和结束位置
        snr_matrix.append(snr_lead) # snr值小于0的心拍的位置 这是一个矩阵

    return record_for_supress

# 把snr值小于0的心拍的位置，把他们这些位置的所有概率值全部置为0
def beat_suppression(label_12, record_for_setdefault):
    for lead_idx, segments in record_for_setdefault.items():
        for start, end in segments:
            label_12[lead_idx][start:end + 1] = 0 # 这步已经无法在原始数据上体现，只能在预测结果上体现
    return label_12

# 抑制操作主函数 (12,5000,4) in & out 
def process_and_suppress_beats(result_in_probability, denoise_data, original_data):
    # 先对第三维度求最大值索引,用于计算心拍信噪比
    result_in_index = np.argmax(result_in_probability, axis=2) # (12,5000)
    record_for_supress = beat_snr(denoise_data, original_data, result_in_index) # snr值小于0的心拍的位置，这是一个字典
    result_in_probability = beat_suppression(result_in_probability, record_for_supress) # in (12,5000,4)  |  out (12,5000,4)

    result_in_probability[3:6,:,:] = 0 # 把avL，avR，avF的概率值都设为0，不让他们影响其他导联
    return result_in_probability
# endregion

# region 各导联分割结果对齐 in (12, 5000, 4)    out (12, 5000)
def align_points(result_12_lead, activate = True):
    if activate:

        # 创建一个数组来储存对齐之后标签，形状为 (12, 5000)
        max_indices = np.zeros((12, 5000), dtype=int)

        # 对每个采样点执行12个导联的值比较操作，取最大值的索引覆盖所有导联
        for i in range(5000):
            # 获取在采样点i的每个类别（除了背景类）在12个导联中的概率值，并把这些概率值按类别分别排序
            top3_per_class = np.sort(result_12_lead[:, i, -3:], axis=0)[-3:] # 取每个类别在导联纬度top3大的值（3，3）
            # 求top3的平均值
            top3_mean = np.mean(top3_per_class, axis=0) # (3,)
            # 找到3个类别里的概率最大值
            max_class_value = np.max(top3_mean)
            if max_class_value > 0.5:# 如果最大值大于0.5，就找到这个最大值属于哪一个类
                max_class_index = np.argmax(top3_mean)+1 # 因为是从第二个类，P波开始的，所以要加1
            else:
                max_class_index = 0
            # 把这个类索引赋值给所有导联
            max_indices[:, i] = max_class_index
    else:
        max_indices = np.argmax(result_12_lead, axis=2)# (12,5000) 12个导联的分割结果不对齐
    return max_indices

def align_points_2backup(result_12_lead, activate = True):
    if activate: # 这个是根据类别之间的top3之间的比较，决定用谁来覆盖

        # 创建一个数组来储存对齐之后标签，形状为 (12, 5000)
        max_indices = np.zeros((12, 5000), dtype=int)

        # 对每个采样点执行12个导联的值比较操作，取最大值的索引覆盖所有导联
        for i in range(5000):
            # 获取在采样点i的每个类别在12个导联中的概率值，并把这些概率值按类别分别排序
            top3_per_class = np.sort(result_12_lead[:, i, :], axis=0)[-3:] # 取每个类别在导联纬度top3大的值（3，4）
            # 求平均值
            top3_mean = np.mean(top3_per_class, axis=0) # (4,)

            # 比较4类的top3平均值中的最大值，再找到这个最大值属于哪一个类。
            max_class_index = np.argmax(top3_mean)

            # 把这个类索引赋值给所有导联
            max_indices[:, i] = max_class_index
    else:
        max_indices = np.argmax(result_12_lead, axis=2)# (12,5000) 12个导联的分割结果不对齐
    return max_indices

def align_points_bakup(result_12_lead, activate = True):
    if activate:# 这个是按单独的点的概率值比较得出结果
        # 创建一个数组来储存对齐之后标签，形状为 (12, 5000)
        max_indices = np.zeros((12, 5000), dtype=int)
        # 对每个采样点执行12个导联的值比较操作，取最大值的索引覆盖所有导联
        for i in range(5000):
            # 获取每个导联在采样点i的4个概率值中的最大值
            max_per_lead = np.max(result_12_lead[:, i, :], axis=1)       # 每个导联在该点的最大概率值（12，）

            # 比较12个最大值中的最大值，再找到这个最大值属于哪一个类索引。
            max_lead = np.argmax(max_per_lead)
            max_class_index = np.argmax(result_12_lead[max_lead, i, :])

            # 把这个类索引赋值给所有导联
            max_indices[:, i] = max_class_index
    else:
        max_indices = np.argmax(result_12_lead, axis=2)# (12,5000) 12个导联的分割结果不对齐
    return max_indices
# endregion

# region 加 0 padding
def zero_box_padding(data_12):
    ecg_zero_box = np.concatenate((data_12,np.zeros((8,12))),axis=0) # 在后面补8个0，用于适配滑窗截取，(5008,12)
    return ecg_zero_box
# endregion

# 分割的主函数
def segment_2s(denoise_data, original_data):
    ecg_zero_box = zero_box_padding(denoise_data) # in (5000,12)   |  out (5008,12)
    overlapping_clips = go_throught_model(ecg_zero_box)# in (5008,12)   |  out (12,9,1008,4)(12导联,9个有重叠片段,每段长度1008)
    # 循环12个导联，得到12个导联的滑窗拼接好的标签
    result_in_probability = splice_windows(overlapping_clips)# in(12,9,1008,4)  |  out(12,5000,4) 12个导联的分割结果

    # 抑制噪声过大的心拍的概率值
    result_in_probability = process_and_suppress_beats(result_in_probability, denoise_data, original_data)

    # 对齐所有导联的分割点
    result_align = align_points(result_in_probability, activate = True) #(12,5000) 12个导联的分割结果对齐操作可以不激活，但是这个函数要过一遍
    
    # result_align = temp_change(result_align, q_h = 0,  num = 10) # 往前改用1，往后改用-1
    
    # 转置成（5000，12）    
    label_12 = result_align.transpose() 


    return label_12 #（5000，12）


# region 临时改变标签
def temp_change(data, q_h = 1, num=10):
    if q_h == 1: # 往前改
        for idx, lst in enumerate(data):
            for i in range(len(lst)):
                if lst[i] == 2:
                    # 找到2的起始点
                    start_index = i
                    # 检查是否存在足够的0在2之前，如果有就修改
                    if start_index >= num and all(lst[start_index-num:start_index] == [0]*num):
                        lst[start_index-num:start_index] = [2]*num
                    
            # 把修改后的 lst 放回 data
            data[idx] = lst
    else: # 往后改
        for idx, lst in enumerate(data):
            for i in range(len(lst)):
                if lst[i] == 2:
                    # 找到2的起始点
                    start_index = i
                    # 检查是否存在足够的0在2之后，如果有就修改
                    if start_index + num < len(lst) and all(lst[start_index+1:start_index+num+1] == [0]*num):
                        lst[start_index+1:start_index+num+1] = [2]*num
                    
            # 把修改后的 lst 放回 data
            data[idx] = lst

    return data
# endregion

'''
对齐之后的流程图
segment_2s
    ├──> go_throught_model
    │       ├──> 1. data_clip2s_single_lead (9, 1008, 1)    //(9)是片段数
    │       ├──> 2. openvino_predict -> pred_single_lead (9, 1008, 4)   //(4)是4个类别的概率值
    │       └──> 3. 12导联推理完 -> overlapping_clips (12, 9, 1008, 4) //(12导联,9个有重叠片段,每段长度1008, 4个类别的概率值)
    │
    ├──> splice_windows
    │        └──> 调用 concatenate_windows (拼接滑窗) 输出 combine_result_12_lead (12, 5000, 4)
    │
    ├──> process_and_suppress_beats 
    │        ├──> 1.调用 split_by_twos_with_indices 把qrs波的位置分成n个区域
    │        ├──> 2.调用 beat_snr 定位到SNR值小于0的心拍的位置
    │        └──> 3.调用 beat_suppression 抑制噪声过大的心拍的概率值
    │
    └──> align_points
            └──> 对齐所有导联的分割点(12,5000)
            
            
            
改之前的流程图
segment_2s
    ├──> go_throught_model 
    │       ├──> 1. data_clip2s_single_lead (9, 1008, 1)    //模型的输入 (9)是片段数
    │       ├──> 2. openvino_predict -> pred_single_lead (9, 1008, 4)     //模型的输出 (4)是4个类别的概率值
    │       ├──> 3. 取最大值索引 -> pred_single_lead (9, 1008) //(9)是片段数
    │       └──> 4. 12导联推理完 -> overlapping_clips (12, 9, 1008) (12导联,9个有重叠片段,每段长度1008)
    │               │
    │               └──> 输出传递至 loop_12_lead 准备拼接滑窗
    │
    └──> loop_12_lead
            └──> 调用 concatenate_windows (拼接滑窗) 输出 combine_result_12_lead (12,5000)         
'''


# endregion ------------------------------------------------------------- 分割的代码




# region ----------------------------------------------------------参数提取处理代码

# region 返回单导联的标签，单导联的电压信号，单导联在3个特征波出现处的电压信号，单导联的3个特征波的索引
def extract_wave(data, label):

    label_1 = label.copy()  # 单根导联的标签(5000,)
    data_1 = data.copy()  # 单根导联的电压信号(5000,)

    # 初始化特征波的电压信号，全都赋值为-10
    data_lead_p = np.full_like(data_1, -10)
    data_lead_qrs = np.full_like(data_1, -10)

    data_lead_t = np.full_like(data_1, -10)

    # 给特征波的索引赋值
    label_p = np.where(label_1 == 1)[0].tolist()
    label_qrs = np.where(label_1 == 2)[0].tolist()
    label_t = np.where(label_1 == 3)[0].tolist()

    # 插入缺失点
    label_p = insert_missing_points(label_p)
    label_qrs = insert_missing_points(label_qrs)
    label_t = insert_missing_points(label_t)

    # 过滤掉长度小于指定阈值的连续段
    label_p = remove_short_continuous_segments(label_p)
    label_qrs = remove_short_continuous_segments(label_qrs)
    label_t = remove_short_continuous_segments(label_t)

    # 将特征波的电压信号赋值为原始数值
    data_lead_p[label_p] = data_1[label_p]
    data_lead_qrs[label_qrs] = data_1[label_qrs]
    data_lead_t[label_t] = data_1[label_t]

    return data_lead_p, data_lead_qrs, data_lead_t, label_p, label_qrs, label_t

def insert_missing_points(label_x, missing_threshold=30):# 插入缺失点
    miss_value = []
    for i in range(len(label_x)-1):
        if 0 < label_x[i+1]-label_x[i] < missing_threshold: # 如果两个标签之间的间隔大于0小于30
            miss_range = range(label_x[i]+1, label_x[i+1]) # 两个标签之间的间隔
            miss_value.extend(miss_range)
    for value in miss_value:
        label_x.insert(label_x.index(value-1)+1, value)# 在前一个值的索引位置+1的位置插入这个值

    return label_x

def remove_short_continuous_segments(label_x, length_threshold=20):# 过滤掉长度小于指定阈值的连续段
    """
    过滤掉长度小于指定阈值的连续段。
    
    Args:
        label_x (list): 包含所有点的列表，假定点是已排序且唯一的。
        length_threshold (int): 连续段的最小长度阈值。长度小于该值的段会被删除。
    
    Returns:
        list: 去除短连续段后的新列表。
    """
    filtered_label_x = []
    current_segment = [label_x[0]] if label_x else []

    for i in range(1, len(label_x)):
        # 如果当前点与前一个点是连续的
        if label_x[i] == label_x[i - 1] + 1:
            current_segment.append(label_x[i])
        else:
            # 如果当前段长度大于等于阈值，将其加入过滤后的列表
            if len(current_segment) >= length_threshold:
                filtered_label_x.extend(current_segment)
            # 开始一个新的连续段
            current_segment = [label_x[i]]

    # 检查最后一个段
    if len(current_segment) >= length_threshold:
        filtered_label_x.extend(current_segment)

    return filtered_label_x
# endregion 

# region 返回每个波段的连续长度， onset的索引和值
def find_consecutive_lengths(label_x, data_lead_x):
    lengths = []
    current_length = 1

    for i in range(1, len(label_x)):
        if label_x[i] == label_x[i - 1] + 1:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1

    lengths.append(current_length)
    lengths = [length * 2 for length in lengths]  # Convert lengths to milliseconds

    onset_index = []
    onset_value = []
    for i in range(len(label_x)):
        if i == 0 or label_x[i] != label_x[i - 1] + 1:
            onset_index.append(label_x[i])
            onset_value.append(data_lead_x[label_x[i]])
    return onset_index, onset_value, lengths
# endregion

# region 返回每个波的峰谷值
def found_peaks(data_lead, onset_values):
    # Initialize empty lists to store peak and valley coordinates
    peak_indices, peak_values, valley_indices, valley_values, peak_amplitudes, valley_amplitudes = [], [], [], [], [], []
    # Initialize variables to track current peak and valley
    current_peak, current_peak_index = None, None
    current_valley, current_valley_index = None, None

    wave_regions = split_wave_regions(data_lead)# 把信号按心搏分成n个区域
    
    # Iterate over the data_lead_x array
    for region in wave_regions: # 遍历每个区域
        # 找到区域中的最大值（峰值）及其索引
        current_peak_index, current_peak = max(region, key=lambda x: x[1])
        # 找到区域中的最小值（谷值）及其索引
        current_valley_index, current_valley = min(region, key=lambda x: x[1])

        peak_indices.append(current_peak_index)
        peak_values.append(current_peak)
        valley_indices.append(current_valley_index)
        valley_values.append(current_valley)

        peak_amplitudes = [y - onset_y for y, onset_y in zip(peak_values, onset_values)]
        valley_amplitudes = [y - onset for y, onset in zip(valley_values, onset_values)]
    
    return peak_indices, peak_amplitudes, valley_indices, valley_amplitudes

def qrs_found_peaks(data_lead, onset_values):
    # Initialize empty lists to store peak and valley coordinates
    q_indices, q_values, q_amplitudes, \
        r_indices, r_values, r_amplitudes, \
            s_indices, s_values, s_amplitudes = [], [], [], [], [], [], [], [], []
    # Initialize variables to track current peak and valley
    current_peak, current_peak_index = None, None
    current_valley, current_valley_index = None, None

    wave_regions = split_wave_regions(data_lead)# 把信号按心搏分成n个区域
    
    # Iterate over the data_lead_x array
    for region in wave_regions: # 遍历N个心搏区域
        # 找到区域中的最大值（峰值）及其索引
        current_peak_index, current_peak = max(region, key=lambda x: x[1])
        # 找到区域中的最小值（谷值）及其索引
        current_valley_index, current_valley = min(region, key=lambda x: x[1])

        # Add the current peak coordinates to the peak lists
        r_indices.append(current_peak_index)
        r_values.append(current_peak)

        if current_valley_index < current_peak_index:
            q_indices.append(current_valley_index)
            q_values.append(current_valley)
        else:
            s_indices.append(current_valley_index)
            s_values.append(current_valley)

        q_amplitudes = [round(y - onset, 6) for y, onset in zip(q_values, onset_values)] # 取相对值并保留小数点后6位
        r_amplitudes = [round(y - onset, 6) for y, onset in zip(r_values, onset_values)] # 取相对值并保留小数点后6位
        s_amplitudes = [round(y - onset, 6) for y, onset in zip(s_values, onset_values)] # 取相对值并保留小数点后6位
        # 找到最长的那个波形列表，然后把所有列表的长度都填充到最长的长度
        max_length = max(len(q_amplitudes), len(r_amplitudes), len(s_amplitudes))
        q_amplitudes.extend([0] * (max_length - len(q_amplitudes)))
        r_amplitudes.extend([0] * (max_length - len(r_amplitudes)))
        s_amplitudes.extend([0] * (max_length - len(s_amplitudes)))
    
    return q_indices, q_amplitudes, r_indices, r_amplitudes, s_indices, s_amplitudes

# endregion

# region 返回正向波和反向波的面积及长度
# 返回正向波和反向波的面积及长度
def seperate_significant_waves(data_lead, on_set_y):
    """
    data_lead (list): 一个长度为5000的list，其中在x波区域的数据点的电压值，而不在x波区域的数据点为-10。
    on_set_y (list): 一个长度为n的list，其中包含n个波的起始点的电压值。
    在每一个心搏区域内，从起点开始，程序会找到data_lead的数据点穿过on_set_y的点。
    如果这些穿越点定义的波形面积大于或等于160 µV-ms，该波形就被定义为显著波形。
    如果面积小于这个值，程序会将该波形视为不显著波形，并不会将其标记为一个单独的波形。
    不符合最低160 µV-ms波形标准的复合波部分将与相邻的显著波形合并。
    """
    
    # 初始化存储各区域计算结果的列表
    positive_area_list = []     # 存储所有正向波的面积
    negative_area_list = []     # 存储所有反向波的面积
    combine_area_list = []      # 存储所有波形的综合面积
    positive_length_list = []   # 存储所有正向波的长度
    negative_length_list = []   # 存储所有反向波的长度
    threshold_area = 0.08  # 定义显著波形的最小面积为160µV-ms，一个采样点是2ms，所以160µV-ms对应的阈值就是160/2/1000=0.08，单位是mv*采样点

    wave_regions = split_wave_regions(data_lead) 
    # 把信号按心搏分成n个区域,每个区域是储存着一串坐标点，每个坐标点是一个元组，第一个元素是索引，第二个元素是电压值
    
    # 初始化当前波形的累积量
    current_area = 0  # 当前波形的面积
    current_length = 0  # 当前波形的长度
    positive = None  # 记录波形正负向，一旦开始负值就之后true和false
    

    # Iterate over the data_lead_x array
    for i, region in enumerate(wave_regions): # 遍历N个心搏区域
        if i < len(on_set_y): # 没搞懂为什么i会遍历到比len(wave_regions)还多的地方
            baseline_reference = on_set_y[i] # 从on_set_y中取出当前心搏的起始点的电压值
        for _, value in region: # 遍历每个区域的每个点, _是绝对位置的索引，value是电压值
            # 每一个点进来，先判断是否有正负状态变化，当正负变化时，说明当前点穿过基线参考点，记录前一个波形并重置正负标志和面积长度
            if (positive and value < baseline_reference) or (not positive and value > baseline_reference) :
                if positive and current_area >= threshold_area: # 如果状态改变前是正向波,且这个波的面积大于160
                    positive_area_list.append(current_area) # 把当前波形的面积和长度添加到正向波列表中
                    positive_length_list.append(current_length)
                elif not positive and current_area >= threshold_area: # 如果状态改变前是反向波,且这个波的面积大于160
                    negative_area_list.append(current_area) # 把当前波形的面积和长度添加到反向波列表中
                    negative_length_list.append(current_length)
                else: # 如果这个波的面积小于160，就不要重置，直接进入下一个循环，让这个波形和下一个波形合并
                    pass
                combine_area_list.append(current_area) # 把当前波形的面积添加到综合面积列表中
                current_area = 0
                current_length = 0
            # 做完变号的判断之后，再更新当前波形的面积和长度
            positive = value > baseline_reference # 更新正负波状态
            current_area = current_area + abs(value - baseline_reference) # 累积面积，累加值：绝对电压-基线电压 
            current_length += 1

        if positive:
            positive_area_list.append(current_area)
            positive_length_list.append(current_length)
        else:
            negative_area_list.append(current_area)
            negative_length_list.append(current_length)
        combine_area_list.append(current_area)
        current_area = 0 # 重置当前波形的面积和长度
        current_length = 0
        positive = None # 重置正负波

    # 所有面积先乘以2，单位就是mv*ms
    positive_area_list = [area * 2 for area in positive_area_list]
    negative_area_list = [area * 2 for area in negative_area_list]
    combine_area_list = [area * 2 for area in combine_area_list]
    # 把所有列表转成ms
    positive_length_list = [length * 2 for length in positive_length_list]
    negative_length_list = [length * 2 for length in negative_length_list]

    # 把所有list里的值呈现两极分化的情况的列表做一些筛选
    positive_area_list = filter_list(positive_area_list)
    negative_area_list = filter_list(negative_area_list)
    combine_area_list = filter_list(combine_area_list)
    positive_length_list = filter_list(positive_length_list)
    negative_length_list = filter_list(negative_length_list)

    # 返回五个列表
    return combine_area_list, positive_length_list, negative_length_list

def qrs_significant_waves(data_lead, on_set_y, r_peak_indices):
    """
    data_lead (list): 一个长度为5000的list，其中在x波区域的数据点的电压值，而不在x波区域的数据点为-10。
    on_set_y (list): 一个长度为n的list，其中包含n个波的起始点的电压值。
    在每一个心搏区域内，从起点开始，程序会找到data_lead的数据点穿过on_set_y的点。
    如果这些穿越点定义的波形面积大于或等于160 µV-ms，该波形就被定义为显著波形。
    如果面积小于这个值，程序会将该波形视为不显著波形，并不会将其标记为一个单独的波形。
    不符合最低160 µV-ms波形标准的复合波部分将与相邻的显著波形合并。
    """
    
    # 初始化存储各区域计算结果的列表
    q_area_list, r_area_list, rr_area_list, s_area_list, ss_area_list, combine_area_list = [], [], [], [], [], []
    q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list = [], [], [], [], []
    threshold_area = 0.08  # 定义显著波形的最小面积为160µV.ms，一个采样点是2ms，所以160µV-ms对应的阈值就是160/2/1000=0.08，单位是mv*采样点

    wave_regions = split_wave_regions(data_lead) 
    # 把信号按心搏分成n个区域,每个区域是储存着一串坐标点，每个坐标点是一个元组，第一个元素是索引，第二个元素是电压值
    
    # 初始化当前波形的累积量
    current_area = 0  # 当前波形的面积
    current_length = 0  # 当前波形的长度
    positive = None  # 记录波形正负向，一旦开始负值就之后true和false
    
    # Iterate over the data_lead_x array
    for i, region in enumerate(wave_regions): # 遍历N个心搏区域
        if i < len(on_set_y): # 没搞懂为什么i会遍历到比len(wave_regions)还多的地方
            baseline_reference = on_set_y[i] # 从on_set_y中取出当前心搏的起始点的电压值
            r_peak_index = r_peak_indices[i] # 从r_peak_indices中取出当前心搏的R波的索引
            r_added = False
            s_added = False
        for j, value in region: # 遍历每个区域的每个点, j是绝对位置的索引，value是电压值
            # 每一个点进来，先判断是否有正负状态变化，当正负变化时，说明当前点穿过基线参考点，记录前一个波形并重置正负标志和面积长度
            if ((positive and value < baseline_reference) or (not positive and value > baseline_reference)) \
                and current_area >= threshold_area : # 且这个波的面积大于160
                if positive: # 如果状态改变前是正向波, 说明是向下穿越
                    # 这一个for里r[ ]是否被append过
                    if not r_added:
                        r_area_list.append(current_area) # 把当前波形的面积和长度添加到正向波列表中
                        r_length_list.append(current_length)
                        r_added = True
                    else:
                        rr_area_list.append(current_area)
                        rr_length_list.append(current_length)
                else: # 如果状态改变前是反向波,说明是向上穿越
                    if j < r_peak_index:
                        q_area_list.append(current_area) # 把当前波形的面积和长度添加到反向波列表中
                        q_length_list.append(current_length)
                    else:
                        if not s_added:
                            s_area_list.append(current_area)
                            s_length_list.append(current_length)
                            s_added = True
                        else:
                            ss_area_list.append(current_area)
                            ss_length_list.append(current_length)

                combine_area_list.append(current_area) # 把当前波形的面积添加到综合面积列表中
                current_area = 0
                current_length = 0
            # 做完变号的判断之后，再更新当前波形的面积和长度
            positive = value > baseline_reference # 更新正负波状态
            current_area = current_area + abs(value - baseline_reference) # 累积面积，累加值：绝对电压-基线电压 
            current_length += 1

        if current_area >= threshold_area: # 到了某region的最后一个点
            if positive:
                if not r_added:
                    r_area_list.append(current_area)
                    r_length_list.append(current_length)
                else:
                    rr_area_list.append(current_area)
                    rr_length_list.append(current_length)
            else: # 如果最后一个点是反向波，且这个波的面积大于160
                if j < r_peak_index:
                    q_area_list.append(current_area)
                    q_length_list.append(current_length)
                else:
                    if not s_added:
                        s_area_list.append(current_area)
                        s_length_list.append(current_length)
                    else:
                        ss_area_list.append(current_area)
                        ss_length_list.append(current_length)
        
        combine_area_list.append(current_area)
        current_area = 0 # 重置当前波形的面积和长度
        current_length = 0
        positive = None # 重置正负波

    # 所有面积先乘以2，单位就是mv*ms
    q_area_list, r_area_list, rr_area_list, s_area_list, ss_area_list, combine_area_list = \
        [[area * 2 for area in lst] for lst in [q_area_list, r_area_list, rr_area_list, s_area_list, ss_area_list, combine_area_list]]
    # 把所有列表转成ms
    q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list = \
        [[length * 2 for length in lst] for lst in [q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list]]
    
    # 找到最长的那个波形，然后把所有列表的长度都填充到最长的长度
    max_length = max([len(lst) for lst in [q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list]])
    q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list = \
        [lst + [0] * (max_length - len(lst)) for lst in [q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list]]
    # 返回五个列表
    return combine_area_list, q_length_list, r_length_list, rr_length_list, s_length_list, ss_length_list

def split_wave_regions(data_lead_x, noise_value=-10):
    regions = []
    current_region = []

    for i, value in enumerate(data_lead_x):
        if value != noise_value:
            # If the current value is part of the x-wave, add it to the current region
            current_region.append((i, value))
        else:
            # If we encounter a noise value and the current region is not empty,
            # save the current region and reset it
            if current_region:
                regions.append(current_region)
                current_region = []
    
    # Append the last region if it exists
    if current_region:
        regions.append(current_region)

    return regions
 #endregion  

# region 过滤列表中的两极分化的小值
def filter_list(list_unfiltered):
    # 判断是否存在两极分化
    if not list_unfiltered:
        return list_unfiltered
    
    max_value = max(list_unfiltered)
    
    # 找到非零的最小值
    non_zero_values = [num for num in list_unfiltered if num != 0]
    if non_zero_values:
        min_value = min(non_zero_values)
    else:
        min_value = max_value  # 如果列表中全是0，设定 min_value 为 max_value
    
    avg_value = sum(list_unfiltered) / len(list_unfiltered)
    
    # 设定两极分化的标准，例如：最大值和最小值的差异显著
    max_min_ratio_threshold = 2  # 比例阈值
    
    # 检查极值是否超出阈值，且避免除零错误
    if min_value != 0 and (max_value / min_value) > max_min_ratio_threshold:
        # 保留大于平均值的元素，或等于 0 的元素
        list_top = [num for num in non_zero_values if num > avg_value or num == 0]
        return list_top
    else:
        return list_unfiltered
# endregion    

# region 返回每搏RR间隙，平均RR间隔时间和心率
def get_heart_rate(peaks):
    
    if len(peaks) < 2:
        return [], 0, 0

    interval_time_list = []
    heart_rate_list = []

    for i in range(len(peaks) - 1):
        interval_point = peaks[i + 1] - peaks[i]  # 计算相邻峰值点间隔时间
        interval_time = interval_point * 2  # 500Hz的采样点要转换为毫秒就是乘以2
        interval_time_list.append(interval_time)

        heart_rate = int(60000 / interval_time)
        heart_rate_list.append(heart_rate)

    average_interval_time = round(sum(interval_time_list) / len(interval_time_list), 0)
    average_heart_rate = int(sum(heart_rate_list) / len(heart_rate_list))

    return interval_time_list, average_interval_time, average_heart_rate
# endregion

# region 返回ST段和QT段索引，平均ST段长度和QT段长度
def findout_st_qt(label, average_rr_interval):
    st_indices = []  # ST段索引
    qt_indices = []  # QT段索引
    stj_indices = []  # STJ 索引
    stm_indices = []  # STM 索引
    ste_indices = []  # STE 索引
    i = 0
    label = list(map(int, label))  # 将label列表中的元素转换为整数列表
    average_rr_interval = average_rr_interval / 2  # 将平均 RR 间隔转换为点数

    rr_1_16 = round(average_rr_interval / 16)
    rr_1_8 = round(average_rr_interval / 8)

    while i < len(label):
        try: 
            qrs_begin_index = label[i:].index(2) + i  # 往后遍历，直到找到QRS波起始点的索引
            qrs_end_index = label[qrs_begin_index:].index(0) + qrs_begin_index  # 往后遍历，找到QRS波结束点的索引
        except ValueError:
            break
        try:
            t_begin_index = label[qrs_end_index:].index(3) + qrs_end_index  # 往后遍历，找到T波的起始位置
        except ValueError:
            break

        # 计算STJ、STM、STE
        # stj_indices 定义为 QRS 终点处（通常称为 "J 点"）的索引值。STM 是 QRS 终点加上平均 RR 间隔的 1/16 处的索引值。STE 是 QRS 终点加上平均 RR 间隔的 1/8 处的索引值。
        stj_index = qrs_end_index
        stm_index = qrs_end_index + rr_1_16
        ste_index = qrs_end_index + rr_1_8

        # 找到ST段和QT段
        st_segment = [j for j in range(qrs_end_index, t_begin_index)]
        qt_segment = [j for j in range(qrs_begin_index, t_begin_index)]
        i = t_begin_index

        if len(st_segment) > 140:
            st_segment = st_segment[:40]  # ST段强制限制在40个点
            qt_segment = qt_segment[:50]
            i -= 60

        st_indices.extend(st_segment)
        qt_indices.extend(qt_segment)
        stj_indices.append(stj_index)
        stm_indices.append(stm_index)
        ste_indices.append(ste_index)

    return st_indices, qt_indices, stj_indices, stm_indices, ste_indices
# endregion

# region 返回PR间期的索引，PR间期长度和平均PR间期长度
def findout_pr_interval(label):
    pr_indices = []
    pr_len = []
    i = 0
    label = list(map(int, label))  # 将label列表中的元素转换为list 后面的index才能用
    while i < len(label):
        try: #寻找下一个p波的起始点的索引
            p_index = label[i:].index(1)+i
        except ValueError:
            break
        try: #寻找下一个qrs波的起始点的索引
            qrs_index = label[p_index:].index(2)+p_index
        except ValueError:
            break
        # 找到qrs波的索引后，从P波起始点开始一直到这个qrs波的起始点之间，所有元素的索引取到，这就是pr段
        interval = [j for j in range(p_index,qrs_index)]
        i = qrs_index
        if len(interval) > 300:
            interval = interval[0:300]#pr段强制限制在300个点
            i -= 60

        pr_len.append(len(interval))
        pr_indices.extend(interval)

    pr_len = [length * 2 for length in pr_len]  # PR间期长度转换为毫秒

    average_pr_len = round(sum(pr_len) / len(pr_len), 0) if pr_len else 0
    
    return pr_indices,pr_len,average_pr_len
# endregion

# region 返回测量点的电压值
def measure_point(stj_indices, stm_indices, ste_indices, data, on_set_y):
    stj_values_abs = [data[stj_index] for stj_index in stj_indices if 0 <= stj_index < len(data)]# 这里的ABS指的是绝对电压值，不是绝对值的意思
    stm_values_abs = [data[stm_index] for stm_index in stm_indices if 0 <= stm_index < len(data)]
    ste_values_abs = [data[ste_index] for ste_index in ste_indices if 0 <= ste_index < len(data)]

    stj_values = [y - onset_y for y, onset_y in zip(stj_values_abs, on_set_y)]# 这里取到的是相对电压值
    stm_values = [y - onset_y for y, onset_y in zip(stm_values_abs, on_set_y)]
    ste_values = [y - onset_y for y, onset_y in zip(ste_values_abs, on_set_y)]

    return stj_values, stm_values, ste_values

def elevation_assess(list1, list2):
    # 评估两个列表的差异，返回评估列表、差异列表和平均差异值
    threshold = 0.01
    assess_list = []
    level_list = []

    for i in range(min(len(list1), len(list2))):
        diff = list1[i] - list2[i]
        if abs(diff) < threshold:
            assess_list.append(0)
        else:
            assess_list.append(1 if diff > 0 else -1)
        level_list.append(diff)

    if len(list1) != len(list2):
        assess_list.append(None)

    average_level = round(sum(level_list) / len(level_list), 3) if level_list else 0

    return assess_list, level_list, average_level

# region 电轴计算
def calculate_electric_axis(wave_details, wave_type):
    if wave_type == 'QRS':
        # 计算QRS 波群的电轴
        average_amplitude_of_I = np.nan_to_num(np.median(wave_details['I']['r_amplitudes']))
        average_amplitude_of_III = np.nan_to_num(np.median(wave_details['III']['r_amplitudes']))
        average_valley_of_I = np.nan_to_num(np.median(wave_details['I']['s_amplitudes']))
        average_valley_of_III = np.nan_to_num(np.median(wave_details['III']['s_amplitudes']))
        # QRS 波负向波是取绝对值的，所以这里要用减
        LI = average_amplitude_of_I - average_valley_of_I
        LIII = average_amplitude_of_III - average_valley_of_III
    else:
        # 计算 T P 波 的电轴
        average_amplitude_of_I = np.nan_to_num(np.median(wave_details['I']['amplitudes']))
        average_amplitude_of_III = np.nan_to_num(np.median(wave_details['III']['amplitudes']))
        average_valley_of_I = np.nan_to_num(np.median(wave_details['I']['valleys']))
        average_valley_of_III = np.nan_to_num(np.median(wave_details['III']['valleys']))
        # T P 波负向波是没有取绝对值的，所以这里要用加
        LI = average_amplitude_of_I + average_valley_of_I
        LIII = average_amplitude_of_III + average_valley_of_III

    # 根据经典振幅法作图进行心电轴测量
    # 其中LI为标准I导联的QRS波群的各个波深度或高度算术和，LIII为标准III导联的QRS波群的各个波深度或高度算术和
    degree = np.arctan((2 * LIII + LI) / (np.sqrt(3)* LI))* 180 / np.pi
    if LI<0 and LIII>0: 
        deviation = 180 + degree
    elif LI<0 and LIII<0:
        deviation = -(180 - degree)
    else:
        deviation = degree
    return deviation
# endregion 电轴计算
# endregion

# region 参数delta波的测量点
def find_delta_start(data_lead, onset_values): # data_lead是一个导联的5000个信号值，onset_values是这个导联的所有qrs波的起始点的电压值

    wave_regions = split_wave_regions(data_lead)# 把信号按心搏分成n个区域
    # 遍历每个区域，把每个区域中比onset_values大0.02点的索引取出来
    delta_start_indices = []
    delta_start_values = []
    delta_start_16ms_indices = []
    delta_start_16ms_values = []
    delta_start_28ms_indices = []
    delta_start_28ms_values = []

    for i, region in enumerate(wave_regions):
        if i < len(onset_values):
            onset = onset_values[i]
            for j, value in region:
                if value > onset + 0.02:
                    delta_start_indices.append(j)
                    break

    delta_start_values = [round(y - onset, 6) for y, onset in zip(data_lead[delta_start_indices], onset_values)]
    delta_start_16ms_indices = [index + 8 for index in delta_start_indices]
    delta_start_16ms_indices = [i for i in delta_start_16ms_indices if i < len(data_lead)]# 限制长度，不然容易越界
    delta_start_16ms_values = [round(y - onset, 6) for y, onset in zip(data_lead[delta_start_16ms_indices], onset_values)]
    delta_start_28ms_indices = [index + 14 for index in delta_start_indices]
    delta_start_28ms_indices = [i for i in delta_start_28ms_indices if i < len(data_lead)]
    delta_start_28ms_values = [round(y - onset, 6) for y, onset in zip(data_lead[delta_start_28ms_indices], onset_values)]

    points = [delta_start_values, delta_start_16ms_values, delta_start_28ms_values]
    return points
# endregion 参数delta波的测量点

# region QT间期计算
def findout_qt_interval(label, rr_intervals):
    qt_indices = []
    qt_len = []
    i = 0
    label = list(map(int, label))  # 确保 label 是整数列表
    while i < len(label):
        try:  # 寻找 Q 波起点的索引
            q_index = label[i:].index(2) + i
        except ValueError:
            break
        try:  # 寻找 T 波结束点的索引
            t_index = label[q_index:].index(3) + q_index
        except ValueError:
            break
        interval = [j for j in range(q_index, t_index)]  # 获取 QT 间期索引
        i = t_index  # 更新索引
        if len(interval) > 500:  # QT 间期通常不会超过 500ms
            interval = interval[:500]
            i -= 100
        qt_len.append(len(interval))
        qt_indices.extend(interval)
    qt_len = [length * 2 for length in qt_len]  # QT 间期转换为毫秒
    average_qt_len = round(sum(qt_len) / len(qt_len), 0) if qt_len else 0
    
    # 计算 QTc（Bazett 公式：QTc = QT / sqrt(RR)）
    if rr_intervals:
        rr_avg = sum(rr_intervals) / len(rr_intervals)  # 计算 RR 平均间期（ms）
        qt_c = round(average_qt_len / math.sqrt(rr_avg / 1000), 2) if rr_avg else 0
    else:
        qt_c = 0
    
    return qt_indices, qt_len, average_qt_len, qt_c
# endregion QT间期计算


# 参数提取总函数
def extract_parameter(data_12, label_12): # in（5000，12）（5000，12） ｜  out 字典

    # region -------------初始化wave_details字典
    leads = ['I', 'II', 'III', 'aVL', 'aVR', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    P_wave_details = {
        lead: {
            'amplitudes': None,
            'valleys': None,
            'durations': None,
            'areas': None,
            'wave_indices': None,
            'onset_amplitudes': None,
            'onset_indices': None
        } for lead in leads
    }
    QRS_wave_details = {
        lead: {
            'q_amplitudes': None,
            'r_amplitudes': None,
            's_amplitudes': None,
            'durations': None,
            'q_durations': None,
            'r_durations': None,
            'rr_durations': None,
            's_durations': None,
            'ss_durations': None,
            'areas': None,
            'q_valley_indices': None,
            'wave_indices': None,
            's_valley_indices': None,
            'onset_amplitudes': None,
            'delta_points': None
        } for lead in leads
    }
    T_wave_details = {
        lead: {
            'amplitudes': None, 
            'valleys': None, 
            'durations': None, 
            'areas': None, 
            'wave_indices': None, 
            'onset_amplitudes': None
        } for lead in leads
    }
    other_details = {
        lead:{
            'rr_intervals': None, 
            'heart_rate': None, 
            'atrial_rate': None, 
            'pr_intervals': None, 
            'stj_values': None, 
            'stm_values': None, 
            'ste_values': None
        } for lead in leads
    }
    #endregion -------------初始化wave_details字典

    # 遍历label的12和导联，遍历data的12导联
    for i in range(12):
        # region ------------------------------提取每个导联的特征参数
        data_1 = data_12[:, i]
        label_1 = label_12[:, i]
        lead = leads[i]

        # 调用extract_wave函数，选择分析导联
        data_lead_p, data_lead_qrs, data_lead_t, label_p, label_qrs, label_t = extract_wave(data_1,label_1)
        # 前三个是用于储存电压值(5000,), data_lead_p是5000个点，在是p波的地方是电压值，在不是p波的地方就是-10
        # 后三个用于储存时间戳 list:700~1400 

        # 获取每个波段起始点坐标和持续时间，这个高度是绝对值（电压值）
        p_onset_indices, p_onset_values, p_wave_durations = find_consecutive_lengths(label_p, data_lead_p)
        qrs_onset_indices, qrs_onset_values, qrs_wave_durations = find_consecutive_lengths(label_qrs, data_lead_qrs)
        _, t_onset_y, t_wave_durations = find_consecutive_lengths(label_t, data_lead_t) # t_onset_x 是没有用的

        # 获取每个波的峰谷值,这个值是相对于当前波onset的相对高度值
        p_peak_indices, p_peak_amplitudes, p_valley_indices, p_valley_amplitudes = found_peaks(data_lead_p, p_onset_values)
        q_valley_indices, q_valley_amplitudes, r_peak_indices, r_peak_amplitudes, s_valley_indices, s_valley_amplitudes = \
            qrs_found_peaks(data_lead_qrs, qrs_onset_values)
        t_peak_indices, t_peak_amplitudes, t_valley_indices, t_valley_amplitudes = found_peaks(data_lead_t, qrs_onset_values)
        # s_valley_y is a list of negative values, need to be converted to positive values:s_amplitudes_y
        q_amplitudes = [-value for value in q_valley_amplitudes]
        r_amplitudes = [value for value in r_peak_amplitudes]
        s_amplitudes = [-value for value in s_valley_amplitudes]

        # 在基础波形里切分显著波型
        p_combine_areas,  _, _ = seperate_significant_waves(data_lead_p, p_onset_values)
        qrs_combine_areas, q_durations, r_durations, rr_durations, s_durations, ss_durations = \
            qrs_significant_waves(data_lead_qrs, qrs_onset_values, r_peak_indices)
        t_combine_areas,  _, _ = seperate_significant_waves(data_lead_t, qrs_onset_values)

        # 求心率
        rr_intervals, average_rr_interval, heart_rate= get_heart_rate(r_peak_indices)
        pp_intervals, average_pp_interval, atrial_rate= get_heart_rate(p_peak_indices)
        # 找到st段和qt段，并计算平均长度
        st_indices, qt_indices, stj_indices, stm_indices, ste_indices = findout_st_qt(label_1, average_rr_interval)
        # 找到pr段，并计算平均长度
        pr_indices, pr_intervals, average_pr_len= findout_pr_interval(label_1)

        # 根据st段的长度，确定st段测量的参考点，得到测量值，并计算平均值
        stj_values, stm_values, ste_values = measure_point(stj_indices, stm_indices, ste_indices, data_1, qrs_onset_values)

        # 用上面得到的测量值根等电位线的瞬时值做对比
        # assess_list, level_list, average_st_level = elevation_assess(qrs_on_y, measurement_value)

        # delta波的测量起点
        delta_points = find_delta_start(data_lead_qrs, qrs_onset_values)

        # QT间期计算
        qt_indices, qt_len, average_qt_len, qt_c = findout_qt_interval(label_1, rr_intervals)
        
        # endregion ------------------------------提取每个导联的特征参数


        # region ------------------------把每个导联的特征参数存到字典里
        P_wave_details[lead]['amplitudes'] = p_peak_amplitudes # 每个p波的峰值 这是一个list
        P_wave_details[lead]['valleys'] = p_valley_amplitudes
        P_wave_details[lead]['durations'] = p_wave_durations # 每个p波的持续时间
        P_wave_details[lead]['areas'] = p_combine_areas # 面积
        P_wave_details[lead]['wave_indices'] = p_peak_indices
        P_wave_details[lead]['onset_values'] = p_onset_values
        P_wave_details[lead]['onset_indices'] = p_onset_indices

        QRS_wave_details[lead]['q_amplitudes'] = q_amplitudes
        QRS_wave_details[lead]['r_amplitudes'] = r_amplitudes
        QRS_wave_details[lead]['s_amplitudes'] = s_amplitudes
        QRS_wave_details[lead]['durations'] = qrs_wave_durations
        QRS_wave_details[lead]['q_durations'] = q_durations
        QRS_wave_details[lead]['r_durations'] = r_durations
        QRS_wave_details[lead]['rr_durations'] = rr_durations
        QRS_wave_details[lead]['s_durations'] = s_durations
        QRS_wave_details[lead]['ss_durations'] = ss_durations
        QRS_wave_details[lead]['areas'] = qrs_combine_areas
        QRS_wave_details[lead]['q_valley_indices'] = q_valley_indices
        QRS_wave_details[lead]['wave_indices'] = r_peak_indices
        QRS_wave_details[lead]['s_valley_indices'] = s_valley_indices
        QRS_wave_details[lead]['onset_values'] = qrs_onset_values
        QRS_wave_details[lead]['onset_indices'] = qrs_onset_indices
        QRS_wave_details[lead]['delta_points'] = delta_points

        T_wave_details[lead]['amplitudes'] = t_peak_amplitudes
        T_wave_details[lead]['valleys'] = t_valley_amplitudes
        T_wave_details[lead]['durations'] = t_wave_durations
        T_wave_details[lead]['areas'] = t_combine_areas
        T_wave_details[lead]['wave_indices'] = t_peak_indices
        T_wave_details[lead]['onset_values'] = t_onset_y

        other_details[lead]['rr_intervals'] = rr_intervals # 每搏RR间隙 这是一个list
        other_details[lead]['pr_intervals'] = pr_intervals # PR间期长度 这是一个list
        other_details[lead]['heart_rate'] = heart_rate # 心率
        other_details[lead]['atrial_rate'] = atrial_rate # 房率
        other_details[lead]['stj_values'] = stj_values
        other_details[lead]['stm_values'] = stm_values
        other_details[lead]['ste_values'] = ste_values
        other_details[lead]['stj_indices'] = stj_indices
        other_details[lead]['stm_indices'] = stm_indices
        other_details[lead]['ste_indices'] = ste_indices
        # endregion ------------------------把每个导联的特征参数存到字典里

        
    axis = calculate_electric_axis(QRS_wave_details, 'QRS')
    t_axis = calculate_electric_axis(T_wave_details, 'T')
    p_axis = calculate_electric_axis(P_wave_details, 'P')
    return {'P_wave_details': P_wave_details, 'QRS_wave_details': QRS_wave_details, 'T_wave_details': T_wave_details, 'other_details': other_details, 'axis': axis, 't_axis': t_axis, 'p_axis': p_axis}


# endregion --------------------------------------------------------参数提取处理代码






