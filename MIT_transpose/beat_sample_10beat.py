import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy import signal
'''
把欧阳给的动态心电数据（初步被截成10个心拍的，但是还没有重采样的数据）转成我可以读的形式，这个脚本是处理小片数据的
'''
def segment_beats(ecg_signal, RR_Interval):
    """
    根据RR间期将ECG信号分割成心拍片段
    
    参数:
    ecg_signal: ECG信号数据，一维数组
    RR_Interval: RR间期数据，包含10个心拍的间隔
    
    返回:
    beat_segments: 分割后的心拍片段列表，每个元素是一个心拍信号片段
    """
    R_location = 0  # R波位置标记
    start = 0       # 心拍起始位置
    end = 0         # 心拍结束位置
    beat_segments = []  # 存储分割后的心拍
    
    for k in range(10):  # 处理10个心拍


        position_rate = 0.5 #心拍右侧占总窗口长度的比例
        # 第一个心拍的特殊处理
        if k == 0:
            start = 0  # 第一个心拍从信号开始
            R_location = int(0.6 * RR_Interval[k])  # 计算R波位置(假设在60%处)
            end = int(0.6 * RR_Interval[k]) + int(position_rate * RR_Interval[k + 1])  # 结束位置到下一个心拍的position_rate%
        
        # 中间心拍的处理(第2-8个)
        if k != 0 and k != 9:
            start = end  # 当前心拍开始于上一个心拍的结束
            R_location = int(R_location) + int(RR_Interval[k])  # 更新R波位置
            end = int(R_location) + int(position_rate * RR_Interval[k + 1])  # 结束到下一个心拍的position_rate%，可以随便调整
        
        # 最后一个心拍的特殊处理
        if k == 9:
            start = end
            R_location = int(R_location) + int(RR_Interval[k])
            end = int(R_location) + int(position_rate * RR_Interval[k])
            
        # 截取心拍片段并保存
        Beat_seg = ecg_signal[start: end]
        if len(Beat_seg) > 0:  # 只处理非空片段
            Beat_seg = signal.resample(Beat_seg, 128)
            beat_segments.append(Beat_seg)
        else:
            # 如果片段为空，添加一个全零数组
            beat_segments.append(np.zeros(128))

    beat_segments = np.concatenate(beat_segments) # 将10个心拍片段连接成一个(1280,)的一维数组
    
    return beat_segments


# 不再使用这个 （1280，2）的格式存储数据，
# 因为这个格式的标签占用内存太大了 ，
# 只需要把10个心拍的类型放到信号的后面就可以了，新格式（1290，）
# def generate_beat_label(beat_segments, beat_type, save_path=None):
#     """
#     生成心拍标签，根据心拍类别生成标签
    
#     参数:
#     beat_segments: 心拍片段列表
#     beat_type: 心拍类别数组 (1-5)
#     sample_idx: 样本索引
#     save_path: 图片保存路径
#     """
#     label = np.zeros(1280)
#     current_position = 0

#     for beat, b_type in zip(beat_segments, beat_type):
#         b_type = int(b_type) if not np.isnan(b_type) else 0  # 处理NaN值
#         end_position = current_position + len(beat)

#         if end_position > 1280:
#             end_position = 1280

#         label[current_position:end_position] = b_type
#         current_position = end_position

#         if current_position >= 1280:
#             break

    
#     beat_data = np.concatenate(beat_segments)[:1280] # 将心拍片段拼接成一个数组，并限制长度为1280
#     labeled_data = np.column_stack((beat_data, label[:len(beat_data)]))

#     save_path = save_path

    
#     np.save(save_path, labeled_data)
#     print(f"标签已保存至: {save_path}")

def plot_beat_sequence(beat_segments, beat_type, signal=None, prediction=None, title='ECG Beat Signal Sequence', save_path=None):
    """
    绘制心拍信号序列图，根据心拍类别使用背景色块标注
    
    参数:
    beat_segments: 心拍片段列表
    beat_type: 心拍类别数组 (1-5)
    signal: 原始ECG信号（用于整段绘制）
    prediction: 每个样本点的预测标签（可选）
    title: 图表标题
    save_path: 图片保存路径
    """
    try:
        plt.figure(figsize=(20, 4))

        # 如果没有传入完整信号，则拼接心拍片段作为显示用信号
        if signal is None:
            signal = np.concatenate(beat_segments)

        # 使用传入的prediction或从beat_type生成扩展版本
        if prediction is None:
            prediction = []
            for beat, b_type in zip(beat_segments, beat_type):
                prediction.extend([int(b_type) if not np.isnan(b_type) else 0] * len(beat))
            prediction = np.array(prediction[:len(signal)])

        # 绘制ECG信号
        plt.plot(signal, 'b-', alpha=0.7, label='ECG Signal')

        # 定义标签对应的颜色
        label_colors = {
            0: 'lightgreen',   # N类型
            1: 'skyblue',      # S类型
            2: 'lightcoral',   # V类型
            3: 'moccasin',     # X类型
            4: 'plum',         # AF类型
            5: 'purple'        # VF类型
        }

        # 为每个样本点上色（使用透明度较低的背景色）
        for i in range(len(prediction)):
            label = prediction[i]
            color = label_colors.get(label, 'gray')
            plt.axvspan(i, i + 1, alpha=0.3, color=color)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', alpha=0.5, label='Normal (0)'),
            Patch(facecolor='skyblue', alpha=0.5, label='S Type (1)'),
            Patch(facecolor='lightcoral', alpha=0.5, label='V Type (2)'),
            Patch(facecolor='moccasin', alpha=0.5, label='X Type (3)'),
            Patch(facecolor='plum', alpha=0.5, label='AF Type (4)'),
            Patch(facecolor='purple', alpha=0.5, label='VF Type (5)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

        # 在 x 轴每隔 128 个点画一条竖线
        for x in range(0, len(signal), 128):
            plt.axvline(x=x, color='black', linestyle='--', linewidth=1, alpha=0.7)

        plt.title(title)
        plt.xlabel("Sample Points")
        plt.ylabel("Amplitude")
        plt.grid(True)

        # 保存图像
        plt.savefig(save_path)
        print(f"图片已保存至: {save_path}")
        plt.close()

    except Exception as e:
        print(f"绘图时出错: {e}")


if __name__ == '__main__':
    npy_folder = '/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/心拍断掉/train'
    save_folder = '/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/心拍断掉/numpy'
    picture_folder = '/Users/xingyulu/Public/监护心电预警/存储数据转换成可训练数据/心拍断掉/picture_label'

    # 获取路径A下的所有文件名
    files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
    for file in files:
        data = np.load(os.path.join(npy_folder, file), allow_pickle=True)# 形状: 128hz的(1301, 12)  256hz的(2581, 12)
        npy_save_path = os.path.join(save_folder, file)
        image_save_path = os.path.join(picture_folder, os.path.splitext(file)[0] + ".png")

        # 把第二导联作为ECG信号
        ecg_signal_label = data[:, 1] # ECG信号形状: 128hz的 (1301, )    256hz的(2581, )    （总的样本数， 样本序列长度）

        # 到数的11个值代表了这10个心拍的RR间期
        RR_Interval = ecg_signal_label[-11:]  # eg:[80. 80. 80. 80. 79. 80. 79. 79. 80. 48.]

        # 倒数第20到倒数第21，这10个值代表了这10个心拍的类型
        beat_type = ecg_signal_label[-21:-11]  # eg:[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]s

        # 取前10000个点作为ECG信号
        ecg_signal = ecg_signal_label[:10000]

        nan_indices = np.where(np.isnan(ecg_signal))[0]

        # 调用函数获取心拍片段
        beat_segments = segment_beats(ecg_signal, RR_Interval) # 这个操作会把每一个心拍直接转换成128长度，所以你之后不再需要重采样


        # 把心拍片段和心拍类型拼接成一个数组
        labeled_data = np.concatenate((beat_segments, beat_type))
        # 保存为npy文件
        np.save(npy_save_path, labeled_data)

        # 调用函数显示心拍序列
        # generate_beat_label(beat_segments, beat_type, save_path=npy_save_path)

        plot_beat_sequence(beat_segments, beat_type, save_path=image_save_path)




