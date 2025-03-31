import os
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

# 定义一个函数，将信号数据重采样到指定采样率
def resample_signal(data, original_fs=128, target_fs=250):
    """
    将信号数据重采样到指定采样率
    
    参数:
        data: 原始信号数据
        original_fs: 原始采样率，默认为128Hz
        target_fs: 目标采样率，默认为250Hz
    
    返回:
        重采样后的信号数据
    """
    from scipy import signal
    # 计算重采样后的点数
    new_length = int(len(data) * target_fs / original_fs)
    # 使用signal.resample进行重采样
    resampled_data = signal.resample(data, new_length)
    return resampled_data

# 定义一个函数，把原来的标签转化成0，2，3，4，5个类别，代替掉原来的那标签
def label_transform(ecg_labels):
    label_out = np.zeros(len(ecg_labels))
    for i in range(len(ecg_labels)):
        if ecg_labels[i] == 1 or ecg_labels[i] == 2 or ecg_labels[i] == 3 or \
                ecg_labels[i] == 25 or ecg_labels[i] == 42 or ecg_labels[i] == 43 or ecg_labels[i] == 11 or ecg_labels[i] == 34 or ecg_labels[i] == 35:
            label_out[i] = 0 # N

        # elif ecg_labels[i] == 8 or ecg_labels[i] == 9 or ecg_labels[i] == 7 or \
        #         ecg_labels[i] == 4:
        #     label_out[i] = 1

        elif ecg_labels[i] == 5 or ecg_labels[i] == 10 or ecg_labels[i] == 41 or ecg_labels[i] == 17:
            label_out[i] = 2 # V

        elif ecg_labels[i] == 45 or ecg_labels[i] == 46:
            label_out[i] = 4 # af

        else:
            label_out[i] = 3 # else

    return label_out

# 定义一个函数，把原来的标注数据转化成分段的标注
def segment_label_make(data, old_label, qrs_types):
    # 初始化 label_out
    label_out = np.zeros(len(data))
    # 把原先的标注数据里的R波位置和R-R间期取出来变成列表，并转换为数值类型
    R_Position = [int(x) for x in old_label['RPos'].tolist()[1:]]
    R_R = [int(x) for x in old_label['RR'].tolist()[1:]]

    # 增加一个功能，这里的R_Position是在128hz下标注的，要把它转化成250hz下的位置
    R_Position = [int(x * 250 / 128) for x in R_Position]
    R_R = [int(x * 250 / 128) for x in R_R]

    # 定义存储心拍段的列表
    segments = []

    previous_type = qrs_types[0]
    start_position = R_Position[0]

    # 遍历数据 (从第二行开始)
    for i in range(1, len(R_Position)):
        # 获取当前行数据
        current_r_pos = R_Position[i]
        current_type = qrs_types[i]
        current_r_r = R_R[i]
        
        if current_type != previous_type:  # 检测变号
            # 计算结束位置 = 当前行的R波位置 + 下一行的R-R间期/2
            end_position = current_r_pos + R_R[i+1] / 2
            # 记录当前段
            segments.append((start_position, end_position, previous_type))
            # 更新起始位置和类型
            start_position = current_r_pos - current_r_r / 2 #当前行的R波位置 - 当前行的R-R间期/2
            previous_type = current_type

    # 处理最后一个心拍段
    if len(R_Position) > 0:
        end_position = R_Position[-1] + R_R[-1] / 2
        segments.append((start_position, end_position, previous_type))

    # 将分段信息映射到 label_out
    for start, end, qrs_type in segments:
        start_idx = int(start)
        end_idx = int(end)
        # 确保不越界
        end_idx = min(end_idx, len(label_out)-1)
        label_out[start_idx:end_idx+1] = qrs_type  # +1 确保包含结束位置

    return label_out

# 定义一个函数，将数据和标签划分成1280长度的片段并保存
def save_10s_data_with_label(data, segmentation_label, filename, segments_path):
    """
    将数据和标签划分成2500长度的片段并保存
    
    参数:
        data: 信号数据
        segmentation_label: 分段标注
        filename: 文件名，用于保存
        segments_path: 片段保存路径
    """
    segment_length = 2500
    total_length = len(data)
    
    # 计算可以划分的完整片段数
    num_segments = total_length // segment_length
    
    # 只处理完整的片段
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length
        
        # 提取当前片段的数据和标签
        segment_data = data[start_idx:end_idx]
        segment_label = segmentation_label[start_idx:end_idx].astype(np.int32)
        
        # 确保两个数组长度相同
        if len(segment_data) == segment_length and len(segment_label) == segment_length:
            # 将数据和标签组合成shape为(1280,2)的数组
            combined_data = np.column_stack((segment_data, segment_label))
            
            # 保存为npy文件
            save_path = os.path.join(segments_path, f"{filename.split('.')[0]}_segment_{i}.npy")
            np.save(save_path, combined_data)
            print(f"片段 {i} 已保存到: {save_path}")
    
# 定义一个函数，可视化信号数据和标签并保存图像
def visualize_and_save_signal(segments_path, picture_path, max_samples=1280):
    """
    从保存的片段文件中读取数据并可视化
    
    参数:
        segments_path: 片段保存路径
        picture_path: 图片保存路径
        max_samples: 最大显示的样本数
    """
    # 遍历segments_path中的所有npy文件
    for filename in os.listdir(segments_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(segments_path, filename)
            try:
                # 加载npy文件，其中包含信号数据和标签
                combined_data = np.load(file_path)
                
                # 分离信号数据和标签
                signal_data = combined_data[:, 0]
                segmentation_label = combined_data[:, 1]
                
                plt.figure(figsize=(12, 6))
                
                # 绘制信号数据
                plt.plot(signal_data, 'b-', alpha=0.7, label='Signal')
                
                # 使用颜色区分不同标签
                # 定义标签对应的颜色
                label_colors = {
                    0: 'green',   # N类型
                    2: 'red',     # V类型
                    3: 'orange',  # 其他类型
                    4: 'purple'   # af类型
                }
                
                # 为每个样本点上色
                for i in range(len(signal_data)):
                    label = int(segmentation_label[i])
                    color = label_colors.get(label, 'gray')
                    plt.axvspan(i, i+1, alpha=0.3, color=color)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='green', alpha=0.3, label='N Type (0)'),
                    Patch(facecolor='red', alpha=0.3, label='V Type (2)'),
                    Patch(facecolor='orange', alpha=0.3, label='Other Type (3)'),
                    Patch(facecolor='purple', alpha=0.3, label='AF Type (4)')
                ]
                plt.legend(handles=legend_elements, loc='upper right')
                
                plt.title(f"Signal Data: {filename}")
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.grid(True)
                
                # 保存图像
                save_path = os.path.join(picture_path, f"{filename.replace('.npy', '.png')}")
                plt.savefig(save_path)
                print(f"图像已保存到: {save_path}")
                plt.close()
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


def main(directory, ref_path=None, picture_path=None, segments_path=None):
    """
    加载指定目录下的所有npy文件并进行可视化，同时读取同名标注文件
    
    参数:
        directory: 包含npy文件的目录路径
        ref_path: 标注文件路径
        picture_path: 图片保存路径
        segments_path: 片段保存路径
    """
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.npy'):
            file_path = os.path.join(directory, filename)
            try:
                # 加载npy文件并将所有数据除以200
                data = np.load(file_path) / 200
                # 使用封装的函数进行重采样
                data = resample_signal(data)
                print(f"成功加载文件: {filename}, 形状: {data.shape}")
                
                # 读取同名标注文件
                annotation_file = os.path.join(ref_path, filename.replace('.npy', '.txt').replace('raw', 'ref'))
                
                # 读取数据并手动指定列名
                annotation = pd.read_csv(annotation_file, sep='\t', header=None, names=['RPos', 'QRSType', 'RR'])

                # 提取QRSType列
                qrs_types = annotation['QRSType'].tolist()[1:]

                # 把标注类型都转化成0，2，3，4，5个类别，代替掉原来的那标签
                new_qrs_types = label_transform(qrs_types) 

                # 把标注数据转化成分段的标注
                segment_label = segment_label_make(data, annotation, new_qrs_types) 

                save_10s_data_with_label(data, segment_label, filename, segments_path)

                
            except Exception as e:
                print(f"加载文件 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 定义所有路径
    data_path = '/Users/xingyulu/Public/afafaf/try/data'
    ref_path = '/Users/xingyulu/Public/afafaf/try/ref'
    picture_path = '/Users/xingyulu/Public/afafaf/try/picture'
    segments_path = '/Users/xingyulu/Public/afafaf/try/segments'
    
    # 确保所有目录都存在
    for path in [data_path, ref_path, picture_path, segments_path]:
        os.makedirs(path, exist_ok=True)
    
    # 加载所有npy文件并处理
    main(data_path, ref_path, picture_path, segments_path)

        
    # 从保存的片段文件中可视化数据
    visualize_and_save_signal(segments_path, picture_path)
