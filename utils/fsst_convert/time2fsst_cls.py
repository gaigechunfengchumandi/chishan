import os
import numpy as np
from utils.fsst_convert.extractFeaturesFSST import extract_features_fsst # 整个文件夹一起转
from utils.fsst_convert.extractFeaturesFSST import extract_features_fsst_1 #单个文件转 
import matplotlib.pyplot as plt
from scipy.signal import stft, windows
import sys
import traceback

'''这个脚本可以把时间域的numpy直接转成单导联的fsst_numpy'''

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TimeToFSSTConverter:
    """
    将时间域数据转换为FSST特征的类
    支持分类(cls)和分割(seg)两种模式
    """
    
    def __init__(self, input_path, output_path, mode='cls', fs=250):
        """
        初始化转换器
        
        参数:
            input_path: 输入文件夹路径
            output_path: 输出文件夹路径
            mode: 处理模式，'cls'(分类)或'seg'(分割)
            fs: 采样率
        """
        self.input_path = input_path
        self.output_path = output_path
        self.mode = mode
        self.fs = fs
        
        # 确保输出文件夹存在
        os.makedirs(self.output_path, exist_ok=True)
        
        # 声明输入输出的列表
        self.input_txt = [None] * 1
        self.feature_fsst = [None] * 1
        
        print(f"初始化完成，当前处理模式: {self.mode}")
    
    def process_file(self, file_path):
        """处理单个文件"""
        try:
            if self.mode == 'cls':
                return self._process_cls_file(file_path)
            elif self.mode == 'seg':
                return self._process_seg_file(file_path)
            else:
                print(f"不支持的处理模式: {self.mode}")
                return False
        except Exception as e:
            print(f"处理文件 {os.path.basename(file_path)} 时出错: {e}")
            traceback.print_exc()
            return False
    
    def _process_cls_file(self, file_path):
        """处理分类模式的文件"""
        # 分类模式 - 只处理信号数据
        data = np.load(file_path).squeeze()  # (2500,)
        self.input_txt[0] = data  # 这个list里面只放一个元素就好
        print(f"传入函数的数据形状: {self.input_txt[0].shape}")
        
        # 调用extract_features_fsst函数得到基于FSST的特征
        self.feature_fsst = extract_features_fsst_1(self.input_txt[0], self.fs)  # list:1  ndarray:(40,2500)
        feature_data = np.array(self.feature_fsst[0])
        
        # 保存结果
        file_name = os.path.basename(file_path)
        save_path = os.path.join(self.output_path, file_name)
        np.save(save_path, feature_data)
        print(f"成功处理并保存: {file_name}")
        return True
    
    def _process_seg_file(self, file_path):
        """处理分割模式的文件"""
        # 分割模式 - 处理信号数据和标签
        data_label = np.load(file_path)  # 把数据和标签一起取出来
        data = data_label[:, 0]  # 用来转成fsst
        self.input_txt[0] = data  # 这个list里面只放一个元素就好(2500,) 
        label = data_label[:, 1]  # (2500,)  把标签取出来
        label_row = label[np.newaxis, :]  # 转成ndarray:(1,2500)以便后面拼接
        
        # 调用extract_features_fsst函数得到基于FSST的特征
        self.feature_fsst = extract_features_fsst_1(self.input_txt[0], self.fs)  # list:1  ndarray:(40,2500)
        feature_data = np.array(self.feature_fsst[0])
        
        # 将特征和标签合并
        combined_data = np.vstack((feature_data, label_row))  # (41,2500)
        
        # 保存结果
        file_name = os.path.basename(file_path)
        save_path = os.path.join(self.output_path, file_name)
        np.save(save_path, combined_data)
        print(f"成功处理并保存: {file_name} (包含标签)")
        return True

        
    
    def process_all_files(self):
        """处理文件夹中的所有文件"""
        # 获取要处理的文件总数
        npy_files = [f for f in os.listdir(self.input_path) if f.endswith(".npy")]
        total_files = len(npy_files)
        processed_files = 0
        success_count = 0
        
        print(f"开始处理，共有 {total_files} 个文件")
        
        for file in npy_files:
            processed_files += 1
            file_path = os.path.join(self.input_path, file)
            print(f"处理文件: {file} ({processed_files}/{total_files})")
            
            if self.process_file(file_path):
                success_count += 1
        
        print(f"处理完成! 共处理 {processed_files} 个文件，成功 {success_count} 个")
        return success_count


def time2fsst_for_loader(data_label, fs=250):
    """
    将时间域数据转换为FSST特征的函数，适用于dataloader调用
    
    参数:
        data_label: numpy数组，shape为(..., 2)，第一列为信号数据，第二列为标签
        fs: 采样率，默认250Hz
    
    返回:
        combined_data: numpy数组，shape为(41, ...)，前40行为FSST特征，最后一行为标签
    """
    # 处理信号数据和标签
    data = data_label[:, 0]  # 用来转成fsst
    label = data_label[:, 1]  # 取出标签
    label_row = label[np.newaxis, :]  # 转成ndarray:(1,...)以便后面拼接
    
    # 准备输入数据
    input_txt = [data]  # 创建包含一个元素的列表
    
    # 调用extract_features_fsst函数得到基于FSST的特征
    feature_fsst = extract_features_fsst_1(input_txt[0], fs)  # list:1  ndarray:(40,...)
    feature_data = np.array(feature_fsst[0])
    
    # 将特征和标签合并
    combined_data = np.vstack((feature_data, label_row))  # (41,...)
    
    return combined_data

# if __name__ == "__main__":
#     # 处理模式: 'cls'(分类)或'seg'(分割)
#     mode = 'seg'  
#     # 要读取的文件夹路径
#     input_path = '/Users/xingyulu/Downloads/fsst_code/try/time'
#     # 要保存的文件夹路径
#     output_path = '/Users/xingyulu/Downloads/fsst_code/try/fsst'
    
#     # 创建转换器实例
#     converter = TimeToFSSTConverter(
#         input_path=input_path,
#         output_path=output_path,
#         mode=mode,
#         fs=250
#     )
    
#     # 处理所有文件
#     converter.process_all_files()


