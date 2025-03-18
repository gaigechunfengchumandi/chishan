"""
室颤分割任务模型

这个模型用于ECG信号的室颤分割任务，输入为ECG信号，输出为每个时间点的室颤概率。
模型基于1D-CNN和BiLSTM的混合架构，能够有效捕捉ECG信号的时序特征和局部特征。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ECGDataset(Dataset):
    """ECG数据集类，用于加载和预处理ECG数据"""
    
    def __init__(self, data_dir, transform=None):
        """
        初始化数据集
        
        Args:
            data_dir: 数据目录路径
            transform: 数据变换函数
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_list = list(self.data_dir.glob('*.npy'))
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 加载数据
        data = np.load(self.file_list[idx])
        
        # 分离信号和标签
        signal = data[:, 0].astype(np.float32).squeeze()  # 确保1D形状
        label = data[:, 1].astype(np.float32).squeeze()
        
        # 应用变换
        if self.transform:
            signal = self.transform(signal)
            
        # 新增标准化 (假设ECG信号范围在±5mV之间)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        signal = np.clip(signal, -5.0, 5.0) / 5.0  # 归一化到[-1, 1]
        
        return torch.tensor(signal), torch.tensor(label)

class VFSegmentationModel(nn.Module):
    """室颤分割模型，基于1D-CNN和BiLSTM的混合架构"""
    
    def __init__(self, input_channels=1, hidden_size=64, num_layers=2, dropout=0.3):
        """
        初始化模型
        
        Args:
            input_channels: 输入通道数，默认为1（单导联ECG）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(VFSegmentationModel, self).__init__()
        
        # 1D卷积层用于提取局部特征
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # 双向LSTM层用于捕捉长期依赖关系
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # 输出层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 新增权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, sequence_length]
            
        Returns:
            输出张量，形状为 [batch_size, sequence_length, 1]
        """
        # 修复维度问题：输入应为 [batch_size, channels, sequence_length]
        x = x.unsqueeze(1)  # 添加通道维度 [batch_size, 1, sequence_length]
        
        # 卷积层处理后的形状应为 [batch_size, 128, sequence_length]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 调整维度以适应LSTM [batch_size, sequence_length, channels]
        x = x.permute(0, 2, 1)
        
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = torch.sigmoid(self.attention(lstm_out))
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        context_vector = attention_weights * lstm_out
        
        # 全连接层
        x = self.dropout(F.relu(self.fc1(context_vector)))
        x = self.fc2(x)
        
        return torch.sigmoid(x)




# 在主程序或需要的地方调用visualize_model函数
if __name__ == "__main__":
    # 初始化模型
    model = VFSegmentationModel(input_channels=1, hidden_size=64, num_layers=2, dropout=0.3)
    
    # 可视化模型
    model.visualize_simple()

