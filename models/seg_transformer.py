import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

class ECGDataset(Dataset):
    def __init__(self, data_dir, mode='time', augment=False):
        self.data_dir = data_dir
        self.file_list = list(self.data_dir.glob('*.npy'))
        self.mode = mode
        self.augment = augment  # 新增数据增强开关
        # 如果启用数据增强,生成所有可能的位置交换对
        # 例如,对于长度为10的序列,生成(0,1),(0,2)...(8,9)等45个交换对
        # 如果不启用数据增强,则返回空列表
        self.augment_pairs = [(i,j) for i in range(10) for j in range(i+1,10)] if augment else []

        
    def __len__(self):
        # 如果是增强模式，每个有效样本会产生45个增强样本
        return len(self.file_list) * (1 + len(self.augment_pairs)) if self.augment else len(self.file_list)

    
    def __getitem__(self, idx):
        # 根据是否启用数据增强来计算实际的文件索引和配对索引
        if self.augment:
            # 计算原始文件索引:总样本数除以每个样本产生的增强样本数
            file_idx = idx // (1 + len(self.augment_pairs))
            # 计算增强配对索引:总样本数对每个样本产生的增强样本数取余
            pair_idx = idx % (1 + len(self.augment_pairs)) - 1
        else:
            # 不进行数据增强时,直接使用原始索引
            file_idx = idx
            pair_idx = -1

        # 加载数据文件
        data = np.load(self.file_list[file_idx])
        # 实际访问的文件索引会通过file_idx来访问原始文件
        # 这种设计使得：
        # - 每个原始样本会被访问多次(1次原始+多次增强)
        # - 但DataLoader看到的 idx 仍然是连续递增的
        
        if self.mode == 'time':  
            # 时域模式:提取信号和标签
            signal = data[:, 0].astype(np.float32).squeeze()  # 第一列为信号数据
            label = data[:, 1].astype(np.int64).squeeze()    # 第二列为标签数据

            # 标准化 (假设ECG信号范围在±5mV之间)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)  # Z-score标准化
            signal = np.clip(signal, -5.0, 5.0) / 5.0  # 归一化到[-1, 1]

            # 如果启用数据增强且信号长度为1280(10个心拍)，执行信号和标签的心拍互换
            if pair_idx >= 0 and len(signal) == 1280:
                i, j = self.augment_pairs[pair_idx]  # 获取要交换的心拍索引对
                signal = signal.copy()  # 创建信号的副本
                label = label.copy()    # 创建标签的副本
                # 交换指定心拍的信号和标签(每个心拍128个点)
                signal[i*128:(i+1)*128], signal[j*128:(j+1)*128] = signal[j*128:(j+1)*128].copy(), signal[i*128:(i+1)*128].copy()
                label[i*128:(i+1)*128], label[j*128:(j+1)*128] = label[j*128:(j+1)*128].copy(), label[i*128:(i+1)*128].copy()
            
            # 将numpy数组转换为PyTorch张量并返回
            return torch.tensor(signal), torch.tensor(label)

        elif self.mode == 'fsst':
            # FSST模式 - 使用FSST转换
            # 转换为FSST特征
            fsst_data = time2fsst_for_loader(data)  # shape: (41, sequence_length)
            
            # 分离特征和标签
            signal = fsst_data[:-1, :].astype(np.float32)  # 前40行是特征
            label = fsst_data[-1, :].astype(np.float32)    # 最后一行是标签
            
            # 将numpy数组转换为PyTorch张量并返回
            return torch.tensor(signal), torch.tensor(label)
        else:
            # 如果模式既不是time也不是fsst,抛出异常
            raise ValueError(f"不支持的模式: {self.mode}")



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim) # 注意力子层后的归一化
        self.norm2 = nn.LayerNorm(embed_dim) # 前馈子层后的归一化
        self.ff = nn.Sequential(     # 前馈神经网络（Feed Forward Network）
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), # ff_dim​：前馈神经网络的隐藏层维度
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(x, x, x) # 计算自注意力输出
        x = x + attn_output # 残差连接 + 归一化
        x = self.norm1(x)
        ff_output = self.ff(x) # 前馈网络处理
        x = x + ff_output # 残差连接 + 归一化
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        # 注意：如果 embed_dim 已经是 256，这个线性层可能不是必须的，
        # 但保留它通常无害，除非有特定原因要移除。
        self.output_layer = nn.Linear(embed_dim, 256) 

    def forward(self, x):
        # 输入形状: [num_patches, batch_size, embed_dim] e.g., [25, 2, 256]
        x = x + self.pos_encoder.unsqueeze(1) # 添加位置编码
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
            
        # 输出层调整
        x = self.output_layer(x)  # [num_patches, batch_size, 256]
        x = x.permute(1, 2, 0)    # 转换到 [batch_size, 256, num_patches] e.g., [2, 256, 25]
        
        # 不在此处进行上采样，将上采样移至主模型
        return x

class TransformerSegmentation(nn.Module):
    def __init__(self, input_size=2500, num_classes=5, mode='time'):
        super(TransformerSegmentation, self).__init__()
        self.mode = mode
        input_channels = 1 if mode == 'time' else 40
        patch_size = 100
        num_patches = input_size // patch_size
        embed_dim = 256 # 定义 embed_dim
        
        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=embed_dim, # 使用定义的 embed_dim
            kernel_size=patch_size,
            stride=patch_size
        )
        self.transformer = TransformerEncoder(
            num_patches=num_patches,
            embed_dim=embed_dim, # 使用定义的 embed_dim
            num_heads=8,
            ff_dim=512,
            num_layers=3,
            dropout=0.1
        )
        
        # 移除复杂的解码器，替换为简单的上采样和最终卷积层
        # self.decoder = nn.Sequential(...) # 移除旧的解码器定义

        # 添加最终的分类卷积层
        self.final_conv = nn.Conv1d(embed_dim, num_classes, kernel_size=1)
        self.input_size = input_size # 保存 input_size 以便在 forward 中使用
        
    def forward(self, x):
        # 输入形状: [batch_size, channels, seq_len] e.g., [2, 1, 2500] or [2, 40, 2500]

        # 输入验证 (保持不变)
        if x.dim() != 3:
            raise ValueError(f"输入形状应为[batch, channels, seq_len], 但得到的是{x.shape}")
        if self.mode == 'time' and x.size(1) != 1:
            raise ValueError(f"time模式输入通道数应为1, 但得到的是{x.shape}")
        if self.mode == 'fsst' and x.size(1) != 40:
            raise ValueError(f"fsst模式输入通道数应为40, 但得到的是{x.shape}")
            
        # 使用自定义的patch embedding
        x = self.patch_embed(x)
        x = x.permute(2, 0, 1)
        
        # Transformer处理
        features = self.transformer(x)  # [batch_size, embed_dim, num_patches] e.g., [2, 256, 25]
        
        # 上采样回原始序列长度
        # 使用 'linear' 或 'nearest' 插值模式，取决于具体需求
        # align_corners=False 通常是推荐的设置
        features_interpolated = nn.functional.interpolate(
            features, 
            size=self.input_size, # 上采样到原始输入长度
            mode='linear', 
            align_corners=False
        ) # [batch_size, embed_dim, seq_len] e.g., [2, 256, 2500]
        
        # 应用最终的卷积层进行分类
        out = self.final_conv(features_interpolated) # [batch_size, num_classes, seq_len] e.g., [2, 5, 2500]

        return out

if __name__ == "__main__":
    # 设置较小的 batch size 来减少内存使用
    batch_size = 2
    writer = SummaryWriter('runs/ecg_transformer_segmentation_model')
    
    # 创建模型实例
    print("初始化模型...")
    model = TransformerSegmentation(input_size=2500, num_classes=5, mode='time')
    
    # 生成较小的测试输入
    test_input = torch.randn(batch_size, 1, 2500)

    # 测试模型
    try:
        output = model(test_input)
        writer.add_graph(model, test_input)
        print(f"模型测试成功，输出形状: {output.shape}")
        
        # 查看参数量
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数量: {num_params}")
    except Exception as e:
        print(f"模型错误: {e}")
    finally:
        writer.close()
        print("\nTensorBoard数据已写入。请运行以下命令查看:")
        print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/ecg_transformer_segmentation_model")