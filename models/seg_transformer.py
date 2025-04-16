import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

class ECGDataset(Dataset):
    def __init__(self, data_dir, mode='time'):
        self.data_dir = data_dir
        self.file_list = list(self.data_dir.glob('*.npy'))
        self.mode = mode
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        if self.mode == 'time':  
            signal = data[:, 0].astype(np.float32).squeeze()
            label = data[:, 1].astype(np.int64).squeeze()

            # 标准化 (假设ECG信号范围在±5mV之间)
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            signal = np.clip(signal, -5.0, 5.0) / 5.0  # 归一化到[-1, 1]

            return torch.tensor(signal), torch.tensor(label)

        elif self.mode == 'fsst':
            # FSST模式 - 使用FSST转换
            # 转换为FSST特征
            fsst_data = time2fsst_for_loader(data)  # shape: (41, sequence_length)
            
            # 分离特征和标签
            signal = fsst_data[:-1, :].astype(np.float32)  # 前40行是特征
            label = fsst_data[-1, :].astype(np.float32)    # 最后一行是标签
            
            return torch.tensor(signal), torch.tensor(label)
        else:
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
        # 创建可学习的位置编码参数
        # num_patches: patch的数量
        # embed_dim: 每个位置编码向量的维度
        # 初始化为全0，在训练过程中会学习合适的位置编码值
        self.pos_encoder = nn.Parameter(torch.zeros(num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 256)

    def forward(self, x):
        # 输入形状: [num_patches, batch_size, embed_dim] [25, 2, 256]
        # 添加位置编码
        x = x + self.pos_encoder.unsqueeze(1) # [25, 2, 256]
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
            
        # 输出层调整 [batch_size, 256, num_patches]
        x = self.output_layer(x)  # [num_patches, batch_size, 256]
        x = x.permute(1, 2, 0)    # 转换到[batch_size, 256, num_patches] [2,256,25]
        
        # 上采样回原始长度 [batch_size, 256, seq_len]
        # x = nn.functional.interpolate(x, size=2500, mode='linear', align_corners=False)
        return x

class TransformerSegmentation(nn.Module):
    def __init__(self, input_size=2500, num_classes=5, mode='time'):
        super(TransformerSegmentation, self).__init__()
        self.mode = mode
        input_channels = 1 if mode == 'time' else 40
        patch_size = 100
        num_patches = input_size//patch_size
        
        self.patch_embed = nn.Conv1d(
            in_channels=input_channels,
            out_channels=256,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.transformer = TransformerEncoder(
            num_patches=num_patches,
            embed_dim=256,
            num_heads=8,
            ff_dim=512,
            num_layers=3,
            dropout=0.1
        )
        
        # 重新调整解码器结构以精确输出2500长度
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=5),  # 25 -> 121
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2),   # 121 -> 250
            nn.ReLU(),
            nn.ConvTranspose1d(64, num_classes, kernel_size=9, stride=10)  # 250 -> 2500
        )
        
    def forward(self, x):
        # 输入形状: [batch_size, channels, seq_len] [2,40,2500]

        # 输入验证
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
        features = self.transformer(x)  # [batch_size, 256, 25]
        
        # 使用调整后的解码器结构
        out = self.decoder(features)  # [2, 5, 2500] [batch_size, num_classes, seq_len]seq_len有时候不是2500
        
        # 确保输出长度正确
        if out.size(2) != 2500:
            out = nn.functional.interpolate(out, size=2500, mode='linear', align_corners=False)
            
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