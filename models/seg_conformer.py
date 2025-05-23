import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

class ConformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 前馈模块
        self.ff1 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_ff1 = nn.LayerNorm(embed_dim)
        
        # 多头注意力模块
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm_attn = nn.LayerNorm(embed_dim)
        
        # 卷积模块
        self.norm_conv = nn.LayerNorm(embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 4*embed_dim, kernel_size=31, padding='same', groups=embed_dim),
            nn.GLU(dim=1),
            nn.Conv1d(2*embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        # 第二个前馈模块
        self.ff2 = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_ff2 = nn.LayerNorm(embed_dim)
        
        # 最后整体LayerNorm
        self.norm_final = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 输入形状: (seq_len, batch, embed_dim)
        # 第一个前馈模块
        residual = x
        x = self.norm_ff1(x)
        x = self.ff1(x)
        x = residual + 0.5 * x

        # 多头自注意力模块
        residual = x
        x = self.norm_attn(x)
        attn_output, _ = self.attention(x, x, x)
        x = residual + attn_output

        # 卷积模块
        residual = x
        x = self.norm_conv(x)
        x = x.permute(1, 2, 0)  # (batch, embed_dim, seq_len)
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # (seq_len, batch, embed_dim)
        x = residual + x

        # 第二个前馈模块
        residual = x
        x = self.norm_ff2(x)
        x = self.ff2(x)
        x = residual + 0.5 * x

        # 最后整体LayerNorm
        x = self.norm_final(x)
        return x

class ConformerSegmentation(nn.Module):
    def __init__(self, input_size=1280, num_classes=5, patch_size=128):
        super().__init__()
        num_patches = input_size // patch_size
        
        # 残差连接
        self.residual_pool = nn.AdaptiveAvgPool1d(num_patches)
        self.residual_conv = nn.Conv1d(1, 128, kernel_size=1)
        
        # Patch embedding (与seg_transformer_128.py保持一致)
        self.patch_embed = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 128, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Conformer编码器
        self.conformer = nn.ModuleList([
            ConformerBlock(embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1)
            for _ in range(3)
        ])
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(num_patches, 128))
        
        # 输出层
        self.conv_last = nn.Conv1d(128, out_channels=num_classes, kernel_size=1)
        self.upsampling = nn.Upsample(size=1280, mode='linear')
    
    def forward(self, x):
        # 输入形状: (batch, 1, 1280)
        residual = self.residual_conv(x)
        residual = self.residual_pool(residual)  # (batch, 128, 10)
        
        x_patch = self.patch_embed(x)  # (batch, 128, 10)
        x = x_patch + residual
        
        # 添加位置编码
        x = x.permute(2, 0, 1)  # (10, batch, 128)
        x = x + self.pos_encoder.unsqueeze(1)
        
        # 通过Conformer块
        for layer in self.conformer:
            x = layer(x)
        
        # 输出处理
        x = x.permute(1, 2, 0)  # (batch, 128, 10)
        out_features = self.conv_last(x)  # (batch, 5, 10)
        out = self.upsampling(out_features)  # (batch, 5, 1280)
        
        return out

if __name__ == "__main__":
    # 初始化SummaryWriter
    writer = SummaryWriter('runs/seg_conformer_model')
    
    # 测试模型
    model = ConformerSegmentation(input_size=1280, num_classes=5)
    test_input = torch.randn(2, 1, 1280)
    output, features = model(test_input)
    
    # 将模型结构图写入TensorBoard
    writer.add_graph(model, test_input)
    writer.close()
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    
    # 查看TensorBoard的命令提示
    print("\nTensorBoard数据已写入。请运行以下命令查看:")
    print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/seg_conformer_model")