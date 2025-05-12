import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
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
        self.output_layer = nn.Linear(embed_dim, 128)

    def forward(self, x):
        x = x + self.pos_encoder.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = x.permute(1, 2, 0)
        
        return x

class TransformerSegmentation128(nn.Module):
    def __init__(self, input_size=1280, num_classes=5, patch_size=128):
        super().__init__()
        num_patches = input_size // patch_size

        # Add adaptive pooling to match dimensions
        self.residual_pool = nn.AdaptiveAvgPool1d(num_patches)
        self.residual_conv = nn.Conv1d(1, 128, kernel_size=1) #(stride)默认为1

        # Patch embedding
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
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_patches=num_patches,
            embed_dim=128,
            num_heads=4,
            ff_dim=256,
            num_layers=3,
            dropout=0.1
        )
        
        self.conv_last = nn.Conv1d(128, out_channels=num_classes, kernel_size=1)
        self.upsampling = nn.Upsample(size=1280, mode='linear')
  
    def forward(self, x): # x(2,1,1280)
        residual = self.residual_conv(x)# residual (2,128,1280)
        residual = self.residual_pool(residual)  # Match dimensions        residual (2,128,10)
        x = self.patch_embed(x)  # (batch, 128, num_patches) (2,128,10)
        x = x + residual  # Now dimensions match  (2,128,10)
        x = x.permute(2, 0, 1)  #(10,2,128)
        
        features = self.transformer(x) # （2，128，10）

        out = self.conv_last(features) # （2，5，10）

        out = self.upsampling(out) # 使用线性插值将特征图上采样到原始输入大小(1280) (2, 5, 1280)
        return out

if __name__ == "__main__":
    # Test the model
    model = TransformerSegmentation128(input_size=1280, num_classes=5)
    writer = SummaryWriter('runs/patch_128')
    test_input = torch.randn(2, 1, 1280)
    output = model(test_input)
    writer.add_graph(model, test_input)
    writer.close()
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    print("\nTensorBoard数据已写入。请运行以下命令查看:")
    print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/patch_128")