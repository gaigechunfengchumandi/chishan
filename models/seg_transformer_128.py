import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

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
        residual = self.residual_pool(residual)  # Match dimensions residual (2,128,10)
        x_patch = self.patch_embed(x)  # (batch, 128, num_patches) (2,128,10)
        x = x_patch + residual  # Now dimensions match  (2,128,10)
        x = x.permute(2, 0, 1)  #(10,2,128)
        
        features = self.transformer(x) # （2，128，10）
        out_features = self.conv_last(features) # （2，5，10）
        out = self.upsampling(out_features) # (2, 5, 1280)
        
        # Return both output and features
        return out, x_patch

def plot_features(features, save_path=None):
    """
    使用黄绿色主题热力图可视化特征图，颜色亮度表示概率值
    Args:
        features: 特征图张量 (batch_size, channels, length)
        save_path: 图片保存路径，如果为None则直接显示
    """
    features = features.detach().cpu().numpy()
    batch_size, channels, length = features.shape
    
    plt.figure(figsize=(15, 5))
    # 使用黄绿色主题(YlGn)
    img = plt.imshow(features[0], cmap='YlGn', aspect='auto',
                    interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(img, label='Probability')
    plt.title('Feature Maps Heatmap (Yellow-Green Theme)')
    plt.xlabel('Time Steps')
    plt.ylabel('Channels')
    
    # 添加网格线
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    # 设置网格线位置
    plt.xticks(np.arange(-0.5, 10, 1), [])
    # 修改yticks设置，将网格线放在通道之间
    plt.yticks(np.arange(0, channels, 1), [])
    # 显示通道标签（位置在单元格中心）
    plt.yticks(np.arange(0.5, channels+0.5, 1), [f'Ch {i}' for i in range(channels)])
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_patch_features(features, save_path=None):
    """
    可视化Patch Embedding后的特征图
    Args:
        features: 特征图张量 (batch_size, channels, length)
        save_path: 图片保存路径，如果为None则直接显示
    """
    features = features.detach().cpu().numpy()
    batch_size, channels, length = features.shape
    
    plt.figure(figsize=(15, 5))
    # 使用蓝紫色主题(PuRd)
    img = plt.imshow(features[0], cmap='PuRd', aspect='auto',
                    interpolation='nearest', vmin=features.min(), vmax=features.max())
    plt.colorbar(img, label='Feature Value')
    plt.title('Patch Embedding Features Heatmap (Purple-Red Theme)')
    plt.xlabel('Time Steps')
    plt.ylabel('Channels')
    
    # 添加网格线
    plt.grid(True, color='black', linestyle='-', linewidth=0.5)
    plt.xticks(np.arange(-0.5, length, 1), [])
    plt.yticks(np.arange(0, channels, 1), [])
    # 每10个通道显示一个标签
    plt.yticks(np.arange(0.5, channels+0.5, 10), [f'Ch {i}' for i in range(0, channels, 10)])
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test the model
    model = TransformerSegmentation128(input_size=1280, num_classes=5)
    test_input = torch.randn(2, 1, 1280)
    output, features = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    
    # 可视化特征图
    plot_patch_features(features)