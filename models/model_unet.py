import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.pool(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        # 裁剪skip_connection以匹配x的尺寸
        diff = skip_connection.size()[2] - x.size()[2]
        if diff > 0:
            skip_connection = skip_connection[:, :, diff//2:-(diff-diff//2)]
        elif diff < 0:
            x = x[:, :, -diff//2:(diff-diff//2)]
        x = torch.cat([x, skip_connection], dim=1)
        x = self.double_conv(x)
        return x

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
        # x shape: [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(1, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(input_size, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.output_layer = nn.Linear(embed_dim, 256)
        
    def forward(self, x):
        # 输入形状: [batch_size, 1, seq_len]
        batch_size, _, seq_len = x.shape

        # 转换为 [seq_len, batch_size, 1]
        x = x.permute(2, 0, 1)

        # 嵌入 [seq_len, batch_size, embed_dim]
        x = self.embedding(x)

        # 添加位置编码
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(1)

        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 输出层调整
        x = self.output_layer(x)            # 形状 [seq_len, batch_size, 256]
        x = x.permute(1, 0, 2)             # 形状 [batch_size, seq_len, 256]
        x = x.permute(0, 2, 1)             # 形状 [batch_size, 256, seq_len]

        return x

class UNet(nn.Module):
    def __init__(self, input_size=2500, use_transformer=True):
        super(UNet, self).__init__()
        self.use_transformer = use_transformer
        
        # Encoder
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        
        # Transformer特征提取器（仅当use_transformer=True时使用）
        if self.use_transformer:
            self.transformer = TransformerEncoder(
                input_size=input_size,
                embed_dim=256,  # 确保与channel_adjust输入维度一致
                num_heads=8,
                ff_dim=512,
                num_layers=3,
                dropout=0.1
            )
            # 修正融合层输入通道为256（卷积特征）+256（Transformer特征）=512，输出为 256
            self.fusion = nn.Conv1d(512, 256, kernel_size=1)
        
        # Bridge
        self.bridge = DoubleConv(256, 512)
        
        # Decoder
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        
        # Output
        self.conv_last = nn.Conv1d(64, 2, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def _fuse_features(self, conv_features, transformer_input):
        """
        融合卷积特征和Transformer特征
        
        Args:
            conv_features: 卷积网络提取的特征
            transformer_input: 原始输入，用于Transformer特征提取
            
        Returns:
            融合后的特征
        """
        # Transformer特征提取
        transformer_features = self.transformer(transformer_input)
        
        # 调整transformer_features的大小以匹配conv_features
        target_size = conv_features.size(2)
        transformer_features = nn.functional.adaptive_avg_pool1d(
            transformer_features, 
            output_size=target_size
        )
        
        # 特征融合
        combined_features = torch.cat([conv_features, transformer_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features
        
    def forward(self, x):
        # 保存原始输入用于Transformer
        transformer_input = x
        
        # Encoder
        conv1 = self.encoder1(x)
        conv2 = self.encoder2(conv1)
        conv3 = self.encoder3(conv2)
        
        # 如果启用Transformer，则进行特征融合
        if self.use_transformer:
            conv3 = self._fuse_features(conv3, transformer_input)
        
        # Bridge
        bridge = self.bridge(conv3)
        
        # Decoder
        up3 = self.decoder3(bridge, conv3)
        up2 = self.decoder2(up3, conv2)
        up1 = self.decoder1(up2, conv1)
        
        # Output
        out = self.conv_last(up1)
        out = self.adaptive_pool(out)
        out = out.squeeze(2)
        
        return out

if __name__ == "__main__":
    # 设置较小的 batch size 来减少内存使用
    batch_size = 2
    writer = SummaryWriter('runs/ecg_unet_transformer_model')
    # 创建模型实例
    print("初始化模型...")
    model_with_transformer = UNet(input_size=2500, use_transformer=True)
    
    # 生成较小的测试输入
    test_input = torch.randn(batch_size, 1, 2500)

    # 先运行一次前向传播
    try:
        output = model_with_transformer(test_input)
        print(f"模型测试成功，输出形状: {output.shape}")
    
        # 使用 torch.jit.trace 创建可追踪的模型
        script_model = torch.jit.script(model_with_transformer, test_input)
        writer.add_graph(script_model, test_input)
        print("模型图保存成功")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试不带Transformer的模型
    # try:
    #     output = model_without_transformer(test_input)
    #     print(f"模型测试成功，输出形状: {output.shape}")
    #     # 添加模型图到 TensorBoard
    #     writer.add_graph(model_without_transformer, test_input)
    # except Exception as e:
    #     print(f"错误: {e}")


    finally:
        writer.close()
    
        print("\nTensorBoard数据已写入。请运行以下命令查看:")
        print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/ecg_unet_transformer_model")