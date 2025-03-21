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
        # 添加通道维度转换层
        self.channel_adjust = nn.Conv1d(embed_dim, 256, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, seq_len = x.shape
        
        x = x.permute(2, 0, 1)  # [seq_len, batch_size, 1]
        x = self.embedding(x)    # [seq_len, batch_size, embed_dim]
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x)
        
        # 转换形状并调整通道数
        x = x.permute(1, 2, 0)  # [batch_size, embed_dim, seq_len]
        x = self.channel_adjust(x)  # [batch_size, 256, seq_len]
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
            # 修正融合层输入通道为256（卷积特征）+256（Transformer特征）=512
            self.fusion = nn.Conv1d(512, 512, kernel_size=1)
        
        # Bridge
        self.bridge = DoubleConv(256 if not self.use_transformer else 512, 512)
        
        # Decoder
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        
        # Output
        self.conv_last = nn.Conv1d(64, 2, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # 保存原始输入用于Transformer
        transformer_input = x
        
        # Encoder
        conv1 = self.encoder1(x)
        conv2 = self.encoder2(conv1)
        conv3 = self.encoder3(conv2)
        
        # 如果启用Transformer，则进行特征融合
        if self.use_transformer:
            # Transformer特征提取
            transformer_features = self.transformer(transformer_input)
            
            # 调整transformer_features的大小以匹配conv3
            if transformer_features.size(2) != conv3.size(2):
                transformer_features = nn.functional.adaptive_avg_pool1d(
                    transformer_features, conv3.size(2)
                )
            
            # 特征融合
            combined_features = torch.cat([conv3, transformer_features], dim=1)
            conv3 = self.fusion(combined_features)
        
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
    # 创建 TensorBoard writer
    writer = SummaryWriter('runs/ecg_unet_transformer_model')
    # 创建模型实例 - 可以选择是否使用Transformer
    model_with_transformer = UNet(input_size=2500, use_transformer=True)
    # model_without_transformer = UNet(input_size=2500, use_transformer=False)
    
    # 生成测试输入
    test_input = torch.randn(32, 1, 2500)  # batch_size=32, channels=1, sequence_length=2500
    
    # 测试带有Transformer的模型
    try:
        print("测试带有Transformer的模型:")
        output = model_with_transformer(test_input)
        print(f"模型测试成功，输出形状: {output.shape}")
        
        # 添加模型图到 TensorBoard
        writer.add_graph(model_with_transformer, test_input)
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试不带Transformer的模型
    # try:
    #     print("\n测试不带Transformer的模型:")
    #     output = model_without_transformer(test_input)
    #     print(f"模型测试成功，输出形状: {output.shape}")
    #     # 添加模型图到 TensorBoard
    #     writer.add_graph(model_without_transformer, test_input)
    # except Exception as e:
    #     print(f"错误: {e}")
    
    writer.close()
    
    print("\nTensorBoard数据已写入。请运行以下命令查看:")
    print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/ecg_unet_transformer_model")