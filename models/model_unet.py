import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

class ECGDataset(Dataset):
    def __init__(self, data_dir, mode='time'):
        self.data_dir = data_dir
        self.mode = mode
        self.data = []
        self.labels = []
        
        # 加载数据文件
        self._load_data()
        
    def _load_data(self):
        """加载数据文件"""
        for file in os.listdir(self.data_dir):
            if file.endswith('.npy'):
                file_path = os.path.join(self.data_dir, file)
                arr = np.load(file_path, allow_pickle=True)
                
                # 确保数据格式正确
                if isinstance(arr, np.ndarray) and arr.ndim == 2:
                    self.data.append(arr[0])  # 信号数据
                    self.labels.append(arr[1])  # 标签数据
                else:
                    raise ValueError(f"文件 {file} 格式不正确，应为二维数组")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 转换为张量并返回
        signal = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal, label

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
    def __init__(self, in_channels, out_channels, output_size=None):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.output_size = output_size
        
    def forward(self, x, skip_connection):
        x = self.upconv(x)
        
        # 如果指定了输出尺寸，则调整x的尺寸
        if self.output_size is not None:
            x = nn.functional.interpolate(x, size=self.output_size, mode='linear', align_corners=False)
            
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
    def __init__(self, input_size, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1, patch_size=50):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv1d(
            in_channels=1, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 计算patch数量
        self.num_patches = input_size // patch_size
        if input_size % patch_size != 0:
            raise ValueError(f"输入尺寸{input_size}必须能被patch_size{patch_size}整除")
            
        self.pos_encoder = nn.Parameter(torch.zeros(self.num_patches, embed_dim))
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_dim, 256)

    def forward(self, x):
        # 输入形状: [batch_size, 1, seq_len]
        batch_size = x.shape[0]
        
        # 使用卷积进行patch划分 [batch_size, embed_dim, num_patches]
        x = self.patch_embed(x)
        
        # 转换为Transformer输入格式 [num_patches, batch_size, embed_dim]
        x = x.permute(2, 0, 1)
        
        # 添加位置编码
        x = x + self.pos_encoder.unsqueeze(1)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
            
        # 输出层调整 [batch_size, 256, num_patches]
        x = self.output_layer(x)  # [num_patches, batch_size, 256]
        x = x.permute(1, 2, 0)    # 转换到[batch_size, 256, num_patches]
        
        # 上采样回原始长度 [batch_size, 256, seq_len]
        x = nn.functional.interpolate(x, size=2500, mode='linear', align_corners=False)
        return x

class UNet(nn.Module):
    def __init__(self, input_size=2500, use_transformer=True, num_classes=5):
        super(UNet, self).__init__()
        self.use_transformer = use_transformer
        self.num_classes = num_classes
        
        # Encoder
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        
        # Transformer特征提取器（仅当use_transformer=True时使用）
        if self.use_transformer:
            self.transformer = TransformerEncoder(
                input_size=input_size,
                embed_dim=256,  # embed_dim​：嵌入向量的维度，表示每个token/时间步的特征维度
                num_heads=8,
                ff_dim=512,
                num_layers=3,
                dropout=0.1,
                patch_size=50  # 新增patch_size参数
            )
        
        # Bridge
        self.bridge = DoubleConv(256, 512)
        
        # Decoder
        self.decoder3 = DecoderBlock(512, 256)
        self.decoder2 = DecoderBlock(256, 128)
        self.decoder1 = DecoderBlock(128, 64)
        
        self.conv_last = nn.Conv1d(64, num_classes, kernel_size=1)
        # 添加上采样层以恢复原始序列长度
        self.upsample = nn.Upsample(size=input_size, mode='linear', align_corners=False)
        
    def _fuse_features(self, conv_features, transformer_features):
        """
        融合卷积特征和Transformer特征
        
        Args:
            conv_features: 卷积网络提取的特征
            transformer_features: Transformer的特征提取结果
            
        Returns:
            融合后的特征
        """
            
        # 获取并验证在Unet那个底部的卷积特征的时间维度大小（也就是encoder3的最后一个维度的数值，目前是312）
        target_size = int(conv_features.size(2))  # 显式转换为整数
        if target_size <= 0:
            raise ValueError(f"无效的target_size: {target_size}")
            
        # 调整transformer_features的大小，将Transformer输出（原始长度2500）下采样到与卷积特征相同的时间维度（312）
        # ​为什么需要对齐​：
        # 卷积路径通过步长卷积逐步减小序列长度（2500→1250→625→312）
        # Transformer路径保持原始长度（2500）
        # 必须统一时间维度才能进行特征拼接
        transformer_features = nn.functional.adaptive_avg_pool1d(
            transformer_features, 
            output_size=target_size
        )
        
        # 特征拼接​：沿通道维度（dim=1）拼接：
        # conv_features: [batch, 256, 312]
        # transformer_features: [batch, 256, 312]
        # 拼接后: [batch, 512, 312]
        combined_features = torch.cat([conv_features, transformer_features], dim=1)

        # 获取输入输出通道数
        in_channels = combined_features.size(1)  # 512
        out_channels = conv_features.size(1)    # 256
        self.fusion_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 卷积层处理融合后的特征融合层​：
        # 用一个卷积层，用于将512通道的特征映射到256通道
        # 作用：将拼接后的512通道压缩回256通道，同时学习两种特征的交互
        fused_features = self.fusion_layer(combined_features)
        return fused_features
        
    def forward(self, x, return_features=False):
        # 输入验证
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f"输入形状应为[batch, 1, seq_len], 但得到的是{x.shape}")
            
        # 保存原始输入用于Transformer
        transformer_input = x  # shape: [batch_size, 1, 2500]
        
        # CNN路径（逐步下采样）
        conv1 = self.encoder1(x)  #conv1 shape: [batch_size, 64, 1250]
        conv2 = self.encoder2(conv1)  #conv2 shape: [batch_size, 128, 625]
        conv3 = self.encoder3(conv2)  #conv3 shape: [batch_size, 256, 312]

        # 如果启用Transformer，则进行特征融合
        if self.use_transformer:
            # Transformer路径（保持原始长度）
            transformer_output = self.transformer(transformer_input)  # shape: [batch_size, 256, 2500]
            # 特征融合（关键步骤）
            conv3 = self._fuse_features(conv3, transformer_output)  # shape: [batch_size, 256, 312]
        
        # Bridge
        bridge = self.bridge(conv3)  # shape: [batch_size, 512, 312]
        
        # Decoder
        up3 = self.decoder3(bridge, conv3)  # up3 shape: [batch_size, 256, 312]
        up2 = self.decoder2(up3, conv2)  # up2 shape: [batch_size, 128, 624]
        up1 = self.decoder1(up2, conv1)  # up1 shape: [batch_size, 64, 1248]
        
        # Output
        out = self.conv_last(up1)  # shape: [batch_size, 5, 1248]
        out = self.upsample(out)  # shape: [batch_size, 5, 2500]
        
        if return_features:
            return out, {
                'encoder_features': [conv1, conv2, conv3],
                'decoder_features': [up1, up2, up3],
                'bridge': bridge
            }
        return out

if __name__ == "__main__":
    # 设置较小的 batch size 来减少内存使用
    batch_size = 2
    writer = SummaryWriter('runs/ecg_unet_segmentation_model')
    # 创建模型实例
    print("初始化模型...")
    model_without_transformer = UNet(input_size=2500, use_transformer=False, num_classes=5)
    model_with_transformer = UNet(input_size=2500, use_transformer=True, num_classes=5)
    
    # 生成较小的测试输入
    test_input = torch.randn(batch_size, 1, 2500)

    # 测试带Transformer的模型
    try:
        output, features = model_with_transformer(test_input, return_features=True)

        writer.add_graph(model_with_transformer, test_input)
        print(f"带Transformer模型测试成功，输出形状: {output.shape}")
        # 查看参数量
        num_params = sum(p.numel() for p in model_with_transformer.parameters() if p.requires_grad)
        print(f"模型参数量: {num_params}")
    except Exception as e:
        print(f"带Transformer模型错误: {e}")
    
    # 测试不带Transformer的模型
    try:
        output, features  = model_without_transformer(test_input, return_features=True)
        print(f"不带Transformer模型测试成功，输出形状: {output.shape}")
        num_params = sum(p.numel() for p in model_without_transformer.parameters() if p.requires_grad)
        print(f"模型参数量: {num_params}")
        # 添加模型图到 TensorBoard
        # writer.add_graph(model_without_transformer, test_input)
    except Exception as e:
        print(f"错误: {e}")

    finally:
        writer.close()
    
        print("\nTensorBoard数据已写入。请运行以下命令查看:")
        print("tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/ecg_unet_segmentation_model")