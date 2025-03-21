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

class UNet(nn.Module):
    def __init__(self, input_size=2500):
        super(UNet, self).__init__()
        
        # Encoder
        self.conv1 = DoubleConv(1, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bridge
        self.bridge = DoubleConv(256, 512)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(512, 256)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)
        self.upconv1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        # Output
        self.conv_last = nn.Conv1d(64, 2, kernel_size=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Bridge
        bridge = self.bridge(pool3)
        
        # Decoder - 添加尺寸调整以确保特征图匹配
        up3 = self.upconv3(bridge)
        # 裁剪conv3以匹配up3的尺寸
        diff3 = conv3.size()[2] - up3.size()[2]
        if diff3 > 0:
            conv3 = conv3[:, :, diff3//2:-(diff3-diff3//2)]
        elif diff3 < 0:
            up3 = up3[:, :, -diff3//2:(diff3-diff3//2)]
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.conv_up3(up3)
        
        up2 = self.upconv2(up3)
        # 裁剪conv2以匹配up2的尺寸
        diff2 = conv2.size()[2] - up2.size()[2]
        if diff2 > 0:
            conv2 = conv2[:, :, diff2//2:-(diff2-diff2//2)]
        elif diff2 < 0:
            up2 = up2[:, :, -diff2//2:(diff2-diff2//2)]
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.conv_up2(up2)
        
        up1 = self.upconv1(up2)
        # 裁剪conv1以匹配up1的尺寸
        diff1 = conv1.size()[2] - up1.size()[2]
        if diff1 > 0:
            conv1 = conv1[:, :, diff1//2:-(diff1-diff1//2)]
        elif diff1 < 0:
            up1 = up1[:, :, -diff1//2:(diff1-diff1//2)]
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.conv_up1(up1)
        
        # Output
        out = self.conv_last(up1)
        out = self.adaptive_pool(out)
        out = out.squeeze(2)
        
        return out

if __name__ == "__main__":
    # 创建 TensorBoard writer
    writer = SummaryWriter('runs/ecg_unet_model')
    
    # 创建模型实例
    model = UNet(input_size=2500)
    
    # 生成测试输入
    test_input = torch.randn(32, 1, 2500)  # batch_size=32, channels=1, sequence_length=2500
    
    # 先测试模型是否能正常运行
    try:
        output = model(test_input)
        print(f"模型测试成功，输出形状: {output.shape}")
        
        # 添加模型图到 TensorBoard
        writer.add_graph(model, test_input)
        writer.close()
    except Exception as e:
        print(f"错误: {e}")
        writer.close()
    
    # 测试输出
    output = model(test_input)
    print(f"输出形状: {output.shape}")  # 应该输出 torch.Size([32, 2])



    # tensorboard --logdir=/Users/xingyulu/Public/physionet/runs/ecg_unet_model