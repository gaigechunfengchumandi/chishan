import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 定义数据集类（与model.py保持一致）
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ECGClassifier(nn.Module):
    def __init__(self, input_size=2500, hidden_size=128, num_layers=2):
        super(ECGClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=1,  # 每个时间步的输入特征数
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # 双向LSTM，所以hidden_size * 2
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)  # 二分类问题
        
    def forward(self, x):
        # 输入形状: [batch_size, 1, sequence_length]
        batch_size = x.size(0)
        
        # 将输入reshape为 (batch_size, sequence_length, 1)
        x = x.permute(0, 2, 1)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)  # 双向LSTM
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        x = torch.relu(self.fc1(out))
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x

# 使用示例
if __name__ == "__main__":
    # 创建 TensorBoard writer
    writer = SummaryWriter('runs/ecg_model')
    
    # 创建模型实例
    model = ECGClassifier(input_size=2500)
    
    # 生成测试输入
    test_input = torch.randn(32, 1, 2500)  # batch_size=32, channels=1, sequence_length=2500
    
    # 添加模型图到 TensorBoard
    writer.add_graph(model, test_input)
    writer.close()
    
    # 测试输出
    output = model(test_input)
    print(f"输出形状: {output.shape}")  # 应该输出 torch.Size([32, 2])