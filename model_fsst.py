import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 设置随机种子以确保结果可复现
torch.manual_seed(42)

# 定义数据集类
class ECGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ECGClassifier(nn.Module):
    def __init__(self, input_size=2500, num_channels=40):
        super(ECGClassifier, self).__init__()
        self.num_channels = num_channels
        
        # 卷积层保持不变
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # # 计算全连接层的输入大小
        # self.fc_input_size = 128 * (input_size // 16)
        
        # 添加全局平均池化层
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 修改全连接层
        self.fc1 = nn.Linear(128 * num_channels, 256)  # 修改输入维度
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 2)  # 二分类问题
        
    def forward(self, x):
        # 输入形状: [batch_size, num_channels, sequence_length]
        batch_size = x.size(0)
        
        # 将输入reshape为 (batch_size * num_channels, 1, sequence_length)
        x = x.view(-1, 1, x.size(2))
        
        # 卷积层
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 将特征图reshape回 (batch_size, num_channels, features)
        x = x.view(batch_size, self.num_channels, -1)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    
    def visualize_simple(self, save_path="/Users/xingyulu/Public/physionet/plots/model_simple.png"):
        """
        使用matplotlib绘制美观的模型结构图
        Args:
            save_path: 保存图像的路径
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 获取模型的所有层并按类型分组
            layers = []
            layer_types = {
                'Conv1d': '#FF9999',     # 红色系
                'BatchNorm1d': '#99FF99', # 绿色系
                'MaxPool1d': '#9999FF',   # 蓝色系
                'Linear': '#FFCC99',      # 橙色系
                'Dropout': '#FF99FF'      # 紫色系
            }
            
            for name, module in self.named_children():
                if isinstance(module, nn.Sequential):
                    for n, m in module.named_children():
                        layers.append({
                            'name': f"{name}.{n}",
                            'type': m.__class__.__name__,
                            'color': layer_types.get(m.__class__.__name__, '#CCCCCC')
                        })
                else:
                    layers.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'color': layer_types.get(module.__class__.__name__, '#CCCCCC')
                    })
            
            # 创建图形
            plt.style.use('seaborn')
            fig = plt.figure(figsize=(12, len(layers) * 0.7))
            ax = plt.gca()
            y_positions = np.arange(len(layers))
            
            # 绘制每一层
            bars = ax.barh(y_positions, [1] * len(layers), height=0.5, 
                          align='center', 
                          color=[layer['color'] for layer in layers],
                          alpha=0.7)
            
            # 添加层名称和类型
            for i, layer in enumerate(layers):
                # 添加层名称
                ax.text(0.02, i, layer['name'], 
                       ha='left', va='center',
                       fontsize=10, fontweight='bold')
                # 添加层类型
                ax.text(0.98, i, layer['type'],
                       ha='right', va='center',
                       fontsize=9, style='italic')
            
            # 设置图形属性
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim(-0.1, 1.1)
            
            # 添加标题和边框
            plt.title('神经网络模型结构', pad=20, fontsize=14, fontweight='bold')
            
            # 添加图例
            legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7) 
                             for color in layer_types.values()]
            ax.legend(legend_elements, layer_types.keys(), 
                     loc='center left', bbox_to_anchor=(1, 0.5))
            
            # 添加层间连接线
            for i in range(len(layers)-1):
                ax.plot([0.5, 0.5], [i+0.3, i+0.7], 
                       color='gray', linestyle='--', alpha=0.5)
            
            # 保存图形
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"美化后的模型结构图已保存至: {save_path}")
        except ImportError:
            print("请先安装matplotlib: pip install matplotlib")

# 使用示例
if __name__ == "__main__":
    model = ECGClassifier(input_size=2500, num_channels=40)
    x = torch.randn(32, 40, 2500)  # batch_size=32, num_channels=40, sequence_length=2500
    output = model(x)
    print(output.shape)  # 应该输出 torch.Size([32, 2])
    model.visualize_simple()