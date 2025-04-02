import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shutil
from tqdm import tqdm  

# 如果有其他自定义模块或类需要导入，请确保它们在项目路径中可用
import sys
sys.path.append('/Users/xingyulu/Public/physionet')
sys.path.append('/Users/xingyulu/Public/physionet/utils/fsst_convert')
from models.seg_model_cnn_lstm import AFSegmentationModel, ECGDataset


# 结合和了时间和频率两种模式的训练脚本
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda', patience=10, model_save_path='best_model.pth'):
    """
    训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 训练设备
        patience: 早停耐心值
        model_save_path: 模型保存路径
        
    Returns:
        训练历史记录
    """
    model = model.to(device)
    best_val_loss = float('inf')
    counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'train_accuracy': []}
    
    # 添加epoch进度条
    epoch_pbar = tqdm(range(num_epochs), desc='Training', position=0)
    
    for epoch in epoch_pbar:
        # region ==================训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0  # 新增：训练正确预测数
        train_total = 0    # 新增：训练总样本数
        
        # 添加batch进度条
        batch_pbar = tqdm(enumerate(train_loader), 
                         total=len(train_loader),
                         desc=f'Epoch {epoch+1}/{num_epochs}',
                         position=1,
                         leave=False)
        
        for batch_idx, (signals, labels) in batch_pbar:
            signals = signals.to(device)
            labels = labels.long().to(device)  # 提前转换类型 shape(batch_size, 2500)

            optimizer.zero_grad()
            # 前向传播
            outputs = model(signals) #shape(batch_size, 2500, 5)

            # 调整维度顺序
            outputs = outputs.permute(0, 2, 1) # shape(batch_size, 5, 2500)
            
            # 获取预测类别并计算准确率
            _, preds = torch.max(outputs, dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.numel()
            
            # 累加到总计数器
            train_correct += batch_correct
            train_total += batch_total
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
            
            # 更新进度条显示当前批次损失和准确率
            batch_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'batch_acc': f'{batch_correct/batch_total:.4f}'
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = train_correct / train_total  # 计算整体训练准确率
        # endregion ====================训练阶段
        
        # region ====================验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0  # 正确预测的样本数
        val_total = 0    # 总样本数
        # 添加验证集进度条
        val_pbar = tqdm(val_loader, 
                       desc='Validating',
                       position=1,
                       leave=False)
        with torch.no_grad():  # 确保验证阶段不计算梯度
            for signals, labels in val_pbar:
                # 数据转移到设备并转换标签类型
                signals = signals.to(device)
                labels = labels.long().to(device)  # 转换类型 shape(batch_size, 2500)
                
                # 前向传播
                outputs = model(signals)
                outputs = outputs.permute(0, 2, 1) # shape(batch_size, 5, 2500)
                
                # 计算损失
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)
                
                # 获取预测类别
                _, preds = torch.max(outputs, dim=1)
                
                # 计算当前批次的正确预测数
                batch_correct = (preds == labels).sum().item()
                batch_total = labels.numel()  # 当前批次的总样本数
                
                # 累加到总计数器
                val_correct += batch_correct
                val_total += batch_total
                
                # 更新进度条显示当前批次准确率
                val_pbar.set_postfix({'batch_acc': f'{batch_correct/batch_total:.4f}'})
        
        # 计算总体验证损失和准确率
        val_loss = val_loss / len(val_loader.dataset)
        accuracy = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        # 在保存历史记录时添加train_accuracy
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)
        history['train_accuracy'].append(train_accuracy)  # 新增：计算并保存训练集准确率

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.4f}, train_accuracy: {train_accuracy:.4f}')
        # endregion ====================验证阶段

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
        
                

def evaluate_model(model, test_loader, device='cuda'):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            labels = labels.long()

            # 前向传播
            outputs = model(signals)
            
            # 获取预测类别 (形状 [batch_size, 2500])
            _, preds = torch.max(outputs, dim=-1)
            
            # 收集预测结果和标签
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro')
    }
    
    return metrics

def plot_training_history(history, save_path=None):
    """
    绘制训练历史记录
    
    Args:
        history: 训练历史记录字典
        save_path: 图像保存路径
    """
    # 添加history参数验证
    if history is None:
        print("警告：history参数为None，无法绘制训练历史")
        return
        
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制评估指标曲线
    plt.subplot(2, 1, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Evaluation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Training history plot saved to {save_path}')
    
    plt.show()


def main():
    """
    主函数，演示模型的训练和评估流程
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置参数
    data_dir = '/Users/xingyulu/Public/afafaf/用于尝试训练'
    batch_size = 16
    learning_rate = 0.0001
    num_epochs = 50
    patience = 40
    data_mode = 'time' # 这里要指定是'fsst'还是'time'模式
    model_save_path = '/Users/xingyulu/Public/physionet/models/saved/af_segmentation_best.pth'
    history_plot_path = '/Users/xingyulu/Public/physionet/result/training_history.png'
    

    # region 1.0 数据读取和模型配置
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)

    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # 创建数据集和数据加载器
    train_dataset = ECGDataset(train_dir, mode=data_mode) # 设置模式为'fsst' 或 'time'
    val_dataset = ECGDataset(val_dir, mode=data_mode)
    test_dataset = ECGDataset(test_dir, mode=data_mode)
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 初始化模型
    model = AFSegmentationModel(mode=data_mode, hidden_size=64, num_layers=2, dropout=0.3)
    print(f'模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # endregion 1.0
    
    # region 1.1 模型训练前添加参数验证
    sample, label = train_dataset[0]
    print(f"输入形状验证: 样本形状={sample.shape}（应为[2500]）, 标签形状={label.shape}（应为[2500]）")
    
    # 在模型初始化后添加参数验证
    print(f"第一个卷积层权重形状: {model.conv1.weight.shape}（应为[32, 1, 5]）")
    print(f"标签极值验证: min={label.min().item():.4f}, max={label.max().item():.4f}") 
    
    # 检查是否存在非0/1标签
    invalid_labels = ((label != 0) & (label != 1)).sum()
    print(f"无效标签数量: {invalid_labels}")
    print(f"信号形状: {sample.shape}, 标签形状: {label.shape}")
    print(f"标签取值范围: min={label.min().item():.4f}, max={label.max().item():.4f}")
    
    # 检查是否包含异常值
    unique_labels = torch.unique(label)
    print(f"标签唯一值: {unique_labels}")
    # endregion 1.1

    
    # 训练模型
    print('开始训练模型...')
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
        patience=patience,
        model_save_path=model_save_path
    )

    
    # 绘制训练历史记录
    plot_training_history(history, save_path=history_plot_path)
    
    # 加载最佳模型进行评估
    model.load_state_dict(torch.load(model_save_path))
    
    # 在测试集上评估模型
    print('在测试集上评估模型...')
    metrics = evaluate_model(model, test_loader, device=device)
    
    print('\n测试集评估结果:')
    for metric_name, metric_value in metrics.items():
        print(f'{metric_name}: {metric_value:.4f}')
    
    
    print('模型训练和评估完成!')


if __name__ == "__main__":
    main()