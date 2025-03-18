import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shutil

# 如果有其他自定义模块或类需要导入，请确保它们在项目路径中可用
import sys
sys.path.append('/Users/xingyulu/Public/physionet')
from models.vf_segmentation_model import VFSegmentationModel, ECGDataset

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
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1': []}
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            # 前向传播前添加梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(signals).squeeze(-1)
            
            # 新增梯度裁剪和更严格的数值稳定处理
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            outputs = torch.clamp(outputs, min=1e-4, max=1.0-1e-4)
            
            # 新增输入数据验证
            if torch.isnan(signals).any():
                print("[CRITICAL] 输入信号包含NaN值！")
            
            # 调试信息增加NaN检查
            print(f"[DEBUG] 输出范围: ({outputs.min().item():.4f}, {outputs.max().item():.4f})",
                  f"NaN数量: {torch.isnan(outputs).sum().item()}")
            
            # 新增数值稳定处理（关键修复）
            outputs = torch.clamp(outputs, min=1e-7, max=1.0-1e-7)
            
            # 新增标签类型转换（确保标签为浮点型）
            labels = labels.float()
            
            # 新增调试信息（定位具体问题）
            print(f"[DEBUG] 输出范围: ({outputs.min().item():.4f}, {outputs.max().item():.4f})")
            print(f"[DEBUG] 标签范围: ({labels.min().item():.4f}, {labels.max().item():.4f})")
            
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(signals).squeeze(-1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * signals.size(0)
                
                # 收集预测结果和标签
                preds = (outputs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # 计算评估指标
        all_preds = np.concatenate([p.flatten() for p in all_preds])
        all_labels = np.concatenate([l.flatten() for l in all_labels])
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)
        history['val_f1'].append(f1)
        
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}')
        
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
                break
    
    return history


def evaluate_model(model, test_loader, device='cuda'):
    """
    评估模型
    
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        device: 评估设备
        
    Returns:
        评估指标字典
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for signals, labels in test_loader:
            signals, labels = signals.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(signals).squeeze(-1)
            
            # 新增数值检查
            if (outputs.min() < 0) or (outputs.max() > 1):
                print(f"测试输出越界: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
            
            # 收集预测结果和标签
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    
    # 合并所有批次的结果
    all_probs = np.concatenate([p.flatten() for p in all_probs])
    all_preds = np.concatenate([p.flatten() for p in all_preds])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics


def plot_training_history(history, save_path=None):
    """
    绘制训练历史记录
    
    Args:
        history: 训练历史记录字典
        save_path: 图像保存路径
    """
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
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1 Score')
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


def predict_and_visualize(model, signal, device='cuda', window_size=2500):
    """
    对单个信号进行预测并可视化结果
    
    Args:
        model: 模型实例
        signal: 输入信号，形状为 [sequence_length]
        device: 预测设备
        window_size: 窗口大小
        
    Returns:
        预测结果
    """
    model = model.to(device)
    model.eval()
    
    # 确保信号长度是窗口大小的倍数
    if len(signal) % window_size != 0:
        pad_length = window_size - (len(signal) % window_size)
        signal = np.pad(signal, (0, pad_length), 'constant')
    
    # 将信号分割成窗口
    num_windows = len(signal) // window_size
    windows = np.array_split(signal, num_windows)
    
    predictions = []
    
    with torch.no_grad():
        for window in windows:
            # 转换为张量并添加批次维度
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 前向传播
            output = model(window_tensor).squeeze(-1)
            
            # 收集预测结果
            pred = (output > 0.5).float().cpu().numpy()
            predictions.append(pred[0])
    
    # 合并所有窗口的预测结果
    predictions = np.concatenate(predictions)
    
    # 可视化
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('ECG Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions)
    plt.title('VF Prediction (1 = VF, 0 = Non-VF)')
    plt.xlabel('Sample')
    plt.ylabel('Prediction')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return predictions


def main():
    """
    主函数，演示模型的训练和评估流程
    """
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置参数
    data_dir = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/分割任务/transition_segments'
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    patience = 10
    model_save_path = '/Users/xingyulu/Public/physionet/models/saved/vf_segmentation_best.pth'
    history_plot_path = '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/分割任务/results/training_history.png'
    

    # region 1.0 数据读取和模型配置
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建保存目录
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_plot_path), exist_ok=True)
    
    # 数据集划分
    all_files = list(Path(data_dir).glob('*.npy'))
    np.random.shuffle(all_files)
    
    # 按照8:1:1的比例划分训练集、验证集和测试集
    train_size = int(0.8 * len(all_files))
    val_size = int(0.1 * len(all_files))
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size+val_size]
    test_files = all_files[train_size+val_size:]
    
    # 创建临时目录存放划分后的数据
    temp_dirs = {
        'train': Path('/Users/xingyulu/Public/physionet/data/temp/train'),
        'val': Path('/Users/xingyulu/Public/physionet/data/temp/val'),
        'test': Path('/Users/xingyulu/Public/physionet/data/temp/test')
    }
    
    for dir_path in temp_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        # 清空目录
        for file in dir_path.glob('*.npy'):
            file.unlink()
    
    # 复制文件到临时目录
    import shutil
    for file, dir_name in zip(
        [train_files, val_files, test_files],
        ['train', 'val', 'test']
    ):
        for f in file:
            shutil.copy(f, temp_dirs[dir_name])
    
    # 创建数据集和数据加载器
    train_dataset = ECGDataset(temp_dirs['train'])
    val_dataset = ECGDataset(temp_dirs['val'])
    test_dataset = ECGDataset(temp_dirs['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 初始化模型
    model = VFSegmentationModel(input_channels=1, hidden_size=64, num_layers=2, dropout=0.3)
    print(f'模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # 定义损失函数、优化器和学习率调度器
    criterion = nn.BCELoss()
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
    
    # 可视化一个测试样本的预测结果
    print('\n可视化预测结果...')
    test_sample, test_label = test_dataset[0]
    predictions = predict_and_visualize(
        model=model,
        signal=test_sample.numpy(),
        device=device,
        window_size=len(test_sample)
    )
    
    # 清理临时目录
    for dir_path in temp_dirs.values():
        shutil.rmtree(dir_path, ignore_errors=True)
    
    print('模型训练和评估完成!')


if __name__ == "__main__":
    main()