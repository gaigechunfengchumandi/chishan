import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader  # 重命名导入
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import pandas as pd
from tqdm import tqdm
from model import ECGDataset, ECGClassifier
from config import PATH_CONFIG, DATA_CONFIG, MODEL_CONFIG, AUGMENTATION_CONFIG

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 加载和预处理数据
class CustomDataLoader:  # 重命名您的自定义类
    def __init__(self, config):
        self.config = config
        self.non_vf_dir = PATH_CONFIG['non_vf_dir']
        self.vf_dir = PATH_CONFIG['vf_dir']
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler('data_loader.log')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_single_folder(self, folder_path):
        data = []
        self.logger.info(f'开始加载文件夹: {folder_path}')
        target_length = DATA_CONFIG['max_sequence_length']
        
        for filename in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)
            try:
                file_data = np.load(file_path)  # 从文件中加载数据
                if len(file_data) > 0:  # 确保数据不为空
                    # 标准化数据形状
                    if len(file_data) > target_length:
                        file_data = file_data[:target_length]  # 截断过长的数据到目标长度
                    elif len(file_data) < target_length:
                        # 填充
                        padding = np.zeros((target_length - len(file_data),) + file_data.shape[1:])  # 创建零填充数组
                        file_data = np.concatenate([file_data, padding])  # 将原始数据和填充数据连接起来
                    
                    data.append(file_data)  # 将处理后的数据添加到数据列表中
            except Exception as e:
                self.logger.error(f'加载文件失败: {file_path}, 错误: {str(e)}')
        self.logger.info(f'文件夹加载完成: {folder_path}, 加载数据量: {len(data)}')
        return data

    def load_all_data(self):
        """
        加载所有数据
        Returns:
            tuple: (non_vf_data, vf_data)
        """
        non_vf_data = self.load_single_folder(self.non_vf_dir)
        vf_data = self.load_single_folder(self.vf_dir)
        return non_vf_data, vf_data

# 修改load_data函数，移除筛选逻辑
def load_data(data_augmentation=DATA_CONFIG['data_augmentation']):
    """
    加载数据的主函数
    Args:
        data_augmentation: 是否进行数据增强
    Returns:
        tuple: (all_data, all_labels, max_length)
    """
    try:
        loader = CustomDataLoader(PATH_CONFIG)  # 使用新的类名
        non_vf_data, vf_data = loader.load_all_data()
        
        # 检查数据是否成功加载
        if not non_vf_data or not vf_data:
            raise ValueError("数据加载失败:数据集为空")
            
        print(f"非VF窗口数量: {len(non_vf_data)}")
        print(f"VF窗口数量: {len(vf_data)}")
        
        # 使用配置中的最大序列长度，不再计算实际数据中的最小长度
        max_length = DATA_CONFIG['max_sequence_length']
        
        # 检查数据形状
        sample_shape = non_vf_data[0].shape
        print(f"样本数据形状: {sample_shape}")
        
        # 数据增强 - 只对少数类(VF)进行增强
        if data_augmentation:
            print("执行数据增强...")
            augmented_vf_data = []
            
            for data in tqdm(vf_data):
                # 确保数据形状一致
                if data.shape != sample_shape:
                    print(f"警告: 跳过形状不一致的数据 {data.shape}")
                    continue
                    
                # 添加噪声
                noise = np.random.normal(0, AUGMENTATION_CONFIG['noise_std'], size=data.shape)
                augmented_vf_data.append(data + noise)
                
                # 时间偏移
                shift = np.random.randint(*AUGMENTATION_CONFIG['shift_range'])
                shifted_data = np.roll(data, shift)
                augmented_vf_data.append(shifted_data)
                
                # 振幅缩放
                scale_factor = np.random.uniform(*AUGMENTATION_CONFIG['scale_range'])
                augmented_vf_data.append(data * scale_factor)
            
            vf_data.extend(augmented_vf_data)
            print(f"数据增强后的VF窗口数量: {len(vf_data)}")
        
        # 创建标签
        non_vf_labels = np.zeros(len(non_vf_data))
        vf_labels = np.ones(len(vf_data))
        
        # 合并数据和标签
        print("合并数据...")
        try:
            all_data = np.stack(non_vf_data + vf_data)
            print(f"合并后的数据形状: {all_data.shape}")
        except Exception as e:
            print(f"合并数据时出错: {str(e)}")
            print("尝试检查并修复数据形状...")
            
            # 检查所有数据形状
            shapes = [arr.shape for arr in non_vf_data + vf_data]
            unique_shapes = set(str(shape) for shape in shapes)
            if len(unique_shapes) > 1:
                print(f"检测到不同的数据形状: {unique_shapes}")
                
                # 过滤掉形状不一致的数据
                filtered_non_vf = [arr for arr in non_vf_data if arr.shape == sample_shape]
                filtered_vf = [arr for arr in vf_data if arr.shape == sample_shape]
                
                print(f"过滤后的非VF窗口数量: {len(filtered_non_vf)}")
                print(f"过滤后的VF窗口数量: {len(filtered_vf)}")
                
                # 更新标签
                non_vf_labels = np.zeros(len(filtered_non_vf))
                vf_labels = np.ones(len(filtered_vf))
                
                # 重新尝试合并
                all_data = np.stack(filtered_non_vf + filtered_vf)
                print(f"过滤后合并的数据形状: {all_data.shape}")
            else:
                raise ValueError("无法解决数据形状不一致问题")
        
        all_labels = np.concatenate([non_vf_labels, vf_labels])
        
        # 数据标准化
        mean = np.mean(all_data)
        std = np.std(all_data)
        all_data = (all_data - mean) / std
        
        return all_data, all_labels, max_length
        
    except Exception as e:
        print(f"数据加载过程中发生错误: {str(e)}")
        raise

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=MODEL_CONFIG['num_epochs'], patience=MODEL_CONFIG['patience']):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_no_improve = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': train_correct / train_total
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': val_correct / val_total
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
              f'Time: {epoch_time:.2f}s')
        
        # 保存最佳模型并检查早停
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"早停: {patience} 个epoch没有改善")
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}秒")
    
    return model, train_losses, val_losses, train_accs, val_accs

# 评估模型
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="评估模型"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取正类的概率
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # 计算PR曲线和AUC
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集精确率: {precision:.4f}")
    print(f"测试集召回率: {recall:.4f}")
    print(f"测试集F1分数: {f1:.4f}")
    print(f"测试集AUC: {roc_auc:.4f}")
    print(f"测试集PR AUC: {pr_auc:.4f}")
    
    # 保存评估结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非VF', 'VF'], 
                yticklabels=['非VF', 'VF'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig(os.path.join(PATH_CONFIG['plot_dir'], 'confusion_matrix.png'))
    plt.close()
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PATH_CONFIG['plot_dir'], 'roc_curve.png'))
    plt.close()
    
    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR曲线 (AUC = {pr_auc:.4f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(PATH_CONFIG['plot_dir'], 'pr_curve.png'))
    plt.close()
    
    return accuracy, precision, recall, f1, cm

# 绘制训练过程
def plot_training_process(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练和验证损失')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('训练和验证准确率')
    
    plt.tight_layout()
    plt.savefig('/Users/xingyulu/Public/physionet/training_process.png')
    plt.close()

# 保存结果到CSV
def save_results(results, filename=None):
    if filename is None:
        filename = PATH_CONFIG['results_file']
    df = pd.DataFrame([results])
    df.to_csv(filename, index=False)
    print(f"结果已保存到 {filename}")

# 主函数
def main():
    # 设置设备
    device = MODEL_CONFIG['device']
    print(f"使用设备: {device}")
    
    # 创建必要的目录
    os.makedirs(PATH_CONFIG['model_save_dir'], exist_ok=True)
    os.makedirs(PATH_CONFIG['plot_dir'], exist_ok=True)
    
    # 加载数据
    print("加载和筛选数据...")
    all_data, all_labels, max_length = load_data()
    
    # 数据分割
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        all_data, all_labels, test_size=DATA_CONFIG['test_size'], random_state=42, stratify=all_labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=DATA_CONFIG['val_size'], random_state=42, stratify=y_train_val
    )
    
    # 打印数据形状以便调试
    print(f"训练数据形状: {X_train.shape}")
    
    # 修改数据处理方式，确保维度正确
    # 将数据从(N, 2500, 1)转换为(N, 1, 2500)，适合conv1d操作
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    print(f"处理后的训练数据形状: {X_train.shape}")
    
    # 创建数据集和数据加载器 - 不再需要unsqueeze(1)
    train_dataset = ECGDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = ECGDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = ECGDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    train_loader = TorchDataLoader(train_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=True, num_workers=DATA_CONFIG['num_workers'])
    val_loader = TorchDataLoader(val_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False, num_workers=DATA_CONFIG['num_workers'])
    test_loader = TorchDataLoader(test_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False, num_workers=DATA_CONFIG['num_workers'])
    
    # 初始化模型
    model = ECGClassifier(input_size=max_length).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=MODEL_CONFIG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练模型
    print("开始训练模型...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )
    
    # 绘制训练过程
    plot_training_process(train_losses, val_losses, train_accs, val_accs)
    
    # 在测试集上评估模型
    print("在测试集上评估模型...")
    accuracy, precision, recall, f1, cm = evaluate_model(model, test_loader, device)
    
    # 保存结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc
    }
    save_results(results)
    
    # 保存模型
    model_save_path = os.path.join(PATH_CONFIG['model_save_dir'], 'best_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存到: {model_save_path}")

if __name__ == "__main__":
    main()