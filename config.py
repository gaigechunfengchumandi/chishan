import torch

# 文件路径配置
PATH_CONFIG = {
    'non_vf_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/data_proccess/window_data/non_vf_windows',
    'vf_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/data_proccess/window_data/vf_windows',
    'model_save_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/models',
    'plot_dir': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/plots',
    'results_file': '/Users/xingyulu/Public/监护心电预警/公开数据/室颤/results.csv'
}

# 数据处理配置
DATA_CONFIG = {
    'min_size': 20 * 1024,  # 最小文件大小（20KB）
    'data_augmentation': True,  # 是否进行数据增强
    'max_sequence_length': 2500,  # 最大序列长度
    'test_size': 0.2,  # 测试集比例
    'val_size': 0.2,  # 验证集比例
    'batch_size': 32,  # 批次大小
    'num_workers': 4  # 数据加载线程数
}

# 模型配置
MODEL_CONFIG = {
    'input_size': 5000,  # 输入大小
    'learning_rate': 0.001,  # 学习率
    'weight_decay': 1e-5,  # 权重衰减
    'num_epochs': 50,  # 训练轮数
    'patience': 10,  # 早停耐心值
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 训练设备
}

# 数据增强配置
AUGMENTATION_CONFIG = {
    'noise_std': 0.01,  # 噪声标准差
    'shift_range': (10, 50),  # 时间偏移范围
    'scale_range': (0.9, 1.1)  # 振幅缩放范围
}