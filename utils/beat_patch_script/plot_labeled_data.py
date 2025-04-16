import numpy as np
import matplotlib.pyplot as plt
import os

def plot_labeled_data(data_path, save_path=None):
    """
    可视化带标签的心拍数据(优化内存版本)
    """
    try:
        # 使用with语句确保文件正确关闭
        labeled_data = np.load(data_path)
        beat_data = labeled_data[:, 0].copy()  # 创建副本避免内存泄漏
        labels = labeled_data[:, 1].copy()
        del labeled_data  # 及时释放内存
        
        plt.figure(figsize=(15, 5))
        
        # 定义5种心拍类别的颜色和标签(与plot_beat_sequence保持一致)
        type_colors = {
            0: 'green',    # 正常心拍
            1: 'blue',     # S
            2: 'green',    # V
            3: 'purple',   # X
            4: 'red',      # AF
        }
        
        type_labels = {
            0: 'N',
            1: 'S',
            2: 'V',
            3: 'X',
            4: 'AF'
        }
        
        # 根据标签变化点分割数据
        change_points = np.where(np.diff(labels) != 0)[0] + 1
        segments = np.split(beat_data, change_points)
        label_segments = np.split(labels, change_points)
        
        current_position = 0
        legend_handles = []
        
        for seg, lbl_seg in zip(segments, label_segments):
            if len(seg) == 0:
                continue
                
            b_type = int(lbl_seg[0])  # 获取当前段的标签类型
            color = type_colors.get(b_type, 'gray')
            x_values = np.arange(current_position, current_position + len(seg))
            line = plt.plot(x_values, seg, color=color)
            
            # 只为每种类型添加一次图例
            if b_type not in [h.get_label() for h in legend_handles]:
                line[0].set_label(type_labels[b_type])
                legend_handles.append(line[0])
                
            current_position += len(seg)
        
        plt.xlabel('Sample Points')
        plt.ylabel('Amplitude')
        plt.title('ECG Beat Signal Sequence with Labels')
        plt.legend(handles=legend_handles)
        plt.grid(True)
        
        # 保存图片后立即关闭图形
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = os.path.join(save_path, os.path.basename(data_path).replace('.npy', '.png'))
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图片已保存至: {filename}")
        
        plt.close('all')  # 关闭所有图形释放内存
        del beat_data, labels  # 释放数组内存
        
    except Exception as e:
        print(f"处理文件 {data_path} 时出错: {str(e)}")
        plt.close('all')

if __name__ == "__main__":
    # 设置路径
    data_dir = '/Users/xingyulu/Public/afafaf/Holter_Data_3例/npy'
    picture_dir = '/Users/xingyulu/Public/afafaf/Holter_Data_3例/picture'
    
    # 确保picture目录存在
    if not os.path.exists(picture_dir):
        os.makedirs(picture_dir)
    
    # 分批处理文件(每次处理100个)
    batch_size = 100
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]
        print(f"正在处理第 {i//batch_size + 1} 批文件(共 {len(batch_files)} 个)...")
        
        for npy_file in batch_files:
            file_path = os.path.join(data_dir, npy_file)
            plot_labeled_data(file_path, save_path=picture_dir)
            
        # 手动调用垃圾回收
        import gc
        gc.collect()
    
    print(f"已完成所有文件的处理，图片保存在: {picture_dir}")