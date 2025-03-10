import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns

class SignalPlotter:
    def __init__(self, config):
        """初始化信号绘图器"""
        self.config = config
        self.setup_directories()
        self.setup_plot_style()
        
    def setup_directories(self):
        """设置输入输出目录"""
        self.input_dir = Path(self.config['input_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(exist_ok=True)
        
    def setup_plot_style(self):
        """设置绘图样式"""
        sns.set_style(self.config['plot_style'])
        self.figure_size = self.config['figure_size']
        self.dpi = self.config['dpi']
        
    def process_window(self, signal, window_idx, total_windows, filename, fs):
        """处理单个时间窗口的数据"""
        window_size = self.config['window_seconds'] * fs
        start_idx = window_idx * window_size
        end_idx = min((window_idx + 1) * window_size, signal.shape[0])
        
        plt.figure(figsize=self.figure_size)
        
        # 绘制每个导联的数据
        for lead_idx in range(signal.shape[1]):
            window_data = signal[start_idx:end_idx, lead_idx]
            time_axis = np.arange(len(window_data)) / fs
            plt.plot(time_axis, window_data, label=f'Lead {lead_idx+1}')
        
        self.set_plot_attributes(filename, window_idx+1, total_windows)
        self.save_plot(filename, window_idx+1)
        
    def set_plot_attributes(self, filename, window_num, total_windows):
        """设置图形的标题和标签"""
        plt.title(f'Signal: {filename} (Window {window_num}/{total_windows})')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.xlim(0, self.config['window_seconds'])
        plt.legend()
        plt.grid(True)
        
    def save_plot(self, filename, window_num):
        """保存图形"""
        output_path = self.output_dir / f'{filename}_window{window_num}.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def process_file(self, npy_file):
        """处理单个文件"""
        try:
            signal = np.load(npy_file)
            fs = self.config['sampling_rate']
            window_size = self.config['window_seconds'] * fs
            total_windows = int(np.ceil(signal.shape[0] / window_size))
            
            for window in range(total_windows):
                self.process_window(signal, window, total_windows, npy_file.stem, fs)
                
            print(f"Successfully plotted {npy_file.name}")
            
        except Exception as e:
            print(f"Error processing {npy_file.name}: {str(e)}")
            plt.close()
            
    def run(self):
        """运行绘图程序"""
        npy_files = list(self.input_dir.glob('*.npy'))
        for npy_file in npy_files:
            self.process_file(npy_file)

def main():
    """主程序入口"""
    config = {
        'input_dir': '/Users/xingyulu/Downloads/心电预警/公开数据/vf_segments',
        'output_dir': '/Users/xingyulu/Downloads/心电预警/公开数据/picture',
        'sampling_rate': 250,
        'window_seconds': 10,
        'plot_style': 'darkgrid',
        'figure_size': (15, 5),
        'dpi': 300
    }
    
    plotter = SignalPlotter(config)
    plotter.run()

if __name__ == "__main__":
    main()