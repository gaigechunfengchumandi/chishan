import numpy as np

def extract_second_column_to_npy(file_path, output_path):
    """
    读取txt文件并提取第二列数据保存为npy文件
    
    参数:
        file_path (str): 要读取的txt文件路径
        output_path (str): 输出的npy文件路径
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            # 跳过前两行
            data_lines = lines[2:] if len(lines) > 2 else []
            
            # 提取第二列数据
            second_column = []
            for line in data_lines:
                columns = line.strip().split()
                if len(columns) >= 2:
                    try:
                        point_value = int(columns[1])
                        second_column.append(point_value)
                    except ValueError:
                        continue
            
            # 保存为npy文件
            np.save(output_path, np.array(second_column))
            print(f"成功保存第二列数据到: {output_path}")
            
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    except Exception as e:
        print(f"处理文件时出错: {e}")

# 示例用法
if __name__ == "__main__":
    input_file = "/Users/xingyulu/Public/afafaf/室性心搏/12导联24小时数据/劳键潼-20220301-160817_raw_0_11059200.txt"
    output_file = "/Users/xingyulu/Public/afafaf/室性心搏/II导联24小时信号数据/data/劳键潼-20220301-160817_raw_0_11059200.npy"
    extract_second_column_to_npy(input_file, output_file)