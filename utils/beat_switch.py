import numpy as np
import random

def beat_switch(data):
    """
    将1280行数据分成10份，随机交换其中两份的位置

    Args:
        data: shape=(1280,2)的numpy数组
    Returns:
        交换后的数组
    """
    # 确保输入数据形状正确
    assert data.shape[0] == 1280, "数据行数必须为1280"

    # 计算每份的大小
    segment_size = 128  # 1280/10 = 128

    # 分割数据
    segments = []
    for i in range(10):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size
        segments.append(data[start_idx:end_idx])

    # 随机选择两个不同的段进行交换
    i, j = random.sample(range(10), 2)
    segments[i], segments[j] = segments[j], segments[i]

    # 重新组合数据
    result = np.concatenate(segments, axis=0)

    return result



if __name__ == "__main__":
    # 生成示例数据
    data = np.arange(1280 * 2).reshape(1280, 2)
    print("原始数据:")
    print(data[:5])

    # 调用函数
    switched = beat_switch(data)

    print("交换后数据:")
    print(switched[:5])



