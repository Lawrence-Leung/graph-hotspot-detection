import numpy as np

# 填充 list 长度到指定长度，所填充的值为 0
def pad_list(lst, target_length, pad_value=0):
    lst.extend([pad_value] * (target_length - len(lst)))  # 仅当列表长度不足时补充
    return lst

# 将一个一维度数据分布，转换为傅里叶变换。
# data：原始的数据 list
# num_features：需要的维度值（不能是奇数）
def extract_frequency_features(data: list, num_features = 4, a=1200):
    if num_features % 2 == 1:
        raise ValueError("Oops! num_features is not even number.")

    data_temp = data
    if len(data) < num_features:
        data_temp = pad_list(data_temp, num_features)
    data_temp = sorted(data_temp)   # 按递增次序排序

    print(data_temp)
    # 计算 DFT（傅里叶变换）
    fft_result = np.fft.fft(data_temp)

    # 计算幅度谱（忽略复数部分）
    magnitudes = np.abs(fft_result)[:num_features]

    return np.clip(magnitudes, 0, a) / (a / 5)

# 计算 2D 坐标点的傅里叶变换特征。
def extract_2d_frequency_features(points, num_features=4, a=1200):
    """
    :param points: 二维点列表 [(x_1, y_1), ..., (x_n, y_n)]
    :param num_features: 目标维度（必须是偶数）
    :param a: 限制最大特征值
    :return: (num_features,) 大小的 list
    """
    if num_features % 2 == 1:
        raise ValueError("Oops! num_features is not even number.")
    # 分离 x 和 y 坐标
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    # 确保每个维度的长度足够，否则填充
    half_features = num_features // 2
    if len(x_coords) < half_features:
        x_coords = pad_list(x_coords, half_features)
    if len(y_coords) < half_features:
        y_coords = pad_list(y_coords, half_features)
    # 排序（可以确保频率变化平稳）
    x_coords = sorted(x_coords)
    y_coords = sorted(y_coords)
    # 计算 DFT
    fft_x = np.fft.fft(x_coords)
    fft_y = np.fft.fft(y_coords)
    # 取前 half_features 个幅度特征
    mag_x = np.abs(fft_x)[:half_features]
    mag_y = np.abs(fft_y)[:half_features]
    # 合并 x 和 y 方向的特征，保证总大小是 num_features
    features = np.concatenate([mag_x, mag_y])
    return np.clip(features, 0, a) / (a / 5)  # 限制数值范围

if __name__ == "__main__":
    # 示例数据
    data = [1, 3, 2, 4, 5, 2, 1, 4, 6, 2, 1, 1]
    features = extract_frequency_features(data, 4)
    print(features)  # 输出 3 个数值作为特征

    points = [(1, 2), (3, 4), (1, 2), (3, 4)]
    features = extract_2d_frequency_features(points, num_features=4)
    print(features)  # 输出 (4,) 形状的 list