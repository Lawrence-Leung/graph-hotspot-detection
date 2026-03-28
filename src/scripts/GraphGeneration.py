# 图数据生成。
# GraphGeneration.py
# by Lawrence Leung 2025.3.12

# 引入库
import numpy as np
import hashlib, os
import torch
from torch_geometric.data import Data, Dataset, Batch
import glob
import random
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.loader import DataLoader

import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt

from ClipPartition import ClipPartition

# 辅助函数，将文件名（不带后缀）编码为哈希值。函数可用。
def filename_to_hash(filepath, algorithm="sha256"):
    '''
    :param filepath: 文件名
    :param algorithm: 算法，可以选择"sha256"等。
    :return: 哈希值的十六进制编码
    '''
    # 获取不带后缀的文件名
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # 选择哈希算法
    hash_func = hashlib.new(algorithm)
    hash_func.update(filename.encode('utf-8'))
    # 返回哈希值的十六进制编码
    return hash_func.hexdigest()

# 辅助函数，提取文件名，然后通过文件名判断是否为光刻热点，放到这个 clip 的图的 y(label) 中去。
def generate_graph_label(filepath):
    # 获取不带后缀的文件名
    filename = os.path.splitext(os.path.basename(filepath))[0]
    return 1 if filename.startswith("HS") else 0

# 辅助函数，填充 list 长度到指定长度，所填充的值为 0。函数可用。
def pad_list(lst, target_length, pad_value=0):
    lst.extend([pad_value] * (target_length - len(lst)))  # 仅当列表长度不足时补充
    return lst

# 辅助函数，将一个一维度数据分布，转换为傅里叶变换。
# data：原始的数据 list
# num_features：需要的维度值（不能是奇数）
def extract_frequency_features(data: list, num_features = 4, a=1200):
    if num_features % 2 == 1:
        raise ValueError("Oops! num_features is not even number.")

    data_temp = data
    if len(data) < num_features:
        data_temp = pad_list(data_temp, num_features)
    data_temp = sorted(data_temp)   # 按递增次序排序

    # 计算 DFT（傅里叶变换）
    fft_result = np.fft.fft(data_temp)

    # 计算幅度谱（忽略复数部分）
    magnitudes = np.abs(fft_result)[:num_features]

    return np.clip(magnitudes, 5e-4, a)

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

    x_coords = [x / a for x in x_coords]
    y_coords = [y / a for y in y_coords]

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
    return np.clip(features, 5e-4, a) # 限制数值范围

# 生成图的类！================================================================================
class GraphGeneration:
    def __init__(self, save_dir: str = "./dataset"):
        # 临时存储每一个 ClipPartition 变量类
        # 以下是类内成员。"sc" == "single clip"。
        self.sc_hor_blocks = []
        self.sc_ver_blocks = []
        self.sc_poly_per_block = []
        self.sc_lg_cons = {}
        self.a = 0  # 图像的边长，归一化要除以它。
        # 以下是大图的分辨信息
        self.sc_clipid = "" # 分辨不同的 clip，用 hashlib 编码
        self.sc_blkid = (-100, -100)  # 分辨同一 clip 下的不同子图。一个 (2,) tuple，命名如下：
        # 小块图层次：(hor_blk_index(y), ver_blk_index)
        # 大块图层次：(-1, -1)
        # 这里暂且初始化为 (-100, -100) 这样的无效信息。
        self.sc_graph_y = -1   # 整个 clip 的标签信息，说明这个 clip 是热点与否。注意。这个是整数，不是 torch_geometric.Data 的格式！
        self.sc_ahp = []
        self.sc_avp = []
        self.save_dir = save_dir

        # 对所有的 25 个子图的临时变量
        # 用于临时存储一些信息
        # "idx" == "index"，意思是对于每个大块图内而言的操作。
        self.idx_table_perblock = {}
        self.idx_graph_x = torch.tensor([[-1]], dtype=torch.float)   # 每个子图的 x 变量，用于 torch_geometric.Data 中。
        self.idx_graph_edgeindex = torch.tensor([[0], [0]], dtype=torch.long)   # 边的连接关系。注意类型为 torch.long。
        self.idx_graph_edgeattr = torch.tensor([[-1]], dtype=torch.int) # 边的特征信息。
        self.idx_graph_pos = torch.tensor([[-1]], dtype=torch.float) # 每个子图的 node 的位置信息。

        # 大图相关的变量，在函数 `generate_large_graph` 中呈现。
        self.output_graphlist = []  # 为了提升IO效率。IO太恶心了！

    # 加载一个光刻 clip 的原始信息。注意需要保证这些原始信息是完整的。
    def load_single_clip(self, clip: ClipPartition):
        # 导入信息
        self.sc_hor_blocks = clip.hor_blocks
        self.sc_ver_blocks = clip.ver_blocks
        self.sc_poly_per_block = clip.poly_per_block
        self.sc_lg_cons = clip.largegraph_connections
        self.a = clip.a
        self.sc_ahp = clip.all_hor_percentiles
        self.sc_avp = clip.all_ver_percentiles
        self.idx_table_perblock = {}    # 清空！
        # 对于每个clip图像文件，作为每个clip的唯一区分依据，这里编码为哈希值
        self.sc_clipid = filename_to_hash(clip.image_path)
        self.sc_graph_y = generate_graph_label(clip.image_path) # 是 hotspot 还是 non_hotspot
        self.output_graphlist = []

    # 下面仅做对每个子图的操作。分为三个步骤：制定索引表、填充信息、建立连接。
    # 对于每个子图，指定索引表。
    def build_block_index_table(self, hor_idx: int, ver_idx: int):
        """
        为 (hor_idx, ver_idx) 处的 block 生成索引表。
        索引表用于将 block 级别的 node 索引号，与每个 box (gap_index, box_index) 和 polygon 索引相关联。

        参数:
        hor_idx : int - 水平 block 索引
        ver_idx : int - 垂直 block 索引

        结果:
        self.idx_table_perblock[(hor_idx, ver_idx)] = {
            "hor_start": 0,   # 水平方向起始节点
            "ver_start": 1,   # 垂直方向起始节点
            "hor_boxes": { (gap_index, box_index): node_index, ... },
            "ver_boxes": { (gap_index, box_index): node_index, ... },
            "polygons": { polygon_index: node_index, ... }

        返回：node_numbers，就是整个图的所有节点个数。
        }
        """
        idx_table = {
            "hor_start": 0,  # 水平入口节点编号
            "ver_start": 1,  # 垂直入口节点编号
            "hor_boxes": {},  # 水平方向 box 对应的索引 (gap_idx, box_idx): node_idx
            "ver_boxes": {},  # 垂直方向 box 对应的索引 (gap_idx, box_idx): node_idx
            "polygons": {}  # polygon 对应的索引 polygon_idx: node_idx
        }

        # 初始索引号，从2开始（因为 0 和 1 被 horizontal 和 vertical 入口节点占用）
        current_node_index = 2

        # 处理 horizontal block 的 box (gap_index, box_index)
        hor_block_list = self.sc_hor_blocks[hor_idx][ver_idx]
        for gap_idx, box_idx_dict in enumerate(hor_block_list):
            for box_idx in range(len(box_idx_dict)):
                idx_table["hor_boxes"][(gap_idx, box_idx)] = current_node_index
                current_node_index += 1

                # 处理 vertical block 的 box (gap_index, box_index)
        ver_block_list = self.sc_ver_blocks[hor_idx][ver_idx]
        for gap_idx, box_idx_dict in enumerate(ver_block_list):
            for box_idx in range(len(box_idx_dict)):
                idx_table["ver_boxes"][(gap_idx, box_idx)] = current_node_index
                current_node_index += 1

                # 处理 polygon 索引
        poly_list = self.sc_poly_per_block[hor_idx][ver_idx]
        for poly_idx in range(len(poly_list)):
            idx_table["polygons"][poly_idx] = current_node_index
            current_node_index += 1

            # 存储索引表
        self.idx_table_perblock[(hor_idx, ver_idx)] = idx_table
        return current_node_index

    # 对于每个子图，填充节点 feature 信息、位置信息
    def build_node_feat_and_pos(self, hor_idx: int, ver_idx: int, node_numbers = 0):
        if node_numbers == 0:
            raise ValueError("Oops! Node Numbers == 0!")

        # 1. 初始化 idx_graph_x (node_numbers, 10) 和 idx_graph_pos (node_numbers, 3)
        self.idx_graph_x = torch.zeros((node_numbers, 10), dtype=torch.float)  # 存储10个特征
        self.idx_graph_pos = torch.zeros((node_numbers, 3), dtype=torch.float)  # 存储位置信息 (x, y, z)

        # 首先处理 "hot_boxes"、"ver_boxes"：------------------------------------------------------------
        # 对于所有的 "hor_boxes"、"ver_boxes"，满足这些条件。
        # 这10个特征为：[0] 类型，白=1、黑=0；[1] 长；[2] 宽；[3] 重叠边界长度最小值；[4] 重叠边界长度最大值；
        # [5] 重叠边界数量；[6][7][8][9]：重叠边界DFT分量。
        # 位置：
        # 对于 "hor_boxes"：(box的中心x坐标, box的中心y坐标, 1.0)
        # 对于 "ver_boxes"：(box的中心x坐标, box的中心y坐标, 0.0)
        # 2. 获取 hor_boxes 和 ver_boxes 的索引映射
        idx_table = self.idx_table_perblock[(hor_idx, ver_idx)]
        hor_boxes = idx_table["hor_boxes"]
        ver_boxes = idx_table["ver_boxes"]

        # 3. 处理 hor_boxes（水平 box）
        for (gap_idx, box_idx), node_index in hor_boxes.items():
            box_data = self.sc_hor_blocks[hor_idx][ver_idx][gap_idx][box_idx]  # 取出 box 的信息
            # a. 计算坐标信息
            x = (box_data["right"] + box_data["left"])
            y = (box_data["bottom"] + box_data["top"])
            z = 1.0  # 水平方向 box z = 0.5
            self.idx_graph_pos[node_index] = torch.tensor([x, y, z], dtype=torch.float)
            # b. 计算特征信息
            adj_lengths = [adj["adj_length"] for adj in box_data["adj"]] if box_data["adj"] else [0]
            if box_data["color"] == 0:  # 颜色
                self.idx_graph_x[node_index, 0] = 5e-4
            else:
                self.idx_graph_x[node_index, 0] = 1 - (5e-4)
            self.idx_graph_x[node_index, 1] = (box_data["right"] - box_data["left"])  # 长度
            self.idx_graph_x[node_index, 2] = (box_data["bottom"] - box_data["top"])  # 宽度
            self.idx_graph_x[node_index, 3] = min(adj_lengths)  # 最小重叠边界
            self.idx_graph_x[node_index, 4] = max(adj_lengths)   # 最大重叠边界
            self.idx_graph_x[node_index, 5] = len(adj_lengths) / ((box_data["right"] - box_data["left"]) + (box_data["bottom"] - box_data["top"])) # 重叠边界数量
            # 计算频率特征 [6] ~ [9]
            freq_features = extract_frequency_features(adj_lengths, 4, self.a)
            self.idx_graph_x[node_index, 6:10] = torch.tensor(freq_features, dtype=torch.float)

        # 4. 处理 ver_boxes（垂直 box）
        for (gap_idx, box_idx), node_index in ver_boxes.items():
            box_data = self.sc_ver_blocks[hor_idx][ver_idx][gap_idx][box_idx]  # 取出 box 的信息
            # a. 计算坐标信息
            x = (box_data["right"] + box_data["left"])
            y = (box_data["bottom"] + box_data["top"])
            z = 0.0  # 垂直方向 box z = 0.0
            self.idx_graph_pos[node_index] = torch.tensor([x, y, z], dtype=torch.float)
            # b. 计算特征信息
            adj_lengths = [adj["adj_length"] for adj in box_data["adj"]] if box_data["adj"] else [0]
            if box_data["color"] == 0:  # 颜色
                self.idx_graph_x[node_index, 0] = 5e-4
            else:
                self.idx_graph_x[node_index, 0] = 1 - (5e-4)
            self.idx_graph_x[node_index, 1] = (box_data["right"] - box_data["left"])  # 长度
            self.idx_graph_x[node_index, 2] = (box_data["bottom"] - box_data["top"])  # 宽度
            self.idx_graph_x[node_index, 3] = min(adj_lengths) # 最小重叠边界
            self.idx_graph_x[node_index, 4] = max(adj_lengths) # 最大重叠边界
            self.idx_graph_x[node_index, 5] = len(adj_lengths) / ((box_data["right"] - box_data["left"]) + (box_data["bottom"] - box_data["top"])) # 重叠边界数量
            # 计算频率特征 [6] ~ [9]
            freq_features = extract_frequency_features(adj_lengths, 4, self.a)
            self.idx_graph_x[node_index, 6:10] = torch.tensor(freq_features, dtype=torch.float)

        # 接着处理 "polygons"：-------------------------------------------------------------------------
        # 这10个特征为：[0] 类型，固定为 0.5；[1] 面积；[2][3][4]：转角频率分布；[5] 转角数量；
        # [6][7] 水平边界的DFT分量；[8][9] 垂直边界的 DFT 分量。
        # 位置：(所有转角的x轴平均值；所有转角的y轴平均值，0.5)
        polygons = idx_table["polygons"]

        # 处理 polygon 节点
        for polygon_index, node_index in polygons.items():
            polygon_data = self.sc_poly_per_block[hor_idx][ver_idx][polygon_index]  # 取出 polygon 信息

            # a. 计算坐标信息
            vertices = polygon_data["vertices"]  # [(vert_x, vert_y), ...]
            x = sum(v[0] for v in vertices) / len(vertices) / self.a  # x 坐标均值归一化
            y = sum(v[1] for v in vertices) / len(vertices) / self.a  # y 坐标均值归一化
            z = 0.5  # z 恒定为 0.5
            self.idx_graph_pos[node_index] = torch.tensor([x, y, z], dtype=torch.float)

            # b. 计算特征信息
            self.idx_graph_x[node_index, 0] = 0.5  # 类型，polygon 固定为 0.5
            self.idx_graph_x[node_index, 1] = polygon_data["area"] / (self.a * self.a) # 直接取面积的归一化结果

            # 计算 vertices 的 2D 频率特征 [2] ~ [5]
            vertices_freq_features = extract_2d_frequency_features(vertices, 4, self.a)
            self.idx_graph_x[node_index, 2:6] = torch.tensor(vertices_freq_features, dtype=torch.float)

            # 计算水平边界长度分布特征 [6] ~ [7]
            hor_edges = polygon_data["hor_edges"]
            hor_freq_features = extract_frequency_features(hor_edges, 2, self.a)
            self.idx_graph_x[node_index, 6:8] = torch.tensor(hor_freq_features, dtype=torch.float)

            # 计算垂直边界长度分布特征 [8] ~ [9]
            ver_edges = polygon_data["ver_edges"]
            ver_freq_features = extract_frequency_features(ver_edges, 2, self.a)
            self.idx_graph_x[node_index, 8:10] = torch.tensor(ver_freq_features, dtype=torch.float)

        # 最后再填充水平、垂直方向的初始节点。
        for i in range(0, 10):
            self.idx_graph_x[0, i] = 0.5
            self.idx_graph_x[1, i] = 0.5
        self.idx_graph_pos[0, 2] = 1.0  # 水平入口节点 z 坐标
        self.idx_graph_pos[1, 2] = 0.0  # 垂直入口节点 z 坐标
        self.idx_graph_pos[0, 0] = (self.sc_ahp[ver_idx] + self.sc_ahp[ver_idx + 1]) / 2 # 水平入口节点，从上到下
        self.idx_graph_pos[0, 1] = self.sc_avp[hor_idx]
        self.idx_graph_pos[1, 0] = self.sc_ahp[ver_idx]    # 垂直入口节点，从左到右
        self.idx_graph_pos[1, 1] = (self.sc_avp[hor_idx] + self.sc_avp[hor_idx + 1]) / 2

        # 对 [1] ~ [9] 维度的特征进行 z-score 归一化
        features_to_normalize = self.idx_graph_x[:, 1:10]  # 选取 [1] ~ [9] 列
        mean = torch.mean(features_to_normalize, dim=0, keepdim=True)  # 计算均值
        std = torch.std(features_to_normalize, dim=0, keepdim=True)  # 计算标准差
        # 避免除零错误，标准差小于某个极小值时设为 1
        std[std < 1e-6] = 1.0
        # 应用 z-score 归一化
        self.idx_graph_x[:, 1:10] = (features_to_normalize - mean) / std

    # 对于每个子图，建立边的连接
    def build_edges(self, hor_idx:int, ver_idx:int, node_numbers=0):
        if node_numbers == 0:
            raise ValueError("Oops! Node Numbers == 0!")

        # 初始化 edge_index 和 edge_attr
        self.idx_graph_edgeindex = torch.empty((2, 0), dtype=torch.long)  # (2, num_edges)
        self.idx_graph_edgeattr = torch.empty((0,), dtype=torch.int)  # (num_edges,)
        # 注意，从上到下的边，edge_attr = 1；从左到右的边，edge_attr = 2；涉及到 polygon 的边，edge_attr = 3。

        # 获取 hor_boxes 的索引映射
        idx_table = self.idx_table_perblock[(hor_idx, ver_idx)]

        # 从上到下 -------------------------------------------------------------------------------
        hor_boxes = idx_table["hor_boxes"]
        # 1. 从水平入口节点（node_index=0）连接到所有 `gap_idx == 0` 的 box
        for (gap_idx, box_idx), node_index in hor_boxes.items():
            if gap_idx == 0:
                self.idx_graph_edgeindex = torch.cat(
                    [self.idx_graph_edgeindex, torch.tensor([[0], [node_index]], dtype=torch.long)], dim=1
                )
                self.idx_graph_edgeattr = torch.cat(
                    [self.idx_graph_edgeattr, torch.tensor([1], dtype=torch.int)], dim=0
                )
        # 2. 每个 box 连接到 `gap_idx+1` 的相邻 box
        connected_nodes = set()  # 记录已连接的 box
        for (gap_idx, box_idx), node_index in hor_boxes.items():
            box_data = self.sc_hor_blocks[hor_idx][ver_idx][gap_idx][box_idx]
            for adj in box_data["adj"]:
                neighbor_gap = adj["neighbor_gap"]
                neighbor_box = adj["neighbor_box"]
                if neighbor_gap == gap_idx + 1 and (neighbor_gap, neighbor_box) in hor_boxes:
                    neighbor_node_index = hor_boxes[(neighbor_gap, neighbor_box)]
                    self.idx_graph_edgeindex = torch.cat(
                        [self.idx_graph_edgeindex,
                         torch.tensor([[node_index], [neighbor_node_index]], dtype=torch.long)], dim=1
                    )
                    self.idx_graph_edgeattr = torch.cat(
                        [self.idx_graph_edgeattr, torch.tensor([1], dtype=torch.int)], dim=0
                    )
                    connected_nodes.add(node_index)  # 记录已连接的 box
                    connected_nodes.add(neighbor_node_index)
        # 3. 处理孤立 box，连接到水平入口节点。注意这个代码需要在第 2 步运行结束之后才能运行，否则会出错！
        for (gap_idx, box_idx), node_index in hor_boxes.items():
            if node_index not in connected_nodes:
                # print("Isolated!")
                self.idx_graph_edgeindex = torch.cat(
                    [self.idx_graph_edgeindex, torch.tensor([[0], [node_index]], dtype=torch.long)], dim=1
                )
                self.idx_graph_edgeattr = torch.cat(
                    [self.idx_graph_edgeattr, torch.tensor([1], dtype=torch.int)], dim=0
                )

        # 从左到右 -------------------------------------------------------------------------------
        # 获取 ver_boxes 的索引映射
        ver_boxes = idx_table["ver_boxes"]
        # 1. 从垂直入口节点（node_index=1）连接到所有 `gap_idx == 0` 的 box
        for (gap_idx, box_idx), node_index in ver_boxes.items():
            if gap_idx == 0:
                self.idx_graph_edgeindex = torch.cat(
                    [self.idx_graph_edgeindex, torch.tensor([[1], [node_index]], dtype=torch.long)], dim=1
                )
                self.idx_graph_edgeattr = torch.cat(
                    [self.idx_graph_edgeattr, torch.tensor([2], dtype=torch.int)], dim=0
                )
        # 2. 每个 box 连接到 `gap_idx + 1` 处的相邻 box
        connected_nodes = set()  # 记录已连接的 box
        for (gap_idx, box_idx), node_index in ver_boxes.items():
            box_data = self.sc_ver_blocks[hor_idx][ver_idx][gap_idx][box_idx]
            for adj in box_data["adj"]:
                neighbor_gap = adj["neighbor_gap"]
                neighbor_box = adj["neighbor_box"]
                if neighbor_gap == gap_idx + 1 and (neighbor_gap, neighbor_box) in ver_boxes:
                    neighbor_node_index = ver_boxes[(neighbor_gap, neighbor_box)]
                    self.idx_graph_edgeindex = torch.cat(
                        [self.idx_graph_edgeindex, torch.tensor([[node_index], [neighbor_node_index]], dtype=torch.long)],
                        dim=1
                    )
                    self.idx_graph_edgeattr = torch.cat(
                        [self.idx_graph_edgeattr, torch.tensor([2], dtype=torch.int)], dim=0
                    )
                    connected_nodes.add(node_index)
                    connected_nodes.add(neighbor_node_index)
        # 3. 处理孤立 box，连接到垂直入口节点（node_index=1）。注意这个代码需要在第 2 步运行结束之后才能运行，否则会出错！
        for (gap_idx, box_idx), node_index in ver_boxes.items():
            if node_index not in connected_nodes:
                self.idx_graph_edgeindex = torch.cat(
                    [self.idx_graph_edgeindex, torch.tensor([[1], [node_index]], dtype=torch.long)], dim=1
                )
                self.idx_graph_edgeattr = torch.cat(
                    [self.idx_graph_edgeattr, torch.tensor([2], dtype=torch.int)], dim=0
                )

        # 对于 polygons --------------------------------------------------------------------------
        # 获取 polygons 的索引映射
        polygons = idx_table["polygons"]
        # 遍历所有 polygon
        for polygon_index, polygon_node_index in polygons.items():
            polygon_data = self.sc_poly_per_block[hor_idx][ver_idx][polygon_index]  # 取出 polygon 信息
            # 连接 hor_boxes 到 polygon
            for (gap_idx, box_idx) in polygon_data["hor_boxes"]:
                if (gap_idx, box_idx) in hor_boxes:
                    box_node_index = hor_boxes[(gap_idx, box_idx)]
                    self.idx_graph_edgeindex = torch.cat(
                        [self.idx_graph_edgeindex,
                         torch.tensor([[box_node_index], [polygon_node_index]], dtype=torch.long)], dim=1
                    )
                    self.idx_graph_edgeattr = torch.cat(
                        [self.idx_graph_edgeattr, torch.tensor([3], dtype=torch.int)], dim=0
                    )
            # 连接 ver_boxes 到 polygon
            for (gap_idx, box_idx) in polygon_data["ver_boxes"]:
                if (gap_idx, box_idx) in ver_boxes:
                    box_node_index = ver_boxes[(gap_idx, box_idx)]
                    self.idx_graph_edgeindex = torch.cat(
                        [self.idx_graph_edgeindex,
                         torch.tensor([[box_node_index], [polygon_node_index]], dtype=torch.long)], dim=1
                    )
                    self.idx_graph_edgeattr = torch.cat(
                        [self.idx_graph_edgeattr, torch.tensor([3], dtype=torch.int)], dim=0
                    )
        # 完成！

    # 将构建的子图图数据封装为 `torch_geometric.Data`，用于 PyG 训练。
    def build_torch_geometric_data_for_block(self, hor_idx: int, ver_idx: int):
        """
        将当前 (hor_idx, ver_idx) 处的 block 组织为 torch_geometric.Data，并保存到文件。

        参数：
        hor_idx : int - 水平 block 索引
        ver_idx : int - 垂直 block 索引
        save_dir : str - 存储目录，默认为 ./dataset
        """
        # 构建 PyG Data 对象
        data = Data(
            x=self.idx_graph_x,  # 节点特征
            edge_index=self.idx_graph_edgeindex,  # 边索引
            edge_attr=self.idx_graph_edgeattr,  # 边属性
            pos=self.idx_graph_pos,  # 位置特征
        )

        # 添加 clipid（唯一标识）
        data.clipid = self.sc_clipid

        # 添加 blockid（标识子图位置）
        data.blockid = (hor_idx, ver_idx)

        # 计算 y（标签），利用 self.sc_graph_y
        data.y = torch.tensor([self.sc_graph_y], dtype=torch.long)

        # 生成唯一文件名
        # filename = f"{self.sc_clipid}_blk_{hor_idx}_{ver_idx}.pt"

        # 存储数据
        self.output_graphlist.append(data)
        # torch.save(data, f"{self.save_dir}/{filename}")

        # print(f"Block ({hor_idx}, {ver_idx}) has saved to {self.save_dir}/{filename}")

    def generate_block_graph(self):
        """
        生成某个 block (hor_idx, ver_idx) 的图，并填充相关信息。
        这里一次对所有 5 * 5 个子图进行处理。然后再封装。
        """
        for hor_idx in range(5):
            for ver_idx in range(5):
                # 1. 构建索引表，完成！
                node_nums = self.build_block_index_table(hor_idx, ver_idx)
                # 2. 填充节点 feature 信息及位置信息，完成！
                self.build_node_feat_and_pos(hor_idx, ver_idx, node_nums)
                # 3. 建立边连接
                self.build_edges(hor_idx, ver_idx, node_nums)
                # 4. 将每个子图封装为 torch.geometric.Data。
                # 别忘了，每个子图有自己的 clip_id、blkid！
                self.build_torch_geometric_data_for_block(hor_idx, ver_idx) # 这样可以保证顺序！

    def generate_large_graph(self):
        """
        生成整个 clip 的大块图层次（large graph）。
        注意，需要保证每个 clip 对应的 self.sc_clipid 与 self.sc_graph_y 是保持一致的。
        """
        # 1. 填充 data.pos
        pos_list = []
        for i in range(5):
            for j in range(5):
                x = (self.sc_avp[j] + self.sc_avp[j + 1]) / (2 * self.a)
                y = (self.sc_ahp[i] + self.sc_ahp[i + 1]) / (2 * self.a)
                pos_list.append([x, y])
        large_graph_pos = torch.tensor(pos_list, dtype=torch.float)

        # 保存到成员变量中，供后续步骤使用
        self.large_graph_pos = large_graph_pos

        # 2. 填充 data.x
        large_graph_features = []

        for hor_idx in range(5):
            for ver_idx in range(5):
                # 检查该 block 是否有数据
                hor_block = self.sc_hor_blocks[hor_idx][ver_idx]
                ver_block = self.sc_ver_blocks[hor_idx][ver_idx]

                if hor_block is None or ver_block is None:
                    raise ValueError(f"Missing data at block ({hor_idx}, {ver_idx})")

                # ------- 统计 horizontal block -------
                hor_boxes_count = 0
                hor_coords = []
                for gap_idx, gap_list in enumerate(hor_block):
                    for box_idx, box in enumerate(gap_list):
                        hor_boxes_count += 1
                        x_coord = (box["right"] + box["left"]) / (2 * self.a)
                        y_coord = (box["bottom"] + box["top"]) / (2 * self.a)
                        hor_coords.append((x_coord, y_coord))
                hor_freq_features = extract_2d_frequency_features(hor_coords, 8, self.a) if hor_coords else [0.0] * 8

                # ------- 统计 vertical block -------
                ver_boxes_count = 0
                ver_coords = []
                for gap_idx, gap_list in enumerate(ver_block):
                    for box_idx, box in enumerate(gap_list):
                        ver_boxes_count += 1
                        x_coord = (box["right"] + box["left"]) / (2 * self.a)
                        y_coord = (box["bottom"] + box["top"]) / (2 * self.a)
                        ver_coords.append((x_coord, y_coord))
                ver_freq_features = extract_2d_frequency_features(ver_coords, 8, self.a) if ver_coords else [0.0] * 8

                # ------- 统计白色 box -------
                white_boxes_coords = []
                white_boxes_count = 0
                for block_list in [hor_block, ver_block]:
                    for gap_list in block_list:
                        for box in gap_list:
                            if box["color"] == 1:
                                white_boxes_count += 1
                                x_coord = (box["right"] + box["left"]) / (2 * self.a)
                                y_coord = (box["bottom"] + box["top"]) / (2 * self.a)
                                white_boxes_coords.append((x_coord, y_coord))
                white_freq_features = extract_2d_frequency_features(white_boxes_coords, 8, self.a) if white_boxes_coords else [0.0] * 8

                # ------- 统计黑色 box -------
                black_boxes_coords = []
                black_boxes_count = 0
                for block_list in [hor_block, ver_block]:
                    for gap_list in block_list:
                        for box in gap_list:
                            if box["color"] == 0:
                                black_boxes_count += 1
                                x_coord = (box["right"] + box["left"]) / (2 * self.a)
                                y_coord = (box["bottom"] + box["top"]) / (2 * self.a)
                                black_boxes_coords.append((x_coord, y_coord))
                black_freq_features = extract_2d_frequency_features(black_boxes_coords, 8, self.a) if black_boxes_coords else [0.0] * 8

                # ------- 拼接大图节点 feature -------
                feature_vector = [hor_boxes_count] + hor_freq_features + \
                                 [ver_boxes_count] + ver_freq_features + \
                                 [white_boxes_count] + white_freq_features + \
                                 [black_boxes_count] + black_freq_features

                large_graph_features.append(feature_vector)

        large_graph_features = np.array(large_graph_features)       # 各种归一化处理
        lgf_mean = large_graph_features.mean(axis = 0)
        lgf_std = large_graph_features.std(axis = 0)
        lgf_std[lgf_std == 0] = 1e-8  # 避免除以0
        large_graph_features = (large_graph_features - lgf_mean) / lgf_std
        # print(large_graph_features)

        self.large_graph_x = torch.tensor(large_graph_features, dtype=torch.float)

        # 3. 填充 data.edge_index 和 data.edge_attr
        edge_list = []
        edge_attr_list = []

        for hor_idx in range(5):
            for ver_idx in range(5):
                current_node = hor_idx * 5 + ver_idx  # 计算当前节点索引

                # 计算当前节点的宽度和高度
                node_width = (self.sc_avp[ver_idx + 1] - self.sc_avp[ver_idx])
                node_height = (self.sc_ahp[hor_idx + 1] - self.sc_ahp[hor_idx])

                # ------- 连接上方节点 -------
                if hor_idx > 0:
                    upper_node = (hor_idx - 1) * 5 + ver_idx  # 计算上方节点索引
                    upper_boxes = self.sc_hor_blocks[hor_idx][ver_idx][0]  # gap_index = 0 处的 box

                    total_width = sum(box["right"] - box["left"] for box in upper_boxes if box["color"] == 1)
                    edge_weight = total_width / node_width if node_width > 0 else 0

                    if edge_weight > 0:
                        edge_list.append([current_node, upper_node])
                        edge_attr_list.append(edge_weight)

                # ------- 连接下方节点 -------
                if hor_idx < 4:
                    lower_node = (hor_idx + 1) * 5 + ver_idx  # 计算下方节点索引
                    lower_boxes = self.sc_hor_blocks[hor_idx][ver_idx][-1]  # gap_index = -1 处的 box

                    total_width = sum(box["right"] - box["left"] for box in lower_boxes if box["color"] == 1)
                    edge_weight = total_width / node_width if node_width > 0 else 0

                    if edge_weight > 0:
                        edge_list.append([current_node, lower_node])
                        edge_attr_list.append(edge_weight)

                # ------- 连接左方节点 -------
                if ver_idx > 0:
                    left_node = hor_idx * 5 + (ver_idx - 1)  # 计算左方节点索引
                    left_boxes = self.sc_ver_blocks[hor_idx][ver_idx][0]  # gap_index = 0 处的 box

                    total_height = sum(box["bottom"] - box["top"] for box in left_boxes if box["color"] == 1)
                    edge_weight = total_height / node_height if node_height > 0 else 0

                    if edge_weight > 0:
                        edge_list.append([current_node, left_node])
                        edge_attr_list.append(edge_weight)

                # ------- 连接右方节点 -------
                if ver_idx < 4:
                    right_node = hor_idx * 5 + (ver_idx + 1)  # 计算右方节点索引
                    right_boxes = self.sc_ver_blocks[hor_idx][ver_idx][-1]  # gap_index = -1 处的 box

                    total_height = sum(box["bottom"] - box["top"] for box in right_boxes if box["color"] == 1)
                    edge_weight = total_height / node_height if node_height > 0 else 0

                    if edge_weight > 0:
                        edge_list.append([current_node, right_node])
                        edge_attr_list.append(edge_weight)

        # 对于低 50% 数据，删除掉。
        # 将 edge_attr_list 转为 numpy 数组
        edge_attr_array = np.array(edge_attr_list)
        # 找出 50% 分位数 (中位数)
        threshold = np.median(edge_attr_array)
        # 找到大于中位数的元素索引
        indices_to_keep = np.where(edge_attr_array > threshold)[0]
        # indices_to_remove = np.where(edge_attr_array <= threshold)[0]
        # 根据索引保留对应的元素
        edge_list = [edge_list[i] for i in indices_to_keep]
        edge_attr_list = [edge_attr_list[i] for i in indices_to_keep]

        # 转换为 torch 张量
        if edge_list:
            self.large_graph_edge_index = torch.tensor(edge_list, dtype=torch.long).T  # 转置为 (2, num_edges)
            self.large_graph_edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        else:
            # 如果没有边，创建空 tensor，避免后续操作报错
            self.large_graph_edge_index = torch.empty((2, 0), dtype=torch.long)
            self.large_graph_edge_attr = torch.empty((0,), dtype=torch.float)

        # 4. 封装为 torch_geometric.Data
        from torch_geometric.data import Data
        data = Data(
            x=self.large_graph_x,  # (25, 20)
            edge_index=self.large_graph_edge_index,  # (2, num_edges)
            edge_attr=self.large_graph_edge_attr,  # (num_edges,)
            pos=self.large_graph_pos  # (25, 2)
        )
        # 添加唯一标识 clipid
        data.clipid = self.sc_clipid
        # blockid 固定为 (-1, -1) 表示大块图
        data.blockid = (-1, -1)
        # 添加标签
        data.y = torch.tensor([self.sc_graph_y], dtype=torch.long)
        # 生成唯一文件名
        # filename = f"{self.sc_clipid}_blk_-1_-1.pt"
        # 保存
        # torch.save(data, f"{self.save_dir}/{filename}")
        self.output_graphlist.append(data)
        # print(f"Entire Clip has saved to {self.save_dir}/{filename}")

    # 一次全部运行
    def runall(self, clip: ClipPartition):
        self.load_single_clip(clip) # 加载单个 clip
        self.generate_block_graph() # 加载所有小图
        self.generate_large_graph() # 加载所有大图
        filename = f"{self.sc_clipid}_GphLst_{self.sc_graph_y}.pt"    # 生成唯一文件名
        torch.save(self.output_graphlist, f"{self.save_dir}/{filename}") # 保存
        # print(f"Graph Generator Completed: {self.sc_clipid}")

# 设计批量加载的类 ============================================================================
# 注意，这个类是 clip 级别处理使用的类！（已弃用）
class ClipGroupedDatasetOld(Dataset):
    # TODO: 使得 hotspot 的数量，上升到与 non-hotspot 的数量相同。
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.clipid_dict = defaultdict(str)

        # 扫描所有图文件
        all_files = glob.glob(os.path.join(root, "*.pt"))
        for file in all_files:
            basename = os.path.basename(file)
            clipid = basename.split("_GphLst_")[0]
            self.clipid_dict[clipid] = file
        self.clipids = list(self.clipid_dict.keys())    # 键值对，键：clipid(sha256码), 值：文件名

    def len(self):
        return len(self.clipids)

    def get(self, idx):
        clipid = self.clipids[idx]
        singlept = self.clipid_dict[clipid] # singlept 是一个 filename!
        data_list = torch.load(singlept, weights_only=False)
        return data_list

# 可用的版本：
class ClipGroupedDataset(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.clipid_dict = dict()
        self.pos_clipids = []
        self.neg_clipids = []
        random.seed(12345)

        # 扫描所有图文件
        all_files = glob.glob(os.path.join(root, "*.pt"))
        for file in all_files:
            basename = os.path.basename(file)
            clipid = basename.split("_GphLst")[0]
            y = int(basename.split("_GphLst_")[1].split(".")[0])
            self.clipid_dict[clipid] = file
            if y == 1:
                self.pos_clipids.append(clipid)
            else:
                self.neg_clipids.append(clipid)

        # 平衡样本
        diff = len(self.neg_clipids) - len(self.pos_clipids)
        if diff > 0:
            # 上采样正样本
            sampled_pos = random.choices(self.pos_clipids, k=diff)
            self.pos_clipids.extend(sampled_pos)
        elif diff < 0:
            # 如果负样本反而少，可以补齐负样本（极少见）
            sampled_neg = random.choices(self.neg_clipids, k=(-diff))
            self.neg_clipids.extend(sampled_neg)

        # 合并为 clipids
        self.clipids = self.pos_clipids + self.neg_clipids
        random.shuffle(self.clipids)  # 打乱，避免排序偏差

    def len(self):
        return len(self.clipids)

    def get(self, idx):
        clipid = self.clipids[idx]
        singlept = self.clipid_dict[clipid]
        data_list = torch.load(singlept, weights_only=False)
        return data_list

class ClipGroupedDatasetForTest(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None, minnum = 0.0, maxnum = 1.0):
        super().__init__(root, transform, pre_transform)
        self.clipid_dict = dict()
        self.pos_clipids = []
        self.neg_clipids = []
        if minnum >= 0.0 and maxnum <= 1.0 and minnum < maxnum:
            self.minnum = minnum
            self.maxnum = maxnum
        else:
            self.minnum = 0.0
            self.maxnum = 1.0
        random.seed(12345)

        # 扫描所有图文件
        all_files = glob.glob(os.path.join(root, "*.pt"))
        for file in all_files:
            basename = os.path.basename(file)
            clipid = basename.split("_GphLst")[0]
            y = int(basename.split("_GphLst_")[1].split(".")[0])
            self.clipid_dict[clipid] = file
            if y == 1:
                self.pos_clipids.append(clipid)
            else:
                self.neg_clipids.append(clipid)

        # 平衡样本
        diff = len(self.neg_clipids) - len(self.pos_clipids)
        if diff > 0:
            # 上采样正样本
            sampled_pos = random.choices(self.pos_clipids, k=diff)
            self.pos_clipids.extend(sampled_pos)
        elif diff < 0:
            # 如果负样本反而少，可以补齐负样本（极少见）
            sampled_neg = random.choices(self.neg_clipids, k=(-diff))
            self.neg_clipids.extend(sampled_neg)

        # 合并为 clipids
        self.clipids = self.pos_clipids + self.neg_clipids
        random.shuffle(self.clipids)  # 打乱，避免排序偏差
        self.clipids = self.clipids[int(self.minnum * len(self.clipids)):int(self.maxnum * len(self.clipids))]

    def len(self):
        return len(self.clipids)

    def get(self, idx):
        clipid = self.clipids[idx]
        singlept = self.clipid_dict[clipid]
        data_list = torch.load(singlept, weights_only=False)
        return data_list

class ClipGroupedDatasetForTest02(Dataset):
    def __init__(self, root: str, transform=None, pre_transform=None, ratioleft = 0.0, ratioright = 1.0 ):
        super().__init__(root, transform, pre_transform)
        self.clipid_dict = dict()
        self.pos_clipids = []
        self.neg_clipids = []
        random.seed(12345)
        if ratioleft < ratioright and ratioleft >= 0.0 and ratioright <= 1.0:
            self.ratioleft = ratioleft
            self.ratioright = ratioright
        else:
            self.ratioleft = 0.0
            self.ratioright = 1.0

        # 扫描所有图文件
        all_files = glob.glob(os.path.join(root, "*.pt"))
        for file in all_files:
            basename = os.path.basename(file)
            clipid = basename.split("_GphLst")[0]
            y = int(basename.split("_GphLst_")[1].split(".")[0])
            self.clipid_dict[clipid] = file
            if y == 1:
                self.pos_clipids.append(clipid)
            else:
                self.neg_clipids.append(clipid)

        # 平衡样本
        # diff = len(self.neg_clipids) - len(self.pos_clipids)
        # if diff > 0:
        #     # 上采样正样本
        #     sampled_pos = random.choices(self.pos_clipids, k=diff)
        #     self.pos_clipids.extend(sampled_pos)
        # elif diff < 0:
        #     # 如果负样本反而少，可以补齐负样本（极少见）
        #     sampled_neg = random.choices(self.neg_clipids, k=(-diff))
        #     self.neg_clipids.extend(sampled_neg)

        # 合并为 clipids
        self.clipids = self.pos_clipids + self.neg_clipids
        random.shuffle(self.clipids)  # 打乱，避免排序偏差
        self.clipids = self.clipids[int(self.ratioleft * len(self.clipids)):int(self.ratioright * len(self.clipids))]

    def len(self):
        return len(self.clipids)

    def get(self, idx):
        clipid = self.clipids[idx]
        singlept = self.clipid_dict[clipid]
        data_list = torch.load(singlept, weights_only=False)
        return data_list

# Dataset 的 Batch-ification 转置操作
def collate_clip_batches(batch):
    # batch: List[data_list]  (batch_size 个样本，每个样本是 26 个 Data)
    # 转置：将 list of (list of Data) 变为 list of (list of 同位置 Data)
    transposed = list(zip(*batch))  # 长度为 26，每个元素是一个长度为 batch_size 的元组

    # 针对每个位置，将这个位置上所有 clip 的 Data 进行 batch
    batch_list = [Batch.from_data_list(list(graphs_at_same_index)) for graphs_at_same_index in transposed]

    return batch_list

# 仅测试用。
if __name__ == "__main__":

    # 测试1
    # ClipPartitioner：
    # image_path = "test_origin_picts/HS002.png"  # 需要存在该图像文件才能运行此示例
    # partitioner = ClipPartition(image_path=image_path, a=1200)
    # partitioner.runall()    # 一次全部运行
    #
    # # 创建一个图生成器对象
    # generator = GraphGeneration(save_dir="./dataset")
    # generator.runall(partitioner)

    # 已有图，创建一个 dataset
    dataset = ClipGroupedDataset(root="./dataset")
    # 创建一个 DataLoader
    clip_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_clip_batches)
    for step, clipdata in enumerate(clip_loader):
        print(f"Step {step + 1}:")
        print(clipdata)
        print(len(clipdata))
        # data_list 是一个 List[Data]
        # 你可以将它们 batch 一起丢入 GNN 或 clip 级别处理

    # loaded_data_list = torch.load('./dataset/cae493cb271cdb4853f617ad5edee5a3d9b56a76b6acca0c8c2a62870168f6b3_GphLst.pt', weights_only=False)
    # print(len(loaded_data_list))  # 应该是 26
    # print(type(loaded_data_list))  # 应该是 <class 'torch_geometric.data.data.Data'>
    # print(loaded_data_list[0].y.item())
