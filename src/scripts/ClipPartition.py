# 图像划分。
# ClipPartition.py
# by Lawrence Leung 2025.3.12

# 引入库
import numpy as np
import collections
from PIL import Image
from scipy.stats import norm
from typing import List, Tuple
from itertools import groupby

import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt

"""
将光刻 clip 图像从 'png' 等单通道图像转换成可进行边缘检测和后续分割的工具类。
"""
class ClipPartition:
    def __init__(self, image_path: str, a: int = 1200):
        """
        image_path : str
            输入图像的路径（假设是单通道或可转单通道）。
        a : int
            图像假设为正方形, 默认为 1200。
        """
        self.image_path = image_path        # 图像路径
        self.a = a                          # 边长，像素值。
        self.img = self._load_image()       # 加载图像

        # 所有的水平/垂直划分
        self.hor_edges = []                 # 水平边缘 [(y_pos, length), ...]，未归一化
        self.ver_edges = []                 # 垂直边缘 [(x_pos, length), ...]，未归一化

        # 大块划分，利用每 20% 分位点将整个图划分为 5 * 5 的 grid。
        self.hor_percentiles = ()           # 水平分位点，长度为4，(20%pos, 40%pos, ..., 80% pos)，未归一化
        self.ver_percentiles = ()           # 垂直分位点，长度为4，(20%pos, 40%pos, ..., 80% pos)，未归一化
        self.all_hor_percentiles = []   # 加上 0 和 a 后的水平分位点
        self.all_ver_percentiles = []   # 加上 0 和 a 后的垂直分位点

        # 对于每个大块，每两个水平边缘基础上划分垂直线；每两个垂直边缘基础上划分水平线，得到两个多维list。
        # 这个多维list shape 如下：(hor_blk_index(y), ver_blk_index(x), gap_index, block_index)。
        # 前两个维度大小均为5，后面的大小不固定。
        # 其中每个元素是一个 dict，代表一个 box，内容为 "top", "bottom", "left", "right", "color", "adj".
        # color = 0：黑；color = 1：白。
        # "adj":[{'neighbor_gap'（临近的box的gap索引）, 'neighbor_box'（临近的box的box索引）, 'adj_length'（相邻的长度）},...] 。
        # 注意这个neighbor是水平的还是垂直的看属于 hor_blocks 还是 ver_blocks！
        self.hor_blocks = []
        self.ver_blocks = []

        # 每个 block 内共同的多边形。shape 如下：(hor_blk_index(y), ver_blk_index(x))。每个元素（一个 polygon）就是一个 list，由
        # 大量dict 组成：{"hor_boxes":[(gap_index, box_index), ...], "ver_boxes":[(gap_index, box_index), ...],
        #               "hor_edges":[length1, length2, ...], "ver_edges":[length1, length2],
        #               "area":float, "vertices": [(vert_x, vert_y), ...]}
        self.poly_per_block = []

        # 大块图和大块图之间的连接关系。是一个dict。键值对如下：
        # 键：((from_y_idx, from_x_idx),(to_y_idx, to_x_idx))
        # 值：连接端点频率
        # 其中每个元素是与相邻大块图之间的连接像素数。
        self.largegraph_connections = {}

    def _load_image(self) -> np.ndarray:
        """
        加载并转换图像为 (a x a) 的 numpy 数组, 单通道 0~255。
        如果图像本身不是 (a x a), 此处可根据需要做 resize 或 assert 检查。
        """
        image = Image.open(self.image_path).convert('L')
        # 如果需要强制 resize 成 (a, a)，可取消注释以下两行：
        # image = image.resize((self.a, self.a), Image.BILINEAR)
        img_array = np.array(image, dtype=np.uint8)
        assert img_array.shape[0] == self.a and img_array.shape[1] == self.a, (
            f"图像尺寸应当为 {self.a}x{self.a}，"
            f"但当前读取到的图像尺寸为 {img_array.shape}."
        )
        img_array = np.where(img_array > 127, 255, 0).astype(np.uint8)
        return img_array

    def det_hor_edges(self) -> List[Tuple[int, int]]:
        """
        检测与 x 轴平行的白色图形边缘。
        思路:
        ----
        在图像坐标中, 水平边缘存在于行 y-1 和 y 之间,
        只要二者在同一个列上出现像素值不一致,
        且至少一方是白色 (255), 即可认为此处是一个水平边缘像素。

        输出:
        ----
        List[Tuple[int, int]]
            其中每个元组 (y_index, edge_length) 表示:
            - y_index : 对应边缘所属的行索引(较小的那一行, 即 y-1)
            - edge_length : 该行对应的水平边缘总长度(像素数)
        """
        edges_list = []
        # 从第一行与第二行之间开始(1)到倒数第二行与最后一行之间(a-1)
        for y in range(1, self.a):
            transitions = 0
            # 对该行与上一行在每个像素列上进行比较
            for x in range(self.a):
                if self.img[y, x] != self.img[y - 1, x]:
                    # 若出现不一致, 并且至少一方是白色, 视为有效边缘
                    if self.img[y, x] == 255 or self.img[y - 1, x] == 255:
                        transitions += 1
            # y-1 代表了该水平边界“在图像坐标中的行索引”
            if transitions > 0:
                edges_list.append((y - 1, transitions))
        self.hor_edges = edges_list
        return edges_list

    def det_ver_edges(self) -> List[Tuple[int, int]]:
        """
        检测与 y 轴平行的白色图形边缘。
        思路:
        ----
        在图像坐标中, 垂直边缘存在于列 x-1 和 x 之间,
        只要二者在同一个行上出现像素值不一致,
        且至少一方是白色 (255), 即可认为此处是一个垂直边缘像素。

        输出:
        ----
        List[Tuple[int, int]]
            其中每个元组 (x_index, edge_length) 表示:
            - x_index : 对应边缘所属的列索引(较小的那一列, 即 x-1)
            - edge_length : 该列对应的垂直边缘总长度(像素数)
        """
        edges_list = []
        # 从第一列与第二列之间开始(1)到倒数第二列与最后一列之间(a-1)
        for x in range(1, self.a):
            transitions = 0
            # 对该列与上一列在每个像素行上进行比较
            for y in range(self.a):
                if self.img[y, x] != self.img[y, x - 1]:
                    # 若出现不一致, 并且至少一方是白色, 视为有效边缘
                    if self.img[y, x] == 255 or self.img[y, x - 1] == 255:
                        transitions += 1
            # x-1 代表了该垂直边界“在图像坐标中的列索引”
            if transitions > 0:
                edges_list.append((x - 1, transitions))
        self.ver_edges = edges_list
        return edges_list

    def debug_edge_dist(self, edges_list: List[Tuple[int, int]], title: str = "Null") -> None:
        """
        对边缘长度进行频率分布直方图 + 正态分布拟合，并叠加若干分位点。

        参数:
        ----
        edges_list : List[Tuple[int, int]]
            形如 [(index, edge_length), ...] 的边缘信息列表，
            我们只关心其中的 edge_length 部分 (即第2个元素)。
        title : str
            为绘图设置的标题, 例如 "Horizontal Edges Distribution"。
        """
        # 仅提取边缘长度
        data = [edge[0] for edge in edges_list]
        if not data:
            print(f"{title}：edges_list 为空，无法绘图。")
            return

        # 启动一个新的绘图窗口
        plt.figure()

        # 绘制频率分布直方图 (非归一化)，以便后续叠加正态分布曲线
        counts, bin_edges, _ = plt.hist(data, bins='auto', alpha=0.6, label='Histogram')

        # 计算正态分布所需的 mean 和 std
        mu, sigma = np.mean(data), np.std(data, ddof=1)  # ddof=1 做样本标准差
        # 若 std=0 (数据全相同)，直接跳过拟合
        if sigma < 1e-9:
            print(f"{title}：数据方差几乎为0，跳过正态分布拟合。")
        else:
            # 拟合曲线的 x 坐标范围
            x_min, x_max = min(data), max(data)
            x_vals = np.linspace(x_min, x_max, 200)
            # 直方图是频数(counts)，非概率密度 -> 需要将 pdf * 总样本数 * bin_width 才能对齐
            bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0
            pdf_vals = norm.pdf(x_vals, mu, sigma)
            pdf_scaled = pdf_vals * len(data) * bin_width

            # 绘制正态分布拟合曲线
            plt.plot(x_vals, pdf_scaled, linewidth=2, label='Fitted Normal Distribution')

        # 叠加若干分位点的竖直线
        percentiles = [20, 40, 60, 80, 100]
        for p in percentiles:
            q = np.percentile(data, p)
            plt.axvline(q, linestyle='--', label=f'{p}%={q:.1f}')

        # 设置坐标轴标签、标题和图例
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend()

        # 显示图像
        plt.show()

    def compute_edge_percentiles_old(self):
        """
        计算并保存 self.hor_edges、self.ver_edges 中的 4 个分位点 (20%, 40%, 60%, 80%)。

        规则:
        ----
        1. 首先将 (index, length) 中的 length 视为该 index 出现的次数, 重复展开到 hor_data / ver_data.
        2. 对 hor_data / ver_data 分别计算 [20, 40, 60, 80] 四个分位点, 得到 pval.
        3. 找到 hor_data / ver_data 中与 pval 最近的元素, 若二者绝对差值 <= 2, 则用该元素替换 pval.
           若找不到(或差值 > 2), 则将 pval 四舍五入为整数.
        4. 最后将该结果裁剪到 [0, self.a] 范围之内.
        5. 存储结果到 self.hor_percentiles、self.ver_percentiles (shape=(4,))
        """
        if not self.hor_edges:
            raise ValueError("self.hor_edges 为空，无法计算分位点。")
        if not self.ver_edges:
            raise ValueError("self.ver_edges 为空，无法计算分位点。")

        # 将每个 (index, length) 展开到多重列表, 以 index 出现 length 次
        hor_data = [idx[0] for idx in self.hor_edges]
        ver_data = [idx[0] for idx in self.ver_edges]

        if not hor_data or not ver_data:
            raise ValueError("展开后数据为空，无法计算分位点。")

        # 目标分位点
        perc_list = [20, 40, 60, 80]

        def adjust_value(pval, data):
            """
            在 data 中找到与 pval 最近的元素，若差值 <= 2 则用该值替换，否则四舍五入为整数。
            然后裁剪到 [0, self.a].
            """
            # 找到 data 中与 pval 最近的值
            closest_val = min(data, key=lambda x: abs(x - pval))
            if abs(closest_val - pval) <= 2:
                final_val = closest_val
            else:
                final_val = round(pval)
            # 限制在 [0, self.a] 范围
            final_val = max(min(final_val, self.a), 0)
            return final_val

        # 分别计算水平方向与垂直方向的分位点，并做上述修正
        hor_pvals = [np.percentile(hor_data, p) for p in perc_list]
        self.hor_percentiles = tuple(adjust_value(v, hor_data) for v in hor_pvals)

        ver_pvals = [np.percentile(ver_data, p) for p in perc_list]
        self.ver_percentiles = tuple(adjust_value(v, ver_data) for v in ver_pvals)
        self.all_hor_percentiles = [0] + list(self.hor_percentiles) + [self.a]
        self.all_ver_percentiles = [0] + list(self.ver_percentiles) + [self.a]


    def compute_edge_percentiles(self):
        """
        计算并保存 self.hor_edges、self.ver_edges 中的 4 个分位点 (20%, 40%, 60%, 80%)。

        规则:
        ----
        1. 若 hor_edges 或 ver_edges 为空，自动取 [0.2*a, 0.4*a, 0.6*a, 0.8*a]。
        2. 否则：  
           (1) 将 (index, length) 中的 length 视为该 index 出现的次数, 展开为 hor_data / ver_data。  
           (2) 分别计算 [20, 40, 60, 80] 四个分位点。  
           (3) 找到 hor_data / ver_data 中与 pval 最近的元素, 若差值 <= 2, 则用该元素，否则四舍五入。  
           (4) 最后裁剪到 [0, self.a] 范围。  
        3. 将结果保存为 self.hor_percentiles、self.ver_percentiles，并生成 all_hor_percentiles / all_ver_percentiles。
        """
        perc_list = [20, 40, 60, 80]

        def adjust_value(pval, data):
            """
            在 data 中找到与 pval 最近的元素，若差值 <= 2 则用该值替换，否则四舍五入。
            然后裁剪到 [0, self.a]。
            """
            closest_val = min(data, key=lambda x: abs(x - pval))
            if abs(closest_val - pval) <= 2:
                final_val = closest_val
            else:
                final_val = round(pval)
            final_val = max(min(final_val, self.a), 0)
            return final_val

        # 水平分位点
        if not self.hor_edges:
            # print("self.hor_edges 为空，水平分位点自动取 [20%, 40%, 60%, 80%]。")
            self.hor_percentiles = tuple(round(p * self.a) for p in [0.2, 0.4, 0.6, 0.8])
        else:
            hor_data = []
            for idx, length in self.hor_edges:
                hor_data.extend([idx] * length)
            hor_pvals = [np.percentile(hor_data, p) for p in perc_list]
            self.hor_percentiles = tuple(adjust_value(v, hor_data) for v in hor_pvals)

        # 垂直分位点
        if not self.ver_edges:
            # print("self.ver_edges 为空，垂直分位点自动取 [20%, 40%, 60%, 80%]。")
            self.ver_percentiles = tuple(round(p * self.a) for p in [0.2, 0.4, 0.6, 0.8])
        else:
            ver_data = []
            for idx, length in self.ver_edges:
                ver_data.extend([idx] * length)
            ver_pvals = [np.percentile(ver_data, p) for p in perc_list]
            self.ver_percentiles = tuple(adjust_value(v, ver_data) for v in ver_pvals)

        # 拼接 0 和 a 边界
        self.all_hor_percentiles = [0] + list(self.hor_percentiles) + [self.a]
        self.all_ver_percentiles = [0] + list(self.ver_percentiles) + [self.a]


    # 辅助函数：在 (up, low)×(left, right) 范围内，查找垂直方向的边缘位置。这是正确的代码。
    def find_vertical_edges_in_subregion(self, up, low, left, right):
        """
        返回 subregion 中所有与 y 轴平行的“白色边缘”所在 x_pos 的集合(不含重复)。
        相当于在 [left+1, right) 之间, 逐列检查像素值在 [up, low) 范围内的黑白切换(类似 detect_vertical_edges 的逻辑)。
        然后再把 left, right 这两个边界也加入进来。
        """
        ver_edges = set()
        # 不必在单列内收集所有切换点, 只要该列存在至少一次黑白切换, 就认为此 x 是一个垂直边缘位置
        for x in range(left + 1, right):
            column_has_edge = False
            # 在该列 x, 从 up+1 到 low-1 检查是否存在黑白切换
            # (若 up+1 == low, 则无可检查像素)
            for y in range(up + 1, low):
                if self.img[y, x] != self.img[y, x - 1]:
                    # 检查是否至少一方是白色像素
                    if self.img[y, x] == 255 or self.img[y, x - 1] == 255:
                        column_has_edge = True
                        break
            if column_has_edge:
                ver_edges.add(x)
        # 添加左右边界
        ver_edges.add(left)
        ver_edges.add(right)
        ver_edges = sorted(ver_edges)
        return ver_edges

    # 辅助函数：在 (up, low)×(left, right) 范围内，查找水平方向的边缘位置。这是正确的代码。
    def find_horizontal_edges_in_subregion(self, up, low, left, right):
        # print(f"youknow {up} {low}")
        hor_edges = set()
        # 不必在单列内收集所有切换点, 只要该列存在至少一次黑白切换, 就认为此 x 是一个垂直边缘位置
        for y in range(up + 1, low):
            row_has_edge = False
            # 在该列 x, 从 up+1 到 low-1 检查是否存在黑白切换
            # (若 up+1 == low, 则无可检查像素)
            for x in range(left + 1, right):
                if self.img[y, x] != self.img[y - 1, x]:    # 按 [y, x] 访问。
                    # 检查是否至少一方是白色像素
                    if self.img[y, x] == 255 or self.img[y - 1, x] == 255:
                        row_has_edge = True
                        break
            if row_has_edge:
                hor_edges.add(y)
        # 添加左右边界
        hor_edges.add(up)
        hor_edges.add(low)
        hor_edges = sorted(hor_edges)
        return hor_edges

    # 填充每个大方块下的小方块
    def fill_hor_and_ver_blocks(self):
        """
        根据 self.hor_percentiles、self.ver_percentiles 将原图划分为 5*5 大块，
        并在每个大块下细分“gap”与“box”，最终填充 self.hor_blocks 与 self.ver_blocks。

        self.hor_blocks 的形状:
          [hor_blk_index=0..4][ver_blk_index=0..4][gap_index][box_index] = (5,) 的元组
          其中 (5,) = "top", "bottom", "left", "right", "color", "adjacents"

        color: 1 表示白色, 0 表示黑色.

        self.ver_blocks 以此类推。

        逻辑:
        ----
        1) 水平方向的 5 条边界: [0%, 20%, 40%, 60%, 80%, 100%] => [0, p20, p40, p60, p80, self.a]
        2) 垂直方向同理 => boundariesVer = [0, p20, p40, p60, p80, self.a]
        3) 最外层循环: 对每个大块 (i=0..4, j=0..4):
             up_blk_bound   = boundariesHor[i]
             low_blk_bound  = boundariesHor[i+1]
             left_blk_bound = boundariesVer[j]
             right_blk_bound= boundariesVer[j+1]

           然后在这个大块范围内, 收集并排序所有水平边缘的 y_pos => "hor_edges_within_blk"
           最后遍历其中每对相邻 y_pos => gap; 在 gap 范围内(纵向固定), 从 left_blk_bound 到 right_blk_bound 查找垂直边缘 => 细分为若干 box.
        """
        # 1) 检查分位点是否可用
        if not hasattr(self, 'hor_percentiles') or len(self.hor_percentiles) != 4:
            raise ValueError("self.hor_percentiles 无效, 应该包含4个分位点(0%,20%,40%,60%,80%,100%).")
        if not hasattr(self, 'ver_percentiles') or len(self.ver_percentiles) != 4:
            raise ValueError("self.ver_percentiles 无效, 应该包含4个分位点(0%,20%,40%,60%,80%,100%).")

        # 2) 组合成包含 0 与 self.a 在内的 5 条边界
        boundariesHor = [0] + sorted(self.hor_percentiles) + [self.a]   # hor_percentiles 是 y 值，指示水平线
        boundariesVer = [0] + sorted(self.ver_percentiles) + [self.a]   # ver_percentiles 是 x 值，指示垂直线

        # 3) 初始化 self.hor_blocks、self.ver_blocks 为 5×5 的空壳 (每个位置先放一个空列表, 内部还会细分 gap, box)
        self.hor_blocks = [[[] for _ in range(5)] for _ in range(5)]
        self.ver_blocks = [[[] for _ in range(5)] for _ in range(5)]

        # 为了后面复用：将 self.hor_edges 转成一个快速可检索的结构
        # self.hor_edges 形如 [(y_index, transitions), ...]
        # 我们只关心 y_index, 故可直接使用
        # 若 y_index 可能重复, 这里直接用列表保存即可
        y_edge_positions = [y for (y, _) in self.hor_edges]

        # 4) 逐个大块区域进行处理
        for i in range(5):      # y 方向
            up_blk_bound = boundariesHor[i]
            low_blk_bound = boundariesHor[i + 1]
            for j in range(5):  # x 方向
                left_blk_bound = boundariesVer[j]
                right_blk_bound = boundariesVer[j + 1]

                # 取得所有的水平边界，进入到 gap 层次
                hor_edges = self.find_horizontal_edges_in_subregion(up_blk_bound, low_blk_bound, left_blk_bound,
                                                                    right_blk_bound)
                ver_edges = self.find_vertical_edges_in_subregion(up_blk_bound, low_blk_bound, left_blk_bound,
                                                                    right_blk_bound)

                # 准备存放 gap -> box 的 2层列表结构
                # self.hor_blocks[i][j] 将是一个列表(gap 级别),
                # 其中每个 gap 又是一个列表(box 级别)
                gap_list_for_hor = []
                gap_list_for_ver = []

                # 5) 对于 horizontal 方向而言，对相邻 y_pos 形成 gap 区域
                for gap_index in range(len(hor_edges) - 1):
                    gap_top = hor_edges[gap_index]
                    gap_bottom = hor_edges[gap_index + 1]
                    if gap_bottom <= gap_top:
                        # 忽略空或反序区域
                        continue

                    # 在这个 gap 范围内, 向左右方向查找垂直边缘
                    ver_edges_within_gap = self.find_vertical_edges_in_subregion(
                        gap_top, gap_bottom,
                        left_blk_bound, right_blk_bound
                    )

                    # 逐 box 细分
                    box_list = []
                    for box_index in range(len(ver_edges_within_gap) - 1):
                        x_left = ver_edges_within_gap[box_index]
                        x_right = ver_edges_within_gap[box_index + 1]
                        if x_right <= x_left:
                            continue

                        # 选取盒子的中心像素(或者左上像素等)来判断颜色
                        center_y = (gap_top + gap_bottom) // 2
                        center_x = (x_left + x_right) // 2
                        # 防止越界
                        if center_y >= self.a:
                            center_y = self.a - 1
                        if center_x >= self.a:
                            center_x = self.a - 1

                        pixel_val = self.img[center_y, center_x]
                        color = 1 if pixel_val == 255 else 0

                        # 拼装 (up_bound, low_bound, left_bound, right_bound, color)
                        box_dict = {
                            'top': gap_top, 'bottom': gap_bottom, 'left': x_left, 'right': x_right, 'color': color, "adj": []
                        }
                        box_list.append(box_dict)

                    # 将该 gap 下的全部 box 存入 gap_list
                    gap_list_for_hor.append(box_list)

                # 6) 对于 vertical 方向而言，对相邻 x_pos 形成 gap 区域。仍然是对于相同大块图而言。
                for gap_index in range(len(ver_edges) - 1):
                    gap_left = ver_edges[gap_index]
                    gap_right = ver_edges[gap_index + 1]
                    if gap_right <= gap_left:
                        # 忽略空或反序区域
                        continue

                    # 在这个 gap 范围内, 向上下方向查找水平边缘
                    hor_edges_within_gap = self.find_horizontal_edges_in_subregion(
                        up_blk_bound, low_blk_bound,
                        gap_left, gap_right
                    )

                    # 逐 box 细分
                    box_list = []
                    for box_index in range(len(hor_edges_within_gap) - 1):
                        y_up = hor_edges_within_gap[box_index]
                        y_low = hor_edges_within_gap[box_index + 1]
                        if y_low <= y_up:
                            continue

                        # 选取盒子的中心像素(或者左上像素等)来判断颜色
                        center_y = (y_up + y_low) // 2
                        center_x = (gap_left + gap_right) // 2
                        # 防止越界
                        if center_y >= self.a:
                            center_y = self.a - 1
                        if center_x >= self.a:
                            center_x = self.a - 1

                        pixel_val = self.img[center_y, center_x]
                        color = 1 if pixel_val == 255 else 0

                        # 拼装 (up_bound, low_bound, left_bound, right_bound, color, [])，[] for **kwargs
                        box_dict = {
                            "top": y_up,
                            "bottom": y_low,
                            "left": gap_left,
                            "right": gap_right,
                            "color": color,
                            "adj": []
                        }
                        box_list.append(box_dict)

                    # 将该 gap 下的全部 box 存入 gap_list
                    gap_list_for_ver.append(box_list)

                # 最终赋值给 self.hor_blocks[i][j]
                self.hor_blocks[i][j] = gap_list_for_hor
                self.ver_blocks[i][j] = gap_list_for_ver
        # 填充完毕！

    # 计算水平与垂直的邻接性
    def compute_hor_ver_box_adjacency(self):
        """
        针对 self.hor_blocks 里每个大块(i, j)，逐个 gap(g)、逐个 box(b)，
        寻找同一大块内上下左右相邻的 box，计算其与当前 box 的公共边界长度，
        并将信息存储到 box 的 extra_list 中。

        相邻定义:
        --------
        1) 左右相邻: 同一个 gap 内，box_index 相邻 (b-1, b+1)；
           - 条件: neighbor.right_bound == cur.left_bound (或 neighbor.left_bound == cur.right_bound)
           - 重叠长度: 在垂直方向(y)上 two-box 的 [up_bound, low_bound) 区间求交集
        2) 上下相邻: gap_index 相邻 (g-1, g+1)；
           - 条件: neighbor.low_bound == cur.up_bound (或 neighbor.up_bound == cur.low_bound)
           - 重叠长度: 在水平方向(x)上 two-box 的 [left_bound, right_bound) 区间求交集

        存储方式:
        --------
        对每个 box，有 dict: "top", "bottom", "left", "right", "color", "adj"。其中 "adj" 是一个 dict。
        最后我们会在 extra_list 中 append 若干字典，每个字典内容例如:
          {
            "neighbor_gap": gap_idx_of_neighbor,
            "neighbor_box": box_idx_of_neighbor,
            "adj_length": length_of_shared_boundary
          }

        垂直的也在这里呈现。以此类推。
        """

        # 为了方便处理，我们定义一个内部函数，用于计算一维区间 [start1, end1)、[start2, end2) 的重叠长度
        def overlap_length(start1, end1, start2, end2):
            return max(0, min(end1, end2) - max(start1, start2))

        # 外层迭代 5x5 block
        for i in range(5):
            for j in range(5):

                # 对于水平的处理：---------------------------------------------------------------------
                # 取出某个 block 下的 gap 列表
                block_gap_list = self.hor_blocks[i][j]  # 这是一个 list(gap_index) -> list(box_index)

                # 逐 gap 处理
                for g in range(len(block_gap_list)):
                    gap_box_list = block_gap_list[g]

                    for b in range(len(gap_box_list)):  # 对于每个 box
                        # 当前 box 的数据结构
                        # 注意: 如果真的是 tuple，就需要先转换为 list 才能改动 extra_list
                        cur_box = gap_box_list[b]
                        up_b, low_b, left_b, right_b, color_b, extra_list = cur_box["top"], cur_box["bottom"], cur_box["left"], cur_box["right"], cur_box["color"], cur_box["adj"]
                            # 可用

                        # adjacency_info 将临时保存当前 box 的所有邻接信息
                        adjacency_info = []

                        ###############################
                        # 1) 检查同 gap 左右相邻的 box
                        ###############################
                        # 1.1) 左邻居: b - 1
                        if b > 0:
                            left_neighbor = gap_box_list[b - 1]
                            up_n, low_n, left_n, right_n, color_n, extra_n = left_neighbor["top"], left_neighbor["bottom"], left_neighbor["left"], left_neighbor["right"], left_neighbor["color"], left_neighbor["adj"]
                            # 判断是否真正共享竖直边: right_n == left_b
                            if right_n == left_b:
                                # 计算垂直方向的重叠长度
                                vert_overlap = overlap_length(up_b, low_b, up_n, low_n)
                                if vert_overlap > 0:
                                    adjacency_info.append({
                                        "neighbor_gap": g,
                                        "neighbor_box": b - 1,
                                        "adj_length": vert_overlap
                                    })

                        # 1.2) 右邻居: b + 1
                        if b < len(gap_box_list) - 1:
                            right_neighbor = gap_box_list[b + 1]
                            up_n, low_n, left_n, right_n, color_n, extra_n = right_neighbor["top"], right_neighbor["bottom"], right_neighbor["left"], right_neighbor["right"], right_neighbor["color"], right_neighbor["adj"]
                            # 判断是否真正共享竖直边: left_n == right_b
                            if left_n == right_b:
                                # 计算垂直方向的重叠长度
                                vert_overlap = overlap_length(up_b, low_b, up_n, low_n)
                                if vert_overlap > 0:
                                    adjacency_info.append({
                                        "neighbor_gap": g,
                                        "neighbor_box": b + 1,
                                        "adj_length": vert_overlap
                                    })

                        ###############################
                        # 2) 检查相邻 gap (上/下) 的 box
                        ###############################
                        # 2.1) 上 gap: g - 1
                        if g > 0:
                            up_gap_box_list = block_gap_list[g - 1]
                            # 遍历上 gap 里的所有 box，找出与当前 box 共享水平边界的
                            for ub_idx, up_box in enumerate(up_gap_box_list):
                                up_u, low_u, left_u, right_u, color_u, extra_u = up_box["top"], up_box["bottom"], up_box["left"], up_box["right"], up_box["color"], up_box["adj"]
                                # 条件: low_u == up_b
                                if low_u == up_b:
                                    # 计算水平方向的重叠长度
                                    horiz_overlap = overlap_length(left_b, right_b, left_u, right_u)
                                    if horiz_overlap > 0:
                                        adjacency_info.append({
                                            "neighbor_gap": g - 1,
                                            "neighbor_box": ub_idx,
                                            "adj_length": horiz_overlap
                                        })

                        # 2.2) 下 gap: g + 1
                        if g < len(block_gap_list) - 1:
                            down_gap_box_list = block_gap_list[g + 1]
                            # 遍历下 gap 里的所有 box
                            for db_idx, down_box in enumerate(down_gap_box_list):
                                up_d, low_d, left_d, right_d, color_d, extra_d = down_box["top"], down_box["bottom"], down_box["left"], down_box["right"], down_box["color"], down_box["adj"]
                                # 条件: up_d == low_b
                                if up_d == low_b:
                                    # 计算水平方向的重叠长度
                                    horiz_overlap = overlap_length(left_b, right_b, left_d, right_d)
                                    if horiz_overlap > 0:
                                        adjacency_info.append({
                                            "neighbor_gap": g + 1,
                                            "neighbor_box": db_idx,
                                            "adj_length": horiz_overlap
                                        })

                        # 将 adjacency_info 添加到当前 box 的 extra_list
                        extra_list.extend(adjacency_info)

                        # 最后写回去(若原先是 tuple，需要重新组装)
                        gap_box_list[b]['adj'] = extra_list

                # 写回 self.hor_blocks[i][j]（通常不需要显式写回，如果是可变引用，则已经更新）
                self.hor_blocks[i][j] = block_gap_list

                # 对于垂直的处理：---------------------------------------------------------------------
                # 取出某个 block 下的 gap 列表
                block_gap_list = self.ver_blocks[i][j]  # 这是一个 list(gap_index) -> list(box_index)

                # 逐 gap 处理
                for g in range(len(block_gap_list)):
                    gap_box_list = block_gap_list[g]

                    for b in range(len(gap_box_list)):  # 对于每个 box
                        # 当前 box 的数据结构
                        # 注意: 如果真的是 tuple，就需要先转换为 list 才能改动 extra_list
                        cur_box = gap_box_list[b]
                        up_b, low_b, left_b, right_b, color_b, extra_list = cur_box["top"], cur_box["bottom"], cur_box[
                            "left"], cur_box["right"], cur_box["color"], cur_box["adj"]
                        # 可用

                        # adjacency_info 将临时保存当前 box 的所有邻接信息
                        adjacency_info = []

                        ###############################
                        # 1) 检查同 gap 上下相邻的 box
                        ###############################
                        # 1.1) 上邻居: b - 1
                        if b > 0:
                            up_neighbor = gap_box_list[b - 1]
                            up_n, low_n, left_n, right_n, color_n, extra_n = up_neighbor["top"], up_neighbor[
                                "bottom"], up_neighbor["left"], up_neighbor["right"], up_neighbor["color"], \
                                                                             up_neighbor["adj"]
                            # 判断是否真正共享水平边
                            if low_n == up_b:
                                # 计算水平方向的重叠长度
                                horiz_overlap = overlap_length(left_b, right_b, left_n, right_n)
                                if horiz_overlap > 0:
                                    adjacency_info.append({
                                        "neighbor_gap": g,
                                        "neighbor_box": b - 1,
                                        "adj_length": horiz_overlap
                                    })

                        # 1.2) 下邻居: b + 1
                        if b < len(gap_box_list) - 1:
                            low_neighbor = gap_box_list[b + 1]
                            up_n, low_n, left_n, right_n, color_n, extra_n = low_neighbor["top"], low_neighbor[
                                "bottom"], low_neighbor["left"], low_neighbor["right"], low_neighbor["color"], \
                                                                             low_neighbor["adj"]
                            # 判断是否真正共享水平边
                            if up_n == low_b:
                                # 计算水平方向的重叠长度
                                horiz_overlap = overlap_length(left_b, right_b, left_n, right_n)
                                if horiz_overlap > 0:
                                    adjacency_info.append({
                                        "neighbor_gap": g,
                                        "neighbor_box": b + 1,
                                        "adj_length": horiz_overlap
                                    })

                        ###############################
                        # 2) 检查相邻 gap (左/右) 的 box
                        ###############################
                        # 2.1) 左 gap: g - 1
                        if g > 0:
                            left_gap_box_list = block_gap_list[g - 1]
                            # 遍历左 gap 里的所有 box，找出与当前 box 共享水平边界的
                            for lb_idx, left_box in enumerate(left_gap_box_list):
                                up_l, low_l, left_l, right_l, color_l, extra_l = left_box["top"], left_box["bottom"], \
                                                                                 left_box["left"], left_box["right"], \
                                                                                 left_box["color"], left_box["adj"]
                                # 条件: low_u == up_b
                                if right_l == left_b:
                                    # 计算垂直方向的重叠长度
                                    vert_overlap = overlap_length(up_b, low_b, up_l, low_l)
                                    if vert_overlap > 0:
                                        adjacency_info.append({
                                            "neighbor_gap": g - 1,
                                            "neighbor_box": lb_idx,
                                            "adj_length": vert_overlap
                                        })

                        # 2.2) 右 gap: g + 1
                        if g < len(block_gap_list) - 1:
                            right_gap_box_list = block_gap_list[g + 1]
                            # 遍历右 gap 里的所有 box
                            for rb_idx, right_box in enumerate(right_gap_box_list):
                                up_r, low_r, left_r, right_r, color_r, extra_r = right_box["top"], right_box["bottom"], \
                                                                                 right_box["left"], right_box["right"], \
                                                                                 right_box["color"], right_box["adj"]
                                # 条件: left_r == right_b
                                if left_r == right_b:
                                    # 计算垂直方向的重叠长度
                                    vert_overlap = overlap_length(up_b, low_b, up_r, low_r)
                                    if vert_overlap > 0:
                                        adjacency_info.append({
                                            "neighbor_gap": g + 1,
                                            "neighbor_box": rb_idx,
                                            "adj_length": vert_overlap
                                        })

                        # 将 adjacency_info 添加到当前 box 的 extra_list
                        extra_list.extend(adjacency_info)

                        # 最后写回去(若原先是 tuple，需要重新组装)
                        gap_box_list[b]['adj'] = extra_list

                # 写回 self.hor_blocks[i][j]（通常不需要显式写回，如果是可变引用，则已经更新）
                self.ver_blocks[i][j] = block_gap_list

    # 辅助函数: 对某个 block 的 box-list 做连通分组 (DFS/BFS)，为构建多边形服务。这是正确的代码。
    def find_polygons_in_block(self, box_list):
        """
        参数: box_list = 一个列表, 下标=gap_idx, 内容= [ { box_dict_1 }, { box_dict_2 }, ... ]
          其中 box_dict 的结构:
            {
              "top": int,
              "bottom": int,
              "left": int,
              "right": int,
              "color": 0/1,
              "adj": [
                { "neighbor_gap": g, "neighbor_box": b, "adj_length": L }, ...
              ]
            }
          同一个 gap_idx 下可能有多 box。我们把所有 gap 下的 box 合并在一起做连通分析。

        返回: polygons, 其中 polygons 是一个列表, 每个元素是 set((gap_idx, box_idx)) 表示同一个连通白色多边形
        """
        visited = set() # 宽度优先搜索
        polygons = []

        # 先收集所有 color=1 的 (g, b) 以及对应的 box_dict
        # 这里把 block 内所有 gap 的 box 摊平 => (g, b) => box_dict
        all_white_boxes = {}
        for g_idx, gap_boxes in enumerate(box_list):
            for b_idx, box_d in enumerate(gap_boxes):
                if box_d["color"] == 1:
                    all_white_boxes[(g_idx, b_idx)] = box_d

        # BFS/DFS 寻找连通分量（宽度优先搜索、深度优先搜索）
        for (g, b), box_d in all_white_boxes.items():
            if (g, b) in visited:
                continue
            # 新建一个 polygon 集合
            polygon_set = set()
            # BFS/DFS
            queue = collections.deque() # 队列
            queue.append((g, b))
            visited.add((g, b))
            polygon_set.add((g, b))

            while queue:
                cur_g, cur_b = queue.popleft()
                cur_box = all_white_boxes[(cur_g, cur_b)]
                # 遍历它的邻接
                for adj_info in cur_box["adj"]:
                    ng, nb = adj_info["neighbor_gap"], adj_info["neighbor_box"]
                    # 判断邻居是否白色
                    if (ng, nb) in all_white_boxes:
                        # 说明邻居与当前 box 同属白色, 同一多边形
                        if (ng, nb) not in visited:
                            visited.add((ng, nb))
                            queue.append((ng, nb))
                            polygon_set.add((ng, nb))
            # BFS 完成, 获得一个连通分量
            polygons.append(polygon_set)

        return polygons

    # 辅助函数: 计算 mini-box 分割，为构建多边形服务。这是正确的代码。
    def generate_miniboxes(self, box_indices, gap_list):
        """
        根据某个 polygon 下的 box 列表，基于上下左右边界进行最小单元划分。
        返回:
          mini_boxes: 一个 set(), 其中每个元素是 (top, bottom, left, right)
        """
        top_edges = set()
        bottom_edges = set()
        left_edges = set()
        right_edges = set()

        # 1. 遍历所有 box，收集所有独立边界
        for (g_idx, b_idx) in box_indices:
            box = gap_list[g_idx][b_idx]
            top_edges.add(box["top"])
            bottom_edges.add(box["bottom"])
            left_edges.add(box["left"])
            right_edges.add(box["right"])

        # 2. 将边界进行排序
        hor_div_lines = sorted(top_edges.union(bottom_edges))
        ver_div_lines = sorted(left_edges.union(right_edges))

        if hor_div_lines:  # 检查集合是否为空
            min_element = min(hor_div_lines)  # 获取最小元素
            max_element = max(hor_div_lines)  # 获取最大元素
            hor_div_lines.remove(min_element)  # 删除最小元素
            hor_div_lines.remove(max_element)  # 删除最大元素
        if ver_div_lines:  # 检查集合是否为空
            min_element = min(ver_div_lines)  # 获取最小元素
            max_element = max(ver_div_lines)  # 获取最大元素
            ver_div_lines.remove(min_element)  # 删除最小元素
            ver_div_lines.remove(max_element)  # 删除最大元素

        # 3. 生成 mini-box
        mini_boxes = set()
        for (g_idx, b_idx) in box_indices:
            box = gap_list[g_idx][b_idx]
            bt, bb, bl, br = box["top"], box["bottom"], box["left"], box["right"]
            sub_hor_div = sorted({y for y in hor_div_lines if bt < y < bb}.union({bt, bb}))
            sub_ver_div = sorted({x for x in ver_div_lines if bl < x < br}.union({bl, br}))
            for y in range(len(sub_hor_div) - 1):
                for x in range(len(sub_ver_div) - 1):
                    mini_boxes.add((
                        sub_hor_div[y], sub_hor_div[y + 1],  # top, bottom
                        sub_ver_div[x], sub_ver_div[x + 1]  # left, right
                    ))

        return mini_boxes

    # 辅助函数：合并 (y, x1, x2)、(x, y1, y2) 相互连接的线段，为构建多边形服务。这是正确的代码。
    def merge_segments(self, segments):
        # 1. 按 y 值分组
        segments.sort(key=lambda seg: seg[0])  # 先按 y 排序
        grouped = groupby(segments, key=lambda seg: seg[0])

        merged_segments = []

        # 2. 对每个 y 值的分组进行处理
        for y, group in grouped:
            sorted_group = sorted(group, key=lambda seg: seg[1])  # 按 x1 排序
            merged = []

            # 3. 合并相连的线段
            for seg in sorted_group:
                if not merged:
                    merged.append(seg)
                else:
                    last = merged[-1]
                    # 检查是否能合并
                    if last[2] >= seg[1]:
                        merged[-1] = (y, last[1], max(last[2], seg[2]))
                    else:
                        merged.append(seg)

            # 4. 添加合并后的线段
            merged_segments.extend(merged)

        return merged_segments

    # 辅助函数：使用 contour tracing 获取 polygon 的外轮廓边长、顶点。这是正确的代码。
    def extract_contour_edges(self, miniboxes, i, j):
        """
        计算 miniboxes 形成的 polygon 轮廓边界，并返回水平/垂直边长列表。
        miniboxes: set((top, bottom, left, right)) 格式
        返回:
          - h_edges: List[int]  所有外轮廓上的水平边长
          - v_edges: List[int]  所有外轮廓上的垂直边长
        """
        # 1. 记录所有水平、垂直边的集合 (以 tuple 记录完整坐标)
        hor_lines = []  # {(y, x1, x2)}
        ver_lines = []  # {(x, y1, y2)}

        for (top, bottom, left, right) in miniboxes:
            # 水平边
            hor_lines.append((top, left, right))  # 上边
            hor_lines.append((bottom, left, right))  # 下边
            # 垂直边
            ver_lines.append((left, top, bottom))  # 左边
            ver_lines.append((right, top, bottom))  # 右边

        # 2. 找到只出现一次的边 (即轮廓边)
        hor_counts = collections.Counter(hor_lines)
        ver_counts = collections.Counter(ver_lines)

        hor_lines = [item for item in hor_lines if hor_counts[item] == 1]
        ver_lines = [item for item in ver_lines if ver_counts[item] == 1]

        hor_lines = self.merge_segments(hor_lines)
        ver_lines = self.merge_segments(ver_lines)

        # 计算长度
        h_edges = [x2 - x1 for (y, x1, x2) in hor_lines]
        v_edges = [y2 - y1 for (x, y1, y2) in ver_lines]

        # 3. 计算 num_verts
        verts = []
        # 遍历所有的水平线段
        for y, x1, x2 in hor_lines:
            # 遍历所有的垂直线段
            for x, y1, y2 in ver_lines:
                # 检查交点是否满足条件
                if (y == y1 or y == y2) and (x == x1 or x == x2):
                    verts.append((x, y))

        return h_edges, v_edges, verts, hor_lines, ver_lines

    # 辅助函数：寻找大图和大图之间的连接关系。固定某个特定的 i, j。
    def find_largeblock_connections(self, hor_lines, ver_lines, i, j):
        """
        hor_lines, ver_lines：block 以内的水平、垂直线，一个 list，大量 (y, x1, x2) 或 (x, y1, y2)
        i：hor_blk_index
        j: ver_blk_index
        """
        up_b, low_b = self.all_hor_percentiles[i], self.all_hor_percentiles[i + 1]
        left_b, right_b = self.all_ver_percentiles[j], self.all_ver_percentiles[j + 1]

        up_pixs, low_pixs, left_pixs, right_pixs = 0, 0, 0, 0

        for y, x1, x2 in hor_lines:
            if abs(up_b - y) <= 2 and i > 0:  # 检测是否上界存在连接
                up_pixs += abs(x2 - x1)
            if abs(low_b - y) <= 2 and i < 4:   # 检测是否下界存在连接
                low_pixs += abs(x2 - x1)

        for x, y1, y2 in ver_lines:
            if abs(left_b - x) <= 2 and j > 0:  # 检测是否左界存在连接
                left_pixs += abs(y2 - y1)
            if abs(right_b - x) <= 2 and j < 4: # 检测是否右界存在连接
                right_pixs += abs(y2 - y1)

        if up_pixs > 0:
            self.largegraph_connections[((i, j), (i-1, j))] = up_pixs
        if low_pixs > 0:
            self.largegraph_connections[((i, j), (i+1, j))] = low_pixs
        if left_pixs > 0:
            self.largegraph_connections[((i, j), (i, j-1))] = left_pixs
        if right_pixs > 0:
            self.largegraph_connections[((i, j), (i, j+1))] = right_pixs

    # 构建多边形，以及加入“大块图”的连接关系
    def extract_polygons_per_block(self):
        """
        在每个 block 内:
          1. 找出所有 color=1 的水平 box，通过 DFS/BFS 在 'adj' 中找连通分量 => 得到 horizontal polygons
          2. 找出所有 color=1 的垂直 box，同理 => 得到 vertical polygons
          3. 将二者进行“一一对应”匹配，合并为同一个 polygon
          4. 计算该 polygon 的 hor_boxes、ver_boxes、hor_edges、ver_edges、area、num_vertices
          5. 存入 self.poly_per_block[i][j] 中

        self.poly_per_block 的形状: 5x5, 每个元素是一个 list, 其中每个元素是:
        {
          "hor_boxes": [(gap_idx, box_idx), ...],
          "ver_boxes": [(gap_idx, box_idx), ...],
          "hor_edges": [e1, e2, ...],
          "ver_edges": [e1, e2, ...],
          "area": float,
          "num_vertices": int
        }
        """

        # 先初始化 poly_per_block => 5x5, 每个位置一个空 list
        self.poly_per_block = [[[] for _ in range(5)] for _ in range(5)]

        #----------------------------------------------------------------------
        # 2) 逐 block 处理: 找到 horizontal polygons / vertical polygons
        #----------------------------------------------------------------------
        for i in range(5):
            for j in range(5):
                hor_gap_list = self.hor_blocks[i][j]  # shape: [gap_index -> list_of_box_dict]
                ver_gap_list = self.ver_blocks[i][j]

                # 2.1) 找到所有 horizontal polygons
                hor_polygons = self.find_polygons_in_block(hor_gap_list)
                # 2.2) 找到所有 vertical polygons
                ver_polygons = self.find_polygons_in_block(ver_gap_list)

                # 2.3) 生成 hor_polygons 和 ver_polygons 的 mini-box 结构，并匹配
                # 计算 hor_polygons 和 ver_polygons 的 mini-boxes
                hor_poly_miniboxes = [self.generate_miniboxes(poly, hor_gap_list) for poly in hor_polygons]
                ver_poly_miniboxes = [self.generate_miniboxes(poly, ver_gap_list) for poly in ver_polygons]

                # ------------------------------------------------------------------
                # 3) 进行匹配: 找到 mini-box 完全相同的 hor_polygon 与 ver_polygon
                # ------------------------------------------------------------------
                matched_pairs = []
                used_ver_idx = set()

                for h_idx, h_mini in enumerate(hor_poly_miniboxes):
                    for v_idx, v_mini in enumerate(ver_poly_miniboxes):
                        if v_idx in used_ver_idx:
                            continue
                        # 匹配新方法
                        _, _, verts_1, _, _ = self.extract_contour_edges(hor_poly_miniboxes[h_idx], i, j)
                        _, _, verts_2, _, _ = self.extract_contour_edges(ver_poly_miniboxes[v_idx], i, j)
                        # 若两个 polygon 生成的 mini-box 列表完全一致，则匹配
                        if set(verts_1) == set(verts_2):
                            # print(f"Matched! h_idx = {h_idx}, v_idx = {v_idx}")
                            matched_pairs.append((h_idx, v_idx))  # hor_polygon 与 ver_polygon 的 index。
                            used_ver_idx.add(v_idx)
                            break

                #------------------------------------------------------------------
                # 4) 构建 self.poly_per_block[i][j] => 每个 polygon 一个 dict
                #------------------------------------------------------------------
                # 对于成功匹配的 pair, 合并 hor_boxes / ver_boxes

                block_polygons = [] # 每个大的 block polygon，也是最终输出的对象。
                hor_lines, ver_lines = [], []
                for (h_idx, v_idx) in matched_pairs:
                    # horizontal polygon
                    h_poly_set = hor_polygons[h_idx]
                    # vertical polygon
                    v_poly_set = ver_polygons[v_idx]

                    # 4.1) 收集 hor_boxes / ver_boxes
                    hor_box_indices = sorted(list(h_poly_set))
                    ver_box_indices = sorted(list(v_poly_set))

                    ## 4.2) 提取 polygon 的水平边 (h_edges) 与垂直边 (v_edges)，以及计算 num_vertices
                    # 目标: 找到 polygon 外轮廓上的所有水平/垂直边，并记录它们的长度
                    h_edges, v_edges, verts_, hor_lines, ver_lines = self.extract_contour_edges(hor_poly_miniboxes[h_idx], i, j)

                    # 4.3) 计算 “area”： 直接累加所有 mini-box 的 (bottom - top) * (right - left)
                    area_ = sum(
                        (bottom - top) * (right - left)
                        for (top, bottom, left, right) in hor_poly_miniboxes[h_idx])

                    poly_dict = {
                        "hor_boxes": hor_box_indices,
                        "ver_boxes": ver_box_indices,
                        "hor_edges": h_edges,
                        "ver_edges": v_edges,
                        "area": area_,
                        "vertices": verts_
                    }
                    block_polygons.append(poly_dict)    # 最终将合并起来的 block polygon 添加进去！

                self.poly_per_block[i][j] = block_polygons
                # 到此，每个 block 的 polygon 信息写入 self.poly_per_block

                # 5) 构建大图间的连接关系
                self.find_largeblock_connections(hor_lines, ver_lines, i, j)

    # 全流程一次运行完毕
    def runall(self):
        # 1) 检测水平方向边缘、垂直方向边缘
        self.det_hor_edges()
        self.det_ver_edges()
        # 2) 计算水平、垂直的分位点
        self.compute_edge_percentiles()
        # 3) 填充水平方块、垂直方块
        self.fill_hor_and_ver_blocks()
        # 4) 邻接性处理，计算与其相连接的东西
        self.compute_hor_ver_box_adjacency()
        # 5) 寻找多边形，建立大块图之间的连接关系
        self.extract_polygons_per_block()
        return 0

# 仅测试用。
if __name__ == "__main__":
    # 以下为简单测试示例，需自行修改 image_path 为你实际的图像路径
    image_path = "test_origin_picts/HS002.png"  # 需要存在该图像文件才能运行此示例
    partitioner = ClipPartition(image_path=image_path, a=1200)

    # 一次全部运行
    partitioner.runall()

    # 调用调试分布绘图
    # partitioner.debug_edge_dist(horizontal_edges, "Horizontal Edges Distribution")
    # partitioner.debug_edge_dist(vertical_edges, "Vertical Edges Distribution")

    print("Horizontal Percentiles (y):", partitioner.all_hor_percentiles)
    print("Vertical Percentiles (x):", partitioner.all_ver_percentiles)

    print(partitioner.hor_blocks[2][4])
    print(partitioner.ver_blocks[2][4])

    k = 0
    for i in range(5):
        for j in range(5):
            if len(partitioner.poly_per_block[i][j]) != 0:
                k += len(partitioner.poly_per_block[i][j])
    print((partitioner.poly_per_block[2][4]))
    print(f"polys :: {k}")

    print(partitioner.largegraph_connections)
'''
TODO:
1. 将每个大块图中发生重叠的部分，得到一个独立的 polygon 建立一个list，(hor_blk_index(y), ver_blk_index(x), every_polygon)。
    每个元素就是一个 polygon list，所包含的 tuple：(isfrom_horblocks, gapindex, blockindex)。
2. 建构每个大块图的从下到上的纵向图，以及从左到右的横向图。
'''