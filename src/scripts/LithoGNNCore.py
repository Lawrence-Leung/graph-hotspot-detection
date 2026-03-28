# GNN 核心构建代码
# LithoGNNCore.py
# by Lawrence Leung 2025.3.18

# 引入库
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, BatchNorm, global_max_pool, LayerNorm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph
from tqdm import tqdm
import time
import json
import os

# 从已有目录中导入
from GraphGeneration import ClipGroupedDataset, collate_clip_batches

import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt


# 神经网络构造类 ============================================================================

# 自定义 MPNN 层（max 聚合）
class ResidualMPNNLayer(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='max')
        self.lin = nn.Linear(channels, channels)
        self.bn = LayerNorm(channels)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        out = self.lin(out)
        out = self.bn(out)
        out = F.relu(out + x)  # 残差连接
        return out

    def message(self, x_j):
        return x_j

# 可学习门控融合模块
class GatedFusion(nn.Module):
    def __init__(self, channels, num_hops):
        super().__init__()
        self.gates = nn.Parameter(torch.randn(num_hops, channels))

    def forward(self, hop_outputs):
        # hop_outputs: List of [num_nodes, channels]
        fused = 0
        for i, out in enumerate(hop_outputs):
            gate = torch.sigmoid(self.gates[i])  # [channels]
            fused = fused + out * gate  # 广播 [num_nodes, channels] * [channels]
        return fused

# 对小块图三步走的策略
class ThreeHopSmallBlock(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=64, num_hops=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.hop_layers = nn.ModuleList([
            ResidualMPNNLayer(hidden_channels) for _ in range(num_hops)
        ])
        self.gate_fusion = GatedFusion(hidden_channels, num_hops)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        hop_outputs = []

        for conv in self.hop_layers:
            x = conv(x, edge_index)
            hop_outputs.append(x)

        x = self.gate_fusion(hop_outputs)
        x = self.dropout(x)
        # 此处不 pool，因为是子图节点特征输出
        return x

# [1] 小块图处理模块，整合 small graph 特征并根据 edge_attr=3 和节点属性进一步处理
class SingleUnitSmallBlock(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=128, num_hops=3, dropout=0.2):
        super().__init__()
        self.small_gnn = ThreeHopSmallBlock(in_channels, hidden_channels, num_hops, dropout)
        self.final_mpn = ResidualMPNNLayer(hidden_channels)  # 第 4 hop 消息传递
        self.hidden_channels = hidden_channels

    def forward(self, data):
        # step1: 整图 3-hop MPNN
        node_features = self.small_gnn(data.x, data.edge_index)  # [num_nodes, 64]
        # step2: 提取 edge_attr=3 的边，sink 节点 max pooling
        polygon_mask = (data.edge_attr == 3)
        polygon_edges = data.edge_index[:, polygon_mask]
        sink_nodes = polygon_edges[1].unique()
        if sink_nodes.numel() > 0:
            subgraph_features = self.final_mpn(node_features, polygon_edges)
            polygon_features = subgraph_features[sink_nodes]
            vector_step2 = polygon_features.max(dim=0, keepdim=True)[0]  # (1, hidden_channels)
        else:
            vector_step2 = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device, dtype=node_features.dtype)
        # step3: 根据输入节点特征 [0] > 0.1 进行删除，也就是只剩下白色的东西
        node_property_mask = (data.x[:, 0] > 0.1)  # True 表示需要删除
        remaining_mask = ~node_property_mask
        remaining_features = node_features[remaining_mask]
        if remaining_features.numel() > 0:
            vector_step3 = remaining_features.max(dim=0, keepdim=True)[0]  # (1, hidden_channels)
        else:
            vector_step3 = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device, dtype=node_features.dtype)  # 默认填充
        # step4: 拼接输出。白色的和 polygon 的信息。
        output = torch.cat([vector_step2, vector_step3], dim=0)  # (2,64)
        return output

# [1a] 带 batch 的小块图处理模块
class SingleUnitSmallBlockBatch(nn.Module):
    def __init__(self, in_channels=10, hidden_channels=128, num_hops=3, dropout=0.2):
        super().__init__()
        self.small_gnn = ThreeHopSmallBlock(in_channels, hidden_channels, num_hops, dropout)
        self.final_mpn = ResidualMPNNLayer(hidden_channels)
        self.hidden_channels = hidden_channels

    def forward(self, data):
        # 整图 3-hop MPNN
        node_features = self.small_gnn(data.x, data.edge_index)  # [total_nodes, hidden]

        batch_size = data.batch.max().item() + 1
        outputs = []

        for graph_id in range(batch_size):
            # 当前图的节点索引
            node_mask = (data.batch == graph_id)
            node_idx = node_mask.nonzero(as_tuple=False).view(-1)
            if node_idx.numel() == 0:
                # 空图，返回填充
                vector_step2 = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device)
                vector_step3 = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device)
                outputs.append(torch.cat([vector_step2, vector_step3], dim=0).unsqueeze(0))  # (1,2,hidden)
                continue

            # 提取这个图的子边索引
            edge_mask = node_mask[data.edge_index[0]] & node_mask[data.edge_index[1]]
            edge_idx = data.edge_index[:, edge_mask]
            edge_attr = data.edge_attr[edge_mask]

            sub_node_features = node_features[node_idx]

            # step2: polygon feature pooling
            polygon_mask = (edge_attr == 3)
            polygon_edges_global = edge_idx[:, polygon_mask]
            if polygon_edges_global.numel() > 0:
                # 将 polygon_edges_global 映射为 local 索引
                polygon_edges_local, _ = subgraph(
                    node_idx, polygon_edges_global, relabel_nodes=True, num_nodes=node_features.shape[0]
                )
                subgraph_features = self.final_mpn(sub_node_features, polygon_edges_local)
                sink_nodes = polygon_edges_local[1].unique()
                polygon_features = subgraph_features[sink_nodes]
                vector_step2 = polygon_features.max(dim=0, keepdim=True)[0]
            else:
                vector_step2 = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device)

            # step3: 按照 x[:,0] 特征删除节点
            x_sub = data.x[node_idx]
            delete_mask = (x_sub[:, 0] > 0.1)
            keep_mask = ~delete_mask
            remaining_features = sub_node_features[keep_mask]
            if remaining_features.numel() > 0:
                remain_vec = remaining_features.max(dim=0, keepdim=True)[0]
            else:
                remain_vec = torch.full((1, self.hidden_channels), 1e-4, device=node_features.device)

            # 拼接
            out_feat = torch.cat([vector_step2, remain_vec], dim=0).unsqueeze(0)  # (1,2,hidden)
            outputs.append(out_feat)

        # 最后 concat
        output = torch.cat(outputs, dim=0)  # (batch_size, 2, hidden)
        return output


# 自定义大块图 MPNN 层（可根据边权重调节消息）
class WeightedMPNNLayer(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(channels, channels)
        self.bn = LayerNorm(channels)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.lin(out)
        out = self.bn(out)
        out = F.relu(out + x)  # 残差连接
        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr.view(-1, 1)  # 按边权重调整消息

# [2] 大块图 GNN 模块
class LargeBlockGraphNet(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=128, num_hops=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.hop_layers = nn.ModuleList([
            WeightedMPNNLayer(hidden_channels) for _ in range(num_hops)
        ])
        self.pool_fc = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x = F.relu(self.input_proj(data.x))

        for conv in self.hop_layers:
            x = conv(x, data.edge_index, data.edge_attr)

        # 对 25 个节点特征取全局均值聚合 (也可尝试 max 或 concat)
        pooled = x.mean(dim=0, keepdim=True)  # (1, 64)
        output = self.dropout(F.relu(self.pool_fc(pooled)))
        return output  # (1, 64)

# [2a] 带 batch 的大块图 GNN 模块
class LargeBlockGraphNetBatch(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=128, num_hops=3, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.hop_layers = nn.ModuleList([
            WeightedMPNNLayer(hidden_channels) for _ in range(num_hops)
        ])
        self.pool_fc = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # data.x, data.edge_index, data.edge_attr, data.batch
        x = F.relu(self.input_proj(data.x))
        for conv in self.hop_layers:
            x = conv(x, data.edge_index, data.edge_attr)

        # 根据 batch 将节点特征聚合成图特征
        pooled = global_max_pool(x, data.batch)  # shape=(batch_size, hidden_channels)
        output = self.dropout(F.relu(self.pool_fc(pooled)))  # (batch_size, hidden_channels)
        return output

# [3] 最终的聚合操作
class FinalAggregator(nn.Module):
    def __init__(self, dropout=0.2, hidden_channels = 1024):
        super().__init__()
        torch.manual_seed(12345)
        self.outer_ring_block = SingleUnitSmallBlock(hidden_channels=hidden_channels)
        self.middle_ring_block = SingleUnitSmallBlock(hidden_channels=hidden_channels)
        self.inner_ring_block = SingleUnitSmallBlock(hidden_channels=hidden_channels)
        self.large_block_net = LargeBlockGraphNet(hidden_channels=hidden_channels)

        self.hidden_channels = hidden_channels

        # 小图处理索引
        self.outer_ring_idx = [
                             (i, 0) for i in range(5)
                         ] + [
                             (0, i) for i in range(1, 5)
                         ] + [
                             (i, 4) for i in range(1, 5)
                         ] + [
                             (4, i) for i in range(1, 4)
                         ]  # 共16个索引

        self.middle_ring_idx = [
                              (i, 1) for i in range(1, 4)
                          ] + [
                              (i, 3) for i in range(1, 4)
                          ] + [(1, 2), (3, 2)]  # 共8个索引

        self.inner_ring_idx = [(2, 2)]  # 单个小图

        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_channels, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, data_list):

        def get_block_feature(block, indices):
            outputs = []
            for (r, c) in indices:
                idx = r * 5 + c
                single_data = data_list[idx]
                out = block(single_data)
                pooled = out.max(dim=0, keepdim=True)[0]  # (1,64)
                outputs.append(pooled)
            return torch.stack(outputs).max(dim=0)[0]  # (1,64)

        outer_ring_feat = get_block_feature(self.outer_ring_block, self.outer_ring_idx)
        middle_ring_feat = get_block_feature(self.middle_ring_block, self.middle_ring_idx)
        inner_ring_feat = get_block_feature(self.inner_ring_block, self.inner_ring_idx)

        # 大块图输入
        large_graph_data = data_list[25]
        large_block_feat = self.large_block_net(large_graph_data)  # (1,64)

        # concat 聚合
        concat_features = torch.cat([
            outer_ring_feat, middle_ring_feat, inner_ring_feat, large_block_feat
        ], dim=0).view(1, -1)  # (1, 256)

        output = self.mlp(concat_features)  # (1, 2)

        return output

# [3a] 带 batch 的最终聚合操作
class FinalAggregatorBatch(nn.Module):
    def __init__(self, dropout=0.2, hidden_channels = 1024):
        super().__init__()
        torch.manual_seed(12345)
        self.outer_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.middle_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.inner_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.large_block_net = LargeBlockGraphNetBatch(hidden_channels=hidden_channels)

        self.hidden_channels = hidden_channels

        # 小图处理索引
        self.outer_ring_idx = [
                             (i, 0) for i in range(5)
                         ] + [
                             (0, i) for i in range(1, 5)
                         ] + [
                             (i, 4) for i in range(1, 5)
                         ] + [
                             (4, i) for i in range(1, 4)
                         ]  # 共16个索引

        self.middle_ring_idx = [
                              (i, 1) for i in range(1, 4)
                          ] + [
                              (i, 3) for i in range(1, 4)
                          ] + [(1, 2), (3, 2)]  # 共8个索引

        self.inner_ring_idx = [(2, 2)]  # 单个小图

        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_channels, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, data_list):

        def get_block_feature(block, indices, data_list):
            outputs = []  # 用来存储 (batch_size, hidden_channels)
            for (r, c) in indices:
                idx = r * 5 + c
                batch_data = data_list[idx]  # Batch
                out = block(batch_data)  # (batch_size, 2, hidden_channels)

                # 对第二维（2个通道：polygon 和 remaining）取 max
                pooled = out.max(dim=1)[0]  # (batch_size, hidden_channels)
                outputs.append(pooled)

            # 现在 outputs 是 (num_indices, batch_size, hidden_channels)
            stacked = torch.stack(outputs, dim=0)  # (num_indices, batch_size, hidden_channels)

            # 对 num_indices 维度 max pooling，得到 (batch_size, hidden_channels)
            final = stacked.max(dim=0)[0]
            return final  # (batch_size, hidden_channels)

        # batch_size = data_list[0].num_graphs  # 每个 databatch 中 graph 数量就是 batch_size

        outer_ring_feat = get_block_feature(self.outer_ring_block, self.outer_ring_idx, data_list)
        middle_ring_feat = get_block_feature(self.middle_ring_block, self.middle_ring_idx, data_list)
        inner_ring_feat = get_block_feature(self.inner_ring_block, self.inner_ring_idx, data_list)

        large_graph_data = data_list[25]
        large_block_feat = self.large_block_net(large_graph_data)

        # 合并 (batch_size, 4 * hidden_channels)
        concat_features = torch.cat([
            outer_ring_feat, middle_ring_feat, inner_ring_feat, large_block_feat
        ], dim=1)

        output = self.mlp(concat_features)  # (batch_size, 2)
        return output

class FinalAggregatorBatchDebug01(nn.Module):
    def __init__(self, dropout=0.05, hidden_channels = 1024):
        super().__init__()
        torch.manual_seed(12345)
        self.outer_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.middle_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.inner_ring_block = SingleUnitSmallBlockBatch(hidden_channels=hidden_channels)
        self.large_block_net = LargeBlockGraphNetBatch(hidden_channels=hidden_channels)

        self.hidden_channels = hidden_channels

        # 小图处理索引
        self.outer_ring_idx = [
                             (i, 0) for i in range(5)
                         ] + [
                             (0, i) for i in range(1, 5)
                         ] + [
                             (i, 4) for i in range(1, 5)
                         ] + [
                             (4, i) for i in range(1, 4)
                         ]  # 共16个索引

        self.middle_ring_idx = [
                              (i, 1) for i in range(1, 4)
                          ] + [
                              (i, 3) for i in range(1, 4)
                          ] + [(1, 2), (3, 2)]  # 共8个索引

        self.inner_ring_idx = [(2, 2)]  # 单个小图

        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_channels, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 2)
        )

    def forward(self, data_list):

        def get_block_feature(block, indices, data_list):
            outputs = []  # 用来存储 (batch_size, hidden_channels)
            for (r, c) in indices:
                idx = r * 5 + c
                batch_data = data_list[idx]  # Batch
                out = block(batch_data)  # (batch_size, 2, hidden_channels)

                # 对第二维（2个通道：polygon 和 remaining）取 max
                pooled = out.max(dim=1)[0]  # (batch_size, hidden_channels)
                outputs.append(pooled)

            # 现在 outputs 是 (num_indices, batch_size, hidden_channels)
            stacked = torch.stack(outputs, dim=0)  # (num_indices, batch_size, hidden_channels)

            # 对 num_indices 维度 max pooling，得到 (batch_size, hidden_channels)
            final = stacked.max(dim=0)[0]
            return final  # (batch_size, hidden_channels)

        # batch_size = data_list[0].num_graphs  # 每个 databatch 中 graph 数量就是 batch_size

        outer_ring_feat = get_block_feature(self.outer_ring_block, self.outer_ring_idx, data_list)
        middle_ring_feat = get_block_feature(self.middle_ring_block, self.middle_ring_idx, data_list)
        inner_ring_feat = get_block_feature(self.inner_ring_block, self.inner_ring_idx, data_list)

        large_graph_data = data_list[25]
        large_block_feat = self.large_block_net(large_graph_data)

        # 合并 (batch_size, 4 * hidden_channels)
        concat_features = torch.cat([
            outer_ring_feat, middle_ring_feat, inner_ring_feat, large_block_feat
        ], dim=1)

        output = self.mlp(concat_features)  # (batch_size, 2)
        return output

# 带有计时的版本
class FinalAggregatorTimed(FinalAggregatorBatch):
    def forward(self, data_list):
        torch.cuda.synchronize()
        t0 = time.time()

        # === 小图处理 ===
        outer = self._get_block(self.outer_ring_block, self.outer_ring_idx, data_list)
        middle = self._get_block(self.middle_ring_block, self.middle_ring_idx, data_list)
        inner = self._get_block(self.inner_ring_block, self.inner_ring_idx, data_list)

        torch.cuda.synchronize()
        t1 = time.time()

        # === 大图处理 ===
        large = self.large_block_net(data_list[25])

        torch.cuda.synchronize()
        t2 = time.time()

        # === MLP 分类 ===
        concat_features = torch.cat([outer, middle, inner, large], dim=1)
        output = self.mlp(concat_features)

        torch.cuda.synchronize()
        t3 = time.time()

        return output, {
            "small_block_time": t1 - t0,
            "large_block_time": t2 - t1,
            "mlp_time": t3 - t2,
            "total_time": t3 - t0
        }

    def _get_block(self, block, indices, data_list):
        outputs = []
        for (r, c) in indices:
            idx = r * 5 + c
            out = block(data_list[idx])              # (batch_size, 2, hidden)
            pooled = out.max(dim=1)[0]               # (batch_size, hidden)
            outputs.append(pooled)
        stacked = torch.stack(outputs, dim=0)        # (num_blocks, batch, hidden)
        return stacked.max(dim=0)[0]                 # (batch, hidden)

# 神经网络训练 ==============================================================================
# 训练函数（带进度条）
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training', leave=False)
    for data_list in pbar:
        data_list = [data.to(device) for data in data_list]
        optimizer.zero_grad()

        out = model(data_list)  # 输出 (batch_size, 2)
        label = data_list[0].y.view(-1).to(device)  # (batch_size,)

        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

# 测试函数
def test(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data_list in loader:
            data_list = [data.to(device) for data in data_list]
            out = model(data_list)
            pred = out.argmax(dim=1)  # (batch_size,)
            label = data_list[0].y.view(-1).to(device)
            correct += int((pred == label).sum())

    return correct / len(loader)

if __name__ == "__main__":
    # 已有图，创建一个 dataset
    dataset = ClipGroupedDataset(root="D:\\GradProject\\OAS_Transformer\\dataset")
    # 创建一个 DataLoader
    clip_loader = DataLoader(dataset, batch_size=5, shuffle=True, collate_fn=collate_clip_batches)

    model = FinalAggregatorBatch(hidden_channels=1024)
    model1 = SingleUnitSmallBlockBatch(hidden_channels=1024)
    model2 = LargeBlockGraphNetBatch(hidden_channels=1024)

    for step, data_list in enumerate(clip_loader):
        print(f"Step {step + 1}:")
        print(f"{len(data_list)} graphsets")
        print(f"{data_list[0].num_graphs} samples")
        try:
            # sdata = data_list[0]
            # ldata = data_list[-1]

            # print(data_list[0].y.view(-1))

            out = model(data_list)
            print(out)
            print(out.shape)
        finally:
            pass
