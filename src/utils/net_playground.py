import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.loader import DataLoader
# 从已有目录中导入
from GraphGeneration import ClipGroupedDataset, collate_clip_batches

'''
TEST 002
'''
# 自定义 MPNN 层（max 聚合）
class ResidualMPNNLayer(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='max')
        self.lin = nn.Linear(channels, channels)
        self.bn = BatchNorm(channels)

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

# 自定义大块图 MPNN 层（可根据边权重调节消息）
class WeightedMPNNLayer(MessagePassing):
    def __init__(self, channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(channels, channels)
        self.bn = BatchNorm(channels)

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
    def __init__(self, in_channels=4, hidden_channels=128, num_hops=3, dropout=0.2):
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


# 主函数，用于测试 ============================================================
if __name__ == "__main__":

    # 已有图，创建一个 dataset
    dataset = ClipGroupedDataset(root="D:\\GradProject\\OAS_Transformer\\dataset")
    # 创建一个 DataLoader
    clip_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_clip_batches)
    data, largedata = [], []
    datalist = []

    model = FinalAggregator(hidden_channels=1024)

    for step, data_list in enumerate(clip_loader):
        print(f"Step {step + 1}:")
        print(f"Loaded {len(data_list)} graphs (expected {len(data_list) // 26} clips)")
        print(f"{data_list[0].clipid}")

        try:
            out = model(data_list)
            print(out)
        finally:
            pass

