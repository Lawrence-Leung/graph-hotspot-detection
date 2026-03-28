# AblationStudies.py
# Train.py
# by Lawrence Leung 2025.4.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from LithoGNNCore import FinalAggregatorBatch, SingleUnitSmallBlockBatch, LargeBlockGraphNetBatch, ResidualMPNNLayer, ThreeHopSmallBlock
from GraphGeneration import ClipGroupedDatasetForTest02
from Test import ModelTester


# ========== 1. 去除某一部分的输入模块 ============

class FinalAggregatorAblation_NoOuterRing(FinalAggregatorBatch):
    def forward(self, data_list):
        middle = self._get_block(self.middle_ring_block, self.middle_ring_idx, data_list)
        inner = self._get_block(self.inner_ring_block, self.inner_ring_idx, data_list)
        large = self.large_block_net(data_list[25])
        outer = torch.zeros_like(middle)  # 全 0 替代
        return self.mlp(torch.cat([outer, middle, inner, large], dim=1))

class FinalAggregatorAblation_NoMiddleRing(FinalAggregatorBatch):
    def forward(self, data_list):
        outer = self._get_block(self.outer_ring_block, self.outer_ring_idx, data_list)
        inner = self._get_block(self.inner_ring_block, self.inner_ring_idx, data_list)
        large = self.large_block_net(data_list[25])
        middle = torch.zeros_like(inner)
        return self.mlp(torch.cat([outer, middle, inner, large], dim=1))

class FinalAggregatorAblation_NoInnerRing(FinalAggregatorBatch):
    def forward(self, data_list):
        outer = self._get_block(self.outer_ring_block, self.outer_ring_idx, data_list)
        middle = self._get_block(self.middle_ring_block, self.middle_ring_idx, data_list)
        large = self.large_block_net(data_list[25])
        inner = torch.zeros_like(middle[:, :self.hidden_channels])
        return self.mlp(torch.cat([outer, middle, inner, large], dim=1))

class FinalAggregatorAblation_NoLargeBlock(FinalAggregatorBatch):
    def forward(self, data_list):
        outer = self._get_block(self.outer_ring_block, self.outer_ring_idx, data_list)
        middle = self._get_block(self.middle_ring_block, self.middle_ring_idx, data_list)
        inner = self._get_block(self.inner_ring_block, self.inner_ring_idx, data_list)
        large = torch.zeros_like(inner)
        return self.mlp(torch.cat([outer, middle, inner, large], dim=1))

# ========== 修复版 no_gate：保留结构，只替换 GatedFusion ============
class GatedFusionAvg(nn.Module):
    def forward(self, hop_outputs):
        return torch.stack(hop_outputs).mean(dim=0)

class FinalAggregatorAblation_NoGate(FinalAggregatorBatch):
    def __init__(self, dropout=0.2, hidden_channels=1024):
        super().__init__(dropout, hidden_channels)

        def replace_fusion(module):
            module.small_gnn.gate_fusion = GatedFusionAvg()

        replace_fusion(self.outer_ring_block)
        replace_fusion(self.middle_ring_block)
        replace_fusion(self.inner_ring_block)

# ========== 修复版 one_hop：保持3层结构，仅禁用后两层传播 ============
class ResidualMPNNLayerNoOp(nn.Module):
    def forward(self, x, edge_index):
        return x

class FinalAggregatorAblation_OneHop(FinalAggregatorBatch):
    def __init__(self, dropout=0.2, hidden_channels=1024):
        super().__init__(dropout, hidden_channels)

        def disable_extra_hops(module):
            module.small_gnn.hop_layers[1] = ResidualMPNNLayerNoOp()
            module.small_gnn.hop_layers[2] = ResidualMPNNLayerNoOp()

        disable_extra_hops(self.outer_ring_block)
        disable_extra_hops(self.middle_ring_block)
        disable_extra_hops(self.inner_ring_block)

# ========== 修复版 shallow_mlp：保持结构，替换中间层为恒等映射 ============
class FinalAggregatorAblation_ShallowMLP(FinalAggregatorBatch):
    def __init__(self, dropout=0.2, hidden_channels=1024):
        super().__init__(dropout, hidden_channels)

        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear) and i not in [0, len(self.mlp)-1]:
                in_dim = layer.in_features
                out_dim = layer.out_features
                identity = nn.Linear(in_dim, out_dim)
                with torch.no_grad():
                    identity.weight.copy_(torch.eye(out_dim, in_dim)[:out_dim])
                    identity.bias.zero_()
                self.mlp[i] = identity
            elif isinstance(layer, (nn.LayerNorm, nn.ReLU, nn.Dropout)):
                self.mlp[i] = nn.Identity()

# ========== 5. 工具方法重用 ============
def _get_block(self, block, indices, data_list):
    outputs = []
    for (r, c) in indices:
        idx = r * 5 + c
        out = block(data_list[idx])
        pooled = out.max(dim=1)[0]
        outputs.append(pooled)
    stacked = torch.stack(outputs, dim=0)
    return stacked.max(dim=0)[0]

FinalAggregatorBatch._get_block = _get_block

## 主程序
if __name__ == "__main__":

    ablation_variants = {
        # "baseline": FinalAggregatorBatch,
        # "no_outer": FinalAggregatorAblation_NoOuterRing,
        # "no_middle": FinalAggregatorAblation_NoMiddleRing,
        # "no_inner": FinalAggregatorAblation_NoInnerRing,
        # "no_large": FinalAggregatorAblation_NoLargeBlock,
        "no_gate": FinalAggregatorAblation_NoGate,
        "one_hop": FinalAggregatorAblation_OneHop,
        "shallow_mlp": FinalAggregatorAblation_ShallowMLP,
    }


    for tag, model_cls in ablation_variants.items():
        print(f"=== Testing ablation: {tag} ===")
        model = model_cls()
        model.load_state_dict(torch.load("/ai/edallx/Graduate_Project_2025/NewMethods/temp_logs/best_model.pt",
                                         weights_only=False), strict=False)  # 使用相同初始模型

        dataset = ClipGroupedDatasetForTest02(root="/ai/edallx/Graduate_Project_2025/iccad19/iccad19-6/train/", ratioleft=0, ratioright=0.01)
        tester = ModelTester(model, dataset,
                             batch_size=64,
                             mode='geometric',
                             device='cuda',
                             save_dir=f"./ablation_results/{tag}/")

        tester.test()