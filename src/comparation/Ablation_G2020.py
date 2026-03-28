# Ablation Studies 使用代码。
# Ablation_G2020.py
# by Lawrence Leung 2025.3.26

import os
import torch
from Test import ModelTesterGeng2020
from train_branch import DatasetForTestNew
import torch.nn as nn
from backbone import Geng20Backbone, Geng20Br2, Geng20CWA, Geng20SWA, Geng20Inception

# ========== 工具模块：恒等 Inception 与 Attention 替代 ==========
class IdentityInception(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

class IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# ========== 1. 移除某一阶段（第5阶段） ==========
class Geng20Ablation_RemoveStage5(Geng20Backbone):
    def __init__(self):
        super().__init__()
        self.i5 = IdentityInception()
        self.c5 = IdentityAttention()
        self.s5 = IdentityAttention()

# ========== 2. 移除所有 Attention（仅保留 Inception） ==========
class Geng20Ablation_NoAttention(Geng20Backbone):
    def __init__(self):
        super().__init__()
        for i in range(1, 6):
            setattr(self, f"c{i}", IdentityAttention())
            setattr(self, f"s{i}", IdentityAttention())

# ========== 3. 仅保留前 3 层（简化版网络） ==========
class Geng20Ablation_First3LayersOnly(Geng20Backbone):
    def __init__(self):
        super().__init__()
        self.i4 = IdentityInception()
        self.c4 = IdentityAttention()
        self.s4 = IdentityAttention()
        self.i5 = IdentityInception()
        self.c5 = IdentityAttention()
        self.s5 = IdentityAttention()

# ========== 4. 替换最后的 fc 为浅层结构（兼容参数） ============
class Geng20Ablation_ShallowFC(Geng20Backbone):
    def __init__(self):
        super().__init__()
        # 替换 fc 为恒等映射（保留 Linear(2048, 1024)）
        linear = self.fc[0]
        with torch.no_grad():
            in_dim = linear.in_features
            out_dim = linear.out_features
            identity_w = torch.eye(out_dim, in_dim)
            linear.weight.copy_(identity_w[:out_dim])
            linear.bias.zero_()
        self.fc[1] = nn.Identity()  # 去除非线性

# ========== 5. 替换 Br2 分类器为线性映射（兼容加载） ============
class Geng20Br2Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 250),  # 原始结构
            nn.ReLU(),
            nn.Linear(250, 2)
        )
        # 将中间层权重变成恒等映射（模拟 shallow）
        with torch.no_grad():
            lin0 = self.fc[0]
            lin0.weight.zero_()
            lin0.bias.zero_()
        self.fc[1] = nn.Identity()

    def forward(self, x):
        return self.fc(x)

# ========== 6. 注册所有变体 ==========
ablation_variants_geng20 = {
    # "baseline": (Geng20Backbone, Geng20Br2),
    # "no_stage5": (Geng20Ablation_RemoveStage5, Geng20Br2),
    # "no_attention": (Geng20Ablation_NoAttention, Geng20Br2),
    # "first3_only": (Geng20Ablation_First3LayersOnly, Geng20Br2),
    "shallow_fc": (Geng20Ablation_ShallowFC, Geng20Br2),
    "linear_br2": (Geng20Backbone, Geng20Br2Linear)
}
# 设置数据集路径和模型路径
DATASET_ROOT = "/ai/edallx/Graduate_Project_2025/iccad12/iccad-official/iccad1/test/"
WEIGHT_BACKBONE = "/ai/edallx/Graduate_Project_2025/models/BB_iccad1_lrb1_0.01_bs_128_20250302_062441.pt"
WEIGHT_BR2 = "/ai/edallx/Graduate_Project_2025/models/Br2_iccad2_lrb1_0.01_bs_128_20250303_032131.pt"

SAVE_ROOT = "./ablation_results_geng20"

def main():
    dataset = DatasetForTestNew(DATASET_ROOT, minnum=0, maxnum=0.005)
    print("Dataset loaded!")

    for tag, (BackboneCls, Br2Cls) in ablation_variants_geng20.items():
        print(f"\n=== Running ablation: {tag} ===")

        model1 = BackboneCls()
        model2 = Br2Cls()

        tester = ModelTesterGeng2020(
            model1,
            model2,
            dataset,
            batch_size=64,
            device='cuda',
            save_dir=os.path.join(SAVE_ROOT, tag)
        )

        tester.load_weights_loose(WEIGHT_BACKBONE, WEIGHT_BR2)
        tester.test()

if __name__ == "__main__":
    main()
