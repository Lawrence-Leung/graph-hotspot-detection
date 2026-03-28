# Geng2020 复现 Branch 1: Triplet loss
# loss01.py
# by Lawrence Leung 2025

# 头文件引用
import torch
from torch import nn
import torch.nn.functional as F

import backbone as Geng20BB

##### Branch 1: Metric Learning, 通过 Triplet loss 实现。
# 在 Geng2020 论文中，Backbone 输出的 feature map 为 (batch, 1024)。
# 这里是 Geng2020 论文的第一个 branch: metric learning。旨在得到一个合适的变换操作。
# TODO: 不能这样做！因为训练过程中使用的不是继承 nn.Module，而是在 StepRunner 这个 class 中处理！
class Geng20TrLoss(nn.Module):  # deprecated, Feb 25th 2025
    def __init__(self, margin = 1.0):       # 注意，这里的 margin 还需要通过论文原文的情形进行修改。
        super(Geng20TrLoss, self).__init__()
        self.M = margin

    def forward(self, anc, pos, neg):
        # 归一化 anchor、positive 和 negative 特征向量
        anc = F.normalize(anc, p=2, dim=1)
        pos = F.normalize(pos, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=1)
        # 计算 anchor 与 positive 间的欧氏距离
        pos_dist = F.pairwise_distance(anc, pos, p = 2)
        # 计算 anchor 与 negative 间的欧式距离
        neg_dist = F.pairwise_distance(anc, neg, p = 2)
        # 计算 triplet loss
        loss = torch.clamp(pos_dist - neg_dist + self.M, min = 0.0)
        return loss.mean()  # 计算平均值，返回

# 引入了 Geng2020 原文中的 Training Strategy：semi-hard examples 的训练方式
class Geng20TrLossWithStrategy(nn.Module):
    def __init__(self, margin=1.0):
        super(Geng20TrLossWithStrategy, self).__init__()
        self.M = margin  # margin 是超参数，用来控制负样本的选择

        # print("Geng20TrLossWithStrategy Working... ")

    def forward(self, anc, pos, neg):
        # 归一化 anchor、positive 和 negative 特征向量
        anc = F.normalize(anc, p=2, dim=1)
        pos = F.normalize(pos, p=2, dim=1)
        neg = F.normalize(neg, p=2, dim=1)
        # 计算 anchor 与 positive 间的欧氏距离
        pos_dist = F.pairwise_distance(anc, pos, p=2)
        # 计算 anchor 与 negative 间的欧氏距离
        neg_dist = F.pairwise_distance(anc, neg, p=2)

        # 选择符合 semi-hard 条件的负样本
        # semi-hard negative:  ||d(a, p)||^2 <= ||d(a, n)||^2 <= margin + ||d(a, n)||^2
        semi_hard_negatives = (pos_dist ** 2 <= neg_dist ** 2) & (neg_dist ** 2 <= pos_dist ** 2 + self.M)

        # print("Semi Hard Negatives", semi_hard_negatives)

        # semi_hard_negatives 是一个布尔条件。

        if semi_hard_negatives.any():  # 检查是否有至少一个 semi-hard negatives
            # 计算 semi-hard negatives 的 triplet loss
            loss = torch.clamp(pos_dist - neg_dist + self.M, min=0.0)
            # 仅对符合 semi-hard 条件的负样本计算损失
            loss = loss * semi_hard_negatives.float()

            return loss.mean()  # 返回符合条件的负样本的损失平均值

        # 如果没有符合 semi-hard 条件的负样本，则回退到默认的 triplet loss
        # 注意此时 loss 还是一个向量，不是标量。
        loss = torch.clamp(pos_dist - neg_dist + self.M, min=0.0)
        # 返回所有符合条件的 loss 的平均值
        return loss.mean()

##### 调试操作
if __name__ == '__main__':
    anchor = torch.randn(32, 128)  # 假设每个样本的特征维度为 128
    positive = torch.randn(32, 128)
    negative = torch.randn(32, 128)

    triplet_loss = Geng20TrLoss(margin=1.0)
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item()}")

    # 为防止 out of memory 问题。仅作 debug 使用！
    del triplet_loss
    torch.cuda.empty_cache()


