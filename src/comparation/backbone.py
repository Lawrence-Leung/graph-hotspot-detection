# Geng2020 复现 模型定义
# backbone.py
# by Lawrence Leung 2025

# 头文件引用
import torch
from torch import nn

##### 1) Geng20 Inception 定义
class Geng20Inception (nn.Module):

    # 构造函数
    def __init__(self, input_size = 64, input_channel_numbers = 32):
        # input_size 默认为 128。这里是输入 feature map 的边长。
        # input_channel_numbers 是输入的 feature map 的层数。
        super(Geng20Inception, self).__init__()
        self.input_size = input_size  # 输入 feature map 的边长
        self.icn = input_channel_numbers # input_channel_numbers 是输入的 feature map 的层数。
        # 以下的维度顺序为：(c, h, w)
        # 输入 -> pooling -> a1 (1, 1200, 1200)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # 输入 -> conv 1*1 -> a2 (16, 1200, 1200)
        self.conv1x1_a = nn.Conv2d(in_channels=self.icn, out_channels=8, kernel_size=1)
        # 输入 -> conv 1*1 -> a3 (16, 1200, 1200)
        self.conv1x1_b = nn.Conv2d(in_channels=self.icn, out_channels=16, kernel_size=1)
        # 输入 -> conv 1*1 -> a4 (16, 1200, 1200)
        self.conv1x1_c = nn.Conv2d(in_channels=self.icn, out_channels=16, kernel_size=1)

        # a1 -> conv 1*1 -> b1
        self.conv1x1_b1 = nn.Conv2d(in_channels=self.icn, out_channels=8, kernel_size=1)
        # a2 -> pooling -> b2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # a3 -> conv 3*3 -> b3
        self.conv3x3_b3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        # a4 -> conv 5*5 -> b4
        self.conv5x5_b4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, padding=2)
        # b3 -> pooling -> c3
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # b4 -> pooling -> c4
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    # 前进方法
    def forward(self, input):
        # a1 -> pooling
        a1 = self.pool1(input)
        # a2 -> conv 1*1
        a2 = self.conv1x1_a(input)
        # a3 -> conv 1*1
        a3 = self.conv1x1_b(input)
        # a4 -> conv 1*1
        a4 = self.conv1x1_c(input)
        # a1 -> conv 1*1 -> b1
        b1 = self.conv1x1_b1(a1)
        # a2 -> pooling -> b2
        b2 = self.pool2(a2)
        # a3 -> conv 3*3 -> b3
        b3 = self.conv3x3_b3(a3)
        # a4 -> conv 5*5 -> b4
        b4 = self.conv5x5_b4(a4)
        # b3 -> pooling -> c3
        c3 = self.pool3(b3)
        # b4 -> pooling -> c4
        c4 = self.pool4(b4)
        # Concatenate b1, b2, c3, c4 along channel axis
        output = torch.cat((b1, b2, c3, c4), dim=1)  # 在通道维度拼接
        return output
        # 最后得到的输出：C = 32, W = H = input_shape = 1200. (input_shape 不变)

##### 2) Geng20 Attention Block 部分
#### 2.1) Channel-wise attention
class Geng20CWA(nn.Module):

    # 构造函数
    def __init__(self, input_size = 64):
        # TODO: input_size 默认为 128。这里是输入 feature map 的边长。
        super(Geng20CWA, self).__init__()
        self.input_size = input_size  # 输入 feature map 的边长
        # Encoder-Decoder 网络（MLP）
        self.fc = nn.Sequential(
            nn.Linear(1, 32),  # 输入 1 维，输出 32 维
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出通道数维度
        )
        # 池化操作
        self.maxpool = nn.MaxPool2d(kernel_size=(self.input_size, self.input_size), stride=(self.input_size, self.input_size))  # 输入: (B, C, H, W) -> 输出: (B, C, 1, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=(self.input_size, self.input_size), stride=(self.input_size, self.input_size))  # 输入: (B, C, H, W) -> 输出: (B, C, 1, 1)

    # 前进方法
    def forward(self, input):
        # 获取输入的形状
        h, w = input.shape[2], input.shape[3]
        # Input -> Maxpool -> a1
        a1 = self.maxpool(input)  # (B, C, 1, 1)
        # Input -> Avepool -> a2
        a2 = self.avgpool(input)  # (B, C, 1, 1)
        # a1 -> ED -> b1
        b1 = self.fc(a1.view(a1.size(0), a1.size(1), -1))  # Flatten and pass through MLP
        b1 = b1.view(b1.size(0), b1.size(1), 1, 1)  # (B, C, 1, 1)
        # a2 -> ED -> b2
        b2 = self.fc(a2.view(a2.size(0), a2.size(1), -1))  # Flatten and pass through MLP
        b2 = b2.view(b2.size(0), b2.size(1), 1, 1)  # (B, C, 1, 1)
        # b1 与 b2 逐元素相加 -> c1
        c1 = torch.sigmoid(b1 + b2)  # (B, C, 1, 1)
        # c1 -> 在 H、W 方向广播操作 -> d1
        d1 = c1.expand(-1, -1, h, w)  # 广播到 (B, C, H, W)
        # d1 与 Input -> 逐元素相乘 -> Output
        output = input * d1  # (B, C, H, W)
        return output

#### 2.2) Spatial-wise attention
class Geng20SWA(nn.Module):

    # 构造函数
    def __init__(self):
    # 注意，channels_size 必须是 2 的倍数！

        super(Geng20SWA, self).__init__()
        # 卷积层：7x7卷积，保持空间大小不变，通过padding补充
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=7, padding=3)  # padding=3 保持 H, W 不变

    # 前进方法
    def forward(self, input):
        # 获取输入的形状
        c = input.shape[1]
        # Input -> Maxpool -> a1
        a1 = torch.max(input, dim=1, keepdim=True)[0]  # (B, 1, H, W) : 对 channels 维度进行 maxpool 操作，得到 1 个通道
        # Input -> Avepool -> a2
        a2 = torch.mean(input, dim=1, keepdim=True)  # (B, 1, H, W) : 对 channels 维度进行 avgpool 操作，得到 1 个通道
        # a1 与 a2 在 channel 维度 concat -> b1
        b1 = torch.cat((a1, a2), dim=1)  # (B, 2, H, W)
        # b1 -> conv7*7 -> c1 -> sigmoid 激活函数
        c1 = torch.sigmoid(self.conv(b1))  # (B, 2, H, W)
        # c1 -> 在 channel 方向广播 -> d1
        d1 = c1.repeat(1, c // 2, 1, 1)  # 广播到 (B, C, H, W)
        # d1 与 Input -> 逐元素相乘 -> Output
        output = input * d1  # (B, C, H, W)
        return output

# 注意：一个完整的 Attention Block，由 Channel-wise 和 Spatial-wise 两者级联而成。

##### 3) Backbone 的组合操作
class Geng20Backbone(nn.Module):
    # 构造函数
    def __init__(self, input_size = 64, output_size = 1024, input_channel_numbers = 32):
        # TODO: input_size 默认为 128。这里是输入 feature map 的边长。output_size 为输出的一维度向量长度。
        super(Geng20Backbone, self).__init__()
        self.input_size = input_size    # 输入 feature map 的边长
        self.output_size = output_size  # 输出的一维度向量长度
        self.icn = input_channel_numbers  # input_channel_numbers 是输入的 feature map 的层数。

        # 已有的模块
        self.i1 = Geng20Inception(input_channel_numbers = 1)  # Inception 模块
        self.c1 = Geng20CWA()  # Channel-wise Attention 模块  (1, 128, 128 -> 32, 128, 128)
        self.s1 = Geng20SWA()  # Spatial-Wise Attention 模块  (1, 128, 128 -> 32, 128, 128)
        self.i2 = Geng20Inception()  # Inception 模块
        self.c2 = Geng20CWA()  # Channel-wise Attention 模块
        self.s2 = Geng20SWA()  # Spatial-Wise Attention 模块
        self.i3 = Geng20Inception()  # Inception 模块
        self.c3 = Geng20CWA()  # Channel-wise Attention 模块
        self.s3 = Geng20SWA()  # Spatial-Wise Attention 模块
        self.i4 = Geng20Inception()  # Inception 模块
        self.c4 = Geng20CWA()  # Channel-wise Attention 模块
        self.s4 = Geng20SWA()  # Spatial-Wise Attention 模块
        self.i5 = Geng20Inception()  # Inception 模块
        self.c5 = Geng20CWA()  # Channel-wise Attention 模块
        self.s5 = Geng20SWA()  # Spatial-Wise Attention 模块 (32, 128, 128)

        # 池化层进行下采样，减少空间维度。注意由于论文原文埋了一个坑，所以这里可能需要自主发挥 :(
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # 自适应池化，将空间维度降低到 8 * 8
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 1024),  # 输入尺寸为 32 * 128 * 128
            nn.ReLU(),
        )

    # 前进方法
    def forward(self, x):   # x 就是 input，同样放在除了 self 之后的第一个参数。
        x = self.i1(x)  # 首先通过 Inception 模块
        x = self.c1(x)  # 然后通过 Channel-wise Attention
        x = self.s1(x)  # 再通过 Spatial-Wise Attention
        x = self.i2(x)  # 首先通过 Inception 模块
        x = self.c2(x)  # 然后通过 Channel-wise Attention
        x = self.s2(x)  # 再通过 Spatial-Wise Attention
        x = self.i3(x)  # 首先通过 Inception 模块
        x = self.c3(x)  # 然后通过 Channel-wise Attention
        x = self.s3(x)  # 再通过 Spatial-Wise Attention
        x = self.i4(x)  # 首先通过 Inception 模块
        x = self.c4(x)  # 然后通过 Channel-wise Attention
        x = self.s4(x)  # 再通过 Spatial-Wise Attention
        x = self.i5(x)  # 首先通过 Inception 模块
        x = self.c5(x)  # 然后通过 Channel-wise Attention
        x = self.s5(x)  # 再通过 Spatial-Wise Attention

        # 下采样，先通过平均池化。这一步不需要任何参数。
        x = self.adaptive_pool(x)   # 这一步到(32, 8, 8)
        # 展平 feature map 为一维向量
        x = x.view(x.size(0), -1)  # 拉平为 (batch_size, 32*8*8=2048)
        x = self.fc(x)  # 最后通过全连接层输出 (2048 -> 1024)
        return x

##### 4) Branch 2，将 Backbone 进一步处理，1024 -> 250 -> 2
class Geng20Br2(nn.Module):
    def __init__(self):
        super(Geng20Br2, self).__init__()
        self.fc = nn.Sequential(    # 仅供 Branch 2 使用
            nn.Linear(1024, 250),  # 输入 1024 维，输出 250 维
            nn.ReLU(),
            nn.Linear(250, 2)  # 输出通道数维度
        )

    def forward(self, x):
        x = self.fc(x)
        return x

##### 5) 仅作 debug 使用
class EmptyBackbone(nn.Module):
    def __init__(self):
        super(EmptyBackbone, self).__init__()

        # 使用一个全连接层将卷积后的结果展平并映射到 (1024,) 的向量
        # 计算展平后的输入大小为 1 * 64 * 64
        self.fc1 = nn.Linear(1 * 64 * 64, 1024)

    def forward(self, x):

        # 展平 (batch_size, channels, height, width) -> (batch_size, channels * height * width)
        x = x.view(x.size(0), -1)

        # 全连接层映射到目标的 1024 维
        x = self.fc1(x)

        return x
##### 调试操作
if __name__ == '__main__':
    # CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Inception 测试专用
    # input_tensor = torch.randn(2, 1, 1200, 1200)    # batch_size, channel, height, weight
    # input_tensor = input_tensor.to(device)
    # model = Geng20Inception(input_size=1200)
    # model.to(device)

    # 2) Channel-wise Attention 测试专用
    # input_tensor = torch.randn(2, 64, 1200, 1200)  # Batch size = 2, Channel = 64, H = W = 1200。
    # 注意：Channel 的数量在实际情况下是可以变动的。
    # input_tensor = input_tensor.to(device)
    # model = Geng20CWA()
    # model.to(device)

    # 3) Spatial-wise Attention 测试专用
    # input_tensor = torch.randn(2, 288, 1200, 1200)  # Batch size = 2, Channel = 288, H = W = 1200
    # 注意：Channel 的数量在实际情况下是可以变动的。
    # input_tensor = input_tensor.to(device)
    # model = Geng20SWA()
    # model.to(device)

    # 4) Backbone 测试专用
    # input_tensor = torch.randn(2, 1, 128, 128)    # batch_size, channel, height, weight
    input_tensor = torch.randn(2, 1024)
    input_tensor = input_tensor.to(device)
    model = Geng20Br2()
    # model = EmptyBackbone()
    model.to(device)

    # 使用 torchkeras 来显示网络结构
    print(model)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    # summary(model)

    # 为了防止 out of memory 问题。仅作 debug 使用！
    del model
    del input_tensor
    torch.cuda.empty_cache()




