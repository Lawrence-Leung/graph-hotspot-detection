# Geng2020 训练，Branch 1 & 2
# train_branch.py
# by Lawrence Leung 2025

# 导入库
# PyTorch 相关库
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from PIL import Image
import random

# 系统相关库
import os, sys, time
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt


# 自定义库
import backbone as Geng20BB
import loss01 as Geng20Loss01

##### 1 准备数据
# ICCAD12 数据集 (Branch 1) -> 所使用的是 triplet loss，因此 Dataloader 要用 anchor, pos, neg 三元组。
class ICCAD12DatasetBr1(Dataset):
    def __init__(self, root_dir, transform=None, isTrain=True, training_set_proportion=0.7):
        """
        初始化数据集，加载图片路径和标签
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        :param isTrain：若为真则为训练集 70%，假为测试集 30%。
        :param training_set_proportion：训练集占总集合的比例，0到1间的小数。
        """
        self.root_dir = root_dir
        self.transform = transform  # 图像变换方法
        self.hotspot_images = []
        self.non_hotspot_images = []

        # 检查 root_dir 本身是否包含图片文件
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                # 检查文件是否是 .png 格式
                if filename.lower().endswith('.png'):
                    if filename.startswith('HS'):
                        self.hotspot_images.append(file_path)
                    else:  # 包括 "NNHS" 这样的文件
                        self.non_hotspot_images.append(file_path)

        # 遍历子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    # 检查文件是否是 .png 格式
                    if filename.lower().endswith('.png'):
                        if filename.startswith('HS'):
                            self.hotspot_images.append(file_path)
                        else:  # 包括 "NNHS" 这样的文件
                            self.non_hotspot_images.append(file_path)

        # 计算训练集和测试集的划分
        if isTrain:
            # 训练集：70%
            self.hotspot_images = self.hotspot_images[:int(training_set_proportion * len(self.hotspot_images))]
            self.non_hotspot_images = self.non_hotspot_images[:int(training_set_proportion * len(self.non_hotspot_images))]
        else:
            # 测试集：30%
            self.hotspot_images = self.hotspot_images[int(training_set_proportion * len(self.hotspot_images)):]
            self.non_hotspot_images = self.non_hotspot_images[int(training_set_proportion * len(self.non_hotspot_images)):]

        # 填充较短的一侧
        max_len = max(len(self.hotspot_images), len(self.non_hotspot_images))
        while len(self.hotspot_images) < max_len:
            self.hotspot_images.append(random.choice(self.hotspot_images))  # 从已有数据中随机填充
        while len(self.non_hotspot_images) < max_len:
            self.non_hotspot_images.append(random.choice(self.non_hotspot_images))  # 从已有数据中随机填充

        self.hotspot_images = self.hotspot_images[:min(len(self.hotspot_images), 22400)]
        self.non_hotspot_images = self.non_hotspot_images[:min(len(self.non_hotspot_images), 9600)]

    def __len__(self):  # 返回数据集的长度
        return len(self.hotspot_images) + len(self.non_hotspot_images)

    def __getitem__(self, idx): # 获得一个 item，根据索引 (idx)
        # 获取 anchor 图像 (可以随机选择 hotspot 或 non-hotspot)
        if idx < len(self.hotspot_images):
            anchor_path = self.hotspot_images[idx]
            label = 1  # 热点
        else:
            anchor_path = self.non_hotspot_images[idx - len(self.hotspot_images)]
            label = 0  # 非热点

        # 加载图像
        anchor_img = Image.open(anchor_path).convert('L')  # 单通道图像
        if self.transform:  # 如果存在图像变换
            anchor_img = self.transform(anchor_img)

        # 根据标签选择 positive 或 negative 图像
        if label == 1:  # 如果 anchor 是热点，选择另一个热点作为 positive，非热点作为 negative
            positive_path = random.choice(self.hotspot_images)
            negative_path = random.choice(self.non_hotspot_images)
        else:  # 如果 anchor 是非热点，选择另一个非热点作为 positive，热点作为 negative
            positive_path = random.choice(self.non_hotspot_images)
            negative_path = random.choice(self.hotspot_images)

        # 加载 positive 和 negative 图像
        positive_img = Image.open(positive_path).convert('L')
        negative_img = Image.open(negative_path).convert('L')

        if self.transform:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img

# ICCAD12 数据集 (Branch 2) -> 所使用的是 cross-entropy loss，因此使用正常的 Dataloader 即可。
# TODO: 标签为 0 或 1，分别代表非热点（non-hotspot）和热点（hotspot）。
class ICCAD12DatasetBr2(Dataset):
    def __init__(self, root_dir, transform=None, isTrain=True, training_set_proportion=0.7):
        """
        初始化数据集，加载图片路径和标签
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        :param isTrain：若为真则为训练集 70%，假为测试集 30%。
        :param training_set_proportion：训练集占总集合的比例，0到1间的小数。
        """
        self.root_dir = root_dir
        self.transform = transform  # 图像变换方法
        self.hotspot_images = []
        self.non_hotspot_images = []

        # 检查 root_dir 本身是否包含图片文件
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                # 检查文件是否是 .png 格式
                if filename.lower().endswith('.png'):
                    if filename.startswith('HS'):
                        self.hotspot_images.append(file_path)
                    else:  # 包括 "NNHS" 这样的文件
                        self.non_hotspot_images.append(file_path)

        # 遍历子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    # 检查文件是否是 .png 格式
                    if filename.lower().endswith('.png'):
                        if filename.startswith('HS'):
                            self.hotspot_images.append(file_path)
                        else:  # 包括 "NNHS" 这样的文件
                            self.non_hotspot_images.append(file_path)

        # 计算训练集和测试集的划分
        if isTrain:
            # 训练集：70%
            self.hotspot_images = self.hotspot_images[:int(training_set_proportion * len(self.hotspot_images))]
            self.non_hotspot_images = self.non_hotspot_images[:int(training_set_proportion * len(self.non_hotspot_images))]
        else:
            # 测试集：30%
            self.hotspot_images = self.hotspot_images[int(training_set_proportion * len(self.hotspot_images)):]
            self.non_hotspot_images = self.non_hotspot_images[int(training_set_proportion * len(self.non_hotspot_images)):]

        # 填充较短的一侧
        max_len = max(len(self.hotspot_images), len(self.non_hotspot_images))
        while len(self.hotspot_images) < max_len:
            self.hotspot_images.append(random.choice(self.hotspot_images))  # 从已有数据中随机填充
        while len(self.non_hotspot_images) < max_len:
            self.non_hotspot_images.append(random.choice(self.non_hotspot_images))  # 从已有数据中随机填充

        self.hotspot_images = self.hotspot_images[:min(len(self.hotspot_images), 22400)]
        self.non_hotspot_images = self.non_hotspot_images[:min(len(self.non_hotspot_images), 9600)]

    def __len__(self):      # 返回数据集的长度
        return len(self.hotspot_images) + len(self.non_hotspot_images)

    def __getitem__(self, idx):     # 获得一个 item，根据索引 (idx)
        # 获取图像和标签
        if idx < len(self.hotspot_images):
            img_path = self.hotspot_images[idx]
            label = 1  # 热点
        else:
            img_path = self.non_hotspot_images[idx - len(self.hotspot_images)]
            label = 0  # 非热点

        # 加载图像
        img = Image.open(img_path).convert('L')  # 单通道图像
        if self.transform:  # 如果存在图像变换
            img = self.transform(img)

        return img, label   # 返回图像、标签(是hotspot或否)

class DatasetForTest(Dataset):
    def __init__(self, root_dir):

        """
        初始化数据集，加载图片路径和标签
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        :param isTrain：若为真则为训练集 70%，假为测试集 30%。
        :param training_set_proportion：训练集占总集合的比例，0到1间的小数。
        """
        self.root_dir = root_dir
        self.transform = T.Compose([
        T.Resize((64, 64)),  # 将图片调整为 64*64
        T.ToTensor(),])  # 图像变换方法
        self.hotspot_images = []
        self.non_hotspot_images = []

        # 检查 root_dir 本身是否包含图片文件
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                # 检查文件是否是 .png 格式
                if filename.lower().endswith('.png'):
                    if filename.startswith('HS'):
                        self.hotspot_images.append(file_path)
                    else:  # 包括 "NNHS" 这样的文件
                        self.non_hotspot_images.append(file_path)

        # 遍历子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    # 检查文件是否是 .png 格式
                    if filename.lower().endswith('.png'):
                        if filename.startswith('HS'):
                            self.hotspot_images.append(file_path)
                        else:  # 包括 "NNHS" 这样的文件
                            self.non_hotspot_images.append(file_path)

        # 填充较短的一侧
        max_len = max(len(self.hotspot_images), len(self.non_hotspot_images))
        while len(self.hotspot_images) < max_len:
            self.hotspot_images.append(random.choice(self.hotspot_images))  # 从已有数据中随机填充
        while len(self.non_hotspot_images) < max_len:
            self.non_hotspot_images.append(random.choice(self.non_hotspot_images))  # 从已有数据中随机填充

        self.hotspot_images = self.hotspot_images[:min(len(self.hotspot_images), 22400)]
        self.non_hotspot_images = self.non_hotspot_images[:min(len(self.non_hotspot_images), 9600)]

    def __len__(self):      # 返回数据集的长度
        return len(self.hotspot_images) + len(self.non_hotspot_images)

    def __getitem__(self, idx):     # 获得一个 item，根据索引 (idx)
        # 获取图像和标签
        if idx < len(self.hotspot_images):
            img_path = self.hotspot_images[idx]
            label = 1  # 热点
        else:
            img_path = self.non_hotspot_images[idx - len(self.hotspot_images)]
            label = 0  # 非热点

        # 加载图像
        img = Image.open(img_path).convert('L')  # 单通道图像
        if self.transform:  # 如果存在图像变换
            img = self.transform(img)

        return img, label   # 返回图像、标签(是hotspot或否)

class DatasetForTestNew(Dataset):
    def __init__(self, root_dir, minnum = 0.0, maxnum = 1.0):

        """
        初始化数据集，加载图片路径和标签
        :param root_dir: 数据集根目录
        :param transform: 图像变换
        :param isTrain：若为真则为训练集 70%，假为测试集 30%。
        :param training_set_proportion：训练集占总集合的比例，0到1间的小数。
        """
        self.root_dir = root_dir
        self.transform = T.Compose([
        T.Resize((64, 64)),  # 将图片调整为 64*64
        T.ToTensor(),])  # 图像变换方法
        self.hotspot_images = []
        self.non_hotspot_images = []
        if minnum < maxnum and minnum >= 0.0 and maxnum <= 1.0:
            self.minnum = minnum
            self.maxnum = maxnum
        else:
            self.minnum = 0.0
            self.maxnum = 1.0

        # 检查 root_dir 本身是否包含图片文件
        for filename in os.listdir(root_dir):
            file_path = os.path.join(root_dir, filename)
            if os.path.isfile(file_path):
                # 检查文件是否是 .png 格式
                if filename.lower().endswith('.png'):
                    if filename.startswith('HS'):
                        self.hotspot_images.append(file_path)
                    else:  # 包括 "NNHS" 这样的文件
                        self.non_hotspot_images.append(file_path)

        # 遍历子文件夹
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, filename)
                    # 检查文件是否是 .png 格式
                    if filename.lower().endswith('.png'):
                        if filename.startswith('HS'):
                            self.hotspot_images.append(file_path)
                        else:  # 包括 "NNHS" 这样的文件
                            self.non_hotspot_images.append(file_path)

        # 填充较短的一侧
        max_len = max(len(self.hotspot_images), len(self.non_hotspot_images))
        while len(self.hotspot_images) < max_len:
            self.hotspot_images.append(random.choice(self.hotspot_images))  # 从已有数据中随机填充
        while len(self.non_hotspot_images) < max_len:
            self.non_hotspot_images.append(random.choice(self.non_hotspot_images))  # 从已有数据中随机填充

        self.hotspot_images = self.hotspot_images[int(self.minnum * len(self.hotspot_images)) : int(self.maxnum * len(self.hotspot_images))]
        self.non_hotspot_images = self.non_hotspot_images[
                              int(self.minnum * len(self.non_hotspot_images)): int(self.maxnum * len(self.non_hotspot_images))]

    def __len__(self):      # 返回数据集的长度
        return len(self.hotspot_images) + len(self.non_hotspot_images)

    def __getitem__(self, idx):     # 获得一个 item，根据索引 (idx)
        # 获取图像和标签
        if idx < len(self.hotspot_images):
            img_path = self.hotspot_images[idx]
            label = 1  # 热点
        else:
            img_path = self.non_hotspot_images[idx - len(self.hotspot_images)]
            label = 0  # 非热点

        # 加载图像
        img = Image.open(img_path).convert('L')  # 单通道图像
        if self.transform:  # 如果存在图像变换
            img = self.transform(img)

        return img, label   # 返回图像、标签(是hotspot或否)

# 测试脚本（只用于查看脚本, Branch 1 & 2）
def Stage1Debug(is_branch2 = False):    # 若为 Branch 1，False；否则为 True。
    # 变换：将图片转为 Tensor，并归一化到 [0, 1]
    transform_img = T.Compose([
        T.Resize((128, 128)),  # 将图片调整为 128x128
        T.ToTensor(),
    ])

    # 加载训练和测试数据集，假设以 iccad1 作为例子。
    train_dir = "../iccad-official/iccad_debug/train"  # 修改为你的训练数据集路径。这里无需修改，Windows 和 Linux 都可以用它。
    # test_dir = "../iccad-official/iccad_debug/test"  # 修改为你的测试数据集路径

    if is_branch2 == False: # Branch 1
        ds_train = ICCAD12DatasetBr1(root_dir=train_dir, transform=transform_img)
        # ds_test = ICCAD12DatasetBr1(root_dir=test_dir, transform=transform_img)   # 用不上。

        # 使用 DataLoader 加载数据。注意这里的 batch size！
        dl_train = DataLoader(ds_train, batch_size=10, shuffle=True)
        # dl_test = DataLoader(ds_test, batch_size=10, shuffle=False)   # 用不上。

        # 查看部分样本
        plt.figure(figsize=(8, 8))
        for i, (anchor, positive, negative) in enumerate(dl_train):  # 迭代器
            if i == 3:  # 只显示三个 batch
                break
            plt.subplot(3, 3, i * 3 + 1)
            plt.imshow(anchor[0].permute(1, 2, 0).numpy(), cmap='gray')
            plt.title("Anchor")
            plt.subplot(3, 3, i * 3 + 2)
            plt.imshow(positive[0].permute(1, 2, 0).numpy(), cmap='gray')
            plt.title("Positive")
            plt.subplot(3, 3, i * 3 + 3)
            plt.imshow(negative[0].permute(1, 2, 0).numpy(), cmap='gray')
            plt.title("Negative")
        plt.show()

        # 查看 DataLoader 输出的形状
        for anchor, positive, negative in dl_train:
            print(
                f"Anchor shape: {anchor.shape}, \nPositive shape: {positive.shape}, \nNegative shape: {negative.shape}")
            break

    else:       # Branch 2
        ds_train = ICCAD12DatasetBr2(root_dir=train_dir, transform=transform_img)
        # ds_test = ICCAD12DatasetBr2(root_dir=test_dir, transform=transform_img) # 用不上。

        # 使用 DataLoader 加载数据。注意这里的 batch size！
        dl_train = DataLoader(ds_train, batch_size=10, shuffle=True)
        # dl_test = DataLoader(ds_test, batch_size=10, shuffle=False)   # 用不上。

        # 查看部分样本
        # plt.figure(figsize=(8, 8))
        # for i, (img, label) in enumerate(dl_train):  # 迭代器
        #     if i == 3:  # 只显示三个 batch
        #         break
        #     plt.subplot(3, 3, i * 3 + 1)
        #     plt.imshow(img[0].permute(1, 2, 0).numpy(), cmap='gray')
        #     plt.title(f"Label: {label[0].item()}")
        # plt.show()

        # 查看 DataLoader 输出的形状
        for img, label in dl_train:
            print(f"Image shape: {img.shape}, Label shape: {label.shape}, Labels: {label}")
            break

##### 2 模型定义
# 放在 __main__ 中。不赘述。

##### 3 训练模型
# 打印日志
def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")

#### 3.1 Branch 1 训练类与方法：
# StepRunner 类 (Branch 1)
class StepRunnerBr1:
    def __init__(self, net, loss_fn, stage="train", optimizer=None):
        self.net, self.loss_fn, self.stage = net, loss_fn, stage
        self.optimizer = optimizer

    def step(self, anchor, positive, negative): # 一个 Br1 dataloader 的 __getitem__ 有 anchor, pos, neg 三元组。
        # 获取网络输出（这是一个 1024 长度的向量）
        anchor_out = self.net(anchor)
        positive_out = self.net(positive)
        negative_out = self.net(negative)

        # 计算 Triplet Loss
        loss = self.loss_fn(anchor_out, positive_out, negative_out)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()  # 别忘了清除梯度！

        return loss.item()	# 注意，这里相比 reference.py 少了 metrics_dict，而只有 loss 的数值在这里出现！

    def train_step(self, anchor, positive, negative):   # 模式 1
        self.net.train()  # 训练模式
        return self.step(anchor, positive, negative)

    @torch.no_grad()
    def eval_step(self, anchor, positive, negative):    # 模式 2
        self.net.eval()  # 预测模式
        return self.step(anchor, positive, negative)

    def __call__(self, anchor, positive, negative):     # 调用使用。
        if self.stage == "train":
            return self.train_step(anchor, positive, negative)
        else:
            return self.eval_step(anchor, positive, negative)

# EpochRunner 类 (Branch 1)
class EpochRunnerBr1:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):	# 少了 step_log。
        total_loss, step = 0, 0

        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 别忘了将数据移动到 GPU 上！

        epoch_loss = 0.0    # 为了避免 bug 出现。

        for i, batch in loop:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            loss = self.steprunner(anchor, positive, negative)
            total_loss += loss
            step += 1
            epoch_loss = total_loss / step
            loop.set_postfix({self.stage + "_loss": epoch_loss})

        return {self.stage + "_loss": epoch_loss}

# train_model 函数 (Branch 1)
def train_modelBr1(net, optimizer, loss_fn,
                   train_data, val_data=None,
                   epochs=10, ckpt_path='checkpoint.pt',
                   patience=5, monitor="val_loss", mode="min"):
    history = {}

    for epoch in range(1, epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunnerBr1(net=net, stage="train",
                                          loss_fn=loss_fn, optimizer=optimizer)

        train_epoch_runner = EpochRunnerBr1(train_step_runner)

        train_metrics = train_epoch_runner(train_data)  # 开始运行

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunnerBr1(net=net, stage="val", loss_fn=loss_fn)
            val_epoch_runner = EpochRunnerBr1(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        try:
            torch.save(net.state_dict(), f"../models/BB_in_training.pt")
            print("\r\n-- Geng2020 Backbone Dict Saved. --")
        except Exception as e:
            print(e)
        finally:
            print("\r\n-- Save Models During Training End. --")


        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                              arr_scores[best_score_idx]), file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience), file=sys.stderr)
            break
        net.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return pd.DataFrame(history)

#### 3.2 Branch 2 训练类
# 注意。在训练 Branch 2 时，此时 Backbone 是固定的。
# StepRunner 类 (Branch 2)
class StepRunnerBr2:
    def __init__(self, netbr1, netbr2, loss_fn, stage="train", optimizer=None):
        # 在每次推理中，首先通过 Branch 1 网络推理得到特征，然后传递给 Branch 2 网络进行分类。
        # 因此有 netbr1, netbr2 两个模型。其中 netbr1 不参与训练，而 netbr2 参与训练。
        self.netbr1, self.netbr2, self.loss_fn, self.stage = netbr1, netbr2, loss_fn, stage
        self.optimizer = optimizer

    def step(self, images, labels): # 一个 Br2 的 __getitem__ 有 images, labels 二元组。
        # 使用 Branch 1 的模型提取特征（固定权重）
        with torch.no_grad():  # 这里固定 Branch 1 模型的权重。torch.no_grad 这里是一直被使用的。
            features = self.netbr1(images)  # 此时 netbr1 输出大小为 (batch_size, 1024)

        # 将 Branch 1 提取的特征输入到 Branch 2 模型中
        outputs = self.netbr2(features)  # 这里 netbr2 输出大小为 (batch_size, 2)

        # 计算交叉熵损失
        loss = self.loss_fn(outputs, labels)

        # backward() 进行反向传播
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()  # 清除梯度

        return loss.item()  # 返回当前的损失值

    def train_step(self, images, labels):
        self.netbr2.train()  # 切换到训练模式
        return self.step(images, labels)

    @torch.no_grad()
    def eval_step(self, images, labels):
        self.netbr2.eval()  # 切换到评估模式
        return self.step(images, labels)

    def __call__(self, images, labels):
        if self.stage == "train":
            return self.train_step(images, labels)
        else:
            return self.eval_step(images, labels)

# EpochRunner 类 (Branch 2)
class EpochRunnerBr2:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        epoch_loss = 0.0

        for i, batch in loop:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            loss = self.steprunner(images, labels)
            total_loss += loss
            step += 1
            epoch_loss = total_loss / step
            loop.set_postfix({self.stage + "_loss": epoch_loss})

        return {self.stage + "_loss": epoch_loss}

# train_model 函数 (Branch 2)
def train_modelBr2(netbackbone, netbr2, optimizer, loss_fn,
                   train_data, val_data=None,
                   epochs=10, ckpt_path='checkpoint.pt',
                   patience=5, monitor="val_loss", mode="min"):
    history = {}

    for epoch in range(1, epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunnerBr2(netbr1 = netbackbone, netbr2=netbr2, stage="train", loss_fn=loss_fn, optimizer=optimizer)
        # TODO，被修改
        train_epoch_runner = EpochRunnerBr2(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunnerBr2(netbr1 = netbackbone, netbr2=netbr2, stage="val", loss_fn=loss_fn)
            # TODO，被修改
            val_epoch_runner = EpochRunnerBr2(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        try:
            torch.save(netbr2.state_dict(), f"../models/Br2_in_training.pt")
            print("\r\n-- Geng2020 Branch 2 Dict Saved. --")
        except Exception as e:
            print(e)
        finally:
            print("\r\n-- Save Models During Training End. --")

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        if best_score_idx == len(arr_scores) - 1:
            torch.save(netbr2.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                              arr_scores[best_score_idx]), file=sys.stderr)
        if len(arr_scores) - best_score_idx > patience:
            print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                monitor, patience), file=sys.stderr)
            break
        netbr2.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return pd.DataFrame(history)


##### for debug only，用于测试各个组件的完整性
if __name__ == '__main__':
    ### -------------------- 0. 超参数 -------------------- ###
    # TODO 所有的超参数在这里！
    batch_sizes = [128]  # batch 太大会出现显存溢出问题。
    epochs_br1 = 60
    epochs_br2 = 20
    # lr_rates_br1 = [0.01, 0.005, 0.0075, 0.015, 0.02]
    lr_rates_br1 = [0.01]
    isStrategyForBranch1 = True   # Branch 1 是否采用 Geng2020 论文中的训练策略：balanced mini-batch、semi-hard negatives

    # 加载训练和测试数据集，假设以 iccad1 作为例子。
    train_dir = "/ai/edallx/Graduate_Project_2025/iccad-official/"  # 修改为你的训练数据集路径。这里无需修改，Windows 和 Linux 都可以用它。
    test_dir = "/ai/edallx/Graduate_Project_2025/iccad-official/"  # 修改为你的测试数据集路径
    dirs = ['iccad1', 'iccad2', 'iccad3', 'iccad4', 'iccad5']
    # dirs = ['']
    # usertext = "ICCAD1201"  # 自定义名称标签

    random.seed(100)    # 设置全局随机数种子

    ### -------------------------------------------------- ###
    ### 使用 CPU 或 GPU？
    if torch.cuda.is_available():
        print("-- USING GPU --")
        torch.cuda.empty_cache()
    else:
        print("-- USING CPU --")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用 CUDA

    ### -------------------- 1(a). 读取数据(a) -------------------- ###
    # 变换：先将图片调整为 128x128，然后转换为 Tensor，并归一化到 [0, 1]
    transform_img = T.Compose([
        T.Resize((64, 64)),  # 将图片调整为 64x64
        T.ToTensor(),
    ])

    ### -------------------- 2. 定义模型 -------------------- ###
    Geng20BBInst = Geng20BB.Geng20Backbone()      # 定义Backbone 模型，用于训练 Branch 1 的 deep metric learner。
    # Geng20BBInst = Geng20BB.EmptyBackbone()     # 仅用作 debug。
    Geng20BBInst.to(device)     # 别忘了使用 GPU 推理！
    # 备注：BBInst = backbone instance.

    Geng20Br2Inst = Geng20BB.Geng20Br2()    # 定义用于训练 Branch 2 的 hotspot classification 模型
    Geng20Br2Inst.to(device)    # 别忘了使用 GPU 推理！

    ### -------------------- 3. 训练模型 -------------------- ###
    # 定义网络的特征提取模型，在上一步中已经定义好了 net。
    if isStrategyForBranch1 is not True:
        loss_fn_br1 = Geng20Loss01.Geng20TrLoss(margin=1.0) # loss_fn_br1 使用 TripletLoss
    else:
        loss_fn_br1 = Geng20Loss01.Geng20TrLossWithStrategy(margin=1.0)  # loss_fn_br1 使用 TripletLoss

    loss_fn_br2 = nn.CrossEntropyLoss()  # loss_fn_br2 使用 nn.CrossEntropyLoss

    ### -------------------- 两轮大循环 -------------------- ###
    for eachlr in lr_rates_br1:     # 遍历五个 learning rate
        for eachbs in batch_sizes:  # 遍历所有 batch sizes
            for dis in dirs:    # 遍历五个不同的子目录
                usertext = f"{dis}_lrb1_{eachlr}_bs_{eachbs}" # 区分字符

                ### -------------------- 1(b). 读取数据(b) -------------------- ###
                ## Branch 1 的 dataset & dataloader。注意 Branch 1 的 dataloader 的 __getitem__ 输出 anchor, pos, neg 三元组！
                print('\n' + train_dir + dis)
                ds_trainbr1 = ICCAD12DatasetBr1(root_dir=train_dir+dis, transform=transform_img, isTrain=True)    # TODO: 小心！
                print('\n' + train_dir + dis)
                ds_testbr1 = ICCAD12DatasetBr1(root_dir=test_dir+dis, transform=transform_img, isTrain=False)      # TODO: 小心！
                # 注意这里的 batch size！
                print(len(ds_trainbr1), ',', len(ds_testbr1))
                if isStrategyForBranch1 is not True:
                    dl_trainbr1 = DataLoader(ds_trainbr1, batch_size=eachbs, shuffle=True)
                    dl_testbr1 = DataLoader(ds_testbr1, batch_size=eachbs, shuffle=False)
                else:
                    print('\n' + train_dir + dis)
                    print('\n' + train_dir + dis)
                    # dl_trainbr1 = BalancedDataLoaderForBr1(ds_trainbr1, batch_size=eachbs, shuffle=True)  # 有问题！
                    # dl_testbr1 = BalancedDataLoaderForBr1(ds_testbr1, batch_size=eachbs, shuffle=False)
                    dl_trainbr1 = DataLoader(ds_trainbr1, batch_size=eachbs, shuffle=True)
                    dl_testbr1 = DataLoader(ds_testbr1, batch_size=eachbs, shuffle=False)

                ## Branch 2 的 dataset & dataloader。注意 Branch 2 的 dataloader 的 __getitem__ 输出 image, label 二元组。
                ds_trainbr2 = ICCAD12DatasetBr2(root_dir=train_dir+dis, transform=transform_img, isTrain=True)        # TODO: 小心！
                ds_testbr2 = ICCAD12DatasetBr2(root_dir=test_dir+dis, transform=transform_img, isTrain=False)          # TODO: 小心！

                # 注意这里的 batch size！
                dl_trainbr2 = DataLoader(ds_trainbr2, batch_size=eachbs, shuffle=True)
                dl_testbr2 = DataLoader(ds_testbr2, batch_size=eachbs, shuffle=False)
                ### -------------------- end of 1(b). 读取数据(b) -------------------- ###

                ## Branch 1
                optimizer_br1 = torch.optim.Adam(Geng20BBInst.parameters(), lr=eachlr)
                # 调用训练函数。别忘了输出的 dfhistory_br1, br2 均为 pandas 的 DataFrame 结构！
                print("\r\n-- Geng2020 Branch 1 Training -- \r\n")
                try:
                    dfhistory_br1 = train_modelBr1(Geng20BBInst,
                                               optimizer_br1,
                                               loss_fn_br1,
                                               train_data=dl_trainbr1,
                                               val_data=dl_testbr1,
                                               epochs=epochs_br1,
                                               patience=10,
                                               monitor="val_loss",
                                               mode="min",
                                               )
                except Exception as e:
                    print(e)
                finally:
                    print("\r\n-- Geng2020 Branch 1 Training Finished -- \r\n")

                ## Branch 2
                optimizer_br2 = torch.optim.Adam(Geng20Br2Inst.parameters(), lr=0.01)

                print("\r\n-- Geng2020 Branch 2 Training -- \r\n")
                # 调用训练函数
                try:
                    dfhistory_br2 = train_modelBr2(Geng20BBInst,    # 新增的：因为 Branch 2 需要使用到 Branch 1 的训练好的参数。注意。
                                               Geng20Br2Inst,   # 这个就是对应的需要被训练的模型权重。后续的顺序和 train_modelBr1 一致。
                                               optimizer_br2,
                                               loss_fn_br2,
                                               train_data=dl_trainbr2,
                                               val_data=dl_testbr2,
                                               epochs=epochs_br2,
                                               patience=5,
                                               monitor="val_loss",
                                               mode="min")
                except Exception as e:
                    print(e)
                finally:
                    print("\r\n-- Geng2020 Branch 2 Training Finished -- \r\n")

                ### -------------------- 4. 导出训练数据 -------------------- ###
                timestamp = time.strftime("%Y%m%d_%H%M%S")

                try:
                    # 构造文件名
                    dfhistory_br1_name = f"../training_history/BB_{usertext}_{timestamp}.csv"
                    dfhistory_br2_name = f"../training_history/Br2_{usertext}_{timestamp}.csv"
                    # 导出为 csv 文件
                    dfhistory_br1.to_csv(dfhistory_br1_name, index=False)
                    print(f"\r\n-- Geng2020 Training Backbone History Saved to {dfhistory_br1_name} --")
                    dfhistory_br2.to_csv(dfhistory_br2_name, index=False)
                    print(f"\r\n-- Geng2020 Training Branch 2 History Saved to {dfhistory_br2_name} --")
                except Exception as e:
                    print(e)
                finally:
                    print("\r\n-- Export to CSV End. --")

                # 绘制训练历史图片
                def plot_metric(dfhistory, metric, directoryname):
                    train_metrics = dfhistory["train_" + metric]
                    val_metrics = dfhistory['val_' + metric]
                    epochs = range(1, len(train_metrics) + 1)

                    plt.plot(epochs, train_metrics, 'bo--')
                    plt.plot(epochs, val_metrics, 'ro-')
                    plt.title('Training and validation ' + metric)
                    plt.xlabel("Epochs")
                    plt.ylabel(metric)
                    plt.legend(["train_" + metric, 'val_' + metric])

                    # 获取当前时间戳，并生成文件名
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    file_name = f"{directoryname}{usertext}_{metric}_{timestamp}.png"

                    # 将图表保存为 PNG 图片
                    plt.savefig(file_name)
                    print(f"\r\n Training History has saved to: {file_name}")
                    plt.close()  # 关闭当前图表，释放资源

                # 调用函数进行绘图
                try:
                    plot_metric(dfhistory_br1, "loss", "../training_history/BB_")
                    plot_metric(dfhistory_br2, "loss", "../training_history/Br2_")
                except Exception as e:
                    print(e)
                finally:
                    print("\r\n-- Export to Figures End. --")

                ### -------------------- 5. 使用模型推理 -------------------- ###
                # 在 test_branch.py 中出现。

                ### -------------------- 6. 保存模型 -------------------- ###
                try:
                    torch.save(Geng20BBInst.state_dict(), f"../models/BB_{usertext}_{timestamp}.pt")
                    print("\r\n-- Geng2020 Backbone Dict Saved. --")
                    torch.save(Geng20Br2Inst.state_dict(), f"../models/Br2_{usertext}_{timestamp}.pt")
                    print("\r\n-- Geng2020 Branch 2 Dict Saved. --")
                except Exception as e:
                    print(e)
                finally:
                    print("\r\n-- Save Models End. --")

    print("\r\n-- Congrats! All Ended. --")