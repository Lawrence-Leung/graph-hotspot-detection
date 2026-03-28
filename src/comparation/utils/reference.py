###### 1 准备数据

import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision import datasets

transform_img = T.Compose(
    [T.ToTensor()])

def transform_label(x):
    return torch.tensor([x]).float()

# dataset_train/val
ds_train = datasets.ImageFolder("./eat_pytorch_datasets/cifar2/train/",
            transform = transform_img,target_transform = transform_label)
ds_val = datasets.ImageFolder("./eat_pytorch_datasets/cifar2/test/",
            transform = transform_img,target_transform = transform_label)
print(ds_train.class_to_idx)

dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True)
dl_val = DataLoader(ds_val,batch_size = 50,shuffle = False)

#查看部分样本
from matplotlib import pyplot as plt

plt.figure(figsize=(8,8))
for i in range(9):
    img,label = ds_train[i]
    img = img.permute(1,2,0)
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label.item())
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

# Pytorch的图片默认顺序是 Batch,Channel,Width,Height
for features, labels in dl_train:
    print(features.shape, labels.shape)
    break

##### 2 定义模型

#测试AdaptiveMaxPool2d的效果
pool = nn.AdaptiveMaxPool2d((1,1))
t = torch.randn(10,8,32,32)
pool(t).shape

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


net = Net()
print(net)

import torchkeras
torchkeras.summary(net,input_data = features);

##### 3 训练模型

import os, sys, time
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

import torch
from torch import nn
from copy import deepcopy

def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")

class StepRunner:
    def __init__(self, net, loss_fn,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer

    def step(self, features, labels):
        # loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, features, labels):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(features, labels)

    @torch.no_grad()
    def eval_step(self, features, labels):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(features, labels)

    def __call__(self, features, labels):
        if self.stage == "train":
            return self.train_step(features, labels)
        else:
            return self.eval_step(features, labels)


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics = self.steprunner(*batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, optimizer, loss_fn, metrics_dict,
                train_data, val_data=None,
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min"):
    history = {}

    for epoch in range(1, epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net, stage="train",
                                       loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(net=net, stage="val",
                                         loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

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


import torchmetrics


class Accuracy(torchmetrics.Accuracy):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        super().update(torch.sigmoid(preds), targets.long())

    def compute(self):
        return super().compute()


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
metrics_dict = {"acc": Accuracy(task='binary')}

dfhistory = train_model(net,
                        optimizer,
                        loss_fn,
                        metrics_dict,
                        train_data=dl_train,
                        val_data=dl_val,
                        epochs=10,
                        patience=5,
                        monitor="val_acc",
                        mode="max")

##### 4 评估模型
dfhistory

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory,"loss")

##### 5 使用模型

def predict(net,dl):
    net.eval()
    with torch.no_grad():
        result = nn.Sigmoid()(torch.cat([net.forward(t[0]) for t in dl]))
    return(result.data)

#预测概率
y_pred_probs = predict(net,dl_val)
y_pred_probs

#预测类别
y_pred = torch.where(y_pred_probs>0.5,
        torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))
y_pred

##### 6 保存模型

print(net.state_dict().keys())

# 保存模型参数
torch.save(net.state_dict(), "./data/net_parameter.pt")

net_clone = Net()
net_clone.load_state_dict(torch.load("./data/net_parameter.pt",weights_only=True))

predict(net_clone,dl_val)

################ 无用代码

# ICCAD12 的 Branch 1 的新的 dataloader，用于 hotspot 与 nonhotspot 数据的 balanced mini-batch 的生成。
# 弃用，bug 太多了！
# class BalancedDataLoaderForBr1(DataLoader):
#     def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
#         super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
#         # self.dataset = dataset
#         # self.batch_size = batch_size
#         self.shuffle = shuffle
#
#     def __iter__(self):
#         # 获取 hotspot 和 non-hotspot 样本
#         hotspot_images = self.dataset.hotspot_images    # 只适用于 ICCAD12DatasetBr1 class！
#         non_hotspot_images = self.dataset.non_hotspot_images    # 只适用于 ICCAD12DatasetBr1 class！
#
#         # 打乱 hotspot 和 non-hotspot 样本
#         if self.shuffle:
#             random.shuffle(hotspot_images)
#             random.shuffle(non_hotspot_images)
#
#         # 计算最大长度，并填充较小一侧的样本
#         max_len = len(non_hotspot_images) # max(len(hotspot_images), len(non_hotspot_images))
#         while len(hotspot_images) < max_len:
#             hotspot_images.append(random.choice(hotspot_images))  # 从已有数据中随机填充
#         while len(non_hotspot_images) < max_len:
#             non_hotspot_images.append(random.choice(non_hotspot_images))  # 从已有数据中随机填充
#
#         # 每次生成 batch_size 个样本
#         for i in range(0, max_len, self.batch_size // 2):
#             # 选择 batch_size // 2 个 hotspot 样本
#             batch_hotspot = hotspot_images[i:i + self.batch_size // 2]
#             # 选择 batch_size // 2 个 non-hotspot 样本
#             batch_non_hotspot = non_hotspot_images[i:i + self.batch_size // 2]
#
#             # 合并成一个平衡的 mini-batch
#             batch_images = batch_hotspot + batch_non_hotspot
#             random.shuffle(batch_images)
#
#             # 构建 anchor, positive, negative 三元组
#             batch_anchors = []
#             batch_positives = []
#             batch_negatives = []
#
#             for anchor_path in batch_images:
#                 if anchor_path in batch_hotspot:
#                     label = 1  # hotspot
#                     positive_path = random.choice(batch_hotspot)
#                     negative_path = random.choice(batch_non_hotspot)
#                 else:
#                     label = 0  # non-hotspot
#                     positive_path = random.choice(batch_non_hotspot)
#                     negative_path = random.choice(batch_hotspot)
#
#                 anchor_img = Image.open(anchor_path).convert('L')
#                 positive_img = Image.open(positive_path).convert('L')
#                 negative_img = Image.open(negative_path).convert('L')
#
#                 # 应用 transform
#                 if self.dataset.transform:
#                     anchor_img = self.dataset.transform(anchor_img)
#                     positive_img = self.dataset.transform(positive_img)
#                     negative_img = self.dataset.transform(negative_img)
#
#                 batch_anchors.append(anchor_img)
#                 batch_positives.append(positive_img)
#                 batch_negatives.append(negative_img)
#
#             # 将三元组样本转换为 Tensor 并返回
#             batch_anchors = torch.stack(batch_anchors)
#             batch_positives = torch.stack(batch_positives)
#             batch_negatives = torch.stack(batch_negatives)
#
#             yield batch_anchors, batch_positives, batch_negatives


