# Geng2020 测试，Branch 1 & 2
# test_branch.py
# by Lawrence Leung 2025

# 导入库
# PyTorch 相关库
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
import pandas as pd
import time
from sklearn.metrics import accuracy_score

# 自定义库
import backbone as Geng20BB
from train_branch import ICCAD12DatasetBr2

##### 主程序
if __name__ == '__main__':

    ### -------------------- 0. 超参数 -------------------- ###
    # TODO 所有的超参数在这里！
    batch_size = 20

    train_dir = "/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad6/train"  # 修改为你的训练数据集路径。这里无需修改，Windows 和 Linux 都可以用它。
    test_dir = "/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad6/test"  # 修改为你的测试数据集路径

    bbmodel_dir = "../models/BB_iccad6_lrb1_0.01_bs_64_20250312_031224.pt"
    br2model_dir = "../models/Br2_iccad6_lrb1_0.01_bs_64_20250312_031224.pt"

    ### -------------------------------------------------- ###
    ### 使用 CPU 或 GPU？
    if torch.cuda.is_available():
        print("-- USING GPU --")
        torch.cuda.empty_cache()
    else:
        print("-- USING CPU --")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用 CUDA

    ### -------------------- 1. 读取数据 -------------------- ###
    # 变换：先将图片调整为 64x64，然后转换为 Tensor，并归一化到 [0, 1]
    transform_img = T.Compose([
        T.Resize((64, 64)),  # 将图片调整为 64*64
        T.ToTensor(),
    ])

    ## Branch 2 的 dataset & dataloader。注意 Branch 2 的 dataloader 的 __getitem__ 输出 image, label 二元组。
    try:
        # 在测试过程中，无论是 branch 1 还是 branch 2，都使用 branch 2 的 dataloader，而不是 branch 1 的 dataloader。
        ds_testbr2 = ICCAD12DatasetBr2(root_dir=test_dir, transform=transform_img, isTrain=False, training_set_proportion=0.95)
        # 只取其中 95% 用于训练。
        # 注意这里的 batch size！
        dl_testbr2 = DataLoader(ds_testbr2, batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(e)
    finally:
        print("\r\n--- Dataset Loaded. ---")

    ### -------------------- 2. 定义模型 -------------------- ###
    Geng20BBInst = Geng20BB.Geng20Backbone()      # 定义Backbone 模型，用于训练 Branch 1 的 deep metric learner。
    # Geng20BBInst = Geng20BB.EmptyBackbone()     # 仅用作 debug。
    Geng20BBInst.to(device)     # 别忘了使用 GPU 推理！
    # 备注：BBInst = backbone instance.

    Geng20Br2Inst = Geng20BB.Geng20Br2()    # 定义用于训练 Branch 2 的 hotspot classification 模型
    Geng20Br2Inst.to(device)    # 别忘了使用 GPU 推理！

    ### -------------------- 3. 加载模型参数 -------------------- ###
    try:
        Geng20BBInst.load_state_dict(torch.load(f"../models/{bbmodel_dir}" ,weights_only=True))
    except Exception as e:
        print(e)
    finally:
        print("\r\n--- Backbone Params Loaded. ---")

    try:
        Geng20Br2Inst.load_state_dict(torch.load(f"../models/{br2model_dir}" ,weights_only=True))
    except Exception as e:
        print(e)
    finally:
        print("\r\n--- Branch 2 Params Loaded. ---")

    ### -------------------- 4. 使用模型推理 -------------------- ###
    def predict_regular(net, images):   # image，即每个图片 (batchsize, 128, 128) -> (batchsize, 1024)
        with torch.no_grad():
            result = net.forward(images.to(device))
            # 为什么是 t[0]，因为这里假设一个 t 是 image label 集合，其中 [0] 就是 image。
            # 别忘了让所有的 tensor 都进入 to_device 模式！
        return (result.data)

    def predict_with_sigmoid(net, rd):   # rd = raw data，不是 dataloader，(batchsize, 1024) -> (batchsize, 2)
        with torch.no_grad():
            result = net.forward(rd.to(device))
            # 为什么是 t[0]，因为这里假设一个 t 是 image label 集合，其中 [0] 就是 image。
            # 别忘了让所有的 tensor 都进入 to_device 模式！
        return (torch.sigmoid(result.data))

    def evaluate_model(dl_testbr2, Geng20Br2Inst, Geng20BBInst):
        # 用于记录每个item的结果
        batch_accuracies = []
        batch_false_alarms = []
        batch_preds = []
        batch_labels = []

        # 用于记录总体结果
        total_preds = []
        total_labels = []
        item_numbers_list = []

        # 设置模型为评估模式
        Geng20Br2Inst.eval()
        Geng20BBInst.eval()

        # 记录推理开始时间
        print("\r\n--- Start Timing ---")
        start_time = time.time()  # 开始计时

        # 对 dataloader 中的每一批数据进行迭代
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in dl_testbr2:
                # 将数据移到 GPU 如果有必要
                images, labels = images.to(device), labels.to(device)   # a batch of images, labels

                # 计算模型输出
                probs2 = predict_with_sigmoid(Geng20Br2Inst, predict_regular(Geng20BBInst, images))

                # 获取预测标签 (predictions = 0 or 1)
                preds = torch.argmax(probs2, dim=1)  # 选择概率较大的类别

                # 计算准确率
                accuracy = accuracy_score(labels.cpu(), preds.cpu())
                batch_accuracies.append(accuracy)

                # 计算 false alarm
                false_alarm = ((preds == 1) & (labels == 0)).sum().item()  # label=1 且预测为 label=0
                batch_false_alarms.append(false_alarm)

                # 记录每个item的预测标签和真实标签
                batch_preds.append(preds.cpu().tolist())
                batch_labels.append(labels.cpu().tolist())

                item_numbers_list.append(len(batch_preds[-1]))

                # 记录总体预测标签和真实标签
                total_preds.extend(preds.cpu().tolist())
                total_labels.extend(labels.cpu().tolist())

        # 计算总体准确率和总体假警报
        overall_accuracy = sum(x * y for x, y in zip(batch_accuracies, item_numbers_list)) / sum(item_numbers_list)
        total_false_alarms = sum(batch_false_alarms)

        # 结束计时，计算总推理时间
        end_time = time.time()
        print("\r\n--- End Timing ---")
        inference_time = end_time - start_time  # 计算总推理时间

        # 创建 DataFrame 来保存结果
        item_results_df = pd.DataFrame({
            'Size of this Batch': item_numbers_list,
            'Batch Accuracies': batch_accuracies,
            'Batch False Alarms': batch_false_alarms,
            'Batch Preds': batch_preds,
            'Batch Labels': batch_labels
        })

        overall_results_df = pd.DataFrame({
            'Overall Item Numbers': [sum(item_numbers_list)],    # 小心，标量必须得加上一个方括号，否则报错！
            'Overall Accuracies': [overall_accuracy],
            'Overall False Alarms': [total_false_alarms],
            'Inference Time (sec)': [inference_time]
        })

        # 保存为 CSV 文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        item_results_df.to_csv(f'../test_history/item_results_{timestamp}.csv', index=False)
        overall_results_df.to_csv(f'../test_history/overall_results_{timestamp}.csv', index=False)

        return item_results_df, overall_results_df

    # 注意，branch 2 的输出 (2,)，第0个元素为label为0的概率；而第1个元素为label为1的概率。注意这个顺序！
    # 假设 Geng20Br2Inst 和 Geng20BBInst 是你的神经网络实例，dl_testbr2 是你的 dataloader
    # 调用评估函数

    item_results, overall_results = evaluate_model(dl_testbr2, Geng20Br2Inst, Geng20BBInst)
    print("\r\n--- End Testing ---")
