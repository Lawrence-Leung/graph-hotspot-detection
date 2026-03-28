# 测试用代码。
# Test.py
# by Lawrence Leung 2025.3.26

import time
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt

class ModelTesterGeng2020:
    def __init__(self, model1, model2, dataset, batch_size=32, device='cuda', save_dir = './my_test_results'):
        """
        :param model: 需要测试的模型
        :param dataset: 测试数据集
        :param batch_size: 批量大小
        :param mode: 'torch' 或 'geometric'
        :param device: 运行设备
        """
        self.model1 = model1.to(device) # branch 1 模型
        self.model2 = model2.to(device) # branch 2 模型
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.loader = self._init_loader()
        self.savedir = save_dir

    def _init_loader(self):
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def load_weights(self, weight1_path, weight2_path):
        self.model1.load_state_dict(torch.load(weight1_path, map_location=self.device, weights_only=True))
        self.model2.load_state_dict(torch.load(weight2_path, map_location=self.device, weights_only=True))
        print(f"Loaded model weights from {weight1_path}, {weight2_path}")

    def load_weights_loose(self, weight1_path, weight2_path):
        self.model1.load_state_dict(torch.load(weight1_path, map_location=self.device, weights_only=True), strict=False)
        self.model2.load_state_dict(torch.load(weight2_path, map_location=self.device, weights_only=True), strict=False)
        print(f"Loaded model weights from {weight1_path}, {weight2_path}")

    def test(self):
        self.model1.eval()
        self.model2.eval()

        all_infer_times = []
        all_accuracies = []
        all_false_alarms = []
        all_f1s = []
        all_recalls = []

        with torch.no_grad():
            for batch in self.loader:
                start_time = time.time()

                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                middlestate = self.model1(inputs)
                outputs = self.model2(middlestate)

                infer_time = time.time() - start_time
                all_infer_times.append(infer_time)

                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()

                TP = ((preds == 1) & (labels == 1)).sum().item()
                TN = ((preds == 0) & (labels == 0)).sum().item()
                FP = ((preds == 1) & (labels == 0)).sum().item()
                FN = ((preds == 0) & (labels == 1)).sum().item()

                accuracy = (TP + TN) / labels.size(0)
                false_alarm = FP / labels.size(0)
                recall = TP / (TP + FN + 1e-8)
                precision = TP / (TP + FP + 1e-8)
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                all_accuracies.append(accuracy)
                all_false_alarms.append(false_alarm)
                all_recalls.append(recall)
                all_f1s.append(f1_score)
                print(f"Batch Inference Time: {infer_time:.4f}s | Accuracy: {accuracy:.4f} | False Alarm: {false_alarm:.4f} | Recall: {recall:.4f} | F1-score: {f1_score:.4f}")

        self._summary(all_infer_times, all_accuracies, all_false_alarms, all_recalls, all_f1s)

    def _summary(self, infer_times, accuracies, false_alarms, recalls, f1s):
        import os
        os.makedirs(self.savedir, exist_ok=True)

        summary_text = (
            f"===== Summary =====\n"
            f"Average inference time per batch: {np.mean(infer_times):.4f}s\n"
            f"Average accuracy: {np.mean(accuracies):.4f}\n"
            f"Average false alarm: {np.mean(false_alarms):.4f}\n"
            f"Average Recall: {np.mean(recalls):.4f}\n"
            f"Average F1-score: {np.mean(f1s):.4f}\n"
            f"\nFrequency Distributions:\n"
            f"Inference time distribution: {np.histogram(infer_times, bins=25)}\n"
            f"Accuracy distribution: {np.histogram(accuracies, bins=25)}\n"
            f"False alarm distribution: {np.histogram(false_alarms, bins=25)}\n"
            f"Recall distribution: {np.histogram(recalls, bins=25)}\n"
        )

        print(summary_text)

        summary_array = np.vstack([
            infer_times,
            accuracies,
            false_alarms,
            recalls,
            f1s
        ]).T  # shape: (num_batches, 5)

        np.save(os.path.join(self.savedir, 'batch_metrics.npy'), summary_array)

        with open(os.path.join(self.savedir, 'summary.txt'), 'w') as f:
            f.write(summary_text)

        # self._plot_distributions(infer_times, accuracies, false_alarms)

    def _plot_distributions(self, infer_times, accuracies, false_alarms):
        plt.figure()
        plt.hist(infer_times, bins=5)
        plt.title("Inference Time Distribution")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.savedir, "inference_time_distribution.png"))
        plt.close()

        plt.figure()
        plt.hist(accuracies, bins=5)
        plt.title("Accuracy Distribution")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.savedir, "accuracy_distribution.png"))
        plt.close()

        plt.figure()
        plt.hist(false_alarms, bins=5)
        plt.title("False Alarm Distribution")
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(self.savedir, "false_alarm_distribution.png"))
        plt.close()

if __name__ == "__main__":
    # 示例使用:
    # 假设你有一个 torch 模型 `cnn_model` 和一个 torch 数据集 `test_dataset`
    # 或者一个 torch_geometric 模型 `gnn_model` 和 `geo_test_dataset`

    # from your_model import cnn_model, test_dataset  # torch
    # tester = ModelTester(cnn_model, test_dataset, batch_size=64, mode='torch', device='cuda')

    # from your_gnn_model import gnn_model, geo_test_dataset  # torch_geometric
    # tester = ModelTester(gnn_model, geo_test_dataset, batch_size=32, mode='geometric', device='cuda')

    # tester.test()

    # ------------------------------------------------------
    # 导入类
    import backbone as Geng20BB
    from train_branch import DatasetForTestNew

    print("Starting...")

    dataset = DatasetForTestNew("/ai/edallx/Graduate_Project_2025/iccad12/iccad-official/iccad5/train/",
                                          minnum=0, maxnum=0.005)
    print("Set Dataset!")
    model1 = Geng20BB.Geng20Backbone()
    model2 = Geng20BB.Geng20Br2()

    tester = ModelTesterGeng2020(model1,
                         model2,
                         dataset,
                         batch_size=64,
                         device='cuda',
                         save_dir=f"./my_test_results_12_1_bs64"
                         )

    tester.load_weights("/ai/edallx/Graduate_Project_2025/models/BB_iccad5_lrb1_0.01_bs_128_20250306_131107.pt",
                        "/ai/edallx/Graduate_Project_2025/models/Br2_iccad5_lrb1_0.01_bs_128_20250306_131107.pt")

    tester.test()

    # for i in range(5):
    #     print(f"ROUND {i}:")
    #     dataset = DatasetForTestNew("/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad6/train", minnum = i * 0.2, maxnum = 0.2 + i * 0.2)    # 仍未调整
    #     print("Set Dataset!")
    #     model1 = Geng20BB.Geng20Backbone()
    #     model2 = Geng20BB.Geng20Br2()
    #
    #     tester = ModelTesterGeng2020(model1,
    #                          model2,
    #                          dataset,
    #                          batch_size = 64,
    #                          device = 'cuda',
    #                          save_dir=f"my_test_results_19_{i}_batch64_G2020"
    #                          )
    #
    #     tester.load_weights("/ai/edallx/Graduate_Project_2025/models/BB_iccad2_lrb1_0.01_bs_128_20250303_032131.pt",
    #                         "/ai/edallx/Graduate_Project_2025/models/Br2_iccad2_lrb1_0.01_bs_128_20250303_032131.pt")
    #
    #     tester.test()

