# 测试用代码。
# Test.py
# by Lawrence Leung 2025.3.26

import time
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import os
from torch_geometric.data import Batch

import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt

def collate_clip_batches(batch):
    # batch: List[data_list]  (batch_size 个样本，每个样本是 26 个 Data)
    # 转置：将 list of (list of Data) 变为 list of (list of 同位置 Data)
    transposed = list(zip(*batch))  # 长度为 26，每个元素是一个长度为 batch_size 的元组

    # 针对每个位置，将这个位置上所有 clip 的 Data 进行 batch
    batch_list = [Batch.from_data_list(list(graphs_at_same_index)) for graphs_at_same_index in transposed]

    return batch_list

class ModelTester:
    def __init__(self, model, dataset, batch_size=32, mode='torch', device='cuda', save_dir = './my_test_results'):
        """
        :param model: 需要测试的模型
        :param dataset: 测试数据集
        :param batch_size: 批量大小
        :param mode: 'torch' 或 'geometric'
        :param device: 运行设备
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        self.loader = self._init_loader()
        self.savedir = save_dir

    def _init_loader(self):
        if self.mode == 'torch':
            return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        elif self.mode == 'geometric':
            return GeoDataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_clip_batches)
        else:
            raise ValueError("mode 只支持 'torch' 或 'geometric'，或请检查 'geometric' 的 collate_fn 是否正确")

    def load_weights(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        print(f"Loaded model weights from {weight_path}")

    def test(self):
        self.model.eval()

        all_infer_times = []
        all_accuracies = []
        all_false_alarms = []
        all_f1s = []
        all_recalls = []

        with torch.no_grad():
            for batch in self.loader:
                start_time = time.time()

                if self.mode == 'torch':
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                else:  # geometric 模式
                    batch = [data.to(self.device) for data in batch]
                    outputs = self.model(batch)
                    labels = batch[0].y

                infer_time = time.time() - start_time
                all_infer_times.append(infer_time)

                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()
                accuracy = correct / labels.size(0)
                false_alarm = ((preds == 1) & (labels == 0)).sum().item() / labels.size(0)
                recall = ((preds == 1) & (labels == 1)).sum().item() / (((preds == 1) & (labels == 1)).sum().item() + ((preds == 0) & (labels == 0)).sum().item())

                TP = ((preds == 1) & (labels == 1)).sum().item()
                TN = ((preds == 0) & (labels == 0)).sum().item()
                FP = ((preds == 1) & (labels == 0)).sum().item()
                FN = ((preds == 0) & (labels == 1)).sum().item()

                accuracy = (TP + TN) / labels.size(0)
                false_alarm = FP / labels.size(0)
                recall = TP / (TP + FN + 1e-8)
                precision = TP / (TP + FP + 1e-8)
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                # 输出
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

        # self._plot_distributions(infer_times, accuracies, false_alarms, self.savedir)

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
    from LithoGNNCore import FinalAggregatorBatch
    from GraphGeneration import ClipGroupedDataset, ClipGroupedDatasetForTest02

    print("Starting... 1201")

    dataset = ClipGroupedDatasetForTest02(root="/ai/edallx/Graduate_Project_2025/iccad19/iccad19-6-new/",
                                          ratioleft=0.0, ratioright=1.0)
    print("Set Dataset!")
    model = FinalAggregatorBatch()

    tester = ModelTester(model,
                         dataset,
                         batch_size=64,
                         mode='geometric',
                         device='cuda',
                         save_dir=f"./my_test_results_12_51_bs64"
                         )

    tester.load_weights("/ai/edallx/Graduate_Project_2025/NewMethods/temp_logs/best_model.pt")

    tester.test()

    # for i in range(5):
    #     print(f"ROUND {i} :")
    #     dataset = ClipGroupedDatasetForTest02(root="/ai/edallx/Graduate_Project_2025/iccad19/iccad19-6/train/", ratioleft = i * 0.2, ratioright = 0.2 + i * 0.2)
    #     print("Set Dataset!")
    #     model = FinalAggregatorBatch()
    #
    #     tester = ModelTester(model,
    #                          dataset,
    #                          batch_size = 64,
    #                          mode = 'geometric',
    #                          device = 'cuda',
    #                          save_dir=f"./my_test_results_19_{i}_bs64"
    #                          )
    #
    #     tester.load_weights("/ai/edallx/Graduate_Project_2025/NewMethods/temp_logs/best_model.pt")
    #
    #     tester.test()

