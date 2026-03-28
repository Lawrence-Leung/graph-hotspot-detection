# 新方法训练代码！加油！
# Train.py
# by Lawrence Leung 2025.3.20

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import os
import logging

from GraphGeneration import ClipGroupedDataset, collate_clip_batches, ClipGroupedDatasetForTest
from LithoGNNCore import FinalAggregatorBatch

# 训练类
class Trainer:
    def __init__(self, model, train_loader, test_loader, device, save_dir="training_logs", lr=0.001, warmup_epochs=5, total_epochs=120, early_stop_patience=15):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs)

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.early_stop_patience = early_stop_patience

        self.train_losses, self.train_accs, self.test_accs = [], [], []
        self.best_test_acc = 0
        self.epochs_no_improve = 0

        self.checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
        self.start_epoch = 1
        self._try_load_checkpoint()

    def _try_load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            print("Loading checkpoint...")
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.test_accs = checkpoint['test_accs']

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}', leave=False)
        for data_list in pbar:
            data_list = [data.to(self.device) for data in data_list]
            self.optimizer.zero_grad()

            out = self.model(data_list)  # (1, 2)
            label = data_list[0].y.view(-1).to(self.device)

            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        # Warm-up 调整
        if epoch <= self.warmup_epochs:
            warmup_lr = self.optimizer.param_groups[0]['lr'] * (epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            self.scheduler.step()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0

        with torch.no_grad():
            for data_list in loader:
                data_list = [data.to(self.device) for data in data_list]
                out = self.model(data_list)
                pred = out.argmax(dim=1)
                label = data_list[0].y.view(-1).to(self.device)
                correct += int((pred == label).sum())

        return correct / len(loader)

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'test_accs': self.test_accs
        }
        torch.save(checkpoint, self.checkpoint_path)

    def save_logs_and_plot(self):
        log_dict = {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "test_accs": self.test_accs
        }
        with open(os.path.join(self.save_dir, "train_log.json"), "w") as f:
            json.dump(log_dict, f, indent=4)

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, label='Train Acc')
        plt.plot(epochs, self.test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curve.png"))
        # plt.show()

    def run(self):
        for epoch in range(self.start_epoch, self.total_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            train_acc = self.evaluate(self.train_loader)
            test_acc = self.evaluate(self.test_loader)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

            print(f"Epoch {epoch:03d}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

            # early stopping 检查
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.epochs_no_improve = 0
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch)

            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        self.save_logs_and_plot()

# 假设 Trainer 类和模型定义已在上方导入
def setup_logger(save_dir):
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    log_file = os.path.join(save_dir, "training_output.log")

    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# 主函数
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')        # 注意！需要修改！

    train_roots = ["/ai/edallx/Graduate_Project_2025/iccad12/iccad12-01/test"]
    test_roots = ["/ai/edallx/Graduate_Project_2025/iccad12/iccad12-01/test"]

    if len(test_roots) is not len(train_roots):
        print("len(test_roots) is not len(train_roots)! ")

    # 每轮循环
    for i in range(len(train_roots)):

        # 创建数据集和数据加载器
        # train_dataset = ClipGroupedDataset(root="D:\\GradProject\\OAS_Transformer\\dataset")
        # test_dataset = ClipGroupedDataset(root="D:\\GradProject\\OAS_Transformer\\dataset")

        print(f"=== ROUND {i + 1} ===")
        train_dataset = ClipGroupedDatasetForTest(root = train_roots[i], minnum=0.0, maxnum=0.7)
        test_dataset = ClipGroupedDatasetForTest(root = test_roots[i], minnum=0.7, maxnum=1.0)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_clip_batches)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_clip_batches)

        # 实例化模型
        model = FinalAggregatorBatch()

        # 创建保存目录并初始化日志
        save_dir = "temp_logs_1201"
        os.makedirs(save_dir, exist_ok=True)
        logger = setup_logger(save_dir)

        # 创建 Trainer 实例
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            save_dir=save_dir,
            lr=0.01,
            warmup_epochs=5,
            total_epochs=100,
            early_stop_patience=15
        )

        # 将 logger 绑定到 trainer（可在 Trainer 中集成 logger 使用）
        trainer.logger = logger

        # 启动训练
        trainer.run()