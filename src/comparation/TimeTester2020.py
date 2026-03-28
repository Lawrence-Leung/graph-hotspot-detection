# 时间测试
# TimeTester2020.py
# by Lawrence Leung 2025.3.26

import torch
import time
import numpy as np
import os
from torch.utils.data import DataLoader
from train_branch import DatasetForTestNew

class TimedTesterGeng2020:
    def __init__(self, model1, model2, dataset, batch_size=32, device='cuda', save_dir='./timed_results'):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def load_weights(self, weight1_path, weight2_path):
        state_dict1 = torch.load(weight1_path, map_location=self.device)
        state_dict2 = torch.load(weight2_path, map_location=self.device)
        self.model1.load_state_dict(state_dict1, strict=False)
        self.model2.load_state_dict(state_dict2, strict=False)
        print("Loaded model weights.")

    def test_with_timing(self):
        self.model1.eval()
        self.model2.eval()
        stage_times = []  # 每个 batch: [backbone_time, br2_time, total_time]

        with torch.no_grad():
            for batch in self.loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)

                torch.cuda.synchronize()
                t0 = time.time()
                middle = self.model1(inputs)
                torch.cuda.synchronize()
                t1 = time.time()
                outputs = self.model2(middle)
                torch.cuda.synchronize()
                t2 = time.time()

                stage_times.append([
                    t1 - t0,  # backbone 时间
                    t2 - t1,  # br2 时间
                    t2 - t0   # 总时间
                ])

                print(f"Batch time (s): backbone={t1 - t0:.4f}, br2={t2 - t1:.4f}, total={t2 - t0:.4f}")

        np.save(os.path.join(self.save_dir, 'inference_timing_breakdown.npy'), np.array(stage_times))
        print("Saved timing results to:", self.save_dir)

if __name__ == "__main__":
    from backbone import Geng20Backbone, Geng20Br2
    from train_branch import DatasetForTestNew

    dataset = DatasetForTestNew("/ai/edallx/Graduate_Project_2025/debug_datasets/iccad_debug_new/test", minnum=0, maxnum=1)
    model1 = Geng20Backbone()
    model2 = Geng20Br2()

    tester = TimedTesterGeng2020(model1, model2, dataset, batch_size=1, device='cuda',
                                 save_dir='./timed_results_geng20')
    tester.load_weights("/ai/edallx/Graduate_Project_2025/models/BB_iccad5_lrb1_0.01_bs_128_20250306_131107.pt",
                        "/ai/edallx/Graduate_Project_2025/models/Br2_iccad5_lrb1_0.01_bs_128_20250306_131107.pt")
    tester.test_with_timing()


    

