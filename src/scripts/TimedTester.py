# 测试用代码，带有计时。
# Test.py
# by Lawrence Leung 2025.3.26

import torch
import time
import numpy as np
import os
from Test import collate_clip_batches
from torch_geometric.loader import DataLoader as GeoDataLoader
from GraphGeneration import ClipGroupedDatasetForTest02
from LithoGNNCore import FinalAggregatorTimed

class TimedModelTester:
    def __init__(self, model, dataset, batch_size=32, device='cuda', save_dir='./timed_results'):
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir
        self.loader = GeoDataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_clip_batches)
        os.makedirs(self.save_dir, exist_ok=True)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {path}")

    def test_with_timing(self):
        self.model.eval()
        all_times = []
        all_total_times = []

        with torch.no_grad():
            for batch in self.loader:
                batch = [data.to(self.device) for data in batch]
                torch.cuda.synchronize()
                start_time = time.time()

                output, timing = self.model(batch)

                torch.cuda.synchronize()
                end_time = time.time()

                all_times.append([
                    timing['small_block_time'],
                    timing['large_block_time'],
                    timing['mlp_time'],
                    timing['total_time']
                ])

                all_total_times.append(end_time - start_time)

                print(
                    f"Batch time breakdown (s): small={timing['small_block_time']:.4f}, large={timing['large_block_time']:.4f}, mlp={timing['mlp_time']:.4f}, total={timing['total_time']:.4f}")

        # 保存为 npy
        np.save(os.path.join(self.save_dir, 'inference_timing_breakdown.npy'), np.array(all_times))
        np.save(os.path.join(self.save_dir, 'wall_clock_total.npy'), np.array(all_total_times))

        print("Saved timing analysis to:", self.save_dir)

if __name__ == "__main__":
    dataset = ClipGroupedDatasetForTest02(root="/ai/edallx/Graduate_Project_2025/iccad19/iccad19-6/train/",
                                          ratioleft=0, ratioright=0.1)
    model = FinalAggregatorTimed()

    tester = TimedModelTester(model, dataset, batch_size=64, device='cuda', save_dir='./timed_results_baseline')
    tester.load_weights("/ai/edallx/Graduate_Project_2025/NewMethods/temp_logs/best_model.pt")
    tester.test_with_timing()