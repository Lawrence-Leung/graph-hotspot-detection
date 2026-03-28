# 嵌入分析代码
# EmbAnalyzer.py
# by Lawrence Leung 2025.4.1

import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use('Agg')   # 注意！
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import umap
except ImportError:
    umap = None
    print("[Warning] 'umap-learn' is not installed. UMAP mode will not be available.")

class TrainTestImageEmbeddingAnalyzer:
    def __init__(self, train_dir, test_dir, save_dir, dim=10, method='pca', max_images=None):
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.save_dir = Path(save_dir)
        self.dim = dim
        self.method = method.lower()
        self.max_images = max_images
        self.images = []
        self.labels = []
        self.domains = []  # 'train' or 'test'
        self.file_names = []
        self.result = None
        self._load_images()

    def _load_from_folder(self, folder, domain_label):
        all_files = sorted(list(folder.glob("*.png")))
        if self.max_images is not None:
            step = max(1, len(all_files) // self.max_images)
            selected_files = all_files[::step][:self.max_images // 2]  # half from each domain
        else:
            selected_files = all_files

        print(f"[INFO] Loading from {folder} ({domain_label}), total files: {len(selected_files)}")
        for file in tqdm(selected_files, desc=f"Loading {domain_label}"):
            label = 1 if file.stem.startswith("HS") else 0
            img = Image.open(file).convert("L").resize((32, 32))
            img_array = np.asarray(img).flatten()
            self.images.append(img_array)
            self.labels.append(label)
            self.domains.append(domain_label)
            self.file_names.append(file.name)

    def _load_images(self):
        self._load_from_folder(self.train_dir, 'train')
        self._load_from_folder(self.test_dir, 'test')
        self.images = np.stack(self.images)
        self.labels = np.array(self.labels)
        self.domains = np.array(self.domains)
        print(f"[INFO] Total loaded: {len(self.labels)} images")

    def compute_embedding(self):
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        n_samples, n_features = self.images.shape
        n_components = min(self.dim, n_samples, n_features)
        print(f"[INFO] Starting dimensionality reduction using {self.method.upper()} with dim={n_components}")

        if self.method == 'pca':
            reducer = PCA(n_components=n_components)
        elif self.method == 'tsne':
            reducer = TSNE(n_components=n_components, init='pca', random_state=42)
        elif self.method == 'umap':
            if umap is None:
                raise ImportError("UMAP is not installed. Please run: pip install umap-learn")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {self.method}")

        self.result = reducer.fit_transform(self.images)
        print(f"[INFO] Dimensionality reduction completed. Result shape: {self.result.shape}")
        return self.result

    def save_embedding(self):
        os.makedirs(self.save_dir, exist_ok=True)
        df = pd.DataFrame(self.result, columns=[f"{self.method}_{i+1}" for i in range(self.result.shape[1])])
        df["label"] = self.labels
        df["domain"] = self.domains
        df["filename"] = self.file_names
        out_path = self.save_dir / f"{self.method}_embeddings.csv"
        df.to_csv(out_path, index=False)
        print(f"[INFO] Embeddings saved to: {out_path}")

    def plot_embedding(self, extra_text:str):
        def normalize_to_unit_interval(data, axis=0):
            min_val = np.min(data[:, axis])
            max_val = np.max(data[:, axis])
            return 2 * (data[:, axis] - min_val) / (max_val - min_val + 1e-8) - 1

        os.makedirs(self.save_dir, exist_ok=True)
        dim = self.result.shape[1]
        print(f"[INFO] Plotting 2D projections (data normalized to [-1, 1])")
        for i in range(min(dim, 3)):
            for j in range(i + 1, min(dim, 3)):
                plt.figure(figsize=(6, 6))
                for domain, color in zip(['train', 'test'], ['blue', 'red']):
                    idx = np.where(self.domains == domain)
                    x_vals = normalize_to_unit_interval(self.result, i)[idx]
                    y_vals = normalize_to_unit_interval(self.result, j)[idx]
                    plt.scatter(x_vals, y_vals, c=color, alpha=0.5, label=f"{domain.capitalize()} Hotspot")
                plt.xlabel(f"{self.method.upper()}-{i + 1} (norm)")
                plt.ylabel(f"{self.method.upper()}-{j + 1} (norm)")
                plt.title(f"{extra_text} : {self.method.upper()}-{i + 1} vs {self.method.upper()}-{j + 1}")
                plt.legend()
                plt.grid(True)
                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                fig_path = self.save_dir / f"{self.method}_{i + 1}_{j + 1}_norm.png"
                plt.savefig(fig_path)
                plt.close()
                print(f"[INFO] Saved normalized plot: {fig_path}")

    def wasserstein_scores(self):
        scores = []
        print("[INFO] Calculating Wasserstein distances between train/test hotspot distributions")
        for i in tqdm(range(self.result.shape[1]), desc="Computing Wasserstein"):
            test_vals = self.result[(self.labels == 1) & (self.domains == 'test'), i]
            train_vals = self.result[(self.labels == 1) & (self.domains == 'train'), i]
            if len(test_vals) > 0 and len(train_vals) > 0:
                wd = wasserstein_distance(train_vals, test_vals)
                scores.append(wd)
            else:
                scores.append(np.nan)
        return scores

# 主程序
if __name__ == "__main__":
    analyzer = TrainTestImageEmbeddingAnalyzer(
        train_dir="/ai/edallx/Graduate_Project_2025/iccad12/iccad-official/iccad1/test",  # 替换为你的图像路径
        test_dir="/ai/edallx/Graduate_Project_2025/iccad12/iccad-official/iccad1/train",
        # train_dir = "/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad7/train",
        # test_dir = "/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad7/test",
        save_dir="./EmbAnalyzerNew/iccad19-total",  # 替换为你的保存路径
        dim=3,
        method='pca',  # 或 'tsne', 'umap'
        max_images = 2000
    )

    analyzer.compute_embedding()
    analyzer.save_embedding()
    analyzer.plot_embedding("ICCAD'19")
    scores = analyzer.wasserstein_scores()

    for i, s in enumerate(scores):
        print(f"Dimension {i + 1}: Wasserstein Distance = {s:.4f}")