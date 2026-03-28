# 批量图生成器。
# GraphGeneration.py
# by Lawrence Leung 2025.3.20
# 注意，这个是更新后的版本，小心 IO 问题！

import os
import glob
import logging
import time
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
from functools import wraps

import pickle
import logging
from pathlib import Path

from GraphGeneration import ClipPartition, GraphGeneration  # 根据你的工程实际修改导入

# 配置 logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 装饰器：失败重试
def retry_on_failure(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"函数 {func.__name__} 出错：{e}，第 {retries+1} 次重试...")
                    time.sleep(delay)
                    retries += 1
            logging.error(f"函数 {func.__name__} 重试 {max_retries} 次仍失败。")
        return wrapper
    return decorator

class BatchGraphGenerator:
    def __init__(self, input_dir: str, output_dir: str, a: int = 1200):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.a = a
        self.cache_file = os.path.join(self.input_dir, "png_list_cache.pkl")
        self.png_files = []

    # 以前的单项处理
    def batch_generate_graphs(self, png_files: List[str]):
        from GraphGeneration import ClipPartition, GraphGeneration  # 根据我的工程实际修改导入

        for image_path in tqdm(png_files, desc="Generating graphs (single batch)"):
            relative_path = os.path.relpath(image_path, self.input_dir)
            output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))
            os.makedirs(output_subdir, exist_ok=True)

            partitioner = ClipPartition(image_path=image_path, a=self.a)
            partitioner.runall()

            generator = GraphGeneration(save_dir=output_subdir)
            generator.runall(partitioner)

            clipid = generator.sc_clipid
            expected_files = [f"{clipid}_GphLst.pt"]

            missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_subdir, f))]
            if missing_files:
                logging.warning(f"Clip {clipid} 缺少以下图文件： {missing_files}")

    @retry_on_failure(max_retries=3, delay=3)
    def _generate_one_clip_deprecated(self, image_path: str):
        from GraphGeneration import ClipPartition, GraphGeneration  # 根据你的工程实际修改导入

        relative_path = os.path.relpath(image_path, self.input_dir)
        output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))

        partitioner = ClipPartition(image_path=image_path, a=self.a)
        partitioner.runall()

        generator = GraphGeneration(save_dir=output_subdir)
        generator.runall(partitioner)

        clipid = generator.sc_clipid
        y = generator.sc_graph_y
        expected_files = [f"{clipid}_GphLst_{y}.pt"]

        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_subdir, f))]
        if missing_files:
            logging.warning(f"Clip {clipid} 缺少以下图文件： {missing_files}")


    def _generate_one_clip(self, image_path: str):
        try:
            
            relative_path = os.path.relpath(image_path, self.input_dir)
            output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))
    
            partitioner = ClipPartition(image_path=image_path, a=self.a)
            partitioner.runall()
    
            generator = GraphGeneration(save_dir=output_subdir)
            generator.runall(partitioner)
    
            clipid = generator.sc_clipid
            y = generator.sc_graph_y
            expected_files = [f"{clipid}_GphLst_{y}.pt"]
    
            missing_files = [f for f in expected_files if not os.path.exists(os.path.join(output_subdir, f))]
            if missing_files:
                logging.warning(f"Clip {clipid} 缺少以下图文件：{missing_files}，跳过。")
                return
            else:
                pass
                # logging.info(f"Clip {clipid} 生成成功。")
        except Exception as e:
            logging.warning(f"生成 clip 时发生异常（跳过）：{image_path}，错误信息：{e}")
            return


    # 新的多进程处理，正式时使用这个。
    # memory_limit_gb：最大使用内存GB数
    def multi_process_generate(self, memory_limit_gb: int = 20):
        logging.info("正在创建输出目录结构...")
        dirs_to_create = set()
        for image_path in self.png_files:
            relative_path = os.path.relpath(image_path, self.input_dir)
            output_subdir = os.path.join(self.output_dir, os.path.dirname(relative_path))
            dirs_to_create.add(output_subdir)
        for d in tqdm(dirs_to_create, desc="Creating directories"):
            os.makedirs(d, exist_ok=True)

        cpu_count = multiprocessing.cpu_count()
        logging.info(f"检测到 {cpu_count} 核心，启动多进程...")

        # 动态调整 worker 数量，防止单机内存不足
        est_mem_per_proc_gb = 0.4
        max_workers = min(cpu_count, int(memory_limit_gb // est_mem_per_proc_gb))
        if max_workers < 1:
            max_workers = 1

        logging.info(f"根据 {memory_limit_gb} GB 限制，最大进程数调整为 {max_workers}。")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._generate_one_clip, image_path) for image_path in self.png_files]

            for future in tqdm(as_completed(futures), total=len(futures), desc="生成图数据"):
                _ = future.result()  # 即便出错也不会影响进度条

        logging.info("✅ 多进程批量生成完成！")

    def load_or_scan_png_files(self, force_refresh=False):
        """
        如果缓存存在且不强制刷新，直接加载缓存；
        否则从磁盘扫描并生成缓存（带进度条）。
        """
        if os.path.exists(self.cache_file) and not force_refresh:
            logging.info("从缓存加载 PNG 文件列表...")
            with open(self.cache_file, 'rb') as f:
                self.png_files = pickle.load(f)
            logging.info(f"从缓存加载完成，共 {len(self.png_files)} 个文件。")
        else:
            logging.info("开始扫描 PNG 文件...")
            # 先计算文件总数，避免 tqdm 进度条没有 total
            total_files = sum(1 for _ in Path(self.input_dir).rglob("*.png"))
            self.png_files = []
            for file in tqdm(Path(self.input_dir).rglob("*.png"), total=total_files, desc="扫描中"):
                self.png_files.append(str(file))
            logging.info(f"扫描完成，总共读取 {len(self.png_files)} 个 PNG 文件，保存缓存...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.png_files, f)
            logging.info("缓存保存成功。")

    def clear_cache(self):
        """清除 PNG 文件列表缓存"""
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            logging.info("缓存已清除。")
        else:
            logging.info("没有找到缓存文件，无需清理。")


# 主函数
if __name__ == "__main__":
    b = BatchGraphGenerator("/ai/edallx/Graduate_Project_2025/iccad19/iccad-Geng2020/iccad6",      # input_dir
                            "/ai/edallx/Graduate_Project_2025/iccad19/iccad19-6")                # output_dir
    b.load_or_scan_png_files()
    b.multi_process_generate()