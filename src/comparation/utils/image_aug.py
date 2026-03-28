import os
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def process_image(file_path, output_dir):
    """处理单个图像并保存不同的变种"""
    try:
        # 读取图像
        image = cv2.imread(file_path)
        if image is None:
            print(f"无法读取文件 {file_path}, 跳过该文件。")
            return

        # 获取图像的基本信息
        filename = Path(file_path).stem
        dir_path = Path(file_path).parent

        print(f"开始处理: {file_path}")

        # 如果已经处理过该图像，跳过
        for i in [0, 1, 2, 3]:
            nonflip_filename = f"{filename}_{i}_nonflip.png"
            flip_filename = f"{filename}_{i}_flip.png"

            nonflip_path = os.path.join(dir_path, nonflip_filename)
            flip_path = os.path.join(dir_path, flip_filename)

            # 如果文件已经存在，跳过该图像
            if os.path.exists(nonflip_path) and os.path.exists(flip_path):
                print(f"文件 {nonflip_path} 和 {flip_path} 已存在，跳过处理。")
                return

        # 1. 不旋转图像，作左右翻转，保存为 *_0_nonflip.png
        flipped_image = cv2.flip(image, 1)
        flip_filename = f"{filename}_0_nonflip.png"
        flip_filepath = os.path.join(output_dir, flip_filename)
        cv2.imwrite(flip_filepath, flipped_image)
        # print(f"保存图像: {flip_filepath}")

        # 2. 旋转 90、180、270 度，不作左右翻转
        for i in [1]:
            rotated_image = np.rot90(image, k=i)
            rotated_filename = f"{filename}_{i}_nonflip.png"
            rotated_filepath = os.path.join(output_dir, rotated_filename)
            cv2.imwrite(rotated_filepath, rotated_image)
            # print(f"保存图像: {rotated_filepath}")

            # 3. 旋转 i * 90 度，作左右翻转
            rotated_flipped_image = cv2.flip(rotated_image, 1)
            rotated_flipped_filename = f"{filename}_{i}_flip.png"
            rotated_flipped_filepath = os.path.join(output_dir, rotated_flipped_filename)
            cv2.imwrite(rotated_flipped_filepath, rotated_flipped_image)
            # print(f"保存图像: {rotated_flipped_filepath}")

    except Exception as e:
        print(f"处理图像 {file_path} 时出错: {e}")


def process_directory(entrance_dir):
    """遍历入口目录并并行处理每个图像"""
    # 获取所有子文件夹和文件
    for root, dirs, files in os.walk(entrance_dir):
        # 在每个子文件夹中并行处理每个图像文件
        with ProcessPoolExecutor() as executor:
            futures = []
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(process_image, file_path, root))
            # 等待所有任务完成
            for future in futures:
                future.result()

    print("所有图像处理完成！")


# 输入目录路径
entrance_dir = input("请输入图像数据目录路径: ")
process_directory(entrance_dir)
