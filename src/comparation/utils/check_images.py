import os
import argparse
from PIL import Image
from tqdm import tqdm


def is_valid_image(image_path):
    """检查图像文件是否可以被PIL打开"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # 仅验证，不加载图像数据
        return True
    except Exception:
        return False


def clean_invalid_images(directory):
    """遍历目录中的所有 PNG 图片，删除无法识别的文件"""
    total_checked = 0
    invalid_count = 0

    # 获取所有 PNG 文件
    png_files = [os.path.join(root, file)
                 for root, _, files in os.walk(directory)
                 for file in files if file.lower().endswith('.png')]

    with tqdm(total=len(png_files), desc="Checking Images") as pbar:
        for image_path in png_files:
            total_checked += 1
            if not is_valid_image(image_path):
                invalid_count += 1
                os.remove(image_path)
                print(f"Deleted invalid image: {image_path}")
            pbar.update(1)

    print(f"Total checked: {total_checked}, Invalid images deleted: {invalid_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and clean invalid PNG images")
    parser.add_argument("directory", type=str, help="Path to the image directory")
    args = parser.parse_args()

    clean_invalid_images(args.directory)
