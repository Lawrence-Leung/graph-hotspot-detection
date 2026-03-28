import os
import gdstk
from PIL import Image, ImageDraw

# ========== 3. 处理单个 cell 的函数（并行时要能被子进程导入） ==========
def export_cell_png(cell, output_dir):  # 注意，这个函数输入的是 gdstk.Polygon 对象。
    """
    将指定 cell 的 layer=(10,10) 绘制为白色(255)，背景为黑色(0)，输出 1200×1200 PNG。
    参考点偏移由 layer=(0,0) 的多边形（原先为 box）确定。
    """
    # 1) 在 layer=(0,0) 的所有多边形中寻找 "边界"，
    #    假设它的坐标可给出 cell 的左下 (x0,y0) 和右上 (x1,y1)。
    box_poly = None
    for poly in cell.polygons:
        if poly.layer == 0 and poly.datatype == 0:
            box_poly = poly
            break

    if not box_poly:
        print(f"Warning: cell {cell.name} has no layer=(0,0) polygon. Skip.")
        return

    box_points = box_poly.points  # (N,2) array-like
    x_coords = [p[0] for p in box_points]
    y_coords = [p[1] for p in box_points]
    x0, x1 = min(x_coords), max(x_coords)
    y0, y1 = min(y_coords), max(y_coords)

    # 2) 创建一张单通道图像（L模式，黑色背景）
    img = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(img)

    # 用于将 (x, y) 转为图像像素坐标
    def to_pixel(pt):
        x_um = pt[0] - x0  # 相对左下角 x0
        y_um = pt[1] - y0  # 相对左下角 y0
        px = x_um * SCALE
        py = y_um * SCALE
        # PIL 的 (0,0) 在左上方，所以翻转 Y 以让下方坐标对齐
        py = IMAGE_SIZE - py
        return (px, py)

    # 3) 在 layer=(10,10) 的 polygons 上绘制白色(255)
    for poly in cell.polygons:
        if poly.layer == 10 and poly.datatype == 0:
            pts_pixel = [to_pixel(p) for p in poly.points]
            draw.polygon(pts_pixel, fill=255)

    # 4) 输出图像
    try:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{cell.name}.png")
        img.save(out_path, "PNG")
        print(f"[{cell.name}] -> {out_path}")
    except Exception as e:
        print(e)

# 返回一个包含所有 .oas 文件相对路径的列表。
def find_oas_files(root_dir):
    # 获取当前脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 用来存储找到的 .oas 文件的相对路径
    oas_files = []
    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.oas'):
                # 获取文件的绝对路径
                absolute_path = os.path.join(root, file)
                # 计算该文件相对于脚本所在目录的相对路径
                relative_path = os.path.relpath(absolute_path, script_dir)
                # 将相对路径加入结果列表
                oas_files.append(relative_path)
    return oas_files

# 检查文件夹是否存在并创建新文件夹
def create_directory_if_not_exists(parent_dir, folder_name):
    # 构建完整的文件夹路径
    folder_path = os.path.join(parent_dir, folder_name)
    # 如果文件夹不存在，则创建它
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    return folder_path

# Debug 输出专用
def debug_print(cell):
    print(f"Cell name: {cell.name}")
    # 你可以访问 Cell 内的所有元素，例如它包含的路径、矩形、圆形等
    i = 0
    for element in cell.polygons:
        i += 1
        print(f"Element type: {type(element)}")
        print(f"Element details: {element}")
    print(f"Polygon Numbers: {i}")

# ========== 主函数 ==========
if __name__ == "__main__":

    # ========== 1. 全局配置 ==========
    root_directory = '/ai/edallx/Graduate_Project_2025/iccad19/ICCAD2019Benchmarks-master/ICCAD_2019_benchmarks'  # 替换为实际的文件夹路径
    output_parent_dir = "/ai/edallx/Graduate_Project_2025/iccad19/iccad-official"
    # 每个 cell 在 (0,0) layer 的 box 宽、高均为 4.8um
    CELL_SIZE_UM = 4.8
    # 输出图像分辨率
    IMAGE_SIZE = 1200
    # 缩放因子
    SCALE = IMAGE_SIZE / CELL_SIZE_UM  # => 1200 / 4.8 = 250 px/μm

    # ========== 2. 读取 OAS 并获取所有目标 cell ==========
    # 遍历入口文件夹下的所有目录，列出所有的 OAS 文件
    oas_files_list = find_oas_files(root_directory)

    # 读取 OAS 文件
    for oas_file in oas_files_list:
        print(f"--- Entering {oas_file}: ---")

        library = gdstk.read_oas(oas_file)
        print("--- OAS file Loaded. ---")
        # 过滤想处理的 cell，例如名字含 "patid_"
        target_cells = [c for c in library.cells if c.name.startswith("patid_")]
        print("--- Cells Filtered. ---")
        print("Target Cells Length: ", len(target_cells))

        output_stem = os.path.splitext(os.path.basename(oas_file))[0]
        # 确保输出目录存在
        os.makedirs(output_parent_dir, exist_ok=True)
        folder_path = create_directory_if_not_exists(output_parent_dir, output_stem)

        for cell in target_cells:
            export_cell_png(cell, folder_path)

    print("Done. Exported images for cells.")
