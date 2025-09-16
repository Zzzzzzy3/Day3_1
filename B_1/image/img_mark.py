import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import os
import glob
import numpy as np


def draw_quadrilaterals(image_path, all_coordinates, categories):
    """
    在图像上绘制多个四边形标注框
    参数:
    image_path: 图像路径
    all_coordinates: 多个四边形坐标列表 [[x1,y1,x2,y2,x3,y3,x4,y4], ...]
    categories: 每个框对应的类别列表
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    # 定义不同颜色用于区分不同的类别
    color_map = {}
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink']

    # 绘制每个四边形
    for idx, (coordinates, category) in enumerate(zip(all_coordinates, categories)):
        # 提取坐标点
        points = [(coordinates[i], coordinates[i + 1]) for i in range(0, 8, 2)]

        # 创建四边形路径
        codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        vertices = points + [points[0]]  # 闭合多边形
        path = Path(vertices, codes)

        # 为每个类别分配颜色
        if category not in color_map:
            color_map[category] = colors[len(color_map) % len(colors)]
        color = color_map[category]

        # 创建patch并添加到图中
        patch = patches.PathPatch(path, linewidth=2, edgecolor=color,
                                  facecolor='none', alpha=0.7)
        ax.add_patch(patch)

        # 计算中心点用于标注编号和类别
        center_x = np.mean([p[0] for p in points])
        center_y = np.mean([p[1] for p in points])

        # 在框中心标注编号和类别
        label = f"{idx + 1}:{category}"
        ax.text(center_x, center_y, label, color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, boxstyle='round'),
                ha='center', va='center')

    # 添加图例
    legend_elements = []
    for category, color in color_map.items():
        legend_elements.append(patches.Patch(facecolor=color, label=category, alpha=0.7))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.title(f"path: {os.path.basename(image_path)} - totally{len(all_coordinates)}RBB")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def process_coordinates(line):
    """
    处理单行数据，提取坐标和类别
    格式: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    """
    parts = line.split()
    if len(parts) < 8:
        return None, None, None

    # 提取前8个元素并转换为浮点数
    coordinates = [float(part) for part in parts[:8]]

    # 提取类别（如果有）
    category = parts[8] if len(parts) > 8 else "unknown"

    # 提取难度（如果有）
    difficulty = parts[9] if len(parts) > 9 else "0"

    return coordinates, category, difficulty


def process_txt_file(txt_folder, image_folder, file_index):
    """
    处理指定文件夹中的第n个txt文件
    参数:
    txt_folder: 包含txt文件的文件夹路径
    image_folder: 包含png图像的文件夹路径
    file_index: 要处理的文件序号(从1开始)
    """
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(txt_folder, "*.txt"))
    txt_files.sort()  # 按文件名排序

    if not txt_files:
        print(f"错误: 在文件夹 {txt_folder} 中未找到任何txt文件")
        return

    if file_index < 1 or file_index > len(txt_files):
        print(f"错误: 文件序号 {file_index} 超出范围 (1-{len(txt_files)})")
        return

    # 选择指定序号的txt文件
    txt_file = txt_files[file_index - 1]
    txt_filename = os.path.basename(txt_file)
    print(f"处理文件: {txt_filename}")

    # 读取文件内容
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        print(f"警告: 文件 {txt_file} 为空")
        return

    # 收集所有标注框的坐标和类别
    all_coordinates = []
    all_categories = []
    difficulties = []

    # 处理每一行
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        try:
            coordinates, category, difficulty = process_coordinates(line)
            if coordinates:
                all_coordinates.append(coordinates)
                all_categories.append(category)
                difficulties.append(difficulty)
                print(f"第 {i + 1} 行: 类别={category}, 难度={difficulty}, 坐标={coordinates}")
            else:
                print(f"警告: 第 {i + 1} 行格式不正确: {line}")

        except Exception as e:
            print(f"处理第 {i + 1} 行时出错: {e}")

    if not all_coordinates:
        print("错误: 未找到有效的标注框数据")
        return

    print(f"共找到 {len(all_coordinates)} 个标注框")

    # 构建对应的图像文件名
    base_name = os.path.splitext(txt_filename)[0]
    image_filename = base_name + ".png"
    image_path = os.path.join(image_folder, image_filename)

    # 检查图像文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件 {image_path} 不存在")
        # 尝试查找其他可能的图像文件
        png_files = glob.glob(os.path.join(image_folder, "*.png"))
        if png_files:
            print(f"在图像文件夹中找到 {len(png_files)} 个PNG文件")
            # 尝试找到文件名最接近的
            for png_file in png_files:
                png_base = os.path.splitext(os.path.basename(png_file))[0]
                if png_base == base_name:
                    image_path = png_file
                    print(f"找到匹配的图像文件: {image_path}")
                    break
        return

    print(f"找到对应图像: {image_path}")

    # 绘制所有标注框
    draw_quadrilaterals(image_path, all_coordinates, all_categories)


if __name__ == '__main__':
    # 用户输入
    txt_folder ="D:/user_3188/AIC_SAR/source/train/annfiles"
    image_folder ="D:/user_3188/AIC_SAR/source/train/images"

    try:
        file_index = int(input("请输入要处理的文件序号(从0开始): ").strip())+1
    except ValueError:
        print("错误: 请输入有效的数字")
        exit(1)

    # 检查文件夹是否存在
    if not os.path.exists(txt_folder):
        print(f"错误: txt文件夹 {txt_folder} 不存在")
        exit(1)

    if not os.path.exists(image_folder):
        print(f"错误: 图像文件夹 {image_folder} 不存在")
        exit(1)

    # 处理指定的txt文件
    process_txt_file(txt_folder, image_folder, file_index)