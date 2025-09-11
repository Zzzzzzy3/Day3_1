import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path


def draw_quadrilateral(coordinates):
    """
    在图像上绘制四边形标注框
    参数:
    image_path: 图像路径
    coords: 四边形坐标列表 [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    image_path="D:/user_3188/AIC_SAR/Day3_1/B_1/image/000001.png"
    coords=coordinates
    # 读取图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR转RGB

    # 创建图形
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # 提取坐标点
    points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]

    # 创建四边形路径
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices = points + [points[0]]  # 闭合多边形
    path = Path(vertices, codes)

    # 创建patch并添加到图中
    patch = patches.PathPatch(path, linewidth=2, edgecolor='red',
                              facecolor='none', alpha=0.7)
    ax.add_patch(patch)

    # 可选：标注顶点
    for i, (x, y) in enumerate(points):
        ax.plot(x, y, 'ro', markersize=4)  # 绘制顶点
        ax.text(x, y, f'P{i + 1}', color='blue', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def process_coordinates(data_string):
    # 分割字符串
    parts = data_string.split()

    # 提取前8个元素并转换为浮点数
    coordinates = [float(part) for part in parts[:8]]

    return coordinates

if __name__ == '__main__':

    with open('000001.txt', 'r', encoding='utf-8') as file:
        data_string = file.readline().strip()

    coordinates=process_coordinates(data_string)
    draw_quadrilateral(coordinates)