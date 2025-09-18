import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


# ========== 1. 数据转换：DOTA -> YOLOv8-OBB ==========
def convert_dota_to_yoloobb(txt_path, save_path, class2id, img_width=256, img_height=256):
    os.makedirs(save_path, exist_ok=True)
    for file in os.listdir(txt_path):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(txt_path, file), 'r') as f:
            lines = f.readlines()
        out = []
        for line in lines:
            vals = line.strip().split()
            if len(vals) < 9:
                continue
            x = list(map(float, vals[:8]))
            cls = vals[8]
            if cls not in class2id:
                continue

            # 归一化坐标到0-1范围
            normalized_x = [coord / img_width if i % 2 == 0 else coord / img_height
                            for i, coord in enumerate(x)]

            # YOLOv8-OBB 格式：class_id x1 y1 x2 y2 x3 y3 x4 y4 (归一化到0-1)
            out.append(
                f"{class2id[cls]} {normalized_x[0]:.6f} {normalized_x[1]:.6f} {normalized_x[2]:.6f} {normalized_x[3]:.6f} {normalized_x[4]:.6f} {normalized_x[5]:.6f} {normalized_x[6]:.6f} {normalized_x[7]:.6f}\n")

        with open(os.path.join(save_path, file), 'w') as f:
            f.writelines(out)
    print(f"数据转换完成: {txt_path} -> {save_path}")


# ========== 2. 生成数据配置文件 ==========
def write_yaml(save_path="rsar.yaml"):
    # 使用绝对路径确保YOLO能找到数据
    current_dir = os.getcwd().replace('\\', '/')

    yaml_str = f"""# RSAR SAR目标检测数据集 (YOLOv8-OBB)
path: {current_dir}
train: train/images
val: val/images

names:
  0: ship
  1: aircraft
  2: car
  3: tank
  4: bridge
  5: harbor
"""
    with open(save_path, "w") as f:
        f.write(yaml_str)
    print(f"YAML配置文件已生成: {save_path}")
    print(f"数据集路径: {current_dir}")


# ========== 3. 检查标注文件格式 ==========
def check_label_format(label_dir):
    """检查标注文件格式是否正确"""
    print("检查标注文件格式...")
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    if not label_files:
        print("没有找到标注文件！")
        return False

    for label_file in label_files[:3]:  # 检查前3个文件
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            if lines:
                first_line = lines[0].strip()
                values = first_line.split()
                print(f"文件: {label_file}, 第一行: {first_line}")
                print(f"列数: {len(values)}")

                # 检查是否为9列（class + 8个坐标）
                if len(values) != 9:
                    print(f"错误：应该有9列，实际有{len(values)}列")
                    return False

                # 检查坐标值是否在0-1范围内
                try:
                    coords = list(map(float, values[1:]))  # 跳过class_id
                    for i, coord in enumerate(coords):
                        if coord < 0 or coord > 1:
                            print(f"警告：坐标 {coord} 不在0-1范围内 (位置 {i + 1})")
                    print("坐标范围检查完成")
                except ValueError:
                    print("错误：坐标值不是有效的浮点数")
                    return False

    return True


# ========== 4. 可视化旋转框 ==========
def visualize_predictions(results, class_names, save_dir="runs/detect/vis"):
    os.makedirs(save_dir, exist_ok=True)
    for r in results:
        img = r.orig_img.copy()
        if hasattr(r, 'obb') and r.obb is not None:
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            clss = r.obb.cls.cpu().numpy().astype(int)

            for pts, conf, cls_id in zip(boxes, confs, clss):
                pts = pts.reshape(4, 2).astype(int)
                color = (0, 255, 0)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=1)
                label = f"{class_names[cls_id]} {conf:.2f}"
                cv2.putText(img, label, (pts[0][0], pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        save_path = os.path.join(save_dir, os.path.basename(r.path))
        cv2.imwrite(save_path, img)
        print(f"可视化结果保存到: {save_path}")


# ========== 5. 绘制 Loss 和 mAP 曲线 ==========
def plot_training_curves(csv_path="runs/detect/train/results.csv", save_path="runs/detect/train/curves.png"):
    if not os.path.exists(csv_path):
        print(f"未找到 {csv_path}，无法绘制曲线")
        return
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    if "train/box_loss" in df.columns:
        plt.plot(df["epoch"], df["train/box_loss"], label="box_loss")
    if "train/cls_loss" in df.columns:
        plt.plot(df["epoch"], df["train/cls_loss"], label="cls_loss")
    if "train/dfl_loss" in df.columns:
        plt.plot(df["epoch"], df["train/dfl_loss"], label="dfl_loss")
    if "train/obb_loss" in df.columns:
        plt.plot(df["epoch"], df["train/obb_loss"], label="obb_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    # mAP 曲线
    plt.subplot(1, 2, 2)
    if "metrics/mAP50" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50"], label="mAP@0.5")
    if "metrics/mAP50-95" in df.columns:
        plt.plot(df["epoch"], df["metrics/mAP50-95"], label="mAP@0.5:0.95")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.title("Validation mAP")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"训练曲线已保存到 {save_path}")


# ========== 6. 训练 & 评估 ==========
def train_and_eval():
    class2id = {"ship": 0, "aircraft": 1, "car": 2, "tank": 3, "bridge": 4, "harbor": 5}
    id2class = {v: k for k, v in class2id.items()}

    print("=" * 50)
    print("SAR图像目标检测训练开始")
    print("图像尺寸: 256x256像素")
    print("=" * 50)

    # 检查数据目录
    print("检查数据目录...")
    required_dirs = [
        "train/labelTxt",
        "val/labelTxt",
        "train/images",
        "val/images"
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"错误：目录不存在 - {dir_path}")
            return
        print(f"✓ {dir_path}")

    # 创建labels目录
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    # 检查图像文件
    train_images = [f for f in os.listdir("train/images") if f.endswith('.png')]
    val_images = [f for f in os.listdir("val/images") if f.endswith('.png')]

    print(f"训练图像数量: {len(train_images)}")
    print(f"验证图像数量: {len(val_images)}")

    if len(train_images) == 0 or len(val_images) == 0:
        print("错误：没有找到PNG图像文件！")
        return

    # 转换数据格式 - 使用正确的YOLOv8-OBB格式（归一化坐标）
    print("开始转换数据格式...")
    convert_dota_to_yoloobb("train/labelTxt", "train/labels", class2id, img_width=256, img_height=256)
    convert_dota_to_yoloobb("val/labelTxt", "val/labels", class2id, img_width=256, img_height=256)

    # 检查转换后的标注文件格式
    if not check_label_format("train/labels"):
        print("标注文件格式检查失败！")
        return

    if not check_label_format("val/labels"):
        print("验证集标注文件格式检查失败！")
        return

    # 生成配置文件
    write_yaml("rsar.yaml")

    # 加载模型
    print("加载YOLOv8-OBB模型...")
    try:
        model = YOLO("yolov8m-obb.pt")
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 训练配置
    print("开始训练...")
    model.train(
        data="rsar.yaml",
        epochs=50,  # 减少epoch数以便快速测试
        imgsz=256,
        batch=16,
        device=0,
        optimizer="AdamW",
        lr0=0.001,
        workers=4,
        verbose=True
    )

    # 验证
    print("开始验证...")
    try:
        metrics = model.val()
        print("评估结果：")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
    except Exception as e:
        print(f"验证失败: {e}")

    # 推理测试
    print("进行推理测试...")
    try:
        results = model.predict(
            source="val/images",
            save=True,
            imgsz=256,
            conf=0.3,
            device=0
        )
        print("推理完成")

        # 可视化
        visualize_predictions(results, id2class)

    except Exception as e:
        print(f"推理失败: {e}")

    # 绘制训练曲线
    plot_training_curves()

    print("=" * 50)
    print("训练完成！")
    print("=" * 50)


if __name__ == "__main__":
    train_and_eval()