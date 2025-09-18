import os
import cv2
import math
import random

# 类别映射 (RSAR 数据集)
CLASS_MAP = {
    "ship": 0,
    "aircraft": 1,
    "car": 2,
    "tank": 3,
    "bridge": 4,
    "harbor": 5
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

def load_dota_boxes(txt_path):
    """读取 DOTA 标注 (像素坐标)"""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            cls_name = parts[8]
            boxes.append(((x1,y1),(x2,y2),(x3,y3),(x4,y4),cls_name))
    return boxes

def load_yolo_obb(txt_path, img_w, img_h):
    """读取 YOLO OBB (归一化) 并反归一化为像素坐标"""
    boxes = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id, cx, cy, w, h, angle = parts
            cls_id = int(cls_id)
            cx, cy, w, h, angle = map(float, (cx, cy, w, h, angle))

            # 反归一化
            cx *= img_w
            cy *= img_h
            w *= img_w
            h *= img_h

            # 旋转框四点坐标
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            dx = w / 2
            dy = h / 2

            points = [
                (cx - dx*cos_a + dy*sin_a, cy - dx*sin_a - dy*cos_a),
                (cx + dx*cos_a + dy*sin_a, cy + dx*sin_a - dy*cos_a),
                (cx + dx*cos_a - dy*sin_a, cy + dx*sin_a + dy*cos_a),
                (cx - dx*cos_a - dy*sin_a, cy - dx*sin_a + dy*cos_a),
            ]
            boxes.append((points, INV_CLASS_MAP[cls_id]))
    return boxes

def draw_boxes(img, boxes, color, label_prefix=""):
    """在图像上画旋转框"""
    for box in boxes:
        if isinstance(box[0], tuple):
            # DOTA: ( (x1,y1),(x2,y2),(x3,y3),(x4,y4),cls )
            pts = [box[0], box[1], box[2], box[3]]
            cls_name = box[4]
        else:
            # YOLO: ([(x1,y1)...(x4,y4)], cls)
            pts = box[0]
            cls_name = box[1]

        pts = [(int(x), int(y)) for x, y in pts]
        cv2.polylines(img, [np.array(pts)], isClosed=True, color=color, thickness=2)
        cv2.putText(img, f"{label_prefix}{cls_name}", pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    return img


if __name__ == "__main__":
    import numpy as np

    img_dir = "dataset/images"
    dota_dir = "dataset/annfiles"
    yolo_dir = "dataset/labels"
    save_dir = "check_vis"
    os.makedirs(save_dir, exist_ok=True)

    all_imgs = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    sample_imgs = random.sample(all_imgs, 5)  # 随机挑 5 张

    for img_name in sample_imgs:
        img_path = os.path.join(img_dir, img_name)
        dota_path = os.path.join(dota_dir, img_name.replace(".png", ".txt"))
        yolo_path = os.path.join(yolo_dir, img_name.replace(".png", ".txt"))

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 加载 DOTA 框 (绿色)
        dota_boxes = load_dota_boxes(dota_path)
        img = draw_boxes(img, dota_boxes, (0,255,0), "DOTA:")

        # 加载 YOLO 框 (红色)
        yolo_boxes = load_yolo_obb(yolo_path, w, h)
        img = draw_boxes(img, yolo_boxes, (0,0,255), "YOLO:")

        # 保存对比结果
        save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(save_path, img)

    print(f"检查完成，可视化结果保存在 {save_dir}/")
