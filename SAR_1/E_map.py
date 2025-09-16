import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 1. 数据集类
class SARDetectionDataset(Dataset):
    def __init__(self, image_dir, ann_dir, transform=None, max_samples=2000):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.transform = transform

        self.image_files = []
        for f in os.listdir(image_dir):
            if f.lower().endswith('.png'):
                self.image_files.append(f)

        if max_samples and len(self.image_files) > max_samples:
            import random
            random.seed(42)
            self.image_files = random.sample(self.image_files, max_samples)

        print(f'Found {len(self.image_files)} valid image-annotation pairs')

        self.class_dict = {
            'ship': 1, 'aircraft': 2, 'car': 3,
            'tank': 4, 'bridge': 5, 'harbor': 6
        }

    def __len__(self):
        return len(self.image_files)

    def parse_annotation(self, ann_path):
        boxes, labels = [], []
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line or line.startswith(('imagesource', 'gsd')):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                coords = list(map(float, parts[:8]))
                class_name = parts[8]
                xs, ys = coords[0::2], coords[1::2]
                x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

                if (x_max - x_min >= 2 and y_max - y_min >= 2 and
                        x_min >= 0 and y_min >= 0 and x_max <= 256 and y_max <= 256):
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(self.class_dict.get(class_name, 0))
        except Exception as e:
            print(f"Warning: Error parsing {ann_path}: {e}")
            return [], []
        return boxes, labels

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            ann_file = os.path.splitext(img_name)[0] + '.txt'
            ann_path = os.path.join(self.ann_dir, ann_file)

            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            boxes, labels = self.parse_annotation(ann_path)
            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]
                labels = [0]

            if self.transform:
                try:
                    transformed = self.transform(
                        image=image,
                        bboxes=boxes,
                        class_labels=labels
                    )
                    image = transformed['image']
                    boxes = transformed['bboxes']
                    labels = transformed['class_labels']
                except Exception as e:
                    print(f"Transform error: {e}")

            if isinstance(image, np.ndarray):  # 没有ToTensorV2时兜底
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

            target = {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': torch.tensor([idx]),
                'area': (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0]),
                'iscrowd': torch.zeros(len(boxes_tensor), dtype=torch.int64)
            }

            return image, target
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            empty_image = torch.zeros((3, 256, 256), dtype=torch.float32)
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
            return empty_image, empty_target


# 2. 数据增强
def get_simple_transform(train=True):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


# 3. 模型
def create_model(num_classes=7):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


# 4. collate
def collate_fn(batch):
    images, targets = [], []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


# 5. mAP 计算
def calculate_map(detections, ground_truths, iou_threshold=0.5, num_classes=7):
    aps = []
    for cls in range(1, num_classes):
        cls_dets, cls_gts = [], []
        for det, gt in zip(detections, ground_truths):
            det_mask = det['labels'] == cls
            gt_mask = gt['labels'] == cls
            det_boxes = det['boxes'][det_mask]
            det_scores = det['scores'][det_mask]
            gt_boxes = gt['boxes'][gt_mask]
            if len(det_boxes) == 0 and len(gt_boxes) == 0:
                continue
            cls_dets.append((det_boxes, det_scores))
            cls_gts.append(gt_boxes)

        if not cls_dets:
            continue

        tp, fp, total_gts = [], [], 0
        for (det_boxes, det_scores), gt_boxes in zip(cls_dets, cls_gts):
            total_gts += len(gt_boxes)
            if len(gt_boxes) == 0:
                tp.extend([0] * len(det_boxes))
                fp.extend([1] * len(det_boxes))
                continue
            ious = box_iou(det_boxes, gt_boxes)
            max_ious, indices = ious.max(dim=1)
            matched = set()
            for i, iou in enumerate(max_ious):
                if iou >= iou_threshold and indices[i].item() not in matched:
                    tp.append(1)
                    fp.append(0)
                    matched.add(indices[i].item())
                else:
                    tp.append(0)
                    fp.append(1)

        if total_gts == 0:
            continue

        tp, fp = np.array(tp), np.array(fp)
        scores = np.concatenate([d[1].cpu().numpy() for d in cls_dets])
        order = np.argsort(-scores)
        tp, fp = tp[order], fp[order]
        tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
        recalls = tp_cum / total_gts
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = 0.0
        for r in np.linspace(0, 1, 11):
            prec = precisions[recalls >= r].max() if np.any(recalls >= r) else 0
            ap += prec / 11.0
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def evaluate_map(model, data_loader, num_classes=7, iou_threshold=0.5):
    model.eval()
    detections, ground_truths = [], []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i, output in enumerate(outputs):
                detections.append({
                    'boxes': output['boxes'].cpu(),
                    'scores': output['scores'].cpu(),
                    'labels': output['labels'].cpu()
                })
                ground_truths.append({
                    'boxes': targets[i]['boxes'].cpu(),
                    'labels': targets[i]['labels'].cpu()
                })
    return calculate_map(detections, ground_truths, iou_threshold, num_classes)


# 6. 训练函数
def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, num_classes=7):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_map = 0.0
    train_losses, val_maps = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, batch_count = 0, 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
            if not valid_indices:
                continue

            images = [images[i].to(device) for i in valid_indices]
            targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid_indices]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                continue

            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            batch_count += 1

            if batch_count % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {losses.item():.4f}')

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

        val_map = evaluate_map(model, val_loader, num_classes=num_classes)
        val_maps.append(val_map)
        print(f'Epoch {epoch+1}, Validation mAP: {val_map:.4f}')

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best mAP: {best_map:.4f}')

    return model, train_losses, val_maps


# 7. 主函数
def main():
    image_dir = r'D:\user_3188\AIC_SAR\source\train\images'
    ann_dir = r'D:\user_3188\AIC_SAR\source\train\annfiles'

    print("Loading datasets...")
    dataset = SARDetectionDataset(image_dir, ann_dir, get_simple_transform(train=True), max_samples=2000)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    print("Creating model...")
    model = create_model(num_classes=7)

    print("Starting training...")
    trained_model, train_losses, val_maps = train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, num_classes=7)

    # 绘制曲线
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_maps)+1), val_maps, marker='s', color='orange', label='Validation mAP')
    plt.title('Validation mAP Curve')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    print("Training completed successfully!")


if __name__ == '__main__':
    main()
