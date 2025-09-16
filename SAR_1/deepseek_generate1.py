import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou, nms
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


# 1. 修复的数据集类
#把“图像文件夹 + 标签文件夹”里能对的上的文件全部扫出来，做成列表。
class SARDetectionDataset(Dataset):
    def __init__(self, image_dir, ann_dir, transform=None, max_samples=2000):
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.transform = transform

        # 获取图像文件列表
        # self.image_files = []
        # valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        # for f in os.listdir(image_dir):
        #     if f.lower().endswith(valid_extensions):
        #         ann_file = os.path.splitext(f)[0] + '.txt'
        #         ann_path = os.path.join(ann_dir, ann_file)
        #         if os.path.exists(ann_path):
        #             self.image_files.append(f)
        self.image_files=[]
        valid_extensions = '.png'
        for f in os.listdir(image_dir):
            if f.lower().endswith(valid_extensions):
                self.image_files.append(f)

        # 限制样本数量以避免内存问题
        if max_samples and len(self.image_files) > max_samples:
            import random
            random.seed(42)
            self.image_files = random.sample(self.image_files, max_samples)

        print(f'Found {len(self.image_files)} valid image-annotation pairs')

        # 类别映射
        self.class_dict = {'ship': 1, 'aircraft': 2, 'car': 3,
                           'tank': 4, 'bridge': 5, 'harbor': 6}

    def __len__(self):
        return len(self.image_files)

    def parse_annotation(self, ann_path):
        boxes = []
        labels = []

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

                try:
                    # 解析坐标
                    coords = list(map(float, parts[:8]))
                    class_name = parts[8]

                    # 转换为水平矩形框
                    xs = coords[0::2]
                    ys = coords[1::2]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)

                    # 过滤无效框
                    if (x_max - x_min >= 2 and y_max - y_min >= 2 and
                            x_min >= 0 and y_min >= 0 and x_max <= 256 and y_max <= 256):
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(self.class_dict.get(class_name, 0))

                except (ValueError, IndexError):
                    continue

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

            # 读取图像
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            # 解析标注
            boxes, labels = self.parse_annotation(ann_path)

            # 如果没有标注，创建空标注
            if len(boxes) == 0:
                boxes = [[0, 0, 1, 1]]  # 添加一个虚拟框避免错误
                labels = [0]

            # 应用变换
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
                    # 使用原始数据
                    pass

            # 转换为tensor
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
            # 返回一个空的样本
            empty_image = np.zeros((256, 256, 3), dtype=np.uint8)
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros(0),
                'iscrowd': torch.zeros(0, dtype=torch.int64)
            }
            return empty_image, empty_target


# 2. 简化的数据增强
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


# 3. 创建模型
def create_model(num_classes=7):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


# 4. 自定义collate函数 - 修复版本
def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    return images, targets


# 5. 简化的mAP计算(检测时没有考虑分类
def calculate_simple_map(detections, ground_truths, iou_threshold=0.5):
    """简化的mAP计算，避免复杂逻辑"""
    try:
        aps = []

        for dets, gts in zip(detections, ground_truths):
            if len(dets['boxes']) == 0 or len(gts['boxes']) == 0:
                continue

            # 计算IoU
            iou_matrix = box_iou(dets['boxes'], gts['boxes'])

            # 找到最佳匹配
            max_ious, _ = torch.max(iou_matrix, dim=1)
            correct = (max_ious >= iou_threshold).float()

            if len(correct) > 0:
                ap = correct.mean().item()
                aps.append(ap)

        return np.mean(aps) if aps else 0.0

    except Exception as e:
        print(f"mAP calculation error: {e}")
        return 0.0


# 6. 稳定的训练函数
def train_stable_model(model, train_loader, val_loader, num_epochs=15, lr=0.001):
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_map = 0.0
    train_losses = []

    for epoch in range(num_epochs):
        # 训练
        model.train()
        epoch_loss = 0
        batch_count = 0

        try:
            for batch_idx, (images, targets) in enumerate(train_loader):
                try:
                    # 过滤空目标
                    valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
                    if not valid_indices:
                        continue

                    images = [images[i].to(device) for i in valid_indices]
                    targets = [{k: v.to(device) for k, v in targets[i].items()} for i in valid_indices]

                    optimizer.zero_grad()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    if torch.isnan(losses) or torch.isinf(losses):
                        continue

                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += losses.item()
                    batch_count += 1

                    if batch_idx % 50 == 0:
                        print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {losses.item():.4f}')

                except Exception as e:
                    print(f"Batch error: {e}")
                    continue

            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                train_losses.append(avg_loss)
                print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

            # 验证
            if (epoch + 1) % 3 == 0:
                val_map = evaluate_simple(model, val_loader)
                print(f'Epoch {epoch + 1}, Validation mAP: {val_map:.4f}')

                if val_map > best_map:
                    best_map = val_map
                    # torch.save(model.state_dict(), 'best_model.pth')
                    print(f'New best mAP: {best_map:.4f}')

        except Exception as e:
            print(f"Epoch error: {e}")
            continue

    return model, train_losses


# 7. 简化的评估函数
def evaluate_simple(model, data_loader):
    model.eval()
    detections = []
    ground_truths = []

    with torch.no_grad():
        for images, targets in data_loader:
            try:
                images = [img.to(device) for img in images]
                outputs = model(images)

                for i, output in enumerate(outputs):
                    if len(output['boxes']) > 0:
                        detections.append({
                            'boxes': output['boxes'].cpu(),
                            'scores': output['scores'].cpu(),
                            'labels': output['labels'].cpu()
                        })
                        ground_truths.append({
                            'boxes': targets[i]['boxes'].cpu(),
                            'labels': targets[i]['labels'].cpu()
                        })
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

    return calculate_simple_map(detections, ground_truths)


# 8. 主函数
def main():
    try:
        # 数据路径
        image_dir = r'D:\user_3188\AIC_SAR\source\train\images'
        ann_dir = r'D:\user_3188\AIC_SAR\source\train\annfiles'

        print("Loading datasets...")
        # 创建数据集
        dataset = SARDetectionDataset(image_dir, ann_dir, get_simple_transform(train=True), max_samples=2000)

        # 分割训练验证集
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

        import matplotlib.patches as patches

        # 数据加载器 - 使用更少的worker避免错误
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # 设置为0避免多进程错误
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        exit()
        # 创建模型
        print("Creating model...")
        model = create_model(num_classes=7)

        # 训练模型
        print("Starting training...")
        trained_model, train_losses = train_stable_model(
            model, train_loader, val_loader, num_epochs=15, lr=0.001
        )

        # 保存最终模型
        # torch.save(trained_model.state_dict(), 'final_model.pth')
        print("Training completed successfully!")

        # 绘制训练曲线
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('training_loss.png')
        plt.show()

    except Exception as e:
        print(f"Main function error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()