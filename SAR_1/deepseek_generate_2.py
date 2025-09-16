# ========== 改进的mAP评估函数 ==========
def calculate_map(detections, ground_truths, iou_threshold=0.5, num_classes=7):
    """
    简化版 mAP 计算 (支持分类)
    detections: [{'boxes':..., 'scores':..., 'labels':...}, ...]
    ground_truths: [{'boxes':..., 'labels':...}, ...]
    """
    aps = []

    for cls in range(1, num_classes):  # 0是背景类，跳过
        cls_dets = []
        cls_gts = []

        # 按类别筛选
        for det, gt in zip(detections, ground_truths):
            det_mask = det['labels'] == cls
            gt_mask = gt['labels'] == cls
            if det_mask.sum() == 0 and gt_mask.sum() == 0:
                continue

            det_boxes = det['boxes'][det_mask]
            det_scores = det['scores'][det_mask]
            gt_boxes = gt['boxes'][gt_mask]

            if len(det_boxes) == 0 and len(gt_boxes) == 0:
                continue

            cls_dets.append((det_boxes, det_scores))
            cls_gts.append(gt_boxes)

        # 没有该类别
        if not cls_dets:
            continue

        # 计算 TP/FP
        tp, fp, total_gts = [], [], 0
        for (det_boxes, det_scores), gt_boxes in zip(cls_dets, cls_gts):
            total_gts += len(gt_boxes)
            if len(gt_boxes) == 0:
                # 全部算 FP
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

        # 按得分排序
        tp = np.array(tp)
        fp = np.array(fp)
        scores = np.concatenate([d[1].cpu().numpy() for d in cls_dets])
        order = np.argsort(-scores)

        tp = tp[order]
        fp = fp[order]

        # 累积
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / total_gts
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)

        # 插值计算 AP (VOC 11 点法)
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            prec = precisions[recalls >= r].max() if np.any(recalls >= r) else 0
            ap += prec / 11.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


# ========== 改进后的 evaluate 函数 ==========
def evaluate_map(model, data_loader, num_classes=7, iou_threshold=0.5):
    model.eval()
    detections, ground_truths = [], []

    with torch.no_grad():
        for images, targets in data_loader:
            try:
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
            except Exception as e:
                print(f"Evaluation error: {e}")
                continue

    return calculate_map(detections, ground_truths, iou_threshold, num_classes)


# ========== 改进后的训练函数（混合精度+评估） ==========
def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.001, num_classes=7):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # 混合精度
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-4)

    best_map = 0.0
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

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

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {losses.item():.4f}')

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            train_losses.append(avg_loss)
            print(f'Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}')

        # 每隔3轮验证一次
        if (epoch + 1) % 3 == 0:
            val_map = evaluate_map(model, val_loader, num_classes=num_classes)
            print(f'Epoch {epoch+1}, Validation mAP: {val_map:.4f}')

            if val_map > best_map:
                best_map = val_map
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best mAP: {best_map:.4f}')

    return model, train_losses
