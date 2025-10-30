import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

from src.config import VOC_CLASSES, VOC_IMG_MEAN, YOLO_IMG_DIM


def decoder(pred):
    """
    pred (tensor) 1xSxSx(B*5+C)  -- in our case with resnet: 1x14x14x(2*5+20)
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    """
    grid_num = pred.squeeze().shape[0]  # 14 for resnet50 base, 7 for vgg16
    assert pred.squeeze().shape[0] == pred.squeeze().shape[1]  # square grid
    boxes = []
    cls_indexs = []
    probs = []
    cell_size = 1.0 / grid_num
    pred = pred.data
    pred = pred.squeeze(0)  # SxSx(B*5+C)
    object_confidence1 = pred[:, :, 4].unsqueeze(2)
    object_confidence2 = pred[:, :, 9].unsqueeze(2)
    object_confidences = torch.cat((object_confidence1, object_confidence2), 2)

    # Select all predictions above the threshold
    min_confidence_threshold = 0.1
    mask1 = object_confidences > min_confidence_threshold

    # We always want to select at least one predictions so we also take the prediction with max confidence
    mask2 = object_confidences == object_confidences.max()
    mask = (mask1 + mask2).gt(0)

    # We need to convert the grid-relative coordinates back to image coordinates
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                if mask[i, j, b] == 1:
                    box = pred[i, j, b * 5 : b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = (
                        torch.FloatTensor([j, i]) * cell_size
                    )  # upper left corner of grid cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(
                        box.size()
                    )  # convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    if float((contain_prob * max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1, 4))
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob * max_prob)
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes, 0)
        probs = torch.cat(probs, 0)
        cls_indexs = torch.stack(cls_indexs, dim=0)

    # Perform non-maximum suppression so that we don't predict many similar and overlapping boxes
    keep = nms(boxes, probs)
    return boxes[keep], cls_indexs[keep], probs[keep]


def nms(bboxes, scores, threshold=0.3):
    """
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0] if order.numel() > 1 else order.item()
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2 - xx1).clamp(min=0)
        h = (yy2 - yy1).clamp(min=0)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr <= threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids + 1]
    return torch.LongTensor(keep)


def compute_iou_wh(box1, box2):
    """Compute IoU between two boxes in (cx, cy, w, h) format (0-1 normalized)."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    # Union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area


def predict_image(model, image_name, root_img_directory=""):
    """
    Predict output for a single image using YOLO v3 multi-scale predictions.

    :param model: detector model for inference
    :param image_name: image file name e.g. '0000000.jpg'
    :param root_img_directory:
    :return: List of lists containing:
        - (x1, y1)
        - (x2, y2)
        - predicted class name
        - image name
        - predicted class probability
    """

    result = []
    image = cv2.imread(os.path.join(root_img_directory + image_name))
    h, w, _ = image.shape
    img = cv2.resize(image, (YOLO_IMG_DIM, YOLO_IMG_DIM))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = VOC_IMG_MEAN
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)

    with torch.no_grad():
        img = Variable(img[None, :, :, :])
        img = img.cuda()
        model.eval()

        # Get raw predictions from all 3 scales (bypass potentially buggy NMS)
        predictions = model(img, return_raw=True)  # Returns tuple of (pred_13, pred_26, pred_52)

        # YOLO v3 anchors (same as in yolo_loss.py and resnet_yolo.py)
        anchors = [
            # Scale 1 (13x13): Large anchors for large objects
            [(0.9555, 0.4289), (0.6300, 0.7966), (1.0154, 0.8347)],
            # Scale 2 (26x26): Medium anchors for medium objects
            [(0.3843, 0.2255), (0.5086, 0.4398), (0.3387, 0.6758)],
            # Scale 3 (52x52): Small anchors for small objects
            [(0.0639, 0.0817), (0.1426, 0.1908), (0.2109, 0.4096)],
        ]

        all_detections = []

        # Process each scale
        for scale_idx, pred in enumerate(predictions):
            grid_size = pred.size(1)
            num_anchors = 3
            num_classes = 20

            # Reshape: (1, H, W, 75) -> (1, H, W, 3, 25)
            pred = pred.view(1, grid_size, grid_size, num_anchors, 5 + num_classes)
            pred = pred.squeeze(0)  # (H, W, 3, 25)

            # Apply sigmoid to appropriate components
            pred_xy = torch.sigmoid(pred[..., 0:2])  # (H, W, 3, 2) - offsets within cell
            pred_wh = pred[..., 2:4]                  # (H, W, 3, 2) - raw w/h to be exp'ed
            pred_conf = torch.sigmoid(pred[..., 4])   # (H, W, 3) - objectness
            pred_cls = torch.sigmoid(pred[..., 5:])   # (H, W, 3, 20) - class probs

            # Create grid
            for i in range(grid_size):
                for j in range(grid_size):
                    for a in range(num_anchors):
                        obj_conf = pred_conf[i, j, a].item()

                        # Filter by objectness confidence
                        if obj_conf < 0.05:
                            continue

                        # Get class probabilities
                        class_probs = pred_cls[i, j, a]
                        max_class_prob, class_id = torch.max(class_probs, 0)

                        # Final confidence
                        final_conf = obj_conf * max_class_prob.item()
                        if final_conf < 0.05:
                            continue

                        # Decode bounding box
                        x_offset = pred_xy[i, j, a, 0].item()
                        y_offset = pred_xy[i, j, a, 1].item()

                        # Center coordinates in image space (0-1)
                        cx = (j + x_offset) / grid_size
                        cy = (i + y_offset) / grid_size

                        # Width and height with anchors
                        w_raw = pred_wh[i, j, a, 0].item()
                        h_raw = pred_wh[i, j, a, 1].item()

                        # Clamp to prevent overflow
                        w_raw = max(-10, min(10, w_raw))
                        h_raw = max(-10, min(10, h_raw))

                        # Get anchor dimensions for this scale
                        anchor_w, anchor_h = anchors[scale_idx][a]

                        # Decode width and height
                        bw = np.exp(w_raw) * anchor_w
                        bh = np.exp(h_raw) * anchor_h

                        # Clamp to reasonable values
                        bw = min(1.5, bw)
                        bh = min(1.5, bh)

                        all_detections.append({
                            'cx': cx,
                            'cy': cy,
                            'w': bw,
                            'h': bh,
                            'conf': final_conf,
                            'class_id': class_id.item()
                        })

        # Apply NMS per class
        for cls_id in range(20):
            cls_detections = [d for d in all_detections if d['class_id'] == cls_id]
            if len(cls_detections) == 0:
                continue

            # Sort by confidence
            cls_detections.sort(key=lambda x: x['conf'], reverse=True)

            # NMS
            keep = []
            while len(cls_detections) > 0:
                # Keep highest confidence
                best = cls_detections.pop(0)
                keep.append(best)

                # Remove overlapping boxes
                remaining = []
                for det in cls_detections:
                    iou = compute_iou_wh(
                        [best['cx'], best['cy'], best['w'], best['h']],
                        [det['cx'], det['cy'], det['w'], det['h']]
                    )
                    if iou <= 0.3:  # NMS threshold
                        remaining.append(det)

                cls_detections = remaining

                if len(keep) >= 100:  # Max detections per class
                    break

            # Convert to output format
            for det in keep:
                cx, cy, bw, bh = det['cx'], det['cy'], det['w'], det['h']
                conf = det['conf']

                # Convert to pixel coordinates
                x1 = int(np.clip((cx - bw/2) * w, 0, w-1))
                x2 = int(np.clip((cx + bw/2) * w, 0, w-1))
                y1 = int(np.clip((cy - bh/2) * h, 0, h-1))
                y2 = int(np.clip((cy + bh/2) * h, 0, h-1))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                result.append([
                    (x1, y1), (x2, y2),
                    VOC_CLASSES[cls_id],
                    image_name,
                    conf
                ])

    return result
