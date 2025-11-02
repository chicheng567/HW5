import sys
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from src.config import VOC_CLASSES, VOC_IMG_MEAN, VOC_IMG_STD, YOLO_IMG_DIM
import albumentations as A
from src.dataset import test_data_pipelines

# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False
):
    """
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    """
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # No predictions made for this class
            ap = 0.0
            print(
                "---class {} ap {}--- (no predictions for this class)".format(
                    class_, ap
                )
            )
            aps += [ap]
            continue
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.0
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    union = (
                        (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                        + (bbgt[2] - bbgt[0] + 1.0) * (bbgt[3] - bbgt[1] + 1.0)
                        - inters
                    )
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # bbox has already been used
                        if len(BBGT) == 0:
                            del target[
                                (image_id, class_)
                            ]  # delete things that don't have bbox
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        print("---class {} ap {}---".format(class_, ap))
        aps += [ap]
    print("---map {}---".format(np.mean(aps)))
    return aps


def evaluate(model, eval_loader):
    """
    Evaluate model using DataLoader with built-in inference method.

    Args:
        model: ODModel instance with inference() method
        eval_loader: DataLoader with encode_target=False

    Returns:
        aps: List of average precision scores per class
    """

    print("---Evaluate model on validation samples---")
    sys.stdout.flush()
    model.eval()
    targets = defaultdict(list)
    preds = defaultdict(list)

    device = next(model.parameters()).device

    # Run inference on all batches
    global_idx = 0
    for images, target_list in tqdm(eval_loader):
        # Move to GPU if available
        images = images.to(device)

        # Run inference with LOW threshold for evaluation
        # Using low conf_thres (0.01) allows model to propose more candidates
        # The mAP metric will properly evaluate them
        with torch.no_grad():
            detections = model.inference(images, conf_thres=0.01, nms_thres=0.4)

        # Process each image in the batch
        for i in range(len(images)):
            # Use global index as unique image_id
            image_id = global_idx + i

            # Build ground truth from target_list
            # target_list[i] is a list of instances: [[x1,y1,x2,y2,label], ...]
            # Note: boxes are normalized (0-1), need to convert to pixels
            for instance in target_list[i]:
                box_norm = instance[:4]  # [x1, y1, x2, y2] normalized
                label = int(instance[4])
                class_name = VOC_CLASSES[label]
                # Convert normalized coords to pixel coords
                box_pixels = [coord * YOLO_IMG_DIM for coord in box_norm]
                targets[(image_id, class_name)].append(box_pixels)

            # Get image dimensions (all images are resized to YOLO_IMG_DIM)
            # We need original dimensions to convert normalized coords back
            # Since we don't have access to original dims, we'll use YOLO_IMG_DIM
            # Note: This assumes target_list contains boxes in pixel coordinates at YOLO_IMG_DIM
            img_dim = YOLO_IMG_DIM

            # Process detections for this image
            if detections[i] is not None:
                for detection in detections[i]:
                    # Detection format: (x, y, w, h, obj_conf, class_conf, class_pred)
                    x_center = detection[0].item()
                    y_center = detection[1].item()
                    w = detection[2].item()
                    h = detection[3].item()
                    obj_conf = detection[4].item()
                    class_conf = detection[5].item()
                    class_id = int(detection[6].item())

                    # Convert from center format (normalized 0-1) to corner format (pixels)
                    x1 = int((x_center - w / 2) * img_dim)
                    y1 = int((y_center - h / 2) * img_dim)
                    x2 = int((x_center + w / 2) * img_dim)
                    y2 = int((y_center + h / 2) * img_dim)

                    # Clip to image boundaries
                    x1 = max(0, min(x1, img_dim - 1))
                    y1 = max(0, min(y1, img_dim - 1))
                    x2 = max(0, min(x2, img_dim - 1))
                    y2 = max(0, min(y2, img_dim - 1))

                    # Skip degenerate boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Combined confidence score
                    prob = obj_conf * class_conf
                    class_name = VOC_CLASSES[class_id]

                    # Add to predictions
                    preds[class_name].append([image_id, prob, x1, y1, x2, y2])

        global_idx += len(images)

    aps = voc_eval(preds, targets, VOC_CLASSES=VOC_CLASSES)
    return aps

def test_evaluate(model, test_dataset_file, img_root, test_loader=None):
    """
    Evaluate model on test set using built-in inference method.

    Args:
        model: ODModel instance with inference() method
        test_dataset_file: Path to test annotation file
        img_root: Root directory containing images
        test_loader: Optional (not used, kept for backward compatibility)

    Returns:
        preds_submission: Dictionary mapping image_id to list of detections
    """
    targets = defaultdict(list)
    preds = defaultdict(list)
    preds_submission = defaultdict(list)
    image_list = []  # image path list

    # Preprocessing transform (same as test_data_pipelines)
    transform = A.Compose([
        A.Resize(height=YOLO_IMG_DIM, width=YOLO_IMG_DIM),
        A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
        A.pytorch.ToTensorV2(),
    ])

    f = open(test_dataset_file)
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()

    # Collect ground truth targets
    for image_file in file_list:
        image_id = image_file[0]
        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = VOC_CLASSES[c]
            targets[(image_id, class_name)].append([x1, y1, x2, y2])

    print("---Evaluate model on test samples---")
    sys.stdout.flush()
    model.eval()

    # Run inference on all images
    for image_path in tqdm(image_list):
        # Load and preprocess image
        img_full_path = os.path.join(img_root, image_path)
        image = cv2.imread(img_full_path)
        if image is None:
            print(f"Warning: Could not load image {img_full_path}")
            continue

        orig_h, orig_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply preprocessing
        transformed = transform(image=image_rgb)
        img_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

        # Move to GPU if available
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Run inference
        with torch.no_grad():
            detections = model.inference(img_tensor)

        # Process detections for this image
        if detections[0] is not None:
            for detection in detections[0]:
                # Detection format: (x, y, w, h, obj_conf, class_conf, class_pred)
                x_center = detection[0].item()
                y_center = detection[1].item()
                w = detection[2].item()
                h = detection[3].item()
                obj_conf = detection[4].item()
                class_conf = detection[5].item()
                class_id = int(detection[6].item())

                # Convert from center format (normalized 0-1) to corner format (pixels)
                x1 = int((x_center - w / 2) * orig_w)
                y1 = int((y_center - h / 2) * orig_h)
                x2 = int((x_center + w / 2) * orig_w)
                y2 = int((y_center + h / 2) * orig_h)

                # Clip to image boundaries
                x1 = max(0, min(x1, orig_w - 1))
                y1 = max(0, min(y1, orig_h - 1))
                x2 = max(0, min(x2, orig_w - 1))
                y2 = max(0, min(y2, orig_h - 1))

                # Skip degenerate boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                # Combined confidence score
                prob = obj_conf * class_conf
                class_name = VOC_CLASSES[class_id]

                # Add to predictions
                preds[class_name].append([image_path, prob, x1, y1, x2, y2])
                preds_submission[image_path].append([class_name, prob, x1, y1, x2, y2])

    return preds_submission