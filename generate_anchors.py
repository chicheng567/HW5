"""
Generate YOLO v3 anchor boxes using K-means clustering on the dataset.
This produces anchors optimized for the Pascal VOC dataset.
"""

import numpy as np
from tqdm import tqdm
from PIL import Image
import os


def iou_width_height(boxes, anchors):
    """
    Compute IoU between boxes and anchors based only on width and height.
    Assumes boxes and anchors are centered at the same point.

    Args:
        boxes: (N, 2) array of [width, height]
        anchors: (K, 2) array of [width, height]

    Returns:
        iou: (N, K) array of IoU values
    """
    intersection = np.minimum(boxes[:, 0:1], anchors[:, 0]) * \
                   np.minimum(boxes[:, 1:2], anchors[:, 1])

    box_area = boxes[:, 0] * boxes[:, 1]
    anchor_area = anchors[:, 0] * anchors[:, 1]

    union = box_area[:, None] + anchor_area - intersection

    return intersection / (union + 1e-16)


def kmeans_anchors(boxes, k=9, max_iter=300, verbose=True):
    """
    Run K-means clustering to generate anchor boxes.

    Args:
        boxes: (N, 2) array of [width, height] in normalized format (0-1)
        k: Number of anchors to generate (default: 9 for YOLO v3)
        max_iter: Maximum iterations
        verbose: Print progress

    Returns:
        anchors: (k, 2) array of anchor [width, height]
    """
    n = boxes.shape[0]

    # Initialize anchors randomly from the boxes
    indices = np.random.choice(n, k, replace=False)
    anchors = boxes[indices]

    if verbose:
        print(f"Running K-means with k={k} on {n} boxes...")

    for iteration in range(max_iter):
        # Assign each box to nearest anchor (using IoU as distance metric)
        ious = iou_width_height(boxes, anchors)
        assignments = np.argmax(ious, axis=1)

        # Update anchors to mean of assigned boxes
        old_anchors = anchors.copy()
        for i in range(k):
            if np.sum(assignments == i) > 0:
                anchors[i] = boxes[assignments == i].mean(axis=0)

        # Check convergence
        if np.allclose(old_anchors, anchors, rtol=1e-5):
            if verbose:
                print(f"Converged at iteration {iteration + 1}")
            break

    # Sort anchors by area (small to large)
    areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(areas)
    anchors = anchors[sorted_indices]

    # Compute average IoU
    ious = iou_width_height(boxes, anchors)
    best_ious = np.max(ious, axis=1)
    avg_iou = np.mean(best_ious)

    if verbose:
        print(f"Average IoU: {avg_iou:.4f}")

    return anchors, avg_iou


def load_voc_boxes(dataset_file, img_dir='VOC/JPEGImages'):
    """
    Load all bounding boxes from VOC dataset file with correct normalization.

    Args:
        dataset_file: Path to dataset annotation file
        img_dir: Directory containing the actual images

    Returns:
        boxes: (N, 2) array of normalized [width, height] (0-1 range)
    """
    boxes = []
    skipped = 0
    total = 0

    with open(dataset_file, 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Loading boxes"):
        parts = line.strip().split()
        img_name = parts[0]
        num_boxes = (len(parts) - 1) // 5

        # Get actual image dimensions
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            skipped += num_boxes
            continue

        try:
            img = Image.open(img_path)
            img_w, img_h = img.size
        except Exception as e:
            print(f"Warning: Cannot read image {img_path}: {e}")
            skipped += num_boxes
            continue

        for i in range(num_boxes):
            total += 1
            x1 = float(parts[1 + 5 * i])
            y1 = float(parts[2 + 5 * i])
            x2 = float(parts[3 + 5 * i])
            y2 = float(parts[4 + 5 * i])

            # Compute width and height in pixels
            w = x2 - x1
            h = y2 - y1

            # Normalize by ACTUAL image dimensions (not target size)
            # This gives us true 0-1 normalized values
            w_norm = w / img_w
            h_norm = h / img_h

            # Sanity check: normalized values should be in (0, 1]
            if w_norm > 0 and h_norm > 0 and w_norm <= 1.0 and h_norm <= 1.0:
                boxes.append([w_norm, h_norm])
            else:
                skipped += 1

    if skipped > 0:
        print(f"Warning: Skipped {skipped} out of {total} boxes due to invalid dimensions")

    return np.array(boxes)


def generate_yolo_v3_anchors(train_file='vocall_train.txt', img_dir='VOC/JPEGImages'):
    """
    Generate YOLO v3 anchors for 3 scales using K-means clustering.

    Args:
        train_file: Path to training annotation file
        img_dir: Directory containing the actual images

    Returns:
        anchors_by_scale: List of 3 arrays, each (3, 2) for [width, height]
    """
    print("="*80)
    print("Generating YOLO v3 Anchors using K-means Clustering")
    print("="*80)

    # Load all boxes with correct normalization
    boxes = load_voc_boxes(train_file, img_dir)
    print(f"\nLoaded {len(boxes)} boxes from dataset")
    print(f"Box size range (normalized 0-1):")
    print(f"  Width:  [{boxes[:, 0].min():.4f}, {boxes[:, 0].max():.4f}]")
    print(f"  Height: [{boxes[:, 1].min():.4f}, {boxes[:, 1].max():.4f}]")

    # Run K-means to get 9 anchors
    print("\nRunning K-means clustering to find 9 optimal anchors...")
    anchors, avg_iou = kmeans_anchors(boxes, k=9, verbose=True)

    # Split into 3 scales (3 anchors each)
    # Small anchors for large grid (52x52 or 56x56)
    # Medium anchors for medium grid (26x26 or 28x28)
    # Large anchors for small grid (13x13 or 14x14)
    anchors_scale_1 = anchors[6:9]  # Largest objects (for 13x13 grid)
    anchors_scale_2 = anchors[3:6]  # Medium objects (for 26x26 grid)
    anchors_scale_3 = anchors[0:3]  # Smallest objects (for 52x52 grid)

    print("\n" + "="*80)
    print("Generated Anchors (normalized 0-1, relative to image dimensions):")
    print("="*80)
    print(f"\nScale 1 (13x13 grid - large objects):")
    for i, anchor in enumerate(anchors_scale_1):
        print(f"  Anchor {i+1}: width={anchor[0]:.4f}, height={anchor[1]:.4f}")

    print(f"\nScale 2 (26x26 grid - medium objects):")
    for i, anchor in enumerate(anchors_scale_2):
        print(f"  Anchor {i+1}: width={anchor[0]:.4f}, height={anchor[1]:.4f}")

    print(f"\nScale 3 (52x52 grid - small objects):")
    for i, anchor in enumerate(anchors_scale_3):
        print(f"  Anchor {i+1}: width={anchor[0]:.4f}, height={anchor[1]:.4f}")

    print(f"\nAverage IoU with ground truth boxes: {avg_iou:.4f}")
    print("\nNote: These anchors are normalized (0-1 range).")
    print("All values should be <= 1.0 for proper YOLO training.")
    print("="*80)

    # Format for easy copy-paste into code
    print("\nPython code format:")
    print("anchors = [")
    print(f"    # Scale 1 (13x13): Large anchors for large objects")
    print(f"    [", end="")
    for i, anchor in enumerate(anchors_scale_1):
        if i > 0:
            print(", ", end="")
        print(f"({anchor[0]:.4f}, {anchor[1]:.4f})", end="")
    print("],")

    print(f"    # Scale 2 (26x26): Medium anchors for medium objects")
    print(f"    [", end="")
    for i, anchor in enumerate(anchors_scale_2):
        if i > 0:
            print(", ", end="")
        print(f"({anchor[0]:.4f}, {anchor[1]:.4f})", end="")
    print("],")

    print(f"    # Scale 3 (52x52): Small anchors for small objects")
    print(f"    [", end="")
    for i, anchor in enumerate(anchors_scale_3):
        if i > 0:
            print(", ", end="")
        print(f"({anchor[0]:.4f}, {anchor[1]:.4f})", end="")
    print("],")
    print("]")

    return [anchors_scale_1, anchors_scale_2, anchors_scale_3], avg_iou


if __name__ == '__main__':
    # Generate anchors using K-means clustering on the training dataset
    # Anchors are normalized (0-1 range) relative to image dimensions
    anchors, avg_iou = generate_yolo_v3_anchors(
        train_file='vocall_train.txt',
        img_dir='VOC/JPEGImages'
    )

    print(f"\nAnchors generated successfully!")
    print(f"Copy the Python code above to your config file.")
