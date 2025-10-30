# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YOLO v3 object detection implementation for the Pascal VOC dataset. The implementation uses a DenseNet/DarkNet backbone (via TIMM) with FPN-style neck and multi-scale predictions at 13x13, 26x26, and 52x52 grid sizes.

## Environment Setup

**Python Environment**: `/miniconda/envs/vl3/bin/python`

Always use this specific Python interpreter when running scripts or installing packages.

## Common Commands

### Training
```bash
# Train YOLO v3 model
/miniconda/envs/vl3/bin/python train_yolov3.py
```

### Evaluation
```bash
# Evaluate on validation set (in Python)
from src.eval_voc import evaluate
from src.yolo import resnet50
model = resnet50(pretrained=False)
model.load_state_dict(torch.load('checkpoints/best_detector.pth'))
val_aps = evaluate(model, val_dataset_file='vocall_val.txt', img_root='./VOC/JPEGImages/')
```

### Generate Anchors
```bash
# Generate custom anchors using K-means on training data
/miniconda/envs/vl3/bin/python generate_anchors.py
```

### Testing Backbone
```bash
# Test backbone architecture
/miniconda/envs/vl3/bin/python src/test_backbone.py
```

## Architecture

### Modular Design

The codebase follows a strict modular design philosophy:

**1. Model Architecture (`src/yolo.py`)**
- `Backbone`: Feature extractor using TIMM models (currently DarkNet53, configurable)
  - Returns 3 feature maps at different scales (13x13, 26x26, 52x52)
- `YOLOv3Head`: Neck + Prediction head combined
  - FPN-like structure with upsampling and concatenation
  - Outputs predictions at 3 scales
- `ODModel`: Top-level model class that encapsulates everything
  - Handles inference mode with NMS
  - Factory function `resnet50()` creates complete model

**2. Loss Function (`yolo_loss.py`)**
- `YOLOv3Loss`: Modular loss with separate components
  - Box loss (MSE on xy and wh)
  - Objectness loss (BCE on object confidence)
  - No-object loss (BCE on background)
  - Class loss (BCE on class probabilities)

**3. Dataset (`src/dataset.py`)**
- `VocDetectorDataset`: Handles Pascal VOC format
  - Returns encoded targets when `encode_target=True`
  - Returns raw boxes/labels when `encode_target=False`
- `collate_fn`: Batches data for different modes
  - Training mode: Returns (images, targets) where targets is list of 3 tensors
  - Evaluation mode: Returns (images, target_list) where target_list contains raw boxes

### Multi-Scale Predictions

YOLO v3 predicts at 3 different scales:
- **Scale 1 (13x13)**: Detects large objects
- **Scale 2 (26x26)**: Detects medium objects
- **Scale 3 (52x52)**: Detects small objects

Each scale uses 3 anchor boxes (9 total anchors). Anchors are defined in `src/config.py` and can be regenerated using `generate_anchors.py`.

### Target Encoding

Ground truth boxes are assigned to the anchor with highest IoU (width-height only) at the appropriate scale. The encoding process:

1. For each GT box, find best matching anchor across all scales
2. Compute grid cell containing box center
3. Encode target as:
   - `tx, ty`: Offsets within grid cell (0-1)
   - `tw, th`: log(w/anchor_w), log(h/anchor_h)
   - Class: One-hot encoded (20 classes)
   - Objectness: 1 if object present, 0 otherwise

Target format: `[grid, grid, 3, 25]` where 25 = 4 (box) + 1 (obj) + 20 (classes)

### Prediction Decoding

During inference (`src/predict.py`):
1. Apply sigmoid to xy offsets and objectness
2. Apply sigmoid to class scores (multi-label)
3. Decode boxes: `cx = (grid_x + sigmoid(tx)) / grid_size`
4. Decode sizes: `w = exp(tw) * anchor_w`
5. Apply per-class NMS with IoU threshold 0.3

## Key Implementation Details

### Model Forward Pass Modes

The `ODModel` has two modes:
- **Training mode** (`model.train()`): Returns raw predictions as tuple of 3 tensors
- **Inference mode** (`model.inference()`): Applies NMS and returns list of detections

For validation loss computation, use raw predictions:
```python
pred = net(images)  # Training mode - returns raw tensors
```

### Loss Function Interface

The loss expects:
```python
loss_dict = criterion(predictions, targets)
```
- `predictions`: Tuple of 3 tensors (one per scale), each `[B, H, W, 75]`
- `targets`: Tuple of 3 tensors (one per scale), each `[B, H, W, 3, 25]`

Returns dictionary with keys: `total`, `box`, `obj`, `noobj`, `cls`

### DataLoader Configuration

**Must use `collate_fn`** in all DataLoaders:
```python
from src.dataset import collate_fn
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, ...)
```

### Image Preprocessing

- Input size: 416x416 (defined in `src/config.py` as `YOLO_IMG_DIM`)
- Normalization: Uses VOC-specific mean/std from `src/config.py`
- Training augmentations: Random crops, flips, rotation, brightness/contrast (albumentations)
- Test augmentations: Only resize and normalize

### Anchor Boxes

Anchors are normalized to [0, 1] range relative to image dimensions. Current anchors in `src/config.py` were generated using K-means clustering on the training set. To regenerate with different settings, modify and run `generate_anchors.py`.

## File Structure

- `train_yolov3.py`: Main training script with complete training loop
- `yolo_loss.py`: YOLO v3 loss implementation
- `generate_anchors.py`: K-means anchor generation utility
- `src/yolo.py`: Model architecture (Backbone, Head, ODModel)
- `src/dataset.py`: Dataset and data loading utilities
- `src/config.py`: Configuration constants (classes, anchors, image size, etc.)
- `src/eval_voc.py`: VOC evaluation metrics (mAP calculation)
- `src/predict.py`: Inference and prediction utilities
- `checkpoints/`: Saved model weights
- `VOC/`: Pascal VOC dataset
- `vocall_train.txt`, `vocall_val.txt`: Dataset annotation files

## Important Notes

### Backbone Switching

To change the backbone, modify the `model_name` parameter in `Backbone.__init__()` (src/yolo.py:16):
```python
self.backbone = timm.create_model(
    model_name,  # Change this (e.g., "densenet121", "resnet50", "efficientnet_b0")
    pretrained=pretrained,
    features_only=True
)
```

You may need to adjust channel dimensions in `YOLOv3Head` if the backbone outputs different feature map sizes.

### Grid Sizes

Grid sizes are fixed at [13, 26, 52] (defined in `src/config.py`). These must match the actual output sizes from the backbone at different scales. If changing image size or backbone, verify feature map dimensions match.

### NMS Configuration

NMS thresholds are set in `ODModel.__init__()`:
- `conf_thres`: 0.5 (objectness threshold)
- `nms_thres`: 0.4 (IoU threshold for NMS)

During prediction (src/predict.py), lower thresholds (0.05 confidence, 0.3 NMS) are used for evaluation.

### Validation Loss vs mAP

During training:
- Validation loss uses raw predictions (no NMS)
- mAP evaluation uses inference mode with NMS
- These are computed separately and may not correlate perfectly

## Training Configuration

Default hyperparameters in `train_yolov3.py`:
- Batch size: 35
- Learning rate: 1e-3 (decayed by 10x at epochs 30 and 40)
- Optimizer: AdamW with weight decay 5e-4
- Loss weights: coord=5.0, obj=1.0, noobj=0.5, class=1.0
- Epochs: 50
- mAP evaluation: Every 5 epochs

Checkpoints saved:
- `best_detector.pth`: Best validation loss
- `detector.pth`: Latest epoch
- `detector_epoch_X.pth`: Snapshots at epochs 5, 10, 20, 30, 40
