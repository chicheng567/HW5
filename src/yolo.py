import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from src.config import GRID_SIZES
class Backbone(nn.Module):
    """
    DenseNet backbone for feature extraction.
    Extracts features at multiple scales for YOLO v3.
    """
    def __init__(self, model_name="darknet53", pretrained=True):
        super(Backbone, self).__init__()
        ### Change here to use Different backbone ###
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True
        )
    def forward(self, x):
        features = self.backbone(x)
        return features[-1], features[-2], features[-3]  # Return feature maps at 3 scales


# ============================================================================
# Neck & Prediction Head
# ============================================================================
class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class YOLOv3Head(nn.Module):
    """
    YOLO v3 detection head with FPN-like neck structure.
    Performs multi-scale predictions.
    """
    def __init__(self, num_classes=20, num_anchors=3):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.output_channels = num_anchors * (5 + num_classes)
        self.scale1_conv = nn.Sequential(
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
        )
        # Classifier for scale 1
        self.scale1_detect_conv = nn.Sequential(
            ConvBlock(512, 1024, 3, 1, 1),
            nn.Conv2d(1024, self.output_channels, 1, 1, 0)
        )

        # Upsample for scale 2
        self.scale_13_upsample = nn.Sequential(
            ConvBlock(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # ==== Scale 2: 26x26 (medium scale - detects medium objects) ====
        self.scale2_conv = nn.Sequential(
            ConvBlock(512 + 256, 256, 1, 1, 0),  # 512 from backbone + 256 from upsampled
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
        )
        self.scale2_detect_conv = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1),
            nn.Conv2d(512, self.output_channels, 1, 1, 0)
        )

        # Upsample for scale 3
        self.scale_26_upsample = nn.Sequential(
            ConvBlock(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # ==== Scale 3: 52x52 (large scale - detects small objects) ====
        self.scale3_conv = nn.Sequential(
            ConvBlock(256 + 128, 128, 1, 1, 0),  # 256 from backbone + 128 from upsampled
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
        )
        self.scale3_detect_conv = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1),
            nn.Conv2d(256, self.output_channels, 1, 1, 0)
        )

    def forward(self, features):
        """
        Args:
            features: Tuple of (feat_13x13, feat_26x26, feat_52x52)

        Returns:
            Tuple of (pred_13x13, pred_26x26, pred_52x52)
            Each prediction shape: (B, H, W, num_anchors * (5 + num_classes))
        """
        feat_13, feat_26, feat_52 = features
        
        # Scale 1: 13x13
        x1 = self.scale1_conv(feat_13)
        pred_13 = self.scale1_detect_conv(x1)
        # Prepare for scale 2
        x1_up = self.scale_13_upsample(x1)
        # Scale 2: 26x26
        x2 = torch.cat([x1_up, feat_26], dim=1)
        x2 = self.scale2_conv(x2)
        pred_26 = self.scale2_detect_conv(x2)

        # Prepare for scale 3
        x2_up = self.scale_26_upsample(x2)

        # Scale 3: 52x52
        x3 = torch.cat([x2_up, feat_52], dim=1)
        x3 = self.scale3_conv(x3)
        pred_52 = self.scale3_detect_conv(x3)
        
        # Reshape predictions: (B, C, H, W) -> (B, H, W, C)
        pred_13 = pred_13.permute(0, 2, 3, 1).contiguous()
        pred_26 = pred_26.permute(0, 2, 3, 1).contiguous()
        pred_52 = pred_52.permute(0, 2, 3, 1).contiguous()

        return pred_13, pred_26, pred_52
# ============================================================================
# NMS for inference
# ============================================================================
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.

    Args:
        prediction: (batch_size, num_boxes, 5 + num_classes)
                   where 5 = (x, y, w, h, objectness)
        conf_thres: object confidence threshold
        nms_thres: IOU threshold for NMS

    Returns:
        detections: List of detections for each image in batch
                   Each detection: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5:], 1, keepdim=True)

        # Detections: (x, y, w, h, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # Perform NMS
        unique_labels = detections[:, -1].unique()

        for c in unique_labels:
            # Get detections of this class
            detections_class = detections[detections[:, -1] == c]

            # Sort by confidence
            conf_sort_index = torch.argsort(detections_class[:, 4] * detections_class[:, 5], descending=True)
            detections_class = detections_class[conf_sort_index]

            # NMS
            keep_boxes = []
            while detections_class.size(0):
                # Keep box with highest confidence
                keep_boxes.append(detections_class[0:1])

                if len(detections_class) == 1:
                    break

                # Compute IoU with remaining boxes
                ious = bbox_iou(detections_class[0:1, :4], detections_class[1:, :4])

                # Remove boxes with IoU > threshold
                detections_class = detections_class[1:][ious.view(-1) < nms_thres]

            if len(keep_boxes) > 0:
                keep_boxes = torch.cat(keep_boxes, 0)
                if output[image_i] is None:
                    output[image_i] = keep_boxes
                else:
                    output[image_i] = torch.cat((output[image_i], keep_boxes), 0)
    return output

def bbox_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    Boxes are in (x, y, w, h) format.
    """
    # Get coordinates of bounding boxes
    b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

    # Get intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area.unsqueeze(1) + b2_area - inter_area + 1e-16)

    return iou


# ============================================================================
# Object Detection Model
# ============================================================================
from src.config import ANCHORS
class ODModel(nn.Module):
    """
    Complete YOLO v3 Object Detection Model.
    Combines DenseNet backbone with YOLO v3 detection head.
    """
    def __init__(self, num_classes=20, num_anchors=3, pretrained=True, nms_thres=0.4, conf_thres=0.5):
        super(ODModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.backbone = Backbone(pretrained=pretrained, model_name="darknet53")
        self.head = YOLOv3Head(num_classes=num_classes, num_anchors=num_anchors)
        self.anchors = ANCHORS
        self.nms_thres = nms_thres
        self.conf_thres = conf_thres
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Get predictions at 3 scales
        predictions = self.head(features)

        return predictions
    def inference(self, x):
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            predictions = self.head(features)
            # Reshape and concatenate all predictions
            pred_13, pred_26, pred_52 = predictions

            # Apply sigmoid to predictions
            pred_13 = self._transform_predictions(pred_13, self.anchors[0])
            pred_26 = self._transform_predictions(pred_26, self.anchors[1])
            pred_52 = self._transform_predictions(pred_52, self.anchors[2])

            # Concatenate all scales
            batch_size = x.size(0)
            all_predictions = []

            for i in range(batch_size):
                pred_i = torch.cat([
                    pred_13[i].view(-1, 5 + self.num_classes),
                    pred_26[i].view(-1, 5 + self.num_classes),
                    pred_52[i].view(-1, 5 + self.num_classes)
                ], dim=0)
                all_predictions.append(pred_i)

            all_predictions = torch.stack(all_predictions, dim=0)

            # Apply NMS
            output = non_max_suppression(all_predictions)

            return output
    def _transform_predictions(self, pred):
        """
        Transform raw predictions to actual bbox coordinates.

        Args:
            pred: (B, H, W, num_anchors * (5 + num_classes))
            anchors: List of (w, h) tuples
            stride: Stride of the feature map

        Returns:
            Transformed predictions with actual coordinates
        """
        batch_size = pred.size(0)
        grid_size = pred.size(1)

        # Reshape: (B, H, W, num_anchors * (5 + C)) -> (B, H, W, num_anchors, 5 + C)
        pred = pred.view(batch_size, grid_size, grid_size, self.num_anchors, 5 + self.num_classes)
        # Get outputs
        x = torch.sigmoid(pred[..., 0])  # Center x
        y = torch.sigmoid(pred[..., 1])  # Center y
        w = pred[..., 2]  # Width
        h = pred[..., 3]  # Height
        obj_conf = torch.sigmoid(pred[..., 4])  # Objectness
        cls_conf = torch.softmax(pred[..., 5:], dim=-1)  # classification scores

        # Calculate offsets for each grid
        grid_x = torch.arange(grid_size, dtype=torch.float, device=pred.device).repeat(grid_size, 1).view(1, grid_size, grid_size, 1)
        grid_y = torch.arange(grid_size, dtype=torch.float, device=pred.device).repeat(grid_size, 1).t().view(1, grid_size, grid_size, 1)

        # Anchor dimensions
        anchor_w = torch.tensor([a[0] for a in self.anchors], dtype=torch.float, device=pred.device).view(1, 1, 1, self.num_anchors)
        anchor_h = torch.tensor([a[1] for a in self.anchors], dtype=torch.float, device=pred.device).view(1, 1, 1, self.num_anchors)

        # Add offset and scale with anchors
        # Clamp w and h before exp to prevent overflow (clamp to [-10, 10])
        w_clamped = torch.clamp(w, -10, 10)
        h_clamped = torch.clamp(h, -10, 10)

        pred_boxes = torch.zeros_like(pred[..., :4])
        pred_boxes[..., 0] = (x + grid_x) / grid_size  # Normalize to 0-1
        pred_boxes[..., 1] = (y + grid_y) / grid_size  # Normalize to 0-1
        pred_boxes[..., 2] = torch.exp(w_clamped) * anchor_w
        pred_boxes[..., 3] = torch.exp(h_clamped) * anchor_h

        # Concatenate predictions
        output = torch.cat([
            pred_boxes,
            obj_conf.unsqueeze(-1),
            cls_conf
        ], dim=-1)

        return output


# ============================================================================
# Factory function (保持與原始程式碼的相容性)
# ============================================================================
def resnet50(pretrained=True):
    """
    Factory function to create YOLO v3 model with DenseNet backbone.
    Renamed to keep compatibility with existing code.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    Returns:
        ODModel: Complete YOLO v3 object detection model
    """
    model = ODModel(num_classes=20, num_anchors=3, pretrained=pretrained)
    return model
