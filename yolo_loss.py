import torch
import torch.nn as nn
import torch.nn.functional as F
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type
    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4]
        target_boxes: [bsz, grid, grid, anchors, 4]
        anchors: list of (w, h) for the anchors at this scale
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        anchors = torch.tensor(anchors, device=pred_boxes.device).unsqueeze(0)  # [1, 3, 2]
        pred_boxes = pred_boxes.view(-1, num_anchors, 4)
        target_boxes = target_boxes.view(-1, num_anchors, 4)
        if self.type == 'giou':
            pxy = torch.sigmoid(pred_boxes[..., :2])  # Apply sigmoid to tx, ty
            pwh = torch.exp(pred_boxes[..., 2:].clamp(max=1e3))  # Apply exp to tw, th
            pwh = pwh * anchors  # Scale by anchors
            txy, twh = target_boxes[..., :2], target_boxes[..., 2:]
            # Find the smallest enclosing box
            p_x1y1 = pxy - pwh / 2
            p_x2y2 = pxy + pwh / 2
            t_x1y1 = txy - twh / 2
            t_x2y2 = txy + twh / 2
            # enclosing box
            c_x1y1 = torch.min(p_x1y1, t_x1y1)
            c_x2y2 = torch.max(p_x2y2, t_x2y2)
            # 中心點都在同一個grid，所以不需要加上grid偏移，WH的時候會被消掉
            # 所有的面積都被 * grid_size^2 ，但是因為算比例所以也被抵消了
            c_area = (c_x2y2[..., 0] - c_x1y1[..., 0]) * (c_x2y2[..., 1] - c_x1y1[..., 1])
            # iou
            eps = 1e-7
            # intersection
            i_x1y1 = torch.max(p_x1y1, t_x1y1)
            i_x2y2 = torch.min(p_x2y2, t_x2y2)
            i_wh = (i_x2y2 - i_x1y1).clamp(min=0)
            i_area = (i_wh[..., 0] * i_wh[..., 1])

            # areas
            p_area = (pwh[..., 0] * pwh[..., 1])
            t_area = (twh[..., 0] * twh[..., 1])

            # union and IoU
            union = p_area + t_area - i_area
            iou = i_area / (union + eps)

            # GIoU and loss
            giou = iou - (c_area - union) / (c_area + eps)
            giou_loss = 1.0 - giou
            return giou_loss.view(bsz, grid, grid, num_anchors)
        elif self.type == 'mse':
            mse_loss = F.mse_loss(pred_boxes, target_boxes, reduction='none')
            pxy = torch.sigmoid(pred_boxes[..., :2])  # Apply sigmoid to tx, ty
            pwh = torch.exp(pred_boxes[..., 2:].clamp(max=1e3))  # Apply exp to tw, th
            pwh = pwh * anchors  # Scale by anchors
            txy, twh = target_boxes[..., :2], target_boxes[..., 2:]
            mse_loss_xy = F.mse_loss(pxy, txy, reduction='none')
            mse_loss_wh = F.mse_loss(pwh, twh, reduction='none')
            return (mse_loss_xy + mse_loss_wh).view(bsz, grid, grid, num_anchors)
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
        anchors=None,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.box_loss = BoxLoss()
        self.anchors = anchors  # List of anchor boxes per scale
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 75]
        targets: list of 3 scales, each [batch, grid, grid, 3, 25]
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_noobj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)

        total_num_pos = 0
        total_num_neg = 0
        batch_size = predictions[0].size(0)

        for pred, gt, anchors in zip(predictions, targets, self.anchors):
            bsz, grid, _, num_anchors, _ = gt.shape
            # Reshape prediction: [B, H, W, 75] -> [B, H, W, 3, 25]
            pred = pred.view(bsz, grid, grid, num_anchors, -1)

            # Create masks
            obj_mask = (gt[..., 4] == 1)
            noobj_mask = (gt[..., 4] == 0)

            # Count samples
            num_pos = obj_mask.sum().item()
            num_neg = noobj_mask.sum().item()
            total_num_pos += num_pos
            total_num_neg += num_neg

            # Box loss (only for positive samples)
            if num_pos > 0:
                ####box loss####
                box_loss = self.box_loss(
                    pred[..., :4],
                    gt[..., :4],
                    anchors
                )
                total_box_loss += box_loss[obj_mask].sum()
                ################
                # Class loss
                cls_loss = self.bce_loss(
                    pred[..., 5:],
                    gt[..., 5:]
                )
                total_cls_loss += cls_loss[obj_mask].sum()

            # Objectness loss for positive samples
            obj_loss_pos = self.bce_loss(
                pred[..., 4],
                gt[..., 4]
            )
            total_obj_loss += obj_loss_pos[obj_mask].sum()
            total_noobj_loss += obj_loss_pos[noobj_mask].sum()

        # Box, obj, and cls losses are normalized by number of positive samples
        # NoObj loss is normalized by number of negative samples
        total_box_loss = total_box_loss / batch_size
        total_obj_loss = total_obj_loss / batch_size
        total_cls_loss = total_cls_loss / batch_size

        total_noobj_loss = total_noobj_loss / batch_size

        # Combined loss
        
        total_loss = (
            self.lambda_coord * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_cls_loss
        )
        
        loss_dict = {
            'total': total_loss,
            'box': total_box_loss,
            'obj': total_obj_loss,
            'noobj': total_noobj_loss,
            'cls': total_cls_loss,
        }
        
        return loss_dict
