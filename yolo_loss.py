import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: predictions after sigmoid, shape [N, *]
        targets: ground truth labels (0 or 1), shape [N, *]
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class BoxLoss(nn.Module):
    def __init__(self, loss_type='giou'):
        super(BoxLoss, self).__init__()
        self.type = loss_type

    def forward(self, pred_boxes, target_boxes, anchors):
        """
        pred_boxes: [bsz, grid, grid, anchors, 4] (raw predictions)
        target_boxes: [bsz, grid, grid, anchors, 4] (encoded targets)
        anchors: list of (w, h) for the anchors at this scale (normalized 0-1)
        """
        bsz, grid, _, num_anchors, _ = pred_boxes.size()
        device = pred_boxes.device
        dtype = pred_boxes.dtype

        anchors = torch.tensor(anchors, device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)

        # coordinate offset for each grid cell
        grid_range = torch.arange(grid, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(grid_range, grid_range, indexing='ij')
        grid_x = grid_x.view(1, grid, grid, 1, 1)
        grid_y = grid_y.view(1, grid, grid, 1, 1)

        if self.type == 'giou':
            # predicted centre (cell offset) and size
            pxy = torch.sigmoid(pred_boxes[..., :2])
            pwh = torch.exp(pred_boxes[..., 2:].clamp(min=-10, max=10)) * anchors

            # target centre still stored as cell offset, convert to same system
            txy = target_boxes[..., :2]
            twh = target_boxes[..., 2:]

            # Convert both to image-normalised coordinates
            pred_center_x = (pxy[..., 0].unsqueeze(-1) + grid_x) / grid
            pred_center_y = (pxy[..., 1].unsqueeze(-1) + grid_y) / grid
            targ_center_x = (txy[..., 0].unsqueeze(-1) + grid_x) / grid
            targ_center_y = (txy[..., 1].unsqueeze(-1) + grid_y) / grid

            # boxes to corner format
            p_x1 = pred_center_x - pwh[..., 0].unsqueeze(-1) / 2
            p_y1 = pred_center_y - pwh[..., 1].unsqueeze(-1) / 2
            p_x2 = pred_center_x + pwh[..., 0].unsqueeze(-1) / 2
            p_y2 = pred_center_y + pwh[..., 1].unsqueeze(-1) / 2

            t_x1 = targ_center_x - twh[..., 0].unsqueeze(-1) / 2
            t_y1 = targ_center_y - twh[..., 1].unsqueeze(-1) / 2
            t_x2 = targ_center_x + twh[..., 0].unsqueeze(-1) / 2
            t_y2 = targ_center_y + twh[..., 1].unsqueeze(-1) / 2

            # Intersection box
            i_x1 = torch.max(p_x1, t_x1)
            i_y1 = torch.max(p_y1, t_y1)
            i_x2 = torch.min(p_x2, t_x2)
            i_y2 = torch.min(p_y2, t_y2)
            i_w = (i_x2 - i_x1).clamp(min=0)
            i_h = (i_y2 - i_y1).clamp(min=0)
            i_area = i_w * i_h

            # Areas
            p_area = ((p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0))
            t_area = ((t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0))

            union = p_area + t_area - i_area
            eps = 1e-7
            iou = i_area / (union + eps)

            # smallest enclosing box
            c_x1 = torch.min(p_x1, t_x1)
            c_y1 = torch.min(p_y1, t_y1)
            c_x2 = torch.max(p_x2, t_x2)
            c_y2 = torch.max(p_y2, t_y2)
            c_w = (c_x2 - c_x1).clamp(min=eps)
            c_h = (c_y2 - c_y1).clamp(min=eps)
            c_area = c_w * c_h

            giou = iou - (c_area - union) / (c_area + eps)
            giou_loss = 1.0 - giou
            return giou_loss.squeeze(-1)

        elif self.type == 'mse':
            pxy = torch.sigmoid(pred_boxes[..., :2])
            pwh = torch.exp(pred_boxes[..., 2:].clamp(min=-10, max=10)) * anchors
            txy, twh = target_boxes[..., :2], target_boxes[..., 2:]

            pred_center = torch.stack([(pxy[..., 0] + grid_x.squeeze(-1)) / grid,
                                       (pxy[..., 1] + grid_y.squeeze(-1)) / grid], dim=-1)
            targ_center = torch.stack([(txy[..., 0] + grid_x.squeeze(-1)) / grid,
                                       (txy[..., 1] + grid_y.squeeze(-1)) / grid], dim=-1)

            mse_loss_xy = F.mse_loss(pred_center, targ_center, reduction='none').sum(-1)
            mse_loss_wh = F.mse_loss(pwh, twh, reduction='none').sum(-1)
            return mse_loss_xy + mse_loss_wh
        else:
            raise NotImplementedError(f"Box loss type '{self.type}' not implemented.")
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=2.0,
        lambda_obj=1.0,
        lambda_noobj=0.2,
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
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.focal_loss = FocalLoss(reduction='none')
        self.box_loss = BoxLoss(loss_type='giou')
        self.anchors = anchors  # List of anchor boxes per scale
    # Check for NaNs in any of the loss scalars and print which one is NaN
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 75]
        targets: list of 3 scales, each [batch, grid, grid, 3, 25]
        """
        device = predictions[0].device

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss_pos = torch.tensor(0.0, device=device)
        total_obj_loss_neg = torch.tensor(0.0, device=device)
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
                assert torch.isnan(box_loss).sum() == 0, "Box loss contains NaN values"
                total_box_loss += box_loss[obj_mask].sum()
                ################
                # Class loss
                cls_loss = self.bce_loss(
                    pred[..., 5:],  # [B, H, W, A, C]
                    gt[..., 5:]
                )
                
                total_cls_loss += cls_loss[obj_mask].sum()

            # Objectness loss for positive samples
            obj_loss = self.focal_loss(
                pred[..., 4],
                gt[..., 4]
            )
            total_obj_loss_pos += obj_loss[obj_mask].sum()
            total_obj_loss_neg += obj_loss[noobj_mask].sum()

        pos_denom = max(total_num_pos, 1)
        neg_denom = max(total_num_neg, 1)

        total_box_loss = total_box_loss / pos_denom
        total_obj_loss = total_obj_loss_pos / pos_denom
        total_cls_loss = total_cls_loss / pos_denom
        total_noobj_loss = total_obj_loss_neg / neg_denom

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
