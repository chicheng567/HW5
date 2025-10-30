import torch
import torch.nn as nn
import torch.nn.functional as F
class YOLOv3Loss(nn.Module):
    def __init__(
        self,
        lambda_coord=5.0,
        lambda_obj=1.0,
        lambda_noobj=0.5,
        lambda_class=1.0,
    ):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        predictions: list of 3 scales, each [batch, grid, grid, 3, 25]
        targets: (target_boxes, target_cls, target_obj)
        """
        target_boxes, target_cls, target_obj = targets
        
        total_box_loss = 0
        total_obj_loss = 0
        total_noobj_loss = 0
        total_cls_loss = 0
        for pred, gt in zip(predictions, targets):
            bsz, grid, _, num_anchors, _ = gt.shape
            pred = pred.view(bsz, grid, grid, num_anchors, -1) 
            # generate objectness mask
            obj_mask = gt[..., 4] == 1
            noobj_mask = gt[..., 4] == 0
            
            # 1. Box Loss (MSE)
            if obj_mask.sum() > 0:
                xy_loss = self.mse_loss(pred[..., 0:2], gt[..., 0:2])
                wh_loss = self.mse_loss(pred[..., 2:4], gt[..., 2:4])
                xy_loss = xy_loss[obj_mask].sum()
                wh_loss = wh_loss[obj_mask].sum()
                box_loss = xy_loss + wh_loss
            else:
                box_loss = 0

            # 2. Objectness Loss (BCE)
            obj_loss = self.bce_loss(pred[..., 4], gt[..., 4])
            #giving different weights to obj and noobj
            noobj_loss = obj_loss[noobj_mask].sum()
            obj_loss = obj_loss[obj_mask].sum()
            

            # 3. Class Loss (BCE)
            if obj_mask.sum() > 0:
                cls_loss = self.bce_loss(pred[..., 5:][obj_mask], gt[..., 5:][obj_mask]).sum()
            else:
                cls_loss = 0
            # 4. sum up losses accross scales
            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_noobj_loss += noobj_loss
            total_cls_loss += cls_loss
        
        # apply weights to each loss component
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
            'cls': total_cls_loss
        }
        
        return loss_dict
