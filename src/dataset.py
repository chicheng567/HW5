import os
import cv2
import numpy as np
import torch
import torch.utils.data as DataLoader
import albumentations as A
from src.config import VOC_IMG_MEAN, VOC_IMG_STD, YOLO_IMG_DIM, ANCHORS, GRID_SIZES
train_data_pipelines = A.Compose([
    # A.RandomSizedBBoxSafeCrop(
    #     width=YOLO_IMG_DIM, 
    #     height=YOLO_IMG_DIM, 
    #     erosion_rate=0,
    #     p=0.3
    # ),
    A.Resize(YOLO_IMG_DIM, YOLO_IMG_DIM),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    # A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    # A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    A.pytorch.ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['cls_labels']
))

test_data_pipelines = A.Compose([
    A.Resize(height=YOLO_IMG_DIM, width=YOLO_IMG_DIM),
    A.Normalize(mean=VOC_IMG_MEAN, std=VOC_IMG_STD),
    A.pytorch.ToTensorV2(),
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['cls_labels'])
)

class VocDetectorDataset(DataLoader.Dataset):
    image_size = YOLO_IMG_DIM
    def __init__(
        self,
        root_img_dir,
        dataset_file,
        train,
        contain_labels=True,
        num_classes=20,
        grid_sizes=[13, 26, 52],
        transform=None,
        return_image_id=False,
        encode_target=True,
    ):
        print("Initializing dataset")
        self.root = root_img_dir
        self.contain_labels = contain_labels
        self.train = train
        self.transform = transform if transform is not None else test_data_pipelines
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.grid_sizes = grid_sizes
        self.num_classes = num_classes
        self.return_image_id = return_image_id
        self.encode_target = encode_target
        with open(dataset_file) as f:
            lines = f.readlines()

        for line in lines:
            if self.contain_labels == False:
                continue
            split_line = line.strip().split()
            self.fnames.append(split_line[0])
            num_boxes = (len(split_line) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x1 = float(split_line[1 + 5 * i])
                y1 = float(split_line[2 + 5 * i])
                x2 = float(split_line[3 + 5 * i])
                y2 = float(split_line[4 + 5 * i])
                c = split_line[5 + 5 * i]
                box.append([x1, y1, x2, y2])
                label.append(int(c)) # Background is not included, so labels are 0-indexed
            self.boxes.append(box)
            self.labels.append(label)
        self.num_samples = len(self.boxes)
    def _compute_iou_wh(self, w1, h1, w2, h2):
        """
        Compute IoU between two boxes given only width and height.
        Assumes boxes are centered at the same point.
        This function is used for bbox assignment to anchors.
        """
        intersection = min(w1, w2) * min(h1, h2)
        union = w1 * h1 + w2 * h2 - intersection
        iou = intersection / (union + 1e-16)
        return iou.item()
    def from_cxcy_to_gridxy(self, cx, cy, grid_size, image_size):
        """
        Convert center coordinates (cx, cy) to grid cell coordinates (grid_x, grid_y)
        and normalized coordinates (tx, ty) within the grid cell.
        """
        stride_x = image_size[0] / grid_size
        stride_y = image_size[1] / grid_size
        grid_x, tx = divmod(cx, stride_x)
        grid_y, ty = divmod(cy, stride_y)
        grid_x = int(grid_x)
        grid_y = int(grid_y)
        tx, ty = tx / stride_x, ty / stride_y #normalize tx, ty to [0, 1]
        return grid_x, grid_y, tx, ty

    def encoder(self, image, boxes:list, labels:list):
        """
        Encode ground truth boxes and labels into YOLO target format.
        
        Args:
            boxes: List of boxes in format [x1, y1, x2, y2], normalized to [0, 1]
            labels: List of class labels (1-indexed, 0 is background)
        
        Returns:
            target_boxes: List of 3 tensors, shape [grid_size, grid_size, 3, 4]
                        Format: [tx, ty, tw, th]
            target_cls: List of 3 tensors, shape [grid_size, grid_size, 3, num_classes]
                    One-hot encoded class labels
            target_obj: List of 3 tensors, shape [grid_size, grid_size, 3]
                    Objectness mask (1 if object present, 0 otherwise)
        """
        num_scales = len(self.grid_sizes)
        image_height, image_width = image.shape[1], image.shape[2]
        # Initialize target tensors
        target_boxes = [torch.zeros(gs, gs, 3, 4) for gs in self.grid_sizes]
        target_cls = [torch.zeros(gs, gs, 3, self.num_classes) for gs in self.grid_sizes]
        target_obj = [torch.zeros(gs, gs, 3) for gs in self.grid_sizes]
        anchors = torch.tensor(ANCHORS)  # Shape: [3, 3, 2]
        assert len(boxes) == len(labels), "Mismatch between boxes and labels length"
        for box, label in zip(boxes, labels):
            # Convert box to center format
            x1, y1, x2, y2 = box # haven't normalized yet!!!!!!
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = (x2 - x1) / image_width #normalize w to [0, 1]
            h = (y2 - y1) / image_height #normalize h to [0, 1]
            assert 0 < w <= 1.0 and 0 < h <= 1.0, f"Box width and height must be in (0, 1]. Got w: {w}, h: {h}"
            #find best anchor
            best_iou = -1
            best_scale_idx = None
            best_anchor_idx = None
            for scale_idx in range(num_scales):
                grid_size = self.grid_sizes[scale_idx]
                grid_x, grid_y, tx, ty = self.from_cxcy_to_gridxy(cx, cy, grid_size, (image_width, image_height))
                for anchor_idx in range(anchors.shape[1]):
                    anchor_w, anchor_h = anchors[scale_idx, anchor_idx]
                    iou = self._compute_iou_wh(w, h, anchor_w, anchor_h)
                    # Check if this anchor is better and not already occupied
                    if iou > best_iou and target_obj[scale_idx][grid_y, grid_x, anchor_idx] == 0:
                        best_iou = iou
                        best_scale_idx = scale_idx
                        best_anchor_idx = anchor_idx
        
            assert best_scale_idx is not None, "No suitable anchor found for box."
            
            # Encode target
            grid_size = self.grid_sizes[best_scale_idx]
            grid_x, grid_y, tx, ty = self.from_cxcy_to_gridxy(cx, cy, grid_size, (image_width, image_height))
            target_boxes[best_scale_idx][grid_y, grid_x, best_anchor_idx] = torch.tensor([tx, ty, w, h]) # Store absolute cx, cy, w, h for loss calculation
            target_cls[best_scale_idx][grid_y, grid_x, best_anchor_idx, int(label)] = 1.0
            target_obj[best_scale_idx][grid_y, grid_x, best_anchor_idx] = 1.0 # There is an object


        # Combine targets into a single structure
        assert len(boxes) == sum([target_obj[scale].sum() for scale in range(num_scales)]), f"Some boxes were not assigned to any scale.{len(boxes)} vs {[target_obj[scale].sum() for scale in range(num_scales)]}, best idxs: {best_idxs}"
        obj_mask = sum(obj_mask.sum() for obj_mask in target_obj)
        assert obj_mask > 0, "No objects were assigned in the target encoding."
        final_target = []
        for index in range(num_scales):
            final_target.append(torch.cat((target_boxes[index], target_obj[index].unsqueeze(-1), target_cls[index]), dim=-1))
        return final_target

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.imread(os.path.join(self.root + fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if not self.contain_labels:
            transformed = self.transform(image=img)
            transformed_image = transformed['image']
            return (transformed_image,)
        try:
            boxes = self.boxes[idx] #list
            labels = self.labels[idx] #list
            transformed = self.transform(
                image=img, bboxes=boxes, cls_labels=labels
            )
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['cls_labels']
            assert len(transformed_bboxes) == len(transformed_labels), "Mismatch between boxes and labels after transformation"
            assert len(transformed_bboxes) > 0, "No bounding boxes after transformation"
        except Exception as e:
            print(f"Error processing index {idx}, file {fname}: {e}")
            print("Using fallback: No augmentation.")
            boxes = self.boxes[idx] #list
            labels = self.labels[idx] #list
            transformed = test_data_pipelines(
                image=img, bboxes=boxes, cls_labels=labels
            )
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_labels = transformed['cls_labels']

        if self.encode_target and len(transformed_bboxes) > 0:
            target = self.encoder(
                transformed_image,  transformed_bboxes, transformed_labels
            )
            return transformed_image, target
        else:
            # Return raw boxes and labels if not encoding
            return transformed_image, transformed_bboxes, transformed_labels

    def __len__(self):
        return len(self.fnames)
    
    
    
def collate_fn(batch):
    images = []
    if len(batch[0]) == 2: #train
        targets_list = [[] for _ in range(len(GRID_SIZES))]  # for each scale
        for image, target in batch:
            images.append(image)
            for scale_idx in range(len(GRID_SIZES)):
                targets_list[scale_idx].append(target[scale_idx])
        targets = [torch.stack(tgt_scale, dim=0) for tgt_scale in targets_list]
        images = torch.stack(images, dim=0)
        return images, targets
    elif len(batch[0]) == 3:
        target_list = []
        for image, boxes, labels in batch:
            images.append(image)
            instances = []
            for box, label in zip(boxes, labels):
                target_instance = box
                target_instance.append(int(label))
                instances.append(target_instance)
            target_list.append(instances)
        images = torch.stack(images, dim=0)
        return images, target_list
    else:
        pass