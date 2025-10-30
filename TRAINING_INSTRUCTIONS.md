# YOLO v3 Training Instructions

## 簡介
此文檔說明如何使用新的 YOLO v3 模型進行訓練。

## 重要變更

### 1. Dataset 輸出格式
Dataset 返回：
- `image`: numpy array (H, W, 3)
- `boxes`: tensor (N, 4) - [x1, y1, x2, y2] 像素座標
- `labels`: tensor (N,) - 類別 ID (1-20)

### 2. Loss 函數介面
新的 loss 函數**不再需要** `has_object_map` 參數！

**舊版 (YOLO v1)**:
```python
loss_dict = criterion(pred, target_boxes, target_cls, has_object_map)
```

**新版 (YOLO v3)**:
```python
loss_dict = criterion(pred, target_boxes, target_cls)
```

`has_object_map` 會在 loss 內部自動從 `target_cls` 生成。

## 訓練方式

### 方式 1: 使用提供的訓練腳本

```bash
cd /workspace/dlhw/hw5
/miniconda/envs/vl3/bin/python train_yolov3.py
```

這個腳本已經配置好所有必要的參數和 collate function。

### 方式 2: 在 Jupyter Notebook 中訓練

在 `A5.ipynb` 中，需要進行以下修改：

#### Step 1: 導入 helper function

在 imports 部分添加：
```python
from train_helper import collate_fn
```

#### Step 2: 修改 DataLoader

將原始的 DataLoader 改為：
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn  # 添加這一行
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn  # 添加這一行
)
```

#### Step 3: 修改 Training Loop

**Training部分** - 改為：
```python
for i, (images, target_boxes, target_cls) in enumerate(train_loader):  # 移除 has_object_map
    images = images.to(device)
    target_boxes = target_boxes.to(device)
    target_cls = target_cls.to(device)

    pred = net(images)
    loss_dict = criterion(pred, target_boxes, target_cls)  # 只傳 3 個參數

    for key in loss_dict:
        total_loss[key] += loss_dict[key].item()

    optimizer.zero_grad()
    loss_dict['total_loss'].backward()
    optimizer.step()

    # ... 其他代碼
```

**Validation部分** - 改為：
```python
net.eval()  # Set to eval mode
for i, (images, target_boxes, target_cls) in enumerate(val_loader):  # 移除 has_object_map
    images = images.to(device)
    target_boxes = target_boxes.to(device)
    target_cls = target_cls.to(device)

    # 重要：使用 return_raw=True 在 eval mode 下獲取 raw predictions
    pred = net(images, return_raw=True)
    loss_dict = criterion(pred, target_boxes, target_cls)  # 只傳 3 個參數
    val_loss += loss_dict['total_loss'].item()
```

## 完整的 Training Loop 示例

```python
import torch
import collections
from torch.utils.data import DataLoader
from src.resnet_yolo import resnet50
from yolo_loss import YoloLoss
from src.dataset import VocDetectorDataset, train_data_pipelines, test_data_pipelines
from src.eval_voc import evaluate
from train_helper import collate_fn

# Hyperparameters
device = torch.device("cuda:0")
S = 14
B = 2
num_epochs = 50
batch_size = 50
learning_rate = 1e-3
lambda_coord = 5
lambda_noobj = 0.5

# Data
file_root_train = './VOC/JPEGImages/'
annotation_file_train = 'vocall_train.txt'
file_root_val = './VOC/JPEGImages/'
annotation_file_val = 'vocall_val.txt'

# Create datasets with collate_fn
train_dataset = VocDetectorDataset(
    root_img_dir=file_root_train,
    dataset_file=annotation_file_train,
    train=True,
    transform=train_data_pipelines,
    S=S
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn  # 重要！
)

val_dataset = VocDetectorDataset(
    root_img_dir=file_root_val,
    dataset_file=annotation_file_val,
    train=False,
    transform=test_data_pipelines,
    S=S
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn  # 重要！
)

# Model, loss, optimizer
net = resnet50(pretrained=True).to(device)
criterion = YoloLoss(S, B, lambda_coord, lambda_noobj)
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=5e-4)

# Training loop
torch.cuda.empty_cache()
best_val_loss = float('inf')

for epoch in range(num_epochs):
    net.train()

    # Update learning rate
    if epoch == 30 or epoch == 40:
        learning_rate /= 10.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

    print(f'\n\nStarting epoch {epoch + 1} / {num_epochs}')
    print(f'Learning Rate: {learning_rate}')

    total_loss = collections.defaultdict(float)

    # Training
    for i, (images, target_boxes, target_cls) in enumerate(train_loader):
        images = images.to(device)
        target_boxes = target_boxes.to(device)
        target_cls = target_cls.to(device)

        pred = net(images)
        loss_dict = criterion(pred, target_boxes, target_cls)  # 注意：只 3 個參數

        for key in loss_dict:
            total_loss[key] += loss_dict[key].item()

        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            outstring = f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}], Loss: '
            outstring += ', '.join(f"{key[:-5]}={val / (i+1):.3f}" for key, val in total_loss.items())
            print(outstring)

    # Validation
    with torch.no_grad():
        val_loss = 0.0
        net.eval()
        for i, (images, target_boxes, target_cls) in enumerate(val_loader):
            images = images.to(device)
            target_boxes = target_boxes.to(device)
            target_cls = target_cls.to(device)

            # 重要：使用 return_raw=True 獲取 raw predictions
            pred = net(images, return_raw=True)
            loss_dict = criterion(pred, target_boxes, target_cls)  # 注意：只 3 個參數
            val_loss += loss_dict['total_loss'].item()

        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}')

    # Save best model
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        print(f'Updating best val loss: {best_val_loss:.5f}')
        torch.save(net.state_dict(), 'checkpoints/best_detector.pth')

    # Save checkpoints
    if (epoch + 1) in [5, 10, 20, 30, 40]:
        torch.save(net.state_dict(), f'checkpoints/detector_epoch_{epoch+1}.pth')

    torch.save(net.state_dict(), 'checkpoints/detector.pth')

    # Evaluate
    if (epoch + 1) % 5 == 0:
        val_aps = evaluate(net, val_dataset_file=annotation_file_val, img_root=file_root_val)
        print(f'Epoch {epoch}, mAP: {sum(val_aps)/len(val_aps):.4f}')
```

## 關鍵點總結

1. ✅ **必須使用 `collate_fn`**: 在 DataLoader 中指定 `collate_fn=collate_fn`
2. ✅ **Loss 只需要 3 個參數**: `criterion(pred, target_boxes, target_cls)`
3. ✅ **DataLoader 返回 3 個值**: `images, target_boxes, target_cls`
4. ✅ **圖像大小**: collate_fn 會自動確保所有圖像是 448x448
5. ✅ **格式轉換**: collate_fn 會自動將 boxes/labels 轉換為 YOLO grid 格式

## 常見問題

### Q: 為什麼不需要 has_object_map？
A: 因為 `has_object_map` 可以直接從 `target_cls` 計算得出（任何有 class 的 cell 就是有 object 的 cell）。這樣更簡潔且不容易出錯。

### Q: 為什麼 validation 需要 return_raw=True？
A: 因為在 `eval()` mode 下，模型默認會應用 NMS 並返回檢測結果（list），而不是 raw predictions。但計算 loss 需要 raw predictions，所以需要使用 `return_raw=True` 參數來獲取它們。

### Q: collate_fn 做了什麼？
A:
1. 將圖像從 numpy array 轉換為 tensor
2. 確保所有圖像都是 448x448
3. 將 boxes (N, 4) 和 labels (N,) 轉換為 YOLO grid 格式 (14, 14, 4) 和 (14, 14, 20)

### Q: 可以直接用原來的 notebook 嗎？
A: 需要做最小修改：
1. 添加 `from train_helper import collate_fn`
2. 在 DataLoader 添加 `collate_fn=collate_fn`
3. 移除 training/validation loop 中的 `has_object_map` 相關代碼

### Q: 模型輸出格式變了嗎？
A: 是的！
- **Training mode**: 返回 tuple of 3 tensors (3 個尺度的預測)
- **Inference mode (return_raw=False)**: 返回 list (每張圖的 NMS 後檢測結果)
- **Inference mode (return_raw=True)**: 返回 tuple of 3 tensors (用於計算 loss)

在 validation 時計算 loss，需要使用 `model(x, return_raw=True)` 來獲取 raw predictions。
