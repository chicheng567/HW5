"""
YOLO v3 Training Script
Compatible with the new YOLO v3 implementation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.yolo import resnet50
from yolo_loss import YOLOv3Loss
from src.dataset import VocDetectorDataset, train_data_pipelines, test_data_pipelines, collate_fn
from src.eval_voc import evaluate
from src.config import GRID_SIZES

def train():
    # Hyperparameters
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 35
    learning_rate = 1e-3
    # Loss coefficients
    lambda_coord=5.0
    lambda_obj=1.0
    lambda_noobj=0.5
    lambda_class=1.0
    # Data paths
    file_root_train = './VOC/JPEGImages/'
    annotation_file_train = 'vocall_train.txt'
    file_root_val = './VOC/JPEGImages/'
    annotation_file_val = 'vocall_val.txt'

    # Create datasets
    print('Loading datasets...')
    train_dataset = VocDetectorDataset(
        root_img_dir=file_root_train,
        dataset_file=annotation_file_train,
        train=True,
        transform=train_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=4,
    )
    print(f'Loaded {len(train_dataset)} train images')
    
    val_dataset = VocDetectorDataset(
        root_img_dir=file_root_val,
        dataset_file=annotation_file_val,
        train=False,
        transform=test_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4,
    )
    #for computing val maps
    eval_dataset = VocDetectorDataset(
        root_img_dir=file_root_val,
        dataset_file=annotation_file_val,
        train=False,
        transform=test_data_pipelines,
        grid_sizes=GRID_SIZES,
        encode_target=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=4
    )
    print(f'Loaded {len(val_dataset)} val images')
    
    # Create model
    print('\nInitializing model...')
    net = resnet50(pretrained=True).to(device)
    print(f'Model parameters: {sum(p.numel() for p in net.parameters()):,}')

    # Create loss and optimizer
    criterion = YOLOv3Loss(lambda_coord, lambda_obj, lambda_noobj, lambda_class).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    # Training loop
    print('\nStarting training...')
    torch.cuda.empty_cache()
    best_val_loss = np.inf
    for epoch in range(num_epochs):
        net.train()
        # Update learning rate late in training
        if epoch == 30 or epoch == 40:
            learning_rate /= 10.0

        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print(f'\n\nStarting epoch {epoch + 1} / {num_epochs}')
        print(f'Learning Rate for this epoch: {learning_rate}')

        for i, (images, target) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            target = [t.to(device) for t in target]
            # Forward pass
            pred = net(images)
            # pred and target are lists of each scales
            loss_dict = criterion(pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            optimizer.step()

            # Print progress
            if (i + 1) % 50 == 0:
                outstring = f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}], Loss: '
                outstring += ', '.join(f"{key}={val / (i+1):.3f}" for key, val in loss_dict.items())
                print(outstring)

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            net.eval()
            for i, (images, target) in enumerate(val_loader):
                # Move to device
                images = images.to(device)
                target = [t.to(device) for t in target]
                # Forward pass
                pred = net(images)
                loss_dict = criterion(pred, target)
                val_loss += loss_dict['total'].item()

            val_loss /= len(val_loader)
            print(f'Validation Loss: {val_loss:.4f}')

        # Save best model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            print(f'Updating best val loss: {best_val_loss:.5f}')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(net.state_dict(), 'checkpoints/best_detector.pth')

        # Save checkpoint
        if (epoch + 1) in [5, 10, 20, 30, 40]:
            torch.save(net.state_dict(), f'checkpoints/detector_epoch_{epoch+1}.pth')

        torch.save(net.state_dict(), 'checkpoints/detector.pth')

        # Evaluate on val set
        if (epoch + 1) % 5 == 0:
            print('\nEvaluating on validation set...')
            val_aps = evaluate(net, val_dataset_file=annotation_file_val, img_root=file_root_val)
            print(f'Epoch {epoch}, mAP: {np.mean(val_aps):.4f}')


if __name__ == '__main__':
    train()
