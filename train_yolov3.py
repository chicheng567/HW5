"""
YOLO v3 Training Script
Compatible with the new YOLO v3 implementation
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import ANCHORS, GRID_SIZES
from src.dataset import (
    VocDetectorDataset,
    collate_fn,
    test_data_pipelines,
    train_data_pipelines,
)
from src.eval_voc import evaluate
from src.yolo import getODmodel
from yolo_loss import YOLOv3Loss

DEFAULT_LAMBDAS = {
    "lambda_coord": 5.0,
    "lambda_obj": 1.0,
    "lambda_noobj": 0.5,
    "lambda_class": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv3 detector.")
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lambda-coord", type=float, default=DEFAULT_LAMBDAS["lambda_coord"]
    )
    parser.add_argument(
        "--lambda-obj", type=float, default=DEFAULT_LAMBDAS["lambda_obj"]
    )
    parser.add_argument(
        "--lambda-noobj", type=float, default=DEFAULT_LAMBDAS["lambda_noobj"]
    )
    parser.add_argument(
        "--lambda-class", type=float, default=DEFAULT_LAMBDAS["lambda_class"]
    )
    parser.add_argument(
        "--train-annotations",
        type=str,
        default="./dataset/vocall_train.txt",
    )
    parser.add_argument(
        "--val-annotations",
        type=str,
        default="./dataset/vocall_val.txt",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="./dataset/image/",
        help="Root directory containing JPEGImages.",
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Optional path to append JSON summary per run.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Identifier stored in the training summary.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Evaluate mAP every N epochs (0 disables).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers passed to DataLoader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override training device (e.g. cuda:0).",
    )
    return parser.parse_args()


def train(args: Optional[argparse.Namespace] = None) -> dict:
    if args is None:
        args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    lambda_coord = args.lambda_coord
    lambda_obj = args.lambda_obj
    lambda_noobj = args.lambda_noobj
    lambda_class = args.lambda_class

    file_root_train = args.images_root
    annotation_file_train = args.train_annotations
    file_root_val = args.images_root
    annotation_file_val = args.val_annotations

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
        num_workers=args.num_workers,
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
        num_workers=args.num_workers,
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
        num_workers=args.num_workers,
    )
    print(f'Loaded {len(val_dataset)} val images')
    
    # Create model
    print('\nInitializing model...')
    net = getODmodel(pretrained=True).to(device)
    print(f'Model parameters: {sum(p.numel() for p in net.parameters()):,}')

    # Create loss and optimizer
    criterion = YOLOv3Loss(lambda_coord, lambda_obj, lambda_noobj, lambda_class, ANCHORS).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=5e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    # Training loop
    print('\nStarting training...')
    torch.cuda.empty_cache()
    best_val_loss = np.inf
    best_epoch = 0
    best_map = float("-inf")
    start_time = time.time()
    last_val_loss = float("nan")
    last_map = float("nan")
    for epoch in range(num_epochs):
        net.train()
        print(f'\n\nStarting epoch {epoch + 1} / {num_epochs}')
        for i, (images, target) in enumerate(train_loader):
            # Move to device
            images = images.to(device)
            target = [t.to(device) for t in target]
            # Forward pass
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                pred = net(images)
                # pred and target are lists of each scales
                loss_dict = criterion(pred, target)
            # Backward pass with mixed precision support
            scaler.scale(loss_dict['total']).backward()
            scaler.step(optimizer)
            scaler.update()
            # Print progress
            if i % 50 == 0:
                outstring = f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}], Loss: '
                outstring += ', '.join(f"{key}={val :.3f}" for key, val in loss_dict.items())
                print(outstring)
        lr_scheduler.step()
        learning_rate = lr_scheduler.get_last_lr()[0]
        print(f'Learning Rate for this epoch: {learning_rate}')
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
            last_val_loss = val_loss

        # Save best model
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            print(f'Updating best val loss: {best_val_loss:.5f}')
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(net.state_dict(), 'checkpoints/best_detector.pth')

        # Save checkpoint
        if (epoch + 1) in [5, 10, 20, 30, 40]:
            torch.save(net.state_dict(), f'checkpoints/detector_epoch_{epoch+1}.pth')

        torch.save(net.state_dict(), 'checkpoints/detector.pth')

        # Evaluate on val set
        if args.eval_every and (epoch + 1) % args.eval_every == 0:
            print('\nEvaluating on validation set...')
            val_aps = evaluate(net, eval_loader)
            epoch_map = float(np.mean(val_aps))
            print(f'Epoch {epoch}, mAP: {epoch_map:.4f}')
            best_map = max(best_map, epoch_map)
            last_map = epoch_map

    print('\nRunning final evaluation for mAP...')
    val_aps = evaluate(net, eval_loader)
    last_map = float(np.mean(val_aps))
    best_map = max(best_map, last_map)
    print(f'Final mAP: {last_map:.4f}')

    summary = {
        "run_id": args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lambda_coord": lambda_coord,
        "lambda_obj": lambda_obj,
        "lambda_noobj": lambda_noobj,
        "lambda_class": lambda_class,
        "best_val_loss": float(best_val_loss),
        "best_epoch": best_epoch,
        "final_val_loss": float(last_val_loss),
        "best_map": float(best_map),
        "final_map": float(last_map),
        "duration_sec": time.time() - start_time,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "device": str(device),
    }
    if args.results_path:
        results_dir = os.path.dirname(args.results_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        with open(args.results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
    print("TRAINING_SUMMARY " + json.dumps(summary))
    return summary


if __name__ == '__main__':
    train(parse_args())
