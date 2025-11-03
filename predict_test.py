import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from src.config import VOC_CLASSES, YOLO_IMG_DIM
from src.dataset import test_data_pipelines
from src.yolo import resnet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv3 inference on the Kaggle test split.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("checkpoints/best_detector.pth"),
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=Path("dataset/vocall_test.txt"),
        help="File containing test image names (one per line).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("dataset/image"),
        help="Directory containing the test images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("result.csv"),
        help="Where to write the prediction CSV (matching Kaggle result format).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--nms-thres", type=float, default=0.4, help="IOU threshold for NMS.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (cuda, cpu, or auto).",
    )
    return parser.parse_args()


def build_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = resnet50(pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: Path) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    transformed = test_data_pipelines(image=image_rgb, bboxes=[], cls_labels=[])
    tensor = transformed["image"]
    return tensor, (orig_w, orig_h)


def convert_detections(
    detections: torch.Tensor,
    orig_size: Tuple[int, int],
) -> List[List]:
    orig_w, orig_h = orig_size
    boxes = []
    for detection in detections:
        x_center, y_center, width, height, obj_conf, cls_conf, cls_idx = detection.tolist()
        prob = float(obj_conf * cls_conf)

        width = max(0.0, min(width, 1.0))
        height = max(0.0, min(height, 1.0))
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)

        x1 = (x_center - width / 2.0) * orig_w
        y1 = (y_center - height / 2.0) * orig_h
        x2 = (x_center + width / 2.0) * orig_w
        y2 = (y_center + height / 2.0) * orig_h

        x1 = int(round(max(0.0, min(x1, orig_w - 1))))
        y1 = int(round(max(0.0, min(y1, orig_h - 1))))
        x2 = int(round(max(0.0, min(x2, orig_w - 1))))
        y2 = int(round(max(0.0, min(y2, orig_h - 1))))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(cls_idx)
        class_name = VOC_CLASSES[class_id]
        boxes.append([class_name, round(prob, 6), x1, y1, x2, y2])
    return boxes


def run_inference(args: argparse.Namespace) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_dir = args.images_dir
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with args.test_list.open() as f:
        image_names = [line.strip() for line in f if line.strip()]

    model = build_model(args.weights, device)

    results = []
    batch_tensors: List[torch.Tensor] = []
    batch_meta: List[Tuple[str, Tuple[int, int]]] = []

    for name in image_names:
        image_path = image_dir / name
        tensor, orig_size = preprocess_image(image_path)
        batch_tensors.append(tensor)
        batch_meta.append((name, orig_size))

        if len(batch_tensors) == args.batch_size:
            results.extend(
                process_batch(
                    model,
                    batch_tensors,
                    batch_meta,
                    device,
                    args.conf_thres,
                    args.nms_thres,
                )
            )
            batch_tensors.clear()
            batch_meta.clear()

    if batch_tensors:
        results.extend(
            process_batch(
                model,
                batch_tensors,
                batch_meta,
                device,
                args.conf_thres,
                args.nms_thres,
            )
        )

    write_predictions(args.output, results)


def process_batch(
    model: torch.nn.Module,
    batch_tensors: List[torch.Tensor],
    batch_meta: List[Tuple[str, Tuple[int, int]]],
    device: torch.device,
    conf_thres: float,
    nms_thres: float,
) -> List[Tuple[str, str]]:
    images = torch.stack(batch_tensors).to(device)
    with torch.no_grad():
        detections = model.inference(images, conf_thres=conf_thres, nms_thres=nms_thres)

    batch_results: List[Tuple[str, str]] = []
    for det, (name, orig_size) in zip(detections, batch_meta):
        if det is None:
            prediction_list = "[]"
        else:
            boxes = convert_detections(det.cpu(), orig_size)
            prediction_list = str(boxes)
        batch_results.append((name, prediction_list))
    return batch_results


def write_predictions(output_path: Path, predictions: List[Tuple[str, str]]) -> None:
    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "prediction_list"])
        for image_name, prediction_list in predictions:
            writer.writerow([image_name, prediction_list])


if __name__ == "__main__":
    run_inference(parse_args())
