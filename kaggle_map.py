"""
Kaggle evaluation metric for VOC-style object detection (mAP @ IoU 0.5).

The solution and submission CSVs provide a ``prediction_list`` column where
each row is encoded as a Python-style list of detections::

    ['class_name', score, xmin, ymin, xmax, ymax]

The metric parses those detections, computes per-class Average Precision
following the VOC2010 protocol (integral interpolation, IoU threshold 0.5),
and returns the mean AP across the 20 Pascal VOC classes.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

CLASSES: Tuple[str, ...] = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
CLASS_TO_INDEX: Dict[str, int] = {cls: idx for idx, cls in enumerate(CLASSES)}
IOU_THRESHOLD: float = 0.5


class ParticipantVisibleError(Exception):
    """Raised for submission issues that the competitor can fix."""


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute VOC-style mAP@0.5 for the competition leaderboard.

    Parameters
    ----------
    solution:
        Ground-truth dataframe supplied by Kaggle.  Must include the row id,
        ``Usage`` flag, and ``prediction_list`` columns.
    submission:
        Competitor predictions containing the row id and ``prediction_list``.
    row_id_column_name:
        Name of the column identifying each image (typically ``"id"``).

    Returns
    -------
    float
        Mean Average Precision across the 20 VOC classes.
    """

    _validate_columns(solution, submission, row_id_column_name)

    sol = solution.set_index(row_id_column_name)
    sub = submission.set_index(row_id_column_name)

    if sol.index.has_duplicates:
        raise ParticipantVisibleError("Solution contains duplicated image ids.")
    if sub.index.has_duplicates:
        raise ParticipantVisibleError("Submission contains duplicated image ids.")

    missing = sol.index.difference(sub.index)
    if not missing.empty:
        raise ParticipantVisibleError(f"Submission is missing predictions for ids: {missing[:5].tolist()}")
    extra = sub.index.difference(sol.index)
    if not extra.empty:
        raise ParticipantVisibleError(f"Submission contains unknown ids: {extra[:5].tolist()}")

    # Align rows in solution order
    sub = sub.loc[sol.index]

    gt_boxes, gt_class_counts = _parse_ground_truth(sol["prediction_list"])
    pred_by_class = _parse_predictions(sub["prediction_list"])

    aps: List[float] = []
    for class_idx in range(len(CLASSES)):
        ap = _average_precision_for_class(
            class_idx,
            gt_boxes,
            gt_class_counts[class_idx],
            pred_by_class[class_idx],
        )
        if ap is not None:
            aps.append(ap)

    result = float(np.mean(aps) if aps else 0.0)
    if not np.isfinite(result):
        raise ParticipantVisibleError("Evaluation produced a non-finite score.")
    return float(result)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _validate_columns(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> None:
    required_solution_cols = {row_id_column_name, "Usage", "prediction_list"}
    required_submission_cols = {row_id_column_name, "prediction_list"}

    missing_sol = required_solution_cols.difference(solution.columns)
    if missing_sol:
        raise ParticipantVisibleError(f"Solution file missing columns: {sorted(missing_sol)}")

    missing_sub = required_submission_cols.difference(submission.columns)
    if missing_sub:
        raise ParticipantVisibleError(f"Submission file missing columns: {sorted(missing_sub)}")


def _parse_ground_truth(
    series: pd.Series,
) -> Tuple[Dict[Tuple[str, int], List[np.ndarray]], np.ndarray]:
    """Decode GT boxes into a lookup structure for evaluation."""
    gt_boxes: Dict[Tuple[str, int], List[np.ndarray]] = defaultdict(list)
    class_counts = np.zeros(len(CLASSES), dtype=np.int64)

    for image_id, value in series.items():
        entries = _decode_prediction_list(value, context="solution")
        for class_idx, _, box in entries:
            gt_boxes[(image_id, class_idx)].append(box)
            class_counts[class_idx] += 1

    return gt_boxes, class_counts


def _parse_predictions(series: pd.Series) -> Dict[int, List[Tuple[str, float, np.ndarray]]]:
    """Group predictions by class, preserving image ids and confidence."""
    preds: Dict[int, List[Tuple[str, float, np.ndarray]]] = defaultdict(list)
    for image_id, value in series.items():
        entries = _decode_prediction_list(value, context="submission")
        for class_idx, score, box in entries:
            if score < 0 or not np.isfinite(score):
                raise ParticipantVisibleError(f"Invalid confidence score {score} for image {image_id}.")
            preds[class_idx].append((image_id, float(score), box))
    return preds


def _decode_prediction_list(
    value: object,
    *,
    context: str,
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Convert the serialized prediction list into (class_idx, score, box) tuples.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            data = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            raise ParticipantVisibleError(f"Could not parse prediction_list for {context}.")
    else:
        data = value

    if data is None or data == "":
        return []
    if not isinstance(data, Sequence):
        raise ParticipantVisibleError("prediction_list must be a sequence of detections.")

    parsed: List[Tuple[int, float, np.ndarray]] = []
    for det in data:
        if not isinstance(det, Sequence) or len(det) != 6:
            raise ParticipantVisibleError("Each detection must be [class_name, score, xmin, ymin, xmax, ymax].")
        class_name, score, xmin, ymin, xmax, ymax = det

        if class_name not in CLASS_TO_INDEX:
            raise ParticipantVisibleError(f"Unknown class '{class_name}'.")

        try:
            score_f = float(score)
            xmin_f = float(xmin)
            ymin_f = float(ymin)
            xmax_f = float(xmax)
            ymax_f = float(ymax)
        except (TypeError, ValueError):
            raise ParticipantVisibleError("Bounding box coordinates and scores must be numeric.")

        if xmax_f < xmin_f or ymax_f < ymin_f:
            raise ParticipantVisibleError("Bounding box has negative area.")

        box = np.array([xmin_f, ymin_f, xmax_f, ymax_f], dtype=np.float32)
        parsed.append((CLASS_TO_INDEX[class_name], score_f, box))

    return parsed


# --------------------------------------------------------------------------- #
# AP computation
# --------------------------------------------------------------------------- #

def _average_precision_for_class(
    class_idx: int,
    gt_boxes: Dict[Tuple[str, int], List[np.ndarray]],
    num_gt: int,
    predictions: Sequence[Tuple[str, float, np.ndarray]],
) -> Optional[float]:
    if num_gt == 0:
        return None
    if not predictions:
        return 0.0

    sorted_preds = sorted(predictions, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(sorted_preds), dtype=np.float32)
    fp = np.zeros(len(sorted_preds), dtype=np.float32)

    gt_used: Dict[Tuple[str, int], np.ndarray] = {
        key: np.zeros(len(boxes), dtype=bool) for key, boxes in gt_boxes.items() if key[1] == class_idx
    }

    for i, (image_id, score, box) in enumerate(sorted_preds):
        key = (image_id, class_idx)
        gts = gt_boxes.get(key, [])
        if gts:
            overlaps = np.array([_bbox_iou(box, gt_box) for gt_box in gts], dtype=np.float32)
            best = overlaps.argmax()
            best_iou = overlaps[best]
            if best_iou >= IOU_THRESHOLD and not gt_used[key][best]:
                tp[i] = 1.0
                gt_used[key][best] = True
            else:
                fp[i] = 1.0
        else:
            fp[i] = 1.0

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)

    recall = tp / num_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return _voc_ap(recall, precision)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])

    iw = max(ixmax - ixmin + 1.0, 0.0)
    ih = max(iymax - iymin + 1.0, 0.0)
    inter = iw * ih

    area_a = (box_a[2] - box_a[0] + 1.0) * (box_a[3] - box_a[1] + 1.0)
    area_b = (box_b[2] - box_b[0] + 1.0) * (box_b[3] - box_b[1] + 1.0)

    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)
