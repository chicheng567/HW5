"""
Hyperparameter evolution driver for YOLOv3 loss lambdas.

This script spawns short training runs of `train_yolov3.py`, mutates the
loss-weight hyperparameters, and records the resulting validation mAP in a
JSONL history file. Use it to explore better lambda combinations without
hand-tuning.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_LAMBDAS = {
    "lambda_coord": 5.0,
    "lambda_obj": 1.0,
    "lambda_noobj": 0.5,
    "lambda_class": 1.0,
}

DEFAULT_BOUNDS = {
    "lambda_coord": (0.25, 15.0),
    "lambda_obj": (0.05, 5.0),
    "lambda_noobj": (0.05, 5.0),
    "lambda_class": (0.1, 5.0),
}

SUMMARY_PREFIX = "TRAINING_SUMMARY "


@dataclass
class Candidate:
    lambdas: Dict[str, float]
    parent_run_id: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolve YOLOv3 loss lambdas via repeated training runs."
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="train_yolov3.py",
        help="Path to the training script to invoke.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="checkpoints/lambda_evolve.jsonl",
        help="History file storing JSONL training summaries.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Number of generations to evolve.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=6,
        help="Number of candidates to evaluate per generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Parent pool size sampled from best historical runs.",
    )
    parser.add_argument(
        "--child-epochs",
        type=int,
        default=3,
        help="Epochs per candidate run (keep small for faster search).",
    )
    parser.add_argument(
        "--mutation-sigma",
        type=float,
        default=0.35,
        help="Standard deviation of multiplicative log-normal mutation.",
    )
    parser.add_argument(
        "--mutation-prob",
        type=float,
        default=0.9,
        help="Probability that each lambda receives a perturbation.",
    )
    parser.add_argument(
        "--reset-prob",
        type=float,
        default=0.05,
        help="Chance to resample a lambda uniformly within bounds.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Pass-through for train.py --eval-every (0 to disable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=80,
        help="Batch size passed to the training script.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate passed to the training script.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device forwarded to train_yolov3.py (--device).",
    )
    parser.add_argument(
        "--discard-threshold",
        type=float,
        default=None,
        help="Discard runs whose best_map falls below this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Evaluate the default lambda configuration before mutating.",
    )
    return parser.parse_args()


def load_history(path: Path) -> List[dict]:
    if not path.exists():
        return []
    history = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] Skipping malformed line in {path}: {line[:120]}")
    return history


def run_map(entry: dict) -> float:
    return entry.get("best_map", float("-inf"))


def select_parent(history: List[dict], top_k: int) -> dict:
    sorted_runs = sorted(history, key=run_map, reverse=True)
    pool = sorted_runs[: max(1, min(top_k, len(sorted_runs)))]
    return random.choice(pool)


def mutate_lambdas(
    base: Dict[str, float],
    bounds: Dict[str, tuple],
    sigma: float,
    mutation_prob: float,
    reset_prob: float,
) -> Dict[str, float]:
    mutated = {}
    for key, value in base.items():
        low, high = bounds.get(key, (value * 0.25, value * 4))
        candidate = value
        if random.random() < reset_prob:
            candidate = random.uniform(low, high)
        elif random.random() < mutation_prob:
            factor = math.exp(random.gauss(0.0, sigma))
            candidate = value * factor
        candidate = max(low, min(high, candidate))
        mutated[key] = candidate
    return mutated


def run_training(
    train_script: str,
    candidate: Candidate,
    args: argparse.Namespace,
    generation: int,
    index: int,
    results_file: Path,
) -> Optional[dict]:
    run_id = f"gen{generation:03d}_cand{index:02d}_{int(time.time())}"
    cmd = [
        "python",
        train_script,
        "--num-epochs",
        str(args.child_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--lambda-coord",
        f"{candidate.lambdas['lambda_coord']}",
        "--lambda-obj",
        f"{candidate.lambdas['lambda_obj']}",
        "--lambda-noobj",
        f"{candidate.lambdas['lambda_noobj']}",
        "--lambda-class",
        f"{candidate.lambdas['lambda_class']}",
        "--results-path",
        str(results_file),
        "--run-id",
        run_id,
        "--eval-every",
        str(args.eval_every),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print(f"\n[gen {generation}] launching candidate {index} with run_id={run_id}")
    print(f"  lambdas={candidate.lambdas}")
    summary_json = None
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    assert process.stdout is not None  # for type checkers
    for line in process.stdout:
        print(line, end="")
        if line.startswith(SUMMARY_PREFIX):
            summary_json = line[len(SUMMARY_PREFIX) :].strip()
    process.wait()
    if process.returncode != 0:
        print(f"[warn] training run {run_id} exited with {process.returncode}")
        return None
    if not summary_json:
        print(f"[warn] training run {run_id} did not emit a summary; skipping.")
        return None
    try:
        summary = json.loads(summary_json)
    except json.JSONDecodeError as exc:
        print(f"[warn] failed to decode summary for {run_id}: {exc}")
        return None
    summary["parent_run_id"] = candidate.parent_run_id
    summary["generation"] = generation
    summary["candidate_index"] = index
    return summary


def ensure_baseline(
    args: argparse.Namespace, results_file: Path, history: List[dict]
) -> List[dict]:
    if not args.include_baseline or history:
        return history
    baseline_candidate = Candidate(lambdas=dict(DEFAULT_LAMBDAS))
    summary = run_training(
        args.train_script,
        baseline_candidate,
        args,
        generation=0,
        index=0,
        results_file=results_file,
    )
    if summary:
        history.append(summary)
    return load_history(results_file)


def evolve():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    results_file = Path(args.results_file)
    if results_file.parent:
        os.makedirs(results_file.parent, exist_ok=True)

    history = load_history(results_file)
    history = ensure_baseline(args, results_file, history)

    best_score = float("-inf")
    best_config = None
    if history:
        best_entry = max(history, key=run_map)
        best_score = run_map(best_entry)
        best_config = {
            k: best_entry.get(k, DEFAULT_LAMBDAS[k]) for k in DEFAULT_LAMBDAS
        }
        print(f"[info] Loaded {len(history)} past runs. Best mAP={best_score:.4f}")

    for generation in range(1, args.generations + 1):
        print(f"\n=== Generation {generation} ===")
        for index in range(args.population_size):
            if history:
                parent_entry = select_parent(history, args.top_k)
                parent_id = parent_entry.get("run_id")
                base_lambdas = {
                    key: parent_entry.get(key, DEFAULT_LAMBDAS[key])
                    for key in DEFAULT_LAMBDAS
                }
            else:
                parent_entry = None
                parent_id = None
                base_lambdas = dict(DEFAULT_LAMBDAS)
            lambdas = mutate_lambdas(
                base_lambdas,
                DEFAULT_BOUNDS,
                args.mutation_sigma,
                args.mutation_prob,
                args.reset_prob,
            )
            candidate = Candidate(lambdas=lambdas, parent_run_id=parent_id)
            summary = run_training(
                args.train_script,
                candidate,
                args,
                generation,
                index,
                results_file,
            )
            if not summary:
                continue
            history.append(summary)
            if (
                args.discard_threshold is not None
                and run_map(summary) < args.discard_threshold
            ):
                print(
                    f"[info] run {summary['run_id']} discarded "
                    f"(mAP {run_map(summary):.4f} < threshold)"
                )
                continue
            if run_map(summary) > best_score:
                best_score = run_map(summary)
                best_config = {
                    key: summary.get(key, DEFAULT_LAMBDAS[key]) for key in DEFAULT_LAMBDAS
                }
                print(
                    f"[best] New best mAP {best_score:.4f} "
                    f"from run {summary['run_id']} (generation {generation})"
                )

        history = load_history(results_file)
        if history:
            top = sorted(history, key=run_map, reverse=True)[:5]
            print("\n[top] Current leaderboard:")
            for entry in top:
                lambdas_repr = ", ".join(
                    f"{key}={entry.get(key, float('nan')):.3f}"
                    for key in DEFAULT_LAMBDAS
                )
                print(
                    f"  mAP={run_map(entry):.4f} | "
                    f"loss={entry.get('best_val_loss', math.inf):.4f} | "
                    f"run={entry.get('run_id')} | {lambdas_repr}"
                )

    if best_config is not None:
        print("\n=== Evolution complete ===")
        print(f"Best validation mAP: {best_score:.4f}")
        print("Best lambdas (paste into train_yolov3.py defaults if desired):")
        for key, value in best_config.items():
            print(f"  {key} = {value:.6f}")
    else:
        print("[warn] No successful runs were recorded.")


if __name__ == "__main__":
    evolve()
