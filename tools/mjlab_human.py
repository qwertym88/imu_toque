#!/usr/bin/env python3
"""Prepare, check, train and evaluate the local mjlab MS-Human tracking baseline."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human_tracking.motion import DEFAULT_DIRECTORY, ROOT, prepare_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser(
        "prepare", help="Convert the Xsens fit offline; no GPU or mjlab required"
    )
    prepare.add_argument("--model", type=Path, default=ROOT / "outputs/y13/human.xml")
    prepare.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    prepare.add_argument("--start", type=float, default=25.0)
    prepare.add_argument("--duration", type=float, default=10.0)
    prepare.add_argument("--speed", type=float, default=3.8 / 3.6)
    check = sub.add_parser("check", help="Validate reference, task and compiled human asset on CPU")
    check.add_argument("--reference-dir", type=Path, default=DEFAULT_DIRECTORY)
    smoke = sub.add_parser(
        "smoke", help="Run untrained GPU environments and assert interface invariants"
    )
    smoke.add_argument("--reference-dir", type=Path, default=DEFAULT_DIRECTORY)
    smoke.add_argument("--num-envs", type=int, default=16)
    smoke.add_argument("--steps", type=int, default=100)
    smoke.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY / "smoke.json")
    train = sub.add_parser("train", help="Train PPO locally; TensorBoard, no WandB upload")
    train.add_argument("--reference-dir", type=Path, default=DEFAULT_DIRECTORY)
    train.add_argument("--num-envs", type=int, default=1024)
    train.add_argument("--iterations", type=int, default=3000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-dir", type=Path)
    train.add_argument(
        "--checkpoint", type=Path, help="Resume this baseline's checkpoint and optimizer"
    )
    evaluate = sub.add_parser(
        "evaluate", help="Single uninterrupted rollout, zero assistance, no automatic resets"
    )
    evaluate.add_argument(
        "--checkpoint", type=Path, help="Omit to evaluate the reference-PD baseline"
    )
    evaluate.add_argument(
        "--reference-dir",
        type=Path,
        help="Defaults to checkpoint's frozen reference, or prepared reference",
    )
    evaluate.add_argument("--duration", type=float, default=10.0)
    evaluate.add_argument("--viewer", action="store_true")
    evaluate.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY / "evaluation.npz")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_reference(
            args.model, args.fit, args.output_dir, args.start, args.duration, speed=args.speed
        )
        return
    from human_tracking.runner import check, smoke, train, evaluate

    handlers = {"check": check, "smoke": smoke, "train": train, "evaluate": evaluate}
    result = handlers[args.command](args)
    if result and result.get("completed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
