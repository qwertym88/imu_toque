#!/usr/bin/env python3
"""Infer candidate Y13 foot-contact states from Xsens kinematics and motion history."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from contact_detection import detect_contact


def plot_contact(path, output, start_s=0.0, duration_s=None):
    import json

    with np.load(path, allow_pickle=False) as data:
        time = data["time_s"]
        high_features = data["features"]
        high_time = data["high_rate_time_s"]
        probability = data["contact_probability"]
        state = data["contact_state_candidate"]
        uncertain = data["uncertain"]
        high_prob = data["high_rate_probability"]
        metadata = json.loads(str(data["metadata_json"]))
    end_s = float(time[-1]) if duration_s is None else min(start_s + duration_s, float(time[-1]))
    selected = (time >= start_s) & (time <= end_s)
    high_selected = (high_time >= start_s) & (high_time <= end_s)
    if not selected.any():
        raise ValueError("Requested plot interval is outside the recording")
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    axes[0].plot(
        high_time[high_selected], high_features[high_selected, 0, 0], label="L foot acceleration"
    )
    axes[0].plot(
        high_time[high_selected], high_features[high_selected, 1, 0], label="R foot acceleration"
    )
    axes[0].set_ylabel("log(1 + m/s^2)")
    axes[1].plot(
        high_time[high_selected],
        high_features[high_selected, 0, 1],
        label="L foot angular velocity",
    )
    axes[1].plot(
        high_time[high_selected],
        high_features[high_selected, 1, 1],
        label="R foot angular velocity",
    )
    axes[1].set_ylabel("log(1 + rad/s)")
    for axis in axes[:2]:
        axis.legend(loc="upper right", ncol=2)
    for side, color in enumerate(("tab:blue", "tab:orange")):
        axes[2].plot(
            high_time[high_selected],
            high_prob[high_selected, side],
            color=color,
            alpha=0.65,
            label=f"{'L' if side == 0 else 'R'} HMM posterior (uncalibrated)",
        )
        axes[2].step(
            time[selected],
            state[selected, side].astype(float),
            where="post",
            color=color,
            linestyle="--",
            alpha=0.8,
        )
        axes[2].fill_between(
            time[selected], 0, 1, where=uncertain[selected, side], color=color, alpha=0.12
        )
    axes[2].axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Posterior / state")
    axes[2].legend(loc="upper right", ncol=2)
    axes[3].step(
        time[selected],
        state[selected, 0].astype(int) + 2 * state[selected, 1].astype(int),
        where="post",
    )
    axes[3].set_yticks(
        [0, 1, 2, 3], ["swing/swing", "L contact-like", "R contact-like", "double contact-like"]
    )
    axes[3].set_ylabel("Candidate state")
    axes[3].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(
        f"Y13 unlabeled temporal contact baseline | {start_s:.2f}-{end_s:.2f} s | no force plate"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/y13_MVN_3_8_01.xlsx")
    parser.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/y13/contact")
    parser.add_argument(
        "--expected-state-duration",
        type=float,
        default=0.45,
        help="Persistence prior only; not a measured gait period",
    )
    parser.add_argument("--uncertain-low", type=float, default=0.35)
    parser.add_argument("--uncertain-high", type=float, default=0.65)
    parser.add_argument("--plot-start", type=float, default=25.0)
    parser.add_argument("--plot-duration", type=float, default=10.0)
    args = parser.parse_args()
    if not 0 < args.uncertain_low < args.uncertain_high < 1:
        parser.error("uncertainty thresholds must be ordered in (0, 1)")
    report = detect_contact(
        args.input,
        args.fit,
        args.output_dir,
        expected_state_duration_s=args.expected_state_duration,
        uncertainty=(args.uncertain_low, args.uncertain_high),
    )
    plot_contact(
        args.output_dir / "contact_inference.npz",
        args.output_dir / "contact_preview.png",
        args.plot_start,
        args.plot_duration,
    )
    print(f"Candidate contact inference saved in {args.output_dir}")
    print(
        f"Valid fraction: {report['valid_fraction']:.4f}; force plate used: {report['force_plate_used']}"
    )


if __name__ == "__main__":
    main()
