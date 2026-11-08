#!/usr/bin/env python3
"""Summarize recorded WBC physics; optionally plot predicted vs measured outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def rms(values):
    return np.sqrt(np.mean(np.square(values), axis=0)).tolist()


def summarize(trace, until=None):
    selected = trace["time"] <= (until if until is not None else np.inf)
    count = int(selected.sum())
    if not count:
        raise ValueError("Trace has no samples in the requested interval")
    dt = float(trace["simulation_dt"]) if "simulation_dt" in trace else 0.002
    time = trace["time"][selected]
    qpos = trace["qpos"][selected]
    tilt = np.arccos(np.clip(1.0 - 2.0 * (qpos[:, 4] ** 2 + qpos[:, 5] ** 2), -1, 1))
    report = {
        "samples": count,
        "analyzed_simulation_seconds": float(time[-1] + dt),
        "base_z_min_m": float(qpos[:, 2].min()),
        "base_tilt_max_deg": float(np.rad2deg(tilt).max()),
        "peak_torque_Nm": float(np.abs(trace["tau"][selected]).max()),
        "max_equality_residual": float(trace["eq_error"][selected].max()),
        "max_inequality_residual": float(trace["ineq_error"][selected].max()),
    }
    if "configuration" in trace:
        config = json.loads(str(trace["configuration"]))
        report["speed"] = config["speed"]
        report["analyzed_reference_seconds"] = (
            report["analyzed_simulation_seconds"] * config["speed"]
        )
    if "succeeded" in trace:
        report["completed_requested_duration"] = bool(trace["succeeded"])
        report["failure"] = str(trace["failure"])
    if "normal_force_actual" in trace:
        report["normal_force_rmse_N"] = rms(
            trace["normal_force_actual"][selected] - trace["normal_force"][selected]
        )
        report["actual_base_acceleration_rmse_m_s2"] = rms(
            trace["qacc_actual"][selected, :3] - trace["qacc"][selected, :3]
        )
        report["actuator_torque_error_max_Nm"] = float(
            np.abs(trace["tau_actual"][selected] - trace["tau"][selected]).max()
        )
        loaded = trace["wrench_actual"][selected, :, 2] > 10.0
        expected = trace["planned_support"][selected]
        for side, name in enumerate(("left", "right")):
            single = expected[:, side] & ~expected[:, 1 - side]
            physical_single = loaded[:, side] & ~loaded[:, 1 - side]
            report[f"{name}_planned_single_seconds"] = float(single.sum() * dt)
            report[f"{name}_physical_single_during_plan_seconds"] = float(
                (single & physical_single).sum() * dt
            )
    if "zmp_actual" in trace:
        predicted, measured = trace["zmp"][selected], trace["zmp_actual"][selected]
        valid = np.isfinite(predicted).all(axis=1) & np.isfinite(measured).all(axis=1)
        report["zmp_comparison_valid_samples"] = int(valid.sum())
        if np.any(valid):
            report["zmp_rmse_xy_m"] = rms(predicted[valid] - measured[valid])
    return report


def plot_trace(trace, output, until=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = trace["time"] <= (until if until is not None else np.inf)
    time = trace["time"][selected]
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True, constrained_layout=True)
    axes[0].plot(time, trace["qpos"][selected, 2], label="pelvis z")
    axes[0].set_ylabel("height (m)")
    for key, label in (("com", "COM y"), ("dcm", "DCM y"), ("dcm_target", "DCM ref y")):
        if key in trace:
            axes[1].plot(time, trace[key][selected, 1], label=label)
    axes[1].set_ylabel("position (m)")
    for key in ("zmp", "zmp_actual", "zmp_target"):
        if key in trace:
            axes[2].plot(time, trace[key][selected, 1], label=key + " y")
    axes[2].set_ylabel("ZMP (m)")
    for side, name in enumerate(("left", "right")):
        axes[3].plot(time, trace["wrench"][selected, side, 2], label=name + " Fz QP")
        if "wrench_actual" in trace:
            axes[3].plot(
                time,
                trace["wrench_actual"][selected, side, 2],
                "--",
                label=name + " Fz actual",
                alpha=0.7,
            )
        axes[4].step(
            time,
            trace["planned_support"][selected, side].astype(int) + 2 * side,
            label=name + " planned",
            where="post",
        )
        if "wrench_actual" in trace:
            loaded = (trace["wrench_actual"][selected, side, 2] > 10).astype(int)
            axes[4].step(time, loaded + 2 * side, "--", label=name + " loaded", where="post")
    axes[3].set_ylabel("force (N)")
    axes[4].set_ylabel("contact state")
    axes[4].set_xlabel("simulation time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=2, fontsize=8)
    fig.suptitle("WBC: planned and measured physics")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--until", type=float, default=None, help="analyze only this simulation-time prefix"
    )
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args()
    with np.load(args.trace, allow_pickle=False) as trace:
        print(json.dumps(summarize(trace, args.until), indent=2, allow_nan=False))
        if args.plot is not None:
            plot_trace(trace, args.plot, args.until)


if __name__ == "__main__":
    main()
