#!/usr/bin/env python3
"""Export a non-resetting mjlab rollout as CSV and diagnostic plots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def export(trace):
    with np.load(trace, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    settings = json.loads(str(data["settings_json"]))
    time = data["time_s"]
    names = [
        f"{side}_{joint}"
        for side in ("right", "left")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
    ] + ["torso_pitch", "torso_roll", "torso_yaw"]
    columns = ["time_s"]
    values = [time[:, None]]
    for label, array in (
        ("angle_rad", data["qpos"][:, 7:]),
        ("reference_rad", data["reference_qpos"][:, 7:]),
        ("torque_Nm", data["torque_Nm"]),
    ):
        columns.extend(f"{name}_{label}" for name in names)
        values.append(array)
    columns.extend(
        f"{side}_{component}"
        for side in ("left", "right")
        for component in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm")
    )
    values.append(data["contact_wrench_world"].reshape(len(time), 12))
    np.savetxt(
        trace.with_suffix(".csv"),
        np.column_stack(values),
        delimiter=",",
        header=",".join(columns),
        comments="",
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    error = np.abs(np.rad2deg(data["joint_error_rad"][:, :12]))
    axes[0].plot(time, np.median(error, axis=1), label="joint error median")
    axes[0].plot(time, np.percentile(error, 95, axis=1), label="joint error p95")
    axes[0].plot(
        time, np.median(data["body_orientation_error_deg"], axis=1), label="7 segment error median"
    )
    axes[0].set_ylabel("Error (deg)")
    for i, side in enumerate(("right", "left")):
        axes[1].plot(
            time,
            np.max(np.abs(data["torque_Nm"][:, i * 6 : i * 6 + 6]), axis=1),
            label=f"{side} max |torque|",
        )
    axes[1].set_ylabel("Actuator torque (Nm)")
    for i, side in enumerate(("left", "right")):
        axes[2].plot(
            time, data["contact_wrench_world"][:, i, 2], label=f"{side} Fz (CPU recomputation)"
        )
    axes[2].set_ylabel("Foot force (N)")
    axes[2].set_xlabel("Uninterrupted rollout time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    fig.suptitle(settings["mode"] + " | zero pelvis assistance | simulated forces")
    fig.tight_layout()
    fig.savefig(trace.with_suffix(".png"), dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for index, (axis, joint) in enumerate(
        zip(
            axes.flat,
            ("Hip pitch", "Hip roll", "Hip yaw", "Knee pitch", "Ankle pitch", "Ankle roll"),
        )
    ):
        for offset, side, color in ((0, "right", "C0"), (6, "left", "C1")):
            axis.plot(
                time, np.rad2deg(data["qpos"][:, 7 + offset + index]), color=color, label=side
            )
            axis.plot(
                time,
                np.rad2deg(data["reference_qpos"][:, 7 + offset + index]),
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{side} reference",
            )
        axis.set_title(joint)
        axis.set_ylabel("deg")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(trace.with_name(trace.stem + "_angles.png"), dpi=140)
    plt.close(fig)
    print(f"Exported {trace.with_suffix('.csv')} and diagnostic PNGs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    export(parser.parse_args().trace)


if __name__ == "__main__":
    main()
