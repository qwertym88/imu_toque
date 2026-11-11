#!/usr/bin/env python3
"""Export the fitted joint angles, SO(3) residuals and simulated torques to CSV/PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, header, arrays):
    np.savetxt(path, np.column_stack(arrays), delimiter=",", header=",".join(header), comments="")


def export(fit_path, trace_path, output):
    output.mkdir(parents=True, exist_ok=True)
    with np.load(fit_path, allow_pickle=False) as fit:
        t = fit["time_s"]
        q = fit["joint_angles_rad"]
        names = list(fit["joint_names"])
        segments = list(fit["segment_names"])
        valid = fit["valid"]
        orientation = fit["orientation_residual_deg"]
        position = fit["segment_position_residual_m"]
        write_csv(
            output / "joint_angles.csv",
            ["time_s", "valid"] + [f"{n}_rad" for n in names],
            [t, valid, q],
        )
        write_csv(
            output / "fit_residuals.csv",
            ["time_s", "valid"]
            + [f'{n.replace(" ", "_")}_orientation_deg' for n in segments]
            + [f'{n.replace(" ", "_")}_position_m' for n in segments],
            [t, valid, orientation, position],
        )
        fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
        for column, side in enumerate(("right", "left")):
            for row, suffix in enumerate(
                ("hip_pitch", "hip_roll", "hip_yaw", "knee_pitch", "ankle_pitch", "ankle_roll")
            ):
                name = f"{side}_{suffix}"
                axes[row, column].plot(t, np.rad2deg(q[:, names.index(name)]), lw=0.7)
                for start in t[np.flatnonzero(np.diff(np.r_[False, ~valid]))[::2]]:
                    axes[row, column].axvline(start, color="red", alpha=0.5, lw=0.7)
                axes[row, column].set_ylabel(f"{suffix} (deg)")
                axes[row, column].grid(alpha=0.25)
            axes[0, column].set_title(side)
            axes[-1, column].set_xlabel("Recording time (s)")
        fig.suptitle("Xsens-to-MuJoCo kinematic fit; neutral-reference calibration is provisional")
        fig.tight_layout()
        fig.savefig(output / "joint_angles.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        for i, name in enumerate(segments[1:], start=1):
            axes[0].plot(t, orientation[:, i], lw=0.65, label=name)
            axes[1].plot(t, position[:, i], lw=0.65, label=name)
        axes[0].set_ylabel("SO(3) orientation residual (deg)")
        axes[0].legend(ncol=3, fontsize=8)
        axes[1].set_ylabel("Root-relative joint-origin residual (m)")
        axes[1].set_xlabel("Recording time (s)")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output / "fit_residuals.png", dpi=140)
        plt.close(fig)
    if trace_path is None:
        return
    with np.load(trace_path, allow_pickle=False) as trace:
        t = trace["source_time_s"]
        names = list(trace["joint_names"])
        tau = trace["torque_Nm"]
        assistance = trace["pelvis_assistance_world"]
        contact = trace["contact_wrench_world"]
        write_csv(
            output / "simulation_torques.csv",
            ["source_time_s"]
            + [f"{n}_Nm" for n in names]
            + [f"pelvis_assist_{n}" for n in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm")]
            + [
                f"{side}_contact_{n}"
                for side in ("left", "right")
                for n in ("Fx_N", "Fy_N", "Fz_N", "Mx_Nm", "My_Nm", "Mz_Nm")
            ],
            [t, tau, assistance, contact.reshape(len(t), 12)],
        )
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
        for name in names:
            if name.startswith("torso"):
                continue
            axes[0].plot(t, tau[:, names.index(name)], lw=0.7, label=name)
        axes[0].set_ylabel("Controller torque (Nm)")
        axes[0].legend(ncol=4, fontsize=7)
        for i, axis_name in enumerate(("x", "y", "z")):
            axes[1].plot(t, assistance[:, i], label=f"pelvis assist F{axis_name}")
        axes[1].set_ylabel("Explicit assistance force (N)")
        axes[1].legend(ncol=3)
        for side, i in (("left", 0), ("right", 1)):
            axes[2].plot(t, contact[:, i, 2], label=f"{side} simulated Fz")
        axes[2].set_ylabel("Simulated ground force (N)")
        axes[2].legend()
        axes[3].plot(t, np.max(np.abs(np.rad2deg(trace["joint_error_rad"])), axis=1))
        axes[3].set_ylabel("Max joint tracking error (deg)")
        axes[3].set_xlabel("Recording time (s)")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(
            "Torque-driven assisted tracking; forces are model outputs, not measured human forces"
        )
        fig.tight_layout()
        fig.savefig(output / "simulation_diagnostics.png", dpi=140)
        plt.close(fig)
    print(f"Plots and CSVs saved to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/y13")
    args = parser.parse_args()
    export(args.fit, args.trace, args.output_dir)


if __name__ == "__main__":
    main()
