#!/usr/bin/env python3
"""Reconstruct net joint torques from a physical trace and its simulated foot wrenches.

These are model/controller quantities. No old force-plate record is read or used
as ground truth. Six base residuals retain the explicit pelvis assistance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reconstruct(model, data, qacc, wrenches, other_constraints=None):
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    result = mass @ qacc + data.qfrc_bias - data.qfrc_passive
    for i, side in enumerate(("left", "right")):
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, model.site(f"{side}_foot").id)
        result -= np.vstack((jp, jr)).T @ wrenches[i]
    if other_constraints is not None:
        result -= other_constraints
    return result


def compute(input_path, output_path):
    with np.load(input_path, allow_pickle=False) as trace:
        if "contact_wrench_world" not in trace.files or "settings_json" not in trace.files:
            raise ValueError(
                "Expected a physical simulation trace with wrenches, not the kinematic fit"
            )
        settings = json.loads(str(trace["settings_json"]))
        model = mujoco.MjModel.from_xml_path(settings["model"])
        data = mujoco.MjData(model)
        # NPZ members are decompressed on access; materialize each trace once.
        states = {
            name: trace[name]
            for name in (
                "time_s",
                "qpos",
                "qvel",
                "qacc",
                "contact_wrench_world",
                "other_constraint_force",
                "torque_Nm",
                "pelvis_assistance_generalized",
            )
        }
        required = np.empty((len(states["time_s"]), model.nv))
        dofs = np.array([model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]])
        for i in range(len(required)):
            data.qpos[:] = states["qpos"][i]
            data.qvel[:] = states["qvel"][i]
            mujoco.mj_forward(model, data)
            required[i] = reconstruct(
                model,
                data,
                states["qacc"][i],
                states["contact_wrench_world"][i],
                states["other_constraint_force"][i],
            )
        difference = required[:, dofs] - states["torque_Nm"]
        base_difference = required[:, :6] - states["pelvis_assistance_generalized"]
        report = {
            "input": str(input_path.resolve()),
            "model": settings["model"],
            "frames": len(required),
            "meaning": "simulation net torques; not measured subject torques",
            "max_torque_reconstruction_error_Nm": float(np.max(np.abs(difference))),
            "max_base_assistance_reconstruction_error": float(np.max(np.abs(base_difference))),
            "torque_abs_Nm_median_p95_max": np.percentile(
                np.abs(required[:, dofs]), [50, 95, 100]
            ).tolist(),
            "base_residual_is_reported_not_hidden": True,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            time_s=trace["source_time_s"],
            joint_names=trace["joint_names"],
            torque_Nm=required[:, dofs],
            base_residual_generalized=required[:, :6],
            controller_torque_difference_Nm=difference,
            foot_wrenches_world=trace["contact_wrench_world"],
        )
        output_path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/y13/walk_assisted.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13/joint_torques.npz")
    args = parser.parse_args()
    compute(args.input, args.output)


if __name__ == "__main__":
    main()
