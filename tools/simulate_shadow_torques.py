#!/usr/bin/env python3
"""Forward-simulate the offline shadow-QP torques without pelvis assistance."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from human_wbc_shadow import load_inputs
from simulate_xsens_walk import contact_wrenches, make_environment

ROOT = Path(__file__).resolve().parents[1]


def simulate(args):
    model, fit, _, fit_meta, _ = load_inputs(args.model, args.fit, args.contacts)
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != fit_meta["model_sha256"]:
        raise ValueError("Model differs from the fit used to calculate torques")
    with np.load(args.shadow, allow_pickle=False) as archive:
        shadow = {key: archive[key].copy() for key in archive.files}
    shadow_meta = json.loads(str(shadow["metadata_json"]))
    if shadow_meta["model_sha256"] != fit_meta["model_sha256"]:
        raise ValueError("Shadow torques were calculated for another model")
    if shadow_meta["contact_source_sha256"] != fit_meta["source_xlsx_sha256"]:
        raise ValueError("Shadow torques and fit use different Xsens samples")

    frame = shadow["frame"].astype(int)
    time_source = fit["time_s"][frame]
    torque_source = shadow["joint_torque_Nm"]
    solved = (shadow["solved"] == 1) & np.isfinite(torque_source).all(axis=1)
    if solved.sum() < 2:
        raise ValueError("At least two valid QP torque frames are required")
    end_time = args.start + args.duration
    if args.start < time_source[0] or end_time > time_source[-1]:
        raise ValueError("Requested interval is outside the saved shadow torque window")
    fit_indices = (fit["time_s"] >= args.start) & (fit["time_s"] <= end_time)
    if not fit_indices.any() or not np.all(
        fit["valid"][fit_indices] & fit["derivative_valid"][fit_indices]
    ):
        raise ValueError("Requested interval crosses invalid/reset frames")
    interval = (time_source >= args.start) & (time_source <= end_time)
    valid_times, valid_torques = time_source[solved], torque_source[solved]
    times = fit["time_s"][fit_indices]
    torques = np.column_stack(
        [np.interp(times, valid_times, valid_torques[:, j]) for j in range(valid_torques.shape[1])]
    )

    sim_model, belt_qadr, belt_dadr = make_environment(args.model, args.timestep, args.belt_speed)
    sim_model.opt.iterations = 150
    sim_model.opt.tolerance = 1e-10
    data = mujoco.MjData(sim_model)
    start_index = np.flatnonzero(fit_indices)[0]
    data.qpos[: model.nq] = fit["qpos"][start_index]
    data.qvel[: model.nv] = fit["qvel"][start_index]
    data.qvel[belt_dadr] = -args.belt_speed
    mujoco.mj_forward(sim_model, data)

    reference_qpos = fit["qpos"][fit_indices]
    rows = {
        key: []
        for key in (
            "time_s",
            "source_time_s",
            "qpos",
            "qvel",
            "joint_error_rad",
            "torque_Nm",
            "contact_wrench_world",
            "contact_count",
            "foot_position_m",
        )
    }
    failure = None
    failure_state = None
    viewer = None
    if args.viewer:
        import mujoco.viewer as viewer_module

        viewer = viewer_module.launch_passive(sim_model, data)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 115, -12
    wall_start = time.monotonic()
    steps = int(np.ceil(args.duration / args.timestep))
    sample_every = max(1, int(round(1 / (args.log_rate * args.timestep))))
    try:
        for step in range(steps):
            elapsed = step * args.timestep
            source_t = args.start + elapsed
            command = np.array(
                [np.interp(source_t, times, torques[:, j]) for j in range(sim_model.nu)]
            )
            data.qpos[belt_qadr] = -args.belt_speed * elapsed
            data.qvel[belt_dadr] = -args.belt_speed
            data.ctrl[:] = np.clip(command, -args.torque_limit, args.torque_limit)
            mujoco.mj_step(sim_model, data)
            if viewer is not None and step % 16 == 0:
                if not viewer.is_running():
                    failure = f"viewer closed at {data.time:.3f} s"
                    break
                viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
                viewer.sync()
                delay = data.time - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(min(delay, 0.05))
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel[: model.nv])) > 100:
                failure = f"nonfinite state or velocity limit at {data.time:.3f} s"
                failure_state = {
                    "max_abs_velocity": float(np.nanmax(np.abs(data.qvel[: model.nv]))),
                    "velocity_dof": int(np.nanargmax(np.abs(data.qvel[: model.nv]))),
                    "finite_qpos": bool(np.isfinite(data.qpos).all()),
                    "finite_qvel": bool(np.isfinite(data.qvel).all()),
                    "torque_command_Nm": data.ctrl.copy().tolist(),
                    "qpos": data.qpos[: model.nq].copy().tolist(),
                    "qvel": data.qvel[: model.nv].copy().tolist(),
                }
                if viewer is not None:
                    viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                break
            if data.qpos[2] < args.fall_height or data.qpos[2] > 1.8:
                failure = f"pelvis height outside [{args.fall_height}, 1.8] m at {data.time:.3f} s"
                break
            if step % sample_every == 0:
                mujoco.mj_forward(sim_model, data)
                wrench, count = contact_wrenches(sim_model, data)
                feet = [
                    data.site_xpos[sim_model.site(f"{side}_foot").id].copy()
                    for side in ("left", "right")
                ]
                ref_index = min(int(np.searchsorted(times, source_t)), len(times) - 1)
                error = data.qpos[7 : model.nq] - reference_qpos[ref_index, 7 : model.nq]
                values = {
                    "time_s": data.time,
                    "source_time_s": source_t,
                    "qpos": data.qpos[: model.nq].copy(),
                    "qvel": data.qvel[: model.nv].copy(),
                    "joint_error_rad": error,
                    "torque_Nm": data.ctrl.copy(),
                    "contact_wrench_world": wrench,
                    "contact_count": count,
                    "foot_position_m": feet,
                }
                for key, value in values.items():
                    rows[key].append(value)
    finally:
        if viewer is not None:
            viewer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    names = [sim_model.actuator(i).name.removesuffix("_torque") for i in range(sim_model.nu)]
    summary = {
        "mode": "open_loop_shadow_qp_torque_forward_simulation",
        "model": str(args.model.resolve()),
        "model_sha256": fit_meta["model_sha256"],
        "source_xsens_sha256": fit_meta["source_xlsx_sha256"],
        "start_s": args.start,
        "requested_duration_s": args.duration,
        "simulated_duration_s": float(data.time),
        "completed": failure is None,
        "failure": failure,
        "failure_state": failure_state,
        "pelvis_assistance": False,
        "belt_speed_m_s": args.belt_speed,
        "torque_interpolation": "linear over solved shadow-QP samples; endpoint values held to interval edge",
        "qp_solved_fraction_in_interval": float(np.mean(solved[interval])),
        "torque_interpolated_frame_fraction": float(1 - np.mean(solved[interval])),
        "peak_interpolated_torque_Nm": float(np.max(np.abs(torques))),
        "torque_saturation_fraction": float(np.mean(np.abs(torques) > args.torque_limit)),
        "pelvis_height_min_max_m": (
            np.percentile(arrays["qpos"][:, 2], [0, 100]).tolist() if len(rows["time_s"]) else []
        ),
        "joint_error_abs_deg_median_p95_max": (
            np.percentile(np.abs(np.rad2deg(arrays["joint_error_rad"])), [50, 95, 100]).tolist()
            if len(rows["time_s"])
            else []
        ),
        "loaded_contact_fraction_left_right": (
            np.mean(arrays["contact_wrench_world"][:, :, 2] > 20, axis=0).tolist()
            if len(rows["time_s"])
            else []
        ),
        "interpretation": "Open-loop test of shadow QP joint torques in MuJoCo with no tracking feedback or pelvis assistance; torque interpolation and uncalibrated contact/model parameters limit interpretation.",
    }
    np.savez_compressed(
        args.output,
        **arrays,
        joint_names=np.asarray(names),
        settings_json=np.asarray(json.dumps(summary)),
    )
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/y13/human.xml")
    parser.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument(
        "--contacts", type=Path, default=ROOT / "outputs/y13/contact/contact_inference.npz"
    )
    parser.add_argument("--shadow", type=Path, default=ROOT / "outputs/y13/wbc_shadow.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13/walk_shadow_torque.npz")
    parser.add_argument("--start", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--belt-speed", type=float, default=3.8 / 3.6)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--log-rate", type=float, default=100.0)
    parser.add_argument("--torque-limit", type=float, default=1500.0)
    parser.add_argument("--fall-height", type=float, default=0.55)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.belt_speed < 0 or args.timestep <= 0:
        parser.error("start/belt speed must be nonnegative; duration and timestep must be positive")
    report = simulate(args)
    if not report["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
