#!/usr/bin/env python3
"""Replay fitted Y13 motion with contact candidates and offline QP force overlays."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
GREEN = np.array([0.15, 0.85, 0.3, 1.0])
GRAY = np.array([0.55, 0.55, 0.55, 1.0])
ORANGE = np.array([1.0, 0.5, 0.05, 1.0])


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    return arrays, json.loads(str(arrays["metadata_json"]))


def load_replay(args):
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    fit, fit_meta = read_npz(args.fit)
    contacts, contact_meta = read_npz(args.contacts)
    shadow, shadow_meta = read_npz(args.shadow)
    model_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    if fit_meta["model_sha256"] != model_hash or shadow_meta["model_sha256"] != model_hash:
        raise ValueError("Replay model does not match the fit/shadow model SHA-256")
    if contact_meta["matched_fit_model_sha256"] != model_hash:
        raise ValueError("Contact candidates refer to another model")
    source_hash = fit_meta["source_xlsx_sha256"]
    if (
        contact_meta["matched_xsens_sha256"] != source_hash
        or shadow_meta["contact_source_sha256"] != source_hash
    ):
        raise ValueError("Replay inputs use different Xsens samples")
    for meta in (contact_meta, shadow_meta):
        if meta.get("sample_id") != "y13" or meta.get("force_plate_used") is not False:
            raise ValueError("Replay requires Y13 results without legacy force-plate inputs")
    if not np.array_equal(contacts["time_s"], fit["time_s"]):
        raise ValueError("Contact and fit timestamps differ")
    frames = shadow["frame"].astype(np.int64)
    if np.any(frames < 0) or np.any(frames >= len(fit["qpos"])) or not np.all(np.diff(frames) > 0):
        raise ValueError("Shadow frame indices must be ordered and within the fit")
    if not np.allclose(shadow["time_s"], fit["time_s"][frames], atol=1e-8, rtol=0):
        raise ValueError("Shadow and fit timestamps differ")
    if not np.array_equal(
        shadow["support_mask_candidate"], contacts["support_mask_candidate"][frames]
    ):
        raise ValueError("Shadow support masks differ from the contact candidates; regenerate QP")
    if not (
        fit["valid"][frames] & fit["derivative_valid"][frames] & contacts["valid"][frames]
    ).all():
        raise ValueError("Shadow contains invalid/reset frames")
    selected = np.ones(len(frames), dtype=bool)
    if args.start is not None:
        selected &= shadow["time_s"] >= args.start
    if args.duration is not None:
        beginning = args.start if args.start is not None else shadow["time_s"][0]
        selected &= shadow["time_s"] <= beginning + args.duration
    rows = np.flatnonzero(selected)
    if not len(rows):
        raise ValueError("Requested replay interval is outside the saved shadow result")
    return model, fit, contacts, shadow, rows


def draw_frame(model, data, scene, fit, contacts, shadow, row, force_scale):
    """Set the recorded pose and annotations; deliberately never call mj_step."""
    frame = int(shadow["frame"][row])
    data.qpos[:] = fit["qpos"][frame]
    data.qvel[:] = fit["qvel"][frame]
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    data.time = float(shadow["time_s"][row])
    mujoco.mj_forward(model, data)
    scene.ngeom = 0
    support = shadow["support_mask_candidate"][row].astype(bool)
    geometry = shadow["mujoco_geometry_contact"][row].astype(bool)
    # Older shadow archives can store NaN for failed solves: only finite 1 is solved.
    solved = bool(shadow["solved"][row] == 1)
    descriptions = []
    for side_index, side in enumerate(SIDES):
        sole_id = model.geom(f"{side}_sole").id
        site_id = model.site(f"{side}_foot").id
        model.geom_rgba[sole_id] = GREEN if support[side_index] else GRAY
        position = data.site_xpos[site_id].copy()
        if support[side_index] != geometry[side_index]:
            marker = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                marker,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.025),
                position + [0, 0, 0.06],
                np.eye(3).ravel(),
                ORANGE,
            )
            scene.ngeom += 1
        force = shadow["wrenches_world_about_foot_sites"][row, side_index, :3]
        if solved and np.isfinite(force).all() and np.linalg.norm(force) > 1e-6:
            arrow = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), position, np.eye(3).ravel(), GREEN
            )
            mujoco.mjv_connector(
                arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.012, position, position + force_scale * force
            )
            scene.ngeom += 1
        score = contacts["contact_probability"][frame, side_index]
        force_text = f"{np.linalg.norm(force):.1f} N" if solved else "unavailable"
        descriptions.append(
            f"{side}: candidate={int(support[side_index])}, "
            f"score={score:.2f}, geom={int(geometry[side_index])}, |F|={force_text}"
        )
    peak = "unavailable"
    if solved:
        torque = shadow["joint_torque_Nm"][row]
        index = int(np.argmax(np.abs(torque)))
        peak = f"{fit['joint_names'][index]}: {torque[index]:+.1f} Nm"
    return (
        "KINEMATIC REPLAY + OFFLINE QP (no torque-driven simulation)\n"
        f"t={data.time:.3f} s | QP: {shadow['status'][row]}\n"
        + "\n".join(descriptions)
        + f"\nPeak joint torque: {peak}\n"
        "Green: candidate / predicted force; orange dot: geometry disagreement\n"
        "Space: pause | Left/Right: step | R: restart | Esc: close"
    )


def run(args):
    model, fit, contacts, shadow, rows = load_replay(args)
    data = mujoco.MjData(model)
    if args.check:
        scene = mujoco.MjvScene(model, maxgeom=16)
        for row in rows:
            draw_frame(model, data, scene, fit, contacts, shadow, row, args.force_scale)
        print(
            f"Replay checked: {len(rows)} frames, "
            f"{shadow['time_s'][rows[0]]:.3f}–{shadow['time_s'][rows[-1]]:.3f} s; "
            "pose/force annotations constructed without a GUI or integration."
        )
        return
    import mujoco.viewer as viewer_module

    keys = deque()
    with viewer_module.launch_passive(model, data, key_callback=keys.append) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 115, -12
        index, paused = 0, False
        step_seconds = 1.0 / (float(fit["frame_rate"]) * args.speed)
        deadline = time.monotonic()
        while viewer.is_running():
            while keys:
                key = keys.popleft()
                if key == 32:
                    paused = not paused
                elif key in (262, 263):
                    paused = True
                    index = int(np.clip(index + (1 if key == 262 else -1), 0, len(rows) - 1))
                elif key in (82, 114):
                    index = 0
                deadline = time.monotonic() + step_seconds
            with viewer.lock():
                text = draw_frame(
                    model,
                    data,
                    viewer.user_scn,
                    fit,
                    contacts,
                    shadow,
                    rows[index],
                    args.force_scale,
                )
                viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
            if hasattr(viewer, "set_texts"):
                viewer.set_texts(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_100,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        text + ("\nPAUSED" if paused else ""),
                        "",
                    )
                )
            viewer.sync()
            now = time.monotonic()
            if not paused and now >= deadline:
                index += 1
                if index == len(rows):
                    index = len(rows) - 1 if args.once else 0
                    paused = args.once
                deadline = now + step_seconds
            time.sleep(min(0.01, step_seconds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/y13/human.xml")
    parser.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument(
        "--contacts", type=Path, default=ROOT / "outputs/y13/contact/contact_inference.npz"
    )
    parser.add_argument("--shadow", type=Path, default=ROOT / "outputs/y13/wbc_shadow.npz")
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--force-scale", type=float, default=0.001, help="Arrow metres per newton")
    parser.add_argument(
        "--once", action="store_true", help="Pause on last frame instead of looping"
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate all replay frames without a GUI"
    )
    args = parser.parse_args()
    values = [args.speed, args.force_scale]
    if args.duration is not None:
        values.append(args.duration)
    if not all(np.isfinite(value) and value > 0 for value in values):
        parser.error("speed, force scale and duration must be finite and positive")
    if args.start is not None and (not np.isfinite(args.start) or args.start < 0):
        parser.error("start must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
