#!/usr/bin/env python3
"""Retarget lower-body Xsens segment motion to the SZAR MuJoCo model."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import openpyxl
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
PARTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
JOINTS = [f"{side}_{part}_joint" for side in SIDES for part in PARTS]
LINK_NAMES = ["pelvis"] + [
    f"{side}_{part}_link" for side in SIDES for part in ("hip_yaw", "knee", "ankle_roll")
]
SEGMENTS = ["Pelvis"] + [
    f"{side} {part}" for side in ("Left", "Right") for part in ("Upper Leg", "Lower Leg", "Foot")
]


def read_sheet(book, name, fields):
    rows = book[name].iter_rows(values_only=True)
    header = next(rows)
    columns = [header.index(field) for field in fields]
    values = np.asarray([[row[i] for i in columns] for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or non-finite values")
    return values


def load_xsens(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        pos = read_sheet(
            book,
            "Segment Position",
            ["Frame"] + [f"{s} {axis}" for s in SEGMENTS for axis in "xyz"],
        )
        quat = read_sheet(
            book,
            "Segment Orientation - Quat",
            ["Frame"] + [f"{s} q{i}" for s in SEGMENTS for i in range(4)],
        )
        info = dict(book["General Information"].iter_rows(values_only=True))
    finally:
        book.close()
    if len(pos) == 0 or not np.array_equal(pos[:, 0], quat[:, 0]):
        raise ValueError("Position and orientation frames do not match")
    frames = pos[:, 0]
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Frame numbers must be strictly increasing")
    positions = pos[:, 1:].reshape(-1, len(SEGMENTS), 3)
    # Xsens q0,q1,q2,q3 is scalar-first; SciPy expects scalar-last.
    quats = quat[:, 1:].reshape(-1, len(SEGMENTS), 4)
    if np.any(np.linalg.norm(quats, axis=-1) < 1e-8):
        raise ValueError("Invalid zero-length quaternion")
    rotations = Rotation.from_quat(quats.reshape(-1, 4)[:, [1, 2, 3, 0]])
    matrices = rotations.as_matrix().reshape(-1, len(SEGMENTS), 3, 3)
    return frames, positions, matrices, float(info["Frame Rate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/xsense_raw.xlsx")
    parser.add_argument("--model", type=Path, default=ROOT / "model/szar_humanoid.xml")
    parser.add_argument("--output", type=Path, default=ROOT / "data/retargeted_motion.npz")
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="zero-based row index for calibration (default: 0)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="human-to-robot translation scale; default: estimate from leg lengths",
    )
    args = parser.parse_args()
    frames, positions, rotations, fps = load_xsens(args.input)
    ref = args.reference_frame
    if not 0 <= ref < len(frames):
        parser.error("--reference-frame is outside the recording")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
    if key < 0 or model.nq != 19 or model.nv != 18:
        raise ValueError("Expected the floating-base SZAR model and its standing keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)
    qref = data.qpos.copy()
    ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
    if np.any(ids < 0):
        raise ValueError("Model is missing a required leg joint")
    qadr = model.jnt_qposadr[ids]
    limits = model.jnt_range[ids]
    body_ids = {
        side: [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{part}_link")
            for part in ("hip_yaw", "knee", "ankle_roll")
        ]
        for side in SIDES
    }
    if any(i < 0 for ids_ in body_ids.values() for i in ids_):
        raise ValueError("Model is missing a required leg body")
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if base_id < 0:
        raise ValueError("Model is missing pelvis body")
    link_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_NAMES]
    )
    if np.any(link_ids < 0):
        raise ValueError("Model is missing a required reference link")

    # Compare articulated segment lengths, not pelvis height, for spatial scaling.
    lengths = []
    for side, offset in zip(SIDES, (1, 4)):
        human = np.linalg.norm(
            positions[ref, offset + 1] - positions[ref, offset]
        ) + np.linalg.norm(positions[ref, offset + 2] - positions[ref, offset + 1])
        hip, knee, foot = body_ids[side]
        robot = np.linalg.norm(data.xpos[knee] - data.xpos[hip]) + np.linalg.norm(
            data.xpos[foot] - data.xpos[knee]
        )
        if human < 1e-6:
            raise ValueError("Cannot estimate scale from zero-length human leg")
        lengths.append(robot / human)
    scale = args.scale if args.scale is not None else float(np.mean(lengths))
    if scale <= 0 or not np.isfinite(scale):
        parser.error("--scale must be positive and finite")

    pelvis_rot = rotations[:, 0]
    local_pos = np.einsum("nji,nsj->nsi", pelvis_rot, positions - positions[:, :1])
    local_rot = np.einsum("nji,nsjk->nsik", pelvis_rot, rotations)
    robot_base_rot = data.xmat[base_id].reshape(3, 3).copy()
    ref_points = {}
    ref_foot_rot = {}
    for side in SIDES:
        _, knee, foot = body_ids[side]
        ref_points[side] = np.array(
            [robot_base_rot.T @ (data.xpos[b] - data.xpos[base_id]) for b in (knee, foot)]
        )
        ref_foot_rot[side] = robot_base_rot.T @ data.xmat[foot].reshape(3, 3)

    qpos = np.empty((len(frames), model.nq))
    errors = np.empty((len(frames), 2, 3))
    link_positions = np.empty((len(frames), len(LINK_NAMES), 3))
    link_orientations = np.empty((len(frames), len(LINK_NAMES), 4))
    previous = qref[qadr].copy()
    for frame in range(len(frames)):
        # Calibrate the initial human pelvis frame to the robot standing frame.
        base_rot = robot_base_rot @ pelvis_rot[ref].T @ pelvis_rot[frame]
        base_delta = robot_base_rot @ pelvis_rot[ref].T @ (positions[frame, 0] - positions[ref, 0])
        data.qpos[:3] = qref[:3] + scale * base_delta
        xyzw = Rotation.from_matrix(base_rot).as_quat()
        data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
        data.qpos[qadr] = previous
        for si, (side, offset) in enumerate(zip(SIDES, (1, 4))):
            sl = slice(si * 6, (si + 1) * 6)
            knee_id, foot_id = body_ids[side][1:]
            target = ref_points[side] + scale * (
                local_pos[frame, [offset + 1, offset + 2]]
                - local_pos[ref, [offset + 1, offset + 2]]
            )
            target_rot = (
                local_rot[frame, offset + 2] @ local_rot[ref, offset + 2].T @ ref_foot_rot[side]
            )

            def residual(q):
                data.qpos[qadr[sl]] = q
                mujoco.mj_forward(model, data)
                points = np.array(
                    [base_rot.T @ (data.xpos[b] - data.xpos[base_id]) for b in (knee_id, foot_id)]
                )
                foot_rot = base_rot.T @ data.xmat[foot_id].reshape(3, 3)
                rot_err = Rotation.from_matrix(target_rot.T @ foot_rot).as_rotvec()
                return np.r_[
                    2.0 * (points[0] - target[0]),
                    4.0 * (points[1] - target[1]),
                    0.25 * rot_err,
                    0.025 * (q - previous[sl]),
                    0.005 * (q - qref[qadr[sl]]),
                ]

            result = least_squares(
                residual,
                previous[sl],
                bounds=(limits[sl, 0], limits[sl, 1]),
                max_nfev=25,
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4,
            )
            residual(result.x)
            previous[sl] = result.x
            points = np.array(
                [base_rot.T @ (data.xpos[b] - data.xpos[base_id]) for b in (knee_id, foot_id)]
            )
            foot_rot = base_rot.T @ data.xmat[foot_id].reshape(3, 3)
            errors[frame, si] = [
                np.linalg.norm(points[0] - target[0]),
                np.linalg.norm(points[1] - target[1]),
                np.linalg.norm(Rotation.from_matrix(target_rot.T @ foot_rot).as_rotvec()),
            ]
        # FK of the solved robot pose: world-frame body origins and wxyz quaternions.
        mujoco.mj_forward(model, data)
        qpos[frame] = data.qpos
        link_positions[frame] = data.xpos[link_ids]
        link_orientations[frame] = data.xquat[link_ids]
        if (frame + 1) % 300 == 0 or frame + 1 == len(frames):
            print(f"Processed {frame + 1}/{len(frames)} frames", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=qpos,
        frame=frames,
        fps=fps,
        joint_names=np.array(JOINTS),
        joint_qpos_addresses=qadr,
        errors=errors,
        scale=scale,
        reference_frame=ref,
        link_names=np.array(LINK_NAMES),
        link_positions=link_positions,
        link_orientations=link_orientations,
    )
    print(f"Saved {args.output}; scale={scale:.3f}")
    for si, side in enumerate(SIDES):
        e = errors[:, si]
        print(
            f"{side}: mean knee/foot position error {e[:, 0].mean():.3f}/{e[:, 1].mean():.3f} m, "
            f"foot orientation error {np.degrees(e[:, 2]).mean():.1f} deg"
        )


if __name__ == "__main__":
    main()
