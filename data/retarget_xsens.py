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
LINK_NAMES = (
    ["pelvis"]
    + [f"{side}_{part}_link" for side in SIDES for part in ("hip_yaw", "knee", "ankle_roll")]
    + ["left_foot", "right_foot"]
)
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


def map_foot_rotations(rotations, world_alignment, robot_reference, reference_frame):
    """Map Xsens world orientations to robot world, preserving rotation changes."""
    aligned_reference = world_alignment @ rotations[reference_frame]
    segment_calibration = aligned_reference.T @ robot_reference
    return world_alignment @ rotations @ segment_calibration


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
    parser.add_argument(
        "--foot-axis-weights",
        type=float,
        nargs=3,
        default=(8.0, 0.5, 8.0),
        metavar=("X", "Y", "Z"),
        help="swing foot position weights in world coordinates (default: 8 0.5 8)",
    )
    parser.add_argument(
        "--support-threshold",
        type=float,
        default=0.01,
        help="support height above each Xsens foot's estimated ground baseline (m)",
    )
    parser.add_argument(
        "--ground-percentile",
        type=float,
        default=7.0,
        help="percentile used to estimate each Xsens foot's ground baseline",
    )
    parser.add_argument(
        "--foot-orientation-weight",
        type=float,
        default=5.0,
        help="SO(3) orientation residual weight, for support and swing feet",
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
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_NAMES[:-2]]
    )
    if np.any(link_ids < 0):
        raise ValueError("Model is missing a required reference link")
    foot_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot") for side in SIDES
    }
    if any(body_id < 0 for body_id in foot_ids.values()):
        raise ValueError("Model must define left_foot and right_foot reference sites")
    foot_axis_weights = np.asarray(args.foot_axis_weights, dtype=float)
    if (
        not np.isfinite(foot_axis_weights).all()
        or np.any(foot_axis_weights < 0)
        or not np.any(foot_axis_weights > 0)
    ):
        parser.error("--foot-axis-weights must be non-negative with at least one positive value")
    if not np.isfinite(args.foot_orientation_weight) or args.foot_orientation_weight <= 0:
        parser.error("--foot-orientation-weight must be positive and finite")
    if args.support_threshold <= 0 or not np.isfinite(args.support_threshold):
        parser.error("--support-threshold must be positive and finite")
    if not 0 <= args.ground_percentile < 50:
        parser.error("--ground-percentile must be in [0, 50)")

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
    foot_segment_indices = np.array([3, 6])
    raw_foot_positions = positions[:, foot_segment_indices]
    ground_baselines = np.quantile(
        raw_foot_positions[:, :, 2], args.ground_percentile / 100.0, axis=0
    )
    support_mask = raw_foot_positions[:, :, 2] <= ground_baselines[None, :] + args.support_threshold
    # Do not allow a frame with no support: choose the lower foot as the fallback.
    no_support = ~support_mask.any(axis=1)
    if np.any(no_support):
        lower_foot = np.argmin(raw_foot_positions[no_support, :, 2], axis=1)
        support_mask[np.flatnonzero(no_support), lower_foot] = True
    support_state = np.where(
        support_mask.all(axis=1),
        "double_support",
        np.where(support_mask[:, 0], "left_support", "right_support"),
    )
    robot_base_rot = data.xmat[base_id].reshape(3, 3).copy()
    ref_points = {}
    ref_foot_rot = {}
    for side in SIDES:
        _, knee, _ = body_ids[side]
        ref_points[side] = np.array(
            [
                robot_base_rot.T @ (data.xpos[knee] - data.xpos[base_id]),
                robot_base_rot.T @ (data.site_xpos[foot_ids[side]] - data.xpos[base_id]),
            ]
        )
        ref_foot_rot[side] = robot_base_rot.T @ data.site_xmat[foot_ids[side]].reshape(3, 3)

    # Map world-space foot orientations with a fixed reference-frame calibration.
    # R_target(t) = C R_xsens(t) [C R_xsens(ref)]^T R_robot(ref).
    # The transform never depends on the current pelvis orientation or support state.
    xsens_to_robot_rot = robot_base_rot @ pelvis_rot[ref].T
    foot_rotation_targets = np.stack(
        [
            map_foot_rotations(
                rotations[:, segment], xsens_to_robot_rot, robot_base_rot @ ref_foot_rot[side], ref
            )
            for side, segment in zip(SIDES, foot_segment_indices)
        ],
        axis=1,
    )
    foot_contact_targets = np.stack(
        [
            qref[:3]
            + robot_base_rot @ ref_points[side][1]
            + scale
            * (raw_foot_positions[:, si] - raw_foot_positions[ref, si])
            @ xsens_to_robot_rot.T
            for si, side in enumerate(SIDES)
        ],
        axis=1,
    )
    ground_z = float(np.mean([data.site_xpos[foot_ids[s]][2] for s in SIDES]))
    # Preserve raw vertical clearance instead of tilting it with initial pelvis pitch.
    foot_contact_targets[:, :, 2] = ground_z + scale * np.maximum(
        raw_foot_positions[:, :, 2] - ground_baselines[None, :], 0.0
    )
    foot_contact_targets[:, :, 2][support_mask] = ground_z

    qpos = np.empty((len(frames), model.nq))
    errors = np.empty((len(frames), 2, 3))
    foot_position_errors = np.empty((len(frames), 2, 3))
    foot_orientation_errors = np.empty((len(frames), 2, 3))
    support_base_adjustment = np.zeros((len(frames), 3))
    link_positions = np.empty((len(frames), len(LINK_NAMES), 3))
    link_orientations = np.empty((len(frames), len(LINK_NAMES), 4))
    previous = qref[qadr].copy()
    site_ids = np.array([foot_ids[side] for side in SIDES])
    knee_ids = np.array([body_ids[side][1] for side in SIDES])
    joint_lower, joint_upper = limits[:, 0], limits[:, 1]
    lower = np.r_[[-np.inf] * 3, joint_lower]
    upper = np.r_[[np.inf] * 3, joint_upper]
    previous_base_adjustment = np.zeros(3)
    solver_nfev = np.empty(len(frames), dtype=int)
    solver_success = np.empty(len(frames), dtype=bool)
    for frame in range(len(frames)):
        base_rot = xsens_to_robot_rot @ pelvis_rot[frame]
        base_nominal = qref[:3] + scale * xsens_to_robot_rot @ (
            positions[frame, 0] - positions[ref, 0]
        )
        xyzw = Rotation.from_matrix(base_rot).as_quat()
        data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
        knee_targets = np.array(
            [
                ref_points[side][0]
                + scale * (local_pos[frame, offset + 1] - local_pos[ref, offset + 1])
                for side, offset in zip(SIDES, (1, 4))
            ]
        )
        position_weights = np.broadcast_to(foot_axis_weights, (2, 3)).copy()
        # Height-only support detection constrains z, not a frozen sole orientation.
        # Track measured xy (including pivots); y remains soft for different hip widths.
        position_weights[support_mask[frame]] = [8.0, 0.5, 1000.0]

        def residual(x):
            data.qpos[:3] = base_nominal + x[:3]
            data.qpos[qadr] = x[3:]
            mujoco.mj_forward(model, data)
            knee_local = (data.xpos[knee_ids] - data.xpos[base_id]) @ base_rot
            foot_world = data.site_xpos[site_ids]
            foot_rot = data.site_xmat[site_ids].reshape(2, 3, 3)
            rot_err = Rotation.from_matrix(
                foot_rotation_targets[frame].transpose(0, 2, 1) @ foot_rot
            ).as_rotvec()
            return np.r_[
                (position_weights * (foot_world - foot_contact_targets[frame])).ravel(),
                (args.foot_orientation_weight * rot_err).ravel(),
                (0.3 * (knee_local - knee_targets)).ravel(),
                0.1 * (x[3:] - previous),
                0.005 * (x[3:] - qref[qadr]),
                0.3 * x[:3],
                0.1 * (x[:3] - previous_base_adjustment),
            ]

        result = least_squares(
            residual,
            np.r_[previous_base_adjustment, previous],
            bounds=(lower, upper),
            max_nfev=160,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        if not result.success:
            # Tight height constraints can require more iterations near leg singularities.
            initial_nfev = result.nfev
            result = least_squares(
                residual,
                result.x,
                bounds=(lower, upper),
                max_nfev=320,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            result.nfev += initial_nfev
        residual(result.x)
        solver_nfev[frame] = result.nfev
        solver_success[frame] = result.success
        previous = result.x[3:].copy()
        previous_base_adjustment = result.x[:3].copy()
        support_base_adjustment[frame] = result.x[:3]
        knee_local = (data.xpos[knee_ids] - data.xpos[base_id]) @ base_rot
        foot_position_errors[frame] = data.site_xpos[site_ids] - foot_contact_targets[frame]
        foot_rot = data.site_xmat[site_ids].reshape(2, 3, 3)
        foot_orientation_errors[frame] = Rotation.from_matrix(
            foot_rotation_targets[frame].transpose(0, 2, 1) @ foot_rot
        ).as_rotvec()
        errors[frame, :, 0] = np.linalg.norm(knee_local - knee_targets, axis=1)
        errors[frame, :, 1] = np.linalg.norm(foot_position_errors[frame], axis=1)
        errors[frame, :, 2] = np.linalg.norm(foot_orientation_errors[frame], axis=1)
        qpos[frame] = data.qpos
        link_positions[frame, :-2] = data.xpos[link_ids]
        link_positions[frame, -2:] = data.site_xpos[site_ids]
        link_orientations[frame, :-2] = data.xquat[link_ids]
        xyzw = Rotation.from_matrix(foot_rot).as_quat()
        link_orientations[frame, -2:] = xyzw[:, [3, 0, 1, 2]]
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
        foot_position_errors=foot_position_errors,
        foot_axis_weights=foot_axis_weights,
        support_mask=support_mask,
        support_state=support_state,
        foot_contact_targets=foot_contact_targets,
        foot_orientation_targets=Rotation.from_matrix(foot_rotation_targets.reshape(-1, 3, 3))
        .as_quat()[:, [3, 0, 1, 2]]
        .reshape(-1, 2, 4),
        foot_orientation_errors=foot_orientation_errors,
        foot_orientation_weight=args.foot_orientation_weight,
        xsens_to_robot_rotation=xsens_to_robot_rot,
        ground_percentile=args.ground_percentile,
        solver_nfev=solver_nfev,
        solver_success=solver_success,
        support_ground_baselines=ground_baselines,
        support_threshold=args.support_threshold,
        support_ground_z=ground_z,
        support_base_adjustment=support_base_adjustment,
    )
    print(f"Saved {args.output}; scale={scale:.3f}")
    for si, side in enumerate(SIDES):
        e = errors[:, si]
        print(
            f"{side}: mean knee/foot position error {e[:, 0].mean():.3f}/{e[:, 1].mean():.3f} m, "
            f"foot orientation mean/max error "
            f"{np.degrees(e[:, 2]).mean():.4f}/{np.degrees(e[:, 2]).max():.4f} deg, "
            f"mean abs foot xyz error {np.abs(foot_position_errors[:, si]).mean(axis=0)} m"
        )


if __name__ == "__main__":
    main()
