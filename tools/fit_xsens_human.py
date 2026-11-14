#!/usr/bin/env python3
"""Fit seven Xsens segment orientations to the simplified MS-Human MJCF."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import openpyxl
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/y13_MVN_3_8_01.xlsx"
TEMPLATE_MODEL = ROOT / "model/ms_human_locomotion/human.xml"
DEFAULT_MODEL = ROOT / "outputs/y13/human.xml"
DEFAULT_OUTPUT = ROOT / "outputs/y13/fit.npz"
SEGMENTS = (
    ("Pelvis", "pelvis", None),
    ("Right Upper Leg", "right_thigh", "right"),
    ("Right Lower Leg", "right_shank", "right"),
    ("Right Foot", "right_foot", "right"),
    ("Left Upper Leg", "left_thigh", "left"),
    ("Left Lower Leg", "left_shank", "left"),
    ("Left Foot", "left_foot", "left"),
)
LEG_JOINTS = {
    "right": (
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee_pitch",
        "right_ankle_pitch",
        "right_ankle_roll",
    ),
    "left": (
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee_pitch",
        "left_ankle_pitch",
        "left_ankle_roll",
    ),
}


def load_xsens(path: Path):
    """Read the seven measured segment quaternions and pelvis positions."""
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        metadata = {
            str(row[0]): row[1]
            for row in workbook["General Information"].iter_rows(values_only=True)
            if len(row) > 1 and row[0] is not None
        }
        rate = float(metadata["Frame Rate"])
        sheet = workbook["Segment Orientation - Quat"]
        header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        names = [str(value) for value in header]
        columns = []
        for segment, _, _ in SEGMENTS:
            columns.append([names.index(f"{segment} q{i}") for i in range(4)])
        frame_col = names.index("Frame")
        rows = np.asarray(list(sheet.iter_rows(min_row=2, values_only=True)), dtype=np.float64)
        frames = rows[:, frame_col].astype(np.int64)
        quaternions = np.stack([rows[:, cols] for cols in columns], axis=1)
        pos_sheet = workbook["Segment Position"]
        pos_header = [
            str(x) for x in next(pos_sheet.iter_rows(min_row=1, max_row=1, values_only=True))
        ]
        position_columns = [
            [pos_header.index(f"{segment} {axis}") for axis in ("x", "y", "z")]
            for segment, _, _ in SEGMENTS
        ]
        position_rows = np.asarray(
            list(pos_sheet.iter_rows(min_row=2, values_only=True)), dtype=np.float64
        )
        if not np.array_equal(position_rows[:, 0].astype(np.int64), frames):
            raise ValueError("Xsens segment-position frame numbers do not match orientation frames")
        segment_positions = np.stack(
            [position_rows[:, columns] for columns in position_columns], axis=1
        )
    finally:
        workbook.close()
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Xsens frame numbers are not continuous from zero")
    return frames, rate, quaternions, segment_positions, metadata


def quaternion_rotations(quaternions):
    norms = np.linalg.norm(quaternions, axis=-1)
    valid = np.isfinite(quaternions).all(axis=-1) & (norms > 0.5) & (norms < 1.5)
    normalized = np.zeros_like(quaternions)
    normalized[..., 0] = 1.0
    normalized[valid] = quaternions[valid] / norms[valid, None]
    # Xsens q0..q3 is scalar-first; scipy and MuJoCo matrices are normalized.
    rotations = Rotation.from_quat(normalized[..., [1, 2, 3, 0]].reshape(-1, 4))
    return rotations.as_matrix().reshape(quaternions.shape[0], quaternions.shape[1], 3, 3), valid


def audit_and_correct(
    rotations,
    quaternion_valid,
    pelvis_position,
    frame_rate,
    common_jump_deg=15.0,
    position_jump_m=1.0,
):
    """Remove detected common navigation-frame resets and flag their neighborhoods."""
    count = len(rotations)
    position = pelvis_position.copy()
    valid = quaternion_valid.all(axis=1) & np.isfinite(position).all(axis=1)
    event = np.zeros(count, dtype=bool)
    rotation_steps = np.rad2deg(
        Rotation.from_matrix(rotations[:-1].transpose(0, 1, 3, 2) @ rotations[1:]).magnitude()
    ).T
    # Use common jumps only as a diagnostic navigation-frame reset hypothesis.
    rotations_corrected = rotations.copy()
    position_corrected = position.copy()
    reset_events = []
    active_world_correction = np.eye(3)
    cumulative_position_offset = np.zeros(3)
    for index in range(1, count):
        rotations_corrected[index] = active_world_correction @ rotations[index]
        position_corrected[index] = position[index] + cumulative_position_offset
        jumps = rotation_steps[:, index - 1]
        common = (
            quaternion_valid[index].all()
            and quaternion_valid[index - 1].all()
            and int(np.sum(jumps > common_jump_deg)) >= 5
        )
        translation = np.linalg.norm(position[index] - position[index - 1]) > position_jump_m
        if common or translation:
            event[
                max(0, index - int(frame_rate * 0.25)) : min(
                    count, index + int(frame_rate * 0.25) + 1
                )
            ] = True
            original_delta = rotations[index, 0] @ rotations[index - 1, 0].T
            corrected_delta = active_world_correction @ original_delta @ active_world_correction.T
            if common:
                active_world_correction = (
                    Rotation.from_matrix(corrected_delta.T).as_matrix() @ active_world_correction
                )
                # Re-apply the updated correction at the discontinuity frame.
                rotations_corrected[index] = active_world_correction @ rotations[index]
                reset_events.append(
                    {
                        "frame": int(index),
                        "time_s": index / frame_rate,
                        "type": "common_orientation_reset",
                        "segment_steps_deg": jumps.tolist(),
                    }
                )
            if translation:
                delta = position[index] - position[index - 1]
                cumulative_position_offset -= delta
                position_corrected[index] = position[index] + cumulative_position_offset
                reset_events.append(
                    {
                        "frame": int(index),
                        "time_s": index / frame_rate,
                        "type": "pelvis_position_reset",
                        "step_m": float(np.linalg.norm(delta)),
                    }
                )
    # Reject any remaining implausible one-frame segment jumps.
    corrected_steps = np.rad2deg(
        Rotation.from_matrix(
            rotations_corrected[:-1].transpose(0, 1, 3, 2) @ rotations_corrected[1:]
        ).magnitude()
    ).T
    isolated = np.any(corrected_steps > 20.0, axis=0)
    for index in np.flatnonzero(isolated):
        event[max(0, index - 1) : min(count, index + 3)] = True
    valid &= ~event
    return rotations_corrected, position_corrected, valid, event, reset_events


def quat_wxyz(matrix):
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return xyzw[[3, 0, 1, 2]]


def body_rotation(data, body_id):
    return data.xmat[body_id].reshape(3, 3).copy()


def infer_heading(position, valid, frame_rate):
    indices = np.flatnonzero(valid)
    if len(indices) < 10:
        raise ValueError("Not enough valid Xsens pelvis-position frames")
    window = max(1, int(round(2.0 * frame_rate)))
    # Estimate forward travel direction from two-second displacements, rejecting
    # gaps and outliers instead of trusting the reset-crossing endpoint.
    angles = []
    for i in indices[:: max(1, window // 4)]:
        j = min(i + window, len(position) - 1)
        if j <= i or not (valid[i] and valid[j]) or np.any(~valid[i : j + 1]):
            continue
        delta = position[j, :2] - position[i, :2]
        if np.linalg.norm(delta) > 0.25:
            angles.append(np.arctan2(delta[1], delta[0]))
    if not angles:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angles)))))


def fit_motion(
    input_path,
    model_path,
    output_path,
    rate=60.0,
    reference_frame=0,
    max_nfev=30,
    align_heading=False,
    root_mode="in_place",
):
    input_path = Path(input_path)
    model_path = Path(model_path)
    output_path = Path(output_path)
    source_xlsx_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    frames, source_rate, quats, segment_positions, metadata = load_xsens(input_path)
    pelvis_position = segment_positions[:, 0].copy()
    raw_rotations, quat_valid = quaternion_rotations(quats)
    rotations, pelvis_position, frame_valid, event, reset_events = audit_and_correct(
        raw_rotations, quat_valid, pelvis_position, source_rate
    )
    world_corrections = rotations[:, 0] @ raw_rotations[:, 0].transpose(0, 2, 1)
    relative_positions = segment_positions - segment_positions[:, :1]
    segment_positions = pelvis_position[:, None, :] + np.einsum(
        "tij,tkj->tki", world_corrections, relative_positions
    )
    if not 0 <= reference_frame < len(frames) or not frame_valid[reference_frame]:
        raise ValueError("Reference frame is invalid or lies in a detected discontinuity")

    heading = infer_heading(pelvis_position, frame_valid, source_rate) if align_heading else 0.0
    world_align = Rotation.from_euler("z", -heading).as_matrix()
    pelvis_position = (world_align @ pelvis_position.T).T
    segment_positions = (world_align @ segment_positions.reshape(-1, 3).T).T.reshape(
        segment_positions.shape
    )
    rotations = world_align @ rotations @ world_align.T

    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
    if key_id < 0:
        raise ValueError("Human model must define the 'standing' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    qref = data.qpos.copy()
    mujoco.mj_forward(model, data)

    segment_names = [entry[0] for entry in SEGMENTS]
    model_body_ids = [model.body(entry[1]).id for entry in SEGMENTS]
    reference_rotations = rotations[reference_frame]
    model_reference = np.stack([body_rotation(data, body) for body in model_body_ids])
    pelvis_body_reference = model_reference[0].copy()
    xsens_pelvis_calibration = pelvis_body_reference.T @ rotations[reference_frame, 0]
    observed_positions_local = np.einsum(
        "tij,tkj->tki",
        rotations[:, 0].transpose(0, 2, 1),
        segment_positions - segment_positions[:, :1],
    )
    observed_positions_reference = observed_positions_local[reference_frame].copy()
    position_joint_names = [
        None,
        "right_hip_pitch",
        "right_knee_pitch",
        "right_ankle_pitch",
        "left_hip_pitch",
        "left_knee_pitch",
        "left_ankle_pitch",
    ]
    position_joint_ids = [
        model.joint(n).id if n is not None else None for n in position_joint_names
    ]

    def proximal_positions():
        return np.stack(
            [
                data.xpos[model_body_ids[0]] if j is None else data.xanchor[j]
                for j in position_joint_ids
            ]
        )

    model_positions_reference = (
        proximal_positions() - data.xpos[model_body_ids[0]]
    ) @ pelvis_body_reference
    targets = np.empty_like(rotations)
    for i in range(len(frames)):
        for segment in range(len(SEGMENTS)):
            delta = rotations[i, segment] @ reference_rotations[segment].T
            targets[i, segment] = delta @ model_reference[segment]

    free_joint = model.joint("pelvis_free").id
    root_qadr = model.jnt_qposadr[free_joint]
    root_xyz_adr = int(root_qadr)
    root_quat_adr = root_xyz_adr + 3
    base_position = qref[root_xyz_adr : root_xyz_adr + 3].copy()
    position_ref = pelvis_position[reference_frame].copy()
    joint_qadr = {
        name: int(model.jnt_qposadr[model.joint(name).id])
        for names in LEG_JOINTS.values()
        for name in names
    }
    bounds = {
        side: (
            np.array([model.jnt_range[model.joint(name).id, 0] for name in LEG_JOINTS[side]]),
            np.array([model.jnt_range[model.joint(name).id, 1] for name in LEG_JOINTS[side]]),
        )
        for side in LEG_JOINTS
    }
    model_frame_indices = np.unique(
        np.rint(np.arange(0, len(frames), source_rate / rate)).astype(int)
    )
    model_frame_indices = model_frame_indices[model_frame_indices < len(frames)]
    output_frames = frames[model_frame_indices]
    qpos = np.empty((len(model_frame_indices), model.nq))
    angle_residual = np.full((len(model_frame_indices), 7), np.nan)
    position_residual = np.full((len(model_frame_indices), 7), np.nan)
    solver_success = np.zeros((len(model_frame_indices), 2), dtype=bool)
    solver_nfev = np.zeros((len(model_frame_indices), 2), dtype=np.int16)
    output_valid = frame_valid[model_frame_indices].copy()
    if rate > source_rate or rate < 2 or len(model_frame_indices) < 3:
        raise ValueError("Output rate must be between 2 Hz and the source rate")
    previous = {side: np.zeros(6) for side in LEG_JOINTS}
    body_slots = {name: i for i, name in enumerate(segment_names)}

    for out_index, source_index in enumerate(model_frame_indices):
        if out_index and out_index % 3000 == 0:
            print(f"Fitted {out_index}/{len(model_frame_indices)} frames", flush=True)
        data.qpos[:] = qref
        pelvis_delta = pelvis_position[source_index] - position_ref
        if root_mode == "in_place":
            pelvis_delta[:2] = 0.0
        data.qpos[root_xyz_adr : root_xyz_adr + 3] = base_position + pelvis_delta
        desired_pelvis = targets[source_index, body_slots["Pelvis"]]
        # A free-joint quaternion directly specifies the body world orientation.
        data.qpos[root_quat_adr : root_quat_adr + 4] = quat_wxyz(desired_pelvis)
        mujoco.mj_forward(model, data)

        for side in LEG_JOINTS:
            names = LEG_JOINTS[side]
            qadr = np.array([joint_qadr[name] for name in names])
            lo, hi = bounds[side]
            body_names = [f"{side}_thigh", f"{side}_shank", f"{side}_foot"]
            body_ids = [model.body(name).id for name in body_names]
            target_slots = [
                body_slots["Right Upper Leg" if side == "right" else "Left Upper Leg"],
                body_slots["Right Lower Leg" if side == "right" else "Left Lower Leg"],
                body_slots["Right Foot" if side == "right" else "Left Foot"],
            ]
            desired = targets[source_index, target_slots]

            def residual(angles):
                data.qpos[qadr] = angles
                mujoco.mj_forward(model, data)
                result = []
                for body_id, target in zip(body_ids, desired):
                    error = target.T @ body_rotation(data, body_id)
                    result.extend(Rotation.from_matrix(error).as_rotvec())
                return np.asarray(result)

            result = least_squares(
                residual,
                np.clip(previous[side], lo + 1e-9, hi - 1e-9),
                bounds=(lo, hi),
                max_nfev=max_nfev,
                ftol=1e-7,
                xtol=1e-7,
                gtol=1e-7,
            )
            previous[side] = result.x
            data.qpos[qadr] = result.x
            solver_success[out_index, 0 if side == "right" else 1] = result.success
            solver_nfev[out_index, 0 if side == "right" else 1] = result.nfev
        mujoco.mj_forward(model, data)
        qpos[out_index] = data.qpos
        angle_residual[out_index, 0] = 0.0
        position_residual[out_index, 0] = 0.0
        for slot, (segment, body_name, _) in enumerate(SEGMENTS[1:], start=1):
            body_id = model.body(body_name).id
            error = targets[source_index, slot].T @ body_rotation(data, body_id)
            angle_residual[out_index, slot] = np.rad2deg(
                np.linalg.norm(Rotation.from_matrix(error).as_rotvec())
            )
            target_local = model_positions_reference[slot] + xsens_pelvis_calibration @ (
                observed_positions_local[source_index, slot] - observed_positions_reference[slot]
            )
            target_world = (
                data.xpos[model_body_ids[0]] + body_rotation(data, model_body_ids[0]) @ target_local
            )
            position_residual[out_index, slot] = np.linalg.norm(
                proximal_positions()[slot] - target_world
            )

    # A trial-specific reference must not send the rigid sole through the floor.
    # This is a model-based walking contact assumption, not measured pelvis Z.
    grounding_offset = np.zeros(len(qpos))
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        bottoms = []
        for side in ("left", "right"):
            geom = model.geom(f"{side}_sole").id
            bottoms.append(
                data.geom_xpos[geom, 2]
                - np.abs(data.geom_xmat[geom].reshape(3, 3)[2]) @ model.geom_size[geom]
            )
        grounding_offset[index] = -min(bottoms) - 0.001
    qpos[:, 2] += grounding_offset
    times = (output_frames - output_frames[0]) / source_rate
    # Derivatives never cross a reset/invalid interval. Keep original fitted poses
    # and differentiate a lightly smoothed copy within each contiguous valid run.
    tracking_qpos = qpos.copy()
    # Keep the kinematic-fit poses unchanged; expose a separate smoothed
    # reference for physical tracking, including the nonsmooth sole envelope.
    runs = np.flatnonzero(np.diff(np.r_[False, output_valid, False]))
    for start, stop in zip(runs[::2], runs[1::2]):
        if stop - start < 7:
            continue
        width = min(int(round(rate * 0.25)) | 1, ((stop - start) // 2) * 2 - 1)
        tracking_qpos[start:stop, :3] = savgol_filter(qpos[start:stop, :3], width, 3, axis=0)
        tracking_qpos[start:stop, 7:] = savgol_filter(qpos[start:stop, 7:], width, 3, axis=0)
        for j in range(1, model.njnt):
            if model.jnt_limited[j]:
                adr = model.jnt_qposadr[j]
                tracking_qpos[start:stop, adr] = np.clip(
                    tracking_qpos[start:stop, adr], *model.jnt_range[j]
                )
    qvel = np.full((len(qpos), model.nv), np.nan)
    qacc = np.full_like(qvel, np.nan)
    derivative_valid = np.zeros(len(qpos), dtype=bool)
    boundaries = np.flatnonzero(np.diff(np.r_[False, output_valid, False]))
    for start, stop in zip(boundaries[::2], boundaries[1::2]):
        if stop - start < 7:
            continue
        poses = tracking_qpos[start:stop].copy()
        window = min(int(round(rate * 0.15)) | 1, (stop - start - 1) | 1)
        if window >= 5:
            poses[:, :3] = savgol_filter(poses[:, :3], window, 3, axis=0)
            poses[:, 7:] = savgol_filter(poses[:, 7:], window, 3, axis=0)
        for i in range(1, len(poses) - 1):
            mujoco.mj_differentiatePos(
                model,
                qvel[start + i],
                times[start + i + 1] - times[start + i - 1],
                poses[i - 1],
                poses[i + 1],
            )
        qvel[start] = qvel[start + 1]
        qvel[stop - 1] = qvel[stop - 2]
        qacc[start:stop] = np.gradient(qvel[start:stop], times[start:stop], axis=0)
        derivative_valid[start + 2 : stop - 2] = True

    residual_summary = {
        "orientation_deg_median_p95_max_by_segment": {
            segment[0]: np.nanpercentile(angle_residual[:, i], [50, 95, 100]).tolist()
            for i, segment in enumerate(SEGMENTS)
        },
        "position_residual_m_median_p95_max_by_segment": {
            segment[0]: np.nanpercentile(position_residual[:, i], [50, 95, 100]).tolist()
            for i, segment in enumerate(SEGMENTS)
        },
        "root_grounding_offset_m_min_median_max": np.percentile(
            grounding_offset, [0, 50, 100]
        ).tolist(),
        "valid_fraction": float(np.mean(output_valid)),
        "invalid_output_frames": int(np.sum(~output_valid)),
        "solver_failure_count": int(np.sum(~solver_success)),
        "validity_meaning": "no detected discontinuity and finite input; not experimental accuracy validation",
        "reference_pose_status": "frame mapped to neutral as a provisional assumption, not confirmed calibration",
        "source_processing_quality": str(metadata.get("Processing Quality", "unknown")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frame=output_frames,
        time_s=times,
        frame_rate=np.asarray(1 / np.median(np.diff(times))),
        qpos=qpos,
        tracking_qpos=tracking_qpos,
        joint_angles_rad=qpos[:, model.jnt_qposadr[1:]],
        qvel=qvel,
        qacc=qacc,
        root_grounding_offset_m=grounding_offset,
        orientation_residual_deg=angle_residual,
        segment_position_residual_m=position_residual,
        segment_positions_xsens_m=segment_positions[model_frame_indices],
        valid=output_valid,
        derivative_valid=derivative_valid,
        segment_names=np.asarray(segment_names),
        segment_target_quat_wxyz=Rotation.from_matrix(
            targets[model_frame_indices].reshape(-1, 3, 3)
        )
        .as_quat()[:, [3, 0, 1, 2]]
        .reshape(-1, 7, 4),
        reset_event=event[model_frame_indices],
        solver_success=solver_success,
        solver_nfev=solver_nfev,
        joint_names=np.asarray([model.joint(j).name for j in range(1, model.njnt)]),
        joint_qpos_addresses=model.jnt_qposadr[1:].copy(),
        joint_dof_addresses=model.jnt_dofadr[1:].copy(),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "source_xlsx": str(input_path.resolve()),
                    "source_xlsx_sha256": source_xlsx_sha256,
                    "source_mvn": str(input_path.with_suffix(".mvn").resolve()),
                    "source_metadata": {key: str(value) for key, value in metadata.items()},
                    "model": str(model_path.resolve()),
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    "reference_frame": reference_frame,
                    "world_heading_alignment_rad": heading,
                    "position_resets_corrected_and_common_orientation_resets_removed": reset_events,
                    "unobserved_torso_joints_fixed_at_standing": [
                        "torso_pitch",
                        "torso_roll",
                        "torso_yaw",
                    ],
                    "method": "bounded SO(3) segment-orientation fit; reference frame mapped to neutral as an explicit provisional calibration",
                    "root_position": "Xsens pelvis position with large jumps recentered; initial model keyframe offset retained",
                    "root_mode": root_mode,
                    "grounding": "shift pelvis z to put the lowest approximate sole at -1 mm in each frame; a walking-contact assumption, not a measured root correction",
                    "torque_status": "not computed; contact/ground reaction must be estimated before physical torque tracking",
                    "residual_summary": residual_summary,
                },
                ensure_ascii=False,
            )
        ),
    )
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(
            {
                "source_xlsx": str(input_path.resolve()),
                "source_xlsx_sha256": source_xlsx_sha256,
                "source_mvn": str(input_path.with_suffix(".mvn").resolve()),
                "frames": len(output_frames),
                "source_frames": len(frames),
                "source_rate_hz": source_rate,
                "output_rate_hz": float(1 / np.median(np.diff(times))),
                "duration_s": float(times[-1]),
                "reference_frame": reference_frame,
                "world_heading_alignment_rad": heading,
                "root_mode": root_mode,
                "reset_events": reset_events,
                **residual_summary,
                "note": "Kinematic fit with provisional reference-pose calibration and model-based grounding. Torso is unobserved; force/contact parameters are not experimentally validated.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return report_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rate", type=float, default=60.0, help="Fitted output rate in Hz")
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=0,
        help="Xsens frame mapped to the model standing pose",
    )
    parser.add_argument("--max-nfev", type=int, default=30)
    parser.set_defaults(no_align_heading=True)
    parser.add_argument(
        "--align-heading",
        dest="no_align_heading",
        action="store_false",
        help="Experimental alignment from Xsens horizontal travel",
    )
    parser.add_argument(
        "--no-align-heading",
        action="store_true",
        help="Keep the Xsens world heading instead of aligning travel to +X",
    )
    parser.add_argument(
        "--root-mode",
        choices=("in_place", "follow_xsens"),
        default="in_place",
        help="Keep treadmill walking in place (default) or follow Xsens pelvis XY",
    )
    args = parser.parse_args()
    if not np.isfinite(args.rate) or args.rate <= 0 or args.max_nfev < 1:
        parser.error("--rate must be positive and --max-nfev must be at least one")
    report = fit_motion(
        args.input,
        args.model,
        args.output,
        args.rate,
        args.reference_frame,
        args.max_nfev,
        not args.no_align_heading,
        args.root_mode,
    )
    print(f"Fit saved to {args.output}")
    print(f"Diagnostics saved to {report}")


if __name__ == "__main__":
    main()
