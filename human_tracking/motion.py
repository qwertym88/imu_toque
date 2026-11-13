"""Convert an audited Xsens fit to an explicitly ordered mjlab reference, offline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIRECTORY = ROOT / "outputs/y13/mjlab"
TRACKED_BODIES = (
    "pelvis",
    "right_thigh",
    "right_shank",
    "right_foot",
    "left_thigh",
    "left_shank",
    "left_foot",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interpolate_poses(times, poses, target_times):
    """Linear joint/position interpolation and shortest-arc quaternion SLERP (wxyz)."""
    result = np.column_stack([np.interp(target_times, times, p) for p in poses.T])
    indices = np.clip(np.searchsorted(times, target_times, side="right") - 1, 0, len(times) - 2)
    blend = ((target_times - times[indices]) / (times[indices + 1] - times[indices]))[:, None]
    a, b = poses[indices, 3:7].copy(), poses[indices + 1, 3:7].copy()
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.sum(a * b, axis=1, keepdims=True)
    b *= np.where(dot < 0, -1, 1)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    denominator = np.maximum(np.sin(angle), 1e-12)
    spherical = (np.sin((1 - blend) * angle) * a + np.sin(blend * angle) * b) / denominator
    linear = (1 - blend) * a + blend * b
    quat = np.where(angle < 1e-6, linear, spherical)
    result[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    return result


def prepare_reference(
    model_path=ROOT / "outputs/y13/human.xml",
    fit_path=ROOT / "outputs/y13/fit.npz",
    output_dir=DEFAULT_DIRECTORY,
    start=25.0,
    duration=10.0,
    fps=50.0,
    speed=3.8 / 3.6,
):
    model_path, fit_path, output_dir = map(Path, (model_path, fit_path, output_dir))
    if (output_dir / "human.xml").resolve() == model_path.resolve() or (
        output_dir / "motion.npz"
    ).resolve() == fit_path.resolve():
        raise ValueError("Output directory must not overwrite the input model or fit")
    if not all(np.isfinite(x) for x in (start, duration, fps, speed)):
        raise ValueError("Motion conversion parameters must be finite")
    if start < 0 or duration <= 0 or fps <= 0 or speed < 0 or duration * fps < 3:
        raise ValueError("Require start/speed >= 0 and a positive clip of at least four frames")
    source = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    if (source.nq, source.nv) != (22, 21):
        raise ValueError("This baseline requires the 15-hinge MS-Human model with a free pelvis")
    with np.load(fit_path, allow_pickle=False) as fit:
        times = fit["time_s"]
        poses = fit["tracking_qpos"] if "tracking_qpos" in fit.files else fit["qpos"]
        metadata = json.loads(str(fit["metadata_json"]))
        if metadata["model_sha256"] != sha256(model_path):
            raise ValueError("Fit/model SHA-256 mismatch; regenerate the fit for this model")
        if poses.shape != (len(times), source.nq) or not np.all(np.diff(times) > 0):
            raise ValueError("Malformed reference shape or timestamps")
        if start < times[0] or start + duration > times[-1]:
            raise ValueError("Requested clip lies outside the recording")
        left = max(0, int(np.searchsorted(times, start)) - 1)
        right = min(len(times), int(np.searchsorted(times, start + duration)) + 2)
        valid = fit["valid"].astype(bool)
        if "derivative_valid" in fit.files:
            valid &= fit["derivative_valid"].astype(bool)
        if not valid[left:right].all() or not np.isfinite(poses[left:right]).all():
            raise ValueError("Clip or interpolation boundary crosses invalid/reset frames")
        if np.any(np.linalg.norm(poses[left:right, 3:7], axis=1) < 1e-8):
            raise ValueError("Invalid root quaternion in the reference")
        names = [source.joint(j).name for j in range(1, source.njnt)]
        if names != fit["joint_names"].tolist():
            raise ValueError("Fit joint ordering does not match the human model")
        elapsed = np.arange(int(np.floor(duration * fps)) + 1) / fps
        qpos = interpolate_poses(times[left:right], poses[left:right], start + elapsed)
    # Galilean frame change: stationary treadmill reference -> forward walking
    # on stationary ground. The speed is an experimental assumption, not measured.
    qpos[:, :2] -= qpos[0, :2]
    qpos[:, 0] += speed * elapsed

    root = ET.parse(model_path).getroot()
    for tag in ("actuator", "keyframe"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    world = root.find("worldbody")
    for element in list(world):
        if element.tag != "body":
            world.remove(element)
    # A fixed-ground environment is supplied by mjlab, not included in the entity.
    for geom in root.iter("geom"):
        if geom.get("name", "").endswith("_sole"):
            geom.set("contype", "0")
            geom.set("conaffinity", "1")
            geom.set("solref", "0.02 1")
            geom.set("friction", "0.8 0.005 0.0001")
    # These are training bounds for the unobserved torso, not measured anatomy.
    torso_ranges = {"torso_pitch": "-0.5 0.5", "torso_roll": "-0.35 0.35", "torso_yaw": "-0.5 0.5"}
    for joint in root.iter("joint"):
        if joint.get("name") in torso_ranges:
            joint.set("limited", "true")
            joint.set("range", torso_ranges[joint.get("name")])
    pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
    ET.SubElement(pelvis, "site", name="imu", size="0.005")
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framelinvel", name="imu_lin_vel", objtype="site", objname="imu")
    ET.SubElement(sensors, "gyro", name="imu_ang_vel", site="imu")
    output_dir.mkdir(parents=True, exist_ok=True)
    entity_path = output_dir / "human.xml"
    ET.indent(root, space="  ")
    entity_path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    model = mujoco.MjModel.from_xml_path(str(entity_path.resolve()))
    qvel = np.empty((len(qpos), model.nv))
    for i in range(len(qpos)):
        a, b = max(0, i - 1), min(len(qpos) - 1, i + 1)
        mujoco.mj_differentiatePos(model, qvel[i], elapsed[b] - elapsed[a], qpos[a], qpos[b])
    data = mujoco.MjData(model)
    positions, rotations, linear_velocities, angular_velocities = [], [], [], []
    # Entity-local body order excludes world. MjSpec fusestatic=false preserves it.
    body_names = [model.body(i).name for i in range(1, model.nbody)]
    for pose, velocity in zip(qpos, qvel):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(model, data)
        v = np.empty((model.nbody - 1, 6))
        for body_id in range(1, model.nbody):
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, v[body_id - 1], 0
            )
            # mj_objectVelocity(BODY) uses the inertial COM; shift to body origin.
            v[body_id - 1, 3:] += np.cross(
                v[body_id - 1, :3], data.xpos[body_id] - data.xipos[body_id]
            )
        positions.append(data.xpos[1:].copy())
        rotations.append(data.xquat[1:].copy())
        linear_velocities.append(v[:, 3:])
        angular_velocities.append(v[:, :3])
    report = {
        "format_version": 1,
        "model": str(entity_path.resolve()),
        "model_sha256": sha256(entity_path),
        "source_model": str(model_path.resolve()),
        "source_model_sha256": sha256(model_path),
        "source_fit": str(fit_path.resolve()),
        "source_fit_sha256": sha256(fit_path),
        "start_s": start,
        "duration_s": float(elapsed[-1]),
        "fps": fps,
        "frames": len(qpos),
        "body_names": body_names,
        "joint_names": names,
        "quaternion_order": "wxyz",
        "ground_frame": "stationary ground; root x += assumed belt speed * elapsed time",
        "assumed_belt_speed_m_s": speed,
        "pelvis_assistance": False,
        "torso": "unobserved neutral reference and explicit training bounds; not measured",
        "reference_pose": metadata.get("reference_pose", metadata.get("reference_frame")),
    }
    np.savez_compressed(
        output_dir / "motion.npz",
        fps=np.asarray([fps]),
        time_s=elapsed,
        qpos=qpos,
        qvel=qvel,
        joint_pos=qpos[:, model.jnt_qposadr[1:]],
        joint_vel=qvel[:, model.jnt_dofadr[1:]],
        body_pos_w=np.asarray(positions),
        body_quat_w=np.asarray(rotations),
        body_lin_vel_w=np.asarray(linear_velocities),
        body_ang_vel_w=np.asarray(angular_velocities),
        body_names=np.asarray(body_names),
        joint_names=np.asarray(names),
        metadata_json=np.asarray(json.dumps(report)),
    )
    (output_dir / "motion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Prepared {len(qpos)} frames at {fps:g} Hz in {output_dir}")
    return report
