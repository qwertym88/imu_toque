#!/usr/bin/env python3
"""Torque-drive the human along an Xsens fit; log any virtual pelvis assistance.

The assisted mode is a dynamics tracking diagnostic, not autonomous balance or
experimental inverse dynamics. Human qpos is set once at initialization; every
subsequent human pose is integrated by mj_step under torques and contact forces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]


def make_environment(model_path, timestep, belt_speed):
    root = ET.parse(model_path).getroot()
    world = root.find("worldbody")
    floor = next(geom for geom in world.findall("geom") if geom.get("name") == "floor")
    world.remove(floor)
    keys = root.find("keyframe")
    if keys is not None:
        root.remove(keys)
    for geom in root.iter("geom"):
        if geom.get("name", "").endswith("_sole"):
            geom.set("solref", "0.08 1")
            geom.set("solimp", "0.5 0.9 0.002")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Only foot/ground contact is modeled here. Approximate sole boxes must not
    # act as an uncalibrated obstacle for the other foot during cross-over.
    ET.SubElement(contact, "exclude", body1="left_foot", body2="right_foot")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Only foot/ground contact is modeled here. Approximate sole boxes must not
    # act as an uncalibrated obstacle for the other foot during cross-over.
    ET.SubElement(contact, "exclude", body1="left_foot", body2="right_foot")
    # The conveyor has a separate externally maintained slide coordinate.
    # Its motion is environmental actuation; no human coordinate is prescribed.
    belt = ET.SubElement(world, "body", name="treadmill", pos="0 0 -0.1", gravcomp="1")
    ET.SubElement(
        belt, "inertial", mass="1000000", pos="0 0 0", diaginertia="1000000 1000000 1000000"
    )
    ET.SubElement(belt, "joint", name="treadmill_x", type="slide", axis="1 0 0", limited="false")
    ET.SubElement(
        belt,
        "geom",
        name="treadmill_surface",
        type="box",
        size="500 10 0.1",
        contype="1",
        conaffinity="1",
        condim="3",
        friction="0.8 0.005 0.0001",
        rgba="0.5 0.55 0.6 1",
        group="0",
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.opt.timestep = timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    belt_joint = model.joint("treadmill_x").id
    return model, int(model.jnt_qposadr[belt_joint]), int(model.jnt_dofadr[belt_joint])


def contact_wrenches(model, data):
    """World [F,M] exerted on each foot, moment about its sole-center site."""
    wrenches = np.zeros((2, 6))
    counts = np.zeros(2, dtype=int)
    for index in range(data.ncon):
        contact = data.contact[index]
        for side_index, side in enumerate(("left", "right")):
            foot = model.geom(f"{side}_sole").id
            if foot not in (contact.geom1, contact.geom2):
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, local)
            sign = 1.0 if contact.geom2 == foot else -1.0
            frame = contact.frame.reshape(3, 3)
            force = sign * frame.T @ local[:3]
            moment = sign * frame.T @ local[3:]
            site = data.site_xpos[model.site(f"{side}_foot").id]
            moment += np.cross(contact.pos - site, force)
            wrenches[side_index] += np.r_[force, moment]
            counts[side_index] += 1
    return wrenches, counts


def floating_base_command(model, data, desired_acceleration):
    """Project acceleration tracking onto zero unactuated inverse force.

    Contact forces remain those of MuJoCo's soft-contact inverse model. This is
    a local inverse-dynamics controller, not a proof of balance feasibility.
    """
    desired = desired_acceleration.copy()
    estimate = desired.copy()
    human_nv = model.nv - 1
    inverse_weight = 1 / np.r_[2, 2, 4, 10, 10, 5, np.ones(human_nv - 6)]
    derivative = np.zeros((model.nv, model.nv))
    for _ in range(4):
        data.qacc[:] = estimate
        mujoco.mj_inverse(model, data)
        force = data.qfrc_inverse[:6].copy()
        if np.linalg.norm(force) < 1e-5:
            break
        mujoco.mjd_inverseFD(
            model, data, 1e-6, False, None, None, derivative, None, None, None, None
        )
        jacobian = derivative.T[:6, :human_nv]
        # Closest point to the original tracking command on this linearization.
        normal = (jacobian * inverse_weight) @ jacobian.T
        rhs = jacobian @ (desired[:human_nv] - estimate[:human_nv]) + force
        correction = inverse_weight * (jacobian.T @ np.linalg.solve(normal + 1e-8 * np.eye(6), rhs))
        candidate = desired[:human_nv] - correction
        estimate[:human_nv] = candidate
    data.qacc[:] = estimate
    mujoco.mj_inverse(model, data)
    return data.qfrc_inverse.copy()


def simulate(args):
    motion = np.load(args.motion, allow_pickle=False)
    times = motion["time_s"]
    poses = motion["tracking_qpos"] if "tracking_qpos" in motion.files else motion["qpos"]
    valid = motion["valid"]
    if args.start < times[0] or args.start + args.duration > times[-1]:
        raise ValueError("Requested simulation interval lies outside the fitted recording")
    selected = (times >= args.start - 0.1) & (times <= args.start + args.duration + 0.1)
    if not np.all(valid[selected]):
        raise ValueError(
            "Requested interval crosses invalid/reset frames; choose a contiguous valid interval"
        )
    # Interpolate only this contiguous run: cubic splines must not bridge resets.
    left = max(0, int(np.searchsorted(times, args.start)) - 3)
    right = min(len(times), int(np.searchsorted(times, args.start + args.duration)) + 4)
    while left > 0 and valid[left - 1] and times[left] > args.start - 0.5:
        left -= 1
    while (
        right < len(times) and valid[right] and times[right - 1] < args.start + args.duration + 0.5
    ):
        right += 1
    ts, qs = times[left:right], poses[left:right]
    joints = CubicSpline(ts, qs[:, 7:], axis=0)
    translation = CubicSpline(ts, qs[:, :3], axis=0)
    orientation = Slerp(ts, Rotation.from_quat(qs[:, [4, 5, 6, 3]]))
    human = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if poses.shape[1] != human.nq or human.nv != 21 or human.nq != 22:
        raise ValueError("This tracker requires the 22-qpos / 21-velocity human model")
    # Verify the fit and the simulation use exactly the same adapted human.
    meta = json.loads(str(motion["metadata_json"]))
    if "model_sha256" in meta:
        actual_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if actual_hash != meta["model_sha256"]:
            raise ValueError("Model differs from the one used for the fit; refit after scaling")
    model, belt_qadr, belt_dadr = make_environment(args.model, args.timestep, args.belt_speed)
    model.opt.iterations = 150
    model.opt.tolerance = 1e-10
    data = mujoco.MjData(model)
    probe = mujoco.MjData(model)
    nq, nv = human.nq, human.nv
    dofs = np.array([model.jnt_dofadr[int(j)] for j in model.actuator_trnid[:, 0]])
    kp = np.r_[
        np.full(3, args.base_kp), np.full(3, args.base_kp), np.full(nv - 6, args.joint_kp), 0.0
    ]
    kd = 2 * np.sqrt(kp)
    error = np.zeros(model.nv)

    def reference(time):
        pose = data.qpos.copy()
        pose[:3] = translation(time)
        pose[3:7] = orientation(time).as_quat()[[3, 0, 1, 2]]
        pose[7:nq] = joints(time)
        velocity = np.zeros(model.nv)
        acceleration = np.zeros(model.nv)
        velocity[:3], acceleration[:3] = translation(time, 1), translation(time, 2)
        velocity[6:nv], acceleration[6:nv] = joints(time, 1), joints(time, 2)
        epsilon = min(0.001, (ts[-1] - ts[0]) / 1000)
        a, b = max(ts[0], time - epsilon), min(ts[-1], time + epsilon)
        ra, rb = orientation(a), orientation(b)
        velocity[3:6] = (ra.inv() * rb).as_rotvec() / (b - a)
        # Slerp has piecewise angular velocity; do not differentiate its knots
        # into artificial angular impulses. Orientation/rate feedback remains.
        acceleration[3:6] = 0.0
        velocity[belt_dadr] = -args.belt_speed
        return pose, velocity, acceleration

    target, velocity, _ = reference(args.start)
    data.qpos[:] = target  # Initialization only; no human pose writes inside the loop.
    data.qvel[:] = velocity
    mujoco.mj_forward(model, data)
    rows = {
        key: []
        for key in (
            "time_s",
            "source_time_s",
            "qpos",
            "qvel",
            "qacc",
            "reference_qpos",
            "joint_error_rad",
            "torque_Nm",
            "pelvis_assistance_generalized",
            "pelvis_assistance_world",
            "contact_wrench_world",
            "contact_count",
            "foot_position_m",
            "dynamics_residual",
            "torque_saturated",
            "other_constraint_force",
        )
    }
    failure = None
    iterations = int(round(args.duration / args.timestep))
    sample_every = max(1, int(round(1 / (args.log_rate * args.timestep))))
    viewer = None
    if getattr(args, "viewer", False):
        import mujoco.viewer as viewer_module

        viewer = viewer_module.launch_passive(model, data)
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 115, -12
    wall_start = time.monotonic()
    try:
        for step in range(iterations):
            source_time = args.start + step * args.timestep
            target, velocity, acceleration = reference(source_time)
            data.qvel[belt_dadr] = -args.belt_speed
            # Keep the finite conveyor box centered while preserving its contact velocity.
            data.qpos[belt_qadr] = -args.belt_speed * (step * args.timestep)
            mujoco.mj_forward(model, data)
            mujoco.mj_differentiatePos(model, error, 1.0, data.qpos, target)
            probe.qpos[:] = data.qpos
            probe.qvel[:] = data.qvel
            probe.qacc[:] = acceleration + kp * error + kd * (velocity - data.qvel)
            probe.qacc[belt_dadr] = 0.0
            # Inverse dynamics computes soft-contact forces compatible with desired
            # accelerations. The six unactuated components are explicit assistance.
            if args.assistance == "none":
                required = floating_base_command(model, probe, probe.qacc.copy())
            else:
                mujoco.mj_inverse(model, probe)
                required = probe.qfrc_inverse.copy()
            command = required[dofs]
            data.ctrl[:] = np.clip(command, -args.torque_limit, args.torque_limit)
            data.qfrc_applied[:] = 0.0
            if args.assistance == "pelvis":
                data.qfrc_applied[:6] = required[:6]
            mujoco.mj_step(model, data)
            if viewer is not None and step % 16 == 0:
                if not viewer.is_running():
                    failure = f"viewer closed at {data.time:.3f} s"
                    break
                viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
                viewer.sync()
                delay = data.time - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(min(delay, 0.05))
            if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel[:nv])) > 100:
                failure = f"nonfinite state or velocity limit at {data.time:.3f} s"
                break
            if data.qpos[2] < 0.55 or data.qpos[2] > 1.8:
                failure = f"pelvis height outside [0.55, 1.8] m at {data.time:.3f} s"
                break
            if step % sample_every == 0:
                mujoco.mj_forward(model, data)
                wrench, counts = contact_wrenches(model, data)
                contact_generalized = np.zeros(model.nv)
                foot_positions = []
                for side_index, side in enumerate(("left", "right")):
                    site = model.site(f"{side}_foot").id
                    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, jp, jr, site)
                    contact_generalized += np.vstack((jp, jr)).T @ wrench[side_index]
                    foot_positions.append(data.site_xpos[site].copy())
                mass = np.empty((model.nv, model.nv))
                mujoco.mj_fullM(model, mass, data.qM)
                residual = (
                    mass @ data.qacc
                    + data.qfrc_bias
                    - data.qfrc_passive
                    - data.qfrc_actuator
                    - data.qfrc_applied
                    - data.qfrc_constraint
                )
                assist = data.qfrc_applied[:6].copy()
                root_rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
                assist_world = np.r_[assist[:3], root_rotation @ assist[3:6]]
                rows["time_s"].append(float(data.time))
                rows["source_time_s"].append(float(args.start + data.time))
                for key, value in (
                    ("qpos", data.qpos[:nq]),
                    ("qvel", data.qvel[:nv]),
                    ("qacc", data.qacc[:nv]),
                    ("reference_qpos", target[:nq]),
                    ("joint_error_rad", error[dofs]),
                    ("torque_Nm", data.ctrl),
                    ("pelvis_assistance_generalized", assist),
                    ("pelvis_assistance_world", assist_world),
                    ("contact_wrench_world", wrench),
                    ("contact_count", counts),
                    ("foot_position_m", foot_positions),
                    ("dynamics_residual", residual[:nv]),
                    ("torque_saturated", np.abs(command) > args.torque_limit),
                    ("other_constraint_force", (data.qfrc_constraint - contact_generalized)[:nv]),
                ):
                    rows[key].append(np.array(value, copy=True))
    finally:
        if viewer is not None:
            viewer.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.asarray(values) for key, values in rows.items()}
    settings = {
        "model": str(args.model.resolve()),
        "motion": str(args.motion.resolve()),
        "mode": f"{args.assistance}_assistance",
        "belt_speed_m_s": args.belt_speed,
        "timestep_s": args.timestep,
        "start_s": args.start,
        "duration_requested_s": args.duration,
        "torque_limit_Nm": args.torque_limit,
        "mass_kg": float(human.body_mass.sum()),
        "mass_is_subject_measurement": (
            json.loads(args.model.with_name("model_scaling.json").read_text()).get("mass_source")
            == "user_supplied"
            if args.model.with_name("model_scaling.json").exists()
            else False
        ),
        "human_pose_updates": "initialized once; all subsequent states from mj_step",
        "foot_to_foot_collision": "excluded; simplified contact boxes",
        "contact_model": "moving massive conveyor, friction 0.8, solref 0.08 1 and solimp 0.5 0.9 0.002; uncalibrated soft contacts",
        "torque_meaning": "simulation controller torques, not experimentally validated subject joint torques",
        "assistance_meaning": (
            "explicit external pelvis force/moment; not autonomous balancing"
            if args.assistance == "pelvis"
            else "zero externally applied pelvis assistance"
        ),
    }
    np.savez_compressed(
        args.output,
        **arrays,
        joint_names=np.asarray(
            [model.actuator(a).name.removesuffix("_torque") for a in range(model.nu)]
        ),
        settings_json=np.asarray(json.dumps(settings)),
    )
    summary = {
        **settings,
        "completed": failure is None,
        "failure": failure,
        "simulated_duration_s": float(data.time),
        "log_samples": len(rows["time_s"]),
        "pelvis_height_min_max_m": np.percentile(arrays["qpos"][:, 2], [0, 100]).tolist(),
        "joint_error_abs_deg_median_p95_max": np.percentile(
            np.abs(np.rad2deg(arrays["joint_error_rad"])), [50, 95, 100]
        ).tolist(),
        "torque_abs_Nm_p95_max": np.percentile(np.abs(arrays["torque_Nm"]), [95, 100]).tolist(),
        "pelvis_force_norm_N_p95_max": np.percentile(
            np.linalg.norm(arrays["pelvis_assistance_world"][:, :3], axis=1), [95, 100]
        ).tolist(),
        "contact_fraction_left_right": np.mean(arrays["contact_count"] > 0, axis=0).tolist(),
        "loaded_contact_fraction_left_right": np.mean(
            arrays["contact_wrench_world"][:, :, 2] > 20.0, axis=0
        ).tolist(),
        "pelvis_assist_force_p95_bodyweights": float(
            np.percentile(np.linalg.norm(arrays["pelvis_assistance_world"][:, :3], axis=1), 95)
            / (human.body_mass.sum() * 9.81)
        ),
        "autonomous_balance_validated": False,
        "torque_saturation_fraction": float(np.mean(arrays["torque_saturated"])),
        "max_abs_dynamics_residual": float(np.max(np.abs(arrays["dynamics_residual"]))),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/y13/human.xml")
    parser.add_argument("--motion", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13/walk_assisted.npz")
    parser.add_argument("--start", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--belt-speed",
        type=float,
        default=3.8 / 3.6,
        help="Assumed treadmill belt speed in m/s; zero for stationary floor",
    )
    parser.add_argument("--assistance", choices=("pelvis", "none"), default="pelvis")
    parser.add_argument(
        "--torque-limit",
        type=float,
        default=1500.0,
        help="Numerical actuator limit, not a physiological strength measurement",
    )
    parser.add_argument("--joint-kp", type=float, default=400.0)
    parser.add_argument("--base-kp", type=float, default=120.0)
    parser.add_argument("--timestep", type=float, default=0.001)
    parser.add_argument("--log-rate", type=float, default=100.0)
    parser.add_argument(
        "--viewer", action="store_true", help="Show the live torque-driven simulation"
    )
    args = parser.parse_args()
    if (
        not np.isfinite(args.start)
        or args.start < 0
        or not np.isfinite(args.belt_speed)
        or args.belt_speed < 0
    ):
        parser.error("--start and --belt-speed must be finite and nonnegative")
    if not all(
        np.isfinite(v) and v > 0
        for v in (
            args.duration,
            args.torque_limit,
            args.joint_kp,
            args.base_kp,
            args.timestep,
            args.log_rate,
        )
    ):
        parser.error("Duration, gains, limit, timestep and log rate must be finite and positive")
    report = simulate(args)
    if not report["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
