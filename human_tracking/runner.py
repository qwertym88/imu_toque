"""Local training and honest, non-resetting policy evaluation for mjlab 1.2."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import signal
import time

import mujoco
import numpy as np
import torch

from . import TASK_ID
from .motion import DEFAULT_DIRECTORY, sha256
from .task import PD_GROUPS, fall_detected, make_env_cfg, make_runner_cfg


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in (
            "mjlab",
            "mujoco",
            "mujoco-warp",
            "warp-lang",
            "torch",
            "rsl-rl-lib",
        )
    }


def ensure_compatible():
    version = importlib.metadata.version("mjlab")
    if version != "1.2.0":
        raise RuntimeError(f"This baseline is verified against mjlab 1.2.0, found {version}")
    if mujoco.__version__ != "3.5.0":
        raise RuntimeError(
            f"Use MuJoCo 3.5.0 with mjlab 1.2.0, found {mujoco.__version__}; "
            "activate the project's .venv-mjlab or install requirements-mjlab.txt in a separate environment"
        )


def require_cuda():
    ensure_compatible()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "mjlab training requires an accessible NVIDIA GPU; run 'check' for CPU validation"
        )
    torch.set_num_threads(4)
    return "cuda:0"


def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def check_reference(directory):
    directory = Path(directory).resolve()
    with np.load(directory / "motion.npz", allow_pickle=False) as motion:
        report = json.loads(str(motion["metadata_json"]))
        if report["model_sha256"] != sha256(directory / "human.xml"):
            raise ValueError("Prepared reference/model SHA-256 mismatch")
        if Path(report["model"]).resolve() != directory / "human.xml":
            raise ValueError("Reference moved; rerun prepare to rebuild its model manifest")
        if not np.isclose(float(motion["fps"].item()), 50.0):
            raise ValueError("This baseline requires a 50 Hz reference")
        from mjlab.entity import Entity

        robot = Entity(make_env_cfg(directory, 1).scene.entities["robot"])
        compiled = robot.spec.compile()
        if list(robot.body_names) != motion["body_names"].tolist():
            raise ValueError("Compiled entity/reference body ordering mismatch")
        if list(robot.joint_names) != motion["joint_names"].tolist():
            raise ValueError("Compiled entity/reference joint ordering mismatch")
        if compiled.nq != 22 or compiled.nv != 21 or compiled.nu != 15:
            raise ValueError(
                "Unexpected model dimensions; requires free pelvis + 15 actuated hinges"
            )
        joints = compiled.actuator_trnid[:, 0]
        if (joints == 0).any() or len(set(joints)) != 15:
            raise ValueError("Each hinge must have one actuator and pelvis must have none")
        expected = np.arange(1, 16)
        if not np.array_equal(joints, expected):
            raise ValueError("PD actuator order does not match reference joints")
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            if not np.isfinite(motion[key]).all():
                raise ValueError(f"Nonfinite reference: {key}")
        if not np.allclose(np.linalg.norm(motion["body_quat_w"], axis=-1), 1.0, atol=1e-5):
            raise ValueError("Reference body quaternions are not normalized")
        return {
            "task": TASK_ID,
            "frames": len(motion["joint_pos"]),
            "fps": 50,
            "nq": compiled.nq,
            "nv": compiled.nv,
            "nu": compiled.nu,
            "mass_kg": float(compiled.body_mass.sum()),
            "body_order_verified": True,
            "joint_and_actuator_order_verified": True,
            "pelvis_actuators": 0,
            "reference_dir": str(directory),
            "versions": versions(),
        }


def check(args):
    ensure_compatible()
    result = check_reference(args.reference_dir)
    print(json.dumps(result, indent=2))
    return result


def no_external_forces(env):
    generalized = float(env.sim.data.qfrc_applied.abs().max())
    body = float(env.sim.data.xfrc_applied.abs().max())
    if generalized != 0 or body != 0:
        raise RuntimeError("Unexpected external generalized/body assistance force")
    return max(generalized, body)


def smoke(args):
    from mjlab.envs import ManagerBasedRlEnv

    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs and steps must be positive")
    device = require_cuda()
    result = check_reference(args.reference_dir)
    env = ManagerBasedRlEnv(make_env_cfg(args.reference_dir, args.num_envs), device=device)
    try:
        env.reset()
        reset_count = 0
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = 0.05 * torch.randn(args.num_envs, 15, device=device)
                obs, reward, terminated, truncated, _ = env.step(actions)
                if (
                    not all(torch.isfinite(value).all() for value in obs.values())
                    or not torch.isfinite(reward).all()
                ):
                    raise RuntimeError("Nonfinite smoke-test observation/reward")
                no_external_forces(env)
                reset_count += int((terminated | truncated).sum())
            cmd = env.command_manager.get_term("motion")
            cmd.time_steps[:] = cmd.motion.time_step_total - 2
            pose_before = env.sim.data.qpos.clone()
            cmd._update_command()
            if not torch.equal(env.sim.data.qpos, pose_before):
                raise RuntimeError("Reference command teleported human at clip boundary")
        result.update(
            {
                "steps": args.steps,
                "num_envs": args.num_envs,
                "training_resets": reset_count,
                "external_assistance_max": 0.0,
                "reference_end_does_not_teleport": True,
                "completed": True,
                "trained_policy": False,
            }
        )
        write_json(args.output, result)
        return result
    finally:
        env.close()


def snapshot_reference(directory, destination):
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(directory) / "human.xml", destination / "human.xml")
    with np.load(Path(directory) / "motion.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    metadata = json.loads(str(arrays["metadata_json"]))
    metadata["model"] = str((destination / "human.xml").resolve())
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(destination / "motion.npz", **arrays)
    write_json(destination / "motion.json", metadata)


def train(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    if args.num_envs < 4 or args.iterations < 1:
        raise ValueError("Training needs at least four environments and positive iterations")
    device = require_cuda()
    directory = args.reference_dir.resolve()
    check_reference(directory)
    run_dir = args.run_dir or DEFAULT_DIRECTORY / "runs" / datetime.now().astimezone().strftime(
        "%Y%m%d_%H%M%S"
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_reference(directory, run_dir / "reference")
    directory = run_dir / "reference"
    cfg = make_env_cfg(directory, args.num_envs)
    cfg.seed = args.seed
    agent = make_runner_cfg()
    agent.seed = args.seed
    agent.max_iterations = args.iterations
    agent_config = asdict(agent)
    manifest = {
        "task": TASK_ID,
        "reference_dir": str(directory),
        "model_sha256": sha256(directory / "human.xml"),
        "motion_sha256": sha256(directory / "motion.npz"),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "iterations_requested": args.iterations,
        "policy_dt": cfg.decimation * cfg.sim.mujoco.timestep,
        "physics_dt": cfg.sim.mujoco.timestep,
        "pd_groups": PD_GROUPS,
        "action": "q_reference + clipped residual; built-in PD, finite torque limits",
        "pelvis_assistance": False,
        "agent": agent_config,
        "versions": versions(),
        "resume_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
    }
    write_json(run_dir / "run.json", manifest)
    env = ManagerBasedRlEnv(cfg, device=device)
    previous_handlers = {}
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapper, agent_config, str(run_dir), device)
        if args.checkpoint:
            validate_checkpoint(args.checkpoint, directory)
            runner.load(str(args.checkpoint.resolve()), map_location=device)

        def interrupt_training(signum, frame):
            raise KeyboardInterrupt

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt_training)
        try:
            runner.learn(args.iterations, init_at_random_ep_len=True)
            manifest["status"] = "finished"
        except KeyboardInterrupt:
            runner.save(str(run_dir / f"model_{runner.current_learning_iteration}.pt"))
            manifest["status"] = "interrupted_checkpoint_saved"
        no_external_forces(env)
        manifest["last_iteration_index"] = runner.current_learning_iteration
        manifest["checkpoint"] = str(run_dir / f"model_{runner.current_learning_iteration}.pt")
        write_json(run_dir / "run.json", manifest)
        return manifest
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        env.close()


def validate_checkpoint(checkpoint, directory):
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads(checkpoint.with_name("run.json").read_text())
    with np.load(Path(directory) / "motion.npz", allow_pickle=False) as motion:
        if manifest["model_sha256"] != sha256(Path(directory) / "human.xml"):
            raise ValueError("Checkpoint was trained on a different human model")
        # Snapshot changes only metadata path, not any actual reference array.
        original = Path(manifest["reference_dir"]) / "motion.npz"
        if sha256(original) != manifest["motion_sha256"]:
            raise ValueError("Frozen training reference was modified after the run started")
        with np.load(original, allow_pickle=False) as trained:
            for key in ("qpos", "qvel", "body_pos_w", "body_quat_w"):
                if not np.array_equal(motion[key], trained[key]):
                    raise ValueError(
                        "Checkpoint reference differs from requested evaluation reference"
                    )
    if manifest["pd_groups"] != [list(group) for group in PD_GROUPS]:
        raise ValueError("PD gains/limits changed since this checkpoint was trained")
    cfg = make_env_cfg(directory, 1, play=True)
    if not np.isclose(manifest["physics_dt"], cfg.sim.mujoco.timestep) or not np.isclose(
        manifest["policy_dt"], cfg.sim.mujoco.timestep * cfg.decimation
    ):
        raise ValueError("Physics/control timestep changed since this checkpoint was trained")
    return manifest


def cpu_contact_wrenches(model, data):
    """Diagnostic CPU recomputation at the recorded GPU state, world [F,M] at sole sites."""
    wrenches = np.zeros((2, 6))
    for i in range(data.ncon):
        contact = data.contact[i]
        for index, side in enumerate(("left", "right")):
            geom = model.geom(f"robot/{side}_sole").id
            if geom not in (contact.geom1, contact.geom2):
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, local)
            sign = 1 if contact.geom2 == geom else -1
            force = sign * contact.frame.reshape(3, 3).T @ local[:3]
            moment = sign * contact.frame.reshape(3, 3).T @ local[3:]
            moment += np.cross(
                contact.pos - data.site_xpos[model.site(f"robot/{side}_foot").id], force
            )
            wrenches[index] += np.r_[force, moment]
    return wrenches


def export_runtime_model(scene, model, output):
    """Export attached MjSpec, including floor/PD, rather than global last-loaded XML."""
    from mjlab.utils.xml import fix_spec_xml, strip_buffer_textures

    spec = scene.spec.copy()
    for name in (
        "timestep",
        "integrator",
        "solver",
        "cone",
        "iterations",
        "ls_iterations",
        "tolerance",
        "ls_tolerance",
        "impratio",
        "jacobian",
    ):
        setattr(spec.option, name, getattr(model.opt, name))
    spec.option.gravity[:] = model.opt.gravity
    strip_buffer_textures(spec)
    output.write_text(fix_spec_xml(spec.to_xml()))
    saved = mujoco.MjModel.from_xml_path(str(output.resolve()))
    for name in ("nq", "nv", "nu", "nbody", "ngeom"):
        if getattr(saved, name) != getattr(model, name):
            raise RuntimeError(f"Exported scene differs from runtime model: {name}")
    if saved.names != model.names:
        raise RuntimeError("Exported scene element names differ from the runtime model")


def evaluate(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    device = require_cuda()
    directory = args.reference_dir
    if directory is None:
        directory = (
            args.checkpoint.resolve().parent / "reference" if args.checkpoint else DEFAULT_DIRECTORY
        )
    directory = directory.resolve()
    check_reference(directory)
    if (
        not np.isfinite(args.duration)
        or args.duration < 0.02
        or not np.isclose(args.duration * 50, round(args.duration * 50))
    ):
        raise ValueError(
            "Evaluation duration must be a positive multiple of the 0.02 s control step"
        )
    with np.load(directory / "motion.npz", allow_pickle=False) as motion:
        available = (len(motion["joint_pos"]) - 1) / 50.0
    if args.duration > available + 1e-6:
        raise ValueError("Evaluation cannot exceed clip length; prepare a longer valid reference")
    if args.checkpoint:
        validate_checkpoint(args.checkpoint, directory)
    cfg = make_env_cfg(directory, 1, play=True)
    cfg.terminations = {}  # Absolutely no automatic resets in reported rollouts.
    env = ManagerBasedRlEnv(cfg, device=device)
    viewer = None
    rows = {
        key: []
        for key in (
            "time_s",
            "qpos",
            "qvel",
            "torque_Nm",
            "reference_qpos",
            "joint_error_rad",
            "body_orientation_error_deg",
            "contact_wrench_world",
            "action",
            "cpu_gpu_torque_difference_Nm",
        )
    }
    failure = None
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=3.0)
        policy = None
        if args.checkpoint:
            runner = MjlabOnPolicyRunner(wrapper, asdict(make_runner_cfg()), None, device)
            runner.load(str(args.checkpoint.resolve()), map_location=device)
            policy = runner.get_inference_policy(device=device)
        model, data = env.sim.mj_model, mujoco.MjData(env.sim.mj_model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        model_path = args.output.with_suffix(".xml")
        export_runtime_model(env.scene, model, model_path)
        if args.viewer:
            import mujoco.viewer as viewer_module

            viewer = viewer_module.launch_passive(model, data)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 115, -12
        observation = wrapper.get_observations()
        wall_start = time.monotonic()
        with torch.inference_mode():
            for step in range(int(round(args.duration / env.step_dt))):
                actions = policy(observation) if policy else torch.zeros(1, 15, device=device)
                observation, _, dones, _ = wrapper.step(actions)
                if dones.any():
                    raise RuntimeError("Evaluation unexpectedly reset; continuity is not valid")
                no_external_forces(env)
                if (
                    not torch.isfinite(env.sim.data.qpos).all()
                    or not torch.isfinite(env.sim.data.qvel).all()
                ):
                    raise RuntimeError("Nonfinite GPU state during evaluation; rollout is invalid")
                data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
                data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
                data.ctrl[:] = env.sim.data.ctrl[0].cpu().numpy()
                mujoco.mj_forward(model, data)
                robot = env.scene["robot"]
                cmd = env.command_manager.get_term("motion")
                torque = robot.data.actuator_force[0].cpu().numpy().copy()
                reference = np.r_[
                    cmd.anchor_pos_w[0].cpu().numpy(),
                    cmd.anchor_quat_w[0].cpu().numpy(),
                    cmd.joint_pos[0].cpu().numpy(),
                ]
                from mjlab.utils.lab_api.math import quat_error_magnitude

                error = (
                    torch.rad2deg(quat_error_magnitude(cmd.robot_body_quat_w, cmd.body_quat_w))[0]
                    .cpu()
                    .numpy()
                )
                for key, value in (
                    ("time_s", (step + 1) * env.step_dt),
                    ("qpos", data.qpos.copy()),
                    ("qvel", data.qvel.copy()),
                    ("torque_Nm", torque),
                    ("reference_qpos", reference),
                    ("joint_error_rad", data.qpos[7:] - reference[7:]),
                    ("body_orientation_error_deg", error),
                    ("contact_wrench_world", cpu_contact_wrenches(model, data)),
                    ("action", actions[0].cpu().numpy().copy()),
                    ("cpu_gpu_torque_difference_Nm", data.actuator_force - torque),
                ):
                    rows[key].append(value)
                if bool(fall_detected(env)[0]):
                    failure = f"fall/invalid state at {(step + 1) * env.step_dt:.3f} s"
                    break
                if viewer:
                    if not viewer.is_running():
                        failure = "viewer closed by user"
                        break
                    viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
                    viewer.sync()
                    delay = (step + 1) * env.step_dt - (time.monotonic() - wall_start)
                    if delay > 0:
                        time.sleep(min(delay, 0.05))
        arrays = {key: np.asarray(value) for key, value in rows.items()}
        settings = {
            "mode": "rl_no_pelvis_assistance" if policy else "reference_pd_no_pelvis_assistance",
            "model": str(model_path.resolve()),
            "reference_dir": str(directory),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "pelvis_assistance": False,
            "automatic_resets": 0,
            "contact_meaning": "CPU MuJoCo recomputation at saved GPU qpos/qvel/ctrl; diagnostic, not measured force",
            "torque_meaning": "actual simulated GPU actuator_force, not policy actions or measured human torque",
            "versions": versions(),
        }
        np.savez_compressed(args.output, **arrays, settings_json=np.asarray(json.dumps(settings)))
        summary = {
            **settings,
            "completed": failure is None,
            "failure": failure,
            "duration_requested_s": args.duration,
            "simulated_duration_s": float(arrays["time_s"][-1]),
            "joint_error_abs_deg_p95": float(
                np.percentile(np.abs(np.rad2deg(arrays["joint_error_rad"])), 95)
            ),
            "root_position_error_m_median_p95_max": np.percentile(
                np.linalg.norm(arrays["qpos"][:, :3] - arrays["reference_qpos"][:, :3], axis=1),
                [50, 95, 100],
            ).tolist(),
            "root_displacement_m": (arrays["qpos"][-1, :3] - arrays["qpos"][0, :3]).tolist(),
            "body_orientation_error_deg_p95": float(
                np.percentile(arrays["body_orientation_error_deg"], 95)
            ),
            "torque_abs_Nm_p95_max": np.percentile(np.abs(arrays["torque_Nm"]), [95, 100]).tolist(),
            "max_abs_cpu_gpu_torque_difference_Nm": float(
                np.abs(arrays["cpu_gpu_torque_difference_Nm"]).max()
            ),
            "external_assistance_max": 0.0,
            "continuous_rollout": True,
        }
        write_json(args.output.with_suffix(".json"), summary)
        return summary
    finally:
        if viewer:
            viewer.close()
        env.close()
