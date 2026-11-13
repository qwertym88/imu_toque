"""MS-Human residual-PD tracking on mjlab 1.2.0, without pelvis actuation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .motion import TRACKED_BODIES, sha256

# Diagnostic bounds and gains, not subject-specific strength measurements.
PD_GROUPS = (
    (r".*_hip_.*", 300.0, 20.0, 200.0),
    (r".*_knee_pitch", 350.0, 20.0, 200.0),
    (r".*_ankle_.*", 120.0, 8.0, 120.0),
    (r"torso_.*", 200.0, 15.0, 150.0),
)


def get_human_cfg(model_path):
    path = str(Path(model_path).resolve())
    template = mujoco.MjModel.from_xml_path(path)
    return EntityCfg(
        spec_fn=lambda: mujoco.MjSpec.from_file(path),
        init_state=EntityCfg.InitialStateCfg(
            pos=tuple(template.qpos0[:3]),
            rot=tuple(template.qpos0[3:7]),
            joint_pos={".*": 0.0},
            joint_vel={".*": 0.0},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=tuple(
                BuiltinPositionActuatorCfg(
                    target_names_expr=(pattern,),
                    stiffness=kp,
                    damping=kd,
                    effort_limit=limit,
                    armature=0.0,
                )
                for pattern, kp, kd, limit in PD_GROUPS
            ),
            soft_joint_pos_limit_factor=1.0,
        ),
        sort_actuators=True,
    )


class ClipMotionCommand(MotionCommand):
    """Freeze at clip end; only environment reset is allowed to write human state."""

    def __init__(self, cfg, env):
        with np.load(cfg.motion_file, allow_pickle=False) as motion:
            fps = float(motion["fps"].item())
            if not np.isclose(fps * env.step_dt, 1.0):
                raise ValueError("Reference fps must equal the policy control frequency")
            entity = env.scene[cfg.entity_name]
            if motion["body_names"].tolist() != list(entity.body_names):
                raise ValueError("Reference body ordering differs from the mjlab human entity")
            if motion["joint_names"].tolist() != list(entity.joint_names):
                raise ValueError("Reference joint ordering differs from the mjlab human entity")
            report = json.loads(str(motion["metadata_json"]))
            if report["model_sha256"] != sha256(report["model"]):
                raise ValueError("Reference model changed since conversion; regenerate motion")
        super().__init__(cfg, env)

    def _update_command(self):
        # Upstream automatically resamples/teleports at clip end, even in play.
        # Clamp instead and let an explicit timeout end the training episode.
        self.time_steps.clamp_(max=self.motion.time_step_total - 2)
        super()._update_command()


@dataclass(kw_only=True)
class ClipMotionCommandCfg(MotionCommandCfg):
    def build(self, env):
        return ClipMotionCommand(self, env)


class ReferenceResidualAction(JointPositionAction):
    def process_actions(self, actions):
        super().process_actions(actions.clamp(-3.0, 3.0))
        command = self._env.command_manager.get_term("motion")
        target = command.joint_pos[:, self._target_ids] + self._processed_actions
        limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
        self._processed_actions = torch.clamp(target, limits[..., 0], limits[..., 1])


@dataclass(kw_only=True)
class ReferenceResidualActionCfg(JointPositionActionCfg):
    def build(self, env):
        return ReferenceResidualAction(self, env)


def joint_tracking(env, std=0.4):
    cmd = env.command_manager.get_term("motion")
    return torch.exp(
        -((cmd.robot_joint_pos[:, :12] - cmd.joint_pos[:, :12]) ** 2).mean(-1) / std**2
    )


def torque_cost(env):
    return (env.scene["robot"].data.actuator_force / 200.0).square().mean(-1)


def power_cost(env):
    robot = env.scene["robot"]
    return (robot.data.actuator_force * robot.data.joint_vel).abs().sum(-1) / 1000.0


def torso_cost(env):
    return env.scene["robot"].data.joint_pos[:, 12:].square().sum(-1)


def foot_slip_cost(env):
    robot = env.scene["robot"]
    ids = robot.find_bodies(("left_foot", "right_foot"), preserve_order=True)[0]
    # Contact proxy: link origin near the floor. No measured contact labels exist.
    near_floor = robot.data.body_link_pos_w[:, ids, 2] < 0.08
    speed2 = robot.data.body_link_lin_vel_w[:, ids, :2].square().sum(-1)
    return (near_floor * speed2).sum(-1)


def reference_end(env):
    cmd = env.command_manager.get_term("motion")
    return cmd.time_steps >= cmd.motion.time_step_total - 2


def fall_detected(env):
    cmd = env.command_manager.get_term("motion")
    height = cmd.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
    # The anatomical pelvis frame is rotated 90 deg relative to world. Compare
    # gravity in the actual/reference frames; don't assume pelvis local z is up.
    gravity = env.scene["robot"].data.gravity_vec_w
    actual = quat_apply_inverse(cmd.robot_anchor_quat_w, gravity)
    desired = quat_apply_inverse(cmd.anchor_quat_w, gravity)
    tilt_cosine = (actual * desired).sum(-1).clamp(-1.0, 1.0)
    finite = torch.isfinite(env.scene["robot"].data.joint_pos).all(-1)
    return (height < 0.55) | (height > 1.8) | (tilt_cosine < 0.5) | ~finite


def make_env_cfg(directory, num_envs=1024, play=False):
    directory = Path(directory).resolve()
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_human_cfg(directory / "human.xml")}
    cfg.scene.num_envs = num_envs
    cfg.scene.sensors = ()
    cfg.events = {}  # No pushes, external assistance or randomization in baseline.
    cfg.observations["actor"].enable_corruption = not play
    cfg.actions = {
        "joint_pos": ReferenceResidualActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale={r".*_(hip|knee|ankle)_.*": 0.35, r"torso_.*": 0.15},
            use_default_offset=False,
        )
    }
    cfg.commands = {
        "motion": ClipMotionCommandCfg(
            entity_name="robot",
            motion_file=str(directory / "motion.npz"),
            anchor_body_name="pelvis",
            body_names=TRACKED_BODIES,
            resampling_time_range=(1e9, 1e9),
            debug_vis=False,
            sampling_mode="start" if play else "adaptive",
            joint_position_range=(0.0, 0.0) if play else (-0.03, 0.03),
            pose_range={} if play else {"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
            velocity_range={} if play else {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        )
    }
    cfg.rewards.pop("self_collisions")
    cfg.rewards["motion_body_ori"].weight = 2.0
    cfg.rewards["motion_body_pos"].weight = 1.0
    cfg.rewards["motion_body_lin_vel"].weight = 0.5
    cfg.rewards["motion_body_ang_vel"].weight = 0.5
    cfg.rewards.update(
        {
            "joint_tracking": RewardTermCfg(func=joint_tracking, weight=1.0),
            "torque": RewardTermCfg(func=torque_cost, weight=-0.1),
            "power": RewardTermCfg(func=power_cost, weight=-0.02),
            "torso_prior": RewardTermCfg(func=torso_cost, weight=-0.5),
            "foot_slip": RewardTermCfg(func=foot_slip_cost, weight=-0.2),
        }
    )
    cfg.terminations.pop("anchor_ori")
    cfg.terminations.pop("ee_body_pos")
    cfg.terminations["fall"] = TerminationTermCfg(func=fall_detected)
    cfg.terminations["reference_end"] = TerminationTermCfg(func=reference_end, time_out=True)
    cfg.episode_length_s = 10.0
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 10
    cfg.decimation = 4  # Policy and reference both 50 Hz, physics 200 Hz.
    cfg.sim.nconmax, cfg.sim.njmax = 32, 128
    cfg.viewer.body_name = "pelvis"
    return cfg


def make_runner_cfg():
    from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 128), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="ms_human_tracking",
        logger="tensorboard",
        upload_model=False,
        max_iterations=3000,
        save_interval=100,
        clip_actions=3.0,
        num_steps_per_env=24,
    )


def register_task(directory, num_envs=1024):
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import register_mjlab_task
    from . import TASK_ID

    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_env_cfg(directory, num_envs),
        play_env_cfg=make_env_cfg(directory, 1, play=True),
        rl_cfg=make_runner_cfg(),
        runner_cls=MjlabOnPolicyRunner,
    )
