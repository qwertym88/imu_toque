#!/usr/bin/env python3
"""Track a retargeted motion with a condensed inverse-dynamics WBC QP."""

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np
import osqp
from scipy.spatial import ConvexHull
from scipy import sparse
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
TASKS = ("left_foot", "right_foot", "pelvis")
NV = 18
NTAU = 12
NLAMBDA = 12
NZ = NV + NLAMBDA + NTAU


def require_id(model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing {name!r}")
    return object_id


def load_reference(model, path, smooth_window):
    with np.load(path, allow_pickle=False) as motion:
        qpos = motion["qpos"].astype(float, copy=True)
        frames = motion["frame"].astype(float, copy=True)
        fps = float(motion["fps"])
        support = motion["support_mask"].astype(bool, copy=True)

    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos):
        raise ValueError("Motion qpos does not match the MuJoCo model")
    if support.shape != (len(qpos), 2) or len(frames) != len(qpos):
        raise ValueError("Motion support mask or frame array has an invalid shape")
    if fps <= 0 or not np.isfinite(qpos).all() or np.any(np.diff(frames) <= 0):
        raise ValueError("Motion contains invalid values or timing")

    dt_ref = 1.0 / fps
    qvel = np.zeros((len(qpos), model.nv))
    for i in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(model, qvel[i], dt_ref, qpos[i], qpos[i + 1])
    qvel[-1] = qvel[-2]
    if smooth_window >= 3 and len(qpos) >= smooth_window:
        qvel = savgol_filter(qvel, smooth_window, 3, axis=0, mode="interp")
    qacc = np.gradient(qvel, dt_ref, axis=0, edge_order=2)

    ref_data = mujoco.MjData(model)
    body_ids = {"pelvis": require_id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")}
    site_ids = {
        f"{side}_foot": require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        for side in SIDES
    }
    sole_geom_ids = {
        side: require_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole_collision")
        for side in SIDES
    }
    ground_geom_id = require_id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    body_ids.update({name: model.site_bodyid[site_ids[name]] for name in site_ids})
    pose = {name: np.empty((len(qpos), 3)) for name in TASKS}
    rot = {name: np.empty((len(qpos), 3, 3)) for name in TASKS}
    vel = {name: np.empty((len(qpos), 6)) for name in TASKS}
    acc = {name: np.empty((len(qpos), 6)) for name in TASKS}
    com = np.empty((len(qpos), 3))
    com_vel = np.empty((len(qpos), 3))

    for i in range(len(qpos)):
        ref_data.qpos[:] = qpos[i]
        ref_data.qvel[:] = qvel[i]
        mujoco.mj_forward(model, ref_data)
        com_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, ref_data, com_jac, 0)
        com[i] = ref_data.subtree_com[0]
        com_vel[i] = com_jac @ qvel[i]
        for name in TASKS:
            if name == "pelvis":
                body_id = body_ids[name]
                jp = np.zeros((3, model.nv))
                jr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, ref_data, jp, jr, body_id)
                point = ref_data.xpos[body_id].copy()
                pose[name][i] = point
                rot[name][i] = ref_data.xmat[body_id].reshape(3, 3)
            else:
                site_id = site_ids[name]
                body_id = body_ids[name]
                jp = np.zeros((3, model.nv))
                jr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, ref_data, jp, jr, site_id)
                point = ref_data.site_xpos[site_id].copy()
                pose[name][i] = point
                rot[name][i] = ref_data.site_xmat[site_id].reshape(3, 3)
            jdp = np.zeros((3, model.nv))
            jdr = np.zeros((3, model.nv))
            mujoco.mj_jacDot(model, ref_data, jdp, jdr, point, body_id)
            jac = np.vstack((jp, jr))
            jac_dot = np.vstack((jdp, jdr))
            vel[name][i] = jac @ qvel[i]
            acc[name][i] = jac @ qacc[i] + jac_dot @ qvel[i]

    return {
        "qpos": qpos,
        "qvel": qvel,
        "qacc": qacc,
        "support": support,
        "time": (frames - frames[0]) / fps,
        "fps": fps,
        "pose": pose,
        "com": com,
        "com_vel": com_vel,
        "com_acc": np.gradient(com_vel, dt_ref, axis=0, edge_order=2),
        "rot": {name: Rotation.from_matrix(rot[name]) for name in TASKS},
        "vel": vel,
        "acc": acc,
        "duration": (frames[-1] - frames[0] + 1.0) / fps,
        "site_ids": site_ids,
        "body_ids": body_ids,
        "sole_geom_ids": sole_geom_ids,
        "ground_geom_id": ground_geom_id,
    }


def interpolate_rotation(rotations, index, alpha):
    r0 = rotations[index]
    r1 = rotations[min(index + 1, len(rotations) - 1)]
    return r0 * Rotation.from_rotvec(alpha * (r1 * r0.inv()).as_rotvec())


def interp(array, index, alpha):
    next_index = min(index + 1, len(array) - 1)
    return (1.0 - alpha) * array[index] + alpha * array[next_index]


def task_state(model, data, name, body_id=None, site_id=None):
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    if site_id is not None:
        mujoco.mj_jacSite(model, data, jp, jr, site_id)
        point = data.site_xpos[site_id].copy()
        rotation = data.site_xmat[site_id].reshape(3, 3).copy()
    else:
        mujoco.mj_jacBody(model, data, jp, jr, body_id)
        point = data.xpos[body_id].copy()
        rotation = data.xmat[body_id].reshape(3, 3).copy()
    jdp = np.zeros((3, model.nv))
    jdr = np.zeros((3, model.nv))
    mujoco.mj_jacDot(model, data, jdp, jdr, point, body_id)
    jac = np.vstack((jp, jr))
    jac_dot = np.vstack((jdp, jdr))
    return jac, jac_dot, point, rotation, jac @ data.qvel, jac_dot @ data.qvel


def add_task(H, g, J, jdot_v, command, weight, scale):
    # Normalize acceleration residuals before applying the dimensionless task weight.
    A = np.zeros((J.shape[0], NZ))
    A[:, :NV] = J
    offset = jdot_v - command
    row_weight = weight / (scale * scale)
    H += A.T @ (row_weight[:, None] * A)
    g += A.T @ (row_weight * offset)


def append_constraint(rows, limits, row, upper):
    rows.append(row)
    limits.append(float(upper))


def skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def support_polygon(data, support, site_ids, cop_x, cop_y):
    corners = []
    for foot, side in enumerate(SIDES):
        if not support[foot]:
            continue
        site_id = site_ids[f"{side}_foot"]
        rotation = data.site_xmat[site_id].reshape(3, 3)
        center = data.site_xpos[site_id]
        for x in (-cop_x, cop_x):
            for y in (-cop_y, cop_y):
                corners.append((center + rotation @ np.array([x, y, 0.0]))[:2])
    if not corners:
        return np.empty((0, 3)), np.zeros(2)
    hull = ConvexHull(np.asarray(corners))
    equations = hull.equations.copy()
    center = hull.points[hull.vertices].mean(axis=0)
    return equations, center


def build_inequalities(
    model,
    data,
    support,
    site_ids,
    ground_geom_id,
    mu,
    cop_x,
    cop_y,
    zmp_margin,
    tau_limits,
):
    rows, upper = [], []
    # Actuator bounds are expressed in ctrl units; model motors have gear=1.
    for i in range(NTAU):
        row = np.zeros(NZ)
        row[NV + NLAMBDA + i] = 1.0
        append_constraint(rows, upper, row, tau_limits[i])
        append_constraint(rows, upper, -row, tau_limits[i])

    # Contact wrench order is world [force, moment] about each foot site.
    for foot, side in enumerate(SIDES):
        if not support[foot]:
            continue
        site_id = site_ids[f"{side}_foot"]
        rotation = data.site_xmat[site_id].reshape(3, 3)
        to_local = np.zeros((6, 6))
        to_local[:3, :3] = rotation.T
        to_local[3:, 3:] = rotation.T
        lam_slice = slice(NV + 6 * foot, NV + 6 * (foot + 1))
        facets = [
            ([1, 0, -mu, 0, 0, 0], 0),
            ([-1, 0, -mu, 0, 0, 0], 0),
            ([0, 1, -mu, 0, 0, 0], 0),
            ([0, -1, -mu, 0, 0, 0], 0),
            ([0, 0, -1, 0, 0, 0], 0),
            ([0, 0, -cop_y, 1, 0, 0], 0),
            ([0, 0, -cop_y, -1, 0, 0], 0),
            ([0, 0, -cop_x, 0, 1, 0], 0),
            ([0, 0, -cop_x, 0, -1, 0], 0),
            ([0, 0, -0.02, 0, 0, 1], 0),
            ([0, 0, -0.02, 0, 0, -1], 0),
        ]
        for coeff, bound in facets:
            row = np.zeros(NZ)
            row[lam_slice] = np.asarray(coeff) @ to_local
            append_constraint(rows, upper, row, bound)

    # Constrain the resultant ZMP to the convex hull of the active sole patches.
    equations, _ = support_polygon(data, support, site_ids, cop_x, cop_y)
    ground_z = data.geom_xpos[ground_geom_id, 2]
    plane_origin = np.array([0.0, 0.0, ground_z])
    for nx, ny, offset in equations:
        bound = -offset - zmp_margin
        row = np.zeros(NZ)
        for foot, side in enumerate(SIDES):
            if not support[foot]:
                continue
            site_id = site_ids[f"{side}_foot"]
            lever = data.site_xpos[site_id] - plane_origin
            local_to_plane = np.eye(6)
            local_to_plane[3:, :3] = skew(lever)
            lambda_slice = slice(NV + 6 * foot, NV + 6 * (foot + 1))
            zmp_coeff = np.array([0.0, 0.0, -bound, ny, -nx, 0.0])
            row[lambda_slice] += zmp_coeff @ local_to_plane
        append_constraint(rows, upper, row, 0.0)
    # Keep a fixed row count per contact mode for the cached OSQP workspaces.
    for _ in range(8 - len(equations)):
        append_constraint(rows, upper, np.zeros(NZ), 0.0)

    return np.asarray(rows), np.asarray(upper)


def solve_reduced_qp(model, data, reference, time_s, args, tau_limits, contact_anchors, qp_solvers):
    index = min(int(time_s * reference["fps"]), len(reference["time"]) - 1)
    alpha = min(max(time_s * reference["fps"] - index, 0.0), 1.0)
    support_index = index if alpha < 0.5 else min(index + 1, len(reference["time"]) - 1)
    scheduled_support = reference["support"][support_index]

    mujoco.mj_forward(model, data)
    physical_support = np.zeros(2, dtype=bool)
    for i in range(data.ncon):
        contact = data.contact[i]
        for foot, side in enumerate(SIDES):
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {reference["ground_geom_id"], reference["sole_geom_ids"][side]}:
                physical_support[foot] = True
    # The retargeted contact schedule authorizes support. A foot that still
    # brushes the ground after liftoff is commanded as swing so it can release.
    support = physical_support & scheduled_support
    for foot, side in enumerate(SIDES):
        if support[foot] and contact_anchors[side] is None:
            site_id = reference["site_ids"][f"{side}_foot"]
            contact_anchors[side] = (
                data.site_xpos[site_id].copy(),
                Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3).copy()),
            )
        elif not support[foot]:
            contact_anchors[side] = None
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    bias = data.qfrc_bias.copy()
    actuation = np.zeros((model.nv, model.nu))
    for actuator, joint_id in enumerate(model.actuator_trnid[:, 0]):
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            raise ValueError("The prototype currently requires joint-transmission actuators")
        dof = int(model.jnt_dofadr[joint_id])
        actuation[dof, actuator] = data.actuator_moment[actuator]

    foot_jac = {}
    foot_jdot_v = {}
    active_sides = []
    for foot, side in enumerate(SIDES):
        site_id = reference["site_ids"][f"{side}_foot"]
        state = task_state(
            model,
            data,
            f"{side}_foot",
            body_id=reference["body_ids"][f"{side}_foot"],
            site_id=site_id,
        )
        foot_jac[side], foot_jdot_v[side] = state[0], state[5]
        if support[foot]:
            active_sides.append(side)

    # The linear part of centroidal momentum rate is m * COM acceleration.
    # Its reference can be reconstructed directly from qpos/qvel without an MPC wrench.
    com_jac = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, com_jac, 0)
    com_velocity = com_jac @ data.qvel
    com_jdot_v = np.zeros(3)
    if np.linalg.norm(data.qvel) > 1e-8:
        epsilon = 1e-5
        probe = mujoco.MjData(model)
        probe.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, probe.qpos, data.qvel, epsilon)
        probe.qvel[:] = data.qvel
        mujoco.mj_forward(model, probe)
        probe_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, probe, probe_jac, 0)
        com_jdot_v = ((probe_jac - com_jac) / epsilon) @ data.qvel
    com_target = interp(reference["com"], index, alpha)
    if np.any(support):
        support_positions = np.array(
            [
                data.site_xpos[reference["site_ids"][f"{side}_foot"]][:2]
                for foot, side in enumerate(SIDES)
                if support[foot]
            ]
        )
        com_target[:2] = np.clip(
            com_target[:2],
            support_positions.min(axis=0) - args.com_support_margin,
            support_positions.max(axis=0) + args.com_support_margin,
        )
    com_velocity_target = args.speed * interp(reference["com_vel"], index, alpha)
    if np.count_nonzero(support) == 1:
        com_velocity_target[:2] = 0.0
    com_acc_target = args.speed**2 * interp(reference["com_acc"], index, alpha)
    com_acc_target = np.clip(com_acc_target, -args.com_accel_limit, args.com_accel_limit)
    com_acc_command = (
        com_acc_target
        + args.kp_com * (com_target - data.subtree_com[0])
        + args.kd_com * (com_velocity_target - com_velocity)
    )
    com_acc_command = np.clip(com_acc_command, -args.com_accel_limit, args.com_accel_limit)
    com_acc_command[2] = max(com_acc_command[2], -args.com_vertical_down_limit)

    # Full equality system for z=[qacc(18), both 6D foot wrenches, tau(12)].
    Aeq = np.zeros((30, NZ))
    beq = np.zeros(30)
    Aeq[:NV, :NV] = mass
    for foot in range(2):
        Aeq[:NV, NV + 6 * foot : NV + 6 * (foot + 1)] = -foot_jac[SIDES[foot]].T
    Aeq[:NV, NV + NLAMBDA :] = -actuation
    beq[:NV] = -bias

    row = NV
    for side in active_sides:
        Aeq[row : row + 6, :NV] = foot_jac[side]
        site_id = reference["site_ids"][f"{side}_foot"]
        current_rotation = Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3))
        anchor_position, anchor_rotation = contact_anchors[side]
        position_error = anchor_position - data.site_xpos[site_id]
        rotation_error = (anchor_rotation * current_rotation.inv()).as_rotvec()
        contact_velocity = foot_jac[side] @ data.qvel
        contact_command = np.r_[
            args.kp_contact * position_error - args.kd_contact * contact_velocity[:3],
            args.kp_contact * rotation_error - args.kd_contact * contact_velocity[3:],
        ]
        beq[row : row + 6] = contact_command - foot_jdot_v[side]
        row += 6
    for foot, side in enumerate(SIDES):
        if side not in active_sides:
            Aeq[row : row + 6, NV + 6 * foot : NV + 6 * (foot + 1)] = np.eye(6)
            row += 6
    if row != 30:
        raise RuntimeError(f"Unexpected equality count {row}; expected 30")

    H = np.eye(NZ) * 1e-10
    g = np.zeros(NZ)
    add_task(
        H,
        g,
        com_jac,
        com_jdot_v,
        com_acc_command,
        np.full(3, args.momentum_weight / (args.linear_accel_scale**2)),
        np.ones(3),
    )
    zmp_faces, zmp_target = support_polygon(
        data, support, reference["site_ids"], args.cop_x, args.cop_y
    )
    if len(zmp_faces):
        zmp_task = np.zeros((2, NZ))
        ground_z = data.geom_xpos[reference["ground_geom_id"], 2]
        plane_origin = np.array([0.0, 0.0, ground_z])
        for foot, side in enumerate(SIDES):
            if not support[foot]:
                continue
            site_id = reference["site_ids"][f"{side}_foot"]
            lever = data.site_xpos[site_id] - plane_origin
            local_to_plane = np.eye(6)
            local_to_plane[3:, :3] = skew(lever)
            lambda_slice = slice(NV + 6 * foot, NV + 6 * (foot + 1))
            zmp_task[0, lambda_slice] = (
                np.array([0.0, 0.0, -zmp_target[1], 1.0, 0.0, 0.0]) @ local_to_plane
            )
            zmp_task[1, lambda_slice] = (
                np.array([0.0, 0.0, -zmp_target[0], 0.0, -1.0, 0.0]) @ local_to_plane
            )
        zmp_scale = model.body_subtreemass[0] * 9.81 * args.zmp_moment_scale
        mode_weight = args.zmp_weight
        if np.count_nonzero(support) == 1:
            mode_weight *= args.single_support_zmp_scale
        zmp_weight = mode_weight / (zmp_scale * zmp_scale)
        H += zmp_task.T @ (zmp_weight * zmp_task)
    task_weights = {
        "swing_position": args.swing_weight,
        "swing_orientation": args.swing_orientation_weight,
        "pelvis_xy_position": args.pelvis_xy_weight,
        "pelvis_z_position": args.pelvis_z_weight,
        "pelvis_orientation": args.pelvis_orientation_weight,
    }
    for name in TASKS:
        is_pelvis = name == "pelvis"
        foot_index = SIDES.index(name.split("_")[0]) if not is_pelvis else -1
        if not is_pelvis and support[foot_index]:
            continue
        site_id = None if is_pelvis else reference["site_ids"][name]
        body_id = reference["body_ids"][name]
        J, Jdot, position, rotation, velocity, jdot_v = task_state(
            model, data, name, body_id=body_id, site_id=site_id
        )
        target_position = interp(reference["pose"][name], index, alpha)
        target_velocity = args.speed * interp(reference["vel"][name], index, alpha)
        target_acc = args.speed**2 * interp(reference["acc"][name], index, alpha)
        target_rotation = interpolate_rotation(reference["rot"][name], index, alpha).as_matrix()
        recovery = not is_pelvis and scheduled_support[foot_index] and not support[foot_index]
        kp_position = args.kp_contact if recovery else args.kp_position
        kd_position = args.kd_contact if recovery else args.kd_position
        kp_orientation = args.kp_contact if recovery else args.kp_orientation
        kd_orientation = args.kd_contact if recovery else args.kd_orientation
        linear_cmd = (
            target_acc[:3]
            + kp_position * (target_position - position)
            + kd_position * (target_velocity[:3] - velocity[:3])
        )
        rot_error = (
            Rotation.from_matrix(target_rotation) * Rotation.from_matrix(rotation).inv()
        ).as_rotvec()
        angular_cmd = (
            target_acc[3:]
            + kp_orientation * rot_error
            + kd_orientation * (target_velocity[3:] - velocity[3:])
        )
        if is_pelvis:
            weights = np.array(
                [task_weights["pelvis_xy_position"]] * 2
                + [task_weights["pelvis_z_position"]]
                + [task_weights["pelvis_orientation"]] * 3
            )
        else:
            weights = np.array(
                [task_weights["swing_position"]] * 3 + [task_weights["swing_orientation"]] * 3
            )
            if recovery:
                weights[:3] = args.support_recovery_weight
                weights[3:] = args.support_recovery_weight * args.swing_orientation_weight
        scales = np.array([args.linear_accel_scale] * 3 + [args.angular_accel_scale] * 3)
        add_task(
            H, g, J, jdot_v, np.r_[linear_cmd, angular_cmd], weights / (scales * scales), np.ones(6)
        )

    joint_ids = model.actuator_trnid[:, 0]
    joint_dofs = model.jnt_dofadr[joint_ids]
    joint_qpos = model.jnt_qposadr[joint_ids]
    q_ref = interp(reference["qpos"][:, joint_qpos], index, alpha)
    qd_ref = args.speed * interp(reference["qvel"][:, joint_dofs], index, alpha)
    qdd_ref = args.speed**2 * interp(reference["qacc"][:, joint_dofs], index, alpha)
    joint_command = (
        qdd_ref
        + args.kp_joint * (q_ref - data.qpos[joint_qpos])
        + args.kd_joint * (qd_ref - data.qvel[joint_dofs])
    )
    joint_A = np.zeros((NTAU, NZ))
    joint_A[np.arange(NTAU), joint_dofs] = 1.0
    joint_weights = args.joint_weight / (args.joint_accel_scale * args.joint_accel_scale)
    H += joint_A.T @ (joint_weights * joint_A)
    g += joint_A.T @ (-joint_weights * joint_command)

    tau_slice = slice(NV + NLAMBDA, NZ)
    H[tau_slice, tau_slice] += np.diag(args.torque_regularization / (tau_limits * tau_limits))
    H *= args.objective_scale
    g *= args.objective_scale

    # Condense qacc and lambda using the equality system, as in the reference benchmark.
    E = Aeq[:, : NV + NLAMBDA]
    D = Aeq[:, NV + NLAMBDA :]
    reciprocal_condition = 1.0 / np.linalg.cond(E)
    if not np.isfinite(reciprocal_condition) or reciprocal_condition < args.min_rcond:
        raise RuntimeError(
            f"Contact-mode elimination is ill-conditioned (rcond={reciprocal_condition:.3e})"
        )
    p = np.zeros(NZ)
    T = np.zeros((NZ, NTAU))
    p[: NV + NLAMBDA] = np.linalg.solve(E, beq)
    T[: NV + NLAMBDA] = np.linalg.solve(E, -D)
    T[NV + NLAMBDA :] = np.eye(NTAU)

    H_reduced = T.T @ H @ T
    H_reduced = 0.5 * (H_reduced + H_reduced.T)
    g_reduced = T.T @ (H @ p + g)
    tau_scale = np.diag(tau_limits)
    H_normalized = tau_scale @ H_reduced @ tau_scale
    H_normalized = 0.5 * (H_normalized + H_normalized.T)
    g_normalized = tau_scale @ g_reduced
    C, upper = build_inequalities(
        model,
        data,
        support,
        reference["site_ids"],
        reference["ground_geom_id"],
        args.friction,
        args.cop_x,
        args.cop_y,
        args.zmp_margin,
        tau_limits,
    )
    A_reduced = C @ T @ tau_scale
    upper_reduced = upper - C @ p
    row_scale = np.maximum(np.max(np.abs(A_reduced), axis=1), 1e-8)
    A_solver = A_reduced / row_scale[:, None]
    upper_solver = upper_reduced / row_scale
    mode = tuple(bool(x) for x in support)
    if mode not in qp_solvers:
        solver = osqp.OSQP()
        p_pattern = sparse.triu(sparse.csc_matrix(np.ones((NTAU, NTAU)))).tocsc()
        a_pattern = sparse.csc_matrix(np.ones(A_solver.shape))
        solver.setup(
            P=p_pattern,
            q=g_normalized,
            A=a_pattern,
            l=np.full(len(upper_solver), -np.inf),
            u=upper_solver,
            verbose=False,
            polishing=False,
            max_iter=args.qp_max_iter,
            eps_abs=args.qp_tolerance,
            eps_rel=args.qp_tolerance,
            adaptive_rho=True,
            warm_starting=True,
        )
        qp_solvers[mode] = (solver, p_pattern, a_pattern)
    solver, p_pattern, a_pattern = qp_solvers[mode]
    p_values = np.array(
        [
            H_normalized[row, col]
            for col in range(NTAU)
            for row in p_pattern.indices[p_pattern.indptr[col] : p_pattern.indptr[col + 1]]
        ]
    )
    solver.update(
        Px=p_values,
        q=g_normalized,
        Ax=A_solver.ravel(order="F"),
        l=np.full(len(upper_solver), -np.inf),
        u=upper_solver,
    )
    result = solver.solve()
    if result.info.status_val not in (1, 2) or result.x is None:
        raise RuntimeError(f"Reduced WBC QP failed: {result.info.status}")
    z = p + T @ (tau_scale @ result.x)
    eq_residual = np.linalg.norm(Aeq @ z - beq, ord=np.inf)
    ineq_violation = max(0.0, float(np.max(C @ z - upper)))
    if eq_residual > args.feasibility_tolerance or ineq_violation > args.feasibility_tolerance:
        raise RuntimeError(
            f"QP residual too large (eq={eq_residual:.3e}, inequality={ineq_violation:.3e})"
        )
    return (
        z,
        result.info.iter,
        eq_residual,
        ineq_violation,
        reciprocal_condition,
        support,
        scheduled_support,
        physical_support,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=ROOT / "data/retargeted_motion.npz")
    parser.add_argument("--model", type=Path, default=ROOT / "model/szar_humanoid.xml")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--output", type=Path, default=None, help="optional NPZ trace for torque/QP analysis"
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--cop-x", type=float, default=0.07, help="sole half-length in local x (m)")
    parser.add_argument("--cop-y", type=float, default=0.04, help="sole half-width in local y (m)")
    parser.add_argument("--swing-weight", type=float, default=20.0)
    parser.add_argument("--swing-orientation-weight", type=float, default=20.0)
    parser.add_argument("--support-recovery-weight", type=float, default=40.0)
    parser.add_argument("--pelvis-xy-weight", type=float, default=1.0)
    parser.add_argument("--pelvis-z-weight", type=float, default=10.0)
    parser.add_argument("--pelvis-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pelvis-orientation-weight", type=float, default=10.0)
    parser.add_argument("--zmp-weight", type=float, default=0.1)
    parser.add_argument("--single-support-zmp-scale", type=float, default=1.0)
    parser.add_argument(
        "--zmp-moment-scale", type=float, default=0.05, help="ZMP moment residual scale in m"
    )
    parser.add_argument(
        "--zmp-margin", type=float, default=0.005, help="inward support-polygon margin in m"
    )
    parser.add_argument("--momentum-weight", type=float, default=100.0)
    parser.add_argument("--com-accel-limit", type=float, default=2.0)
    parser.add_argument("--com-vertical-down-limit", type=float, default=0.25)
    parser.add_argument("--com-support-margin", type=float, default=0.035)
    parser.add_argument("--joint-weight", type=float, default=0.1)
    parser.add_argument("--linear-accel-scale", type=float, default=10.0)
    parser.add_argument("--angular-accel-scale", type=float, default=20.0)
    parser.add_argument("--joint-accel-scale", type=float, default=10.0)
    parser.add_argument("--torque-regularization", type=float, default=1e-3)
    parser.add_argument("--objective-scale", type=float, default=1e4)
    parser.add_argument("--kp-position", type=float, default=100.0)
    parser.add_argument("--kd-position", type=float, default=20.0)
    parser.add_argument("--kp-orientation", type=float, default=100.0)
    parser.add_argument("--kd-orientation", type=float, default=20.0)
    parser.add_argument("--kp-contact", type=float, default=400.0)
    parser.add_argument("--kd-contact", type=float, default=40.0)
    parser.add_argument("--kp-com", type=float, default=100.0)
    parser.add_argument("--kd-com", type=float, default=20.0)
    parser.add_argument("--kp-joint", type=float, default=200.0)
    parser.add_argument("--kd-joint", type=float, default=30.0)
    parser.add_argument("--qp-max-iter", type=int, default=10000)
    parser.add_argument("--qp-tolerance", type=float, default=1e-6)
    parser.add_argument("--feasibility-tolerance", type=float, default=2e-3)
    parser.add_argument("--min-rcond", type=float, default=1e-9)
    parser.add_argument("--smooth-window", type=int, default=9)
    args = parser.parse_args()
    if args.pelvis_weight is not None:
        args.pelvis_xy_weight = args.pelvis_weight
        args.pelvis_z_weight = args.pelvis_weight
    if args.speed <= 0 or args.friction <= 0 or args.cop_x <= 0 or args.cop_y <= 0:
        parser.error("speed, friction, and CoP dimensions must be positive")
    if args.zmp_margin < 0 or args.zmp_margin >= min(args.cop_x, args.cop_y):
        parser.error("zmp margin must be non-negative and smaller than both CoP dimensions")
    if args.zmp_weight < 0 or args.single_support_zmp_scale < 0 or args.zmp_moment_scale <= 0:
        parser.error("zmp weights must be non-negative and zmp moment scale must be positive")

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    reference = load_reference(model, args.motion, args.smooth_window)
    if model.nv != NV or model.nu != NTAU:
        raise ValueError(f"This prototype expects nv=18 and nu=12; got {model.nv}, {model.nu}")
    actuator_joint_ids = model.actuator_trnid[:, 0]
    tau_limits = np.empty(NTAU)
    for i, joint_id in enumerate(actuator_joint_ids):
        if model.jnt_actfrclimited[joint_id]:
            tau_limits[i] = np.min(np.abs(model.jnt_actfrcrange[joint_id]))
        else:
            tau_limits[i] = np.max(np.abs(model.actuator_ctrlrange[i]))

    data.qpos[:] = reference["qpos"][0]
    data.qvel[:] = reference["qvel"][0]
    mujoco.mj_forward(model, data)
    reference_duration = reference["duration"] / args.speed
    duration = (
        reference_duration if args.duration is None else min(args.duration, reference_duration)
    )
    if duration <= 0:
        parser.error("duration must be positive")

    state = {"failed": None, "steps": 0, "max_eq": 0.0, "max_ineq": 0.0, "max_rcond": 1.0}
    contact_anchors = {side: None for side in SIDES}
    qp_solvers = {}
    trace = {
        name: []
        for name in (
            "time",
            "qpos",
            "qvel",
            "qacc",
            "tau",
            "wrench",
            "zmp",
            "zmp_target",
            "support",
            "physical_support",
            "planned_support",
            "eq_error",
            "ineq_error",
        )
    }

    def step_once(viewer=None):
        try:
            z, iterations, eq_error, ineq_error, rcond, support, scheduled, physical_support = (
                solve_reduced_qp(
                    model,
                    data,
                    reference,
                    data.time * args.speed,
                    args,
                    tau_limits,
                    contact_anchors,
                    qp_solvers,
                )
            )
        except Exception as exc:
            mujoco.mj_forward(model, data)
            foot_z = [data.site_xpos[reference["site_ids"][f"{side}_foot"], 2] for side in SIDES]
            joints = data.qpos[model.jnt_qposadr[model.actuator_trnid[:, 0]]]
            state["failed"] = (
                f"t={data.time:.3f}s: {exc}; base_z={data.qpos[2]:.3f}; "
                f"foot_z={np.asarray(foot_z)}; joints={np.round(joints, 3)}; ncon={data.ncon}"
            )
            return False
        data.ctrl[:] = z[NV + NLAMBDA :]
        trace["time"].append(data.time)
        trace["qpos"].append(data.qpos.copy())
        trace["qvel"].append(data.qvel.copy())
        trace["qacc"].append(z[:NV].copy())
        trace["tau"].append(z[NV + NLAMBDA :].copy())
        wrench = z[NV : NV + NLAMBDA].reshape(2, 6).copy()
        trace["wrench"].append(wrench)
        zmp_target = support_polygon(data, support, reference["site_ids"], args.cop_x, args.cop_y)[
            1
        ]
        ground_z = data.geom_xpos[reference["ground_geom_id"], 2]
        moment_at_ground = np.zeros(3)
        force_total = np.zeros(3)
        for foot, side in enumerate(SIDES):
            if not support[foot]:
                continue
            site_id = reference["site_ids"][f"{side}_foot"]
            force = wrench[foot, :3]
            force_total += force
            moment_at_ground += wrench[foot, 3:] + np.cross(
                data.site_xpos[site_id] - np.array([0.0, 0.0, ground_z]), force
            )
        zmp = (
            np.array([-moment_at_ground[1], moment_at_ground[0]]) / force_total[2]
            if force_total[2] > 1e-3
            else np.full(2, np.nan)
        )
        trace["zmp"].append(zmp)
        trace["zmp_target"].append(zmp_target if np.any(support) else np.full(2, np.nan))
        trace["support"].append(support.copy())
        trace["physical_support"].append(physical_support.copy())
        trace["planned_support"].append(scheduled.copy())
        trace["eq_error"].append(eq_error)
        trace["ineq_error"].append(ineq_error)
        mujoco.mj_step(model, data)
        state["steps"] += 1
        state["max_eq"] = max(state["max_eq"], eq_error)
        state["max_ineq"] = max(state["max_ineq"], ineq_error)
        state["max_rcond"] = min(state["max_rcond"], rcond)
        if viewer is not None:
            viewer.sync()
        if state["steps"] % 250 == 0:
            print(
                f"t={data.time:.3f}s support={''.join('L' if x else '-' for x in support[:1])}"
                f"{''.join('R' if x else '-' for x in support[1:])} "
                f"planned={''.join('L' if x else '-' for x in scheduled[:1])}"
                f"{''.join('R' if x else '-' for x in scheduled[1:])} "
                f"base_z={data.qpos[2]:.3f} tau_max={np.max(np.abs(z[NV + NLAMBDA:])):.1f} "
                f"qp_iter={iterations} max_eq={state['max_eq']:.2e} "
                f"max_ineq={state['max_ineq']:.2e}",
                flush=True,
            )
        return True

    if args.headless:
        while data.time < duration:
            if not step_once():
                break
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -15
            while viewer.is_running() and data.time < duration:
                if not step_once(viewer):
                    break
                time.sleep(max(0.0, model.opt.timestep - 0.0002))

    print(
        f"steps={state['steps']} duration={data.time:.3f}s "
        f"max_eq={state['max_eq']:.3e} max_ineq={state['max_ineq']:.3e} "
        f"min_elimination_rcond={state['max_rcond']:.3e}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        actuator_names = np.array(
            [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                for joint_id in actuator_joint_ids
            ]
        )
        np.savez_compressed(
            args.output,
            **{key: np.asarray(value) for key, value in trace.items()},
            joint_names=actuator_names,
            wrench_order=np.array(["Fx", "Fy", "Fz", "Mx", "My", "Mz"]),
        )
        print(f"trace={args.output}")
    if state["failed"]:
        raise RuntimeError(state["failed"])


if __name__ == "__main__":
    main()
