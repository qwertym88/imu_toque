#!/usr/bin/env python3
"""Track a retargeted motion with a condensed inverse-dynamics WBC QP."""

import argparse
import json
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

from wbc_balance import BalanceReference, project_to_polygon


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
    return Rotation.from_rotvec(alpha * (r1 * r0.inv()).as_rotvec()) * r0


def interp(array, index, alpha):
    next_index = min(index + 1, len(array) - 1)
    return (1.0 - alpha) * array[index] + alpha * array[next_index]


def task_state(model, data, name, body_id=None, site_id=None):
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    if site_id is not None:
        if body_id is None:
            body_id = int(model.site_bodyid[site_id])
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


def ground_wrench_matrix(data, support, site_ids, ground_z):
    matrix = np.zeros((6, NZ))
    origin = np.array([0.0, 0.0, ground_z])
    for foot, side in enumerate(SIDES):
        if not support[foot]:
            continue
        lever = data.site_xpos[site_ids[f"{side}_foot"]] - origin
        block = np.eye(6)
        block[3:, :3] = skew(lever)
        matrix[:, NV + 6 * foot : NV + 6 * (foot + 1)] = block
    return matrix


def resultant_zmp(wrench, positions, ground_z, minimum_force=1.0):
    force = wrench[:, :3].sum(axis=0)
    moments = wrench[:, 3:] + np.cross(positions - np.array([0.0, 0.0, ground_z]), wrench[:, :3])
    moment = moments.sum(axis=0)
    zmp = (
        np.array([-moment[1], moment[0]]) / force[2]
        if force[2] > minimum_force
        else np.full(2, np.nan)
    )
    return zmp, force


def linearized_zmp_task(numerator, normal_row, previous, body_weight, target):
    normal = float(normal_row @ previous)
    if normal > 0.2 * body_weight:
        point = numerator @ previous / normal
    else:
        normal = body_weight
        point = target.copy()
    return (numerator - point[:, None] * normal_row) / normal, point - target


def measured_contact_wrenches(model, data, reference):
    """World [force; moment] on each sole, about its site, after mj_step.

    mj_step's contact geometry and force outputs describe its pre-integration
    configuration, matching the state saved for this control tick. Do not call
    mj_forward between the step and reading these outputs.
    """
    wrench = np.zeros((2, 6))
    counts = np.zeros(2, dtype=int)
    ground = reference["ground_geom_id"]
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        pair = (int(contact.geom1), int(contact.geom2))
        for foot, side in enumerate(SIDES):
            sole = reference["sole_geom_ids"][side]
            if set(pair) != {ground, sole} or contact.efc_address < 0:
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, local)
            # Contact normal points from geom1 to geom2; force is on geom2.
            sign = 1.0 if pair[1] == sole else -1.0
            rotation = contact.frame.reshape(3, 3).T
            force = sign * (rotation @ local[:3])
            moment = sign * (rotation @ local[3:])
            site = reference["site_ids"][f"{side}_foot"]
            wrench[foot, :3] += force
            wrench[foot, 3:] += moment + np.cross(contact.pos - data.site_xpos[site], force)
            if local[0] > 1.0:
                counts[foot] += 1
    return wrench, counts


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
    minimum_normal_force=0.0,
    joint_limit_horizon=0.0,
    joint_limit_margin=0.01,
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
    total_wrench = ground_wrench_matrix(data, support, site_ids, ground_z)
    append_constraint(
        rows, upper, -total_wrench[2], -minimum_normal_force if np.any(support) else 0.0
    )
    if joint_limit_horizon > 0:
        for joint in model.actuator_trnid[:, 0]:
            if not model.jnt_limited[joint]:
                continue
            dof, address = int(model.jnt_dofadr[joint]), int(model.jnt_qposadr[joint])
            predicted = data.qpos[address] + joint_limit_horizon * data.qvel[dof]
            factor = 0.5 * joint_limit_horizon**2
            row = np.zeros(NZ)
            row[dof] = 1.0
            append_constraint(
                rows,
                upper,
                row,
                (model.jnt_range[joint, 1] - joint_limit_margin - predicted) / factor,
            )
            append_constraint(
                rows,
                upper,
                -row,
                (predicted - model.jnt_range[joint, 0] - joint_limit_margin) / factor,
            )

    return np.asarray(rows), np.asarray(upper)


def solve_reduced_qp(
    model,
    data,
    reference,
    time_s,
    args,
    tau_limits,
    contact_anchors,
    qp_solvers,
    controller_state,
):
    index = min(int(time_s * reference["fps"]), len(reference["time"]) - 1)
    alpha = min(max(time_s * reference["fps"] - index, 0.0), 1.0)
    # Contact phases are zero-order held; do not round liftoff half a frame early.
    scheduled_support = reference["support"][index]

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
    previous_schedule = controller_state.get("scheduled_support")
    if previous_schedule is None or np.any(previous_schedule != scheduled_support):
        controller_state["mode_changed_at"] = data.time
        controller_state["transition_support"] = controller_state.get("support", support).copy()
    if (
        not np.any(support)
        and data.time - controller_state["mode_changed_at"] < args.touchdown_grace
    ):
        support = physical_support & controller_state["transition_support"]
    controller_state["scheduled_support"] = scheduled_support.copy()
    controller_state["support"] = support.copy()
    for foot, side in enumerate(SIDES):
        if support[foot] and contact_anchors[side] is None:
            site_id = reference["site_ids"][f"{side}_foot"]
            anchor_position = data.site_xpos[site_id].copy()
            anchor_rotation = Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3).copy())
            if args.stance_pose == "flat":
                anchor_position[2] = data.geom_xpos[reference["ground_geom_id"], 2]
                matrix = anchor_rotation.as_matrix()
                anchor_rotation = Rotation.from_euler("z", np.arctan2(matrix[1, 0], matrix[0, 0]))
            contact_anchors[side] = (
                anchor_position,
                anchor_rotation,
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
    angular_jac = np.zeros((3, model.nv))
    mujoco.mj_angmomMat(model, data, angular_jac, 0)
    angular_momentum = angular_jac @ data.qvel
    angular_bias = np.zeros(3)
    if np.linalg.norm(data.qvel) > 1e-8:
        epsilon = 1e-5
        probe = controller_state["probe"]
        probe.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, probe.qpos, data.qvel, epsilon)
        probe.qvel[:] = data.qvel
        mujoco.mj_forward(model, probe)
        probe_jac = np.zeros((3, model.nv))
        mujoco.mj_jacSubtreeCom(model, probe, probe_jac, 0)
        com_jdot_v = ((probe_jac - com_jac) / epsilon) @ data.qvel
        probe_angular = np.zeros((3, model.nv))
        mujoco.mj_angmomMat(model, probe, probe_angular, 0)
        angular_bias = ((probe_angular - angular_jac) / epsilon) @ data.qvel
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
    zmp_faces, zmp_target = support_polygon(
        data, support, reference["site_ids"], args.cop_x, args.cop_y
    )
    ground_z = data.geom_xpos[reference["ground_geom_id"], 2]
    height = max(reference["com"][0, 2] - ground_z, 0.1)
    omega = np.sqrt(9.81 / height)
    dcm = data.subtree_com[0, :2] + com_velocity[:2] / omega
    dcm_target = com_target[:2].copy()
    load_target = scheduled_support.astype(float) / np.count_nonzero(scheduled_support)
    if args.balance_mode == "preview":
        plan = reference["balance"]
        com_xy, vel_xy, dcm_target, zmp_target, load_target, planned_feet = plan.sample(data.time)
        active = np.flatnonzero(support)
        if len(active):
            shift = np.mean(
                [contact_anchors[SIDES[f]][0][:2] - planned_feet[f] for f in active], axis=0
            )
            com_xy += shift
            dcm_target += shift
            zmp_target += shift
        com_target = np.r_[com_xy, reference["com"][0, 2]]
        com_velocity_target = np.r_[vel_xy, 0.0]
        zmp_command = zmp_target + (1.0 + args.dcm_gain / omega) * (dcm - dcm_target)
        if len(zmp_faces):
            zmp_command = project_to_polygon(zmp_command, zmp_faces, args.zmp_margin)
        com_acc_command = np.r_[
            omega**2 * (data.subtree_com[0, :2] - zmp_command),
            args.kp_com * (com_target[2] - data.subtree_com[0, 2]) - args.kd_com * com_velocity[2],
        ]
        com_acc_command = np.clip(com_acc_command, -args.com_accel_limit, args.com_accel_limit)
        com_acc_command[2] = max(com_acc_command[2], -args.com_vertical_down_limit)
        zmp_target = zmp_command

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
    if args.angular_momentum_weight > 0:
        angular_scale = model.body_subtreemass[0] * height
        add_task(
            H,
            g,
            angular_jac / angular_scale,
            angular_bias / angular_scale,
            -args.angular_momentum_gain * angular_momentum / angular_scale,
            np.full(3, args.angular_momentum_weight),
            np.full(3, args.linear_accel_scale),
        )
    if len(zmp_faces):
        total_wrench = ground_wrench_matrix(data, support, reference["site_ids"], ground_z)
        normal_row = total_wrench[2]
        numerator = np.vstack((-total_wrench[4], total_wrench[3]))
        body_weight = model.body_subtreemass[0] * 9.81
        normal_target = model.body_subtreemass[0] * (9.81 + com_acc_command[2])
        if args.zmp_cost == "linearized":
            previous = controller_state.get("previous_z", np.zeros(NZ))
            # First-order ratio, not Fz^2 times a position error. Its affine
            # offset is essential; homogeneous wrench scaling has zero gradient.
            zmp_task, zmp_offset = linearized_zmp_task(
                numerator, normal_row, previous, body_weight, zmp_target
            )
        else:
            zmp_task = (numerator - zmp_target[:, None] * normal_row) / body_weight
            zmp_offset = np.zeros(2)
        mode_weight = args.zmp_weight
        if np.count_nonzero(support) == 1:
            mode_weight *= args.single_support_zmp_scale
        zmp_weight = mode_weight / args.zmp_moment_scale**2
        H += zmp_task.T @ (zmp_weight * zmp_task)
        g += zmp_task.T @ (zmp_weight * zmp_offset)
        force_weight = args.normal_force_weight / body_weight**2
        H += force_weight * np.outer(normal_row, normal_row)
        g -= force_weight * normal_row * normal_target
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
        landing_blend = 0.0
        if not is_pelvis and not support[foot_index] and args.landing_preview > 0:
            future = reference["touchdowns"][foot_index]
            future_index = np.searchsorted(future, time_s, side="left")
            if future_index < len(future):
                until_touchdown = (future[future_index] - time_s) / args.speed
                landing_blend = float(
                    np.clip(1.0 - until_touchdown / args.landing_preview, 0.0, 1.0)
                )
        if recovery or landing_blend > 0:
            # A small landing bias removes the soft-task hovering equilibrium.
            # Once contact is detected the normal hard stance constraint takes over.
            blend = 1.0 if recovery else landing_blend**2 * (3.0 - 2.0 * landing_blend)
            target_position[2] += blend * (ground_z - args.landing_depth - target_position[2])
            target_velocity[2] *= 1.0 - blend
            target_acc[2] *= 1.0 - blend
            yaw = np.arctan2(target_rotation[1, 0], target_rotation[0, 0])
            flat = Rotation.from_euler("z", yaw)
            delta = flat * Rotation.from_matrix(target_rotation).inv()
            target_rotation = (
                Rotation.from_rotvec(blend * delta.as_rotvec())
                * Rotation.from_matrix(target_rotation)
            ).as_matrix()
            target_velocity[3:] *= 1.0 - blend
            target_acc[3:] *= 1.0 - blend
            if recovery:
                target_velocity[:] = 0.0
                target_acc[:] = 0.0
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
                weights[3:] = (
                    args.support_recovery_weight
                    * args.swing_orientation_weight
                    / max(args.swing_weight, 1e-8)
                )
            if recovery or landing_blend > 0:
                weights[2] += blend * (args.landing_height_weight - weights[2])
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
        args.min_support_force_ratio * model.body_subtreemass[0] * 9.81,
        args.joint_limit_horizon,
        args.joint_limit_margin,
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
    controller_state["previous_z"] = z.copy()
    return (
        z,
        result.info.iter,
        eq_residual,
        ineq_violation,
        reciprocal_condition,
        support,
        scheduled_support,
        physical_support,
        {
            "com": data.subtree_com[0].copy(),
            "com_velocity": com_velocity,
            "com_target": com_target,
            "dcm": dcm,
            "dcm_target": dcm_target,
            "com_acc_command": com_acc_command,
            "zmp_target": zmp_target if len(zmp_faces) else np.full(2, np.nan),
            "load_target": load_target,
            "com_acc_qp": com_jac @ z[:NV] + com_jdot_v,
            "com_jac": com_jac,
            "com_jdot_v": com_jdot_v,
            "zmp_faces": zmp_faces,
            "angular_momentum": angular_momentum,
        },
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
    parser.add_argument("--balance-mode", choices=("preview", "retarget"), default="preview")
    parser.add_argument(
        "--load-transfer-time",
        type=float,
        default=2.0,
        help="load-transfer window in simulation seconds",
    )
    parser.add_argument("--dcm-gain", type=float, default=3.0)
    parser.add_argument(
        "--landing-depth", type=float, default=0.002, help="landing target below ground plane in m"
    )
    parser.add_argument(
        "--landing-preview",
        type=float,
        default=0.3,
        help="flatten and lower swing foot before touchdown, simulation seconds",
    )
    parser.add_argument("--landing-height-weight", type=float, default=400.0)
    parser.add_argument(
        "--touchdown-grace",
        type=float,
        default=0.0,
        help="maximum time to retain actual old support if planned new support has not touched down",
    )
    parser.add_argument("--stance-pose", choices=("flat", "anchor"), default="flat")
    parser.add_argument(
        "--joint-limit-horizon",
        type=float,
        default=0.0,
        help="constant-acceleration joint-limit prediction horizon (s), 0 disables",
    )
    parser.add_argument("--joint-limit-margin", type=float, default=0.01)
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
    parser.add_argument("--zmp-cost", choices=("linearized", "moment"), default="linearized")
    parser.add_argument("--normal-force-weight", type=float, default=10.0)
    parser.add_argument("--min-support-force-ratio", type=float, default=0.2)
    parser.add_argument("--single-support-zmp-scale", type=float, default=1.0)
    parser.add_argument(
        "--zmp-moment-scale", type=float, default=0.05, help="ZMP residual length scale in m"
    )
    parser.add_argument(
        "--zmp-margin", type=float, default=0.005, help="inward support-polygon margin in m"
    )
    parser.add_argument("--momentum-weight", type=float, default=1000.0)
    parser.add_argument("--angular-momentum-weight", type=float, default=0.0)
    parser.add_argument("--angular-momentum-gain", type=float, default=5.0)
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
    if not 0 <= args.min_support_force_ratio < 1 or args.normal_force_weight < 0:
        parser.error(
            "minimum support force ratio must be in [0, 1), normal force weight non-negative"
        )

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    reference = load_reference(model, args.motion, args.smooth_window)
    if args.load_transfer_time <= 0 or args.dcm_gain < 0:
        parser.error("load transfer time must be positive and DCM gain non-negative")
    if args.angular_momentum_weight < 0 or args.angular_momentum_gain < 0:
        parser.error("angular momentum gains and weights must be non-negative")
    if not 0 <= args.landing_depth <= 0.01:
        parser.error("landing depth must be between 0 and 0.01 m")
    if args.landing_preview < 0 or args.touchdown_grace < 0 or args.landing_height_weight < 0:
        parser.error("landing preview and touchdown grace must be non-negative")
    if args.joint_limit_horizon < 0 or args.joint_limit_margin < 0:
        parser.error("joint limit horizon and margin must be non-negative")
    reference["touchdowns"] = [
        reference["time"][
            np.flatnonzero(reference["support"][1:, f] & ~reference["support"][:-1, f]) + 1
        ]
        for f in range(2)
    ]
    if args.balance_mode == "preview":
        reference["balance"] = BalanceReference(
            reference,
            args.speed,
            args.load_transfer_time,
            reference["com"][0, 2] - model.geom_pos[reference["ground_geom_id"], 2],
        )
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
    data.qvel[:] = args.speed * reference["qvel"][0]
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
    controller_state = {"probe": mujoco.MjData(model)}
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
            "com",
            "com_velocity",
            "com_target",
            "dcm",
            "dcm_target",
            "com_acc_command",
            "com_acc_qp",
            "com_acc_actual",
            "qacc_actual",
            "wrench_actual",
            "zmp_actual",
            "normal_force",
            "normal_force_actual",
            "contact_points",
            "tau_actual",
            "load_target",
            "zmp_violation",
            "zmp_actual_violation",
            "foot_position",
            "qp_iterations",
            "angular_momentum",
            "eq_error",
            "ineq_error",
        )
    }

    def step_once(viewer=None):
        try:
            (
                z,
                iterations,
                eq_error,
                ineq_error,
                rcond,
                support,
                scheduled,
                physical_support,
                diagnostics,
            ) = solve_reduced_qp(
                model,
                data,
                reference,
                data.time * args.speed,
                args,
                tau_limits,
                contact_anchors,
                qp_solvers,
                controller_state,
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
        ground_z = data.geom_xpos[reference["ground_geom_id"], 2]
        foot_positions = np.array(
            [data.site_xpos[reference["site_ids"][f"{side}_foot"]].copy() for side in SIDES]
        )
        zmp, force_total = resultant_zmp(wrench, foot_positions, ground_z)
        trace["zmp"].append(zmp)
        trace["zmp_target"].append(diagnostics["zmp_target"])
        for key in (
            "com",
            "com_velocity",
            "com_target",
            "dcm",
            "dcm_target",
            "com_acc_command",
            "com_acc_qp",
            "load_target",
            "angular_momentum",
        ):
            trace[key].append(diagnostics[key])
        trace["normal_force"].append(force_total[2])
        trace["foot_position"].append(foot_positions)
        trace["qp_iterations"].append(iterations)
        faces = diagnostics["zmp_faces"]

        def polygon_error(point):
            if not len(faces) or not np.isfinite(point).all():
                return np.nan
            return np.max(faces[:, :2] @ point + faces[:, 2] + args.zmp_margin)

        trace["zmp_violation"].append(polygon_error(zmp))
        trace["support"].append(support.copy())
        trace["physical_support"].append(physical_support.copy())
        trace["planned_support"].append(scheduled.copy())
        trace["eq_error"].append(eq_error)
        trace["ineq_error"].append(ineq_error)
        mujoco.mj_step(model, data)
        measured_wrench, contact_points = measured_contact_wrenches(model, data, reference)
        actual_zmp, actual_force = resultant_zmp(measured_wrench, foot_positions, ground_z)
        trace["wrench_actual"].append(measured_wrench)
        trace["zmp_actual"].append(actual_zmp)
        trace["zmp_actual_violation"].append(polygon_error(actual_zmp))
        trace["normal_force_actual"].append(actual_force[2])
        trace["contact_points"].append(contact_points)
        trace["qacc_actual"].append(data.qacc.copy())
        trace["com_acc_actual"].append(
            diagnostics["com_jac"] @ data.qacc + diagnostics["com_jdot_v"]
        )
        trace["tau_actual"].append(data.qfrc_actuator[model.jnt_dofadr[actuator_joint_ids]].copy())
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
            configuration=np.array(json.dumps(vars(args), default=str)),
            succeeded=np.array(state["failed"] is None and data.time >= duration),
            failure=np.array(state["failed"] or ""),
            simulation_dt=np.array(model.opt.timestep),
            simulated_duration=np.array(data.time),
            requested_duration=np.array(duration),
        )
        print(f"trace={args.output}")
    if state["failed"]:
        raise RuntimeError(state["failed"])


if __name__ == "__main__":
    main()
