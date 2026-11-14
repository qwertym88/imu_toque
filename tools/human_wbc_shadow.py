#!/usr/bin/env python3
"""Audit an Xsens contact candidate with a condensed human inverse-dynamics QP.

This is a kinematic-state shadow evaluation: it never advances MuJoCo or applies
torque to the human. Contact labels and all forces remain diagnostic candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import osqp
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
NV, NLAMBDA, NTAU = 21, 12, 15
NZ = NV + NLAMBDA + NTAU
LAMBDA = slice(NV, NV + NLAMBDA)
TAU = slice(NV + NLAMBDA, NZ)


def load_inputs(model_path, fit_path, contact_path):
    model_path, fit_path, contact_path = map(Path, (model_path, fit_path, contact_path))
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    with (
        np.load(fit_path, allow_pickle=False) as fit,
        np.load(contact_path, allow_pickle=False) as contacts,
    ):
        fit_meta = json.loads(str(fit["metadata_json"]))
        contact_meta = json.loads(str(contacts["metadata_json"]))
        if fit_meta["model_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("Fit/model SHA-256 mismatch")
        if contact_meta["matched_fit_model_sha256"] != fit_meta["model_sha256"]:
            raise ValueError("Contact inference refers to another fitted model")
        if contact_meta["matched_xsens_sha256"] != fit_meta["source_xlsx_sha256"]:
            raise ValueError("Contact inference and fit use different Xsens samples")
        if (
            contact_meta.get("sample_id") != "y13"
            or contact_meta.get("force_plate_used") is not False
        ):
            raise ValueError(
                "The shadow QP only accepts Y13 contact inference without a force plate"
            )
        if model.nv != NV or model.nu != NTAU or model.nq != 22:
            raise ValueError(
                f"Expected human model nq/nv/nu=22/21/15, got {model.nq}/{model.nv}/{model.nu}"
            )
        return (
            model,
            {key: fit[key].copy() for key in fit.files},
            {key: contacts[key].copy() for key in contacts.files},
            fit_meta,
            contact_meta,
        )


def model_ids(model):
    result = {"sites": {}, "geoms": {}, "dofs": []}
    for side in SIDES:
        result["sites"][side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_foot")
        result["geoms"][side] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
        if result["sites"][side] < 0 or result["geoms"][side] < 0:
            raise ValueError(f"Human model lacks {side} sole site/geom")
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT or joint <= 0:
            raise ValueError("Expected one joint motor for each of the 15 human hinges")
        result["dofs"].append(int(model.jnt_dofadr[joint]))
    return result


def task_kinematics(model, data, site_id):
    jp, jr = np.zeros((3, NV)), np.zeros((3, NV))
    mujoco.mj_jacSite(model, data, jp, jr, site_id)
    site = data.site_xpos[site_id].copy()
    body = int(model.site_bodyid[site_id])
    jdp, jdr = np.zeros((3, NV)), np.zeros((3, NV))
    mujoco.mj_jacDot(model, data, jdp, jdr, site, body)
    return np.vstack((jp, jr)), np.r_[jdp @ data.qvel, jdr @ data.qvel]


def assemble_qp(model, data, support, qacc_reference, ids, args):
    mass = np.zeros((NV, NV))
    mujoco.mj_fullM(model, mass, data.qM)
    bias = data.qfrc_bias[:NV].copy()
    actuation = np.zeros((NV, NTAU))
    for actuator, dof in enumerate(ids["dofs"]):
        actuation[dof, actuator] = data.actuator_moment[actuator]

    jacobians, jdotv = [], []
    for side in SIDES:
        jac, bias_acc = task_kinematics(model, data, ids["sites"][side])
        jacobians.append(jac)
        jdotv.append(bias_acc)

    equality = np.zeros((NV + 12, NZ))
    target = np.zeros(NV + 12)
    equality[:NV, :NV] = mass
    equality[:NV, TAU] = -actuation
    target[:NV] = -bias
    for side_index in range(2):
        wrench_slice = slice(NV + 6 * side_index, NV + 6 * (side_index + 1))
        task_row = NV + 6 * side_index
        if support[side_index]:
            equality[:NV, wrench_slice] = -jacobians[side_index].T
            equality[task_row : task_row + 6, :NV] = jacobians[side_index]
            target[task_row : task_row + 6] = -jdotv[side_index]
        else:
            equality[task_row : task_row + 6, wrench_slice] = np.eye(6)

    hessian = np.zeros((NZ, NZ))
    # Track the fitted acceleration while allowing the QP to adjust it for contacts.
    acceleration_weight = np.r_[np.full(6, args.base_accel_weight), np.ones(NV - 6)]
    hessian[:NV, :NV] = np.diag(acceleration_weight)
    hessian[LAMBDA, LAMBDA] = np.eye(NLAMBDA) * args.wrench_regularization
    hessian[TAU, TAU] = np.eye(NTAU) * args.torque_regularization
    gradient = np.zeros(NZ)
    gradient[:NV] = -acceleration_weight * qacc_reference

    rows, lower, upper = [], [], []
    rows.extend(equality)
    lower.extend(target)
    upper.extend(target)
    tau_limits = np.asarray(args.torque_limits, dtype=float)
    if tau_limits.size == 1:
        tau_limits = np.full(NTAU, tau_limits.item())
    if tau_limits.shape != (NTAU,) or np.any(tau_limits <= 0):
        raise ValueError("--torque-limits must provide one positive value or exactly 15 values")
    for motor in range(NTAU):
        row = np.zeros(NZ)
        row[TAU.start + motor] = 1
        rows.append(row)
        lower.append(-tau_limits[motor])
        upper.append(tau_limits[motor])

    friction = np.asarray(
        [
            [1, 0, -args.friction, 0, 0, 0],
            [-1, 0, -args.friction, 0, 0, 0],
            [0, 1, -args.friction, 0, 0, 0],
            [0, -1, -args.friction, 0, 0, 0],
            [0, 0, -1, 0, 0, 0],
        ],
        dtype=float,
    )
    for side_index, side in enumerate(SIDES):
        wrench_slice = slice(NV + 6 * side_index, NV + 6 * (side_index + 1))
        rotation = data.site_xmat[ids["sites"][side]].reshape(3, 3)
        world_to_foot = np.zeros((6, 6))
        world_to_foot[:3, :3] = rotation.T
        world_to_foot[3:, 3:] = rotation.T
        facets = []
        if support[side_index]:
            facets.extend(friction)
            # Fz >= 0 is facet 5. CoP and torsional friction are expressed
            # in the sole site axes; no global ground-height assumption enters.
            facets.extend(
                [
                    [0, 0, -args.cop_half_width, 1, 0, 0],
                    [0, 0, -args.cop_half_width, -1, 0, 0],
                    [0, 0, -args.cop_half_length, 0, 1, 0],
                    [0, 0, -args.cop_half_length, 0, -1, 0],
                    [0, 0, -args.torsion_friction, 0, 0, 1],
                    [0, 0, -args.torsion_friction, 0, 0, -1],
                ]
            )
        for facet in facets:
            row = np.zeros(NZ)
            row[wrench_slice] = np.asarray(facet) @ world_to_foot
            rows.append(row)
            lower.append(-np.inf)
            upper.append(0.0)

    matrix = np.asarray(rows)
    return (
        sparse.csc_matrix(0.5 * (hessian + hessian.T)),
        gradient,
        sparse.csc_matrix(matrix),
        np.asarray(lower),
        np.asarray(upper),
        equality,
        target,
        jacobians,
    )


def solve_frame(model, data, support, qacc_reference, ids, args, solver=None):
    hessian, gradient, matrix, lower, upper, equality, target, jacobians = assemble_qp(
        model, data, support, qacc_reference, ids, args
    )
    if solver is None:
        solver = osqp.OSQP()
        solver.setup(
            P=hessian,
            q=gradient,
            A=matrix,
            l=lower,
            u=upper,
            verbose=False,
            polishing=True,
            max_iter=args.max_iter,
            eps_abs=args.tolerance,
            eps_rel=args.tolerance,
            warm_starting=True,
        )
    else:
        solver.update(Px=hessian.data, q=gradient, Ax=matrix.data, l=lower, u=upper)
    result = solver.solve()
    output = {
        "status": result.info.status,
        "iterations": int(result.info.iter),
        "primal_residual": float(result.info.prim_res),
        "dual_residual": float(result.info.dual_res),
    }
    if result.x is None or result.info.status_val not in (1, 2):
        return output, solver
    z = result.x
    output.update(
        {
            "qacc": z[:NV],
            "wrenches": z[LAMBDA].reshape(2, 6),
            "torques": z[TAU],
            "equality_residual": float(np.max(np.abs(equality @ z - target))),
            "support_accel_residual": np.asarray(
                [
                    np.linalg.norm(
                        jacobians[i] @ z[:NV] + task_kinematics(model, data, ids["sites"][side])[1]
                    )
                    for i, side in enumerate(SIDES)
                    if support[i]
                ]
            ),
            "solved": True,
        }
    )
    return output, solver


def run_shadow(args):
    model, fit, contacts, fit_meta, contact_meta = load_inputs(args.model, args.fit, args.contacts)
    if not np.isclose(contact_meta["output_frame_rate_hz"], fit["frame_rate"].item()):
        raise ValueError("Contact and fit sample rates differ")
    times = fit["time_s"]
    selection = (times >= args.start) & (times <= args.start + args.duration)
    if not selection.any():
        raise ValueError("Requested shadow interval lies outside fitted motion")
    validity = fit["valid"] & fit["derivative_valid"] & contacts["valid"]
    if not validity[selection].all():
        raise ValueError("Shadow interval crosses an invalid/reset interval")
    model.opt.timestep = args.timestep
    ids = model_ids(model)
    data = mujoco.MjData(model)
    rows = []
    for index in np.flatnonzero(selection):
        support = contacts["support_mask_candidate"][index].astype(bool)
        data.qpos[:] = fit["qpos"][index]
        data.qvel[:] = fit["qvel"][index]
        mujoco.mj_forward(model, data)
        geometry_overlap = np.zeros(2, dtype=bool)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for contact_index in range(data.ncon):
            pair = {
                int(data.contact[contact_index].geom1),
                int(data.contact[contact_index].geom2),
            }
            for side_index, side in enumerate(SIDES):
                if floor >= 0 and pair == {floor, ids["geoms"][side]}:
                    geometry_overlap[side_index] = True
        if support.sum() == 0:
            rows.append(
                {
                    "frame": int(index),
                    "time_s": float(times[index]),
                    "solved": False,
                    "status": "skipped_no_confident_contact",
                    "support_left_candidate": bool(support[0]),
                    "support_right_candidate": bool(support[1]),
                    "mujoco_geometry_contact_left": bool(geometry_overlap[0]),
                    "mujoco_geometry_contact_right": bool(geometry_overlap[1]),
                }
            )
            continue
        # Recreate the small OSQP workspace per frame. The active contact rows
        # change with the mode, so reusing a sparsity-frozen solver is unsafe.
        result, _ = solve_frame(model, data, support, fit["qacc"][index], ids, args)
        result.update(
            {
                "frame": int(index),
                "time_s": float(times[index]),
                "support_left_candidate": bool(support[0]),
                "support_right_candidate": bool(support[1]),
                "contact_score_left_uncalibrated": float(contacts["contact_probability"][index, 0]),
                "contact_score_right_uncalibrated": float(
                    contacts["contact_probability"][index, 1]
                ),
                "mujoco_geometry_contact_left": bool(geometry_overlap[0]),
                "mujoco_geometry_contact_right": bool(geometry_overlap[1]),
            }
        )
        rows.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    solved = [row for row in rows if row.get("solved")]
    contact_checked = [row for row in rows if "mujoco_geometry_contact_left" in row]
    candidate_contact = np.asarray(
        [
            [row["support_left_candidate"], row["support_right_candidate"]]
            for row in contact_checked
        ],
        dtype=bool,
    )
    geometry_contact = np.asarray(
        [
            [row["mujoco_geometry_contact_left"], row["mujoco_geometry_contact_right"]]
            for row in contact_checked
        ],
        dtype=bool,
    )
    per_side_overlap = {}
    for side_index, side in enumerate(SIDES):
        predicted = candidate_contact[:, side_index]
        reference = geometry_contact[:, side_index]
        true_positive = int(np.sum(predicted & reference))
        false_positive = int(np.sum(predicted & ~reference))
        false_negative = int(np.sum(~predicted & reference))
        per_side_overlap[side] = {
            "candidate_positive_frames": int(predicted.sum()),
            "geometry_overlap_frames": int(reference.sum()),
            "agreement": float(np.mean(predicted == reference)),
            "precision_against_geometry_overlap": float(
                true_positive / max(true_positive + false_positive, 1)
            ),
            "recall_against_geometry_overlap": float(
                true_positive / max(true_positive + false_negative, 1)
            ),
        }
    report = {
        "sample_id": "y13",
        "force_plate_used": False,
        "model_sha256": fit_meta["model_sha256"],
        "contact_source_sha256": contact_meta["source_xlsx_sha256"],
        "start_s": args.start,
        "duration_s": args.duration,
        "frames_requested": len(rows),
        "frames_solved": len(solved),
        "solve_fraction": float(len(solved) / max(len(rows), 1)),
        "solver_status_counts": {
            status: sum(row.get("status") == status for row in rows)
            for status in sorted({row.get("status", "unknown") for row in rows})
        },
        "max_equality_residual": max(
            (row.get("equality_residual", np.nan) for row in solved), default=np.nan
        ),
        "max_support_acceleration_residual": max(
            (
                np.max(row["support_accel_residual"])
                for row in solved
                if len(row["support_accel_residual"])
            ),
            default=np.nan,
        ),
        "candidate_vs_mujoco_geometry_contact_agreement": (
            float(np.mean(candidate_contact == geometry_contact))
            if len(contact_checked)
            else np.nan
        ),
        "per_side_candidate_vs_geometry_overlap": per_side_overlap,
        "candidate_contact_without_geom_overlap_fraction": (
            float(np.mean(candidate_contact & ~geometry_contact))
            if len(contact_checked)
            else np.nan
        ),
        "torque_abs_p95_peak_Nm": (
            np.percentile(np.abs(np.stack([row["torques"] for row in solved])), [95, 100]).tolist()
            if solved
            else [np.nan, np.nan]
        ),
        "meaning": "offline inverse-dynamics shadow QP on fitted Xsens states; no forward simulation; forces/torques are model predictions, not measured human loads",
        "contact_warning": "MuJoCo geometry overlap is only a pose-based proxy, not measured contact ground truth; HMM states remain unlabeled candidates",
    }
    numeric = {
        key: np.asarray([row.get(key, np.nan) for row in rows])
        for key in (
            "frame",
            "time_s",
            "solved",
            "equality_residual",
            "primal_residual",
            "dual_residual",
            "iterations",
        )
    }
    numeric.update(
        {
            "wrenches_world_about_foot_sites": np.stack(
                [row.get("wrenches", np.full((2, 6), np.nan)) for row in rows]
            ),
            "joint_torque_Nm": np.stack(
                [row.get("torques", np.full(NTAU, np.nan)) for row in rows]
            ),
            "qacc_solution": np.stack([row.get("qacc", np.full(NV, np.nan)) for row in rows]),
            "support_mask_candidate": np.asarray(
                [
                    [
                        row.get("support_left_candidate", False),
                        row.get("support_right_candidate", False),
                    ]
                    for row in rows
                ]
            ),
            "mujoco_geometry_contact": np.asarray(
                [
                    [
                        row.get("mujoco_geometry_contact_left", False),
                        row.get("mujoco_geometry_contact_right", False),
                    ]
                    for row in rows
                ]
            ),
            "status": np.asarray([row.get("status", "skipped") for row in rows]),
        }
    )
    np.savez_compressed(output, **numeric, metadata_json=np.asarray(json.dumps(report)))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/y13/human.xml")
    parser.add_argument("--fit", type=Path, default=ROOT / "outputs/y13/fit.npz")
    parser.add_argument(
        "--contacts", type=Path, default=ROOT / "outputs/y13/contact/contact_inference.npz"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13/wbc_shadow.npz")
    parser.add_argument("--start", type=float, default=25.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--cop-half-length", type=float, default=0.13)
    parser.add_argument("--cop-half-width", type=float, default=0.045)
    parser.add_argument("--torsion-friction", type=float, default=0.02)
    parser.add_argument("--base-accel-weight", type=float, default=0.05)
    parser.add_argument("--wrench-regularization", type=float, default=1e-8)
    parser.add_argument("--torque-regularization", type=float, default=1e-6)
    parser.add_argument(
        "--torque-limits",
        type=float,
        nargs="+",
        default=[1500.0],
        help="Diagnostic numeric bound, not measured human strength",
    )
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.start < 0 or args.duration <= 0 or args.timestep <= 0 or args.friction <= 0:
        parser.error("time, timestep and friction values must be positive")
    run_shadow(args)


if __name__ == "__main__":
    main()
