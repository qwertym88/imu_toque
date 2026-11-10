#!/usr/bin/env python3
"""Inspect the derived human model and optionally render a neutral-pose preview."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "model/ms_human_locomotion/human.xml"


def inspect_model(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("standing").id)
    mujoco.mj_forward(model, data)
    mass = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, mass, data.qM)
    bodies = []
    for b in range(1, model.nbody):
        inertia = model.body_inertia[b]
        if model.body_mass[b] <= 0 or np.any(inertia <= 0):
            raise ValueError(f"Nonpositive inertial parameters on {model.body(b).name}")
        if 2 * inertia.max() > inertia.sum() + 1e-10:
            raise ValueError(f"Inertia triangle inequality failed on {model.body(b).name}")
        bodies.append(
            {
                "name": model.body(b).name,
                "mass_kg": float(model.body_mass[b]),
                "com_world_m": data.xipos[b].tolist(),
                "principal_inertia_kg_m2": inertia.tolist(),
            }
        )
    eigenvalues = np.linalg.eigvalsh(mass)
    if eigenvalues[0] <= 0:
        raise ValueError("Mass matrix is not positive definite")
    soles = {}
    for side in ("left", "right"):
        g = model.geom(f"{side}_sole").id
        bottom = (
            data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
        )
        soles[side] = {
            "bottom_z_m": float(bottom),
            "site_world_m": data.site_xpos[model.site(f"{side}_foot").id].tolist(),
        }
    report = {
        "mujoco_version": mujoco.__version__,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "moving_bodies": model.nbody - 1,
        "neq": model.neq,
        "ntendon": model.ntendon,
        "activation_states": model.na,
        "total_mass_kg": float(model.body_mass.sum()),
        "mass_matrix_shape": list(mass.shape),
        "mass_matrix_min_eigenvalue": float(eigenvalues[0]),
        "mass_matrix_symmetry_error": float(np.max(np.abs(mass - mass.T))),
        "passive_force_max_abs": float(np.max(np.abs(data.qfrc_passive))),
        "whole_body_com_world_m": data.subtree_com[model.body("pelvis").id].tolist(),
        "soles": soles,
        "bodies": bodies,
        "joint_order": [],
        "actuator_order": [],
    }
    for j in range(1, model.njnt):
        report["joint_order"].append(
            {
                "name": model.joint(j).name,
                "qpos_address": int(model.jnt_qposadr[j]),
                "dof_address": int(model.jnt_dofadr[j]),
                "axis_body": model.jnt_axis[j].tolist(),
                "range_rad": model.jnt_range[j].tolist() if model.jnt_limited[j] else None,
            }
        )
    for a in range(model.nu):
        report["actuator_order"].append(
            {
                "name": model.actuator(a).name,
                "joint": model.joint(int(model.actuator_trnid[a, 0])).name,
            }
        )
    return report, data


def render_preview(model, data, destination):
    from PIL import Image

    # Must be called with MUJOCO_GL=egl or osmesa on headless machines.
    with mujoco.Renderer(model, height=640, width=960) as renderer:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = [-0.02, 0, 0.9]
        camera.distance = 2.8
        camera.elevation = -12
        images = []
        for azimuth in (90, 0):
            camera.azimuth = azimuth
            renderer.update_scene(data, camera)
            images.append(renderer.render().copy())
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(images, axis=1)).save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", type=Path)
    parser.add_argument(
        "--viewer", action="store_true", help="Open a static pose viewer; no dynamics step"
    )
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    report, data = inspect_model(model)
    summary = {
        k: report[k]
        for k in (
            "nq",
            "nv",
            "nu",
            "moving_bodies",
            "total_mass_kg",
            "mass_matrix_min_eigenvalue",
            "passive_force_max_abs",
        )
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.preview:
        render_preview(model, data, args.preview)
        print(f"Preview: {args.preview}")
    if args.viewer:
        import mujoco.viewer as viewer

        import time

        with viewer.launch_passive(model, data) as handle:
            handle.cam.lookat[:] = [-0.02, 0, 0.9]
            handle.cam.distance = 2.8
            handle.cam.azimuth = 90
            handle.cam.elevation = -12
            while handle.is_running():
                handle.sync()
                time.sleep(1 / 60)


if __name__ == "__main__":
    main()
