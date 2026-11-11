#!/usr/bin/env python3
"""Derive a muscle-free MJCF from the pinned MS-Human locomotion source subset."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model/ms_human_locomotion"
SOURCE = MODEL_DIR / "source"
SIDES = (("right", "r"), ("left", "l"))
# Names are semantic aliases. The oblique anatomical axes are NOT orthogonalized.
LEG_JOINTS = (
    ("hip_pitch", "hip_flexion"),
    ("hip_roll", "hip_adduction"),
    ("hip_yaw", "hip_rotation"),
    ("knee_pitch", "knee_angle"),
    ("ankle_pitch", "ankle_angle"),
    ("ankle_roll", "subtalar_angle"),
)
SOLE_CENTER = np.array([0.08, -0.015, 0.0])
SOLE_SIZE = np.array([0.13, 0.01, 0.045])


def values(array):
    return " ".join(f"{v:.15g}" for v in np.asarray(array).ravel())


def quaternion(matrix):
    result = np.empty(4)
    mujoco.mju_mat2Quat(result, np.asarray(matrix).ravel())
    return result


def verify_source(source=SOURCE):
    manifest = json.loads((source / "manifest.json").read_text())
    for rel, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256((source / rel).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source checksum mismatch: {rel}")
    return manifest


def expand_includes(element, source):
    """Upstream include paths are all relative to the model root."""
    for child in list(element):
        if child.tag == "include":
            fragment = ET.parse(source / child.attrib["file"]).getroot()
            expand_includes(fragment, source)
            index = list(element).index(child)
            element.remove(child)
            for offset, node in enumerate(list(fragment)):
                element.insert(index + offset, node)
        else:
            expand_includes(child, source)


def load_source_reference(source=SOURCE):
    """Compile inertial/kinematic source, preserving mesh-inferred rib masses.

    Explicit inertias override geometry inference on every other body. Remove
    muscles and visual/collision geoms, retain the original joints, defaults and
    equalities. This reference is for deriving/comparing mechanics, not control.
    """
    verify_source(source)
    original = ET.parse(source / "MS-Human-700-Locomotion.xml").getroot()
    root = ET.Element("mujoco", model="ms_human_inertial_reference")
    for tag in ("compiler", "default"):
        root.append(copy.deepcopy(original.find(tag)))
    root.find("compiler").set("fusestatic", "false")
    world = copy.deepcopy(original.find("worldbody"))
    expand_includes(world, source)
    needed_meshes = set()
    for parent in list(world.iter()):
        for child in list(parent):
            keep_geom = (
                child.tag == "geom" and parent.tag == "body" and parent.find("inertial") is None
            )
            if keep_geom:
                needed_meshes.add(child.attrib["mesh"])
            elif child.tag not in ("body", "joint", "inertial"):
                parent.remove(child)
    asset = ET.SubElement(root, "asset")
    for mesh in ET.parse(source / "Asset/Asset_Torso.xml").getroot().iter("mesh"):
        if mesh.get("name") in needed_meshes:
            mesh = copy.deepcopy(mesh)
            mesh.set("file", str((source / mesh.attrib["file"]).resolve()))
            asset.append(mesh)
    if len(asset) != len(needed_meshes):
        raise ValueError("Missing inferred-inertia rib meshes")
    root.append(world)
    root.append(
        copy.deepcopy(
            ET.parse(source / "Equality/Equality_Lowerbody.xml").getroot().find("equality")
        )
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    # Neutral retained angles, with dependent coordinates at their q_knee=0
    # polynomial values rather than violating the upstream knee equalities.
    for eq in range(model.neq):
        joint = int(model.eq_obj1id[eq])
        data.qpos[model.jnt_qposadr[joint]] = model.eq_data[eq, 0]
    mujoco.mj_forward(model, data)
    # Place our approximate sole boxes on world z=0; do not scale the human.
    minimum = np.inf
    for _, suffix in SIDES:
        foot = model.body(f"calcn_{suffix}").id
        rotation = data.xmat[foot].reshape(3, 3)
        center = data.xpos[foot] + rotation @ SOLE_CENTER
        minimum = min(minimum, center[2] - np.abs(rotation[2]) @ SOLE_SIZE)
    data.qpos[model.jnt_qposadr[model.joint("pelvis_ty").id]] -= minimum
    mujoco.mj_forward(model, data)
    return model, data


def rotation(data, body):
    return data.xmat[body].reshape(3, 3)


def aggregate_inertia(model, data, members, position, orientation):
    masses = model.body_mass[members]
    centers = (data.xipos[members] - position) @ orientation
    mass = float(masses.sum())
    center = np.sum(masses[:, None] * centers, axis=0) / mass
    tensor = np.zeros((3, 3))
    for body, body_mass, com in zip(members, masses, centers):
        inertial_rotation = orientation.T @ data.ximat[body].reshape(3, 3)
        tensor += inertial_rotation @ np.diag(model.body_inertia[body]) @ inertial_rotation.T
        delta = com - center
        tensor += body_mass * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return mass, center, (tensor + tensor.T) / 2


def body_groups(model):
    groups = {"pelvis": [model.body(n).id for n in ("pelvis", "sacrum", "Abdomen")]}
    used = set(groups["pelvis"])
    for side, suffix in SIDES:
        for label, names in (
            ("thigh", [f"femur_{suffix}", f"patella_{suffix}"]),
            ("shank", [f"tibia_{suffix}"]),
            ("talus", [f"talus_{suffix}"]),
            ("foot", [f"calcn_{suffix}", f"toes_{suffix}"]),
        ):
            groups[f"{side}_{label}"] = [model.body(n).id for n in names]
            used.update(groups[f"{side}_{label}"])
    groups["torso"] = [b for b in range(1, model.nbody) if b not in used]
    return groups


def derive(source=SOURCE):
    manifest = verify_source(source)
    model, data = load_source_reference(source)
    groups = body_groups(model)
    root = ET.Element("mujoco", model="ms_human_locomotion_net_torque")
    root.append(
        ET.Comment(
            " Derived from MS-Human-700, Apache-2.0; see source/LICENSE and README.md. "
            "Generated by tools/build_ms_human.py. No muscle or joint strength model. "
        )
    )
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default,
        "joint",
        type="hinge",
        limited="true",
        damping="0",
        armature="0",
        stiffness="0",
        frictionloss="0",
    )
    ET.SubElement(default, "geom", contype="0", conaffinity="0", group="1")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.7 0.7 0.7")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="3 3 0.1",
        rgba="0.88 0.9 0.92 1",
        contype="1",
        conaffinity="1",
        group="0",
    )
    ET.SubElement(world, "light", pos="1 -2 4", dir="-0.2 0.2 -1")
    anchors = {"pelvis": "pelvis", "torso": "lumbar5"}
    parents = {"pelvis": None, "torso": "pelvis"}
    for side, suffix in SIDES:
        for label, src, parent in (
            ("thigh", "femur", "pelvis"),
            ("shank", "tibia", f"{side}_thigh"),
            ("talus", "talus", f"{side}_shank"),
            ("foot", "calcn", f"{side}_talus"),
        ):
            anchors[f"{side}_{label}"] = f"{src}_{suffix}"
            parents[f"{side}_{label}"] = parent
    poses = {
        name: (data.xpos[model.body(src).id].copy(), rotation(data, model.body(src).id).copy())
        for name, src in anchors.items()
    }
    elements = {}
    joint_info = []
    group_info = {}
    for name, members in groups.items():
        position, orientation = poses[name]
        parent_name = parents[name]
        if parent_name is None:
            local_position, local_rotation = position, orientation
            parent = world
        else:
            ppos, prot = poses[parent_name]
            local_position = prot.T @ (position - ppos)
            local_rotation = prot.T @ orientation
            parent = elements[parent_name]
        body = ET.SubElement(
            parent,
            "body",
            name=name,
            pos=values(local_position),
            quat=values(quaternion(local_rotation)),
        )
        elements[name] = body
        mass, center, tensor = aggregate_inertia(model, data, members, position, orientation)
        ET.SubElement(
            body,
            "inertial",
            mass=values([mass]),
            pos=values(center),
            fullinertia=values(tensor[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
        )
        group_info[name] = {
            "source_bodies": [model.body(b).name for b in members],
            "mass_kg": mass,
            "com_local_m": center.tolist(),
            "inertia_local_kg_m2": tensor.tolist(),
        }
        if name == "pelvis":
            ET.SubElement(body, "freejoint", name="pelvis_free")
        elif name == "torso":
            # Added three-axis lumped torso articulation about the lumbar5 origin.
            for label, axis in (("pitch", [0, 0, 1]), ("roll", [1, 0, 0]), ("yaw", [0, 1, 0])):
                local_axis = orientation.T @ poses["pelvis"][1] @ np.array(axis)
                jname = f"torso_{label}"
                ET.SubElement(body, "joint", name=jname, axis=values(local_axis), limited="false")
                joint_info.append(
                    {
                        "name": jname,
                        "source_joint": None,
                        "axis_body": local_axis.tolist(),
                        "range_rad": None,
                        "note": "Added lumped torso coordinate; no strength limits",
                    }
                )
        else:
            side, part = name.split("_", 1)
            suffix = dict(SIDES)[side]
            selected = {
                "thigh": LEG_JOINTS[:3],
                "shank": LEG_JOINTS[3:4],
                "talus": LEG_JOINTS[4:5],
                "foot": LEG_JOINTS[5:],
            }[part]
            for alias, source_joint in selected:
                joint_id = model.joint(f"{source_joint}_{suffix}").id
                axis = orientation.T @ data.xaxis[joint_id]
                anchor = orientation.T @ (data.xanchor[joint_id] - position)
                jname = f"{side}_{alias}"
                ET.SubElement(
                    body,
                    "joint",
                    name=jname,
                    pos=values(anchor),
                    axis=values(axis),
                    range=values(model.jnt_range[joint_id]),
                )
                joint_info.append(
                    {
                        "name": jname,
                        "source_joint": model.joint(joint_id).name,
                        "axis_body": axis.tolist(),
                        "anchor_body_m": anchor.tolist(),
                        "range_rad": model.jnt_range[joint_id].tolist(),
                    }
                )
        # Display geometry has no role in mass/inertia; only soles collide.
        if name == "pelvis":
            ET.SubElement(
                body,
                "geom",
                name="pelvis_visual",
                type="box",
                pos="-0.04 0 0",
                size="0.09 0.055 0.1",
                rgba="0.35 0.4 0.48 1",
            )
        elif name == "torso":
            head = model.body("head_neck").id
            neck = data.xpos[head] - np.array([0, 0, 0.025])
            endpoint = orientation.T @ (neck - position)
            head_center = orientation.T @ (data.xipos[head] - position)
            ET.SubElement(
                body,
                "geom",
                name="torso_visual",
                type="capsule",
                fromto=values(np.r_[np.zeros(3), endpoint]),
                size="0.105",
                rgba="0.4 0.5 0.63 1",
            )
            ET.SubElement(
                body,
                "geom",
                name="head_visual",
                type="sphere",
                pos=values(head_center),
                size="0.09",
                rgba="0.7 0.68 0.6 1",
            )
            for suffix in ("r", "l"):
                arm_points = [
                    orientation.T @ (data.xpos[model.body(n).id] - position)
                    for n in (f"humerus_{suffix}", f"ulna_{suffix}", f"hand_{suffix}")
                ]
                for index in range(2):
                    ET.SubElement(
                        body,
                        "geom",
                        type="capsule",
                        size="0.035",
                        fromto=values(np.r_[arm_points[index], arm_points[index + 1]]),
                        rgba="0.5 0.6 0.7 1",
                    )
        elif name.endswith(("thigh", "shank")):
            side, part = name.split("_", 1)
            next_body = f"{side}_{'shank' if part == 'thigh' else 'talus'}"
            endpoint = orientation.T @ (poses[next_body][0] - position)
            ET.SubElement(
                body,
                "geom",
                type="capsule",
                size="0.045" if part == "thigh" else "0.03",
                fromto=values(np.r_[np.zeros(3), endpoint]),
                rgba="0.3 0.5 0.8 1" if side == "right" else "0.85 0.4 0.25 1",
            )
        elif name.endswith("talus"):
            ET.SubElement(body, "geom", type="sphere", size="0.022", rgba="0.5 0.5 0.5 1")
        elif name.endswith("foot"):
            side = name.split("_", 1)[0]
            ET.SubElement(
                body,
                "geom",
                name=f"{side}_sole",
                type="box",
                pos=values(SOLE_CENTER),
                size=values(SOLE_SIZE),
                contype="1",
                conaffinity="1",
                condim="3",
                friction="0.8 0.005 0.0001",
                rgba="0.22 0.26 0.3 1",
            )
            sitepos = SOLE_CENTER - np.array([0, SOLE_SIZE[1], 0])
            # Wrench frame: world-aligned in neutral, moving with the foot later.
            ET.SubElement(
                body,
                "site",
                name=f"{side}_foot",
                pos=values(sitepos),
                quat=values(quaternion(orientation.T)),
                size="0.008",
                rgba="0 0.8 0.2 1",
            )
            for label, offset in (("heel", -SOLE_SIZE[0]), ("toe", SOLE_SIZE[0])):
                point = sitepos + np.array([offset, 0, 0])
                ET.SubElement(body, "site", name=f"{side}_{label}", pos=values(point), size="0.005")
    actuators = ET.SubElement(root, "actuator")
    for joint in joint_info:
        ET.SubElement(
            actuators,
            "motor",
            name=joint["name"] + "_torque",
            joint=joint["name"],
            gear="1",
            ctrllimited="false",
            forcelimited="false",
        )
    # Resolve qpos order through MuJoCo, which traverses the body tree.
    compiled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    keyframes = ET.SubElement(root, "keyframe")
    ET.SubElement(keyframes, "key", name="standing", qpos=values(compiled.qpos0))
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode") + "\n"
    report = {
        "upstream_commit": manifest["commit"],
        "generator_mujoco_version": mujoco.__version__,
        "world_frame": "x forward, y left, z up; metres, radians, kg, N, Nm",
        "source_mass_kg": float(model.body_mass.sum()),
        "reference_source_qpos": data.qpos.tolist(),
        "neutral_pose_note": "Retained angles zero; dependent knee coordinates at polynomial constants; soles translated to z=0",
        "groups": group_info,
        "joints": joint_info,
        "joint_qpos_order": [compiled.joint(j).name for j in range(1, compiled.njnt)],
        "actuator_order": [j["name"] for j in joint_info],
        "sole_half_length_width_m": [float(SOLE_SIZE[0]), float(SOLE_SIZE[2])],
        "simplifications": [
            "Pelvis+sacrum+Abdomen merged; upper spine+head+arms merged into torso",
            "Three torso coordinates added at lumbar5; upper internal posture fixed",
            "Patella locked to femur and toes locked to calcaneus at neutral",
            "Knee secondary translations/rotations frozen at neutral polynomial values",
            "Original ankle and subtalar oblique axes and offset preserved",
            "No muscles, tendons, equalities, damping, springs, frictionloss or armature",
            "Primitive visuals and approximate sole contacts; no human strength limits",
            "Unscaled source subject; participant adaptations are saved separately",
        ],
    }
    return xml, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=MODEL_DIR / "human.xml")
    parser.add_argument("--report", type=Path, default=MODEL_DIR / "derivation.json")
    args = parser.parse_args()
    xml, report = derive(args.source.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(xml)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    model = mujoco.MjModel.from_xml_path(str(args.output.resolve()))
    print(
        f"Wrote {args.output}: nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"mass={model.body_mass.sum():.6f} kg"
    )


if __name__ == "__main__":
    main()
