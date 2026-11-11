#!/usr/bin/env python3
"""Scale the human model's leg lengths to a new Xsens trial; keep mass explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from fit_xsens_human import DEFAULT_INPUT, TEMPLATE_MODEL, load_xsens

ROOT = Path(__file__).resolve().parents[1]


def scale_vectors(element, attributes, factor):
    for key in attributes:
        if key in element.attrib:
            element.set(
                key,
                " ".join(f"{x:.15g}" for x in np.fromstring(element.get(key), sep=" ") * factor),
            )


def prepare(input_path, base_model, output, mass_kg=None, height_m=None):
    frames, rate, quats, positions, metadata = load_xsens(input_path)
    # XLSX segment origins are hip, knee and ankle, not IMU mounting locations.
    source = mujoco.MjModel.from_xml_path(str(base_model.resolve()))
    data = mujoco.MjData(source)
    mujoco.mj_forward(source, data)
    lengths = {}
    factors = {}
    for side, slots in (("right", (1, 2, 3)), ("left", (4, 5, 6))):
        for part, first, second in (("thigh", slots[0], slots[1]), ("shank", slots[1], slots[2])):
            observed = np.linalg.norm(positions[:, second] - positions[:, first], axis=1)
            target_length = float(np.median(observed[np.isfinite(observed)]))
            first_body = source.body(f"{side}_{part}").id
            second_body = source.body(f'{side}_{"shank" if part == "thigh" else "talus"}').id
            base_length = float(np.linalg.norm(data.xpos[second_body] - data.xpos[first_body]))
            lengths[f"{side}_{part}"] = {
                "xsens_origin_distance_m": target_length,
                "template_origin_distance_m": base_length,
            }
            factors[f"{side}_{part}"] = target_length / base_length
    hip_width = float(np.median(np.linalg.norm(positions[:, 1] - positions[:, 4], axis=1)))
    base_hip_width = np.linalg.norm(
        data.xpos[source.body("right_thigh").id] - data.xpos[source.body("left_thigh").id]
    )
    root = ET.parse(base_model).getroot()
    mass_factor = mass_kg / source.body_mass.sum() if mass_kg is not None else 1.0
    for body in root.iter("body"):
        name = body.get("name")
        # Parent segment scaling changes the child joint origin.
        if name.endswith("_shank"):
            side = name.split("_")[0]
            scale_vectors(body, ("pos",), factors[f"{side}_thigh"])
        elif name.endswith("_talus"):
            side = name.split("_")[0]
            scale_vectors(body, ("pos",), factors[f"{side}_shank"])
        elif name.endswith("_thigh"):
            position = np.fromstring(body.get("pos"), sep=" ")
            position[2] *= hip_width / base_hip_width
            body.set("pos", " ".join(map(str, position)))
        segment_scale = factors.get(name, 1.0)
        inertial = body.find("inertial")
        if inertial is not None:
            scale_vectors(inertial, ("pos",), segment_scale)
            scale_vectors(inertial, ("fullinertia", "diaginertia"), segment_scale**2 * mass_factor)
            inertial.set("mass", str(float(inertial.get("mass")) * mass_factor))
        for child in list(body):
            if child.tag == "joint":
                scale_vectors(child, ("pos",), segment_scale)
            elif child.tag == "geom":
                scale_vectors(child, ("pos", "fromto", "size"), segment_scale)
    # Standing qpos must be regenerated from the scaled geometry, not inherited.
    root.remove(root.find("keyframe"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    neutral = mujoco.MjData(model)
    mujoco.mj_forward(model, neutral)
    bottoms = []
    for side in ("left", "right"):
        g = model.geom(f"{side}_sole").id
        bottoms.append(
            neutral.geom_xpos[g, 2]
            - np.abs(neutral.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
        )
    height_offset = -min(bottoms)
    pelvis = next(body for body in root.iter("body") if body.get("name") == "pelvis")
    pos = np.fromstring(pelvis.get("pos"), sep=" ")
    pos[2] += height_offset
    pelvis.set("pos", " ".join(map(str, pos)))
    neutral.qpos[2] += height_offset
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="standing", qpos=" ".join(map(str, neutral.qpos)))
    torso_scale = 1.0
    if height_m is not None:
        # Match the explicit subject stature by scaling the rigid upper-body
        # aggregate above its lumbar pivot. It remains an approximate template.
        compiled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        current = mujoco.MjData(compiled)
        mujoco.mj_forward(compiled, current)
        pivot = current.xpos[compiled.body("torso").id, 2]
        head_geom = compiled.geom("head_visual").id
        head_top = current.geom_xpos[head_geom, 2] + compiled.geom_size[head_geom, 0]
        torso_scale = (height_m - pivot) / (head_top - pivot)
        if not 0.4 <= torso_scale <= 1.6:
            raise ValueError("Requested stature is inconsistent with the estimated leg lengths")
        torso = next(b for b in root.iter("body") if b.get("name") == "torso")
        for element in torso:
            if element.tag == "inertial":
                scale_vectors(element, ("pos",), torso_scale)
                scale_vectors(element, ("fullinertia", "diaginertia"), torso_scale**2)
            elif element.tag in ("geom", "site"):
                scale_vectors(element, ("pos", "fromto", "size"), torso_scale)
    root.insert(
        0,
        ET.Comment(
            " Trial adaptation: Xsens hip width/thigh/shank origin distances; foot and torso geometry retained. See adjacent model_scaling.json. "
        ),
    )
    ET.indent(root, space="  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ET.tostring(root, encoding="unicode") + "\n")
    report = {
        "input": str(input_path.resolve()),
        "base_model": str(base_model.resolve()),
        "leg_lengths": lengths,
        "length_factors": factors,
        "hip_width_m": hip_width,
        "hip_width_factor": float(hip_width / base_hip_width),
        "mass_kg": float(source.body_mass.sum() * mass_factor),
        "mass_source": "user_supplied" if mass_kg is not None else "template_unmeasured",
        "inertia_method": "mass factor times segment length factor squared; subject inertias not measured",
        "unadapted_geometry": ["foot dimensions"],
        "subject_stature_m": height_m,
        "upper_body_length_factor": torso_scale,
        "upper_body_scaling_method": "uniform rigid aggregate scaling to head-visual top; not an anatomical subject model",
        "standing_height_m": float(neutral.qpos[2]),
        "source_xsens_frame_rate": rate,
    }
    output.with_name("model_scaling.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f'Prepared {output}; standing pelvis z={neutral.qpos[2]:.3f} m, mass={report["mass_kg"]:.3f} kg ({report["mass_source"]})'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base-model", type=Path, default=TEMPLATE_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13/human.xml")
    parser.add_argument("--mass-kg", type=float)
    parser.add_argument("--height-m", type=float)
    args = parser.parse_args()
    if args.mass_kg is not None and (not np.isfinite(args.mass_kg) or args.mass_kg <= 0):
        parser.error("--mass-kg must be finite and positive")
    if args.height_m is not None and (not np.isfinite(args.height_m) or args.height_m <= 0):
        parser.error("--height-m must be finite and positive")
    prepare(args.input, args.base_model, args.output, args.mass_kg, args.height_m)


if __name__ == "__main__":
    main()
