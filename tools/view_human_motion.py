#!/usr/bin/env python3
"""Replay a kinematic fit or recorded torque simulation, or render a GIF/PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def load_motion(path, model_override=None):
    with np.load(path, allow_pickle=False) as archive:
        poses = archive["qpos"].copy()
        times = archive["time_s"].copy()
        physical = "settings_json" in archive.files
        settings = json.loads(str(archive["settings_json" if physical else "metadata_json"]))
        model_path = model_override or Path(settings["model"])
        if physical:
            title = "TORQUE SIMULATION | " + settings["mode"].upper()
        else:
            title = "KINEMATIC FIT | NO PHYSICS STEPPING"
    return model_path, poses, times, title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=ROOT / "outputs/y13/walk_assisted.npz")
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--output", type=Path, help="Render .gif or a single .png; otherwise open viewer"
    )
    parser.add_argument("--start", type=float, default=0.0, help="Seconds relative to this trace")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if any(not np.isfinite(v) or v <= 0 for v in (args.duration, args.fps, args.speed)):
        parser.error("--duration, --fps and --speed must be finite and positive")
    model_path, poses, times, title = load_motion(args.motion, args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)
    if poses.ndim != 2 or poses.shape[1] != model.nq:
        raise ValueError("Motion shape does not match its human model")
    times -= times[0]
    if args.start < 0 or args.start > times[-1]:
        parser.error("--start lies outside the recording")
    stop = min(args.start + args.duration, float(times[-1]))
    if args.output:
        if args.output.suffix.lower() not in (".png", ".gif"):
            parser.error("--output must end in .png or .gif")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(camera)
            camera.distance, camera.azimuth, camera.elevation = 3.0, 115, -12
            frame_times = np.arange(args.start, stop + 1e-9, args.speed / args.fps)
            if args.output.suffix.lower() == ".png":
                frame_times = frame_times[:1]
            for elapsed in frame_times:
                index = min(int(np.searchsorted(times, elapsed)), len(poses) - 1)
                data.qpos[:] = poses[index]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [*data.qpos[:2], 0.9]
                renderer.update_scene(data, camera)
                image = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 640, 40), fill=(20, 25, 30))
                draw.text((8, 5), title, fill="white")
                draw.text((8, 23), f"t={elapsed:.2f}s | recorded states replayed", fill="white")
                frames.append(image)
        if args.output.suffix.lower() == ".gif":
            frames[0].save(
                args.output,
                save_all=True,
                append_images=frames[1:],
                duration=int(round(1000 / args.fps)),
                loop=0,
            )
        else:
            frames[0].save(args.output)
        print(f"Rendered {args.output}")
        return
    import mujoco.viewer as viewer_module

    with viewer_module.launch_passive(model, data) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 115, -12
        wall_start = time.monotonic()
        while viewer.is_running():
            elapsed = args.start + ((time.monotonic() - wall_start) * args.speed) % max(
                stop - args.start, 1 / args.fps
            )
            index = min(int(np.searchsorted(times, elapsed)), len(poses) - 1)
            data.qpos[:] = poses[index]
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = [*data.qpos[:2], 0.9]
            viewer.sync()
            time.sleep(1 / 120)


if __name__ == "__main__":
    main()
