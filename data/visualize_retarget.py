#!/usr/bin/env python3
"""Play retargeted qpos in the MuJoCo viewer without stepping dynamics."""
import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, default=ROOT / "data/retargeted_motion.npz")
    parser.add_argument("--model", type=Path, default=ROOT / "model/szar_humanoid.xml")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start-paused", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.vis.headlight.ambient[:] = 0.6
    model.vis.headlight.diffuse[:] = 0.8
    data = mujoco.MjData(model)
    with np.load(args.motion, allow_pickle=False) as motion:
        qpos = motion["qpos"].copy()
        frames = motion["frame"].copy()
        fps = float(motion["fps"])
    if (qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) == 0 or
            len(frames) != len(qpos) or fps <= 0 or not np.isfinite(qpos).all() or
            np.any(np.diff(frames) <= 0)):
        raise ValueError("Motion frames, qpos, or frame rate are invalid for this model")
    times = (frames - frames[0]) / fps
    duration = times[-1] + 1 / fps
    state = {"paused": args.start_paused, "index": 0, "changed": True}

    def on_key(key):
        # GLFW: space, left, right, home. Viewer callbacks receive key presses.
        if key == 32:
            state["paused"] = not state["paused"]
            state["changed"] = True
        elif key in (263, 262):
            state["paused"] = True
            state["index"] = (state["index"] + (-1 if key == 263 else 1)) % len(qpos)
            state["changed"] = True
        elif key == 268:
            state["index"] = 0
            state["changed"] = True

    print("Space: pause/play; Left/Right: previous/next frame; Home: first frame", flush=True)
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = qpos[0, :3]
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15
        anchor = time.monotonic()
        anchor_time = times[0]
        while viewer.is_running():
            now = time.monotonic()
            if state["changed"]:
                anchor = now
                anchor_time = times[state["index"]]
                state["changed"] = False
            if not state["paused"]:
                current = (anchor_time + (now - anchor) * args.speed) % duration
                state["index"] = max(0, int(np.searchsorted(times, current, side="right") - 1))
            data.qpos[:] = qpos[state["index"]]
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = data.qpos[:3]
            viewer.sync()
            time.sleep(0.005)


if __name__ == "__main__":
    main()
