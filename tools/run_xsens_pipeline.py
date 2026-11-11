#!/usr/bin/env python3
"""Build, fit, torque-track and export one explicitly configured Xsens trial."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from prepare_xsens_model import prepare
from fit_xsens_human import fit_motion
from simulate_xsens_walk import simulate
from compute_joint_torques import compute
from plot_xsens_results import export

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/y13_trial.json")
    parser.add_argument(
        "--skip-fit", action="store_true", help="Reuse an existing adapted model and fit"
    )
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = ROOT / config["output_dir"]
    model, motion, trace = output / "human.xml", output / "fit.npz", output / "walk_assisted.npz"
    output.mkdir(parents=True, exist_ok=True)
    if not args.skip_fit:
        print("Preparing trial model...", flush=True)
        prepare(
            ROOT / config["input_xlsx"],
            ROOT / config["base_model"],
            model,
            config["subject"]["mass_kg"],
            config["subject"]["height_m"],
        )
        print("Fitting seven measured body-segment orientations...", flush=True)
        fit_motion(
            ROOT / config["input_xlsx"],
            model,
            motion,
            rate=config["output_rate_hz"],
            reference_frame=config["reference_frame"],
            align_heading=False,
        )
    if args.fit_only:
        export(motion, None, output)
        return
    settings = config["simulation"].copy()
    for name in ("belt_speed_status", "torque_limit_status"):
        settings.pop(name, None)
    settings["start"] = settings.pop("start_s")
    settings["duration"] = settings.pop("duration_s")
    settings["belt_speed"] = settings.pop("belt_speed_m_s")
    settings["torque_limit"] = settings.pop("torque_limit_Nm")
    settings["timestep"] = settings.pop("timestep_s")
    settings["log_rate"] = settings.pop("log_rate_hz")
    print("Running physical torque tracking...", flush=True)
    report = simulate(SimpleNamespace(model=model, motion=motion, output=trace, **settings))
    if report["completed"]:
        compute(trace, output / "joint_torques.npz")
    export(motion, trace, output)
    print(f"Results: {output}", flush=True)
    if not report["completed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
