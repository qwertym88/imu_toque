#!/usr/bin/env python3
"""Audit the seven-IMU recording and export its anatomical SMPL joint binding."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = {
    "01": ("pelvis", 0, None),
    "02": ("right_hip", 2, "01"),
    "03": ("left_hip", 1, "01"),
    "04": ("right_knee", 5, "02"),
    "05": ("left_knee", 4, "03"),
    "06": ("right_ankle", 8, "04"),
    "07": ("left_ankle", 7, "05"),
}
MOUNTING_LOCATIONS = {
    "01": "骨盆",
    "02": "右大腿外侧中部",
    "03": "左大腿外侧中部",
    "04": "右小腿近端，膝盖下方",
    "05": "左小腿近端，膝盖下方",
    "06": "右脚脚背",
    "07": "左脚脚背",
}


def decode_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return [decode_value(element) for element in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_sensor(group, initial_seconds):
    samples = group["data"][:]
    timestamps = group["timestamps"][:]
    if samples.ndim != 2 or samples.shape[1] != 10:
        raise ValueError(f"{group.name}: expected an (N, 10) data array")
    if timestamps.shape != (len(samples),) or len(samples) < 2:
        raise ValueError(f"{group.name}: invalid timestamp shape or sample count")
    if not np.isfinite(samples).all() or not np.isfinite(timestamps).all():
        raise ValueError(f"{group.name}: non-finite data or timestamps")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError(f"{group.name}: timestamps must be strictly increasing")

    quaternions = samples[:, :4]
    acceleration = samples[:, 4:7]
    gyroscope = samples[:, 7:10]
    quaternion_norms = np.linalg.norm(quaternions, axis=1)
    if np.any(quaternion_norms == 0):
        raise ValueError(f"{group.name}: zero-norm quaternion")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    steps = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    largest_step = int(np.argmax(steps))
    initial = timestamps < timestamps[0] + initial_seconds
    acceleration_norms = np.linalg.norm(acceleration, axis=1)
    world_acceleration = rotations.apply(acceleration)
    nonzero_acceleration = acceleration_norms > 0
    up_cosines = np.divide(
        world_acceleration[:, 2],
        acceleration_norms,
        out=np.zeros_like(acceleration_norms),
        where=nonzero_acceleration,
    )
    up_angles = np.rad2deg(np.arccos(np.clip(up_cosines, -1, 1)))

    summary = {
        "samples": len(samples),
        "start_unix_s": float(timestamps[0]),
        "stop_unix_s": float(timestamps[-1]),
        "start_asia_shanghai": datetime.fromtimestamp(
            timestamps[0], ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "mean_rate_hz": float((len(samples) - 1) / (timestamps[-1] - timestamps[0])),
        "interval_ms_min_median_p95_max": (
            1000 * np.percentile(intervals, [0, 50, 95, 100])
        ).tolist(),
        "interval_fraction_below_1ms": float(np.mean(intervals < 0.001)),
        "quaternion_norm_max_error": float(np.max(np.abs(quaternion_norms - 1))),
        "quaternion_sign_flips": int(
            np.sum(np.einsum("ij,ij->i", quaternions[:-1], quaternions[1:]) < 0)
        ),
        "quaternion_step_deg_median_p99_max": np.percentile(steps, [50, 99, 100]).tolist(),
        "quaternion_steps_above_30deg": int(np.sum(steps > 30)),
        "largest_step": {
            "sample_index": largest_step,
            "elapsed_s": float(timestamps[largest_step] - timestamps[0]),
            "host_interval_ms": float(intervals[largest_step] * 1000),
            "gyroscope_norms_assumed_deg_s": np.linalg.norm(
                gyroscope[largest_step : largest_step + 2], axis=1
            ).tolist(),
        },
        "initial_window_samples": int(np.sum(initial)),
        "initial_gyroscope_norm_median_assumed_deg_s": float(
            np.median(np.linalg.norm(gyroscope[initial], axis=1))
        ),
        "initial_acceleration_mean_assumed_g": acceleration[initial].mean(axis=0).tolist(),
        "initial_acceleration_norm_median_assumed_g": float(np.median(acceleration_norms[initial])),
        "initial_world_acceleration_up_error_median_deg": float(
            np.median(up_angles[initial & nonzero_acceleration])
        ),
    }
    return summary


def analyze(input_path, initial_seconds):
    with h5py.File(input_path, "r") as recording:
        actual_sensors = set(recording["imu"])
        if actual_sensors != set(BINDINGS):
            raise ValueError(f"Expected sensors 01–07; found {sorted(actual_sensors)}")
        metadata = {
            name: decode_value(dataset[()]) for name, dataset in recording["metadata"].items()
        }
        sensors = {
            address: summarize_sensor(recording["imu"][address], initial_seconds)
            for address in BINDINGS
        }

    start = max(sensor["start_unix_s"] for sensor in sensors.values())
    stop = min(sensor["stop_unix_s"] for sensor in sensors.values())
    return {
        "input": str(input_path.resolve()),
        "status": "anatomical_binding_only_calibration_required",
        "metadata": metadata,
        "assumptions_not_encoded_as_h5_column_attributes": {
            "data_columns": ["qw", "qx", "qy", "qz", "ax", "ay", "az", "gx", "gy", "gz"],
            "quaternion_action": "active sensor-to-world, wxyz storage",
            "acceleration": "gravity included; g units inferred from stationary magnitude",
            "gyroscope": "sensor-frame angular velocity, deg/s per collection notes",
            "world_up_for_gravity_diagnostic": "+Z; diagnostic hypothesis, not verified calibration",
            "initial_window_seconds": initial_seconds,
            "initial_window_is_known_standing_pose": False,
        },
        "total_samples": sum(sensor["samples"] for sensor in sensors.values()),
        "common_time_interval_unix_s": [start, stop],
        "common_duration_s": stop - start,
        "sensor_binding": {
            address: {
                "mounting_location": MOUNTING_LOCATIONS[address],
                "smpl_joint_name": name,
                "smpl_joint_index_zero_based": index,
                "parent_sensor": parent,
                "meaning": "segment orientation associated with this proximal joint",
                "sensor_to_segment_rotation": None,
                "world_frame_alignment_rotation": None,
                "calibration_status": "unresolved",
            }
            for address, (name, index, parent) in BINDINGS.items()
        },
        "orientation_constrained_joint_indices_after_calibration": [0, 1, 2, 4, 5, 7, 8],
        "unmeasured_joint_indices": sorted(
            set(range(24)) - {binding[1] for binding in BINDINGS.values()}
        ),
        "shape_and_translation_measured": False,
        "sensors": sensors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/y13_level_walking_3_8_trial01.h5"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/y13_smpl_feasibility.json")
    parser.add_argument("--initial-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.initial_seconds) or args.initial_seconds <= 0:
        parser.error("--initial-seconds must be finite and positive")
    report = analyze(args.input, args.initial_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved anatomical binding and diagnostics: {args.output}")
    print(f"Common duration: {report['common_duration_s']:.3f} s")
    print("IMU  SMPL joint    samples  mean Hz  max quaternion step (deg)")
    for address, sensor in report["sensors"].items():
        print(
            f"{address}   {BINDINGS[address][0]:12s} {sensor['samples']:7d} "
            f"{sensor['mean_rate_hz']:8.3f} "
            f"{sensor['quaternion_step_deg_median_p99_max'][2]:12.2f}"
        )


if __name__ == "__main__":
    main()
