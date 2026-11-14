"""Unsupervised temporal foot-contact inference from one matching Xsens export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp


SIDES = ("left", "right")
SEGMENTS = {"left": "Left", "right": "Right"}
FEATURE_NAMES = (
    "foot_acceleration_norm_log",
    "foot_angular_velocity_norm_log",
    "foot_angular_acceleration_norm_log",
    "foot_shank_angular_velocity_difference_log",
    "foot_velocity_residual_log",
)


def transition_events(states, valid, time_s, side):
    events = {f"{side}_contact_on_s": [], f"{side}_contact_off_s": []}
    pair_valid = valid[1:] & valid[:-1]
    changes = np.diff(states.astype(np.int8))
    for index in np.flatnonzero(pair_valid & (changes != 0)) + 1:
        key = f"{side}_contact_{'on' if changes[index - 1] == 1 else 'off'}_s"
        events[key].append(float(time_s[index]))
    return events


def robust_standardize(values):
    center = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - center), axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
    return np.clip((values - center) / scale, -6.0, 6.0), center, scale


def _forward_backward(log_emission, switch_probability):
    transition = np.array(
        [
            [1 - switch_probability, switch_probability],
            [switch_probability, 1 - switch_probability],
        ],
        dtype=float,
    )
    log_transition = np.log(transition)
    initial = np.log(np.array([0.5, 0.5]))
    count = len(log_emission)
    forward = np.empty((count, 2))
    backward = np.zeros((count, 2))
    forward[0] = initial + log_emission[0]
    for frame in range(1, count):
        forward[frame] = log_emission[frame] + logsumexp(
            forward[frame - 1, :, None] + log_transition, axis=0
        )
    for frame in range(count - 2, -1, -1):
        backward[frame] = logsumexp(
            log_transition + log_emission[frame + 1, None, :] + backward[frame + 1, None, :],
            axis=1,
        )
    posterior = forward + backward
    posterior -= logsumexp(posterior, axis=1, keepdims=True)
    return np.exp(posterior), float(logsumexp(forward[-1]))


def _viterbi(log_emission, switch_probability):
    log_transition = np.log(
        [[1 - switch_probability, switch_probability], [switch_probability, 1 - switch_probability]]
    )
    score = np.empty_like(log_emission)
    back = np.zeros(log_emission.shape, dtype=np.int8)
    score[0] = np.log(0.5) + log_emission[0]
    for frame in range(1, len(log_emission)):
        candidates = score[frame - 1, :, None] + log_transition
        back[frame] = np.argmax(candidates, axis=0)
        score[frame] = log_emission[frame] + np.max(candidates, axis=0)
    state = np.empty(len(log_emission), dtype=np.int8)
    state[-1] = np.argmax(score[-1])
    for frame in range(len(state) - 1, 0, -1):
        state[frame - 1] = back[frame, state[frame]]
    return state


def _log_emission(features, means, variances):
    delta = features[:, None, :] - means[None, :, :]
    return -0.5 * np.sum(
        np.log(2 * np.pi * variances)[None, :, :] + delta**2 / variances[None, :, :], axis=-1
    )


def fit_temporal_hmm(features, valid, sample_rate, expected_state_duration_s=0.45, iterations=20):
    """Fit two diagonal-Gaussian emissions; a persistent HMM supplies temporal context.

    State semantics are assigned after fitting: lower dynamic activity is the
    contact-like state. This is an unlabeled detector and requires inspection.
    """
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("Sample rate must be finite and positive")
    if not np.isfinite(expected_state_duration_s) or expected_state_duration_s <= 0:
        raise ValueError("Expected state duration must be finite and positive")
    if np.asarray(features).ndim != 2 or not features.shape[1]:
        raise ValueError("Features must be a nonempty frames-by-features array")
    valid = np.asarray(valid, dtype=bool) & np.isfinite(features).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < max(100, sample_rate):
        raise ValueError("Not enough valid Xsens data to estimate a two-state model")
    x, center, scale = robust_standardize(features[indices])
    segment_bounds = np.flatnonzero(np.diff(indices) > 1) + 1
    segments = np.split(np.arange(len(indices)), segment_bounds)
    activity = np.mean(x, axis=1)
    means = np.stack(
        [
            np.mean(x[activity <= np.quantile(activity, 0.3)], axis=0),
            np.mean(x[activity >= np.quantile(activity, 0.7)], axis=0),
        ]
    )
    overall = np.var(x, axis=0) + 0.1
    variances = np.stack([overall.copy(), overall.copy()])
    switch_probability = float(np.clip(1.0 / (expected_state_duration_s * sample_rate), 1e-4, 0.1))
    posterior = None
    for _ in range(iterations):
        emission = _log_emission(x, means, np.maximum(variances, 0.08))
        posterior = np.zeros((len(x), 2))
        for segment in segments:
            posterior[segment], _ = _forward_backward(emission[segment], switch_probability)
        weights = posterior.sum(axis=0).clip(1e-6)
        means = posterior.T @ x / weights[:, None]
        variances = np.stack(
            [np.average((x - means[k]) ** 2, axis=0, weights=posterior[:, k]) for k in range(2)]
        )
        variances = np.maximum(variances, 0.08)
    # Identifiability: order state labels by aggregate dynamic activity.
    if means[0].mean() > means[1].mean():
        means, variances = means[::-1].copy(), variances[::-1].copy()
    emission = _log_emission(x, means, variances)
    posterior = np.zeros((len(x), 2))
    states = np.zeros(len(x), dtype=np.int8)
    likelihood = 0.0
    for segment in segments:
        posterior[segment], segment_likelihood = _forward_backward(
            emission[segment], switch_probability
        )
        likelihood += segment_likelihood
        states[segment] = _viterbi(emission[segment], switch_probability)
    probability = np.full((len(features), 2), np.nan)
    discrete = np.full(len(features), -1, dtype=np.int8)
    probability[indices] = posterior
    discrete[indices] = states
    return {
        "state": discrete,
        "probability_contact": probability[:, 0],
        "posterior": probability,
        "valid": valid,
        "feature_center": center,
        "feature_scale": scale,
        "state_mean_standardized": means,
        "state_variance_standardized": variances,
        "switch_probability_per_sample": switch_probability,
        "expected_state_duration_s": expected_state_duration_s,
        "log_likelihood": likelihood,
    }


def _read_vector_sheet(workbook, sheet_name, segment_names):
    sheet = workbook[sheet_name]
    header = [str(value) for value in next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))]
    frame_col = header.index("Frame")
    columns = {
        segment: [header.index(f"{segment} {axis}") for axis in ("x", "y", "z")]
        for segment in segment_names
    }
    rows = np.asarray(list(sheet.iter_rows(min_row=2, values_only=True)), dtype=np.float64)
    return rows[:, frame_col].astype(np.int64), {
        segment: rows[:, columns[segment]] for segment in segment_names
    }


def load_xsens_features(xlsx_path):
    import openpyxl

    workbook = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        metadata = {
            str(row[0]): row[1]
            for row in workbook["General Information"].iter_rows(values_only=True)
            if len(row) > 1 and row[0] is not None
        }
        rate = float(metadata["Frame Rate"])
        required = (
            "Segment Acceleration",
            "Segment Angular Velocity",
            "Segment Angular Acceleration",
            "Segment Velocity",
        )
        feet = [f"{side} Foot" for side in ("Left", "Right")]
        shanks = [f"{side} Lower Leg" for side in ("Left", "Right")]
        loaded = {
            name: _read_vector_sheet(workbook, name, feet + shanks + ["Pelvis"])
            for name in required
        }
    finally:
        workbook.close()
    base_frames = loaded[required[0]][0]
    if not np.array_equal(base_frames, np.arange(len(base_frames))):
        raise ValueError("Y13 frame numbers must be contiguous from zero")
    for name in required[1:]:
        if not np.array_equal(loaded[name][0], base_frames):
            raise ValueError(f"Frame mismatch in Xsens worksheet {name}")
    feature_by_side = {}
    for side in ("Left", "Right"):
        foot, shank = f"{side} Foot", f"{side} Lower Leg"
        acceleration = loaded["Segment Acceleration"][1][foot]
        angular_velocity = loaded["Segment Angular Velocity"][1][foot]
        angular_acceleration = loaded["Segment Angular Acceleration"][1][foot]
        shank_angular_velocity = loaded["Segment Angular Velocity"][1][shank]
        relative_velocity = (
            loaded["Segment Velocity"][1][foot] - loaded["Segment Velocity"][1]["Pelvis"]
        )
        # Filter at 240 Hz only to suppress export jitter; event timing is retained.
        signals = [
            acceleration,
            angular_velocity,
            angular_acceleration,
            angular_velocity - shank_angular_velocity,
            relative_velocity,
        ]
        magnitudes = [np.linalg.norm(signal, axis=1) for signal in signals]
        smoothed = [
            gaussian_filter1d(value, sigma=max(rate * 0.015, 1), mode="nearest")
            for value in magnitudes
        ]
        feature_by_side[side.lower()] = np.column_stack([np.log1p(value) for value in smoothed])
    return base_frames, rate, feature_by_side, {str(k): str(v) for k, v in metadata.items()}


def validate_pairing(xlsx_path, fit_metadata):
    source_xlsx = Path(fit_metadata.get("source_xlsx", "")).resolve()
    candidate = Path(xlsx_path).resolve()
    source_hash = fit_metadata.get("source_xlsx_sha256")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if source_hash:
        matches = source_hash == candidate_hash
    else:
        matches = source_xlsx == candidate
    if not matches:
        raise ValueError(
            "The Xsens source does not match the supplied fit; force plates or fits from a different sample cannot be paired"
        )
    source_metadata = fit_metadata.get("source_metadata", {})
    suit = " ".join(source_metadata.get("Suit Label", "").lower().split())
    if "y13" not in suit and "y13" not in source_xlsx.name.lower():
        raise ValueError("This contact workflow only accepts the configured Y13 Xsens sample")
    return {
        "sample_id": "y13",
        "matched_xsens": str(candidate),
        "matched_xsens_sha256": candidate_hash,
        "matched_fit_model_sha256": fit_metadata.get("model_sha256"),
        "force_plate_used": False,
        "force_plate_path": None,
        "pairing_policy": "Y13 sample only; data/fp_raw.csv belongs to the legacy xsense_raw.xlsx sample",
    }


def detect_contact(
    xlsx_path, fit_path, output_dir, expected_state_duration_s=0.45, uncertainty=(0.35, 0.65)
):
    import hashlib

    xlsx_path, fit_path, output_dir = map(Path, (xlsx_path, fit_path, output_dir))
    with np.load(fit_path, allow_pickle=False) as fit:
        fit_metadata = json.loads(str(fit["metadata_json"]))
        pairing = validate_pairing(xlsx_path, fit_metadata)
        fit_times = fit["time_s"].copy()
        valid_fit = (fit["valid"] & fit["derivative_valid"]).copy()
        model_path = Path(fit_metadata["model"])
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != fit_metadata["model_sha256"]:
            raise ValueError("The fitted model changed; rebuild fit before contact inference")
    frames, sample_rate, features, source_metadata = load_xsens_features(xlsx_path)
    high_rate_times = frames / sample_rate
    fit_valid_at_source = valid_fit[
        np.clip(
            np.rint(high_rate_times / np.median(np.diff(fit_times))).astype(int),
            0,
            len(fit_times) - 1,
        )
    ]
    outputs = {}
    for side in SIDES:
        result = fit_temporal_hmm(
            features[side], fit_valid_at_source, sample_rate, expected_state_duration_s
        )
        outputs[side] = result
    # Align high-rate detections to the 60 Hz fitted motion used by WBC.
    output_indices = np.clip(np.rint(fit_times * sample_rate).astype(int), 0, len(frames) - 1)
    states = np.stack([outputs[side]["state"][output_indices] == 0 for side in SIDES], axis=1)
    probabilities = np.stack(
        [outputs[side]["probability_contact"][output_indices] for side in SIDES], axis=1
    )
    valid = np.stack([outputs[side]["valid"][output_indices] for side in SIDES], axis=1).all(axis=1)
    uncertain = (probabilities >= uncertainty[0]) & (probabilities <= uncertainty[1])
    event_frames = {}
    for index, side in enumerate(SIDES):
        event_frames.update(transition_events(states[:, index], valid, fit_times, side))
    report = {
        **pairing,
        "source_metadata": source_metadata,
        "source_frame_rate_hz": sample_rate,
        "output_frame_rate_hz": float(1 / np.median(np.diff(fit_times))),
        "source_xlsx_sha256": hashlib.sha256(xlsx_path.read_bytes()).hexdigest(),
        "feature_names": list(FEATURE_NAMES),
        "method": "unlabeled two-state diagonal Gaussian HMM with persistence prior; low dynamic-activity state is contact-like",
        "expected_state_duration_s": expected_state_duration_s,
        "uncertainty_probability_interval": list(uncertainty),
        "valid_fraction": float(valid.mean()),
        "uncertain_fraction_by_side": dict(zip(SIDES, uncertain[valid].mean(axis=0).tolist())),
        "contact_fraction_by_side": dict(zip(SIDES, states[valid].mean(axis=0).tolist())),
        "state_mean_standardized_by_side": {
            side: outputs[side]["state_mean_standardized"].tolist() for side in SIDES
        },
        "state_variance_standardized_by_side": {
            side: outputs[side]["state_variance_standardized"].tolist() for side in SIDES
        },
        "hmm_log_likelihood_by_side": {
            side: float(outputs[side]["log_likelihood"]) for side in SIDES
        },
        "events": event_frames,
        "limitations": [
            "No contact labels were supplied; probabilities are unsupervised detector scores, not calibrated probabilities.",
            "Y13 is flat treadmill walking only; slope/stair generalization is untested.",
            "Contact state does not identify heel/toe contact geometry or ground normal.",
            "The old force plate file is from another sample and was not loaded or used.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "contact_inference.npz",
        time_s=fit_times,
        frame=np.rint(fit_times * sample_rate).astype(np.int64),
        sides=np.asarray(SIDES),
        contact_state_candidate=states,
        support_mask_candidate=states & (probabilities >= 0.8),
        contact_probability=probabilities,
        valid=valid,
        uncertain=uncertain,
        high_rate_time_s=high_rate_times,
        high_rate_state=np.stack([outputs[side]["state"] == 0 for side in SIDES], axis=1),
        high_rate_probability=np.stack(
            [outputs[side]["probability_contact"] for side in SIDES], axis=1
        ),
        high_rate_valid=fit_valid_at_source,
        feature_center=np.stack([outputs[side]["feature_center"] for side in SIDES]),
        feature_scale=np.stack([outputs[side]["feature_scale"] for side in SIDES]),
        state_mean_standardized=np.stack(
            [outputs[side]["state_mean_standardized"] for side in SIDES]
        ),
        state_variance_standardized=np.stack(
            [outputs[side]["state_variance_standardized"] for side in SIDES]
        ),
        features=np.stack([features[side] for side in SIDES], axis=1),
        feature_names=np.asarray(FEATURE_NAMES),
        metadata_json=np.asarray(json.dumps(report)),
    )
    header = "time_s,contact_left_candidate,contact_right_candidate,score_left_uncalibrated,score_right_uncalibrated,valid,uncertain_left,uncertain_right"
    columns = np.column_stack(
        (
            fit_times,
            states.astype(np.int8),
            probabilities,
            valid.astype(np.int8),
            uncertain.astype(np.int8),
        )
    )
    np.savetxt(
        output_dir / "contact_states.csv", columns, delimiter=",", header=header, comments=""
    )
    (output_dir / "contact_inference.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
