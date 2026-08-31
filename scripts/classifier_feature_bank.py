"""classifier_feature_bank.py -- All 249 windowed features (102 gaze +
96 robot + 51 joint) for the Fine/Gross classifier, combined into one
file. Python port of KD_test/matlab/gaze_feature_extraction.m,
KD_test/matlab/robot_feature_extraction.m, and KD_test/matlab/
joint_feature_extraction.m, reproduced formula-for-formula so the same
window of data gives the same numbers in every language -- exactly the
same relationship KD_test/python/gaze_features.py already has to its
own .m source; this file additionally covers the robot and joint banks
that file doesn't.

Three build_*_feature_bank(config) functions, one per source, each
returning a list of (name, function) pairs with the shared
(..signals..) -> float interface its own .m file established (see each
section's own header comment below for the exact signature and why it
differs between banks -- gaze needs only gaze signals, robot only
robot signals, joint needs both). build_all_feature_banks(config)
below returns all three as one dict, for a training script that wants
every feature at once.

Frame (same decision as gaze_features.py, restated here since it now
governs all three banks, not just gaze's): Gx/Gy (gaze) and Px/Py
(robot position) are calibrated CoppeliaSim-frame METRES, mapped
through this project's established Tobii-UCS -> CoppeliaSim affine
mapping (map_gaze_ucs_to_coppelia below), not gaze_feature_extraction.
m's raw normalized display area. A handful of gaze features that used
to assume a fixed [0, 1] display box, and the robot bank's own
position-entropy features (which have no such fixed box to begin
with), are governed by FeatureConfig instead -- see its own docstring.

Two different velocity philosophies are used on purpose, matching this
project's other derivative chains: the robot bank differentiates
MEASURED velocity (Vx/Vy, one derivative closer to what was actually
measured) for everything except DSJ, which differentiates POSITION
three times (matching this project's other DSJ chains exactly, for
DSJ values comparable project-wide); the gaze bank has no measured
velocity to work with at all, so it differentiates gaze POSITION for
everything, DSJ included.

Every FORCE-derived and COMMAND-TRACKING-derived feature (both banks
that touch either) is partly a function of the CURRENT admittance
damping (B_x_Ns_m/B_y_Ns_m), not just behaviour -- higher damping needs
more force and lags the command more for identical intent. If damping
varies across the trials you pool together (as it does in this
project's fine/gross dataset: d10/d100/d200), these features partly
just encode which damping condition a window came from. They are still
computed (this bank tries to extract everything the raw data can give,
per its own brief), but treat their ranking with that in mind -- each
such feature is marked DAMPING-CONFOUNDED in its own docstring.

Dependencies: numpy (pip install numpy)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

FeatureFunction = Callable[..., float]


@dataclass
class FeatureConfig:
    """Tunable reference points/thresholds, grouped by which bank uses
    them. None of these change the SHAPE of any feature's formula --
    only a reference point or cutoff a formula is evaluated against.

    --- Gaze bank ---
    workspace_center_xy_m: the gaze bank's "display center" replacement
        -- defaults to the calibration JSON's own coppelia_reference_
        xy_m, a fixed, physically meaningful point rather than
        something re-derived per trial.
    workspace_half_extent_m: the gaze bank's "outside display" bounds
        replacement -- a square of this half-width around workspace_
        center_xy_m.
    low/high_speed_threshold_m_s: replace gaze_feature_extraction.m's
        fixed 0.10 / 0.50 "normalized/s" GAZE-speed bands.
    close_thresholds_m: replace gaze_feature_extraction.m's fixed
        (0.05, 0.10, 0.20) "display unit" gaze-to-robot distance bands,
        in metres.
    spatial_bin_count_coarse/fine: unchanged from gaze_feature_
        extraction.m (10, 40) -- its entropy features re-normalize into
        [0, 1] using workspace_center_xy_m/workspace_half_extent_m
        first, then bin exactly as the .m file did.

    --- Robot bank ---
    robot_low/high_speed_threshold_m_s: robot_feature_extraction.m's
        own ROBOT-speed bands (0.02 / 0.08 m/s) -- NOT empirically
        derived from this project's data (flagged honestly, unchanged
        from the .m file, which wasn't derived from data either; redo
        against real TCP-speed percentiles if you want these grounded).
    robot_spatial_bin_count_coarse/fine: unchanged from robot_feature_
        extraction.m (10, 40) -- unlike the gaze bank's fixed display
        box, robot position entropy bins over each WINDOW's own
        observed range (no fixed physical box for robot position to
        bin against), so there is no robot workspace_center/extent
        setting to match the gaze ones above.

    --- Joint bank ---
    joint_close_threshold_m: replaces joint_feature_extraction.m's
        fixed 0.02 m "gaze close to hand" cutoff (used by several
        curvature-/force-conditioned features).
    """

    workspace_center_xy_m: Tuple[float, float]
    workspace_half_extent_m: float = 0.15
    low_speed_threshold_m_s: float = 0.05
    high_speed_threshold_m_s: float = 0.30
    close_thresholds_m: Tuple[float, float, float] = (0.01, 0.02, 0.05)
    spatial_bin_count_coarse: int = 10
    spatial_bin_count_fine: int = 40

    robot_low_speed_threshold_m_s: float = 0.02
    robot_high_speed_threshold_m_s: float = 0.08
    robot_spatial_bin_count_coarse: int = 10
    robot_spatial_bin_count_fine: int = 40

    joint_close_threshold_m: float = 0.02


# ======================================================================
# Shared finite-value statistics helpers -- used by all three banks.
# Ports of each .m file's safe_* / percentile_value / pearson_
# correlation (identical across all three .m files). Implemented
# directly (not via scipy) so the exact MATLAB convention (population-
# moment skew/kurtosis, ddof=1 std, n>=3 minimum for a correlation, ...)
# is guaranteed rather than assumed to match a library default.
# ======================================================================

def finite_values(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    return values[np.isfinite(values)]


def safe_mean(values) -> float:
    values = finite_values(values)
    return float(np.mean(values)) if values.size else np.nan


def safe_std(values) -> float:
    values = finite_values(values)
    return float(np.std(values, ddof=1)) if values.size >= 2 else np.nan


def safe_variance(values) -> float:
    std = safe_std(values)
    return std ** 2 if np.isfinite(std) else np.nan


def safe_minimum(values) -> float:
    values = finite_values(values)
    return float(np.min(values)) if values.size else np.nan


def safe_maximum(values) -> float:
    values = finite_values(values)
    return float(np.max(values)) if values.size else np.nan


def safe_range(values) -> float:
    lo, hi = safe_minimum(values), safe_maximum(values)
    return hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan


def safe_median(values) -> float:
    values = finite_values(values)
    return float(np.median(values)) if values.size else np.nan


def percentile_value(values, percentile: float) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    percentile = min(max(percentile, 0.0), 100.0)
    # numpy's default ('linear') interpolation matches every .m file's
    # hand-rolled rank = 1 + (n-1)*p/100 formula exactly.
    return float(np.percentile(values, percentile))


def safe_interquartile_range(values) -> float:
    return percentile_value(values, 75.0) - percentile_value(values, 25.0)


def safe_median_absolute_deviation(values) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    return float(np.median(np.abs(values - np.median(values))))


def safe_skewness(values) -> float:
    values = finite_values(values)
    if values.size < 3:
        return np.nan
    centered = values - values.mean()
    second_moment = np.mean(centered ** 2)
    if second_moment <= 0:
        return np.nan
    return float(np.mean(centered ** 3) / second_moment ** 1.5)


def safe_kurtosis(values) -> float:
    values = finite_values(values)
    if values.size < 4:
        return np.nan
    centered = values - values.mean()
    second_moment = np.mean(centered ** 2)
    if second_moment <= 0:
        return np.nan
    return float(np.mean(centered ** 4) / second_moment ** 2)


def safe_covariance(first_values, second_values) -> float:
    first_values = np.asarray(first_values, dtype=float).ravel()
    second_values = np.asarray(second_values, dtype=float).ravel()
    valid = np.isfinite(first_values) & np.isfinite(second_values)
    first_values, second_values = first_values[valid], second_values[valid]
    if first_values.size < 2:
        return np.nan
    return float(np.cov(first_values, second_values, ddof=1)[0, 1])


def pearson_correlation(first_values, second_values) -> float:
    first_values = np.asarray(first_values, dtype=float).ravel()
    second_values = np.asarray(second_values, dtype=float).ravel()
    valid = np.isfinite(first_values) & np.isfinite(second_values)
    first_values, second_values = first_values[valid], second_values[valid]
    if first_values.size < 3:
        return np.nan
    first_centered = first_values - first_values.mean()
    second_centered = second_values - second_values.mean()
    denominator = np.sqrt(
        np.sum(first_centered ** 2) * np.sum(second_centered ** 2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(first_centered * second_centered) / denominator)


def safe_root_mean_square(values) -> float:
    values = finite_values(values)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else np.nan


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or \
            abs(denominator) <= 1e-12:
        return np.nan
    return numerator / denominator


# ======================================================================
# Shared planar time-series helpers.
# ======================================================================

def clean_planar_time_series(x, y, time_s):
    """Drop non-finite rows, sort by time, and drop duplicate timestamps
    (keeping the first occurrence) -- matches every .m file's
    clean_planar_time_series() exactly."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    time_s = np.asarray(time_s, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(time_s)
    x, y, time_s = x[valid], y[valid], time_s[valid]

    order = np.argsort(time_s, kind="stable")
    x, y, time_s = x[order], y[order], time_s[order]

    # time_s is now sorted ascending, so np.unique's return_index (the
    # first-occurrence index of each unique value) is itself already
    # ascending -- indexing with it keeps exactly the first sample of
    # each duplicate timestamp, in time order, matching MATLAB's
    # unique(clean_time_s, 'stable') applied to already-sorted input.
    _, unique_indices = np.unique(time_s, return_index=True)
    keep = np.sort(unique_indices)
    return x[keep], y[keep], time_s[keep]


def clean_time_indices(raw_time_s) -> np.ndarray:
    """Index array such that raw_time_s[idx] is finite, sorted, and
    duplicate-free (first occurrence kept). Apply the SAME idx to any
    array paired with raw_time_s (e.g. force[idx]) to keep it aligned
    -- the multi-array generalization of clean_planar_time_series()
    above, which only cleans exactly two paired value arrays; needed
    by the robot/joint banks, which pair MANY robot signals to one
    clock. Port of joint_feature_extraction.m's clean_time_indices()."""
    raw_time_s = np.asarray(raw_time_s, dtype=float)
    valid_idx = np.flatnonzero(np.isfinite(raw_time_s))
    order = np.argsort(raw_time_s[valid_idx], kind="stable")
    sorted_idx = valid_idx[order]
    _, first_occurrence = np.unique(raw_time_s[sorted_idx], return_index=True)
    return sorted_idx[np.sort(first_occurrence)]


def calculate_planar_motion(x, y, time_s) -> dict:
    """Differentiate a planar trajectory in time, from POSITION -- port
    of gaze_feature_extraction.m's calculate_planar_motion(). Returns a
    dict with x, y, time_s, velocity_x, velocity_y, speed,
    acceleration, jerk, direction (NaN-filled fields when there are too
    few samples for that derivative). Used by the gaze bank, which has
    no measured velocity to differentiate instead -- see
    calculate_robot_motion() below for the robot bank's own version,
    which starts from MEASURED velocity and also derives curvature."""
    x, y, time_s = clean_planar_time_series(x, y, time_s)
    n = time_s.size
    motion = {
        "x": x, "y": y, "time_s": time_s,
        "velocity_x": np.full(n, np.nan), "velocity_y": np.full(n, np.nan),
        "speed": np.full(n, np.nan), "acceleration": np.full(n, np.nan),
        "jerk": np.full(n, np.nan), "direction": np.full(n, np.nan),
    }
    if n < 2:
        return motion

    velocity_x = np.gradient(x, time_s)
    velocity_y = np.gradient(y, time_s)
    speed = np.hypot(velocity_x, velocity_y)
    motion["velocity_x"], motion["velocity_y"], motion["speed"] = \
        velocity_x, velocity_y, speed
    moving = speed > 1e-12
    direction = np.full(n, np.nan)
    direction[moving] = np.arctan2(velocity_y[moving], velocity_x[moving])
    motion["direction"] = direction

    if n < 3:
        return motion
    acceleration_x = np.gradient(velocity_x, time_s)
    acceleration_y = np.gradient(velocity_y, time_s)
    motion["acceleration"] = np.hypot(acceleration_x, acceleration_y)

    if n < 4:
        return motion
    jerk_x = np.gradient(acceleration_x, time_s)
    jerk_y = np.gradient(acceleration_y, time_s)
    motion["jerk"] = np.hypot(jerk_x, jerk_y)
    return motion


def planar_path_length(x, y) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.nan
    return float(np.sum(np.hypot(np.diff(x), np.diff(y))))


def planar_net_displacement(x, y) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.nan
    return float(np.hypot(x[-1] - x[0], y[-1] - y[0]))


def circular_mean(angles_rad) -> float:
    angles_rad = finite_values(angles_rad)
    if angles_rad.size == 0:
        return np.nan
    return float(np.arctan2(
        np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad))))


def circular_concentration(angles_rad) -> float:
    angles_rad = finite_values(angles_rad)
    if angles_rad.size == 0:
        return np.nan
    return float(np.hypot(
        np.mean(np.cos(angles_rad)), np.mean(np.sin(angles_rad))))


def count_true_runs(logical_values) -> float:
    logical_values = np.asarray(logical_values, dtype=bool).ravel()
    if logical_values.size == 0:
        return np.nan
    return float(logical_values[0]) + float(np.sum(
        (~logical_values[:-1]) & logical_values[1:]))


def missing_time_fraction(time_s) -> float:
    time_s = np.sort(finite_values(time_s))
    if time_s.size < 3 or time_s[-1] <= time_s[0]:
        return np.nan
    intervals = np.diff(time_s)
    nominal_interval_s = np.median(intervals)
    is_gap = intervals > 1.5 * nominal_interval_s
    missing_time_s = np.sum(intervals[is_gap] - nominal_interval_s)
    return float(missing_time_s / (time_s[-1] - time_s[0]))


def sample_gap_count(time_s) -> float:
    time_s = np.sort(finite_values(time_s))
    if time_s.size < 3:
        return np.nan
    intervals = np.diff(time_s)
    nominal_interval_s = np.median(intervals)
    return float(np.sum(intervals > 1.5 * nominal_interval_s))


def mean_crossing_count(values) -> float:
    values = finite_values(values)
    if values.size < 2:
        return np.nan
    centered_sign = np.sign(values - values.mean())
    centered_sign = centered_sign[centered_sign != 0]
    if centered_sign.size < 2:
        return 0.0
    return float(np.sum(centered_sign[:-1] * centered_sign[1:] < 0))


def lag_one_autocorrelation(values) -> float:
    values = finite_values(values)
    if values.size < 4:
        return np.nan
    return pearson_correlation(values[:-1], values[1:])


def linear_slope(time_s, values) -> float:
    time_s = np.asarray(time_s, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(time_s) & np.isfinite(values)
    time_s, values = time_s[valid], values[valid]
    if time_s.size < 2:
        return np.nan
    centered_time_s = time_s - time_s.mean()
    denominator = np.sum(centered_time_s ** 2)
    if denominator <= 0:
        return np.nan
    return float(np.sum(
        centered_time_s * (values - values.mean())) / denominator)


def best_lagged_correlation(gaze_values, ee_values, time_s, maximum_lag_s):
    """Best Pearson correlation over a +/- maximum_lag_s search -- port
    of every .m file's best_lagged_correlation(). A positive returned
    lag means an earlier gaze sample best matches a later end-effector
    sample; therefore, positive lag indicates gaze leading."""
    gaze_values = np.asarray(gaze_values, dtype=float).ravel()
    ee_values = np.asarray(ee_values, dtype=float).ravel()
    time_s = np.asarray(time_s, dtype=float).ravel()
    valid = np.isfinite(gaze_values) & np.isfinite(ee_values) & \
        np.isfinite(time_s)
    gaze_values, ee_values, time_s = \
        gaze_values[valid], ee_values[valid], time_s[valid]

    if time_s.size < 4:
        return np.nan, np.nan

    nominal_interval_s = np.median(np.diff(time_s))
    if not np.isfinite(nominal_interval_s) or nominal_interval_s <= 0:
        return np.nan, np.nan

    maximum_lag_samples = int(min(
        np.floor(maximum_lag_s / nominal_interval_s), time_s.size - 3))
    candidate_lags = np.arange(-maximum_lag_samples, maximum_lag_samples + 1)
    candidate_correlations = np.full(candidate_lags.size, np.nan)

    for lag_idx, lag_samples in enumerate(candidate_lags):
        if lag_samples >= 0:
            gaze_segment = gaze_values[:gaze_values.size - lag_samples]
            ee_segment = ee_values[lag_samples:]
        else:
            shift = -lag_samples
            gaze_segment = gaze_values[shift:]
            ee_segment = ee_values[:ee_values.size - shift]
        candidate_correlations[lag_idx] = pearson_correlation(
            gaze_segment, ee_segment)

    finite_candidates = np.nonzero(np.isfinite(candidate_correlations))[0]
    if finite_candidates.size == 0:
        return np.nan, np.nan

    local_best_idx = int(np.argmax(candidate_correlations[finite_candidates]))
    best_idx = finite_candidates[local_best_idx]
    best_correlation = float(candidate_correlations[best_idx])
    best_lag_s = float(candidate_lags[best_idx] * nominal_interval_s)
    return best_correlation, best_lag_s


def normalize_finite_to_unit_interval(values) -> np.ndarray:
    values = np.asarray(values, dtype=float).ravel()
    normalized = np.full(values.shape, np.nan)
    finite_mask = np.isfinite(values)
    finite = values[finite_mask]
    if finite.size == 0:
        return normalized
    lo, hi = finite.min(), finite.max()
    value_range = hi - lo
    if value_range <= 0:
        normalized[finite_mask] = 0.0
    else:
        normalized[finite_mask] = (finite - lo) / value_range
    return normalized


# ======================================================================
# Shared entropy helpers (fixed-box variant, used by the gaze bank --
# see the robot bank section below for the window-relative variant it
# uses instead, since robot position has no fixed physical box).
# ======================================================================

def entropy_from_probabilities(probabilities) -> float:
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    probabilities = probabilities[
        np.isfinite(probabilities) & (probabilities > 0)]
    if probabilities.size == 0:
        return 0.0
    return float(-np.sum(probabilities * np.log2(probabilities)))


def shannon_entropy_from_states(states, state_count: int) -> float:
    states = np.asarray(states, dtype=float).ravel()
    valid = np.isfinite(states) & (states >= 1) & (states <= state_count)
    states = np.round(states[valid]).astype(int)
    if states.size == 0:
        return np.nan
    counts = np.bincount(states, minlength=state_count + 1)[1:state_count + 1]
    probabilities = counts / counts.sum()
    return entropy_from_probabilities(probabilities)


def transition_shannon_entropy(states, state_count: int) -> float:
    states = np.asarray(states, dtype=float).ravel()
    valid = np.isfinite(states) & (states >= 1) & (states <= state_count)
    states = np.round(states[valid]).astype(int)
    if states.size < 2:
        return np.nan

    from_states = states[:-1] - 1
    to_states = states[1:] - 1
    transition_counts = np.zeros((state_count, state_count))
    np.add.at(transition_counts, (from_states, to_states), 1.0)
    from_counts = transition_counts.sum(axis=1)
    total_transitions = from_counts.sum()
    if total_transitions <= 0:
        return np.nan

    conditional_entropy = np.zeros(state_count)
    for state_idx in np.nonzero(from_counts > 0)[0]:
        row = transition_counts[state_idx]
        probabilities = row[row > 0] / from_counts[state_idx]
        conditional_entropy[state_idx] = entropy_from_probabilities(
            probabilities)

    weights = from_counts / total_transitions
    return float(np.sum(weights * conditional_entropy))


def _normalize_to_workspace_unit_box(x, y, center_xy, half_extent):
    """Map [center - half_extent, center + half_extent] -> [0, 1] per
    axis -- the calibrated-metres replacement for gaze_feature_
    extraction.m's assumption that Gx/Gy already live in [0, 1]."""
    cx, cy = center_xy
    nx = (x - (cx - half_extent)) / (2.0 * half_extent)
    ny = (y - (cy - half_extent)) / (2.0 * half_extent)
    return nx, ny


def spatial_gaze_states(
        x, y, center_xy, half_extent, bin_count_per_axis: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    nx, ny = _normalize_to_workspace_unit_box(
        x[valid], y[valid], center_xy, half_extent)
    nx = np.clip(nx, 0.0, 1.0 - 1e-12)
    ny = np.clip(ny, 0.0, 1.0 - 1e-12)
    x_bin = np.floor(nx * bin_count_per_axis) + 1
    y_bin = np.floor(ny * bin_count_per_axis) + 1
    return x_bin + (y_bin - 1) * bin_count_per_axis


def one_dimensional_display_entropy(
        values, center: float, half_extent: float, bin_count: int) -> float:
    values = finite_values(values)
    if values.size == 0:
        return np.nan
    normalized = (values - (center - half_extent)) / (2.0 * half_extent)
    clipped = np.clip(normalized, 0.0, 1.0 - 1e-12)
    states = np.floor(clipped * bin_count) + 1
    return shannon_entropy_from_states(states, bin_count)


# ======================================================================
# Shared outcome calculators (DSJ, mean planar force, mean planar
# speed) -- not part of any of the three feature banks, kept as
# separate diagnostic/outcome columns, matching each .m file's own
# window_results / outcome_table split.
# ======================================================================

def calculate_planar_dsj(x_m, y_m, time_s) -> float:
    """Dimensionless squared jerk over one window (paper Eq. 16, same
    formula this project's MATLAB DSJ chains use elsewhere) -- port of
    every .m file's calculate_planar_dsj(). Always from POSITION
    (differentiated three times), matching this project's DSJ
    convention project-wide -- see module docstring."""
    motion = calculate_planar_motion(x_m, y_m, time_s)
    duration_s = safe_range(motion["time_s"])
    path_length_m = planar_path_length(motion["x"], motion["y"])

    if motion["time_s"].size < 4 or not np.isfinite(duration_s) or \
            duration_s <= 0 or not np.isfinite(path_length_m) or \
            path_length_m <= 1e-12 or np.any(~np.isfinite(motion["jerk"])):
        return np.nan

    integrated_squared_jerk = np.trapz(motion["jerk"] ** 2, motion["time_s"])
    return float(
        duration_s ** 5 * integrated_squared_jerk / path_length_m ** 2)


def calculate_mean_planar_force(force_x_n, force_y_n) -> float:
    """DAMPING-CONFOUNDED -- see module docstring."""
    force_magnitude_n = np.hypot(
        np.asarray(force_x_n, dtype=float).ravel(),
        np.asarray(force_y_n, dtype=float).ravel())
    return safe_mean(force_magnitude_n)


def calculate_mean_planar_speed(x_m, y_m, time_s) -> float:
    """Mean speed over one window [m/s], from POSITION -- the
    force-free outcome/feature this project prefers over mean force
    where damping varies across pooled trials (see module docstring)."""
    return safe_mean(calculate_planar_motion(x_m, y_m, time_s)["speed"])


# ======================================================================
# Calibrated Tobii-UCS -> CoppeliaSim mapping. Ported from extract_
# pilot_features_fixed_adm_2d.m's detect_fixations_coppelia_xy(), the
# raw-gaze branch only (not its I-VT fixation clustering, which isn't
# needed here: every bank in this file wants a continuous per-sample
# gaze position, not fixations).
# ======================================================================

def map_gaze_ucs_to_coppelia(
        gaze_ucs_xy_mm: np.ndarray,
        tobii_reference_xy_mm: Tuple[float, float],
        coppelia_reference_xy_m: Tuple[float, float],
        screen_motion_per_robot_motion: float = 1.0,
        tobii_to_coppelia_axis_sign: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Per-axis affine map from Tobii user-coordinate-system gaze [mm]
    to the CoppeliaSim/robot-base frame [m]:

        coppelia_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion
        displacement_mm = gaze_ucs_xy_mm - tobii_reference_xy_mm
        displacement_m = displacement_mm * axis_sign * coppelia_m_per_tobii_mm
        gaze_xy_m = coppelia_reference_xy_m + displacement_m

    gaze_ucs_xy_mm is N-by-2 (or length-2); returns the same shape.
    """
    gaze_ucs_xy_mm = np.asarray(gaze_ucs_xy_mm, dtype=float)
    coppelia_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion
    displacement_mm = gaze_ucs_xy_mm - np.asarray(tobii_reference_xy_mm)
    displacement_m = displacement_mm * np.asarray(
        tobii_to_coppelia_axis_sign) * coppelia_m_per_tobii_mm
    return np.asarray(coppelia_reference_xy_m) + displacement_m


# ======================================================================
# GAZE BANK -- 102 features. Same features, same order, same names
# (the .m file's feature_NNN_<name> prefix stripped) as gaze_feature_
# extraction.m's feature_functions list. Every feature has the shared
# (Gx, Gy, Gt, Ex, Ey, Et) -> float interface: Gx/Gy gaze position [m]
# and its own time [s]; Ex/Ey/Et the robot position this gaze bank
# relates gaze to, and ITS time -- both already calibrated CoppeliaSim-
# frame metres by the time they reach here, not gaze_feature_
# extraction.m's raw normalized display coordinates (see module
# docstring for what that changes).
# ======================================================================

def synchronize_gaze_and_end_effector(Gx, Gy, Gt, Ex, Ey, Et):
    """Interpolate end-effector position onto gaze timestamps, keeping
    only the overlap of the two clocks -- port of gaze_feature_
    extraction.m's synchronize_gaze_and_end_effector()."""
    gx, gy, gaze_time_s = clean_planar_time_series(Gx, Gy, Gt)
    ex_source, ey_source, ee_time_s = clean_planar_time_series(Ex, Ey, Et)

    empty = np.array([])
    if gaze_time_s.size == 0 or ee_time_s.size < 2:
        return empty, empty, empty, empty, empty

    in_overlap = (gaze_time_s >= ee_time_s[0]) & (gaze_time_s <= ee_time_s[-1])
    gx, gy, time_s = gx[in_overlap], gy[in_overlap], gaze_time_s[in_overlap]
    ex = np.interp(time_s, ee_time_s, ex_source)
    ey = np.interp(time_s, ee_time_s, ey_source)

    valid = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(ex) & \
        np.isfinite(ey) & np.isfinite(time_s)
    return gx[valid], gy[valid], ex[valid], ey[valid], time_s[valid]


def relative_offsets(Gx, Gy, Gt, Ex, Ey, Et):
    gx, gy, ex, ey, time_s = synchronize_gaze_and_end_effector(
        Gx, Gy, Gt, Ex, Ey, Et)
    return gx - ex, gy - ey, time_s


def gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et) -> np.ndarray:
    relative_x, relative_y, _ = relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)
    return np.hypot(relative_x, relative_y)


def gaze_ee_velocity_cosine(Gx, Gy, Gt, Ex, Ey, Et) -> np.ndarray:
    gx, gy, ex, ey, time_s = synchronize_gaze_and_end_effector(
        Gx, Gy, Gt, Ex, Ey, Et)
    gaze_motion = calculate_planar_motion(gx, gy, time_s)
    ee_motion = calculate_planar_motion(ex, ey, time_s)
    speed_product = gaze_motion["speed"] * ee_motion["speed"]
    valid = np.isfinite(speed_product) & (speed_product > 1e-12)
    cosine_similarity = (
        gaze_motion["velocity_x"][valid] * ee_motion["velocity_x"][valid] +
        gaze_motion["velocity_y"][valid] * ee_motion["velocity_y"][valid]
    ) / speed_product[valid]
    return np.clip(cosine_similarity, -1.0, 1.0)


def build_gaze_feature_bank(
        config: FeatureConfig) -> List[Tuple[str, FeatureFunction]]:
    cx, cy = config.workspace_center_xy_m
    half_extent = config.workspace_half_extent_m
    low_v = config.low_speed_threshold_m_s
    high_v = config.high_speed_threshold_m_s
    close_a, close_b, close_c = config.close_thresholds_m
    bins_coarse = config.spatial_bin_count_coarse
    bins_fine = config.spatial_bin_count_fine

    bank: List[Tuple[str, FeatureFunction]] = []

    def add(name: str, fn: FeatureFunction) -> None:
        bank.append((name, fn))

    # --- 001-024: gaze position distribution ---
    add("mean_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_mean(Gx))
    add("mean_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_mean(Gy))
    add("std_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_std(Gx))
    add("std_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_std(Gy))
    add("variance_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_variance(Gx))
    add("variance_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_variance(Gy))
    add("minimum_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_minimum(Gx))
    add("maximum_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_maximum(Gx))
    add("range_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_range(Gx))
    add("minimum_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_minimum(Gy))
    add("maximum_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_maximum(Gy))
    add("range_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_range(Gy))
    add("median_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_median(Gx))
    add("median_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_median(Gy))
    add("iqr_gaze_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et: safe_interquartile_range(Gx))
    add("iqr_gaze_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et: safe_interquartile_range(Gy))
    add("mad_gaze_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et: safe_median_absolute_deviation(Gx))
    add("mad_gaze_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et: safe_median_absolute_deviation(Gy))
    add("skewness_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_skewness(Gx))
    add("skewness_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_skewness(Gy))
    add("kurtosis_gaze_x", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_kurtosis(Gx))
    add("kurtosis_gaze_y", lambda Gx, Gy, Gt, Ex, Ey, Et: safe_kurtosis(Gy))
    add("covariance_gaze_xy",
        lambda Gx, Gy, Gt, Ex, Ey, Et: safe_covariance(Gx, Gy))
    add("correlation_gaze_xy",
        lambda Gx, Gy, Gt, Ex, Ey, Et: pearson_correlation(Gx, Gy))

    # --- 025-029: dispersion / path ---
    def _gaze_dispersion_rms(Gx, Gy, Gt, Ex, Ey, Et):
        center_x, center_y = safe_mean(Gx), safe_mean(Gy)
        radius = np.hypot(np.asarray(Gx) - center_x, np.asarray(Gy) - center_y)
        return safe_root_mean_square(radius)
    add("gaze_dispersion_rms", _gaze_dispersion_rms)

    def _bcea_68(Gx, Gy, Gt, Ex, Ey, Et):
        sigma_x, sigma_y = safe_std(Gx), safe_std(Gy)
        rho_xy = pearson_correlation(Gx, Gy)
        if not np.isfinite(sigma_x) or not np.isfinite(sigma_y) or \
                not np.isfinite(rho_xy):
            return np.nan
        ellipse_factor = -np.log(1.0 - 0.68)
        return 2.0 * np.pi * ellipse_factor * sigma_x * sigma_y * \
            np.sqrt(max(0.0, 1.0 - rho_xy ** 2))
    add("bcea_68", _bcea_68)

    def _gaze_path_length(Gx, Gy, Gt, Ex, Ey, Et):
        x, y, _ = clean_planar_time_series(Gx, Gy, Gt)
        return planar_path_length(x, y)
    add("gaze_path_length", _gaze_path_length)

    def _gaze_net_displacement(Gx, Gy, Gt, Ex, Ey, Et):
        x, y, _ = clean_planar_time_series(Gx, Gy, Gt)
        return planar_net_displacement(x, y)
    add("gaze_net_displacement", _gaze_net_displacement)

    def _gaze_path_efficiency(Gx, Gy, Gt, Ex, Ey, Et):
        x, y, _ = clean_planar_time_series(Gx, Gy, Gt)
        return safe_ratio(
            planar_net_displacement(x, y), planar_path_length(x, y))
    add("gaze_path_efficiency", _gaze_path_efficiency)

    # --- 030-043: gaze speed / acceleration / jerk ---
    add("mean_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(calculate_planar_motion(Gx, Gy, Gt)["speed"]))
    add("std_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(calculate_planar_motion(Gx, Gy, Gt)["speed"]))
    add("max_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_maximum(calculate_planar_motion(Gx, Gy, Gt)["speed"]))
    add("median_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_median(calculate_planar_motion(Gx, Gy, Gt)["speed"]))
    add("p90_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            percentile_value(
                calculate_planar_motion(Gx, Gy, Gt)["speed"], 90.0))
    add("rms_gaze_speed",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_root_mean_square(calculate_planar_motion(Gx, Gy, Gt)["speed"]))
    add("mean_gaze_acceleration",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(calculate_planar_motion(Gx, Gy, Gt)["acceleration"]))
    add("std_gaze_acceleration",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(calculate_planar_motion(Gx, Gy, Gt)["acceleration"]))
    add("max_gaze_acceleration",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_maximum(calculate_planar_motion(Gx, Gy, Gt)["acceleration"]))
    add("rms_gaze_acceleration",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_root_mean_square(
                calculate_planar_motion(Gx, Gy, Gt)["acceleration"]))
    add("mean_gaze_jerk",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(calculate_planar_motion(Gx, Gy, Gt)["jerk"]))
    add("std_gaze_jerk",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(calculate_planar_motion(Gx, Gy, Gt)["jerk"]))
    add("max_gaze_jerk",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_maximum(calculate_planar_motion(Gx, Gy, Gt)["jerk"]))
    add("rms_gaze_jerk",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_root_mean_square(calculate_planar_motion(Gx, Gy, Gt)["jerk"]))

    # --- 044-047: velocity components ---
    add("mean_gaze_velocity_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(calculate_planar_motion(Gx, Gy, Gt)["velocity_x"]))
    add("mean_gaze_velocity_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(calculate_planar_motion(Gx, Gy, Gt)["velocity_y"]))
    add("std_gaze_velocity_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(calculate_planar_motion(Gx, Gy, Gt)["velocity_x"]))
    add("std_gaze_velocity_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(calculate_planar_motion(Gx, Gy, Gt)["velocity_y"]))

    # --- 048-050: direction ---
    add("mean_gaze_direction",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            circular_mean(calculate_planar_motion(Gx, Gy, Gt)["direction"]))
    add("gaze_direction_concentration",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            circular_concentration(
                calculate_planar_motion(Gx, Gy, Gt)["direction"]))

    def _mean_absolute_direction_change(Gx, Gy, Gt, Ex, Ey, Et):
        direction = calculate_planar_motion(Gx, Gy, Gt)["direction"]
        direction_change = np.arctan2(
            np.sin(np.diff(direction)), np.cos(np.diff(direction)))
        return safe_mean(np.abs(direction_change))
    add("mean_absolute_direction_change", _mean_absolute_direction_change)

    # --- 051-054: speed-threshold fractions / event counts ---
    add("low_speed_fraction",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((calculate_planar_motion(Gx, Gy, Gt)["speed"] < low_v)
                      .astype(float)))
    add("high_speed_fraction",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((calculate_planar_motion(Gx, Gy, Gt)["speed"] > high_v)
                      .astype(float)))
    add("high_speed_event_count",
        lambda Gx, Gy, Gt, Ex, Ey, Et: count_true_runs(
            calculate_planar_motion(Gx, Gy, Gt)["speed"] > high_v))
    add("low_speed_event_count",
        lambda Gx, Gy, Gt, Ex, Ey, Et: count_true_runs(
            calculate_planar_motion(Gx, Gy, Gt)["speed"] < low_v))

    # --- 055-057: gaze sample quality ---
    def _gaze_missing_time_fraction(Gx, Gy, Gt, Ex, Ey, Et):
        _, _, time_s = clean_planar_time_series(Gx, Gy, Gt)
        return missing_time_fraction(time_s)
    add("gaze_missing_time_fraction", _gaze_missing_time_fraction)

    def _gaze_gap_count(Gx, Gy, Gt, Ex, Ey, Et):
        _, _, time_s = clean_planar_time_series(Gx, Gy, Gt)
        return sample_gap_count(time_s)
    add("gaze_gap_count", _gaze_gap_count)

    def _gaze_sample_rate(Gx, Gy, Gt, Ex, Ey, Et):
        _, _, time_s = clean_planar_time_series(Gx, Gy, Gt)
        if time_s.size < 2 or time_s[-1] <= time_s[0]:
            return np.nan
        return (time_s.size - 1) / (time_s[-1] - time_s[0])
    add("gaze_sample_rate", _gaze_sample_rate)

    # --- 058-061: 10x10-equivalent entropy over the workspace box ---
    add("gaze_x_shannon_entropy",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            one_dimensional_display_entropy(Gx, cx, half_extent, bins_coarse))
    add("gaze_y_shannon_entropy",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            one_dimensional_display_entropy(Gy, cy, half_extent, bins_coarse))
    add("stationary_shannon_entropy",
        lambda Gx, Gy, Gt, Ex, Ey, Et: shannon_entropy_from_states(
            spatial_gaze_states(Gx, Gy, (cx, cy), half_extent, bins_coarse),
            bins_coarse ** 2))
    add("transitional_shannon_entropy",
        lambda Gx, Gy, Gt, Ex, Ey, Et: transition_shannon_entropy(
            spatial_gaze_states(Gx, Gy, (cx, cy), half_extent, bins_coarse),
            bins_coarse ** 2))

    # --- 062-065: workspace-center distance / bounds ---
    add("mean_distance_from_display_center",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(np.hypot(np.asarray(Gx) - cx, np.asarray(Gy) - cy)))
    add("std_distance_from_display_center",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(np.hypot(np.asarray(Gx) - cx, np.asarray(Gy) - cy)))
    add("display_center_fraction",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((np.hypot(np.asarray(Gx) - cx, np.asarray(Gy) - cy)
                       <= 0.5 * half_extent).astype(float)))

    def _outside_display_fraction(Gx, Gy, Gt, Ex, Ey, Et):
        Gx, Gy = np.asarray(Gx), np.asarray(Gy)
        outside = (Gx < cx - half_extent) | (Gx > cx + half_extent) | \
            (Gy < cy - half_extent) | (Gy > cy + half_extent)
        return safe_mean(outside.astype(float))
    add("outside_display_fraction", _outside_display_fraction)

    # --- 066-070: crossings / autocorrelation ---
    add("gaze_x_mean_crossing_count",
        lambda Gx, Gy, Gt, Ex, Ey, Et: mean_crossing_count(Gx))
    add("gaze_y_mean_crossing_count",
        lambda Gx, Gy, Gt, Ex, Ey, Et: mean_crossing_count(Gy))
    add("gaze_x_lag1_autocorrelation",
        lambda Gx, Gy, Gt, Ex, Ey, Et: lag_one_autocorrelation(Gx))
    add("gaze_y_lag1_autocorrelation",
        lambda Gx, Gy, Gt, Ex, Ey, Et: lag_one_autocorrelation(Gy))
    add("gaze_speed_lag1_autocorrelation",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            lag_one_autocorrelation(
                calculate_planar_motion(Gx, Gy, Gt)["speed"]))

    # --- 071-074: relative (gaze - EE) offsets ---
    add("mean_relative_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)[0]))
    add("mean_relative_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)[1]))
    add("std_relative_x",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)[0]))
    add("std_relative_y",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)[1]))

    # --- 075-086: gaze-to-EE distance stats / close-fraction bands ---
    add("mean_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))
    add("std_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_std(gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))
    add("median_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_median(gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))
    add("max_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_maximum(gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))
    add("rms_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_root_mean_square(
                gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))
    add("p90_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            percentile_value(
                gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et), 90.0))
    add("min_gaze_ee_distance",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_minimum(gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)))

    def _final_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et):
        distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)
        return float(distance[-1]) if distance.size else np.nan
    add("final_gaze_ee_distance", _final_gaze_ee_distance)

    def _gaze_ee_distance_slope(Gx, Gy, Gt, Ex, Ey, Et):
        relative_x, relative_y, time_s = relative_offsets(
            Gx, Gy, Gt, Ex, Ey, Et)
        return linear_slope(time_s, np.hypot(relative_x, relative_y))
    add("gaze_ee_distance_slope", _gaze_ee_distance_slope)

    add("fraction_within_0p05",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)
                       <= close_a).astype(float)))
    add("fraction_within_0p10",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)
                       <= close_b).astype(float)))
    add("fraction_within_0p20",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)
                       <= close_c).astype(float)))

    # --- 087-095: gaze-EE correlation / alignment ---
    def _gaze_ee_x_correlation(Gx, Gy, Gt, Ex, Ey, Et):
        gx, _, ex, _, _ = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        return pearson_correlation(gx, ex)
    add("gaze_ee_x_correlation", _gaze_ee_x_correlation)

    def _gaze_ee_y_correlation(Gx, Gy, Gt, Ex, Ey, Et):
        _, gy, _, ey, _ = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        return pearson_correlation(gy, ey)
    add("gaze_ee_y_correlation", _gaze_ee_y_correlation)

    def _gaze_ee_speed_correlation(Gx, Gy, Gt, Ex, Ey, Et):
        gx, gy, ex, ey, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        gaze_motion = calculate_planar_motion(gx, gy, time_s)
        ee_motion = calculate_planar_motion(ex, ey, time_s)
        return pearson_correlation(gaze_motion["speed"], ee_motion["speed"])
    add("gaze_ee_speed_correlation", _gaze_ee_speed_correlation)

    def _relative_path_length(Gx, Gy, Gt, Ex, Ey, Et):
        relative_x, relative_y, _ = relative_offsets(Gx, Gy, Gt, Ex, Ey, Et)
        return planar_path_length(relative_x, relative_y)
    add("relative_path_length", _relative_path_length)

    def _mean_relative_speed(Gx, Gy, Gt, Ex, Ey, Et):
        relative_x, relative_y, time_s = relative_offsets(
            Gx, Gy, Gt, Ex, Ey, Et)
        return safe_mean(
            calculate_planar_motion(relative_x, relative_y, time_s)["speed"])
    add("mean_relative_speed", _mean_relative_speed)

    def _std_relative_speed(Gx, Gy, Gt, Ex, Ey, Et):
        relative_x, relative_y, time_s = relative_offsets(
            Gx, Gy, Gt, Ex, Ey, Et)
        return safe_std(
            calculate_planar_motion(relative_x, relative_y, time_s)["speed"])
    add("std_relative_speed", _std_relative_speed)

    add("mean_velocity_cosine_similarity",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean(gaze_ee_velocity_cosine(Gx, Gy, Gt, Ex, Ey, Et)))
    add("aligned_velocity_fraction",
        lambda Gx, Gy, Gt, Ex, Ey, Et:
            safe_mean((gaze_ee_velocity_cosine(Gx, Gy, Gt, Ex, Ey, Et) > 0)
                      .astype(float)))

    def _gaze_to_ee_speed_ratio(Gx, Gy, Gt, Ex, Ey, Et):
        gx, gy, ex, ey, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        gaze_motion = calculate_planar_motion(gx, gy, time_s)
        ee_motion = calculate_planar_motion(ex, ey, time_s)
        return safe_ratio(
            safe_mean(gaze_motion["speed"]), safe_mean(ee_motion["speed"]))
    add("gaze_to_ee_speed_ratio", _gaze_to_ee_speed_ratio)

    # --- 096-100: cross-correlation / lead-lag ---
    def _max_x_cross_correlation(Gx, Gy, Gt, Ex, Ey, Et):
        gx, _, ex, _, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        value, _ = best_lagged_correlation(gx, ex, time_s, 0.25)
        return value
    add("max_x_cross_correlation", _max_x_cross_correlation)

    def _x_lead_at_max_correlation_s(Gx, Gy, Gt, Ex, Ey, Et):
        gx, _, ex, _, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        _, lag_s = best_lagged_correlation(gx, ex, time_s, 0.25)
        return lag_s
    add("x_lead_at_max_correlation_s", _x_lead_at_max_correlation_s)

    def _max_y_cross_correlation(Gx, Gy, Gt, Ex, Ey, Et):
        _, gy, _, ey, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        value, _ = best_lagged_correlation(gy, ey, time_s, 0.25)
        return value
    add("max_y_cross_correlation", _max_y_cross_correlation)

    def _y_lead_at_max_correlation_s(Gx, Gy, Gt, Ex, Ey, Et):
        _, gy, _, ey, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        _, lag_s = best_lagged_correlation(gy, ey, time_s, 0.25)
        return lag_s
    add("y_lead_at_max_correlation_s", _y_lead_at_max_correlation_s)

    def _mean_gaze_lead_along_ee_motion(Gx, Gy, Gt, Ex, Ey, Et):
        gx, gy, ex, ey, time_s = synchronize_gaze_and_end_effector(
            Gx, Gy, Gt, Ex, Ey, Et)
        ee_motion = calculate_planar_motion(ex, ey, time_s)
        ee_speed = ee_motion["speed"]
        moving = ee_speed > 1e-9
        along_path_lead = np.full(ee_speed.shape, np.nan)
        along_path_lead[moving] = (
            (gx[moving] - ex[moving]) * ee_motion["velocity_x"][moving] +
            (gy[moving] - ey[moving]) * ee_motion["velocity_y"][moving]
        ) / ee_speed[moving]
        return safe_mean(along_path_lead)
    add("mean_gaze_lead_along_ee_motion", _mean_gaze_lead_along_ee_motion)

    # --- 101-102: 40x40-equivalent entropy over the workspace box ---
    add("stationary_shannon_entropy_40x40",
        lambda Gx, Gy, Gt, Ex, Ey, Et: shannon_entropy_from_states(
            spatial_gaze_states(Gx, Gy, (cx, cy), half_extent, bins_fine),
            bins_fine ** 2))
    add("transitional_shannon_entropy_40x40",
        lambda Gx, Gy, Gt, Ex, Ey, Et: transition_shannon_entropy(
            spatial_gaze_states(Gx, Gy, (cx, cy), half_extent, bins_fine),
            bins_fine ** 2))

    assert len(bank) == 102, f"expected 102 gaze features, built {len(bank)}"
    return bank


# ======================================================================
# ROBOT BANK -- 96 features, no gaze data involved at all. Same
# features, same order, same names as robot_feature_extraction.m's
# feature_functions list. Every feature has the shared (Px, Py, Pt, Vx,
# Vy, Fx, Fy, Wmag, Vcx, Vcy) -> float interface: Px/Py robot position
# (CoppeliaSim frame) [m] and its time [s]; Vx/Vy MEASURED TCP velocity
# [m/s] (not re-differentiated from position); Fx/Fy filtered planar
# force [N]; Wmag 3-axis angular speed magnitude [rad/s]; Vcx/Vcy
# COMMANDED planar velocity [m/s] (the admittance controller's own
# setpoint). FORCE- and COMMAND-TRACKING-derived features are DAMPING-
# CONFOUNDED -- see module docstring.
# ======================================================================

def calculate_robot_motion(Vx, Vy, Pt) -> dict:
    """Speed, acceleration, jerk, direction, and signed curvature from
    MEASURED velocity -- port of robot_feature_extraction.m's
    calculate_robot_motion(). One derivative closer to what was
    actually measured than differentiating position twice, matching
    this project's other robot derivative chains (compute_motion_
    derivatives in extract_pilot_features_fixed_adm_2d.m).

    Signed curvature (same convention used throughout this project):

        kappa = (vx*ay - vy*ax) / |v|^3

    set to NaN wherever |v| < MINIMUM_SPEED_M_S, where direction (and
    therefore curvature) is undefined rather than just small."""
    MINIMUM_SPEED_M_S = 0.005

    Vx, Vy, Pt = clean_planar_time_series(Vx, Vy, Pt)
    n = Pt.size
    motion = {
        "time_s": Pt, "velocity_x": Vx, "velocity_y": Vy,
        "speed": np.hypot(Vx, Vy),
        "acceleration": np.full(n, np.nan), "jerk": np.full(n, np.nan),
        "direction": np.full(n, np.nan), "curvature": np.full(n, np.nan),
        "curvature_dot": np.full(n, np.nan),
    }
    moving = motion["speed"] > 1e-12
    direction = np.full(n, np.nan)
    direction[moving] = np.arctan2(Vy[moving], Vx[moving])
    motion["direction"] = direction

    if n < 3:
        return motion

    acceleration_x = np.gradient(Vx, Pt)
    acceleration_y = np.gradient(Vy, Pt)
    motion["acceleration"] = np.hypot(acceleration_x, acceleration_y)

    curvature = (Vx * acceleration_y - Vy * acceleration_x) / \
        motion["speed"] ** 3
    curvature[motion["speed"] < MINIMUM_SPEED_M_S] = np.nan
    motion["curvature"] = curvature
    motion["curvature_dot"] = np.gradient(curvature, Pt)

    if n < 4:
        return motion
    jerk_x = np.gradient(acceleration_x, Pt)
    jerk_y = np.gradient(acceleration_y, Pt)
    motion["jerk"] = np.hypot(jerk_x, jerk_y)
    return motion


def fit_power_law(window_speed_m_s, window_curvature_1_m):
    """Least-squares fit of v = K*kappa^(-beta) within one window, via
    linear regression of log(v) on log(kappa) -- port of robot_
    feature_extraction.m's fit_power_law() (same regression this
    project's other power-law estimators use, e.g. extract_pilot_
    features_fixed_adm_2d.m's fit_power_law_r2()). Returns (r_squared,
    beta); (NaN, NaN) if too few valid samples fall in this window."""
    MINIMUM_SPEED_M_S = 0.005
    MINIMUM_SAMPLE_COUNT = 10

    window_speed_m_s = np.asarray(window_speed_m_s, dtype=float)
    window_curvature_1_m = np.asarray(window_curvature_1_m, dtype=float)
    valid_mask = (window_speed_m_s > MINIMUM_SPEED_M_S) & \
        np.isfinite(window_curvature_1_m) & (window_curvature_1_m != 0)

    log_speed = np.log(window_speed_m_s[valid_mask])
    log_curvature = np.log(np.abs(window_curvature_1_m[valid_mask]))

    sample_count = log_speed.size
    if sample_count < MINIMUM_SAMPLE_COUNT:
        return np.nan, np.nan

    sum_x = np.sum(log_curvature)
    sum_y = np.sum(log_speed)
    sum_xx = np.sum(log_curvature ** 2)
    sum_xy = np.sum(log_curvature * log_speed)

    denominator = sample_count * sum_xx - sum_x ** 2
    if abs(denominator) < 1e-9:
        return np.nan, np.nan

    slope = (sample_count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / sample_count
    beta = -slope

    predicted_log_speed = intercept + slope * log_curvature
    residual_sum_squares = np.sum((log_speed - predicted_log_speed) ** 2)
    total_sum_squares = np.sum((log_speed - log_speed.mean()) ** 2)

    if total_sum_squares < 1e-12:
        r_squared = np.nan
    else:
        r_squared = 1.0 - residual_sum_squares / total_sum_squares
    return float(r_squared), float(beta)


def one_dimensional_entropy_over_range(values, bin_count: int) -> float:
    """Shannon entropy of bin_count equal-width bins spanning THIS
    WINDOW's own [min, max] range [bits] -- port of robot_feature_
    extraction.m's one_dimensional_entropy_over_range(). Unlike the
    gaze bank's one_dimensional_display_entropy(), robot position has
    no fixed physical box to bin against, so the range is taken from
    the data itself -- a window that barely moved shows near-zero
    entropy almost by construction (nearly everything in one bin),
    which is the intended behaviour, not a defect."""
    values = finite_values(values)
    if values.size < 2:
        return np.nan

    minimum_value = values.min()
    value_range = values.max() - minimum_value
    if value_range <= 0:
        return 0.0  # every sample in exactly one bin -> zero entropy

    clipped_values = np.clip(
        (values - minimum_value) / value_range, 0.0, 1.0 - 1e-12)
    states = np.floor(clipped_values * bin_count) + 1
    return shannon_entropy_from_states(states, bin_count)


def spatial_states_over_range(x, y, bin_count_per_axis: int) -> np.ndarray:
    """Convert position into 2D grid state IDs, binning each axis over
    THIS WINDOW's own [min, max] range -- port of robot_feature_
    extraction.m's spatial_states_over_range(), the robot-position
    analogue of the gaze bank's spatial_gaze_states() (which bins over
    a fixed [0, 1] display instead -- robot position has no equivalent
    fixed box, see one_dimensional_entropy_over_range() above)."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.array([])

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    nx = (x - x.min()) / x_range if x_range > 0 else np.zeros_like(x)
    ny = (y - y.min()) / y_range if y_range > 0 else np.zeros_like(y)
    nx = np.clip(nx, 0.0, 1.0 - 1e-12)
    ny = np.clip(ny, 0.0, 1.0 - 1e-12)

    x_bin = np.floor(nx * bin_count_per_axis) + 1
    y_bin = np.floor(ny * bin_count_per_axis) + 1
    return x_bin + (y_bin - 1) * bin_count_per_axis


def build_robot_feature_bank(
        config: FeatureConfig) -> List[Tuple[str, FeatureFunction]]:
    low_v = config.robot_low_speed_threshold_m_s
    high_v = config.robot_high_speed_threshold_m_s
    bins_coarse = config.robot_spatial_bin_count_coarse
    bins_fine = config.robot_spatial_bin_count_fine

    bank: List[Tuple[str, FeatureFunction]] = []

    def add(name: str, fn: FeatureFunction) -> None:
        bank.append((name, fn))

    # --- 001-024: position distribution ---
    add("mean_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Px))
    add("mean_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Py))
    add("std_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Px))
    add("std_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Py))
    add("variance_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_variance(Px))
    add("variance_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_variance(Py))
    add("minimum_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_minimum(Px))
    add("maximum_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_maximum(Px))
    add("range_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_range(Px))
    add("minimum_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_minimum(Py))
    add("maximum_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_maximum(Py))
    add("range_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_range(Py))
    add("median_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_median(Px))
    add("median_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_median(Py))
    add("iqr_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_interquartile_range(Px))
    add("iqr_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_interquartile_range(Py))
    add("mad_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median_absolute_deviation(Px))
    add("mad_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median_absolute_deviation(Py))
    add("skewness_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_skewness(Px))
    add("skewness_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_skewness(Py))
    add("kurtosis_position_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_kurtosis(Px))
    add("kurtosis_position_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_kurtosis(Py))
    add("covariance_position_xy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_covariance(Px, Py))
    add("correlation_position_xy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            pearson_correlation(Px, Py))

    # --- 025-029: dispersion / path ---
    def _position_dispersion_rms(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        center_x, center_y = safe_mean(Px), safe_mean(Py)
        radius = np.hypot(np.asarray(Px) - center_x, np.asarray(Py) - center_y)
        return safe_root_mean_square(radius)
    add("position_dispersion_rms", _position_dispersion_rms)

    def _position_bcea_68(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        sigma_x, sigma_y = safe_std(Px), safe_std(Py)
        rho_xy = pearson_correlation(Px, Py)
        if not np.isfinite(sigma_x) or not np.isfinite(sigma_y) or \
                not np.isfinite(rho_xy):
            return np.nan
        ellipse_factor = -np.log(1.0 - 0.68)
        return 2.0 * np.pi * ellipse_factor * sigma_x * sigma_y * \
            np.sqrt(max(0.0, 1.0 - rho_xy ** 2))
    add("position_bcea_68", _position_bcea_68)

    def _path_length(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        x, y, _ = clean_planar_time_series(Px, Py, Pt)
        return planar_path_length(x, y)
    add("path_length", _path_length)

    def _net_displacement(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        x, y, _ = clean_planar_time_series(Px, Py, Pt)
        return planar_net_displacement(x, y)
    add("net_displacement", _net_displacement)

    def _path_efficiency(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        x, y, _ = clean_planar_time_series(Px, Py, Pt)
        return safe_ratio(
            planar_net_displacement(x, y), planar_path_length(x, y))
    add("path_efficiency", _path_efficiency)

    # --- 030-043: speed / acceleration / jerk ---
    add("mean_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(calculate_robot_motion(Vx, Vy, Pt)["speed"]))
    add("std_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(calculate_robot_motion(Vx, Vy, Pt)["speed"]))
    add("max_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(calculate_robot_motion(Vx, Vy, Pt)["speed"]))
    add("median_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median(calculate_robot_motion(Vx, Vy, Pt)["speed"]))
    add("p90_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            percentile_value(
                calculate_robot_motion(Vx, Vy, Pt)["speed"], 90.0))
    add("rms_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(calculate_robot_motion(Vx, Vy, Pt)["speed"]))
    add("mean_acceleration",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(calculate_robot_motion(Vx, Vy, Pt)["acceleration"]))
    add("std_acceleration",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(calculate_robot_motion(Vx, Vy, Pt)["acceleration"]))
    add("max_acceleration",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(calculate_robot_motion(Vx, Vy, Pt)["acceleration"]))
    add("rms_acceleration",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(
                calculate_robot_motion(Vx, Vy, Pt)["acceleration"]))
    add("mean_jerk",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(calculate_robot_motion(Vx, Vy, Pt)["jerk"]))
    add("std_jerk",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(calculate_robot_motion(Vx, Vy, Pt)["jerk"]))
    add("max_jerk",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(calculate_robot_motion(Vx, Vy, Pt)["jerk"]))
    add("rms_jerk",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(calculate_robot_motion(Vx, Vy, Pt)["jerk"]))

    # --- 044-047: velocity components ---
    add("mean_velocity_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Vx))
    add("mean_velocity_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Vy))
    add("std_velocity_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Vx))
    add("std_velocity_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Vy))

    # --- 048-050: direction ---
    add("mean_direction",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            circular_mean(calculate_robot_motion(Vx, Vy, Pt)["direction"]))
    add("direction_concentration",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            circular_concentration(
                calculate_robot_motion(Vx, Vy, Pt)["direction"]))

    def _mean_absolute_direction_change(
            Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        direction = calculate_robot_motion(Vx, Vy, Pt)["direction"]
        direction_change = np.arctan2(
            np.sin(np.diff(direction)), np.cos(np.diff(direction)))
        return safe_mean(np.abs(direction_change))
    add("mean_absolute_direction_change", _mean_absolute_direction_change)

    # --- 051-054: speed-threshold fractions / event counts ---
    add("low_speed_fraction",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((calculate_robot_motion(Vx, Vy, Pt)["speed"] < low_v)
                      .astype(float)))
    add("high_speed_fraction",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((calculate_robot_motion(Vx, Vy, Pt)["speed"] > high_v)
                      .astype(float)))
    add("high_speed_event_count",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: count_true_runs(
            calculate_robot_motion(Vx, Vy, Pt)["speed"] > high_v))
    add("low_speed_event_count",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: count_true_runs(
            calculate_robot_motion(Vx, Vy, Pt)["speed"] < low_v))

    # --- 055-057: sample quality ---
    def _missing_time_fraction(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        _, _, time_s = clean_planar_time_series(Px, Py, Pt)
        return missing_time_fraction(time_s)
    add("missing_time_fraction", _missing_time_fraction)

    def _gap_count(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        _, _, time_s = clean_planar_time_series(Px, Py, Pt)
        return sample_gap_count(time_s)
    add("gap_count", _gap_count)

    def _sample_rate(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        _, _, time_s = clean_planar_time_series(Px, Py, Pt)
        if time_s.size < 2 or time_s[-1] <= time_s[0]:
            return np.nan
        return (time_s.size - 1) / (time_s[-1] - time_s[0])
    add("sample_rate", _sample_rate)

    # --- 058-061: 10x10-equivalent entropy over this window's own range ---
    add("position_x_shannon_entropy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            one_dimensional_entropy_over_range(Px, bins_coarse))
    add("position_y_shannon_entropy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            one_dimensional_entropy_over_range(Py, bins_coarse))
    add("stationary_shannon_entropy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            shannon_entropy_from_states(
                spatial_states_over_range(Px, Py, bins_coarse),
                bins_coarse ** 2))
    add("transitional_shannon_entropy",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            transition_shannon_entropy(
                spatial_states_over_range(Px, Py, bins_coarse),
                bins_coarse ** 2))

    # --- 062-066: crossings / autocorrelation ---
    add("position_x_mean_crossing_count",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            mean_crossing_count(Px))
    add("position_y_mean_crossing_count",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            mean_crossing_count(Py))
    add("position_x_lag1_autocorrelation",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            lag_one_autocorrelation(Px))
    add("position_y_lag1_autocorrelation",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            lag_one_autocorrelation(Py))
    add("speed_lag1_autocorrelation",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            lag_one_autocorrelation(
                calculate_robot_motion(Vx, Vy, Pt)["speed"]))

    # --- 067-073: curvature ---
    add("mean_curvature",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(calculate_robot_motion(Vx, Vy, Pt)["curvature"]))
    add("std_curvature",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(calculate_robot_motion(Vx, Vy, Pt)["curvature"]))
    add("median_absolute_curvature",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median(np.abs(
                calculate_robot_motion(Vx, Vy, Pt)["curvature"])))
    add("max_absolute_curvature",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(np.abs(
                calculate_robot_motion(Vx, Vy, Pt)["curvature"])))
    add("mean_curvature_dot",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(calculate_robot_motion(Vx, Vy, Pt)["curvature_dot"]))
    add("std_curvature_dot",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(calculate_robot_motion(Vx, Vy, Pt)["curvature_dot"]))

    def _curvature_sign_reversal_rate(
            Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        curvature = calculate_robot_motion(Vx, Vy, Pt)["curvature"]
        valid_curvature = curvature[np.isfinite(curvature)]
        x, y, _ = clean_planar_time_series(Px, Py, Pt)
        distance_travelled_m = planar_path_length(x, y)
        if valid_curvature.size < 2 or not np.isfinite(distance_travelled_m) \
                or distance_travelled_m <= 0:
            return np.nan
        sign_changes = np.sum(np.diff(np.sign(valid_curvature)) != 0)
        return float(sign_changes) / distance_travelled_m
    add("curvature_sign_reversal_rate", _curvature_sign_reversal_rate)

    # --- 074-075: power-law fit ---
    def _power_law_r2(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        motion = calculate_robot_motion(Vx, Vy, Pt)
        r_squared, _ = fit_power_law(motion["speed"], motion["curvature"])
        return r_squared
    add("power_law_r2", _power_law_r2)

    def _power_law_exponent(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        motion = calculate_robot_motion(Vx, Vy, Pt)
        _, beta = fit_power_law(motion["speed"], motion["curvature"])
        return beta
    add("power_law_exponent", _power_law_exponent)

    # --- 076-080: angular speed ---
    add("mean_angular_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Wmag))
    add("std_angular_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Wmag))
    add("max_angular_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_maximum(Wmag))
    add("rms_angular_speed",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(Wmag))
    add("angular_speed_lag1_autocorrelation",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            lag_one_autocorrelation(Wmag))

    # --- 081-082: 40x40-equivalent entropy over this window's own range ---
    add("stationary_shannon_entropy_40x40",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            shannon_entropy_from_states(
                spatial_states_over_range(Px, Py, bins_fine), bins_fine ** 2))
    add("transitional_shannon_entropy_40x40",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            transition_shannon_entropy(
                spatial_states_over_range(Px, Py, bins_fine), bins_fine ** 2))

    # --- 083-091: force. DAMPING-CONFOUNDED -- see module docstring. ---
    add("mean_force_magnitude",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(np.hypot(np.asarray(Fx), np.asarray(Fy))))
    add("std_force_magnitude",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(np.hypot(np.asarray(Fx), np.asarray(Fy))))
    add("max_force_magnitude",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(np.hypot(np.asarray(Fx), np.asarray(Fy))))
    add("median_force_magnitude",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median(np.hypot(np.asarray(Fx), np.asarray(Fy))))
    add("rms_force_magnitude",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(np.hypot(np.asarray(Fx), np.asarray(Fy))))
    add("mean_force_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Fx))
    add("mean_force_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_mean(Fy))
    add("std_force_x",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Fx))
    add("std_force_y",
        lambda Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy: safe_std(Fy))

    # --- 092-093: power. DAMPING-CONFOUNDED -- inherits force's confound. ---
    def _mean_power(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        power_w = np.asarray(Fx) * np.asarray(Vx) + \
            np.asarray(Fy) * np.asarray(Vy)
        return safe_mean(power_w)
    add("mean_power", _mean_power)

    def _std_power(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        power_w = np.asarray(Fx) * np.asarray(Vx) + \
            np.asarray(Fy) * np.asarray(Vy)
        return safe_std(power_w)
    add("std_power", _std_power)

    # --- 094-096: command tracking. DAMPING-CONFOUNDED -- see module
    # docstring. ---
    def _mean_tracking_error(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        tracking_error = np.hypot(
            np.asarray(Vcx) - np.asarray(Vx), np.asarray(Vcy) - np.asarray(Vy))
        return safe_mean(tracking_error)
    add("mean_tracking_error", _mean_tracking_error)

    def _std_tracking_error(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        tracking_error = np.hypot(
            np.asarray(Vcx) - np.asarray(Vx), np.asarray(Vcy) - np.asarray(Vy))
        return safe_std(tracking_error)
    add("std_tracking_error", _std_tracking_error)

    def _commanded_actual_speed_correlation(
            Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        commanded_speed = np.hypot(np.asarray(Vcx), np.asarray(Vcy))
        actual_speed = np.hypot(np.asarray(Vx), np.asarray(Vy))
        return pearson_correlation(commanded_speed, actual_speed)
    add("commanded_actual_speed_correlation",
        _commanded_actual_speed_correlation)

    assert len(bank) == 96, f"expected 96 robot features, built {len(bank)}"
    return bank


# ======================================================================
# JOINT BANK -- 51 features that genuinely need BOTH gaze and robot
# data (remove either and the feature is undefined, not just unused).
# Same features, same order, same names as joint_feature_extraction.
# m's feature_functions list. Every feature has the shared (Gx, Gy, Gt,
# Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy) -> float interface --
# gaze position/time plus every robot signal the robot bank uses.
# FORCE- and COMMAND-TRACKING-derived features are DAMPING-CONFOUNDED
# -- see module docstring.
# ======================================================================

def synchronize_robot_onto_gaze(
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy) -> dict:
    """Interpolate every robot-side raw and derived signal onto gaze
    timestamps, restricted to the time overlap of the two clocks --
    port of joint_feature_extraction.m's synchronize_robot_onto_gaze()
    (which generalizes the gaze bank's own synchronize_gaze_and_end_
    effector(), carrying only position, to also carry speed, curvature,
    force, angular speed, and tracking error). Speed/curvature are
    differentiated from Vx/Vy at the robot's own native (much higher)
    sample rate first (via calculate_robot_motion), and only
    interpolated onto the sparser gaze clock afterward -- never the
    other way around, so they stay as well-resolved as the robot
    bank's own versions of the same quantities.

    Unlike gaze_features.synchronize_gaze_and_end_effector (which
    restricts its query points to its OWN interpolation source's exact
    range, so np.interp's default out-of-range CLAMPING never actually
    triggers), this function interpolates against TWO different-length
    robot arrays (motion's, and the raw Px/Py/Fx/Fy/Wmag/Vcx/Vcy's) --
    left=right=np.nan is passed explicitly so any point outside either
    one's own range reads as missing, matching the .m file's interp1()
    (which returns NaN outside range by default) instead of silently
    clamping to a boundary value."""
    field_names = (
        "gaze_x", "gaze_y", "time_s", "robot_x", "robot_y", "speed",
        "curvature", "velocity_x", "velocity_y", "force_x", "force_y",
        "force_magnitude", "angular_speed", "tracking_error")

    gx, gy, gaze_time_s = clean_planar_time_series(Gx, Gy, Gt)
    robot_index = clean_time_indices(Pt)
    if robot_index.size < 2 or gaze_time_s.size == 0:
        return {name: np.array([]) for name in field_names}
    robot_time_s = np.asarray(Pt, dtype=float)[robot_index]

    motion = calculate_robot_motion(Vx, Vy, Pt)
    # motion's own internal cleaning (inside calculate_robot_motion, via
    # clean_planar_time_series on Vx, Vy, Pt) applies the IDENTICAL
    # sort/dedupe rule as clean_time_indices(Pt) above, so motion.
    # time_s and robot_time_s coincide sample-for-sample -- interpolate
    # speed/curvature against motion["time_s"] directly rather than
    # re-deriving that alignment.

    in_overlap = (gaze_time_s >= motion["time_s"][0]) & \
        (gaze_time_s <= motion["time_s"][-1])
    gt = gaze_time_s[in_overlap]
    gx = gx[in_overlap]
    gy = gy[in_overlap]
    if gt.size == 0:
        return {name: np.array([]) for name in field_names}

    Px_c = np.asarray(Px, dtype=float)[robot_index]
    Py_c = np.asarray(Py, dtype=float)[robot_index]
    Fx_c = np.asarray(Fx, dtype=float)[robot_index]
    Fy_c = np.asarray(Fy, dtype=float)[robot_index]
    Wmag_c = np.asarray(Wmag, dtype=float)[robot_index]
    Vcx_c = np.asarray(Vcx, dtype=float)[robot_index]
    Vcy_c = np.asarray(Vcy, dtype=float)[robot_index]

    def interp(xp, fp):
        return np.interp(gt, xp, fp, left=np.nan, right=np.nan)

    px_i = interp(robot_time_s, Px_c)
    py_i = interp(robot_time_s, Py_c)
    speed_i = interp(motion["time_s"], motion["speed"])
    curvature_i = interp(motion["time_s"], motion["curvature"])
    velocity_x_i = interp(motion["time_s"], motion["velocity_x"])
    velocity_y_i = interp(motion["time_s"], motion["velocity_y"])
    force_x_i = interp(robot_time_s, Fx_c)
    force_y_i = interp(robot_time_s, Fy_c)
    angular_speed_i = interp(robot_time_s, Wmag_c)
    commanded_x_i = interp(robot_time_s, Vcx_c)
    commanded_y_i = interp(robot_time_s, Vcy_c)

    valid = np.isfinite(gx) & np.isfinite(gy) & np.isfinite(px_i) & \
        np.isfinite(py_i)

    return {
        "gaze_x": gx[valid], "gaze_y": gy[valid], "time_s": gt[valid],
        "robot_x": px_i[valid], "robot_y": py_i[valid],
        "speed": speed_i[valid], "curvature": curvature_i[valid],
        "velocity_x": velocity_x_i[valid], "velocity_y": velocity_y_i[valid],
        "force_x": force_x_i[valid], "force_y": force_y_i[valid],
        "force_magnitude": np.hypot(force_x_i[valid], force_y_i[valid]),
        "angular_speed": angular_speed_i[valid],
        "tracking_error": np.hypot(
            commanded_x_i[valid] - velocity_x_i[valid],
            commanded_y_i[valid] - velocity_y_i[valid]),
    }


def curvature_conditioned_on_distance(distance, curvature, want_close: bool):
    """Mean |curvature| restricted to the half of the (paired) samples
    either below (want_close=True) or above (False) this window's own
    median gaze-to-hand distance -- port of joint_feature_extraction.
    m's curvature_conditioned_on_distance()."""
    distance = np.asarray(distance, dtype=float)
    curvature = np.asarray(curvature, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(curvature)
    distance, curvature = distance[valid], curvature[valid]
    if distance.size < 2:
        return np.nan

    median_distance = np.median(distance)
    selected = distance <= median_distance if want_close else \
        distance > median_distance
    if not np.any(selected):
        return np.nan
    return float(np.mean(np.abs(curvature[selected])))


def value_conditioned_on_reference(values, reference, want_low_reference: bool):
    """Mean of values restricted to the half of the (paired) samples
    either below (want_low_reference=True) or above (False) this
    window's own median of reference -- port of joint_feature_
    extraction.m's value_conditioned_on_reference()."""
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(values) & np.isfinite(reference)
    values, reference = values[valid], reference[valid]
    if values.size < 2:
        return np.nan

    median_reference = np.median(reference)
    selected = reference <= median_reference if want_low_reference else \
        reference > median_reference
    if not np.any(selected):
        return np.nan
    return float(np.mean(values[selected]))


def fraction_close_during_high_reference(
        distance, reference, close_threshold_m):
    """Of the (paired) samples above this window's own median of
    reference, the fraction where distance was also <=
    close_threshold_m -- port of joint_feature_extraction.m's
    fraction_close_during_high_reference()."""
    distance = np.asarray(distance, dtype=float)
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(distance) & np.isfinite(reference)
    distance, reference = distance[valid], reference[valid]
    if distance.size < 2:
        return np.nan

    high_reference = reference > np.median(reference)
    if not np.any(high_reference):
        return np.nan
    return float(np.mean(distance[high_reference] <= close_threshold_m))


def signed_angle_series(reference_x, reference_y, target_x, target_y):
    """Per-sample signed angle from (reference_x, reference_y) to
    (target_x, target_y) [deg], via atan2(cross, dot) -- port of
    joint_feature_extraction.m's signed_angle_series() (the vectorized,
    per-sample analogue of extract_pilot_features_fixed_adm_2d.m's
    angle_between_vectors_deg()). NaN wherever either vector is
    (numerically) zero-length, where direction is undefined."""
    reference_x = np.asarray(reference_x, dtype=float)
    reference_y = np.asarray(reference_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    target_y = np.asarray(target_y, dtype=float)

    reference_norm = np.hypot(reference_x, reference_y)
    target_norm = np.hypot(target_x, target_y)

    cross_value = reference_x * target_y - reference_y * target_x
    dot_value = reference_x * target_x + reference_y * target_y
    angle_deg = np.degrees(np.arctan2(cross_value, dot_value))
    angle_deg[(reference_norm < 1e-9) | (target_norm < 1e-9)] = np.nan
    return angle_deg


def lookahead_distance_series(
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
    """Signed projection of (gaze - hand) onto the hand's own (unit)
    velocity direction [m], per synchronized sample -- port of joint_
    feature_extraction.m's lookahead_distance_series(). Positive means
    gaze sits ahead of the hand along its current heading. NaN wherever
    speed is below this project's usual 0.005 m/s at-rest floor
    (calculate_robot_motion()'s own MINIMUM_SPEED_M_S), where direction
    is undefined."""
    MINIMUM_SPEED_M_S = 0.005
    synced = synchronize_robot_onto_gaze(
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
    offset_x = synced["gaze_x"] - synced["robot_x"]
    offset_y = synced["gaze_y"] - synced["robot_y"]

    lookahead = np.full(synced["speed"].shape, np.nan)
    moving = synced["speed"] >= MINIMUM_SPEED_M_S
    unit_velocity_x = synced["velocity_x"][moving] / synced["speed"][moving]
    unit_velocity_y = synced["velocity_y"][moving] / synced["speed"][moving]
    lookahead[moving] = offset_x[moving] * unit_velocity_x + \
        offset_y[moving] * unit_velocity_y
    return lookahead


def joint_gaze_ee_velocity_cosine(
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
    """Cosine similarity of gaze velocity and hand velocity, in
    [-1, 1] -- the joint bank's own version of gaze_ee_velocity_cosine,
    matching the joint bank's larger (Gx...Vcy) interface (see module
    docstring for why the gaze bank can't share this one directly)."""
    synced = synchronize_robot_onto_gaze(
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
    gaze_motion = calculate_robot_motion(
        synced["gaze_x"], synced["gaze_y"], synced["time_s"])

    speed_product = gaze_motion["speed"] * synced["speed"]
    valid = np.isfinite(speed_product) & (speed_product > 1e-12)
    cosine_similarity = (
        gaze_motion["velocity_x"][valid] * synced["velocity_x"][valid] +
        gaze_motion["velocity_y"][valid] * synced["velocity_y"][valid]
    ) / speed_product[valid]
    return np.clip(cosine_similarity, -1.0, 1.0)


def build_joint_feature_bank(
        config: FeatureConfig) -> List[Tuple[str, FeatureFunction]]:
    close_m = config.joint_close_threshold_m

    bank: List[Tuple[str, FeatureFunction]] = []

    def add(name: str, fn: FeatureFunction) -> None:
        bank.append((name, fn))

    def _distance(synced):
        return np.hypot(
            synced["gaze_x"] - synced["robot_x"],
            synced["gaze_y"] - synced["robot_y"])

    # --- 001-030: gaze-vs-robot position/speed relationship (adapted
    # from the gaze bank's own 30 gaze-relative features, now using the
    # calibrated frame instead of the normalized display) ---
    add("mean_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("std_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("median_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_median(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("max_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_maximum(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("rms_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_root_mean_square(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("p90_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            percentile_value(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)), 90.0))
    add("min_gaze_ee_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_minimum(_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))

    def _final_gaze_ee_distance(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        distance = _distance(synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))
        return float(distance[-1]) if distance.size else np.nan
    add("final_gaze_ee_distance", _final_gaze_ee_distance)

    def _gaze_ee_distance_slope(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return linear_slope(synced["time_s"], _distance(synced))
    add("gaze_ee_distance_slope", _gaze_ee_distance_slope)

    add("mean_relative_x",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((lambda s: s["gaze_x"] - s["robot_x"])(
                synchronize_robot_onto_gaze(
                    Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("mean_relative_y",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((lambda s: s["gaze_y"] - s["robot_y"])(
                synchronize_robot_onto_gaze(
                    Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("std_relative_x",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std((lambda s: s["gaze_x"] - s["robot_x"])(
                synchronize_robot_onto_gaze(
                    Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))
    add("std_relative_y",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_std((lambda s: s["gaze_y"] - s["robot_y"])(
                synchronize_robot_onto_gaze(
                    Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))))

    def _relative_path_length(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return planar_path_length(
            synced["gaze_x"] - synced["robot_x"],
            synced["gaze_y"] - synced["robot_y"])
    add("relative_path_length", _relative_path_length)

    def _mean_relative_speed(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        relative_vx = np.gradient(
            synced["gaze_x"] - synced["robot_x"], synced["time_s"]) \
            if synced["time_s"].size >= 2 else np.array([])
        relative_vy = np.gradient(
            synced["gaze_y"] - synced["robot_y"], synced["time_s"]) \
            if synced["time_s"].size >= 2 else np.array([])
        relative_motion = calculate_robot_motion(
            relative_vx, relative_vy, synced["time_s"])
        return safe_mean(relative_motion["speed"])
    add("mean_relative_speed", _mean_relative_speed)

    def _std_relative_speed(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        relative_vx = np.gradient(
            synced["gaze_x"] - synced["robot_x"], synced["time_s"]) \
            if synced["time_s"].size >= 2 else np.array([])
        relative_vy = np.gradient(
            synced["gaze_y"] - synced["robot_y"], synced["time_s"]) \
            if synced["time_s"].size >= 2 else np.array([])
        relative_motion = calculate_robot_motion(
            relative_vx, relative_vy, synced["time_s"])
        return safe_std(relative_motion["speed"])
    add("std_relative_speed", _std_relative_speed)

    add("fraction_within_0p01",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))
                <= 0.01).astype(float)))
    add("fraction_within_0p02",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))
                <= 0.02).astype(float)))
    add("fraction_within_0p05",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((_distance(synchronize_robot_onto_gaze(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy))
                <= 0.05).astype(float)))

    def _gaze_ee_x_correlation(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(synced["gaze_x"], synced["robot_x"])
    add("gaze_ee_x_correlation", _gaze_ee_x_correlation)

    def _gaze_ee_y_correlation(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(synced["gaze_y"], synced["robot_y"])
    add("gaze_ee_y_correlation", _gaze_ee_y_correlation)

    def _gaze_ee_speed_correlation(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return pearson_correlation(gaze_motion["speed"], synced["speed"])
    add("gaze_ee_speed_correlation", _gaze_ee_speed_correlation)

    add("mean_velocity_cosine_similarity",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(joint_gaze_ee_velocity_cosine(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)))
    add("aligned_velocity_fraction",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean((joint_gaze_ee_velocity_cosine(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
                > 0).astype(float)))

    def _gaze_to_ee_speed_ratio(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return safe_ratio(
            safe_mean(gaze_motion["speed"]), safe_mean(synced["speed"]))
    add("gaze_to_ee_speed_ratio", _gaze_to_ee_speed_ratio)

    def _max_x_cross_correlation(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        value, _ = best_lagged_correlation(
            synced["gaze_x"], synced["robot_x"], synced["time_s"], 0.25)
        return value
    add("max_x_cross_correlation", _max_x_cross_correlation)

    def _x_lead_at_max_correlation_s(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        _, lag_s = best_lagged_correlation(
            synced["gaze_x"], synced["robot_x"], synced["time_s"], 0.25)
        return lag_s
    add("x_lead_at_max_correlation_s", _x_lead_at_max_correlation_s)

    def _max_y_cross_correlation(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        value, _ = best_lagged_correlation(
            synced["gaze_y"], synced["robot_y"], synced["time_s"], 0.25)
        return value
    add("max_y_cross_correlation", _max_y_cross_correlation)

    def _y_lead_at_max_correlation_s(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        _, lag_s = best_lagged_correlation(
            synced["gaze_y"], synced["robot_y"], synced["time_s"], 0.25)
        return lag_s
    add("y_lead_at_max_correlation_s", _y_lead_at_max_correlation_s)

    def _mean_gaze_lead_along_ee_motion(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        moving = synced["speed"] > 1e-9
        along_path_lead = np.full(synced["speed"].shape, np.nan)
        along_path_lead[moving] = (
            (synced["gaze_x"][moving] - synced["robot_x"][moving]) *
            synced["velocity_x"][moving] +
            (synced["gaze_y"][moving] - synced["robot_y"][moving]) *
            synced["velocity_y"][moving]
        ) / synced["speed"][moving]
        return safe_mean(along_path_lead)
    add("mean_gaze_lead_along_ee_motion", _mean_gaze_lead_along_ee_motion)

    # --- 031-036: gaze vs. robot path curvature ---
    def _correlation_gaze_ee_distance_curvature(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(
            _distance(synced), np.abs(synced["curvature"]))
    add("correlation_gaze_ee_distance_curvature",
        _correlation_gaze_ee_distance_curvature)

    def _correlation_gaze_speed_absolute_curvature(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return pearson_correlation(
            gaze_motion["speed"], np.abs(synced["curvature"]))
    add("correlation_gaze_speed_absolute_curvature",
        _correlation_gaze_speed_absolute_curvature)

    def _mean_curvature_when_gaze_close(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return curvature_conditioned_on_distance(
            _distance(synced), synced["curvature"], True)
    add("mean_curvature_when_gaze_close", _mean_curvature_when_gaze_close)

    def _mean_curvature_when_gaze_far(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return curvature_conditioned_on_distance(
            _distance(synced), synced["curvature"], False)
    add("mean_curvature_when_gaze_far", _mean_curvature_when_gaze_far)

    def _fraction_gaze_close_during_high_curvature(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return fraction_close_during_high_reference(
            _distance(synced), np.abs(synced["curvature"]), close_m)
    add("fraction_gaze_close_during_high_curvature",
        _fraction_gaze_close_during_high_curvature)

    def _correlation_gaze_offset_angle_curvature(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        offset_angle_deg = signed_angle_series(
            synced["velocity_x"], synced["velocity_y"],
            synced["gaze_x"] - synced["robot_x"],
            synced["gaze_y"] - synced["robot_y"])
        return pearson_correlation(synced["curvature"], offset_angle_deg)
    add("correlation_gaze_offset_angle_curvature",
        _correlation_gaze_offset_angle_curvature)

    # --- 037-042: gaze vs. robot force. DAMPING-CONFOUNDED. ---
    def _angle_gaze_offset_to_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        MINIMUM_FORCE_FOR_DIRECTION_N = 0.5
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        offset_x = synced["gaze_x"] - synced["robot_x"]
        offset_y = synced["gaze_y"] - synced["robot_y"]
        angle_deg = signed_angle_series(
            synced["force_x"], synced["force_y"], offset_x, offset_y)
        angle_deg = angle_deg.copy()
        angle_deg[synced["force_magnitude"] < MINIMUM_FORCE_FOR_DIRECTION_N] \
            = np.nan
        return safe_mean(angle_deg)
    add("angle_gaze_offset_to_force", _angle_gaze_offset_to_force)

    def _correlation_gaze_ee_distance_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(_distance(synced), synced["force_magnitude"])
    add("correlation_gaze_ee_distance_force",
        _correlation_gaze_ee_distance_force)

    def _correlation_gaze_speed_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return pearson_correlation(
            gaze_motion["speed"], synced["force_magnitude"])
    add("correlation_gaze_speed_force", _correlation_gaze_speed_force)

    def _mean_gaze_ee_distance_high_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return value_conditioned_on_reference(
            _distance(synced), synced["force_magnitude"], False)
    add("mean_gaze_ee_distance_high_force", _mean_gaze_ee_distance_high_force)

    def _mean_gaze_ee_distance_low_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return value_conditioned_on_reference(
            _distance(synced), synced["force_magnitude"], True)
    add("mean_gaze_ee_distance_low_force", _mean_gaze_ee_distance_low_force)

    def _fraction_gaze_close_during_high_force(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return fraction_close_during_high_reference(
            _distance(synced), synced["force_magnitude"], close_m)
    add("fraction_gaze_close_during_high_force",
        _fraction_gaze_close_during_high_force)

    # --- 043-044: gaze vs. robot angular speed (wrist rotation) ---
    def _correlation_gaze_speed_angular_speed(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return pearson_correlation(
            gaze_motion["speed"], synced["angular_speed"])
    add("correlation_gaze_speed_angular_speed",
        _correlation_gaze_speed_angular_speed)

    def _correlation_gaze_ee_distance_angular_speed(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(_distance(synced), synced["angular_speed"])
    add("correlation_gaze_ee_distance_angular_speed",
        _correlation_gaze_ee_distance_angular_speed)

    # --- 045-046: gaze vs. robot command-tracking error. DAMPING-
    # CONFOUNDED. ---
    def _correlation_gaze_speed_tracking_error(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        gaze_motion = calculate_robot_motion(
            synced["gaze_x"], synced["gaze_y"], synced["time_s"])
        return pearson_correlation(
            gaze_motion["speed"], synced["tracking_error"])
    add("correlation_gaze_speed_tracking_error",
        _correlation_gaze_speed_tracking_error)

    def _correlation_gaze_ee_distance_tracking_error(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return pearson_correlation(_distance(synced), synced["tracking_error"])
    add("correlation_gaze_ee_distance_tracking_error",
        _correlation_gaze_ee_distance_tracking_error)

    # --- 047-051: gaze lookahead relative to robot motion direction --
    # path-free ports of extract_pilot_features_fixed_adm_2d.m's
    # angle_gaze_v_deg / d_lookahead_over_v_s / d_lookahead_times_
    # curvature / fraction_gaze_ahead, projecting (gaze - hand) onto
    # the velocity direction instead of arc-length along a reference
    # path (no shape .obj is available here to project onto). ---
    def _angle_gaze_offset_to_velocity(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        MINIMUM_SPEED_M_S = 0.005
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        offset_x = synced["gaze_x"] - synced["robot_x"]
        offset_y = synced["gaze_y"] - synced["robot_y"]
        angle_deg = signed_angle_series(
            synced["velocity_x"], synced["velocity_y"], offset_x, offset_y)
        angle_deg = angle_deg.copy()
        angle_deg[synced["speed"] < MINIMUM_SPEED_M_S] = np.nan
        return safe_mean(angle_deg)
    add("angle_gaze_offset_to_velocity", _angle_gaze_offset_to_velocity)

    add("mean_lookahead_distance",
        lambda Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy:
            safe_mean(lookahead_distance_series(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)))

    def _fraction_gaze_ahead(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        lookahead = lookahead_distance_series(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        valid_lookahead = lookahead[np.isfinite(lookahead)]
        if valid_lookahead.size == 0:
            return np.nan
        return float(np.mean(valid_lookahead > 0))
    add("fraction_gaze_ahead", _fraction_gaze_ahead)

    def _lookahead_distance_over_speed(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        MINIMUM_SPEED_M_S = 0.005
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        lookahead = lookahead_distance_series(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        ratio = np.full(lookahead.shape, np.nan)
        moving = synced["speed"] >= MINIMUM_SPEED_M_S
        ratio[moving] = lookahead[moving] / synced["speed"][moving]
        return safe_mean(ratio)
    add("lookahead_distance_over_speed", _lookahead_distance_over_speed)

    def _lookahead_times_curvature(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy):
        synced = synchronize_robot_onto_gaze(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        lookahead = lookahead_distance_series(
            Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        return safe_mean(lookahead * synced["curvature"])
    add("lookahead_times_curvature", _lookahead_times_curvature)

    assert len(bank) == 51, f"expected 51 joint features, built {len(bank)}"
    return bank


# ======================================================================
# Convenience: every bank at once.
# ======================================================================

def build_all_feature_banks(
        config: FeatureConfig,
) -> Dict[str, List[Tuple[str, FeatureFunction]]]:
    """{'gaze': [...102...], 'robot': [...96...], 'joint': [...51...]}
    -- for a training script that wants every feature at once. Feature
    names are unique WITHIN each bank but not necessarily across banks
    (e.g. "mean_relative_x" exists in both the gaze and joint banks,
    computed from different frames/interfaces) -- prefix by bank name
    (e.g. f"gaze_{name}") if you flatten these into one combined table."""
    return {
        "gaze": build_gaze_feature_bank(config),
        "robot": build_robot_feature_bank(config),
        "joint": build_joint_feature_bank(config),
    }
