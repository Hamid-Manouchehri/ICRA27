"""build_modality_feature_datasets.py -- Windowed feature datasets for
the Fine/Gross classifier, built SEPARATELY per modality (robot-only,
gaze-only, joint) from every trial in KD_test/data/raw_data, using
classifier_feature_bank.py's three feature banks directly (no
recomputing 249 features and subsetting columns afterward -- each
modality computes only its own features per window).

Same trial discovery, calibrated gaze mapping, and window/label/
transition/edge-trim logic as KD_test/python/build_feature_dataset.py
(3 shapes x 3 dampings x subjects s2-s5; random_shape has no s3
recording -- a missing design cell, not an error). See that script's
own header for the full reasoning behind each setting below; only
what's new here is called out.

What's new/different from the KD_test single-modality version:
  - A window's validity (label clean, outside transition_margin_s,
    outside the edge-trimmed span, enough samples) is decided ONCE per
    window and required to hold for BOTH gaze and robot data, even for
    the robot-only modality (which doesn't itself need gaze) -- so all
    three modalities are evaluated on the EXACT SAME set of windows.
    Letting robot-only use a more permissive window set (since it
    doesn't technically need valid gaze) would give it strictly more
    training data than gaze/joint, and any accuracy difference later
    could then be "more data" instead of "the feature content" -- the
    same reasoning root as the earlier robot_only-vs-joint fairness
    discussion, applied one level up (window count, not feature count).
  - The train/test split is decided ONCE (by window identity) and
    applied identically to all three modality tables, so "window i" is
    train (or test) the same way in every modality's CSV -- necessary
    for compare_modality_classifiers.py's comparison to mean anything;
    without it, differences could partly reflect different windows
    being evaluated, not different feature content.
  - Output is six CSVs (train/test x robot/gaze/joint), not two.

Dependencies: numpy, pandas, scikit-learn
  pip install numpy pandas scikit-learn
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import classifier_feature_bank as cfb

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "KD_test" / "data"
RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw_data"
CALIBRATION_DIRECTORY = DATA_DIRECTORY / "calibration"

# -------------------- User Settings --------------------
valid_class_labels = ("Fine", "Gross")

write_paths = {
    "robot": {
        "train": SCRIPT_DIRECTORY / "features_robot_train.csv",
        "test": SCRIPT_DIRECTORY / "features_robot_test.csv",
    },
    "gaze": {
        "train": SCRIPT_DIRECTORY / "features_gaze_train.csv",
        "test": SCRIPT_DIRECTORY / "features_gaze_test.csv",
    },
    "joint": {
        "train": SCRIPT_DIRECTORY / "features_joint_train.csv",
        "test": SCRIPT_DIRECTORY / "features_joint_test.csv",
    },
}

# --- Windowing (matches KD_test/python/build_feature_dataset.py) ---
window_size_s = 0.5
window_step_s = 0.5
minimum_gaze_samples = 6
minimum_robot_samples = 10

# --- Ground-truth class label ---
transition_margin_s = 1.5
edge_trim_s = 1.5

# --- Calibrated Tobii-UCS -> CoppeliaSim gaze mapping ---
screen_motion_per_robot_motion = 1.0
tobii_to_coppelia_axis_sign = (1.0, 1.0)
require_gaze_on_display = True
require_both_eyes_valid = True

# --- Feature thresholds (see classifier_feature_bank.FeatureConfig) ---
workspace_half_extent_m = 0.20
low_speed_threshold_m_s = 0.05
high_speed_threshold_m_s = 0.30
close_thresholds_m = (0.01, 0.02, 0.05)

# --- Train/test split ---
train_samples_per_class = None
train_fraction_of_smallest_class = 0.5
random_seed = 42


# ======================================================================
# Stage 0: discover every (robot, gaze, calibration) trial by filename.
# Identical to KD_test/python/build_feature_dataset.py's discover_trials().
# ======================================================================

TRIAL_FILENAME_PATTERN = re.compile(
    r"^ur5e_fixed_adm_2d_labeled_(?P<shape>.+)_d(?P<damping>\d+)"
    r"_s(?P<subject>\d+)\.csv$")


def discover_trials() -> list:
    trials = []
    for robot_csv_file in sorted(
            RAW_DATA_DIRECTORY.glob("ur5e_fixed_adm_2d_labeled_*.csv")):
        match = TRIAL_FILENAME_PATTERN.match(robot_csv_file.name)
        if not match:
            print(f"Skipping {robot_csv_file.name}: unexpected filename.")
            continue
        shape = match["shape"]
        damping_n_s_m = int(match["damping"])
        subject = int(match["subject"])

        gaze_csv_file = RAW_DATA_DIRECTORY / (
            f"tobii_fixed_adm_2d_labeled_{shape}_d{damping_n_s_m}"
            f"_s{subject}.csv")
        calibration_json_file = CALIBRATION_DIRECTORY / (
            f"tobii_coppelia_reference_calibration_{shape}_s{subject}.json")

        if not gaze_csv_file.is_file():
            print(f"Skipping {robot_csv_file.name}: no matching gaze CSV "
                  f"{gaze_csv_file.name}.")
            continue
        if not calibration_json_file.is_file():
            print(f"Skipping {robot_csv_file.name}: no matching "
                  f"calibration {calibration_json_file.name}.")
            continue

        trials.append(dict(
            robot_csv_file=robot_csv_file, gaze_csv_file=gaze_csv_file,
            calibration_json_file=calibration_json_file, shape=shape,
            damping_n_s_m=damping_n_s_m, subject=subject))
    return trials


# ======================================================================
# Stage 1: load one trial's raw data and build calibrated streams.
# Identical to KD_test/python/build_feature_dataset.py's
# load_calibrated_streams(), but ALSO keeps the raw force columns
# (needed by the robot/joint banks, which the KD_test version dropped
# since it had already replaced mean_force_N with mean_ee_speed_m_s).
# ======================================================================

def get_numeric_column(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def parse_vector_cell(cell, size: int) -> np.ndarray:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.full(size, np.nan)
    text = str(cell).strip().strip("[]")
    if not text:
        return np.full(size, np.nan)
    parts = [p for p in text.replace(",", " ").split() if p]
    values = np.full(size, np.nan)
    for i, p in enumerate(parts[:size]):
        try:
            values[i] = float(p)
        except ValueError:
            pass
    return values


def parse_vector_column(series: pd.Series, size: int) -> np.ndarray:
    return np.vstack([parse_vector_cell(c, size) for c in series])


def load_calibrated_streams(
        robot_csv_file: Path, gaze_csv_file: Path,
        calibration_json_file: Path) -> dict:
    # keep_default_na=False: pandas' default NA-string list includes the
    # literal text "None", one of this project's real label values --
    # without this, every "None"-labeled row silently becomes a missing
    # value instead, and later compares unequal to the string "None",
    # letting it slip through as if it were a genuine class.
    robot_table = pd.read_csv(robot_csv_file, keep_default_na=False)
    gaze_table = pd.read_csv(gaze_csv_file, keep_default_na=False)
    calibration = json.loads(Path(calibration_json_file).read_text())
    tobii_reference_xy_mm = tuple(
        calibration["manual_tobii_reference_xy_mm"])
    coppelia_reference_xy_m = tuple(
        calibration["manual_coppelia_reference_xy_m"])

    robot_time_s = get_numeric_column(robot_table["time_s"])
    robot_label = robot_table["label"].astype(str).to_numpy()
    robot_x_m = get_numeric_column(robot_table["x_m"])
    robot_y_m = get_numeric_column(robot_table["y_m"])
    Px_all = get_numeric_column(robot_table["coppeliasim_x_m"])
    Py_all = get_numeric_column(robot_table["coppeliasim_y_m"])
    Vx_all = get_numeric_column(robot_table["tcp_vx_actual_m_s"])
    Vy_all = get_numeric_column(robot_table["tcp_vy_actual_m_s"])
    Fx_all = get_numeric_column(robot_table["Fx_filt_N"])
    Fy_all = get_numeric_column(robot_table["Fy_filt_N"])
    Vcx_all = get_numeric_column(robot_table["vx_cmd_m_s"])
    Vcy_all = get_numeric_column(robot_table["vy_cmd_m_s"])
    wx_all = get_numeric_column(robot_table["tcp_wx_actual_rad_s"])
    wy_all = get_numeric_column(robot_table["tcp_wy_actual_rad_s"])
    wz_all = get_numeric_column(robot_table["tcp_wz_actual_rad_s"])
    Wmag_all = np.sqrt(wx_all ** 2 + wy_all ** 2 + wz_all ** 2)

    robot_valid = np.isfinite(robot_time_s) & np.isfinite(Px_all) & \
        np.isfinite(Py_all) & np.isfinite(robot_x_m) & \
        np.isfinite(robot_y_m) & np.isfinite(Vx_all) & np.isfinite(Vy_all)
    robot_time_s = robot_time_s[robot_valid]
    robot_label = robot_label[robot_valid]
    robot_x_m = robot_x_m[robot_valid]
    robot_y_m = robot_y_m[robot_valid]
    Px_all = Px_all[robot_valid]
    Py_all = Py_all[robot_valid]
    Vx_all = Vx_all[robot_valid]
    Vy_all = Vy_all[robot_valid]
    Fx_all = Fx_all[robot_valid]
    Fy_all = Fy_all[robot_valid]
    Vcx_all = Vcx_all[robot_valid]
    Vcy_all = Vcy_all[robot_valid]
    Wmag_all = Wmag_all[robot_valid]

    left_ucs_mm = parse_vector_column(
        gaze_table["left_gaze_point_in_user_coordinate_system"], 3)
    right_ucs_mm = parse_vector_column(
        gaze_table["right_gaze_point_in_user_coordinate_system"], 3)
    left_display = parse_vector_column(
        gaze_table["left_gaze_point_on_display_area"], 2)
    right_display = parse_vector_column(
        gaze_table["right_gaze_point_on_display_area"], 2)
    left_point_valid = get_numeric_column(
        gaze_table["left_gaze_point_validity"]) == 1
    right_point_valid = get_numeric_column(
        gaze_table["right_gaze_point_validity"]) == 1

    left_point_valid &= np.all(np.isfinite(left_ucs_mm), axis=1)
    right_point_valid &= np.all(np.isfinite(right_ucs_mm), axis=1)
    if require_gaze_on_display:
        left_on_display = np.all(
            np.isfinite(left_display) & (left_display >= 0) &
            (left_display <= 1), axis=1)
        right_on_display = np.all(
            np.isfinite(right_display) & (right_display >= 0) &
            (right_display <= 1), axis=1)
        left_point_valid &= left_on_display
        right_point_valid &= right_on_display

    gaze_system_time_us = get_numeric_column(gaze_table["system_time_stamp"])
    first_timestamp_idx = np.flatnonzero(np.isfinite(gaze_system_time_us))
    if first_timestamp_idx.size == 0:
        raise ValueError("The Tobii CSV contains no finite system timestamps.")
    gaze_start_time_us = gaze_system_time_us[first_timestamp_idx[0]]
    gaze_time_s_all = (gaze_system_time_us - gaze_start_time_us) * 1e-6

    if require_both_eyes_valid:
        gaze_valid = left_point_valid & right_point_valid
        gaze_ucs_xy_mm = 0.5 * (left_ucs_mm[:, :2] + right_ucs_mm[:, :2])
    else:
        eye_count = \
            left_point_valid.astype(float) + right_point_valid.astype(float)
        gaze_valid = eye_count > 0
        gaze_sum = np.zeros((len(gaze_table), 2))
        gaze_sum[left_point_valid] += left_ucs_mm[left_point_valid, :2]
        gaze_sum[right_point_valid] += right_ucs_mm[right_point_valid, :2]
        gaze_ucs_xy_mm = np.divide(
            gaze_sum, eye_count[:, None],
            out=np.full((len(gaze_table), 2), np.nan),
            where=(eye_count[:, None] > 0))

    gaze_valid &= np.isfinite(gaze_time_s_all)
    Gt_all = gaze_time_s_all[gaze_valid]
    Gxy_all = cfb.map_gaze_ucs_to_coppelia(
        gaze_ucs_xy_mm[gaze_valid], tobii_reference_xy_mm,
        coppelia_reference_xy_m, screen_motion_per_robot_motion,
        tobii_to_coppelia_axis_sign)
    Gx_all, Gy_all = Gxy_all[:, 0], Gxy_all[:, 1]

    return dict(
        robot_time_s=robot_time_s, robot_label=robot_label,
        robot_x_m=robot_x_m, robot_y_m=robot_y_m,
        Px_all=Px_all, Py_all=Py_all, Vx_all=Vx_all, Vy_all=Vy_all,
        Fx_all=Fx_all, Fy_all=Fy_all, Wmag_all=Wmag_all,
        Vcx_all=Vcx_all, Vcy_all=Vcy_all,
        Gt_all=Gt_all, Gx_all=Gx_all, Gy_all=Gy_all,
        coppelia_reference_xy_m=coppelia_reference_xy_m,
    )


# ======================================================================
# Stage 2: per-robot-sample "clean" mask. Identical to KD_test/python/
# build_feature_dataset.py's compute_robot_sample_is_clean().
# ======================================================================

def compute_robot_sample_is_clean(
        robot_time_s: np.ndarray, robot_label: np.ndarray) -> np.ndarray:
    is_a_class = np.isin(robot_label, valid_class_labels)

    label_changed = np.empty(robot_label.shape, dtype=bool)
    label_changed[0] = True
    label_changed[1:] = robot_label[1:] != robot_label[:-1]
    boundary_times_s = robot_time_s[label_changed]

    is_near_transition = np.zeros(robot_time_s.shape, dtype=bool)
    for boundary_time_s in boundary_times_s:
        elapsed_since_boundary_s = robot_time_s - boundary_time_s
        is_near_transition |= (elapsed_since_boundary_s >= 0) & \
            (elapsed_since_boundary_s <= transition_margin_s)

    return is_a_class & ~is_near_transition


# ======================================================================
# Stage 3: windows -> features, for one trial, computed for all three
# modalities at once (same kept-window decision shared across them --
# see module docstring for why).
# ======================================================================

def build_windows(streams: dict, trial: dict) -> dict:
    robot_time_s = streams["robot_time_s"]
    robot_label = streams["robot_label"]
    Px_all, Py_all = streams["Px_all"], streams["Py_all"]
    Vx_all, Vy_all = streams["Vx_all"], streams["Vy_all"]
    Fx_all, Fy_all = streams["Fx_all"], streams["Fy_all"]
    Wmag_all = streams["Wmag_all"]
    Vcx_all, Vcy_all = streams["Vcx_all"], streams["Vcy_all"]
    robot_x_m, robot_y_m = streams["robot_x_m"], streams["robot_y_m"]
    Gt_all, Gx_all, Gy_all = \
        streams["Gt_all"], streams["Gx_all"], streams["Gy_all"]

    robot_sample_is_clean = compute_robot_sample_is_clean(
        robot_time_s, robot_label)

    analysis_start_s = max(Gt_all.min(), robot_time_s.min()) + edge_trim_s
    analysis_end_s = min(Gt_all.max(), robot_time_s.max()) - edge_trim_s
    if analysis_end_s - analysis_start_s < window_size_s:
        print(f"  SKIPPING trial (too short after edge_trim_s): "
              f"{trial['shape']} d{trial['damping_n_s_m']} "
              f"s{trial['subject']}")
        return {"robot": [], "gaze": [], "joint": []}

    window_starts = np.arange(
        analysis_start_s, analysis_end_s - window_size_s, window_step_s)

    config = cfb.FeatureConfig(
        workspace_center_xy_m=streams["coppelia_reference_xy_m"],
        workspace_half_extent_m=workspace_half_extent_m,
        low_speed_threshold_m_s=low_speed_threshold_m_s,
        high_speed_threshold_m_s=high_speed_threshold_m_s,
        close_thresholds_m=close_thresholds_m,
    )
    robot_bank = cfb.build_robot_feature_bank(config)
    gaze_bank = cfb.build_gaze_feature_bank(config)
    joint_bank = cfb.build_joint_feature_bank(config)

    rows = {"robot": [], "gaze": [], "joint": []}
    excluded_too_few_samples = 0
    excluded_label = 0
    for window_idx, window_start_s in enumerate(window_starts):
        window_end_s = window_start_s + window_size_s
        is_last_window = window_idx == len(window_starts) - 1

        if is_last_window:
            robot_in_window = (robot_time_s >= window_start_s) & \
                (robot_time_s <= window_end_s)
            gaze_in_window = (Gt_all >= window_start_s) & \
                (Gt_all <= window_end_s)
        else:
            robot_in_window = (robot_time_s >= window_start_s) & \
                (robot_time_s < window_end_s)
            gaze_in_window = (Gt_all >= window_start_s) & \
                (Gt_all < window_end_s)

        robot_count = int(np.count_nonzero(robot_in_window))
        gaze_count = int(np.count_nonzero(gaze_in_window))
        # Both gaze and robot sample-count/validity gates apply to
        # EVERY modality, including robot-only -- see module docstring.
        if robot_count < minimum_robot_samples or \
                gaze_count < minimum_gaze_samples:
            excluded_too_few_samples += 1
            continue
        if not np.all(robot_sample_is_clean[robot_in_window]):
            excluded_label += 1
            continue
        unique_labels = np.unique(robot_label[robot_in_window])
        if unique_labels.size != 1:
            excluded_label += 1
            continue
        window_label = str(unique_labels[0])

        Gx = Gx_all[gaze_in_window]
        Gy = Gy_all[gaze_in_window]
        Gt = Gt_all[gaze_in_window]
        Px = Px_all[robot_in_window]
        Py = Py_all[robot_in_window]
        Pt = robot_time_s[robot_in_window]
        Vx = Vx_all[robot_in_window]
        Vy = Vy_all[robot_in_window]
        Fx = Fx_all[robot_in_window]
        Fy = Fy_all[robot_in_window]
        Wmag = Wmag_all[robot_in_window]
        Vcx = Vcx_all[robot_in_window]
        Vcy = Vcy_all[robot_in_window]

        metadata = {
            "subject": trial["subject"],
            "shape": trial["shape"],
            "damping_n_s_m": trial["damping_n_s_m"],
            "window_index": window_idx,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "window_center_s": 0.5 * (window_start_s + window_end_s),
            "gaze_sample_count": gaze_count,
            "robot_sample_count": robot_count,
            "trial_class_label": window_label,
        }

        robot_row = dict(metadata)
        for name, feature_fn in robot_bank:
            robot_row[name] = feature_fn(
                Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        robot_row["DSJ"] = cfb.calculate_planar_dsj(
            robot_x_m[robot_in_window], robot_y_m[robot_in_window], Pt)
        robot_row["mean_ee_speed_m_s"] = cfb.calculate_mean_planar_speed(
            Px, Py, Pt)
        rows["robot"].append(robot_row)

        gaze_row = dict(metadata)
        for name, feature_fn in gaze_bank:
            gaze_row[name] = feature_fn(Gx, Gy, Gt, Px, Py, Pt)
        rows["gaze"].append(gaze_row)

        joint_row = dict(metadata)
        for name, feature_fn in joint_bank:
            joint_row[name] = feature_fn(
                Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        rows["joint"].append(joint_row)

    print(
        f"  {trial['shape']:<8} d{trial['damping_n_s_m']:<3} "
        f"s{trial['subject']}: {len(window_starts)} candidates, "
        f"{len(rows['robot'])} kept, {excluded_too_few_samples} "
        f"too-few-samples, {excluded_label} label/transition")

    return {name: pd.DataFrame(r) for name, r in rows.items()}


# ======================================================================
# Stage 4: class-balanced train / uneven-by-design test split, decided
# ONCE by window identity and applied identically to all three modality
# tables -- see module docstring for why.
# ======================================================================

def split_train_test_balanced(feature_tables: dict) -> dict:
    key_columns = ["subject", "shape", "damping_n_s_m", "window_index"]
    key_table = feature_tables["robot"][key_columns + ["trial_class_label"]]

    class_counts = key_table["trial_class_label"].value_counts()
    smallest_class_count = int(class_counts.min())

    if train_samples_per_class is None:
        per_class_train_count = int(np.floor(
            smallest_class_count * train_fraction_of_smallest_class))
    else:
        per_class_train_count = train_samples_per_class

    if per_class_train_count <= 0:
        raise ValueError(
            f"train_samples_per_class resolved to {per_class_train_count} "
            f"(smallest class '{class_counts.idxmin()}' has only "
            f"{smallest_class_count} windows) -- raise "
            f"train_fraction_of_smallest_class, or set "
            f"train_samples_per_class explicitly.")
    if per_class_train_count >= smallest_class_count:
        print(
            f"WARNING: train_samples_per_class={per_class_train_count} "
            f">= smallest class's {smallest_class_count} windows -- that "
            f"class will have few or zero windows left for test.")

    train_keys, test_keys = [], []
    for _, class_rows in key_table.groupby("trial_class_label"):
        if len(class_rows) <= per_class_train_count:
            class_train, class_test = class_rows, class_rows.iloc[0:0]
        else:
            class_train, class_test = train_test_split(
                class_rows, train_size=per_class_train_count,
                random_state=random_seed)
        train_keys.append(class_train[key_columns])
        test_keys.append(class_test[key_columns])
    train_keys = pd.concat(train_keys, ignore_index=True)
    test_keys = pd.concat(test_keys, ignore_index=True)

    print(f"\nBalanced training draw: {per_class_train_count} windows "
          f"per class (from a smallest class of {smallest_class_count}).")

    split_tables = {}
    for modality, table in feature_tables.items():
        train_table = table.merge(
            train_keys, on=key_columns, how="inner"
        ).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        test_table = table.merge(
            test_keys, on=key_columns, how="inner"
        ).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        split_tables[modality] = (train_table, test_table)
    return split_tables


def main() -> None:
    trials = discover_trials()
    print(f"Discovered {len(trials)} trials.\n")

    trial_tables = {"robot": [], "gaze": [], "joint": []}
    for trial in trials:
        streams = load_calibrated_streams(
            trial["robot_csv_file"], trial["gaze_csv_file"],
            trial["calibration_json_file"])
        windows = build_windows(streams, trial)
        for modality in trial_tables:
            trial_tables[modality].append(windows[modality])

    feature_tables = {
        modality: pd.concat(tables, ignore_index=True)
        for modality, tables in trial_tables.items()
    }
    pooled_window_count = len(feature_tables["robot"])
    print(f"\nPooled: {pooled_window_count} windows across "
          f"{len(trials)} trials.")
    print(feature_tables["robot"]["trial_class_label"]
          .value_counts().to_string())

    split_tables = split_train_test_balanced(feature_tables)

    for modality, (train_table, test_table) in split_tables.items():
        train_table.to_csv(write_paths[modality]["train"], index=False)
        test_table.to_csv(write_paths[modality]["test"], index=False)
        print(f"\n{modality}: wrote {len(train_table)} train / "
              f"{len(test_table)} test windows "
              f"({len(train_table.columns)} columns) to "
              f"{write_paths[modality]['train'].name} / "
              f"{write_paths[modality]['test'].name}")
        for split_name, table in [("train", train_table), ("test", test_table)]:
            counts = table["trial_class_label"].value_counts()
            counts_text = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"  {split_name}: {counts_text}")


if __name__ == "__main__":
    main()
