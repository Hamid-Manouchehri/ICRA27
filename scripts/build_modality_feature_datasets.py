"""build_modality_feature_datasets.py -- Build windowed features for
the Fine/Gross classifier, per modality (robot/gaze/joint, and "all" --
all 3 banks combined into one row, column names prefixed by bank),
from raw robot + gaze trial CSVs.

Inputs:
- chen_manual (the dataset in use): robot + tobii trial CSVs
  (data/processed_data/) and calibration JSON (data/calibrations/),
  one PER TRIAL for every combination of chen_manual_subjects/
  _bandwidths/_shapes/_directions -- an explicit trial matrix, not a
  glob. Per-trial filenames are built from the explicit
  chen_manual_{robot_csv,gaze_csv,calibration_json}_pattern settings
  (a pattern, not one setting per file, since the trial count is
  data-dependent -- see chen_manual_trial_files()).
- fixed_pilot / chen_otsu (legacy, unused, kept for reference): same
  file kinds, discovered instead via DATASET_REGISTRY[dataset_name]'s
  glob + regex + filename templates.

Outputs (exact filenames in User Settings, write_*_csv):
- 8 feature CSVs: features_{robot,gaze,joint,all}_{train,test}.csv
- data/processed_data/feature_cache/<output_dataset_name>_
  <fingerprint>/ (per-trial cache; inherently dynamic, self-
  invalidating on any feature-affecting setting change)
- data/json/build_modality_feature_datasets_<timestamp>.json (run
  manifest; inherently dynamic, one per run)

Depends on:
- classifier_feature_bank.py
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import classifier_feature_bank as cfb

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
SCRIPT_NAME = Path(__file__).stem
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
RAW_DATA_DIRECTORY = DATA_DIRECTORY / "raw_data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
CALIBRATION_DIRECTORY = DATA_DIRECTORY / "calibrations"
JSON_DIRECTORY = DATA_DIRECTORY / "json"

# -------------------- User Settings --------------------
# TODO which labeled dataset to build from -- "chen_manual" (explicit
# trial matrix below, this is the one in use) or a key of
# DATASET_REGISTRY further down ("chen_otsu"/"fixed_pilot", both
# legacy/unused).
dataset_name = "chen_manual"

# TODO the exact chen_manual trial matrix -- edit these four tuples
# directly to add/remove a subject, shape, bandwidth, or direction.
# Every combination is checked explicitly at startup (see
# discover_chen_manual_trials()) and any missing file is reported by
# name before feature computation starts, instead of being silently
# skipped mid-run one line at a time -- that silent-skip is exactly
# what dropped subject 18 before its calibration files existed (see
# decision_log.md, 2026-09-13).
chen_manual_subjects = (10, 11, 12, 13, 14, 15, 16, 17, 18)
chen_manual_bandwidths = ("narrow", "wide")
chen_manual_shapes = ("ellipse", "pear", "fine_gross")
chen_manual_directions = ("clockwise", "counter_clockwise")

# TODO explicit output filenames -- typed out, not built from a tag, so
# the exact file this script writes is always visible here. Change
# these (not dataset_name above) when you change window_size_s/
# window_step_s below, so the new run doesn't overwrite today's CSVs.
write_robot_train_csv = "features_robot_train_W_2s.csv"
write_robot_test_csv = "features_robot_test_W_2s.csv"
write_gaze_train_csv = "features_gaze_train_W_2s.csv"
write_gaze_test_csv = "features_gaze_test_W_2s.csv"
write_joint_train_csv = "features_joint_train_W_2s.csv"
write_joint_test_csv = "features_joint_test_W_2s.csv"
write_all_train_csv = "features_all_train_W_2s.csv"
write_all_test_csv = "features_all_test_W_2s.csv"

# TODO chen_manual INPUT filename PATTERNS -- one robot/gaze/
# calibration file PER TRIAL (chen_manual_subjects x _bandwidths x
# _shapes x _directions below can give ~100+ trials x 3 files, so
# these stay patterns rather than one explicit setting per file; see
# chen_manual_trial_files()). {stem} is
# "{bandwidth}_{shape}_shape_{direction}_s{subject}".
chen_manual_robot_csv_pattern = "ur5e_chen_var_adm_2d_live_labeled_{stem}.csv"
chen_manual_gaze_csv_pattern = "tobii_chen_var_adm_2d_live_labeled_{stem}.csv"
chen_manual_calibration_json_pattern = (
    "tobii_coppelia_reference_calibration_{stem}.json")

# TODO display/bookkeeping label only -- feature_cache/ subdirectory
# prefix and a field in the run manifest JSON. NOT used to build any of
# the explicit read/write filenames above or below.
output_dataset_name = "W_2s"

# TODO which modality tables to build/write: "robot"|"gaze"|"joint"|
# "all" (all 3 banks combined into one row, column names prefixed by
# bank -- "gaze_"/"robot_"/"joint_"; a few names repeat across banks
# with different bodies). CIRCULARITY on chen_otsu: robot/joint label
# is thresholded damping, a deterministic function of force -- gaze-
# only is the honest test there.
modalities_to_build = ("robot", "gaze", "joint", "all")

# TODO robot CSV stems (filename, no extension) to skip regardless of
# dataset -- trials flagged as unreliable during the calibration audit
# (Tier 3: statistically flagged, judged too risky to label/train on).
excluded_trial_stems = frozenset({
    "ur5e_chen_var_adm_2d_otsu_labeled_narrow_fine_gross_shape_"
    "counter_clockwise_s14",
    "ur5e_chen_var_adm_2d_otsu_labeled_wide_fine_gross_shape_"
    "counter_clockwise_s16",
})

# TODO cache each trial's windows under processed_data/feature_cache/
# so an interrupted run resumes instead of restarting from zero. The
# cache key includes every feature-affecting setting, so it self-
# invalidates; delete the directory to force a full rebuild.
use_trial_cache = False

valid_class_labels = ("Fine", "Gross")

# Built from the explicit write_*_csv settings above -- one dict lookup
# site (main(), save_run_manifest()) instead of re-typing the four-way
# branch everywhere a modality's output path is needed. The original
# pilot run predates this and left unsuffixed features_*_{train,test}.
# csv files; rename those to *_fixed_pilot.csv to reuse them here.
write_paths = {
    "robot": {"train": PROCESSED_DATA_DIRECTORY / write_robot_train_csv,
              "test": PROCESSED_DATA_DIRECTORY / write_robot_test_csv},
    "gaze": {"train": PROCESSED_DATA_DIRECTORY / write_gaze_train_csv,
             "test": PROCESSED_DATA_DIRECTORY / write_gaze_test_csv},
    "joint": {"train": PROCESSED_DATA_DIRECTORY / write_joint_train_csv,
              "test": PROCESSED_DATA_DIRECTORY / write_joint_test_csv},
    "all": {"train": PROCESSED_DATA_DIRECTORY / write_all_train_csv,
            "test": PROCESSED_DATA_DIRECTORY / write_all_test_csv},
}

# --- Windowing ---
window_size_s = 2.0
window_step_s = 2.0  # overlap_fraction = 1 - (window_step_s / window_size_s)
minimum_gaze_samples = 6
minimum_robot_samples = 10

# --- Ground-truth class label ---
# TODO drop windows within this margin [s] of a label change. 1.5 suited
# the pilot set's long scripted blocks; Otsu labels flip far more often
# (median segment 3.5 s, 36% shorter than 3 s), where 1.5 kept only ~38%
# of samples on average and 5% on the worst trial.
transition_margin_s = 0.5
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
# Same logic as KD_test/python/build_feature_dataset.py's
# discover_trials(), generalized over DATASET_REGISTRY so the pilot and
# Chen datasets differ only by their filename patterns and condition
# fields -- own filename pattern + condition fields per dataset.
# ======================================================================

DATASET_REGISTRY = {
    "fixed_pilot": dict(
        robot_directory=RAW_DATA_DIRECTORY,
        robot_glob="ur5e_fixed_adm_2d_labeled_*.csv",
        robot_pattern=re.compile(
            r"^ur5e_fixed_adm_2d_labeled_(?P<shape>.+)"
            r"_d(?P<damping_n_s_m>\d+)_s(?P<subject>\d+)\.csv$"),
        gaze_name=("tobii_fixed_adm_2d_labeled_{shape}_d{damping_n_s_m}"
                   "_s{subject}.csv"),
        calibration_name=("tobii_coppelia_reference_calibration_{shape}"
                          "_s{subject}.json"),
        condition_fields=("shape", "damping_n_s_m"),
    ),
    "chen_otsu": dict(
        robot_directory=PROCESSED_DATA_DIRECTORY,
        robot_glob="ur5e_chen_var_adm_2d_otsu_labeled_*.csv",
        robot_pattern=re.compile(
            r"^ur5e_chen_var_adm_2d_otsu_labeled_"
            r"(?P<bandwidth>wide|narrow)_(?P<shape>.+)_shape_"
            r"(?P<direction>counter_clockwise|clockwise)"
            r"_s(?P<subject>\d+)\.csv$"),
        gaze_name=("tobii_chen_var_adm_2d_otsu_labeled_{bandwidth}_{shape}"
                   "_shape_{direction}_s{subject}.csv"),
        calibration_name=("tobii_coppelia_reference_calibration_{bandwidth}"
                          "_{shape}_shape_{direction}_s{subject}.json"),
        condition_fields=("bandwidth", "shape", "direction"),
    ),
    # chen_manual used to be defined here too (identical shape to
    # chen_otsu above, only the "_live_labeled_" segment differed).
    # It's now built explicitly from the chen_manual_subjects/
    # _bandwidths/_shapes/_directions settings and
    # discover_chen_manual_trials() below instead -- no glob, regex, or
    # registry entry for it any more.
}

# The single source every downstream function reads dataset config
# from, regardless of which discovery mechanism produced the trials.
if dataset_name == "chen_manual":
    ACTIVE_DATASET = dict(
        robot_directory=PROCESSED_DATA_DIRECTORY,
        condition_fields=("bandwidth", "shape", "direction"),
    )
else:
    ACTIVE_DATASET = DATASET_REGISTRY[dataset_name]


def parse_condition_value(text: str):
    """Keep digit-only filename fields numeric (e.g. damping 100)."""
    return int(text) if text.isdigit() else text


def describe_trial(trial: dict) -> str:
    """One-line trial identity for progress printing."""
    condition_text = " ".join(
        str(value) for value in trial["condition"].values())
    return f"{condition_text} s{trial['subject']}"


def chen_manual_trial_files(
        subject: int, bandwidth: str, shape: str, direction: str) -> dict:
    """The three file paths for one chen_manual trial, built directly
    from an f-string -- no regex, no .format() template. If the live-
    labeling filename convention ever changes, this is the one place
    to edit.

    Inputs:
      - subject, bandwidth, shape, direction: one trial identity.
    Outputs:
      - dict with robot_csv_file, gaze_csv_file, calibration_json_file
        (Paths that may or may not exist -- existence is checked by
        the caller)."""
    stem = f"{bandwidth}_{shape}_shape_{direction}_s{subject}"
    return dict(
        robot_csv_file=PROCESSED_DATA_DIRECTORY /
            chen_manual_robot_csv_pattern.format(stem=stem),
        gaze_csv_file=PROCESSED_DATA_DIRECTORY /
            chen_manual_gaze_csv_pattern.format(stem=stem),
        calibration_json_file=CALIBRATION_DIRECTORY /
            chen_manual_calibration_json_pattern.format(stem=stem),
    )


def discover_chen_manual_trials() -> list:
    """Every (subject, bandwidth, shape, direction) combination from
    the chen_manual_* settings, checked explicitly against disk.

    Inputs:
      - chen_manual_subjects/_bandwidths/_shapes/_directions settings.
    Outputs:
      - list of trial dicts (robot_csv_file, gaze_csv_file,
        calibration_json_file, condition, subject) for combinations
        with all three files present. Prints every incomplete
        combination by name, then a total, before any feature
        computation starts -- so a missing file for a whole subject
        (as happened with s18's calibration JSON) cannot be missed in
        a long scrolling log."""
    combinations = [
        (subject, bandwidth, shape, direction)
        for subject in chen_manual_subjects
        for bandwidth in chen_manual_bandwidths
        for shape in chen_manual_shapes
        for direction in chen_manual_directions
    ]
    trials = []
    incomplete = []
    for subject, bandwidth, shape, direction in combinations:
        paths = chen_manual_trial_files(subject, bandwidth, shape, direction)
        if paths["robot_csv_file"].stem in excluded_trial_stems:
            continue
        missing_files = [name for name, path in paths.items()
                         if not path.is_file()]
        if missing_files:
            incomplete.append(
                (subject, bandwidth, shape, direction, missing_files))
            continue
        trials.append(dict(
            **paths, subject=subject,
            condition=dict(bandwidth=bandwidth, shape=shape,
                           direction=direction)))

    print(f"chen_manual trial matrix: {len(chen_manual_subjects)} subjects "
          f"x {len(chen_manual_bandwidths)} bandwidths x "
          f"{len(chen_manual_shapes)} shapes x "
          f"{len(chen_manual_directions)} directions = "
          f"{len(combinations)} combinations.")
    if incomplete:
        print(f"INCOMPLETE -- {len(incomplete)} combination(s) missing a "
              f"file:")
        for subject, bandwidth, shape, direction, missing_files in incomplete:
            print(f"  s{subject} {bandwidth} {shape} {direction}: "
                  f"missing {', '.join(missing_files)}")
    print(f"{len(trials)} complete trials found "
          f"({len(combinations) - len(trials)} incomplete/excluded).\n")
    return trials


def discover_trials_from_registry() -> list:
    """fixed_pilot / chen_otsu only (legacy, unused): glob every robot
    CSV matching the registry's pattern, then derive and check the
    matching gaze CSV and calibration JSON from filename templates."""
    robot_directory = ACTIVE_DATASET["robot_directory"]
    trials = []
    for robot_csv_file in sorted(
            robot_directory.glob(ACTIVE_DATASET["robot_glob"])):
        if robot_csv_file.stem in excluded_trial_stems:
            print(f"Skipping {robot_csv_file.name}: in excluded_trial_stems.")
            continue
        match = ACTIVE_DATASET["robot_pattern"].match(robot_csv_file.name)
        if not match:
            print(f"Skipping {robot_csv_file.name}: unexpected filename.")
            continue
        name_fields = {key: parse_condition_value(value)
                       for key, value in match.groupdict().items()}
        subject = int(name_fields.pop("subject"))
        condition = {field: name_fields[field]
                     for field in ACTIVE_DATASET["condition_fields"]}

        gaze_csv_file = robot_directory / ACTIVE_DATASET["gaze_name"].format(
            subject=subject, **name_fields)
        calibration_json_file = (
            CALIBRATION_DIRECTORY /
            ACTIVE_DATASET["calibration_name"].format(
                subject=subject, **name_fields))

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
            calibration_json_file=calibration_json_file,
            condition=condition, subject=subject))
    return trials


def discover_trials() -> list:
    """Dispatches to the explicit chen_manual matrix, or the legacy
    registry-based glob for fixed_pilot/chen_otsu."""
    if dataset_name == "chen_manual":
        return discover_chen_manual_trials()
    return discover_trials_from_registry()


# ======================================================================
# Stage 1: load one trial's raw data and build calibrated streams.
# Identical to KD_test/python/build_feature_dataset.py's
# load_calibrated_streams(), but ALSO keeps the raw force columns
# (needed by the robot/joint banks, which the KD_test version dropped
# since it had already replaced mean_force_N with mean_ee_speed_m_s).
# ======================================================================

def get_numeric_column(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def optional_numeric_column(
        table: pd.DataFrame, column_name: str,
        fill_value: float = 0.0) -> np.ndarray:
    """get_numeric_column, but fill_value if the column is absent."""
    if column_name not in table.columns:
        return np.full(len(table), fill_value, dtype=float)
    return get_numeric_column(table[column_name])


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
    # ur5e_chen_var_adm_2d.py does not log TCP angular velocity, so
    # these three are optional: absent, Wmag is zero and only the 3
    # robot-bank features built on it go flat (the gaze bank never
    # touches it).
    wx_all = optional_numeric_column(robot_table, "tcp_wx_actual_rad_s")
    wy_all = optional_numeric_column(robot_table, "tcp_wy_actual_rad_s")
    wz_all = optional_numeric_column(robot_table, "tcp_wz_actual_rad_s")
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
# modalities at once (same kept-window decision shared across them, so
# robot-only can't get more windows than gaze/joint).
# ======================================================================

def build_windows(streams: dict, trial: dict, modalities) -> dict:
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
              f"{describe_trial(trial)}")
        return {modality: [] for modality in modalities}

    window_starts = np.arange(
        analysis_start_s, analysis_end_s - window_size_s, window_step_s)

    config = cfb.FeatureConfig(
        workspace_center_xy_m=streams["coppelia_reference_xy_m"],
        workspace_half_extent_m=workspace_half_extent_m,
        low_speed_threshold_m_s=low_speed_threshold_m_s,
        high_speed_threshold_m_s=high_speed_threshold_m_s,
        close_thresholds_m=close_thresholds_m,
    )
    # Only the requested modalities' banks are built, but the window
    # validity gates below still require BOTH gaze and robot data, so a
    # gaze-only run evaluates the exact same windows a three-modality
    # run would. "all" needs every bank, so a bank is built once (never
    # twice) whenever its own modality OR "all" is requested.
    need_robot = "robot" in modalities or "all" in modalities
    need_gaze = "gaze" in modalities or "all" in modalities
    need_joint = "joint" in modalities or "all" in modalities
    robot_bank = cfb.build_robot_feature_bank(config) if need_robot else []
    gaze_bank = cfb.build_gaze_feature_bank(config) if need_gaze else []
    joint_bank = cfb.build_joint_feature_bank(config) if need_joint else []

    rows = {modality: [] for modality in modalities}
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
        # EVERY modality, including robot-only.
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
            **trial["condition"],
            "window_index": window_idx,
            "window_start_s": window_start_s,
            "window_end_s": window_end_s,
            "window_center_s": 0.5 * (window_start_s + window_end_s),
            "gaze_sample_count": gaze_count,
            "robot_sample_count": robot_count,
            "trial_class_label": window_label,
        }

        # Each bank's feature VALUES are computed once here regardless of
        # how many of {that modality, "all"} were requested, then reused
        # below -- "all" never recomputes what an individual modality
        # row already computed.
        robot_features = {}
        if need_robot:
            for name, feature_fn in robot_bank:
                robot_features[name] = feature_fn(
                    Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
            robot_features["DSJ"] = cfb.calculate_planar_dsj(
                robot_x_m[robot_in_window], robot_y_m[robot_in_window], Pt)
            robot_features["mean_ee_speed_m_s"] = \
                cfb.calculate_mean_planar_speed(Px, Py, Pt)

        gaze_features = {}
        if need_gaze:
            for name, feature_fn in gaze_bank:
                gaze_features[name] = feature_fn(Gx, Gy, Gt, Px, Py, Pt)

        joint_features = {}
        if need_joint:
            for name, feature_fn in joint_bank:
                joint_features[name] = feature_fn(
                    Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)

        if "robot" in modalities:
            rows["robot"].append({**metadata, **robot_features})
        if "gaze" in modalities:
            rows["gaze"].append({**metadata, **gaze_features})
        if "joint" in modalities:
            rows["joint"].append({**metadata, **joint_features})
        if "all" in modalities:
            all_row = dict(metadata)
            for name, value in gaze_features.items():
                all_row[f"gaze_{name}"] = value
            for name, value in robot_features.items():
                all_row[f"robot_{name}"] = value
            for name, value in joint_features.items():
                all_row[f"joint_{name}"] = value
            rows["all"].append(all_row)

    kept_count = len(rows[modalities[0]])
    print(
        f"  {describe_trial(trial):<38}: {len(window_starts)} candidates, "
        f"{kept_count} kept, {excluded_too_few_samples} "
        f"too-few-samples, {excluded_label} label/transition")

    return {name: pd.DataFrame(r) for name, r in rows.items()}


# ======================================================================
# Stage 4: class-balanced train / uneven-by-design test split, decided
# ONCE by window identity and applied identically to all three modality
# tables (so "window i" is train/test the same way in every CSV).
# ======================================================================

def split_train_test_balanced(feature_tables: dict) -> dict:
    condition_fields = list(ACTIVE_DATASET["condition_fields"])
    key_columns = ["subject"] + condition_fields + ["window_index"]
    # Any built modality works as the key table: all of them carry the
    # same windows in the same order, by construction above.
    reference_modality = modalities_to_build[0]
    key_table = feature_tables[reference_modality][
        key_columns + ["trial_class_label"]]

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


# ======================================================================
# Per-trial feature cache. Building all trials takes ~30 s each, so an
# interrupted run used to lose everything; each trial's window table is
# now written as it is computed and reloaded on the next run. The cache
# directory name carries a hash of every setting that changes feature
# VALUES, so editing e.g. transition_margin_s starts a fresh cache
# instead of silently mixing old and new windows.
# ======================================================================

def cache_fingerprint() -> str:
    # modalities_to_build is deliberately NOT part of the key: each
    # modality is cached in its own file and its feature VALUES do not
    # depend on which other banks were requested, so adding a modality
    # must not invalidate the ones already computed.
    settings_text = repr((
        dataset_name,
        window_size_s, window_step_s,
        minimum_gaze_samples, minimum_robot_samples,
        transition_margin_s, edge_trim_s, valid_class_labels,
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign,
        require_gaze_on_display, require_both_eyes_valid,
        workspace_half_extent_m, low_speed_threshold_m_s,
        high_speed_threshold_m_s, close_thresholds_m,
    ))
    return hashlib.sha1(settings_text.encode()).hexdigest()[:12]


def trial_cache_directory() -> Path:
    return (PROCESSED_DATA_DIRECTORY / "feature_cache" /
            f"{output_dataset_name}_{cache_fingerprint()}")


def cached_trial_path(
        cache_directory: Path, trial: dict, modality: str) -> Path:
    return cache_directory / f"{trial['robot_csv_file'].stem}_{modality}.csv"


def read_cached_trial(cache_directory: Path, trial: dict) -> dict:
    """Whichever requested modalities are already cached (may be empty),
    so only the missing ones have to be recomputed."""
    if not use_trial_cache:
        return {}
    cached = {}
    for modality in modalities_to_build:
        path = cached_trial_path(cache_directory, trial, modality)
        if path.is_file():
            cached[modality] = pd.read_csv(path, keep_default_na=False)
    return cached


def write_cached_trial(
        cache_directory: Path, trial: dict, windows: dict) -> None:
    if not use_trial_cache:
        return
    for modality, table in windows.items():
        table.to_csv(
            cached_trial_path(cache_directory, trial, modality), index=False)


def save_run_manifest(manifest: dict) -> None:
    JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = JSON_DIRECTORY / f"{SCRIPT_NAME}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Output: {path}")


def main() -> None:
    start_time_s = time.perf_counter()
    if dataset_name == "chen_manual":
        print(f"Input:  {PROCESSED_DATA_DIRECTORY}/"
              f"{chen_manual_robot_csv_pattern.format(stem='<stem>')} "
              f"(+ matching gaze CSV + calibration JSON per trial; "
              f"explicit trial matrix, see chen_manual_* settings)")
    else:
        print(f"Input:  {ACTIVE_DATASET['robot_directory']}/"
              f"{ACTIVE_DATASET['robot_glob']} "
              f"(+ matching gaze CSVs and calibration JSON)")
    for modality in modalities_to_build:
        print(f"Output: {write_paths[modality]['train']}")
        print(f"Output: {write_paths[modality]['test']}")
    print(f"\nDataset: {dataset_name}   output tag: {output_dataset_name}   "
          f"modalities: {', '.join(modalities_to_build)}\n")
    trials = discover_trials()
    print(f"Discovered {len(trials)} trials.\n")
    if not trials:
        raise SystemExit(
            f"No trials found for dataset_name='{dataset_name}' under "
            f"{ACTIVE_DATASET['robot_directory']}.")

    cache_directory = trial_cache_directory()
    if use_trial_cache:
        cache_directory.mkdir(parents=True, exist_ok=True)
        print(f"Trial cache: {cache_directory}\n")

    trial_tables = {modality: [] for modality in modalities_to_build}
    for trial_idx, trial in enumerate(trials, start=1):
        cached = read_cached_trial(cache_directory, trial)
        missing = [m for m in modalities_to_build if m not in cached]

        if missing:
            streams = load_calibrated_streams(
                trial["robot_csv_file"], trial["gaze_csv_file"],
                trial["calibration_json_file"])
            computed = build_windows(streams, trial, missing)
            write_cached_trial(cache_directory, trial, computed)
            cached.update(computed)
            print(f"  [{trial_idx}/{len(trials)}] "
                  f"{describe_trial(trial):<38}: computed "
                  f"{'+'.join(missing)}")
        else:
            print(f"  [{trial_idx}/{len(trials)}] "
                  f"{describe_trial(trial):<38}: fully cached "
                  f"({len(cached[modalities_to_build[0]])} windows)")

        for modality in trial_tables:
            trial_tables[modality].append(cached[modality])

    feature_tables = {
        modality: pd.concat(tables, ignore_index=True)
        for modality, tables in trial_tables.items()
    }
    reference_modality = modalities_to_build[0]
    pooled_window_count = len(feature_tables[reference_modality])
    print(f"\nPooled: {pooled_window_count} windows across "
          f"{len(trials)} trials.")
    print(feature_tables[reference_modality]["trial_class_label"]
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

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")

    save_run_manifest({
        "script": SCRIPT_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "output_dataset_name": output_dataset_name,
        "modalities_built": list(modalities_to_build),
        "trial_count": len(trials),
        "pooled_window_count": pooled_window_count,
        "window_size_s": window_size_s,
        "window_step_s": window_step_s,
        "minimum_gaze_samples": minimum_gaze_samples,
        "minimum_robot_samples": minimum_robot_samples,
        "transition_margin_s": transition_margin_s,
        "edge_trim_s": edge_trim_s,
        "train_fraction_of_smallest_class": train_fraction_of_smallest_class,
        "random_seed": random_seed,
        "elapsed_s": round(elapsed_s, 1),
        "outputs": {
            modality: {"train": str(write_paths[modality]["train"]),
                       "test": str(write_paths[modality]["test"])}
            for modality in modalities_to_build
        },
    })


if __name__ == "__main__":
    main()
