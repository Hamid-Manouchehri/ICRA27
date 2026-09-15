"""compare_loso_feature_sets.py -- LOSO comparison of several feature
sets side by side (see build_feature_sets), to quantify what gaze
contributes over robot data alone. "joint" is NOT robot+gaze
concatenated (zero shared features) -- "robot_plus_gaze"/"all" are the
real tests of gaze's marginal value. Top-N selection is redone inside
each fold (never leaks the held-out subject); baseline is each fold's
own majority-class rate.

Inputs (exact filenames in User Settings, read_*_csv):
- 8 feature CSVs: features_{robot,gaze,joint,all}_{train,test}.csv

Outputs (exact filenames in User Settings, write_*):
- loso_feature_set_comparison.csv / .png
- loso_feature_set_folds.csv
- feature_importance_all.png (top_n_features ranking, fit once on the
  full balanced "all" table -- separate from and not used by any LOSO
  fold's own accuracy)
- feature_importance_all_fold_s<subject>.png, one per held-out subject
  in the base "all" LOSO run (naming PATTERN in User Settings) -- from
  each fold's own model, no extra fit
- loso_lead_time_sweep.csv / .png (if lead_time_sweep_s is non-empty)
- loso_lead_time_sweep_folds.csv
- confusion_matrix_loso_<variant>.png, one per feature-set variant
  (naming PATTERN in User Settings, not a fixed filename -- the
  variant list size depends on feature_sets_to_run/run_top_n_variants/
  the lead-time sweep)

Depends on:
- classifier_feature_bank.py
- build_modality_feature_datasets.py (writes robot/gaze/joint/all CSVs)
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix)
from sklearn.svm import SVC

import classifier_feature_bank as cfb

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
# TODO explicit input filenames (data/processed_data/) -- each typed
# out, not built from a dataset-name tag, so the exact file this script
# reads is always visible here, not reconstructed mentally. Must match
# whatever build_modality_feature_datasets.py's output_dataset_name
# last wrote.
read_robot_train_csv = "features_robot_train_W_1s.csv"
read_robot_test_csv  = "features_robot_test_W_1s.csv"
read_gaze_train_csv  = "features_gaze_train_W_1s.csv"
read_gaze_test_csv   = "features_gaze_test_W_1s.csv"
read_joint_train_csv = "features_joint_train_W_1s.csv"
read_joint_test_csv  = "features_joint_test_W_1s.csv"
read_all_train_csv   = "features_all_train_W_1s.csv"
read_all_test_csv    = "features_all_test_W_1s.csv"

# TODO explicit output filenames (data/processed_data/ for .csv,
# data/images/ for .png) -- same reasoning as the inputs above.
write_feature_set_comparison_csv = "loso_feature_set_comparison_W_1s.csv"
write_feature_set_folds_csv = "loso_feature_set_folds_W_1s.csv"
write_feature_set_comparison_png = "loso_feature_set_comparison_W_1s.png"
write_feature_importance_all_png = "feature_importance_all_W_1s.png"

# TODO one feature-importance plot PER HELD-OUT SUBJECT for the base
# "all" LOSO run (9 subjects here, so 9 files) -- a naming PATTERN
# (see CLAUDE.md), not one setting per file, since subject count can
# change. {subject} is the held-out subject id.
write_feature_importance_per_fold_png_pattern = \
    "feature_importance_all_fold_s{subject}_W_1s.png"
write_lead_time_sweep_csv = "loso_lead_time_sweep_W_1s.csv"
write_lead_time_sweep_folds_csv = "loso_lead_time_sweep_folds_W_1s.csv"
write_lead_time_sweep_png = "loso_lead_time_sweep_W_1s.png"

# TODO how many top-ranked features each "_top" variant keeps [count].
top_n_features = 20

# TODO display-only label for plot titles -- NOT used to build any file
# path (those are all explicit above). Keep it in sync with the tag in
# the filenames above so titles stay accurate.
dataset_label_for_titles = "W_1s"

# TODO confusion-matrix filenames are one PER VARIANT (every entry of
# feature_sets_to_run, x2 if run_top_n_variants, plus every lead-time
# sweep point) -- an inherently dynamic list whose size depends on the
# settings below, so this is a naming PATTERN, not a fixed filename
# per file. {variant} is substituted with e.g. "robot", "robot_top20",
# "robot_top20@lead4s".
confusion_matrix_filename_pattern = "confusion_matrix_loso_{variant}_W_1s.png"

# TODO drop known-confounded features from every feature set before
# comparison -- matches find_optimal_feature_count.py / check_
# overfitting_vs_feature_count.py / losho_classifier.py.
exclude_positional_features = True
exclude_force_features = True

# TODO drop every feature whose VALUE is restricted to one coordinate
# axis (name ends "_x"/"_y", e.g. mean_gaze_x, std_velocity_y) -- not
# a magnitude/angle/correlation-xy that combines both axes (those keep
# names like gaze_dispersion_rms, mean_gaze_speed, correlation_gaze_xy
# and are unaffected). Independent of and overlaps with the two
# toggles above (e.g. mean_gaze_x is both positional AND single-axis).
exclude_single_axis_features = True

# TODO which feature sets to evaluate; keys of build_feature_sets().
feature_sets_to_run = ("robot", "gaze", "joint", "all")

# TODO also run each set restricted to its top N features.
run_top_n_variants = True

# TODO lead times [s] for the anticipation sweep: predict the label
# window_offset steps AHEAD from features at t. Must be multiples of
# window_step_s, since the offset is applied in window steps. Empty
# tuple skips the sweep.
lead_time_sweep_s = (0.0, 1.0, 2.0, 3.0)
window_step_s = 1.0  # TODO must match build_modality_feature_datasets.py

# TODO which feature sets enter the lead-time sweep -- any key of
# build_feature_sets() (e.g. "robot", "all", "gaze_pure", ...).
lead_time_feature_sets = ("robot", "all")

target_column = "trial_class_label"
metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]

# TODO the classifier fit for every LOSO fold -- change this one line
# to switch models, or add another entry to CLASSIFIER_REGISTRY below.
classifier_name = "random_forest"
classifier_kwargs = {"n_estimators": 300, "random_state": 42, "n_jobs": -1}

# Built from the explicit read_*_csv settings above -- one dict lookup
# site (modality_table(), all_modality_table()) instead of re-typing
# the four-way branch everywhere a modality's paths are needed.
all_modality_csv_paths = {
    "robot": {"train": PROCESSED_DATA_DIRECTORY / read_robot_train_csv,
              "test": PROCESSED_DATA_DIRECTORY / read_robot_test_csv},
    "gaze": {"train": PROCESSED_DATA_DIRECTORY / read_gaze_train_csv,
             "test": PROCESSED_DATA_DIRECTORY / read_gaze_test_csv},
    "joint": {"train": PROCESSED_DATA_DIRECTORY / read_joint_train_csv,
              "test": PROCESSED_DATA_DIRECTORY / read_joint_test_csv},
    "all": {"train": PROCESSED_DATA_DIRECTORY / read_all_train_csv,
            "test": PROCESSED_DATA_DIRECTORY / read_all_test_csv},
}

# TODO fraction of the smaller class drawn per class for each fold's
# training set -- matches train_modality_classifiers_loso.py.
train_fraction_of_smallest_class = 0.5
random_seed = 42

# Window identity, used to line robot and gaze rows up for concatenation.
key_columns = ["subject", "bandwidth", "shape", "direction", "window_index"]

# Same confound lists as find_optimal_feature_count.py / losho_
# classifier.py / check_overfitting_vs_feature_count.py -- kept in sync
# there, duplicated here (standalone).
POSITIONAL_FEATURE_NAMES = {
    "robot": {
        "mean_position_x", "mean_position_y", "std_position_x", "std_position_y",
        "variance_position_x", "variance_position_y", "minimum_position_x",
        "maximum_position_x", "range_position_x", "minimum_position_y",
        "maximum_position_y", "range_position_y", "median_position_x",
        "median_position_y", "iqr_position_x", "iqr_position_y",
        "mad_position_x", "mad_position_y", "skewness_position_x",
        "skewness_position_y", "kurtosis_position_x", "kurtosis_position_y",
        "covariance_position_xy", "correlation_position_xy",
        "position_dispersion_rms", "position_bcea_68",
        "position_x_shannon_entropy", "position_y_shannon_entropy",
        "position_x_mean_crossing_count", "position_y_mean_crossing_count",
        "position_x_lag1_autocorrelation", "position_y_lag1_autocorrelation",
    },
    "gaze": {
        "mean_gaze_x", "mean_gaze_y", "std_gaze_x", "std_gaze_y",
        "variance_gaze_x", "variance_gaze_y", "minimum_gaze_x",
        "maximum_gaze_x", "range_gaze_x", "minimum_gaze_y", "maximum_gaze_y",
        "range_gaze_y", "median_gaze_x", "median_gaze_y", "iqr_gaze_x",
        "iqr_gaze_y", "mad_gaze_x", "mad_gaze_y", "skewness_gaze_x",
        "skewness_gaze_y", "kurtosis_gaze_x", "kurtosis_gaze_y",
        "covariance_gaze_xy", "correlation_gaze_xy", "gaze_dispersion_rms",
        "bcea_68", "gaze_x_shannon_entropy", "gaze_y_shannon_entropy",
        "gaze_x_mean_crossing_count", "gaze_y_mean_crossing_count",
        "gaze_x_lag1_autocorrelation", "gaze_y_lag1_autocorrelation",
    },
    "joint": set(),
}
FORCE_FEATURE_NAMES = {
    "robot": {
        "mean_force_magnitude", "std_force_magnitude", "max_force_magnitude",
        "median_force_magnitude", "rms_force_magnitude", "mean_force_x",
        "mean_force_y", "std_force_x", "std_force_y",
    },
    "gaze": set(),
    "joint": {
        "angle_gaze_offset_to_force", "correlation_gaze_ee_distance_force",
        "correlation_gaze_speed_force", "mean_gaze_ee_distance_high_force",
        "mean_gaze_ee_distance_low_force", "fraction_gaze_close_during_high_force",
    },
}
POSITIONAL_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in POSITIONAL_FEATURE_NAMES["robot"]} |
    {f"gaze_{name}" for name in POSITIONAL_FEATURE_NAMES["gaze"]}
)
FORCE_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in FORCE_FEATURE_NAMES["robot"]} |
    {f"joint_{name}" for name in FORCE_FEATURE_NAMES["joint"]}
)

# Single-axis (x-only or y-only) feature names, verified against each
# one's own formula/docstring, not just name pattern -- e.g.
# mean_gaze_velocity_x is included (its OUTPUT is x-only) even though
# its formula also passes Gy in for internal cleaning/sync; combined-
# axis features (hypot/correlation-xy/angle) are excluded. Heavily
# overlaps POSITIONAL_FEATURE_NAMES on purpose -- this is an
# independent, self-contained category (see exclude_single_axis_
# features setting above), not a diff against the other two.
SINGLE_AXIS_FEATURE_NAMES = {
    "robot": {
        "mean_position_x", "mean_position_y", "std_position_x",
        "std_position_y", "variance_position_x", "variance_position_y",
        "minimum_position_x", "maximum_position_x", "range_position_x",
        "minimum_position_y", "maximum_position_y", "range_position_y",
        "median_position_x", "median_position_y", "iqr_position_x",
        "iqr_position_y", "mad_position_x", "mad_position_y",
        "skewness_position_x", "skewness_position_y",
        "kurtosis_position_x", "kurtosis_position_y",
        "mean_velocity_x", "mean_velocity_y", "std_velocity_x",
        "std_velocity_y", "position_x_shannon_entropy",
        "position_y_shannon_entropy", "position_x_mean_crossing_count",
        "position_y_mean_crossing_count",
        "position_x_lag1_autocorrelation",
        "position_y_lag1_autocorrelation", "mean_force_x", "mean_force_y",
        "std_force_x", "std_force_y",
    },
    "gaze": {
        "mean_gaze_x", "mean_gaze_y", "std_gaze_x", "std_gaze_y",
        "variance_gaze_x", "variance_gaze_y", "minimum_gaze_x",
        "maximum_gaze_x", "range_gaze_x", "minimum_gaze_y",
        "maximum_gaze_y", "range_gaze_y", "median_gaze_x", "median_gaze_y",
        "iqr_gaze_x", "iqr_gaze_y", "mad_gaze_x", "mad_gaze_y",
        "skewness_gaze_x", "skewness_gaze_y", "kurtosis_gaze_x",
        "kurtosis_gaze_y", "mean_gaze_velocity_x", "mean_gaze_velocity_y",
        "std_gaze_velocity_x", "std_gaze_velocity_y",
        "gaze_x_shannon_entropy", "gaze_y_shannon_entropy",
        "gaze_x_mean_crossing_count", "gaze_y_mean_crossing_count",
        "gaze_x_lag1_autocorrelation", "gaze_y_lag1_autocorrelation",
        "mean_relative_x", "mean_relative_y", "std_relative_x",
        "std_relative_y", "gaze_ee_x_correlation", "gaze_ee_y_correlation",
        "max_x_cross_correlation", "x_lead_at_max_correlation_s",
        "max_y_cross_correlation", "y_lead_at_max_correlation_s",
    },
    "joint": {
        "mean_relative_x", "mean_relative_y", "std_relative_x",
        "std_relative_y", "gaze_ee_x_correlation", "gaze_ee_y_correlation",
        "max_x_cross_correlation", "x_lead_at_max_correlation_s",
        "max_y_cross_correlation", "y_lead_at_max_correlation_s",
    },
}
SINGLE_AXIS_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in SINGLE_AXIS_FEATURE_NAMES["robot"]} |
    {f"gaze_{name}" for name in SINGLE_AXIS_FEATURE_NAMES["gaze"]} |
    {f"joint_{name}" for name in SINGLE_AXIS_FEATURE_NAMES["joint"]}
)


# Duplicated from train_modality_classifiers.py rather than imported --
# standalone-script convention (see e.g. losho_classifier.py), so this
# script has no cross-file settings coupling.
CLASSIFIER_REGISTRY = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "svm": SVC,
}


def build_classifier(name: str, **kwargs):
    if name not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier_name '{name}'. Available: "
            f"{sorted(CLASSIFIER_REGISTRY)} (or add your own to "
            f"CLASSIFIER_REGISTRY).")
    return CLASSIFIER_REGISTRY[name](**kwargs)


def flatten_classification_report(
        modality: str, report: dict, class_order: list) -> dict:
    """One flat dict: accuracy, then precision/recall/f1 for each class
    in class_order, then macro and weighted averages."""
    row = {"modality": modality, "accuracy": report["accuracy"]}
    for class_name in class_order:
        for metric in ("precision", "recall", "f1-score"):
            row[f"{class_name}_{metric.replace('-score', '')}"] = \
                report[class_name][metric]
    for avg_name in ("macro avg", "weighted avg"):
        prefix = avg_name.split()[0]
        for metric in ("precision", "recall", "f1-score"):
            row[f"{prefix}_{metric.replace('-score', '')}"] = \
                report[avg_name][metric]
    return row


def apply_confound_exclusion(features: list, modality: str) -> list:
    confounded = set()
    if exclude_positional_features:
        confounded |= POSITIONAL_FEATURE_NAMES[modality]
    if exclude_force_features:
        confounded |= FORCE_FEATURE_NAMES[modality]
    if exclude_single_axis_features:
        confounded |= SINGLE_AXIS_FEATURE_NAMES[modality]
    return [f for f in features if f not in confounded]


def modality_table(modality: str) -> pd.DataFrame:
    """One modality's train+test windows pooled back into one table, the
    same way train_modality_classifiers_loso.py pools them."""
    paths = all_modality_csv_paths[modality]
    return pd.concat(
        [pd.read_csv(paths["train"]), pd.read_csv(paths["test"])],
        ignore_index=True)


def feature_columns_of(table: pd.DataFrame) -> list:
    return [c for c in table.columns
            if c not in metadata_columns and c != target_column]


def robot_dependent_gaze_features() -> set:
    """Gaze-bank features whose value changes when only the ROBOT path
    changes -- i.e. those that need ur5e_*.csv despite living in the
    "gaze" bank. Probed empirically rather than matched by name, with
    two different robot paths (position AND speed differ) plus a pure
    translation, since some features are translation-invariant and
    would otherwise look robot-independent."""
    config = cfb.FeatureConfig(
        workspace_center_xy_m=(0.0, -0.65), workspace_half_extent_m=0.2,
        low_speed_threshold_m_s=0.05, high_speed_threshold_m_s=0.3,
        close_thresholds_m=(0.01, 0.02, 0.05))

    sample_count = 40
    gaze_rng = np.random.default_rng(0)
    Gt = np.linspace(0.0, 0.5, sample_count)
    Gx = gaze_rng.normal(0.0, 0.05, sample_count)
    Gy = gaze_rng.normal(-0.65, 0.05, sample_count)
    Pt = np.linspace(0.0, 0.5, sample_count)

    first_rng = np.random.default_rng(1)
    Px1 = first_rng.normal(0.0, 0.05, sample_count)
    Py1 = first_rng.normal(-0.65, 0.05, sample_count)
    second_rng = np.random.default_rng(2)
    Px2 = second_rng.normal(0.1, 0.2, sample_count)
    Py2 = second_rng.normal(-0.5, 0.3, sample_count)

    robot_dependent = set()
    for name, feature_fn in cfb.build_gaze_feature_bank(config):
        baseline = feature_fn(Gx, Gy, Gt, Px1, Py1, Pt)
        for Px, Py in ((Px2, Py2), (Px1 + 0.13, Py1 - 0.07)):
            other = feature_fn(Gx, Gy, Gt, Px, Py, Pt)
            unchanged = (np.isnan(baseline) and np.isnan(other)) or \
                np.isclose(baseline, other, equal_nan=True)
            if not unchanged:
                robot_dependent.add(name)
                break
    return robot_dependent


def all_modality_table() -> pd.DataFrame:
    """read_all_train_csv/read_all_test_csv -- gaze+robot+joint (249
    features, bank-prefixed) already combined, pooled the same way
    modality_table() pools the single-modality CSVs. Same as
    modality_table("all"); kept separate only because build_feature_
    sets() needs this table AND robot/gaze/joint tables together."""
    return modality_table("all")


def build_feature_sets() -> dict:
    """{name: (table, feature_columns)} for every set being compared."""
    robot_table = modality_table("robot")
    gaze_table = modality_table("gaze")
    joint_table = modality_table("joint")
    all_table = all_modality_table()
    all_features = apply_confound_exclusion(
        feature_columns_of(all_table), "all")

    robot_features = apply_confound_exclusion(
        feature_columns_of(robot_table), "robot")
    gaze_features = apply_confound_exclusion(
        feature_columns_of(gaze_table), "gaze")
    joint_features = apply_confound_exclusion(
        feature_columns_of(joint_table), "joint")
    pure_gaze_features = [f for f in gaze_features
                          if f not in robot_dependent_gaze_features()]

    # Robot + gaze concatenated on window identity. Suffix any name
    # present in both banks so neither silently overwrites the other.
    shared_names = set(robot_features) & set(gaze_features)
    gaze_renamed = gaze_table.rename(
        columns={name: f"{name}_gaze" for name in shared_names})
    combined_table = robot_table.merge(
        gaze_renamed[key_columns + [
            f"{f}_gaze" if f in shared_names else f for f in gaze_features]],
        on=key_columns, how="inner", validate="one_to_one")
    combined_gaze_features = [
        f"{f}_gaze" if f in shared_names else f for f in gaze_features]
    combined_pure_features = [
        f"{f}_gaze" if f in shared_names else f for f in pure_gaze_features]

    print(f"robot {len(robot_features)} | gaze {len(gaze_features)} "
          f"(pure {len(pure_gaze_features)}) | joint {len(joint_features)} "
          f"| robot+gaze {len(robot_features) + len(combined_gaze_features)} "
          f"| all {len(all_features)}")
    if len(combined_table) != len(robot_table):
        print(f"WARNING: robot/gaze merge kept {len(combined_table)} of "
              f"{len(robot_table)} windows -- window identities differ.")

    return {
        "robot": (robot_table, robot_features),
        "gaze": (gaze_table, gaze_features),
        "gaze_pure": (gaze_table, pure_gaze_features),
        "joint": (joint_table, joint_features),
        "robot_plus_gaze": (
            combined_table, robot_features + combined_gaze_features),
        "robot_plus_gaze_pure": (
            combined_table, robot_features + combined_pure_features),
        "all": (all_table, all_features),
    }


def draw_balanced_train(table: pd.DataFrame, seed: int,
                        label_column: str = None) -> pd.DataFrame:
    label_column = label_column or target_column
    class_counts = table[label_column].value_counts()
    per_class_count = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)), 1)
    parts = [class_rows.sample(n=min(per_class_count, len(class_rows)),
                               random_state=seed)
             for _, class_rows in table.groupby(label_column)]
    return pd.concat(parts, ignore_index=True)


def shift_labels_forward(table: pd.DataFrame, window_offset: int):
    """Attach, as "future_label", the label of the window window_offset
    steps later in the SAME trial. window_index is assigned before any
    window is filtered out, so index i and i+k are exactly
    k * window_step_s apart in time even where windows were dropped.
    Rows whose future window was dropped are removed."""
    if window_offset == 0:
        shifted = table.copy()
        shifted["future_label"] = shifted[target_column]
        return shifted

    trial_columns = [c for c in key_columns if c != "window_index"]
    future = table[trial_columns + ["window_index", target_column]].copy()
    future["window_index"] = future["window_index"] - window_offset
    future = future.rename(columns={target_column: "future_label"})
    return table.merge(future, on=trial_columns + ["window_index"],
                       how="inner", validate="one_to_one")


def run_loso(set_name: str, table: pd.DataFrame, feature_columns: list,
             keep_top_n: int = None, label_column: str = None,
             on_fold_model: callable = None) -> tuple:
    """LOSO over every subject. Returns (summary_row, fold_rows,
    pooled_true, pooled_predicted) -- the last two are every held-out
    subject's true/predicted label, concatenated in fold order, for
    building a confusion matrix without refitting.

    on_fold_model, if given, is called as
    on_fold_model(held_out_subject, model, fold_features) right after
    each fold's own model is fit -- e.g. to plot that fold's feature
    importances -- reusing the model already fit for that fold's
    prediction, not an extra fit."""
    label_column = label_column or target_column
    subjects = sorted(int(s) for s in table["subject"].unique())
    class_order = sorted(table[label_column].unique())
    fold_rows = []
    all_true, all_predicted = [], []

    for held_out_subject in subjects:
        test_table = table[table["subject"] == held_out_subject]
        train_table = draw_balanced_train(
            table[table["subject"] != held_out_subject], random_seed,
            label_column)

        fold_features = feature_columns
        if keep_top_n is not None and keep_top_n < len(feature_columns):
            # Rank on THIS fold's training data only -- ranking on the
            # pooled table would leak the held-out subject.
            ranker = build_classifier(classifier_name, **classifier_kwargs)
            ranker.fit(train_table[feature_columns], train_table[label_column])
            order = np.argsort(ranker.feature_importances_)[::-1]
            fold_features = [feature_columns[i] for i in order[:keep_top_n]]

        model = build_classifier(classifier_name, **classifier_kwargs)
        model.fit(train_table[fold_features], train_table[label_column])
        if on_fold_model is not None:
            on_fold_model(held_out_subject, model, fold_features)
        predicted = model.predict(test_table[fold_features])

        fold_accuracy = accuracy_score(test_table[label_column], predicted)
        fold_baseline = test_table[label_column].value_counts(
            normalize=True).max()
        # Persistence: assume the regime does not change over the lead
        # time. The controller gets this for free from the force sensor,
        # so an anticipatory model must beat it, not just the majority.
        if label_column != target_column:
            persistence_accuracy = accuracy_score(
                test_table[label_column], test_table[target_column])
        else:
            persistence_accuracy = float("nan")
        fold_rows.append({
            "feature_set": set_name, "subject": held_out_subject,
            "accuracy": fold_accuracy, "majority_baseline": fold_baseline,
            "persistence_baseline": persistence_accuracy,
            "lift": fold_accuracy - fold_baseline,
            "lift_over_persistence": fold_accuracy - persistence_accuracy,
            "train_windows": len(train_table), "test_windows": len(test_table),
            "feature_count": len(fold_features),
        })
        all_true.append(test_table[label_column])
        all_predicted.append(pd.Series(predicted))

    pooled_true = pd.concat(all_true, ignore_index=True)
    pooled_predicted = pd.concat(all_predicted, ignore_index=True)
    report = classification_report(
        pooled_true, pooled_predicted, output_dict=True, zero_division=0)
    summary_row = flatten_classification_report(
        set_name, report, class_order)
    fold_table = pd.DataFrame(fold_rows)
    summary_row["feature_count"] = int(fold_table["feature_count"].iloc[0])
    summary_row["mean_fold_accuracy"] = fold_table["accuracy"].mean()
    summary_row["mean_majority_baseline"] = \
        fold_table["majority_baseline"].mean()
    summary_row["mean_lift"] = fold_table["lift"].mean()
    summary_row["mean_persistence_baseline"] = \
        fold_table["persistence_baseline"].mean()
    summary_row["mean_lift_over_persistence"] = \
        fold_table["lift_over_persistence"].mean()
    summary_row["worst_fold_accuracy"] = fold_table["accuracy"].min()
    summary_row["class_order"] = class_order
    return summary_row, fold_rows, pooled_true, pooled_predicted


def run_lead_time_sweep(feature_sets: dict) -> tuple:
    """Predict the label lead_time_s AHEAD from features at t, for each
    feature set, so gaze's anticipation can be compared against force's
    (and against simply assuming the regime persists). Restricted to
    each set's own top_n_features (ranked per-fold, as in run_loso) so
    sets with more raw columns (e.g. "all" at 172 vs "robot" at 57)
    don't win just from having more features available."""
    summary_rows, fold_rows = [], []
    print(f"\n{'=' * 78}\nLead-time sweep -- predicting label(t + lead) "
          f"from features(t), top {top_n_features} features per set"
          f"\n{'=' * 78}")

    for lead_time_s in lead_time_sweep_s:
        window_offset = int(round(lead_time_s / window_step_s))
        for set_name in lead_time_feature_sets:
            table, feature_columns = feature_sets[set_name]
            shifted = shift_labels_forward(table, window_offset)
            variant_name = f"{set_name}_top{top_n_features}@lead{lead_time_s:g}s"

            summary_row, variant_folds, _, _ = run_loso(
                variant_name, shifted, feature_columns,
                keep_top_n=top_n_features, label_column="future_label")
            summary_row["lead_time_s"] = lead_time_s
            summary_row["base_feature_set"] = set_name
            summary_row["windows"] = len(shifted)
            for fold_row in variant_folds:
                fold_row["lead_time_s"] = lead_time_s
                fold_row["base_feature_set"] = set_name
            summary_rows.append(summary_row)
            fold_rows.extend(variant_folds)

            persistence = summary_row["mean_persistence_baseline"]
            print(f"  {variant_name:<28} acc={summary_row['accuracy']:.3f}  "
                  f"majority={summary_row['mean_majority_baseline']:.3f}  "
                  f"persistence={persistence:.3f}  "
                  f"vs_persistence={summary_row['mean_lift_over_persistence']:+.3f}"
                  f"  n_win={len(shifted)}")
    return summary_rows, fold_rows


def plot_comparison(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(figsize=(11, 0.5 * len(summary_table) + 3))
    ordered = summary_table.sort_values("accuracy")
    positions = np.arange(len(ordered))
    axes.barh(positions, ordered["accuracy"], color="#4C72B0", label="LOSO accuracy")
    axes.plot(ordered["mean_majority_baseline"], positions, "kd",
              label="majority-class baseline")
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{r.modality}  (n={r.feature_count})"
         for r in ordered.itertuples()], fontsize=8)
    axes.set_xlabel("Accuracy")
    axes.set_xlim(0.0, 1.0)
    axes.grid(axis="x", alpha=0.3)
    axes.set_title(f"LOSO by feature set -- {dataset_label_for_titles}")
    axes.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_lead_time_sweep(sweep_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(figsize=(8, 5))
    for set_name, rows in sweep_table.groupby("base_feature_set"):
        rows = rows.sort_values("lead_time_s")
        axes.plot(rows["lead_time_s"], rows["accuracy"], "o-", label=set_name)
    reference = sweep_table.sort_values("lead_time_s").drop_duplicates(
        "lead_time_s")
    axes.plot(reference["lead_time_s"], reference["mean_persistence_baseline"],
              "k--", label="persistence (assume no change)")
    axes.plot(reference["lead_time_s"], reference["mean_majority_baseline"],
              ":", color="gray", label="majority class")
    axes.set_xlabel("Lead time [s]")
    axes.set_ylabel("LOSO accuracy")
    axes.set_ylim(0.4, 1.0)
    axes.grid(alpha=0.3)
    axes.set_title(f"Anticipation: predicting label(t + lead) -- "
                   f"{dataset_label_for_titles}, top {top_n_features} "
                   f"features")
    axes.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(
        model, feature_columns: list, title: str, output_path: Path) -> None:
    """Top top_n_features bar chart from an already-fit model's
    feature_importances_ -- same style as train_simple_classifier.py's
    plot_feature_importance()."""
    ranking = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)
    top_ranking = ranking.head(top_n_features).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_ranking) + 1.5))
    ax.barh(top_ranking.index, top_ranking.values)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_fold_feature_importance(held_out_subject, model, feature_columns):
    """run_loso() on_fold_model callback: plot this fold's own feature-
    importance ranking, from the model already fit for that fold's
    prediction -- no extra fit, no leakage beyond what that fold
    already uses."""
    path = (IMAGE_DATA_DIRECTORY /
            write_feature_importance_per_fold_png_pattern.format(
                subject=held_out_subject))
    plot_feature_importance(
        model, feature_columns,
        f"Top {top_n_features} features -- all, held out s"
        f"{held_out_subject} ({dataset_label_for_titles})",
        path)


def plot_confusion_matrix(
        set_name: str, pooled_true: pd.Series, pooled_predicted: pd.Series,
        class_order: list, output_path: Path) -> None:
    matrix = confusion_matrix(pooled_true, pooled_predicted, labels=class_order)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_order)
    fig, axes = plt.subplots(figsize=(4.5, 5.2))
    display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")
    axes.set_title(
        f"LOSO pooled confusion matrix\n{set_name} "
        f"({dataset_label_for_titles})", fontsize=9)

    report_dict = classification_report(
        pooled_true, pooled_predicted, labels=class_order, output_dict=True,
        zero_division=0)
    macro = report_dict["macro avg"]
    metrics_text = (
        f"Accuracy: {report_dict['accuracy']:.3f}    "
        f"Precision: {macro['precision']:.3f}    "
        f"Recall: {macro['recall']:.3f}    "
        f"F1-score: {macro['f1-score']:.3f}")
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    start_time_s = time.perf_counter()
    for path in all_modality_csv_paths.values():
        print(f"Input:  {path['train']}")
        print(f"Input:  {path['test']}")
    print(f"Output: {PROCESSED_DATA_DIRECTORY / write_feature_set_comparison_csv}")
    print(f"Output: {PROCESSED_DATA_DIRECTORY / write_feature_set_folds_csv}")
    print(f"Output: {IMAGE_DATA_DIRECTORY / write_feature_set_comparison_png}")
    if "all" in feature_sets_to_run:
        print(f"Output: {IMAGE_DATA_DIRECTORY / write_feature_importance_all_png}")
        print(f"Output: {IMAGE_DATA_DIRECTORY / write_feature_importance_per_fold_png_pattern} "
              f"(one per held-out subject)")
    if lead_time_sweep_s:
        print(f"Output: {PROCESSED_DATA_DIRECTORY / write_lead_time_sweep_csv}")
        print(f"Output: {PROCESSED_DATA_DIRECTORY / write_lead_time_sweep_folds_csv}")
        print(f"Output: {IMAGE_DATA_DIRECTORY / write_lead_time_sweep_png}")
    print(f"Output: {IMAGE_DATA_DIRECTORY / confusion_matrix_filename_pattern} "
          f"(one per variant)")
    print()

    feature_sets = build_feature_sets()
    summary_rows, fold_rows = [], []
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for set_name in feature_sets_to_run:
        table, feature_columns = feature_sets[set_name]
        variants = [(set_name, None)]
        if run_top_n_variants and len(feature_columns) > top_n_features:
            variants.append(
                (f"{set_name}_top{top_n_features}", top_n_features))

        for variant_name, keep_top_n in variants:
            fold_callback = (
                save_fold_feature_importance
                if set_name == "all" and keep_top_n is None else None)
            summary_row, variant_folds, pooled_true, pooled_predicted = \
                run_loso(variant_name, table, feature_columns, keep_top_n,
                         on_fold_model=fold_callback)
            summary_rows.append(summary_row)
            fold_rows.extend(variant_folds)
            print(f"  {variant_name:<32} n={summary_row['feature_count']:>3}  "
                  f"acc={summary_row['accuracy']:.3f}  "
                  f"baseline={summary_row['mean_majority_baseline']:.3f}  "
                  f"lift={summary_row['mean_lift']:+.3f}")
            plot_confusion_matrix(
                variant_name, pooled_true, pooled_predicted,
                summary_row["class_order"],
                IMAGE_DATA_DIRECTORY / confusion_matrix_filename_pattern.format(
                    variant=variant_name))

        if set_name == "all":
            # One extra fit on the full balanced table, purely for the
            # ranking plot -- separate from every LOSO fold above (each
            # of which ranks fresh on its own training data only), so
            # this never leaks a held-out subject into a fold's result.
            balanced_full = draw_balanced_train(table, random_seed)
            ranking_model = build_classifier(
                classifier_name, **classifier_kwargs)
            ranking_model.fit(
                balanced_full[feature_columns], balanced_full[target_column])
            feature_importance_path = (
                IMAGE_DATA_DIRECTORY / write_feature_importance_all_png)
            plot_feature_importance(
                ranking_model, feature_columns,
                f"Top {top_n_features} features -- all "
                f"({dataset_label_for_titles})",
                feature_importance_path)
            print(f"  -> {feature_importance_path.name}")

    if lead_time_sweep_s:
        sweep_summary, sweep_folds = run_lead_time_sweep(feature_sets)
        sweep_summary_path = (
            PROCESSED_DATA_DIRECTORY / write_lead_time_sweep_csv)
        sweep_folds_path = (
            PROCESSED_DATA_DIRECTORY / write_lead_time_sweep_folds_csv)
        PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_summary).to_csv(sweep_summary_path, index=False)
        pd.DataFrame(sweep_folds).to_csv(sweep_folds_path, index=False)
        plot_lead_time_sweep(
            pd.DataFrame(sweep_summary),
            IMAGE_DATA_DIRECTORY / write_lead_time_sweep_png)
        print(f"\nWrote:\n  {sweep_summary_path.name}\n"
              f"  {sweep_folds_path.name}")

    if not summary_rows:
        # feature_sets_to_run empty: sweep-only run, leave the existing
        # comparison CSVs alone rather than overwriting them with nothing.
        elapsed_s = time.perf_counter() - start_time_s
        print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")
        return

    summary_table = pd.DataFrame(summary_rows)
    fold_table = pd.DataFrame(fold_rows)

    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = PROCESSED_DATA_DIRECTORY / write_feature_set_comparison_csv
    folds_path = PROCESSED_DATA_DIRECTORY / write_feature_set_folds_csv
    image_path = IMAGE_DATA_DIRECTORY / write_feature_set_comparison_png
    summary_table.to_csv(summary_path, index=False)
    fold_table.to_csv(folds_path, index=False)
    plot_comparison(summary_table, image_path)

    print(f"\n{'=' * 78}\nLOSO feature-set comparison "
          f"({dataset_label_for_titles})\n"
          f"{'=' * 78}")
    print(summary_table[[
        "modality", "feature_count", "accuracy", "macro_f1",
        "mean_majority_baseline", "mean_lift", "worst_fold_accuracy",
    ]].rename(columns={"modality": "feature_set"}).to_string(index=False))
    print(f"\nWrote:\n  {summary_path.name}\n  {folds_path.name}\n"
          f"  {image_path.name}")

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
