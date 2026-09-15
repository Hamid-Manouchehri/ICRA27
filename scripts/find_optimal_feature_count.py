"""find_optimal_feature_count.py -- Per modality, LOSO-sweep the number
of top-ranked features N and find the N with the best held-out
accuracy. Per fold: rank the training subjects' features once, re-fit/
test at each N -- ranking never sees the held-out subject. "joint" =
robot's own features + joint_extra_features merged in (not the full
joint bank).

Inputs (exact filenames in User Settings, read_*_csv):
- 8 feature CSVs: features_{robot,gaze,joint,all}_{train,test}.csv

Outputs (exact filenames in User Settings, write_*):
- optimal_feature_count_{robot,gaze,joint,all}.csv, one per modality in
  modalities_to_sweep
- feature_ranking_{robot,gaze,joint,all}.csv, one per modality in
  modalities_to_sweep
- optimal_feature_count.png
- optimal_feature_count_summary.csv

Depends on:
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
from sklearn.metrics import classification_report

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
# last wrote. "joint" modality reads robot's CSVs as its base and
# merges joint_extra_features on top from read_joint_*_csv.
read_robot_train_csv = "features_robot_train_W_1s.csv"
read_robot_test_csv  = "features_robot_test_W_1s.csv"
read_gaze_train_csv  = "features_gaze_train_W_1s.csv"
read_gaze_test_csv   = "features_gaze_test_W_1s.csv"
read_joint_train_csv = "features_joint_train_W_1s.csv"
read_joint_test_csv  = "features_joint_test_W_1s.csv"
read_all_train_csv   = "features_all_train_W_1s.csv"
read_all_test_csv    = "features_all_test_W_1s.csv"

# TODO explicit output filenames (data/processed_data/ for .csv,
# data/images/ for .png), one PER MODALITY -- a fixed 4-entry lookup,
# not a naming pattern, since the universe of modalities is fixed
# (robot/gaze/joint/all); modalities_to_sweep below just selects which
# of these four get written on a given run.
write_optimal_feature_count_robot_csv = "optimal_feature_count_robot_W_1s.csv"
write_optimal_feature_count_gaze_csv  = "optimal_feature_count_gaze_W_1s.csv"
write_optimal_feature_count_joint_csv = "optimal_feature_count_joint_W_1s.csv"
write_optimal_feature_count_all_csv   = "optimal_feature_count_all_W_1s.csv"
write_feature_ranking_robot_csv = "feature_ranking_robot_W_1s.csv"
write_feature_ranking_gaze_csv  = "feature_ranking_gaze_W_1s.csv"
write_feature_ranking_joint_csv = "feature_ranking_joint_W_1s.csv"
write_feature_ranking_all_csv   = "feature_ranking_all_W_1s.csv"
write_optimal_feature_count_png = "optimal_feature_count_W_1s.png"
write_optimal_feature_count_summary_csv = "optimal_feature_count_summary_W_1s.csv"

# TODO display-only label for plot titles -- NOT used to build any file
# path (those are all explicit above). Keep it in sync with the tag in
# the filenames above so titles stay accurate.
dataset_label_for_titles = "w_1s"

modalities_to_sweep = ("robot", "all")  # TODO subset of "robot","gaze","joint","all"

# TODO "joint" = robot's own features + these specific joint-bank
# columns merged in, instead of the joint bank's ~45 features alone.
joint_extra_features = [
    "mean_curvature_when_gaze_far",
    "mean_curvature_when_gaze_close",
    "gaze_to_ee_speed_ratio",
]

# TODO candidate N values per modality; clipped to what's available, and
# "all features" is always added as a reference point. 125-200 matter
# for "all" (candidate pool ~200), not for robot (~57).
feature_counts_to_try = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50]

# TODO drop known-confounded features from the candidate pool before
# ranking/sweeping -- independent toggles (see POSITIONAL_FEATURE_NAMES/
# FORCE_FEATURE_NAMES below).
exclude_positional_features = True
exclude_force_features = True

# TODO pick the smallest N within this much accuracy of the peak, not
# just the raw argmax -- avoids picking a much bigger N for noise.
selection_tolerance = 0.01

train_fraction_of_smallest_class = 0.5  # TODO balanced-draw fraction per fold
n_estimators = 300
random_seed = 42
# -------------------- End User Settings --------------------

metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]
target_column = "trial_class_label"

# Known label-confounded features (see classifier_feature_bank.py and
# plot_top20_modality_comparison.py's positional_feature_names), split
# by category so exclude_positional_features/exclude_force_features can
# be toggled independently.
POSITIONAL_FEATURE_NAMES = {
    "robot": {
        # correlates with shape geometry, not behaviour
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
    # Same raw-position stats as robot's list above, mirrored onto gaze_x/
    # gaze_y -- same shape-geometry/screen-location confound concern.
    # NOT included here (a judgment call, not asked for): mean_distance_
    # from_display_center / display_center_fraction -- position-DERIVED
    # but a different family; ask if you want those excluded too.
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
        # DAMPING-CONFOUNDED -- see classifier_feature_bank.py docstring
        "mean_force_magnitude", "std_force_magnitude", "max_force_magnitude",
        "median_force_magnitude", "rms_force_magnitude", "mean_force_x",
        "mean_force_y", "std_force_x", "std_force_y",
    },
    "gaze": set(),
    # Unused now that "joint" means robot + joint_extra_features --
    # kept for reference / reverting to the full joint bank.
    "joint": {
        "angle_gaze_offset_to_force", "correlation_gaze_ee_distance_force",
        "correlation_gaze_speed_force", "mean_gaze_ee_distance_high_force",
        "mean_gaze_ee_distance_low_force", "fraction_gaze_close_during_high_force",
    },
}
# "all"'s columns are bank-prefixed (build_modality_feature_dataset_
# all.py) -- reuse the sets above with their bank's prefix added.
POSITIONAL_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in POSITIONAL_FEATURE_NAMES["robot"]} |
    {f"gaze_{name}" for name in POSITIONAL_FEATURE_NAMES["gaze"]}
)
FORCE_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in FORCE_FEATURE_NAMES["robot"]} |
    {f"joint_{name}" for name in FORCE_FEATURE_NAMES["joint"]}
)

# Built from the explicit read_*_csv/write_*_csv settings above -- one
# dict lookup site per direction (read_pooled_table, save_feature_
# ranking, main) instead of re-typing the four-way branch everywhere a
# modality's paths are needed.
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
write_optimal_feature_count_csv_by_modality = {
    "robot": write_optimal_feature_count_robot_csv,
    "gaze": write_optimal_feature_count_gaze_csv,
    "joint": write_optimal_feature_count_joint_csv,
    "all": write_optimal_feature_count_all_csv,
}
write_feature_ranking_csv_by_modality = {
    "robot": write_feature_ranking_robot_csv,
    "gaze": write_feature_ranking_gaze_csv,
    "joint": write_feature_ranking_joint_csv,
    "all": write_feature_ranking_all_csv,
}


def merge_extra_features(base_table: pd.DataFrame, split: str,
                          extra_csv_path: Path, extra_features: list) -> pd.DataFrame:
    """Merge specific named columns from another modality's CSV onto
    base_table, matched by trial+window identity -- same technique as
    train_simple_classifier.py's load_extra_features."""
    extra_table = pd.read_csv(extra_csv_path)
    key_columns = [c for c in
                   ("subject", "shape", "damping_n_s_m", "bandwidth",
                    "direction", "window_index")
                   if c in base_table.columns and c in extra_table.columns]
    merged = base_table.merge(
        extra_table[key_columns + extra_features],
        on=key_columns, how="left", validate="one_to_one", indicator=True)
    unmatched = int((merged["_merge"] == "left_only").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} {split} windows had no matching row in "
            f"{extra_csv_path.name} -- rebuild both modalities with "
            f"matching settings.")
    merged = merged.drop(columns="_merge")

    na_count = int(merged[extra_features].isna().any(axis=1).sum())
    if na_count:
        print(f"  {extra_csv_path.name}->joint: dropping {na_count} of "
              f"{len(merged)} {split} windows (extra feature NaN for "
              f"that window).")
        merged = merged.dropna(subset=extra_features).reset_index(drop=True)
    return merged


def read_pooled_table(modality: str) -> pd.DataFrame:
    if modality == "all":
        train_table = pd.read_csv(all_modality_csv_paths["all"]["train"])
        test_table = pd.read_csv(all_modality_csv_paths["all"]["test"])
        return pd.concat([train_table, test_table], ignore_index=True)

    base_modality = "robot" if modality == "joint" else modality
    train_table = pd.read_csv(all_modality_csv_paths[base_modality]["train"])
    test_table = pd.read_csv(all_modality_csv_paths[base_modality]["test"])
    if modality == "joint" and joint_extra_features:
        train_table = merge_extra_features(
            train_table, "train", all_modality_csv_paths["joint"]["train"],
            joint_extra_features)
        test_table = merge_extra_features(
            test_table, "test", all_modality_csv_paths["joint"]["test"],
            joint_extra_features)
    return pd.concat([train_table, test_table], ignore_index=True)


def candidate_feature_columns(pooled_table: pd.DataFrame, modality: str) -> list:
    all_features = [c for c in pooled_table.columns
                     if c not in metadata_columns and c != target_column]
    # "joint" is robot's own features + joint_extra_features -- filter
    # using ROBOT's confound sets, never drop the extra joint features.
    confound_key = "robot" if modality == "joint" else modality
    confounded = set()
    if exclude_positional_features:
        confounded |= POSITIONAL_FEATURE_NAMES[confound_key]
    if exclude_force_features:
        confounded |= FORCE_FEATURE_NAMES[confound_key]
    confounded -= set(joint_extra_features)
    return [f for f in all_features if f not in confounded]


def fit_model(X, y) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(X, y)
    return model


def draw_balanced_train(table: pd.DataFrame) -> pd.DataFrame:
    class_counts = table[target_column].value_counts()
    per_class = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)), 1)
    parts = [rows.sample(n=min(per_class, len(rows)), random_state=random_seed)
             for _, rows in table.groupby(target_column)]
    return pd.concat(parts, ignore_index=True)


def save_feature_ranking(modality: str, pooled_table: pd.DataFrame,
                          candidate_features: list) -> None:
    """One descriptive ranking, fit on a full balanced draw across ALL
    subjects -- NOT a LOSO evaluation, just showing which candidate
    features the model leans on (same pattern as plot_top20_modality_
    comparison.py's fit_full_importance_ranking)."""
    balanced = draw_balanced_train(pooled_table)
    model = fit_model(
        balanced[candidate_features].to_numpy(dtype=float),
        balanced[target_column].to_numpy())
    ranking = pd.DataFrame({
        "feature": candidate_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    path = (PROCESSED_DATA_DIRECTORY /
            write_feature_ranking_csv_by_modality[modality])
    ranking.to_csv(path, index=False)
    print(f"Wrote: {path}")


def sweep_modality(modality: str) -> tuple:
    pooled_table = read_pooled_table(modality)
    candidate_features = candidate_feature_columns(pooled_table, modality)
    class_order = sorted(pooled_table[target_column].unique())
    subjects = sorted(int(s) for s in pooled_table["subject"].unique())

    feature_counts = sorted(
        set(min(n, len(candidate_features)) for n in feature_counts_to_try)
        | {len(candidate_features)})

    label = (f"{modality.upper()} (robot + {len(joint_extra_features)} "
              f"joint features)" if modality == "joint" else modality.upper())
    print(f"\n{'=' * 70}\n{label}: {len(candidate_features)} "
          f"candidate features ({len(subjects)} subjects), sweeping N = "
          f"{feature_counts}\n{'=' * 70}")

    pooled_true = {n: [] for n in feature_counts}
    pooled_predicted = {n: [] for n in feature_counts}
    for held_out in subjects:
        test_table = pooled_table[pooled_table["subject"] == held_out]
        train_table = draw_balanced_train(
            pooled_table[pooled_table["subject"] != held_out])
        y_train = train_table[target_column].to_numpy()
        y_test = test_table[target_column].to_numpy()

        ranking_model = fit_model(
            train_table[candidate_features].to_numpy(dtype=float), y_train)
        fold_ranking = pd.Series(
            ranking_model.feature_importances_, index=candidate_features
        ).sort_values(ascending=False)

        for n in feature_counts:
            top_n = fold_ranking.index[:n].tolist()
            model = fit_model(train_table[top_n].to_numpy(dtype=float), y_train)
            y_predicted = model.predict(test_table[top_n].to_numpy(dtype=float))
            pooled_true[n].append(y_test)
            pooled_predicted[n].append(y_predicted)

    rows = []
    for n in feature_counts:
        y_true_n = np.concatenate(pooled_true[n])
        y_predicted_n = np.concatenate(pooled_predicted[n])
        report = classification_report(
            y_true_n, y_predicted_n, labels=class_order, output_dict=True)
        rows.append(dict(
            n_features=n, accuracy=report["accuracy"],
            macro_f1=report["macro avg"]["f1-score"]))
        print(f"  N={n:>4}: accuracy={report['accuracy']:.3f}  "
              f"macro_f1={report['macro avg']['f1-score']:.3f}")

    return pd.DataFrame(rows), pooled_table, candidate_features


def pick_best_n(results: pd.DataFrame) -> tuple:
    """(argmax_n, parsimonious_n) -- see selection_tolerance above."""
    argmax_row = results.loc[results["accuracy"].idxmax()]
    near_peak = results[
        results["accuracy"] >= argmax_row["accuracy"] - selection_tolerance]
    parsimonious_row = near_peak.loc[near_peak["n_features"].idxmin()]
    return argmax_row, parsimonious_row


def plot_results(all_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for modality, results in all_results.items():
        ax.plot(results["n_features"], results["accuracy"], marker="o",
                label=modality)
    ax.set_xlabel("Number of features (N)")
    ax.set_ylabel("LOSO accuracy")
    ax.set_title(f"Accuracy vs. feature count -- {dataset_label_for_titles}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = IMAGE_DATA_DIRECTORY / write_optimal_feature_count_png
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote: {path}")


def report_io_paths() -> None:
    for modality in modalities_to_sweep:
        if modality == "all":
            print(f"Input:  {all_modality_csv_paths['all']['train']}")
            print(f"Input:  {all_modality_csv_paths['all']['test']}")
        else:
            base_modality = "robot" if modality == "joint" else modality
            print(f"Input:  {all_modality_csv_paths[base_modality]['train']}")
            print(f"Input:  {all_modality_csv_paths[base_modality]['test']}")
            if modality == "joint" and joint_extra_features:
                print(f"Input:  {all_modality_csv_paths['joint']['train']} "
                      f"(extra columns: {', '.join(joint_extra_features)})")
                print(f"Input:  {all_modality_csv_paths['joint']['test']} "
                      f"(extra columns: {', '.join(joint_extra_features)})")
        print(f"Output: {PROCESSED_DATA_DIRECTORY / write_optimal_feature_count_csv_by_modality[modality]}")
        print(f"Output: {PROCESSED_DATA_DIRECTORY / write_feature_ranking_csv_by_modality[modality]}")
    print(f"Output: {IMAGE_DATA_DIRECTORY / write_optimal_feature_count_png}")
    print(f"Output: {PROCESSED_DATA_DIRECTORY / write_optimal_feature_count_summary_csv}\n")


def main() -> None:
    start_time_s = time.perf_counter()
    report_io_paths()
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []
    for modality in modalities_to_sweep:
        results, pooled_table, candidate_features = sweep_modality(modality)
        all_results[modality] = results
        results.to_csv(
            PROCESSED_DATA_DIRECTORY /
            write_optimal_feature_count_csv_by_modality[modality], index=False)
        save_feature_ranking(modality, pooled_table, candidate_features)

        argmax_row, parsimonious_row = pick_best_n(results)
        summary_rows.append(dict(
            modality=modality,
            best_n_argmax=int(argmax_row["n_features"]),
            best_accuracy_argmax=argmax_row["accuracy"],
            best_n_parsimonious=int(parsimonious_row["n_features"]),
            best_accuracy_parsimonious=parsimonious_row["accuracy"]))

    plot_results(all_results)

    summary_table = pd.DataFrame(summary_rows)
    print(f"\n{'=' * 70}\nBest N per modality (argmax = raw peak; "
          f"parsimonious = smallest N within {selection_tolerance:.3f} of "
          f"peak)\n{'=' * 70}")
    print(summary_table.to_string(index=False))
    summary_table.to_csv(
        PROCESSED_DATA_DIRECTORY / write_optimal_feature_count_summary_csv,
        index=False)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
