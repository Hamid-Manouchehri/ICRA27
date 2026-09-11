"""check_overfitting_vs_feature_count.py -- Non-LOSO overfitting check:
per modality, sweep N (top-ranked features) on the pooled train/test
split, one fit per N, record BOTH train and test accuracy (no folds).
Different question from find_optimal_feature_count.py's LOSO sweep
(new-subject generalization) -- this checks model-capacity/
memorization as N grows.

Inputs:
- features_<modality>_{train,test}_<dataset_name>.csv

Outputs:
- data/images/overfitting_vs_feature_count_<dataset_name>
  [_<output_tag>].png
- data/processed_data/overfitting_vs_feature_count_<modality>_
  <dataset_name>[_<output_tag>].csv
- data/processed_data/overfitting_feature_ranking_<modality>_
  <dataset_name>[_<output_tag>].csv + matching .png (named
  "overfitting_feature_ranking_*", not "feature_ranking_*" -- a
  different draw than find_optimal_feature_count.py's file of that
  name; same name would overwrite one with the other)

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

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
dataset_name = "W_2s"  # TODO must match build_modality_feature_datasets.py
modalities_to_check = ("robot", "all")  # TODO "robot" | "gaze" | "joint" | "all"
output_tag = ""  # TODO appended to output filenames

# TODO candidate N values; clipped to what's available, and "all
# features" is always added as a reference point.
feature_counts_to_try = [
    1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 125, 150, 175, 200]

exclude_positional_features = True  # TODO see POSITIONAL_FEATURE_NAMES below
exclude_force_features = True  # TODO see FORCE_FEATURE_NAMES below

top_n_features_to_plot = 20  # TODO for the ranking bar chart

n_estimators = 300
random_seed = 42
# -------------------- End User Settings --------------------

_output_suffix = f"_{output_tag}" if output_tag else ""

metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]
target_column = "trial_class_label"

# Same confound lists as find_optimal_feature_count.py -- see that
# script for the full reasoning (shape-geometry / damping confounds).
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
# "all"'s columns are bank-prefixed (build_modality_feature_dataset_
# all.py).
POSITIONAL_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in POSITIONAL_FEATURE_NAMES["robot"]} |
    {f"gaze_{name}" for name in POSITIONAL_FEATURE_NAMES["gaze"]}
)
FORCE_FEATURE_NAMES["all"] = (
    {f"robot_{name}" for name in FORCE_FEATURE_NAMES["robot"]} |
    {f"joint_{name}" for name in FORCE_FEATURE_NAMES["joint"]}
)


def read_split_tables(modality: str) -> tuple:
    train_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY / f"features_{modality}_train_{dataset_name}.csv")
    test_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY / f"features_{modality}_test_{dataset_name}.csv")
    return train_table, test_table


def candidate_feature_columns(table: pd.DataFrame, modality: str) -> list:
    all_features = [c for c in table.columns
                     if c not in metadata_columns and c != target_column]
    confounded = set()
    if exclude_positional_features:
        confounded |= POSITIONAL_FEATURE_NAMES[modality]
    if exclude_force_features:
        confounded |= FORCE_FEATURE_NAMES[modality]
    return [f for f in all_features if f not in confounded]


def fit_model(X, y) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(X, y)
    return model


def check_modality(modality: str) -> pd.DataFrame:
    train_table, test_table = read_split_tables(modality)
    candidate_features = candidate_feature_columns(train_table, modality)
    y_train = train_table[target_column].to_numpy()
    y_test = test_table[target_column].to_numpy()

    feature_counts = sorted(
        set(min(n, len(candidate_features)) for n in feature_counts_to_try)
        | {len(candidate_features)})

    print(f"\n{'=' * 70}\n{modality.upper()}: {len(candidate_features)} "
          f"candidate features, {len(train_table)} train / "
          f"{len(test_table)} test windows, sweeping N = "
          f"{feature_counts}\n{'=' * 70}")

    ranking_model = fit_model(
        train_table[candidate_features].to_numpy(dtype=float), y_train)
    ranking = pd.Series(
        ranking_model.feature_importances_, index=candidate_features
    ).sort_values(ascending=False)

    rows = []
    for n in feature_counts:
        top_n = ranking.index[:n].tolist()
        model = fit_model(train_table[top_n].to_numpy(dtype=float), y_train)
        train_accuracy = float(
            model.score(train_table[top_n].to_numpy(dtype=float), y_train))
        test_accuracy = float(
            model.score(test_table[top_n].to_numpy(dtype=float), y_test))
        rows.append(dict(
            n_features=n, train_accuracy=train_accuracy,
            test_accuracy=test_accuracy, gap=train_accuracy - test_accuracy))
        print(f"  N={n:>4}: train={train_accuracy:.3f}  "
              f"test={test_accuracy:.3f}  gap={train_accuracy - test_accuracy:+.3f}")

    return pd.DataFrame(rows), ranking


def save_feature_ranking(modality: str, ranking: pd.Series) -> None:
    """The ranking check_modality() fit on the plain train split (not
    named feature_ranking_* -- that collides with find_optimal_feature_
    count.py's differently-sourced file of that name)."""
    table = ranking.reset_index()
    table.columns = ["feature", "importance"]
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    csv_path = (PROCESSED_DATA_DIRECTORY /
                f"overfitting_feature_ranking_{modality}_{dataset_name}"
                f"{_output_suffix}.csv")
    table.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    top = table.head(top_n_features_to_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top) + 1.5))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"Top {len(top)} features -- {modality} (train split only)")
    fig.tight_layout()
    png_path = (IMAGE_DATA_DIRECTORY /
                f"overfitting_feature_ranking_{modality}_{dataset_name}"
                f"{_output_suffix}.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {png_path}")


def plot_results(all_results: dict) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color, (modality, results) in zip(color_cycle, all_results.items()):
        ax.plot(results["n_features"], results["train_accuracy"], marker="o",
                linestyle="--", color=color, alpha=0.6,
                label=f"{modality} (train)")
        ax.plot(results["n_features"], results["test_accuracy"], marker="o",
                linestyle="-", color=color, label=f"{modality} (test)")
    ax.set_xlabel("Number of features (N)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Train vs. test accuracy vs. feature count "
                 f"(non-LOSO) -- {dataset_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = (IMAGE_DATA_DIRECTORY /
            f"overfitting_vs_feature_count_{dataset_name}{_output_suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote: {path}")


def report_io_paths() -> None:
    for modality in modalities_to_check:
        print(f"Input:  features_{modality}_{{train,test}}_"
              f"{dataset_name}.csv")
        print(f"Output: overfitting_vs_feature_count_{modality}_"
              f"{dataset_name}{_output_suffix}.csv")
        print(f"Output: overfitting_feature_ranking_{modality}_"
              f"{dataset_name}{_output_suffix}.{{csv,png}}")
    print(f"Output: overfitting_vs_feature_count_{dataset_name}"
          f"{_output_suffix}.png\n")


def main() -> None:
    start_time_s = time.perf_counter()
    report_io_paths()
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for modality in modalities_to_check:
        results, ranking = check_modality(modality)
        all_results[modality] = results
        results.to_csv(
            PROCESSED_DATA_DIRECTORY /
            f"overfitting_vs_feature_count_{modality}_{dataset_name}"
            f"{_output_suffix}.csv",
            index=False)
        save_feature_ranking(modality, ranking)

    plot_results(all_results)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
