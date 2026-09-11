"""losho_classifier.py -- Leave-one-shape-out (LOSHO): hold out one
shape at a time, train (balanced) on the rest, test on it. Tests
generalization to a new task geometry, not a new person (compare
train_simple_classifier.py's LOSO). Only 3 shapes exist in this
dataset and their class balance differs sharply (20-80% range), so
per-shape accuracy + per-shape majority baseline are reported, not
just a pooled number.

Inputs:
- features_<modality>_{train,test}_<dataset_name>.csv

Outputs:
- data/images/losho_confusion_matrix_<modality>_<dataset_name>
  [_<output_tag>].png
- data/images/losho_accuracy_by_shape_<modality>_<dataset_name>
  [_<output_tag>].png
- data/processed_data/losho_fold_results_<modality>_<dataset_name>
  [_<output_tag>].csv

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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
modality = "robot"  # TODO "robot" | "gaze" | "joint" | "all"
dataset_name = "W_2s"  # TODO must match build_modality_feature_datasets.py
output_tag = ""  # TODO appended to output filenames

exclude_positional_features = True  # TODO see POSITIONAL_FEATURE_NAMES below
exclude_force_features = True  # TODO see FORCE_FEATURE_NAMES below

train_fraction_of_smallest_class = 0.5  # TODO balanced-draw fraction per fold
n_estimators = 300
random_seed = 42

top_n_features_to_plot = 20  # TODO for the feature-importance bar chart
# -------------------- End User Settings --------------------

read_train_path = (PROCESSED_DATA_DIRECTORY /
                    f"features_{modality}_train_{dataset_name}.csv")
read_test_path = (PROCESSED_DATA_DIRECTORY /
                   f"features_{modality}_test_{dataset_name}.csv")

_output_suffix = f"_{output_tag}" if output_tag else ""
write_confusion_matrix_path = (
    IMAGE_DATA_DIRECTORY /
    f"losho_confusion_matrix_{modality}_{dataset_name}{_output_suffix}.png")
write_accuracy_by_shape_path = (
    IMAGE_DATA_DIRECTORY /
    f"losho_accuracy_by_shape_{modality}_{dataset_name}{_output_suffix}.png")
write_feature_importance_path = (
    IMAGE_DATA_DIRECTORY /
    f"losho_feature_importance_{modality}_{dataset_name}{_output_suffix}.png")
write_fold_results_path = (
    PROCESSED_DATA_DIRECTORY /
    f"losho_fold_results_{modality}_{dataset_name}{_output_suffix}.csv")

metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]
target_column = "trial_class_label"

# Same confound lists as find_optimal_feature_count.py / check_overfitting_
# vs_feature_count.py -- kept in sync there, duplicated here (standalone).
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


def candidate_feature_columns(table: pd.DataFrame) -> list:
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


def draw_balanced_train(table: pd.DataFrame) -> pd.DataFrame:
    class_counts = table[target_column].value_counts()
    per_class = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)), 1)
    parts = [rows.sample(n=min(per_class, len(rows)), random_state=random_seed)
             for _, rows in table.groupby(target_column)]
    return pd.concat(parts, ignore_index=True)


def run_losho(pooled_table: pd.DataFrame, feature_columns: list) -> None:
    class_order = sorted(pooled_table[target_column].unique())
    shapes = sorted(pooled_table["shape"].unique())
    print(f"LOSHO over {len(shapes)} shapes: {shapes}")

    fold_rows = []
    all_true, all_predicted = [], []
    for held_out in shapes:
        test_table = pooled_table[pooled_table["shape"] == held_out]
        train_table = draw_balanced_train(
            pooled_table[pooled_table["shape"] != held_out])

        model = fit_model(
            train_table[feature_columns].to_numpy(dtype=float),
            train_table[target_column].to_numpy())
        y_test = test_table[target_column].to_numpy()
        y_predicted = model.predict(test_table[feature_columns].to_numpy(dtype=float))

        accuracy = float(np.mean(y_predicted == y_test))
        majority_baseline = float(
            test_table[target_column].value_counts(normalize=True).max())
        fold_rows.append(dict(
            shape=held_out, accuracy=accuracy, majority_baseline=majority_baseline,
            lift=accuracy - majority_baseline,
            train_windows=len(train_table), test_windows=len(test_table)))
        all_true.append(y_test)
        all_predicted.append(y_predicted)
        print(f"  held out '{held_out}': accuracy={accuracy:.3f}  "
              f"majority_baseline={majority_baseline:.3f}  "
              f"lift={accuracy - majority_baseline:+.3f}  "
              f"train={len(train_table)}  test={len(test_table)}")

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(write_fold_results_path, index=False)
    print(f"Wrote: {write_fold_results_path}")

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)
    pooled_accuracy = float(np.mean(all_true == all_predicted))

    print(f"\n=== LOSHO pooled test-set performance ({modality}, "
          f"{dataset_name}) ===")
    print(classification_report(all_true, all_predicted, labels=class_order))

    report_dict = classification_report(
        all_true, all_predicted, labels=class_order, output_dict=True)
    macro = report_dict["macro avg"]
    metrics_text = (
        f"Accuracy: {report_dict['accuracy']:.3f}    "
        f"Precision: {macro['precision']:.3f}    "
        f"Recall: {macro['recall']:.3f}    "
        f"F1-score: {macro['f1-score']:.3f}")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        all_true, all_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(f"LOSHO confusion matrix -- {modality} ({dataset_name})")
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(write_confusion_matrix_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {write_confusion_matrix_path}")

    # Per-shape accuracy vs. that shape's OWN majority baseline -- a
    # single pooled baseline would be misleading given how much class
    # balance differs by shape (see module docstring).
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(fold_table))
    ax.bar(x, fold_table["accuracy"], label="accuracy")
    ax.scatter(x, fold_table["majority_baseline"], color="red", zorder=3,
               label="majority baseline (this shape)")
    ax.axhline(pooled_accuracy, color="black", linestyle="--",
               label=f"pooled overall = {pooled_accuracy:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(fold_table["shape"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"LOSHO accuracy by held-out shape -- {modality}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(write_accuracy_by_shape_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {write_accuracy_by_shape_path}")

    balanced_full = draw_balanced_train(pooled_table)
    ranking_model = fit_model(
        balanced_full[feature_columns].to_numpy(dtype=float),
        balanced_full[target_column].to_numpy())
    ranking = pd.Series(
        ranking_model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)
    top_ranking = ranking.head(top_n_features_to_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_ranking) + 1.5))
    ax.barh(top_ranking.index, top_ranking.values)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"Top {len(top_ranking)} features -- {modality}")
    fig.tight_layout()
    fig.savefig(write_feature_importance_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {write_feature_importance_path}")


def main() -> None:
    start_time_s = time.perf_counter()

    print(f"Input:  {read_train_path}")
    print(f"Input:  {read_test_path}")
    print(f"Output: {write_confusion_matrix_path}")
    print(f"Output: {write_accuracy_by_shape_path}")
    print(f"Output: {write_feature_importance_path}")
    print(f"Output: {write_fold_results_path}\n")

    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    pooled_table = pd.concat(
        [pd.read_csv(read_train_path), pd.read_csv(read_test_path)],
        ignore_index=True)
    feature_columns = candidate_feature_columns(pooled_table)
    print(f"Using {len(feature_columns)} candidate features.")

    run_losho(pooled_table, feature_columns)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
