"""find_optimal_feature_count.py -- For each modality (robot/gaze/joint),
sweep the number of top-ranked features N and find the N that gives the
best LOSO accuracy -- instead of assuming top-20 (or "all") is right.

Per fold (held-out subject): rank the OTHER subjects' features by
importance once, then re-fit and test using just the top N, for every N
in feature_counts_to_try. Ranking never sees the held-out subject, so
larger N can't leak subject info the way global ranking would.
Predictions are pooled across folds per N, same as train_modality_
classifiers_loso.py.

Reads features_<modality>_{train,test}_<dataset_name>.csv (written by
build_modality_feature_datasets.py). Standalone -- not imported by, and
can't break, the other training/comparison scripts.

"joint" here means robot's own features PLUS joint_extra_features
merged in (same technique as train_simple_classifier.py's
extra_modality/extra_features_to_include) -- not the joint bank's full
~45 features alone.

Output: a printed table + best-N summary per modality, plus
data/images/optimal_feature_count_<modality>_<dataset_name>.png and
data/processed_data/optimal_feature_count_<modality>_<dataset_name>.csv

Dependencies: numpy, pandas, scikit-learn, matplotlib
  pip install numpy pandas scikit-learn matplotlib
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
dataset_name = "chen_manual"  # TODO must match build_modality_feature_datasets.py
modalities_to_sweep = ("robot", "gaze", "joint")  # TODO subset of these

# TODO "joint" = robot's own features + these specific joint-bank
# columns merged in, instead of the joint bank's ~45 features alone.
joint_extra_features = [
    "mean_curvature_when_gaze_far",
    "mean_curvature_when_gaze_close",
    "gaze_to_ee_speed_ratio",
]

# TODO candidate N values per modality; clipped to what's available, and
# "all features" is always added as a reference point.
feature_counts_to_try = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110]

# TODO drop known-confounded features from the candidate pool (see
# CONFOUNDED_FEATURE_NAMES below) before ranking/sweeping.
exclude_confounded_features = True

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
# plot_top20_modality_comparison.py's positional_feature_names).
CONFOUNDED_FEATURE_NAMES = {
    "robot": {
        # positional -- correlates with shape geometry, not behaviour
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
        # force -- DAMPING-CONFOUNDED (classifier_feature_bank.py docstring)
        "mean_force_magnitude", "std_force_magnitude", "max_force_magnitude",
        "median_force_magnitude", "rms_force_magnitude", "mean_force_x",
        "mean_force_y", "std_force_x", "std_force_y",
    },
    "gaze": set(),
    # Unused now that "joint" means robot + joint_extra_features (see
    # module docstring) -- kept for reference / reverting to the full
    # joint bank.
    "joint": {
        # force-derived -- DAMPING-CONFOUNDED
        "angle_gaze_offset_to_force", "correlation_gaze_ee_distance_force",
        "correlation_gaze_speed_force", "mean_gaze_ee_distance_high_force",
        "mean_gaze_ee_distance_low_force", "fraction_gaze_close_during_high_force",
    },
}


def merge_extra_features(base_table: pd.DataFrame, split: str,
                          extra_modality: str, extra_features: list) -> pd.DataFrame:
    """Merge specific named columns from another modality's CSV onto
    base_table, matched by trial+window identity -- same technique as
    train_simple_classifier.py's load_extra_features."""
    extra_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY /
        f"features_{extra_modality}_{split}_{dataset_name}.csv")
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
            f"{unmatched} {split} windows had no matching {extra_modality} "
            f"row -- rebuild both modalities with the same dataset_name/"
            f"settings.")
    merged = merged.drop(columns="_merge")

    na_count = int(merged[extra_features].isna().any(axis=1).sum())
    if na_count:
        print(f"  {extra_modality}->joint: dropping {na_count} of "
              f"{len(merged)} {split} windows (extra feature NaN for "
              f"that window).")
        merged = merged.dropna(subset=extra_features).reset_index(drop=True)
    return merged


def read_pooled_table(modality: str) -> pd.DataFrame:
    base_modality = "robot" if modality == "joint" else modality
    train_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY /
        f"features_{base_modality}_train_{dataset_name}.csv")
    test_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY /
        f"features_{base_modality}_test_{dataset_name}.csv")
    if modality == "joint" and joint_extra_features:
        train_table = merge_extra_features(
            train_table, "train", "joint", joint_extra_features)
        test_table = merge_extra_features(
            test_table, "test", "joint", joint_extra_features)
    return pd.concat([train_table, test_table], ignore_index=True)


def candidate_feature_columns(pooled_table: pd.DataFrame, modality: str) -> list:
    all_features = [c for c in pooled_table.columns
                     if c not in metadata_columns and c != target_column]
    if not exclude_confounded_features:
        return all_features
    # "joint" is now robot's own features + joint_extra_features (see
    # module docstring) -- filter using ROBOT's confound set, and never
    # drop the deliberately-added extra joint features.
    confound_key = "robot" if modality == "joint" else modality
    confounded = CONFOUNDED_FEATURE_NAMES[confound_key] - set(joint_extra_features)
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


def sweep_modality(modality: str) -> pd.DataFrame:
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

    return pd.DataFrame(rows)


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
    ax.set_title(f"Accuracy vs. feature count -- {dataset_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = IMAGE_DATA_DIRECTORY / f"optimal_feature_count_{dataset_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nWrote: {path}")


def main() -> None:
    start_time_s = time.perf_counter()
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []
    for modality in modalities_to_sweep:
        results = sweep_modality(modality)
        all_results[modality] = results
        results.to_csv(
            PROCESSED_DATA_DIRECTORY /
            f"optimal_feature_count_{modality}_{dataset_name}.csv", index=False)

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
        PROCESSED_DATA_DIRECTORY / f"optimal_feature_count_summary_"
        f"{dataset_name}.csv", index=False)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
