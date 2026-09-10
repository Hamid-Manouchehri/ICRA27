"""train_simple_classifier.py -- Train one Random Forest on a chosen
modality's features, with manual feature include/exclude, either a
pooled train/test split or leave-one-subject-out (LOSO).

Inputs:
- features_<modality>_{train,test}_<dataset_name>.csv
- features_<extra_modality>_{train,test}_<dataset_name>.csv (optional,
  only if extra_modality is set)

Outputs:
- data/images/simple_confusion_matrix_<modality>_<dataset_name>
  [_loso][_<output_tag>].png
- data/images/simple_feature_importance_<modality>_<dataset_name>
  [_loso][_<output_tag>].png
- data/images/simple_accuracy_by_subject_<modality>_<dataset_name>
  [_loso][_<output_tag>].png (LOSO only)
- data/json/train_simple_classifier_<timestamp>.json (run manifest)
- printed classification report

Depends on:
- build_modality_feature_datasets.py (writes robot/gaze/joint CSVs)
- build_modality_feature_dataset_all.py (writes the "all" CSV)
"""

import json
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
SCRIPT_NAME = Path(__file__).stem
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"
JSON_DIRECTORY = DATA_DIRECTORY / "json"

# -------------------- User Settings --------------------
modality = "robot"  # TODO "robot" | "gaze" | "joint"
dataset_name = "chen_manual"  # TODO must match build_modality_feature_datasets.py
output_tag = "lead_win_step_0"  # TODO appended to output filenames

extra_modality = "None"  # TODO "robot" | "gaze" | "joint" | None
extra_features_to_include = [
    # curvature_when_gaze_far/close deliberately left out -- possible
    # curvature/geometry confound, see the "interpret this plot" discussion.
    # "mean_curvature_when_gaze_far",
    # "mean_curvature_when_gaze_close",
    # "gaze_to_ee_speed_ratio",
    # "gaze_ee_x_correlation",
    # "mean_gaze_lead_along_ee_motion",
    # "mean_velocity_cosine_similarity",
    # "mean_lookahead_distance",
    # "max_x_cross_correlation",
]

list_available_features_and_exit = False  # TODO True to just print feature names

use_loso = True  # TODO True for leave-one-subject-out instead of the pooled split
train_fraction_of_smallest_class = 0.5  # TODO LOSO only: balanced-draw fraction

lead_window_steps = 0  # TODO LOSO only: predict the label N windows ahead (window_step_s second ahead); 0 = today's behavior
window_step_s = 1  # TODO must match build_modality_feature_datasets.py; only for printing lead time in seconds

# TODO dropped from "use everything"; ignored if features_to_include is set.
features_to_exclude = [
    # positional (correlates with shape geometry, not behaviour)
    # "mean_position_x", "mean_position_y", "std_position_x", "std_position_y",
    # "variance_position_x", "variance_position_y", "minimum_position_x",
    # "maximum_position_x", "range_position_x", "minimum_position_y",
    # "maximum_position_y", "range_position_y", "median_position_x",
    # "median_position_y", "iqr_position_x", "iqr_position_y",
    # "mad_position_x", "mad_position_y", "skewness_position_x",
    # "skewness_position_y", "kurtosis_position_x", "kurtosis_position_y",
    # "covariance_position_xy", "correlation_position_xy",
    # "position_dispersion_rms", "position_bcea_68",
    # "position_x_shannon_entropy", "position_y_shannon_entropy",
    # "position_x_mean_crossing_count", "position_y_mean_crossing_count",
    # "position_x_lag1_autocorrelation", "position_y_lag1_autocorrelation",
    # # force (DAMPING-CONFOUNDED -- see classifier_feature_bank.py docstring)
    # "mean_force_magnitude", "std_force_magnitude", "max_force_magnitude",
    # "median_force_magnitude", "rms_force_magnitude", "mean_force_x",
    # "mean_force_y", "std_force_x", "std_force_y",
]

# TODO if non-empty, ONLY these are used (overrides features_to_exclude).
features_to_include = [
    # "mean_ee_speed_m_s",
    # "power_law_exponent",
    
]

# TODO see sklearn.ensemble.RandomForestClassifier for more parameters.
n_estimators = 300
random_seed = 42

top_n_features_to_plot = 100  # TODO for the importance bar chart
# -------------------- End User Settings --------------------

read_train_path = (PROCESSED_DATA_DIRECTORY /
                    f"features_{modality}_train_{dataset_name}.csv")
read_test_path = (PROCESSED_DATA_DIRECTORY /
                   f"features_{modality}_test_{dataset_name}.csv")

_tag_parts = (["loso"] if use_loso else []) + ([output_tag] if output_tag else [])
_output_suffix = ("_" + "_".join(_tag_parts)) if _tag_parts else ""
write_confusion_matrix_path = (
    IMAGE_DATA_DIRECTORY /
    f"simple_confusion_matrix_{modality}_{dataset_name}{_output_suffix}.png")
write_feature_importance_path = (
    IMAGE_DATA_DIRECTORY /
    f"simple_feature_importance_{modality}_{dataset_name}{_output_suffix}.png")
write_accuracy_by_subject_path = (
    IMAGE_DATA_DIRECTORY /
    f"simple_accuracy_by_subject_{modality}_{dataset_name}{_output_suffix}.png")

# Non-feature columns present in every features_*.csv.
metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count", "_label_before_shift",
]
target_column = "trial_class_label"


def load_extra_features(base_table: pd.DataFrame, split: str) -> pd.DataFrame:
    """Merge extra_features_to_include from extra_modality's CSV onto
    base_table, matched by trial+window identity. A window with no
    match is a real bug (raises); a matched window whose feature value
    is itself NaN (e.g. a curvature-conditioned-on-gaze-distance
    feature that couldn't split for that window) is just dropped."""
    if not extra_modality or not extra_features_to_include:
        return base_table
    extra_table = pd.read_csv(
        PROCESSED_DATA_DIRECTORY /
        f"features_{extra_modality}_{split}_{dataset_name}.csv")
    key_columns = [c for c in
                   ("subject", "shape", "damping_n_s_m", "bandwidth",
                    "direction", "window_index")
                   if c in base_table.columns and c in extra_table.columns]
    merged = base_table.merge(
        extra_table[key_columns + extra_features_to_include],
        on=key_columns, how="left", validate="one_to_one", indicator=True)
    unmatched = int((merged["_merge"] == "left_only").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} {split} windows had no matching {extra_modality} "
            f"row -- rebuild both modalities with the same dataset_name/"
            f"settings.")
    merged = merged.drop(columns="_merge")

    na_count = int(merged[extra_features_to_include].isna().any(axis=1).sum())
    if na_count:
        print(f"  {extra_modality}->{modality}: dropping {na_count} of "
              f"{len(merged)} {split} windows (extra feature was NaN for "
              f"that window, not a match failure).")
        merged = merged.dropna(
            subset=extra_features_to_include).reset_index(drop=True)
    return merged


def available_feature_columns(table: pd.DataFrame) -> list:
    """Every CSV column that isn't metadata or the label."""
    return [c for c in table.columns
            if c not in metadata_columns and c != target_column]


def select_feature_columns(all_features: list) -> list:
    """Apply features_to_include / features_to_exclude, in that order."""
    if features_to_include:
        unknown = [f for f in features_to_include if f not in all_features]
        if unknown:
            raise ValueError(
                f"features_to_include has unknown names: {unknown}")
        return list(features_to_include)
    unknown = [f for f in features_to_exclude if f not in all_features]
    if unknown:
        raise ValueError(f"features_to_exclude has unknown names: {unknown}")
    return [f for f in all_features if f not in features_to_exclude]


def save_run_manifest(manifest: dict) -> None:
    JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = JSON_DIRECTORY / f"{SCRIPT_NAME}_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Output: {path}")


def fit_model(X, y) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(X, y)
    return model


def shift_labels_forward(table: pd.DataFrame, window_offset: int) -> pd.DataFrame:
    """Re-pair each window's features with the label window_offset steps
    LATER in the same trial (features unchanged, target_column moves
    forward). Keeps the pre-shift label as _label_before_shift, for a
    persistence-baseline check in run_loso. Windows with no future
    partner (trial end, dropped by transition margin/edge trim) are
    removed. LOSO-only -- see module docstring."""
    if window_offset == 0:
        return table
    trial_columns = [c for c in
                      ("subject", "shape", "damping_n_s_m", "bandwidth", "direction")
                      if c in table.columns]
    future = table[trial_columns + ["window_index", target_column]].copy()
    future["window_index"] -= window_offset
    future = future.rename(columns={target_column: "_future_label"})
    shifted = table.rename(columns={target_column: "_label_before_shift"}).merge(
        future, on=trial_columns + ["window_index"],
        how="inner", validate="one_to_one")
    return shifted.rename(columns={"_future_label": target_column})


def draw_balanced_train(table: pd.DataFrame) -> pd.DataFrame:
    """Equal-count draw per class from table, for LOSO fold training."""
    class_counts = table[target_column].value_counts()
    per_class = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)), 1)
    parts = [rows.sample(n=min(per_class, len(rows)), random_state=random_seed)
             for _, rows in table.groupby(target_column)]
    return pd.concat(parts, ignore_index=True)


def plot_confusion_matrix(y_true, y_predicted, class_order, title, path) -> None:
    report_dict = classification_report(
        y_true, y_predicted, labels=class_order, output_dict=True)
    macro = report_dict["macro avg"]
    metrics_text = (
        f"Accuracy: {report_dict['accuracy']:.3f}    "
        f"Precision: {macro['precision']:.3f}    "
        f"Recall: {macro['recall']:.3f}    "
        f"F1-score: {macro['f1-score']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(title)
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_columns, title, path) -> None:
    ranking = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)
    top_ranking = ranking.head(top_n_features_to_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_ranking) + 1.5))
    ax.barh(top_ranking.index, top_ranking.values)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_pooled_split(
        train_table: pd.DataFrame, test_table: pd.DataFrame,
        feature_columns: list) -> None:
    X_train = train_table[feature_columns].to_numpy(dtype=float)
    y_train = train_table[target_column].to_numpy()
    X_test = test_table[feature_columns].to_numpy(dtype=float)
    y_test = test_table[target_column].to_numpy()
    class_order = sorted(np.unique(np.concatenate([y_train, y_test])))

    print(f"Train: {len(y_train)} windows. Test: {len(y_test)} windows. "
          f"Classes: {class_order}")

    model = fit_model(X_train, y_train)
    y_predicted = model.predict(X_test)

    print(f"\n=== test-set performance ({modality}, {dataset_name}) ===")
    print(classification_report(y_test, y_predicted, labels=class_order))

    plot_confusion_matrix(
        y_test, y_predicted, class_order,
        f"Confusion matrix -- {modality} ({dataset_name})",
        write_confusion_matrix_path)
    plot_feature_importance(
        model, feature_columns, f"Top {top_n_features_to_plot} features -- "
        f"{modality}", write_feature_importance_path)


def run_loso(pooled_table: pd.DataFrame, feature_columns: list) -> None:
    class_order = sorted(pooled_table[target_column].unique())
    subjects = sorted(int(s) for s in pooled_table["subject"].unique())
    print(f"LOSO over {len(subjects)} subjects: {subjects}")

    if "_label_before_shift" in pooled_table.columns:
        persistence_accuracy = float((
            pooled_table["_label_before_shift"] ==
            pooled_table[target_column]).mean())
        print(f"  Persistence baseline (assume the label is unchanged "
              f"since the previous window): {persistence_accuracy:.3f} "
              f"-- beat this, not just the majority class, to show real "
              f"anticipation.")

    all_true, all_predicted, fold_accuracies = [], [], []
    for held_out in subjects:
        test_table = pooled_table[pooled_table["subject"] == held_out]
        train_table = draw_balanced_train(
            pooled_table[pooled_table["subject"] != held_out])

        model = fit_model(
            train_table[feature_columns].to_numpy(dtype=float),
            train_table[target_column].to_numpy())
        y_test = test_table[target_column].to_numpy()
        y_predicted = model.predict(test_table[feature_columns].to_numpy(dtype=float))

        fold_accuracies.append(float(np.mean(y_predicted == y_test)))
        all_true.append(y_test)
        all_predicted.append(y_predicted)
        print(f"  held out s{held_out}: accuracy={fold_accuracies[-1]:.3f}, "
              f"train={len(train_table)}, test={len(test_table)}")

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)
    pooled_accuracy = float(np.mean(all_true == all_predicted))

    print(f"\n=== LOSO pooled test-set performance ({modality}, "
          f"{dataset_name}) ===")
    print(classification_report(all_true, all_predicted, labels=class_order))

    plot_confusion_matrix(
        all_true, all_predicted, class_order,
        f"LOSO confusion matrix -- {modality} ({dataset_name})",
        write_confusion_matrix_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([f"s{s}" for s in subjects], fold_accuracies)
    ax.axhline(pooled_accuracy, color="red", linestyle="--",
               label=f"pooled overall = {pooled_accuracy:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"LOSO accuracy by held-out subject -- {modality}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(write_accuracy_by_subject_path, dpi=150)
    plt.close(fig)

    balanced_full = draw_balanced_train(pooled_table)
    ranking_model = fit_model(
        balanced_full[feature_columns].to_numpy(dtype=float),
        balanced_full[target_column].to_numpy())
    plot_feature_importance(
        ranking_model, feature_columns,
        f"Top {top_n_features_to_plot} features -- {modality}",
        write_feature_importance_path)


def main() -> None:
    start_time_s = time.perf_counter()

    print(f"Input:  {read_train_path}")
    print(f"Input:  {read_test_path}")
    if extra_modality and extra_features_to_include:
        print(f"Input:  features_{extra_modality}_{{train,test}}_"
              f"{dataset_name}.csv (extra columns: "
              f"{', '.join(extra_features_to_include)})")
    print(f"Output: {write_confusion_matrix_path}")
    print(f"Output: {write_feature_importance_path}")
    if use_loso:
        print(f"Output: {write_accuracy_by_subject_path}")
    print()

    if lead_window_steps and not use_loso:
        raise ValueError(
            "lead_window_steps requires use_loso = True -- the pooled "
            "window split can put a window and its future label on "
            "opposite sides of train/test.")

    train_table = load_extra_features(pd.read_csv(read_train_path), "train")
    test_table = load_extra_features(pd.read_csv(read_test_path), "test")
    all_features = available_feature_columns(train_table)

    if list_available_features_and_exit:
        print(f"{len(all_features)} available features for "
              f"modality='{modality}', dataset_name='{dataset_name}':\n")
        for name in sorted(all_features):
            print(f'    "{name}",')
        return

    feature_columns = select_feature_columns(all_features)
    print(f"Using {len(feature_columns)} of {len(all_features)} available "
          f"features ({len(all_features) - len(feature_columns)} excluded).")

    save_run_manifest({
        "script": SCRIPT_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset_name,
        "modality": modality,
        "train_file": read_train_path.name,
        "test_file": read_test_path.name,
        "extra_modality": extra_modality if extra_features_to_include else None,
        "extra_features": list(extra_features_to_include),
        "features": feature_columns,
        "n_features": len(feature_columns),
        "classifier": "RandomForest",
        "n_estimators": n_estimators,
        "use_loso": use_loso,
        "lead_window_steps": lead_window_steps if use_loso else None,
        "window_step_s": window_step_s,
        "random_seed": random_seed,
        "output_tag": output_tag or None,
    })

    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if use_loso:
        pooled_table = pd.concat([train_table, test_table], ignore_index=True)
        if lead_window_steps:
            before = len(pooled_table)
            pooled_table = shift_labels_forward(pooled_table, lead_window_steps)
            print(f"Lead time: {lead_window_steps} window(s) = "
                  f"{lead_window_steps * window_step_s:.2f} s ahead. "
                  f"{len(pooled_table)} of {before} windows kept "
                  f"(dropped: no future window in the same trial).")
        run_loso(pooled_table, feature_columns)
    else:
        run_pooled_split(train_table, test_table, feature_columns)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
