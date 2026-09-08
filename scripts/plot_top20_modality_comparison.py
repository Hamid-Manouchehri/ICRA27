"""plot_top20_modality_comparison.py -- Two figures comparing robot-only,
gaze-only (strict, Tobii-derived), and joint features, each restricted
to its own top 20 LOSO-ranked features: one figure with confusion
matrix + classifier metrics per case, a SEPARATE figure with the
feature-importance ranking per case (same column order in both, so
column i is the same case in both figures).

Reuses compare_loso_feature_sets.py's feature-set construction and LOSO
runner directly (import, not duplication), so these numbers are the
SAME evaluation methodology already used there -- per-fold top-N
selection from that fold's training data only (no leakage of the
held-out subject into feature selection), majority-class AND
persistence baselines available in the underlying fold data.

"gaze-only" here means gaze_pure (73 features derived from tobii_*.csv
ALONE) per this project's own definition, NOT compare_loso_feature_
sets.py's "gaze" bank (102 features, 29 of which also read the robot
CSV) -- see that script's docstring for how the 29 were identified.

The feature-importance panel is DIFFERENT from the confusion-matrix
panel's top-20 selection: LOSO selects top-20 fresh inside each of the
9 folds (so no single canonical top-20 list exists), while the ranking
panel fits ONE model on a balanced draw across ALL subjects, purely to
show which features the model leans on -- descriptive, not part of the
generalization evaluation above it.

Inputs:  data/processed_data/features_{robot,gaze,joint}_{train,test}_
         <dataset_name>.csv, written by build_modality_feature_
         datasets.py.
Output:  data/images/top20_confusion_matrix_<dataset_name>.png
         data/images/top20_feature_ranking_<dataset_name>.png

Dependencies: numpy, pandas, scikit-learn, matplotlib
  pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import compare_loso_feature_sets as clfs
import train_modality_classifiers as tmc

DATA_DIRECTORY = clfs.DATA_DIRECTORY
IMAGE_DATA_DIRECTORY = clfs.IMAGE_DATA_DIRECTORY

# -------------------- User Settings --------------------
# TODO the three sets to compare, as (display_label, feature_set_key)
# -- feature_set_key must be one of build_feature_sets()'s keys.
# panels = (
#     ("Robot-only", "robot"),
#     ("Gaze-only", "gaze_pure"),
#     ("Joint", "joint"),
# )

# TODO robot features to drop before ranking/evaluating "robot_no_
# position" -- raw X/Y location, which correlates with curvature by
# construction (shape geometry is fixed per bandwidth/shape), so
# leaving these in lets the classifier learn "where on the shape am I"
# instead of "how is the person behaving."
positional_feature_names = {
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
}

# TODO the three sets to compare, as (display_label, feature_set_key)
# -- feature_set_key must be one of build_feature_sets()'s keys, or
# "robot_no_position" (added in main() below).
panels = (
    ("Robot (no position)", "robot_no_position"),
    ("Gaze-only", "gaze_pure"),
    ("Joint", "joint"),
)

# for i=1:
#     feature_selection = "commanded_actual_speed_correlation",
#     feature_1 

top_n_features = 20
random_seed = clfs.random_seed


def fit_full_importance_ranking(
        table: pd.DataFrame, feature_columns: list) -> pd.Series:
    """One model on a balanced draw across ALL subjects -- descriptive
    ranking only, not an evaluation (see module docstring)."""
    train_table = clfs.draw_balanced_train(table, random_seed)
    model = tmc.build_classifier(clfs.classifier_name, **clfs.classifier_kwargs)
    model.fit(train_table[feature_columns], train_table[clfs.target_column])
    return pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)


def metrics_text(summary_row: dict, class_order: list) -> str:
    lines = [f"Accuracy: {summary_row['accuracy']:.3f}"]
    for class_name in class_order:
        lines.append(
            f"{class_name}: P={summary_row[f'{class_name}_precision']:.2f}  "
            f"R={summary_row[f'{class_name}_recall']:.2f}  "
            f"F1={summary_row[f'{class_name}_f1']:.2f}")
    lines.append(f"Macro F1: {summary_row['macro_f1']:.3f}")
    lines.append(
        f"Baseline: {summary_row['mean_majority_baseline']:.3f}  "
        f"(lift {summary_row['mean_lift']:+.3f})")
    return "\n".join(lines)


def gather_results(feature_sets: dict) -> list:
    """Run LOSO + the descriptive ranking fit once per case, so both
    figures are built from the same numbers without recomputing."""
    results = []
    for display_label, set_key in panels:
        table, feature_columns = feature_sets[set_key]

        summary_row, _, pooled_true, pooled_predicted = clfs.run_loso(
            f"{set_key}_top{top_n_features}", table, feature_columns,
            keep_top_n=top_n_features)
        print(f"{display_label:<12} LOSO top{top_n_features}: "
              f"acc={summary_row['accuracy']:.3f}  "
              f"macro_f1={summary_row['macro_f1']:.3f}")

        ranking = fit_full_importance_ranking(table, feature_columns)
        results.append(dict(
            display_label=display_label, summary_row=summary_row,
            pooled_true=pooled_true, pooled_predicted=pooled_predicted,
            ranking=ranking))
    return results


def build_confusion_matrix_figure(results: list, output_path: Path) -> None:
    fig, axes_row = plt.subplots(
        1, len(results), figsize=(5 * len(results), 6.5))
    for axes, result in zip(axes_row, results):
        summary_row = result["summary_row"]
        class_order = summary_row["class_order"]
        matrix = confusion_matrix(
            result["pooled_true"], result["pooled_predicted"],
            labels=class_order)
        ConfusionMatrixDisplay(matrix, display_labels=class_order).plot(
            ax=axes, cmap="Blues", colorbar=False, values_format="d")
        axes.set_title(
            f"{result['display_label']}  "
            f"(top {summary_row['feature_count']} features)", fontsize=11)
        axes.text(
            0.5, -0.22, metrics_text(summary_row, class_order),
            ha="center", va="top", fontsize=9, family="monospace",
            transform=axes.transAxes)

    fig.suptitle(
        f"Top-{top_n_features} feature comparison, LOSO -- {clfs.dataset_name}",
        fontsize=13)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def build_feature_ranking_figure(results: list, output_path: Path) -> None:
    fig, axes_row = plt.subplots(
        1, len(results), figsize=(5 * len(results), 6))
    for axes, result in zip(axes_row, results):
        top_ranking = result["ranking"].head(top_n_features).iloc[::-1]
        axes.barh(
            np.arange(len(top_ranking)), top_ranking.values, color="#4C72B0")
        axes.set_yticks(np.arange(len(top_ranking)))
        axes.set_yticklabels(top_ranking.index, fontsize=7)
        axes.set_xlabel("Importance", fontsize=8)
        axes.set_title(result["display_label"], fontsize=11)
        axes.tick_params(axis="x", labelsize=7)

    fig.suptitle(
        f"Top-{top_n_features} feature ranking (all subjects) -- "
        f"{clfs.dataset_name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def main() -> None:
    # feature_sets = clfs.build_feature_sets()

    feature_sets = clfs.build_feature_sets()
    robot_table, robot_features = feature_sets["robot"]
    filtered_features = [f for f in robot_features if f not in positional_feature_names]
    feature_sets["robot_no_position"] = (robot_table, filtered_features)
    print(f"robot_no_position: {len(robot_features)} -> {len(filtered_features)} features "
        f"({len(robot_features) - len(filtered_features)} positional dropped)")
    
    results = gather_results(feature_sets)

    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    build_confusion_matrix_figure(
        results,
        IMAGE_DATA_DIRECTORY / f"top20_confusion_matrix_{clfs.dataset_name}_remove_positional_features.png")
    build_feature_ranking_figure(
        results,
        IMAGE_DATA_DIRECTORY / f"top20_feature_ranking_{clfs.dataset_name}_remove_positional_features.png")


if __name__ == "__main__":
    main()
