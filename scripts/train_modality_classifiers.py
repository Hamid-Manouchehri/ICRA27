"""train_modality_classifiers.py -- Train and evaluate one classifier
PER MODALITY (robot-only, gaze-only, joint) on the datasets
build_modality_feature_datasets.py wrote, and report accuracy,
precision, recall, F1 (per-class, macro, and weighted), and a
confusion matrix for EACH modality separately.

Each modality gets its own trained model (not a shared one evaluated
on different feature subsets) -- deliberately matching how
build_modality_feature_datasets.py already built each modality's train
and test windows from the SAME underlying windows and the SAME
train/test split by window identity, so the only thing that differs
between the three runs below is which features the model saw, not
which windows.

Output, per modality (file names carry the modality, e.g.
confusion_matrix_robot.png):
  - confusion_matrix_<modality>.png
  - feature_importance_<modality>.png / feature_importance_ranking_
    <modality>.csv (a random forest's importances specifically, same
    as KD_test/python/train_classifier.py, computed alongside the
    primary metrics as a low-cost bonus -- not the main ask here, but
    cheap given a random forest is already being fit).
  - predictions_<modality>.csv (y_test, y_predicted -- read by
    compare_modality_classifiers.py so it can build a combined view
    without retraining).
Combined across all three modalities:
  - metrics_by_modality.csv (accuracy + per-class/macro/weighted
    precision/recall/F1, one row per modality) -- also read by
    compare_modality_classifiers.py.
  - a classification report printed to the command window per modality.

The classifier is swappable (CLASSIFIER_REGISTRY / classifier_name),
same mechanism as KD_test/python/train_classifier.py -- one setting
change applies to all three modality runs at once.

Dependencies: numpy, pandas, scikit-learn, matplotlib
  pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# -------------------- User Settings --------------------
# TODO input feature CSVs per modality, as written by
# build_modality_feature_datasets.py.
modality_csv_paths = {
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

# TODO metadata columns to exclude from every modality's feature matrix
# (kept only for reference). subject/shape/damping_n_s_m describe the
# recording CONDITION, not behaviour, so are excluded on purpose even
# though damping_n_s_m is numeric.
metadata_columns = [
    "subject", "shape", "damping_n_s_m",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]
target_column = "trial_class_label"

# TODO the classifier to train/evaluate for every modality -- change
# this one line to switch models, or add another entry to
# CLASSIFIER_REGISTRY below for anything not already listed.
classifier_name = "random_forest"
classifier_kwargs = {"n_estimators": 300, "random_state": 42}

# TODO how many top-ranked features to show in each modality's
# importance bar chart.
top_n_features_to_plot = 20

random_seed = 42


# ======================================================================
# Swappable classifier registry.
# ======================================================================

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


# ======================================================================
# Data loading
# ======================================================================

def load_split(csv_path: Path):
    table = pd.read_csv(csv_path)
    feature_columns = [
        c for c in table.columns
        if c not in metadata_columns and c != target_column]
    return table, feature_columns


# ======================================================================
# Train + evaluate one modality
# ======================================================================

def flatten_classification_report(
        modality: str, report: dict, class_order: list) -> dict:
    """One flat dict per modality: accuracy, then precision/recall/f1
    for each class in class_order, then macro and weighted averages --
    the row shape metrics_by_modality.csv ends up with."""
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


def train_and_evaluate_modality(modality: str) -> dict:
    train_table, feature_columns = load_split(
        modality_csv_paths[modality]["train"])
    test_table, test_feature_columns = load_split(
        modality_csv_paths[modality]["test"])
    assert feature_columns == test_feature_columns, (
        f"{modality}: train/test feature CSVs have different columns -- "
        f"were they written by the same run of "
        f"build_modality_feature_datasets.py?")

    X_train = train_table[feature_columns].to_numpy(dtype=float)
    y_train = train_table[target_column].to_numpy()
    X_test = test_table[feature_columns].to_numpy(dtype=float)
    y_test = test_table[target_column].to_numpy()
    class_order = sorted(np.unique(np.concatenate([y_train, y_test])))

    print(f"\n{'=' * 70}\n{modality.upper()} "
          f"({len(feature_columns)} features)\n{'=' * 70}")
    print(f"Train: {len(y_train)} windows. Test: {len(y_test)} windows. "
          f"Classes: {class_order}")

    model = build_classifier(classifier_name, **classifier_kwargs)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    report_text = classification_report(y_test, y_predicted, labels=class_order)
    report_dict = classification_report(
        y_test, y_predicted, labels=class_order, output_dict=True)
    print(f"\n=== {classifier_name} test-set performance ({modality}) ===")
    print(report_text)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(f"Confusion matrix -- {classifier_name} ({modality})")
    fig.tight_layout()
    fig.savefig(SCRIPT_DIRECTORY / f"confusion_matrix_{modality}.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"y_test": y_test, "y_predicted": y_predicted}).to_csv(
        SCRIPT_DIRECTORY / f"predictions_{modality}.csv", index=False)

    # --- Feature ranking: always a random forest, matching KD_test/
    # python/train_classifier.py's own reasoning -- reuses the already-
    # trained model directly when it already IS a random forest. ---
    if isinstance(model, RandomForestClassifier):
        ranking_model = model
    else:
        ranking_model = RandomForestClassifier(
            n_estimators=300, random_state=random_seed)
        ranking_model.fit(X_train, y_train)

    ranking_table = pd.DataFrame({
        "feature": feature_columns,
        "importance": ranking_model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    ranking_table.insert(0, "rank", np.arange(1, len(ranking_table) + 1))
    ranking_table.to_csv(
        SCRIPT_DIRECTORY / f"feature_importance_ranking_{modality}.csv",
        index=False)
    print(f"Top 5 {modality} features by importance:")
    print(ranking_table.head(5).to_string(index=False))

    top_features = ranking_table.head(top_n_features_to_plot)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_features) + 1.5))
    ax.barh(top_features["feature"][::-1], top_features["importance"][::-1])
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"Top {len(top_features)} {modality} features")
    fig.tight_layout()
    fig.savefig(
        SCRIPT_DIRECTORY / f"feature_importance_{modality}.png", dpi=150)
    plt.close(fig)

    return flatten_classification_report(modality, report_dict, class_order)


def main() -> None:
    rows = [train_and_evaluate_modality(modality)
            for modality in modality_csv_paths]
    metrics_table = pd.DataFrame(rows)
    metrics_table.to_csv(
        SCRIPT_DIRECTORY / "metrics_by_modality.csv", index=False)

    print(f"\n{'=' * 70}\nSummary (also written to metrics_by_modality.csv)"
          f"\n{'=' * 70}")
    print(metrics_table.to_string(index=False))


if __name__ == "__main__":
    main()
