"""train_modality_classifiers_loso.py -- Leave-one-subject-out (LOSO)
evaluation. train_modality_classifiers.py's split pools every subject's
windows together and then splits by WINDOW, so every subject is already
seen during training -- its ~75% measures generalization to new time
windows, not to a new person. This script instead holds out ONE ENTIRE
SUBJECT at a time, trains on the rest, and tests only on the held-out
subject -- the honest answer to "how would this do on someone the model
never saw."

Input: the SAME features_<modality>_{train,test}.csv build_modality_
feature_datasets.py already writes. train ∪ test there is a partition
of every window (build_modality_feature_datasets.py's balanced split
puts every window in exactly one of the two), so concatenating them
back together recovers the full pooled window set with no new
extraction step.

Per modality, per left-out subject:
  - Train: a class-balanced draw (same "balanced train" philosophy as
    build_modality_feature_datasets.py) from every OTHER subject's
    windows.
  - Test: ALL of the held-out subject's windows (uneven, natural class
    proportions -- same "balanced train / uneven test" convention used
    throughout this project).
Every fold's held-out predictions are then pooled across ALL subjects
into ONE overall classification report per modality -- the standard
way to summarize LOSO-CV, and directly comparable to train_modality_
classifiers.py's per-modality accuracy.

Output, per modality:
  - loso_fold_results_<modality>.csv (one row per held-out subject:
    accuracy, macro/per-class precision/recall/F1, train/test window
    counts).
  - loso_accuracy_by_subject_<modality>.png (one bar per held-out
    subject, with the pooled overall accuracy marked).
  - loso_confusion_matrix_<modality>.png (pooled over every fold's
    held-out predictions, annotated with accuracy/precision/recall/F1
    -- same style as train_modality_classifiers_top_n.py's).
Combined:
  - loso_metrics_by_modality.csv (the pooled report, one row per
    modality -- same column shape as metrics_by_modality.csv, so it
    can sit side by side with the non-LOSO numbers).
  - a pooled-window-split vs. LOSO accuracy/macro-F1 comparison printed
    to the command window.

Dependencies: numpy, pandas, scikit-learn, matplotlib
  pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import train_modality_classifiers as tmc

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
# Reused directly from train_modality_classifiers.py so a setting only
# has to be changed in one place.
modality_csv_paths = tmc.modality_csv_paths
metadata_columns = tmc.metadata_columns
target_column = tmc.target_column
classifier_name = tmc.classifier_name
classifier_kwargs = tmc.classifier_kwargs

# TODO same balanced-training philosophy as build_modality_feature_
# datasets.py: draw this fraction of the smaller class's window count
# from the OTHER subjects for training each fold. The held-out
# subject's full, uneven-by-design set is always used whole for test.
train_fraction_of_smallest_class = 0.5
random_seed = 42


def draw_balanced_train(table: pd.DataFrame, random_state: int) -> pd.DataFrame:
    class_counts = table[target_column].value_counts()
    per_class_count = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)),
        1)
    parts = [
        class_rows.sample(
            n=min(per_class_count, len(class_rows)),
            random_state=random_state)
        for _, class_rows in table.groupby(target_column)
    ]
    return pd.concat(parts, ignore_index=True)


def run_loso_for_modality(modality: str) -> dict:
    train_csv = pd.read_csv(modality_csv_paths[modality]["train"])
    test_csv = pd.read_csv(modality_csv_paths[modality]["test"])
    pooled_table = pd.concat([train_csv, test_csv], ignore_index=True)
    feature_columns = [
        c for c in pooled_table.columns
        if c not in metadata_columns and c != target_column]
    class_order = sorted(pooled_table[target_column].unique())
    subjects = sorted(int(s) for s in pooled_table["subject"].unique())

    print(f"\n{'=' * 70}\n{modality.upper()} -- leave-one-subject-out "
          f"({len(subjects)} subjects: {subjects})\n{'=' * 70}")

    fold_rows = []
    all_true, all_predicted = [], []
    for held_out_subject in subjects:
        test_table = pooled_table[pooled_table["subject"] == held_out_subject]
        other_table = pooled_table[pooled_table["subject"] != held_out_subject]
        train_table = draw_balanced_train(other_table, random_seed)

        X_train = train_table[feature_columns].to_numpy(dtype=float)
        y_train = train_table[target_column].to_numpy()
        X_test = test_table[feature_columns].to_numpy(dtype=float)
        y_test = test_table[target_column].to_numpy()

        model = tmc.build_classifier(classifier_name, **classifier_kwargs)
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)

        report = classification_report(
            y_test, y_predicted, labels=class_order, output_dict=True)
        row = tmc.flatten_classification_report(
            f"s{held_out_subject}", report, class_order)
        row["subject"] = row.pop("modality")
        row["train_windows"] = len(train_table)
        row["test_windows"] = len(test_table)
        fold_rows.append(row)
        all_true.append(y_test)
        all_predicted.append(y_predicted)

        print(f"  held out s{held_out_subject}: accuracy="
              f"{report['accuracy']:.3f}, train={len(train_table)}, "
              f"test={len(test_table)}")

    fold_table = pd.DataFrame(fold_rows)
    fold_table = fold_table[
        ["subject"] + [c for c in fold_table.columns if c != "subject"]]
    fold_table.to_csv(
        PROCESSED_DATA_DIRECTORY /
        f"loso_fold_results_{modality}_{tmc.dataset_name}.csv",
        index=False)

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)
    pooled_report = classification_report(
        all_true, all_predicted, labels=class_order, output_dict=True)

    plot_loso_results(
        modality, fold_table["subject"], fold_table["accuracy"],
        all_true, all_predicted, class_order, pooled_report)

    return tmc.flatten_classification_report(modality, pooled_report, class_order)


def plot_loso_results(
        modality: str, fold_subjects, fold_accuracies,
        all_true: np.ndarray, all_predicted: np.ndarray, class_order: list,
        pooled_report: dict, filename_suffix: str = "",
        title_suffix: str = "") -> None:
    """Shared by this script's untuned run and tune_random_forest_loso.py's
    tuned run -- same two plots (accuracy-by-subject bar chart, pooled
    confusion matrix annotated with accuracy/precision/recall/F1), just
    a different filename/title suffix so the two don't collide."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(fold_subjects, fold_accuracies)
    ax.axhline(
        pooled_report["accuracy"], color="red", linestyle="--",
        label=f"pooled overall = {pooled_report['accuracy']:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"LOSO accuracy by held-out subject -- {modality}{title_suffix}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        IMAGE_DATA_DIRECTORY /
        f"loso_accuracy_by_subject_{modality}_{tmc.dataset_name}"
        f"{filename_suffix}.png",
        dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.8))
    ConfusionMatrixDisplay.from_predictions(
        all_true, all_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(
        f"LOSO pooled confusion matrix -- {modality}{title_suffix}")
    macro = pooled_report["macro avg"]
    metrics_text = (
        f"Accuracy: {pooled_report['accuracy']:.3f}    "
        f"Precision (macro): {macro['precision']:.3f}    "
        f"Recall (macro): {macro['recall']:.3f}    "
        f"F1-score (macro): {macro['f1-score']:.3f}")
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(
        IMAGE_DATA_DIRECTORY /
        f"loso_confusion_matrix_{modality}_{tmc.dataset_name}"
        f"{filename_suffix}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    pooled_rows = [
        run_loso_for_modality(modality) for modality in modality_csv_paths]
    summary_table = pd.DataFrame(pooled_rows)
    summary_path = (PROCESSED_DATA_DIRECTORY /
                    f"loso_metrics_by_modality_{tmc.dataset_name}.csv")
    summary_table.to_csv(summary_path, index=False)

    print(f"\n{'=' * 70}\nLOSO summary -- pooled over every held-out "
          f"subject (also written to {summary_path.name})\n{'=' * 70}")
    print(summary_table.to_string(index=False))

    full_metrics_path = (PROCESSED_DATA_DIRECTORY /
                         f"metrics_by_modality_{tmc.dataset_name}.csv")
    if full_metrics_path.is_file():
        full_metrics = pd.read_csv(full_metrics_path)
        comparison = full_metrics[["modality", "accuracy", "macro_f1"]].merge(
            summary_table[["modality", "accuracy", "macro_f1"]],
            on="modality", suffixes=("_pooled_split", "_loso"))
        print(f"\nPooled-window-split vs. LOSO comparison:")
        print(comparison.to_string(index=False))
    else:
        print(f"\n({full_metrics_path.name} not found -- run "
              f"train_modality_classifiers.py for a pooled-split "
              f"comparison too.)")


if __name__ == "__main__":
    main()
