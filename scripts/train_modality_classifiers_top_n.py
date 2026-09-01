"""train_modality_classifiers_top_n.py -- Retrains and re-evaluates
each modality's classifier using ONLY its top N most important
features, by the random-forest importance ranking train_modality_
classifiers.py already computed and saved (feature_importance_
ranking_<modality>.csv) -- to see how much of the full-feature-set
performance a much smaller, more interpretable feature set recovers.

Reuses train_modality_classifiers.py's classifier registry/builder and
report-flattening helper directly (import, not duplication) -- this
script's own job is just: load a saved ranking, restrict the feature
matrix to its top N names, retrain, evaluate, and compare against the
full-feature numbers already in metrics_by_modality.csv. Run train_
modality_classifiers.py first if that file doesn't exist yet.

Output, per modality:
  - confusion_matrix_<modality>_top<N>.png
  - predictions_<modality>_top<N>.csv
Combined:
  - metrics_by_modality_top<N>.csv
  - a full-vs-top-N accuracy/macro-F1 comparison printed to the
    command window.

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

# -------------------- User Settings --------------------
# TODO how many top-ranked features (by the already-saved random-forest
# importance ranking) to retrain each modality on.
top_n_features = 10

# Reused directly from train_modality_classifiers.py so a setting only
# has to be changed in one place.
modality_csv_paths = tmc.modality_csv_paths
target_column = tmc.target_column
classifier_name = tmc.classifier_name
classifier_kwargs = tmc.classifier_kwargs


def train_and_evaluate_top_n(modality: str) -> dict:
    ranking_path = \
        SCRIPT_DIRECTORY / f"feature_importance_ranking_{modality}.csv"
    if not ranking_path.is_file():
        raise FileNotFoundError(
            f"{ranking_path.name} not found -- run "
            f"train_modality_classifiers.py first to produce it.")
    ranking_table = pd.read_csv(ranking_path)
    top_features = ranking_table.head(top_n_features)["feature"].tolist()

    train_table = pd.read_csv(modality_csv_paths[modality]["train"])
    test_table = pd.read_csv(modality_csv_paths[modality]["test"])

    X_train = train_table[top_features].to_numpy(dtype=float)
    y_train = train_table[target_column].to_numpy()
    X_test = test_table[top_features].to_numpy(dtype=float)
    y_test = test_table[target_column].to_numpy()
    class_order = sorted(np.unique(np.concatenate([y_train, y_test])))

    print(f"\n{'=' * 70}\n{modality.upper()} -- top {top_n_features} "
          f"features only\n{'=' * 70}")
    print("Features used:", ", ".join(top_features))

    model = tmc.build_classifier(classifier_name, **classifier_kwargs)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    report_text = classification_report(
        y_test, y_predicted, labels=class_order)
    report_dict = classification_report(
        y_test, y_predicted, labels=class_order, output_dict=True)
    print(f"\n=== {classifier_name} test-set performance "
          f"(top {top_n_features} {modality} features) ===")
    print(report_text)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(
        f"Confusion matrix -- top {top_n_features} {modality} features")
    fig.tight_layout()
    fig.savefig(
        SCRIPT_DIRECTORY /
        f"confusion_matrix_{modality}_top{top_n_features}.png", dpi=150)
    plt.close(fig)

    pd.DataFrame({"y_test": y_test, "y_predicted": y_predicted}).to_csv(
        SCRIPT_DIRECTORY /
        f"predictions_{modality}_top{top_n_features}.csv", index=False)

    row = tmc.flatten_classification_report(modality, report_dict, class_order)
    row["feature_count"] = len(top_features)
    return row


def main() -> None:
    rows = [train_and_evaluate_top_n(modality)
            for modality in modality_csv_paths]
    metrics_table = pd.DataFrame(rows)
    metrics_path = \
        SCRIPT_DIRECTORY / f"metrics_by_modality_top{top_n_features}.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print(f"\n{'=' * 70}\nSummary -- top {top_n_features} features per "
          f"modality (also written to {metrics_path.name})\n{'=' * 70}")
    print(metrics_table.to_string(index=False))

    full_metrics_path = SCRIPT_DIRECTORY / "metrics_by_modality.csv"
    if full_metrics_path.is_file():
        full_metrics = pd.read_csv(full_metrics_path)
        comparison = full_metrics[["modality", "accuracy", "macro_f1"]].merge(
            metrics_table[["modality", "accuracy", "macro_f1"]],
            on="modality", suffixes=("_full", f"_top{top_n_features}"))
        print(f"\nFull-feature-set vs. top-{top_n_features} comparison:")
        print(comparison.to_string(index=False))
    else:
        print(f"\n({full_metrics_path.name} not found -- run "
              f"train_modality_classifiers.py to get a full-feature-set "
              f"comparison too.)")


if __name__ == "__main__":
    main()
