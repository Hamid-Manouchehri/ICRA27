"""compare_modality_classifiers.py -- Compares the three per-modality
classifiers train_modality_classifiers.py already trained (robot-only,
gaze-only, joint), by reading its saved metrics_by_modality.csv and
predictions_<modality>.csv -- no retraining here, so this reflects
exactly what that script already computed rather than a fresh (and,
for a random forest, not bit-identical) run.

Output:
  - modality_comparison.png: accuracy per modality, and macro
    precision/recall/F1 per modality, side by side.
  - confusion_matrices_by_modality.png: the three confusion matrices in
    one figure, for a direct visual comparison.
  - the full per-class (not just macro) precision/recall/F1 breakdown,
    printed as a table -- kept in the command window rather than a
    figure, since a 3-modality x 2-class x 3-metric table doesn't
    compress into a bar chart cleanly without becoming hard to read.

Dependencies: pandas, matplotlib
  pip install pandas matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# -------------------- User Settings --------------------
metrics_csv_path = SCRIPT_DIRECTORY / "metrics_by_modality.csv"
modality_order = ["robot", "gaze", "joint"]
predictions_csv_paths = {
    modality: SCRIPT_DIRECTORY / f"predictions_{modality}.csv"
    for modality in modality_order
}
class_order = ["Fine", "Gross"]

write_comparison_plot_path = SCRIPT_DIRECTORY / "modality_comparison.png"
write_confusion_matrices_plot_path = \
    SCRIPT_DIRECTORY / "confusion_matrices_by_modality.png"


def main() -> None:
    metrics_table = pd.read_csv(metrics_csv_path).set_index("modality")
    metrics_table = metrics_table.loc[modality_order]

    print("Per-modality metrics (from metrics_by_modality.csv):")
    print(metrics_table.to_string())

    # --- Comparison figure: accuracy, and macro precision/recall/F1 ---
    fig, (ax_accuracy, ax_macro) = plt.subplots(1, 2, figsize=(12, 5))

    ax_accuracy.bar(modality_order, metrics_table["accuracy"])
    ax_accuracy.set_ylim(0, 1)
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_title("Test-set accuracy by modality")
    ax_accuracy.grid(axis="y", alpha=0.3)
    for i, modality in enumerate(modality_order):
        ax_accuracy.text(
            i, metrics_table.loc[modality, "accuracy"] + 0.01,
            f"{metrics_table.loc[modality, 'accuracy']:.2f}",
            ha="center")

    macro_metrics = ["macro_precision", "macro_recall", "macro_f1"]
    x_positions = np.arange(len(modality_order))
    bar_width = 0.25
    for i, metric in enumerate(macro_metrics):
        offset = (i - 1) * bar_width
        ax_macro.bar(
            x_positions + offset, metrics_table[metric], bar_width,
            label=metric.replace("macro_", ""))
    ax_macro.set_xticks(x_positions)
    ax_macro.set_xticklabels(modality_order)
    ax_macro.set_ylim(0, 1)
    ax_macro.set_ylabel("Score")
    ax_macro.set_title("Macro precision / recall / F1 by modality")
    ax_macro.legend()
    ax_macro.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(write_comparison_plot_path, dpi=150)
    plt.close(fig)
    print(f"\nWrote {write_comparison_plot_path}")

    # --- Combined confusion matrices ---
    fig, axes = plt.subplots(1, len(modality_order), figsize=(15, 4.5))
    for ax, modality in zip(axes, modality_order):
        predictions = pd.read_csv(predictions_csv_paths[modality])
        ConfusionMatrixDisplay.from_predictions(
            predictions["y_test"], predictions["y_predicted"],
            labels=class_order, ax=ax, colorbar=False)
        accuracy = metrics_table.loc[modality, "accuracy"]
        ax.set_title(f"{modality} (accuracy={accuracy:.2f})")
    fig.tight_layout()
    fig.savefig(write_confusion_matrices_plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {write_confusion_matrices_plot_path}")

    # --- Full per-class breakdown, printed (see module docstring for
    # why this isn't also a figure) ---
    print("\nPer-class precision / recall / F1 by modality:")
    per_class_columns = [
        f"{class_name}_{metric}"
        for class_name in class_order
        for metric in ("precision", "recall", "f1")
    ]
    print(metrics_table[per_class_columns].to_string())


if __name__ == "__main__":
    main()
