"""compare_loso_feature_sets.py -- Leave-one-subject-out comparison of
SEVERAL feature sets side by side, to quantify what eye gaze actually
contributes over robot data alone (improve / worsen / maintain).

Why this script exists rather than reusing the others: train_modality_
classifiers_top_n.py does top-N but on the POOLED window split, which
lets the same subject appear in train and test. Only LOSO answers "how
would this do on a NEW person", so the top-N restriction is redone here
inside the LOSO loop.

WHAT "JOINT" IS NOT: classifier_feature_bank.py's joint bank is NOT
robot + gaze concatenated. It shares zero features with the robot bank
-- it is 28 gaze features plus 23 cross-modal ones (gaze offset angle
vs. force, gaze-EE distance vs. curvature, ...). So "joint vs. robot"
does not measure gaze's marginal value. The robot_plus_gaze set below
is the real test: the robot bank's columns and the gaze bank's columns
concatenated on window identity.

FEATURE SETS EVALUATED (each under identical 9-fold LOSO):
  robot            - ur5e_*.csv only (96 features)
  gaze             - the gaze bank as built (102), 30 of which secretly
                     also read the robot CSV (see robot_dependent_gaze_
                     features below)
  gaze_pure        - only the 72 gaze features that are derived from
                     tobii_*.csv ALONE -- gaze-only in the strict sense
  joint            - the cross-modal bank (51)
  robot_plus_gaze  - robot bank + gaze bank concatenated
  robot_plus_gaze_pure - robot bank + strictly-Tobii features
Each is run with ALL of its features and again with only its top N.

TOP-N SELECTION IS DONE INSIDE EACH FOLD, from that fold's TRAINING
data only. Ranking once on all data and then running LOSO would leak
the held-out subject into feature selection and inflate the result.

BASELINE: every accuracy is reported against the majority-class rate of
that fold's own test set, since the classes are not balanced and each
subject's balance differs.

Inputs:  data/processed_data/features_{robot,gaze,joint}_{train,test}_
         <dataset_name>.csv, written by
         build_modality_feature_datasets.py.
Outputs: data/processed_data/loso_feature_set_comparison_<dataset>.csv
         data/processed_data/loso_feature_set_folds_<dataset>.csv
         data/images/loso_feature_set_comparison_<dataset>.png

Dependencies: numpy, pandas, scikit-learn, matplotlib
  pip install numpy pandas scikit-learn matplotlib
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix)

import classifier_feature_bank as cfb
import train_modality_classifiers as tmc

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
# TODO how many top-ranked features each "_top" variant keeps [count].
top_n_features = 20

# TODO which feature sets to evaluate; keys of build_feature_sets().
feature_sets_to_run = (
    "robot", "gaze", "gaze_pure", "joint",
    "robot_plus_gaze", "robot_plus_gaze_pure",
)

# TODO also run each set restricted to its top N features.
run_top_n_variants = True

# TODO lead times [s] for the anticipation sweep: predict the label
# window_offset steps AHEAD from features at t. Must be multiples of
# window_step_s (0.5 s), since the offset is applied in window steps.
# Empty tuple skips the sweep. 0.5 s is omitted deliberately: with
# transition_margin_s = 0.25 the windows either side of a label flip
# are already dropped, so every surviving 0.5 s pair has the same label
# (persistence = 100%) and there is nothing to anticipate. Regimes are
# only genuinely uncertain from ~1 s out (persistence 92% at 1 s, 68%
# at 2 s).
lead_time_sweep_s = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0)
window_step_s = 0.5

# TODO which feature sets enter the lead-time sweep. Gaze vs. robot is
# the comparison that matters: force already predicts the near-future
# label well because force is autocorrelated, so gaze only earns a
# place in the controller if its advantage GROWS with lead time.
lead_time_feature_sets = ("gaze_pure", "robot")

# Reused from train_modality_classifiers.py so settings live in one place.
dataset_name = tmc.dataset_name
target_column = tmc.target_column
metadata_columns = tmc.metadata_columns
classifier_name = tmc.classifier_name
classifier_kwargs = tmc.classifier_kwargs

# TODO fraction of the smaller class drawn per class for each fold's
# training set -- matches train_modality_classifiers_loso.py.
train_fraction_of_smallest_class = 0.5
random_seed = 42

# Window identity, used to line robot and gaze rows up for concatenation.
key_columns = ["subject", "bandwidth", "shape", "direction", "window_index"]


def modality_table(modality: str) -> pd.DataFrame:
    """One modality's train+test windows pooled back into one table, the
    same way train_modality_classifiers_loso.py pools them."""
    paths = tmc.all_modality_csv_paths[modality]
    return pd.concat(
        [pd.read_csv(paths["train"]), pd.read_csv(paths["test"])],
        ignore_index=True)


def feature_columns_of(table: pd.DataFrame) -> list:
    return [c for c in table.columns
            if c not in metadata_columns and c != target_column]


def robot_dependent_gaze_features() -> set:
    """Gaze-bank features whose value changes when only the ROBOT path
    changes -- i.e. those that need ur5e_*.csv despite living in the
    "gaze" bank. Probed empirically rather than matched by name, with
    two different robot paths (position AND speed differ) plus a pure
    translation, since some features are translation-invariant and
    would otherwise look robot-independent."""
    config = cfb.FeatureConfig(
        workspace_center_xy_m=(0.0, -0.65), workspace_half_extent_m=0.2,
        low_speed_threshold_m_s=0.05, high_speed_threshold_m_s=0.3,
        close_thresholds_m=(0.01, 0.02, 0.05))

    sample_count = 40
    gaze_rng = np.random.default_rng(0)
    Gt = np.linspace(0.0, 0.5, sample_count)
    Gx = gaze_rng.normal(0.0, 0.05, sample_count)
    Gy = gaze_rng.normal(-0.65, 0.05, sample_count)
    Pt = np.linspace(0.0, 0.5, sample_count)

    first_rng = np.random.default_rng(1)
    Px1 = first_rng.normal(0.0, 0.05, sample_count)
    Py1 = first_rng.normal(-0.65, 0.05, sample_count)
    second_rng = np.random.default_rng(2)
    Px2 = second_rng.normal(0.1, 0.2, sample_count)
    Py2 = second_rng.normal(-0.5, 0.3, sample_count)

    robot_dependent = set()
    for name, feature_fn in cfb.build_gaze_feature_bank(config):
        baseline = feature_fn(Gx, Gy, Gt, Px1, Py1, Pt)
        for Px, Py in ((Px2, Py2), (Px1 + 0.13, Py1 - 0.07)):
            other = feature_fn(Gx, Gy, Gt, Px, Py, Pt)
            unchanged = (np.isnan(baseline) and np.isnan(other)) or \
                np.isclose(baseline, other, equal_nan=True)
            if not unchanged:
                robot_dependent.add(name)
                break
    return robot_dependent


def build_feature_sets() -> dict:
    """{name: (table, feature_columns)} for every set being compared."""
    robot_table = modality_table("robot")
    gaze_table = modality_table("gaze")
    joint_table = modality_table("joint")

    robot_features = feature_columns_of(robot_table)
    gaze_features = feature_columns_of(gaze_table)
    joint_features = feature_columns_of(joint_table)
    pure_gaze_features = [f for f in gaze_features
                          if f not in robot_dependent_gaze_features()]

    # Robot + gaze concatenated on window identity. Suffix any name
    # present in both banks so neither silently overwrites the other.
    shared_names = set(robot_features) & set(gaze_features)
    gaze_renamed = gaze_table.rename(
        columns={name: f"{name}_gaze" for name in shared_names})
    combined_table = robot_table.merge(
        gaze_renamed[key_columns + [
            f"{f}_gaze" if f in shared_names else f for f in gaze_features]],
        on=key_columns, how="inner", validate="one_to_one")
    combined_gaze_features = [
        f"{f}_gaze" if f in shared_names else f for f in gaze_features]
    combined_pure_features = [
        f"{f}_gaze" if f in shared_names else f for f in pure_gaze_features]

    print(f"robot {len(robot_features)} | gaze {len(gaze_features)} "
          f"(pure {len(pure_gaze_features)}) | joint {len(joint_features)} "
          f"| robot+gaze {len(robot_features) + len(combined_gaze_features)}")
    if len(combined_table) != len(robot_table):
        print(f"WARNING: robot/gaze merge kept {len(combined_table)} of "
              f"{len(robot_table)} windows -- window identities differ.")

    return {
        "robot": (robot_table, robot_features),
        "gaze": (gaze_table, gaze_features),
        "gaze_pure": (gaze_table, pure_gaze_features),
        "joint": (joint_table, joint_features),
        "robot_plus_gaze": (
            combined_table, robot_features + combined_gaze_features),
        "robot_plus_gaze_pure": (
            combined_table, robot_features + combined_pure_features),
    }


def draw_balanced_train(table: pd.DataFrame, seed: int,
                        label_column: str = None) -> pd.DataFrame:
    label_column = label_column or target_column
    class_counts = table[label_column].value_counts()
    per_class_count = max(
        int(np.floor(class_counts.min() * train_fraction_of_smallest_class)), 1)
    parts = [class_rows.sample(n=min(per_class_count, len(class_rows)),
                               random_state=seed)
             for _, class_rows in table.groupby(label_column)]
    return pd.concat(parts, ignore_index=True)


def shift_labels_forward(table: pd.DataFrame, window_offset: int):
    """Attach, as "future_label", the label of the window window_offset
    steps later in the SAME trial. window_index is assigned before any
    window is filtered out, so index i and i+k are exactly
    k * window_step_s apart in time even where windows were dropped.
    Rows whose future window was dropped are removed."""
    if window_offset == 0:
        shifted = table.copy()
        shifted["future_label"] = shifted[target_column]
        return shifted

    trial_columns = [c for c in key_columns if c != "window_index"]
    future = table[trial_columns + ["window_index", target_column]].copy()
    future["window_index"] = future["window_index"] - window_offset
    future = future.rename(columns={target_column: "future_label"})
    return table.merge(future, on=trial_columns + ["window_index"],
                       how="inner", validate="one_to_one")


def run_loso(set_name: str, table: pd.DataFrame, feature_columns: list,
             keep_top_n: int = None, label_column: str = None) -> tuple:
    """LOSO over every subject. Returns (summary_row, fold_rows,
    pooled_true, pooled_predicted) -- the last two are every held-out
    subject's true/predicted label, concatenated in fold order, for
    building a confusion matrix without refitting."""
    label_column = label_column or target_column
    subjects = sorted(int(s) for s in table["subject"].unique())
    class_order = sorted(table[label_column].unique())
    fold_rows = []
    all_true, all_predicted = [], []

    for held_out_subject in subjects:
        test_table = table[table["subject"] == held_out_subject]
        train_table = draw_balanced_train(
            table[table["subject"] != held_out_subject], random_seed,
            label_column)

        fold_features = feature_columns
        if keep_top_n is not None and keep_top_n < len(feature_columns):
            # Rank on THIS fold's training data only -- ranking on the
            # pooled table would leak the held-out subject.
            ranker = tmc.build_classifier(classifier_name, **classifier_kwargs)
            ranker.fit(train_table[feature_columns], train_table[label_column])
            order = np.argsort(ranker.feature_importances_)[::-1]
            fold_features = [feature_columns[i] for i in order[:keep_top_n]]

        model = tmc.build_classifier(classifier_name, **classifier_kwargs)
        model.fit(train_table[fold_features], train_table[label_column])
        predicted = model.predict(test_table[fold_features])

        fold_accuracy = accuracy_score(test_table[label_column], predicted)
        fold_baseline = test_table[label_column].value_counts(
            normalize=True).max()
        # Persistence: assume the regime does not change over the lead
        # time. The controller gets this for free from the force sensor,
        # so an anticipatory model must beat it, not just the majority.
        if label_column != target_column:
            persistence_accuracy = accuracy_score(
                test_table[label_column], test_table[target_column])
        else:
            persistence_accuracy = float("nan")
        fold_rows.append({
            "feature_set": set_name, "subject": held_out_subject,
            "accuracy": fold_accuracy, "majority_baseline": fold_baseline,
            "persistence_baseline": persistence_accuracy,
            "lift": fold_accuracy - fold_baseline,
            "lift_over_persistence": fold_accuracy - persistence_accuracy,
            "train_windows": len(train_table), "test_windows": len(test_table),
            "feature_count": len(fold_features),
        })
        all_true.append(test_table[label_column])
        all_predicted.append(pd.Series(predicted))

    pooled_true = pd.concat(all_true, ignore_index=True)
    pooled_predicted = pd.concat(all_predicted, ignore_index=True)
    report = classification_report(
        pooled_true, pooled_predicted, output_dict=True, zero_division=0)
    summary_row = tmc.flatten_classification_report(
        set_name, report, class_order)
    fold_table = pd.DataFrame(fold_rows)
    summary_row["feature_count"] = int(fold_table["feature_count"].iloc[0])
    summary_row["mean_fold_accuracy"] = fold_table["accuracy"].mean()
    summary_row["mean_majority_baseline"] = \
        fold_table["majority_baseline"].mean()
    summary_row["mean_lift"] = fold_table["lift"].mean()
    summary_row["mean_persistence_baseline"] = \
        fold_table["persistence_baseline"].mean()
    summary_row["mean_lift_over_persistence"] = \
        fold_table["lift_over_persistence"].mean()
    summary_row["worst_fold_accuracy"] = fold_table["accuracy"].min()
    summary_row["class_order"] = class_order
    return summary_row, fold_rows, pooled_true, pooled_predicted


def run_lead_time_sweep(feature_sets: dict) -> tuple:
    """Predict the label lead_time_s AHEAD from features at t, for each
    feature set, so gaze's anticipation can be compared against force's
    (and against simply assuming the regime persists)."""
    summary_rows, fold_rows = [], []
    print(f"\n{'=' * 78}\nLead-time sweep -- predicting label(t + lead) "
          f"from features(t)\n{'=' * 78}")

    for lead_time_s in lead_time_sweep_s:
        window_offset = int(round(lead_time_s / window_step_s))
        for set_name in lead_time_feature_sets:
            table, feature_columns = feature_sets[set_name]
            shifted = shift_labels_forward(table, window_offset)
            variant_name = f"{set_name}@lead{lead_time_s:g}s"

            summary_row, variant_folds, _, _ = run_loso(
                variant_name, shifted, feature_columns,
                label_column="future_label")
            summary_row["lead_time_s"] = lead_time_s
            summary_row["base_feature_set"] = set_name
            summary_row["windows"] = len(shifted)
            for fold_row in variant_folds:
                fold_row["lead_time_s"] = lead_time_s
                fold_row["base_feature_set"] = set_name
            summary_rows.append(summary_row)
            fold_rows.extend(variant_folds)

            persistence = summary_row["mean_persistence_baseline"]
            print(f"  {variant_name:<28} acc={summary_row['accuracy']:.3f}  "
                  f"majority={summary_row['mean_majority_baseline']:.3f}  "
                  f"persistence={persistence:.3f}  "
                  f"vs_persistence={summary_row['mean_lift_over_persistence']:+.3f}"
                  f"  n_win={len(shifted)}")
    return summary_rows, fold_rows


def plot_comparison(summary_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(figsize=(11, 0.5 * len(summary_table) + 3))
    ordered = summary_table.sort_values("accuracy")
    positions = np.arange(len(ordered))
    axes.barh(positions, ordered["accuracy"], color="#4C72B0", label="LOSO accuracy")
    axes.plot(ordered["mean_majority_baseline"], positions, "kd",
              label="majority-class baseline")
    axes.set_yticks(positions)
    axes.set_yticklabels(
        [f"{r.modality}  (n={r.feature_count})"
         for r in ordered.itertuples()], fontsize=8)
    axes.set_xlabel("Accuracy")
    axes.set_xlim(0.0, 1.0)
    axes.grid(axis="x", alpha=0.3)
    axes.set_title(f"LOSO by feature set -- {dataset_name}")
    axes.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_lead_time_sweep(sweep_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(figsize=(8, 5))
    for set_name, rows in sweep_table.groupby("base_feature_set"):
        rows = rows.sort_values("lead_time_s")
        axes.plot(rows["lead_time_s"], rows["accuracy"], "o-", label=set_name)
    reference = sweep_table.sort_values("lead_time_s").drop_duplicates(
        "lead_time_s")
    axes.plot(reference["lead_time_s"], reference["mean_persistence_baseline"],
              "k--", label="persistence (assume no change)")
    axes.plot(reference["lead_time_s"], reference["mean_majority_baseline"],
              ":", color="gray", label="majority class")
    axes.set_xlabel("Lead time [s]")
    axes.set_ylabel("LOSO accuracy")
    axes.set_ylim(0.4, 1.0)
    axes.grid(alpha=0.3)
    axes.set_title(f"Anticipation: predicting label(t + lead) -- {dataset_name}")
    axes.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
        set_name: str, pooled_true: pd.Series, pooled_predicted: pd.Series,
        class_order: list, output_path: Path) -> None:
    matrix = confusion_matrix(pooled_true, pooled_predicted, labels=class_order)
    display = ConfusionMatrixDisplay(matrix, display_labels=class_order)
    fig, axes = plt.subplots(figsize=(4.5, 4.5))
    display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")
    axes.set_title(f"LOSO pooled confusion matrix\n{set_name} ({dataset_name})",
                   fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    feature_sets = build_feature_sets()
    summary_rows, fold_rows = [], []
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for set_name in feature_sets_to_run:
        table, feature_columns = feature_sets[set_name]
        variants = [(set_name, None)]
        if run_top_n_variants and len(feature_columns) > top_n_features:
            variants.append(
                (f"{set_name}_top{top_n_features}", top_n_features))

        for variant_name, keep_top_n in variants:
            summary_row, variant_folds, pooled_true, pooled_predicted = \
                run_loso(variant_name, table, feature_columns, keep_top_n)
            summary_rows.append(summary_row)
            fold_rows.extend(variant_folds)
            print(f"  {variant_name:<32} n={summary_row['feature_count']:>3}  "
                  f"acc={summary_row['accuracy']:.3f}  "
                  f"baseline={summary_row['mean_majority_baseline']:.3f}  "
                  f"lift={summary_row['mean_lift']:+.3f}")
            plot_confusion_matrix(
                variant_name, pooled_true, pooled_predicted,
                summary_row["class_order"],
                IMAGE_DATA_DIRECTORY /
                f"confusion_matrix_loso_{variant_name}_{dataset_name}.png")

    if lead_time_sweep_s:
        sweep_summary, sweep_folds = run_lead_time_sweep(feature_sets)
        sweep_summary_path = (
            PROCESSED_DATA_DIRECTORY /
            f"loso_lead_time_sweep_{dataset_name}.csv")
        sweep_folds_path = (
            PROCESSED_DATA_DIRECTORY /
            f"loso_lead_time_sweep_folds_{dataset_name}.csv")
        PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sweep_summary).to_csv(sweep_summary_path, index=False)
        pd.DataFrame(sweep_folds).to_csv(sweep_folds_path, index=False)
        plot_lead_time_sweep(
            pd.DataFrame(sweep_summary),
            IMAGE_DATA_DIRECTORY / f"loso_lead_time_sweep_{dataset_name}.png")
        print(f"\nWrote:\n  {sweep_summary_path.name}\n"
              f"  {sweep_folds_path.name}")

    if not summary_rows:
        # feature_sets_to_run empty: sweep-only run, leave the existing
        # comparison CSVs alone rather than overwriting them with nothing.
        return

    summary_table = pd.DataFrame(summary_rows)
    fold_table = pd.DataFrame(fold_rows)

    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    summary_path = (PROCESSED_DATA_DIRECTORY /
                    f"loso_feature_set_comparison_{dataset_name}.csv")
    folds_path = (PROCESSED_DATA_DIRECTORY /
                  f"loso_feature_set_folds_{dataset_name}.csv")
    image_path = (IMAGE_DATA_DIRECTORY /
                  f"loso_feature_set_comparison_{dataset_name}.png")
    summary_table.to_csv(summary_path, index=False)
    fold_table.to_csv(folds_path, index=False)
    plot_comparison(summary_table, image_path)

    print(f"\n{'=' * 78}\nLOSO feature-set comparison ({dataset_name})\n"
          f"{'=' * 78}")
    print(summary_table[[
        "modality", "feature_count", "accuracy", "macro_f1",
        "mean_majority_baseline", "mean_lift", "worst_fold_accuracy",
    ]].rename(columns={"modality": "feature_set"}).to_string(index=False))
    print(f"\nWrote:\n  {summary_path.name}\n  {folds_path.name}\n"
          f"  {image_path.name}")


if __name__ == "__main__":
    main()
