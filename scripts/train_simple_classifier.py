"""train_simple_classifier.py -- A deliberately simple, single-file
classifier trainer for ONE modality at a time, meant to be easy to read
top-to-bottom and easy to tweak which features go into the model.

Unlike train_modality_classifiers.py / _loso.py (which loop over all
three modalities, share state via module-level imports, and drive
several downstream comparison scripts), this script does exactly one
thing: load one modality's already-built train/test CSVs, keep or drop
features by NAME, train one classifier, print/plot the result. Nothing
here is imported by another script -- it's safe to edit freely, and
editing it cannot break train_modality_classifiers.py, the LOSO script,
or the top-20/comparison scripts that import that pipeline as a module.

Input: features_<modality>_{train,test}_<dataset_name>.csv, as written
by build_modality_feature_datasets.py. Run that script first if these
don't exist yet.

Workflow for picking features:
  1. Set list_available_features_and_exit = True below and run the
     script -- it prints every feature column found in the CSV, already
     formatted so you can paste lines straight into the lists below,
     then exits without training.
  2. Comment/uncomment the names you want in features_to_exclude (drop a
     few, keep the rest) or features_to_include (keep only these, drop
     everything else -- takes priority if non-empty).
  3. Set list_available_features_and_exit = False and run again to train.

Output:
  - Printed classification report (accuracy, per-class/macro precision/
    recall/F1) on the held-out test split.
  - data/images/simple_confusion_matrix_<modality>_<dataset_name>.png
  - data/images/simple_feature_importance_<modality>_<dataset_name>.png

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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
# TODO which modality's feature CSVs to load.
modality = "robot"  # "robot" | "gaze" | "joint"

# TODO which dataset's feature CSVs to read -- must match the
# dataset_name build_modality_feature_datasets.py was run with.
dataset_name = "chen_manual"

# TODO set True to print every available feature name (and how many)
# then exit without training -- see "Workflow for picking features"
# above.
list_available_features_and_exit = True

# TODO feature names to drop from the "use everything" default.
# Ignored whenever features_to_include (below) is non-empty. Comment/
# uncomment individual lines to toggle a feature on/off.
features_to_exclude = [
    # "mean_position_x",
    # "mean_position_y",
]

# TODO if non-empty, ONLY these feature names are used -- overrides
# features_to_exclude entirely. For building a small, specific feature
# set rather than "everything minus a few".
features_to_include = [
    # "mean_ee_speed_m_s",
    # "power_law_exponent",
]

# TODO random forest settings. See sklearn.ensemble.RandomForestClassifier
# for the full parameter list if you need one not listed here.
n_estimators = 300
random_seed = 42

# TODO how many top-ranked features to show in the importance bar chart.
top_n_features_to_plot = 20
# -------------------- End User Settings --------------------

# Non-feature columns present in every features_*.csv -- excluded from
# the model regardless of dataset (a column a given dataset doesn't
# have is harmless to list here).
metadata_columns = [
    "subject", "shape", "damping_n_s_m", "bandwidth", "direction",
    "window_index", "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count",
]
target_column = "trial_class_label"


def available_feature_columns(table: pd.DataFrame) -> list:
    """Every CSV column that is neither metadata nor the label.

    Input: a features_<modality>_*.csv table.
    Output: list of feature column names.
    """
    return [c for c in table.columns
            if c not in metadata_columns and c != target_column]


def select_feature_columns(all_features: list) -> list:
    """Apply features_to_include / features_to_exclude to the full
    column list, in that priority order (see User Settings above).

    Input: every available feature name.
    Output: the feature names to actually train on.
    """
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


def main() -> None:
    start_time_s = time.perf_counter()

    train_path = (PROCESSED_DATA_DIRECTORY /
                  f"features_{modality}_train_{dataset_name}.csv")
    test_path = (PROCESSED_DATA_DIRECTORY /
                 f"features_{modality}_test_{dataset_name}.csv")
    train_table = pd.read_csv(train_path)
    test_table = pd.read_csv(test_path)

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

    X_train = train_table[feature_columns].to_numpy(dtype=float)
    y_train = train_table[target_column].to_numpy()
    X_test = test_table[feature_columns].to_numpy(dtype=float)
    y_test = test_table[target_column].to_numpy()
    class_order = sorted(np.unique(np.concatenate([y_train, y_test])))

    print(f"Train: {len(y_train)} windows. Test: {len(y_test)} windows. "
          f"Classes: {class_order}")

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)

    print(f"\n=== test-set performance ({modality}, {dataset_name}) ===")
    print(classification_report(y_test, y_predicted, labels=class_order))

    IMAGE_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_predicted, labels=class_order, ax=ax, colorbar=True)
    ax.set_title(f"Confusion matrix -- {modality} ({dataset_name})")
    fig.tight_layout()
    fig.savefig(
        IMAGE_DATA_DIRECTORY /
        f"simple_confusion_matrix_{modality}_{dataset_name}.png", dpi=150)
    plt.close(fig)

    ranking = pd.Series(
        model.feature_importances_, index=feature_columns
    ).sort_values(ascending=False)
    top_ranking = ranking.head(top_n_features_to_plot).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(top_ranking) + 1.5))
    ax.barh(top_ranking.index, top_ranking.values)
    ax.set_xlabel("Random forest feature importance")
    ax.set_title(f"Top {len(top_ranking)} features -- {modality}")
    fig.tight_layout()
    fig.savefig(
        IMAGE_DATA_DIRECTORY /
        f"simple_feature_importance_{modality}_{dataset_name}.png", dpi=150)
    plt.close(fig)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nTotal time: {elapsed_s:.1f} s ({elapsed_s / 60.0:.1f} min)")


if __name__ == "__main__":
    main()
