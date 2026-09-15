"""build_holdout_feature_dataset.py -- Re-pool an existing feature
train/test pair and re-split it by excluding one or more values of a
chosen grouping column (subject/shape/bandwidth/direction), instead of
the original random per-class draw. Output has the same columns as the
input, so train_logistic_regression.py (or any other training script)
can point read_train_csv/read_test_csv at it unchanged.

Inputs (exact filenames in User Settings, read_*_csv):
- one feature train/test CSV pair (e.g. features_all_{train,test}_W_1s.csv)

Outputs (exact filenames in User Settings, write_*_csv):
- write_train_csv, write_test_csv -- same columns as the input, split by
  held_out_column/held_out_values instead of by random draw

Depends on:
- build_modality_feature_datasets.py (writes the read_*_csv files)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed_data"

# -------------------- Start User Settings --------------------
# TODO explicit input filenames (data/processed_data/) -- the ORIGINAL
# train/test pair, pooled together here before re-splitting. That
# original split was a random balanced draw, NOT grouped by subject/
# shape/bandwidth/direction (the same subject can appear in both), so
# it must be re-pooled before splitting by group membership.
read_train_csv = "features_all_train_W_1s.csv"
read_test_csv = "features_all_test_W_1s.csv"

# TODO explicit output filenames (data/processed_data/) -- point
# train_logistic_regression.py's read_train_csv/read_test_csv at these.
# Rename these yourself when held_out_column/_values change below, so a
# new holdout run doesn't silently overwrite a previous one (no
# auto-generated tag in the filename, per this project's convention).
write_train_csv = "features_all_train_holdout_fine_gross_shape_W_1s.csv"
write_test_csv = "features_all_test_holdout_fine_gross_shape_W_1s.csv"

# TODO which column to hold out by, and which of its values become the
# TEST set -- everything else becomes TRAIN. held_out_values must match
# held_out_column's own value type below (from build_modality_feature_
# datasets.py's chen_manual_* settings); every available value is also
# printed at the start of a run, so you can check the spelling/id
# before committing to it.
#   held_out_column     held_out_values must be drawn from
#   "subject"       ->  [10], [11], [12], [13], [14], [15], [16], [17], [18]
#   "shape"         ->  ["ellipse"], ["pear"], ["fine_gross"]
#   "bandwidth"     ->  ["narrow"], ["wide"]
#   "direction"     ->  ["clockwise"], ["counter_clockwise"]
held_out_column = "shape"
held_out_values = ["fine_gross"]

# TODO after excluding held_out_values, draw a class-balanced TRAINING
# set from what's left (matches build_modality_feature_datasets.py's
# own convention) -- set False to keep every remaining row as train,
# whatever the resulting class balance.
balance_train_classes = True
train_fraction_of_smallest_class = 0.5

target_column = "trial_class_label"  # TODO label column
random_seed = 42
# -------------------- End User Settings --------------------


def main() -> None:
    start_time_s = time.perf_counter()
    train_path = PROCESSED_DATA_DIRECTORY / read_train_csv
    test_path = PROCESSED_DATA_DIRECTORY / read_test_csv
    write_train_path = PROCESSED_DATA_DIRECTORY / write_train_csv
    write_test_path = PROCESSED_DATA_DIRECTORY / write_test_csv
    print(f"Input:  {train_path}")
    print(f"Input:  {test_path}")
    print(f"Output: {write_train_path}")
    print(f"Output: {write_test_path}")
    print()

    pooled_table = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)

    if held_out_column not in pooled_table.columns:
        raise ValueError(
            f"held_out_column='{held_out_column}' not found. Available "
            f"columns: {list(pooled_table.columns)}")
    available_values = sorted(
        pooled_table[held_out_column].dropna().unique().tolist())
    print(f"'{held_out_column}' values available: {available_values}")

    missing_values = [v for v in held_out_values if v not in available_values]
    if missing_values:
        raise ValueError(
            f"held_out_values {missing_values} not present in "
            f"'{held_out_column}'. Available: {available_values}")

    is_held_out = pooled_table[held_out_column].isin(held_out_values)
    test_table = pooled_table[is_held_out].reset_index(drop=True)
    remaining_table = pooled_table[~is_held_out].reset_index(drop=True)

    print(f"\nHeld out {held_out_column}={held_out_values}: "
          f"{len(test_table)} windows -> test.")
    print(f"Remaining: {len(remaining_table)} windows available for train.")

    if balance_train_classes:
        class_counts = remaining_table[target_column].value_counts()
        if len(class_counts) < 2 or class_counts.min() == 0:
            raise ValueError(
                f"After excluding {held_out_column}={held_out_values}, "
                f"one class has zero remaining windows for training: "
                f"{class_counts.to_dict()}")
        per_class_count = max(int(np.floor(
            class_counts.min() * train_fraction_of_smallest_class)), 1)
        train_table = pd.concat([
            class_rows.sample(n=min(per_class_count, len(class_rows)),
                               random_state=random_seed)
            for _, class_rows in remaining_table.groupby(target_column)
        ], ignore_index=True).sample(
            frac=1.0, random_state=random_seed).reset_index(drop=True)
        print(f"Balanced train draw: {per_class_count} windows per class "
              f"(from a smallest remaining class of {class_counts.min()}).")
    else:
        train_table = remaining_table

    for split_name, table in [("train", train_table), ("test", test_table)]:
        counts = table[target_column].value_counts()
        counts_text = ", ".join(f"{k}={v}" for k, v in counts.items())
        print(f"  {split_name}: {len(table)} windows ({counts_text})")

    train_table.to_csv(write_train_path, index=False)
    test_table.to_csv(write_test_path, index=False)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nWrote:\n  {write_train_path.name}\n  {write_test_path.name}")
    print(f"Total time: {elapsed_s:.1f} s")


if __name__ == "__main__":
    main()
