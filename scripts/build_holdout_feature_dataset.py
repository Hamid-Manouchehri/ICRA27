"""build_holdout_feature_dataset.py -- Re-pool an existing feature
train/test pair, optionally shift each window's label forward within
its own trial (num_win_label_moveforward), then re-split by excluding
one or more values from one or more grouping columns (subject/shape/
bandwidth/direction) -- a window is TEST if it matches ANY held-out
criterion (union across columns/values), instead of the original random
per-class draw. Output has the same columns as the input, so train_
logistic_regression.py (or any other training script) can point
read_train_csv/read_test_csv at it unchanged.

Inputs (exact filenames in User Settings, read_*_csv):
- one feature train/test CSV pair (e.g. features_all_{train,test}_W_1s.csv)

Outputs (exact filenames in User Settings, write_*_csv):
- write_train_csv, write_test_csv -- same columns as the input, split by
  held_out_criteria instead of by random draw

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
# Rename these yourself when held_out_criteria changes below, so a
# new holdout run doesn't silently overwrite a previous one (no
# auto-generated tag in the filename, per this project's convention).
write_train_csv = "features_all_train_held_out_ellipse_shape_and_s14_1win_ahead_W_1s.csv"
write_test_csv = "features_all_test_held_out_ellipse_shape_and_s14_1win_ahead_W_1s.csv"

# TODO one or more {column: values} criteria to hold out -- a window is
# TEST if it matches ANY criterion (union across columns AND across
# values within a column), everything else becomes TRAIN. This means
# holding out {"shape": ["ellipse"], "subject": [18]} removes ALL of
# subject 18's trials (every shape) AND ALL ellipse trials (every
# subject) from training entirely -- not just their intersection
# (subject 18's ellipse trials alone), which would leave most of each
# dimension still in train and defeat the point of holding it out.
# Values must match each column's own value type (from build_modality_
# feature_datasets.py's chen_manual_* settings); every column's
# available values are also printed at the start of a run, so you can
# check spelling/ids before committing to them.
#   column          values must be drawn from
#   "subject"   ->  10, 11, 12, 13, 14, 15, 16, 17, 18
#   "shape"     ->  "ellipse", "pear", "fine_gross"
#   "bandwidth" ->  "narrow", "wide"
#   "direction" ->  "clockwise", "counter_clockwise"
held_out_criteria = {
    "subject": [14],
    "shape": ["ellipse"],
}

# TODO shift every window's label forward by this many windows within
# its own trial (subject+bandwidth+shape+direction) -- window i's
# FEATURES end up paired with the label from window i + this many
# windows later, so a classifier trains to anticipate an upcoming
# label change instead of reading off the current one. 0 = no shift
# (unchanged behavior). A trial's last N windows have no later window
# in the same trial and are dropped. [count, windows], must be >= 0.
num_win_label_moveforward = 1

# TODO after excluding held_out_criteria, draw a class-balanced TRAINING
# set from what's left (matches build_modality_feature_datasets.py's
# own convention) -- set False to keep every remaining row as train,
# whatever the resulting class balance.
balance_train_classes = True
train_fraction_of_smallest_class = 1.0

target_column = "trial_class_label"  # TODO label column
random_seed = 42
# -------------------- End User Settings --------------------


def shift_labels_forward(
        table: pd.DataFrame, window_offset: int) -> pd.DataFrame:
    """Replace each window's target_column value with the label
    window_offset windows later in the SAME trial (subject, bandwidth,
    shape, direction) -- so a window's FEATURES pair with a future
    window's label. Windows with no later window in their trial (the
    last window_offset of each trial) are dropped."""
    trial_columns = ["subject", "bandwidth", "shape", "direction"]
    future_labels = table[
        trial_columns + ["window_index", target_column]].copy()
    future_labels["window_index"] -= window_offset
    future_labels = future_labels.rename(
        columns={target_column: "_future_label"})
    shifted_table = table.drop(columns=[target_column]).merge(
        future_labels, on=trial_columns + ["window_index"],
        how="inner", validate="one_to_one")
    return shifted_table.rename(columns={"_future_label": target_column})


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

    if num_win_label_moveforward < 0:
        raise ValueError(
            f"num_win_label_moveforward must be >= 0, got "
            f"{num_win_label_moveforward}.")

    pooled_table = pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)

    if num_win_label_moveforward > 0:
        pre_shift_count = len(pooled_table)
        pooled_table = shift_labels_forward(
            pooled_table, num_win_label_moveforward)
        print(f"Shifted labels forward {num_win_label_moveforward} "
              f"window(s) within each trial: {pre_shift_count} -> "
              f"{len(pooled_table)} windows "
              f"({pre_shift_count - len(pooled_table)} dropped, no later "
              f"window in their trial).")

    if not held_out_criteria:
        raise ValueError("held_out_criteria is empty -- nothing to hold out.")

    for column in held_out_criteria:
        if column not in pooled_table.columns:
            raise ValueError(
                f"held_out_criteria column='{column}' not found. Available "
                f"columns: {list(pooled_table.columns)}")

    is_held_out = pd.Series(False, index=pooled_table.index)
    for column, values in held_out_criteria.items():
        available_values = sorted(
            pooled_table[column].dropna().unique().tolist())
        print(f"'{column}' values available: {available_values}")

        missing_values = [v for v in values if v not in available_values]
        if missing_values:
            raise ValueError(
                f"held_out_criteria['{column}'] value(s) {missing_values} "
                f"not present. Available: {available_values}")

        criterion_mask = pooled_table[column].isin(values)
        print(f"  {column}={values}: {int(criterion_mask.sum())} windows "
              f"match (may overlap with other criteria below)")
        is_held_out |= criterion_mask

    test_table = pooled_table[is_held_out].reset_index(drop=True)
    remaining_table = pooled_table[~is_held_out].reset_index(drop=True)

    print(f"\nHeld out {held_out_criteria}: "
          f"{len(test_table)} windows -> test.")
    print(f"Remaining: {len(remaining_table)} windows available for train.")

    if balance_train_classes:
        class_counts = remaining_table[target_column].value_counts()
        if len(class_counts) < 2 or class_counts.min() == 0:
            raise ValueError(
                f"After excluding {held_out_criteria}, one class has zero "
                f"remaining windows for training: {class_counts.to_dict()}")
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
