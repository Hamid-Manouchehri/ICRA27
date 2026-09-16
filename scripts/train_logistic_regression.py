"""train_logistic_regression_simple.py -- Fit one logistic regression on
a manually chosen list of features (standardized first), evaluated on a
single train/test split. Flat script, no LOSO/CV -- for a quick,
easy-to-trace fit, not a benchmark.

Inputs (exact filenames in User Settings, read_*_csv):
- features_<modality>_{train,test}_<dataset_name>.csv

Outputs (exact filenames in User Settings, write_*):
- logistic_regression_<tag>.joblib (scaler + fitted model + settings)
- logistic_regression_<tag>_coefficients.csv (one row per feature)
- logistic_regression_<tag>_predictions.csv (test set: true label,
  predicted label, per-class predicted probability, plus
  identifying_columns/position_columns for trace-back/plotting)
- printed classification report

Depends on:
- build_holdout_feature_dataset.py (writes the read_*_csv files)
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed_data"
MODEL_DIRECTORY = PROJECT_ROOT / "data" / "models"

# -------------------- Start User Settings --------------------
# TODO explicit input filenames (data/processed_data/).
read_train_csv = "features_all_train_held_out_fine_gross_shape_W_1s.csv"
read_test_csv = "features_all_test_held_out_fine_gross_shape_W_1s.csv"

# TODO explicit output filenames (all written to data/processed_data/).
write_model_file = "logistic_regression_held_out_fine_gross_shape_robot_W_1s.joblib"
write_coefficients_csv = "logistic_regression_held_out_fine_gross_shape_robot_W_1s_coefficients.csv"
write_predictions_csv = "logistic_regression_held_out_fine_gross_shape_robot_W_1s_predictions.csv"

# TODO the EXACT features to fit on -- nothing else is used, nothing is
# ranked/selected automatically.
feature_columns = [
    "gaze_gaze_path_efficiency",
    "gaze_p90_gaze_ee_distance",
    "robot_direction_concentration",
    "robot_std_curvature",
]

target_column = "trial_class_label"  # TODO label column, string values

# TODO metadata columns carried into the predictions CSV (not fit on) --
# lets a downstream plot order/group predictions by trial.
identifying_columns = [
    "subject", "bandwidth", "direction", "window_index", "window_center_s",
]

# TODO window position columns [m] (not fit on) -- each window's mean
# TCP position, carried into the predictions CSV so a downstream plot
# can place each window's probability on the shape.
position_columns = ["robot_mean_position_x", "robot_mean_position_y"]

random_seed = 42
max_iterations = 1000  # TODO sklearn LogisticRegression's max_iter
# -------------------- End User Settings --------------------


def main() -> None:
    start_time_s = time.perf_counter()
    train_path = PROCESSED_DATA_DIRECTORY / read_train_csv
    test_path = PROCESSED_DATA_DIRECTORY / read_test_csv
    model_path = MODEL_DIRECTORY / write_model_file
    coefficients_path = PROCESSED_DATA_DIRECTORY / write_coefficients_csv
    predictions_path = PROCESSED_DATA_DIRECTORY / write_predictions_csv
    print(f"Input:  {train_path}")
    print(f"Input:  {test_path}")
    print(f"Output: {model_path}")
    print(f"Output: {coefficients_path}")
    print(f"Output: {predictions_path}")
    print()

    train_table = pd.read_csv(train_path)
    test_table = pd.read_csv(test_path)

    missing = [f for f in feature_columns if f not in train_table.columns]
    if missing:
        raise ValueError(f"feature_columns not found in {read_train_csv}: {missing}")
    missing_identifying = [c for c in identifying_columns if c not in test_table.columns]
    if missing_identifying:
        raise ValueError(
            f"identifying_columns not found in {read_test_csv}: {missing_identifying}")
    missing_position = [c for c in position_columns if c not in test_table.columns]
    if missing_position:
        raise ValueError(
            f"position_columns not found in {read_test_csv}: {missing_position}")

    # Drop rows with any NaN in the chosen features/label -- logistic
    # regression cannot fit through missing values.
    required_columns = feature_columns + [target_column]
    train_clean = train_table.dropna(subset=required_columns)
    test_clean = test_table.dropna(subset=required_columns)
    print(f"Train: {len(train_clean)} of {len(train_table)} rows kept "
          f"({len(train_table) - len(train_clean)} dropped for NaN).")
    print(f"Test:  {len(test_clean)} of {len(test_table)} rows kept "
          f"({len(test_table) - len(test_clean)} dropped for NaN).")

    X_train = train_clean[feature_columns].to_numpy(dtype=float)
    X_test = test_clean[feature_columns].to_numpy(dtype=float)
    y_train = train_clean[target_column].to_numpy()
    y_test = test_clean[target_column].to_numpy()

    # Standardize -- REQUIRED so coefficients are comparable across
    # features of different scale/units (e.g. curvature ~1000s vs.
    # path_efficiency ~0-1).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=max_iterations, random_state=random_seed)
    model.fit(X_train_scaled, y_train)

    y_predicted = model.predict(X_test_scaled)
    y_probability = model.predict_proba(X_test_scaled)  # columns = model.classes_

    accuracy = accuracy_score(y_test, y_predicted)
    print(f"\nClasses (in coef_ order): {list(model.classes_)}")
    print(f"Test accuracy: {accuracy:.4f}  "
          f"(n_train={len(y_train)}, n_test={len(y_test)})\n")
    print(classification_report(y_test, y_predicted))

    # ---- Save trained model (scaler + fitted model together) ----
    joblib.dump({
        "scaler": scaler, "model": model, "feature_columns": feature_columns,
        "target_column": target_column, "classes": list(model.classes_),
    }, model_path)

    # ---- Save coefficients (interpretable since features are
    # standardized; positive = pushes toward model.classes_[1]) ----
    coefficients_table = pd.DataFrame({
        "feature": feature_columns,
        "standardized_coefficient": model.coef_[0],
    }).sort_values("standardized_coefficient", key=abs, ascending=False)
    coefficients_table.to_csv(coefficients_path, index=False)

    # ---- Save test-set predictions, for manual verification ----
    predictions_table = test_clean[
        feature_columns + [target_column] + identifying_columns
        + position_columns].copy()
    predictions_table["predicted_label"] = y_predicted
    for class_index, class_name in enumerate(model.classes_):
        predictions_table[f"probability_{class_name}"] = \
            y_probability[:, class_index]
    predictions_table.to_csv(predictions_path, index=False)

    elapsed_s = time.perf_counter() - start_time_s
    print(f"\nWrote:\n  {model_path.name}\n  {coefficients_path.name}\n"
          f"  {predictions_path.name}")
    print(f"Total time: {elapsed_s:.1f} s")


if __name__ == "__main__":
    main()
