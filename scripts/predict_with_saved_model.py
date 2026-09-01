"""predict_with_saved_model.py -- Classify feature windows with an
ALREADY-TRAINED model that train_modality_classifiers.py saved
(model_<modality>.joblib) -- no retraining, no ground truth required.

Point it at any feature CSV built the same way build_modality_feature_
datasets.py builds one (same columns -- e.g. a new subject's trial run
through that script) and it predicts Fine/Gross per window, plus each
class's probability (a random forest gives this for free -- useful for
a continuous damping-modulation signal, not just a hard label). If the
CSV happens to also carry the true trial_class_label column (e.g.
you're testing on a subject you DO have ground truth for, just one
build_modality_feature_datasets.py hadn't seen when the model was
trained), this also prints accuracy/precision/recall/F1 for reference
-- otherwise it just predicts, no scoring.

The saved .joblib is a small bundle (model + the exact feature_columns
list/order it was trained on), not a bare model -- a bare model has no
memory of column names, so feeding it a DataFrame's columns in the
wrong order would silently produce wrong predictions with no error.
This script selects/orders the new CSV's columns from that saved list
and fails loudly if any are missing, instead.

Dependencies: numpy, pandas, scikit-learn, joblib (joblib ships with
scikit-learn already, no separate install needed)
  pip install numpy pandas scikit-learn
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"

# -------------------- User Settings --------------------
# TODO which modality's saved model to use, and which feature CSV to
# classify (built the same way build_modality_feature_datasets.py
# builds features_<modality>_test.csv -- same columns, any subject).
modality = "robot"
model_path = PROCESSED_DATA_DIRECTORY / f"model_{modality}.joblib"
input_feature_csv_path = \
    PROCESSED_DATA_DIRECTORY / f"features_{modality}_test.csv"

# Only used if this column is present in the input CSV -- scores the
# predictions against it for reference. Leave it out of a CSV with no
# ground truth (e.g. a brand-new, unlabeled subject) and this script
# just predicts, no scoring attempted.
target_column = "trial_class_label"

write_predictions_csv_path = \
    PROCESSED_DATA_DIRECTORY / f"predictions_new_{modality}.csv"


def main() -> None:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    print(f"Loaded {bundle['modality']} model "
          f"({bundle['classifier_name']}, {len(feature_columns)} features) "
          f"from {model_path.name}")

    table = pd.read_csv(input_feature_csv_path)
    missing_columns = [c for c in feature_columns if c not in table.columns]
    if missing_columns:
        raise ValueError(
            f"{input_feature_csv_path.name} is missing "
            f"{len(missing_columns)} feature column(s) this model was "
            f"trained on: {missing_columns[:10]}"
            f"{' ...' if len(missing_columns) > 10 else ''} -- was it "
            f"built with the same modality/version of "
            f"build_modality_feature_datasets.py?")

    X = table[feature_columns].to_numpy(dtype=float)
    predicted_labels = model.predict(X)

    output = table.copy()
    output["predicted_label"] = predicted_labels
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        for class_name, column in zip(model.classes_, probabilities.T):
            output[f"probability_{class_name}"] = column

    output.to_csv(write_predictions_csv_path, index=False)
    print(f"\nWrote {len(output)} predictions to "
          f"{write_predictions_csv_path.name}")

    if target_column in table.columns:
        true_labels = table[target_column].to_numpy()
        print(f"\n'{target_column}' found in the input -- scoring "
              f"predictions against it for reference:\n")
        print(classification_report(true_labels, predicted_labels))
    else:
        print(f"\n(No '{target_column}' column in the input -- treating "
              f"this as genuinely unlabeled data. No scoring done.)")


if __name__ == "__main__":
    main()
