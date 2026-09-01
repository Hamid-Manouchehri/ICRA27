"""tune_random_forest_loso.py -- Hyperparameter-tunes the random forest
against the LOSO (leave-one-subject-out) generalization test, not the
easier pooled-window split, using Optuna (TPE sampler). Two separate,
correctly-scoped searches -- mixing them would leak test/held-out data
into hyperparameter selection and inflate the reported numbers:

1. NESTED LOSO evaluation (the honest, unbiased number). For each of
   the 4 outer held-out-subject folds, the search itself only ever
   sees the OTHER 3 subjects: Optuna picks hyperparameters by k-fold
   CV *within those 3 subjects' windows*, the winning config is refit
   once on a balanced draw from those same 3 subjects (train_modality_
   classifiers_loso.py's own draw_balanced_train, reused directly so
   this isolates "tuned vs. default hyperparameters" as the only
   difference from that script's baseline), and ONLY THEN evaluated
   once on the truly-unseen 4th subject. The held-out predictions from
   all 4 outer folds are pooled into one report per modality --
   directly comparable to train_modality_classifiers_loso.py's
   loso_metrics_by_modality.csv, with tuning as the only difference.
   Because each outer fold gets its own search, this does NOT produce
   one universal "best" hyperparameter set -- see part 2 for that.

2. FINAL DEPLOYABLE MODEL per modality (one hyperparameter set to
   actually use, e.g. with predict_with_saved_model.py). Search is
   k-fold CV on ONLY the ORIGINAL features_<modality>_train.csv (the
   same training data train_modality_classifiers.py's baseline model
   used), refit once on that full training set, evaluated once on the
   original, untouched features_<modality>_test.csv. This reuses
   train_modality_classifiers.py's own train/test split, so its
   resulting accuracy is directly comparable to metrics_by_modality.csv
   -- but note its search and evaluation are NOT subject-disjoint (the
   pooled-window split mixes subjects across train/test, same caveat
   as the baseline model), so treat this number the way you'd treat
   the baseline pooled-split accuracy, not as a LOSO estimate.

Search budget: each study runs a FIXED TRIAL COUNT (inner_search_
trial_count / final_model_search_trial_count below), not a time limit
-- there is no time constraint on this run. Expect this to take
HOURS, not minutes: at the defaults (300 trials/study, ~4 CV fits/
trial), each study runs roughly as long as it did in the earlier
70-second-budget run's per-trial rate x 300 -- call it ~15-20 min/
study x 15 studies (12 nested + 3 final) is on the order of 4-5 hours.
Actual wall-clock is printed at the end regardless.

Output:
  - loso_tuned_fold_results_<modality>.csv (nested search: one row per
    outer-held-out subject, its chosen hyperparameters, inner-CV
    macro-F1, and test accuracy on that subject).
  - loso_accuracy_by_subject_<modality>_tuned.png / loso_confusion_
    matrix_<modality>_tuned.png (same style as train_modality_
    classifiers_loso.py's, via its shared plot_loso_results()).
  - loso_metrics_by_modality_tuned.csv (nested, pooled report per
    modality -- compare directly against loso_metrics_by_modality.csv).
  - model_<modality>_tuned.joblib (final deployable model -- same
    bundle format predict_with_saved_model.py already loads).
  - final_tuned_model_metrics.csv (final-model test-set report per
    modality, plus its chosen hyperparameters -- compare directly
    against metrics_by_modality.csv).
  - both comparisons (tuned vs. baseline, nested-LOSO and final-model)
    printed to the command window.

Dependencies: numpy, pandas, scikit-learn, matplotlib, optuna, joblib
(joblib ships with scikit-learn already)
  pip install numpy pandas scikit-learn matplotlib optuna
"""

import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- this script only saves PNGs
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

import train_modality_classifiers as tmc
import train_modality_classifiers_loso as loso

optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line/trial is too noisy

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESSED_DATA_DIRECTORY = DATA_DIRECTORY / "processed_data"
IMAGE_DATA_DIRECTORY = DATA_DIRECTORY / "images"

# -------------------- User Settings --------------------
# TODO Optuna search budget per study, as a trial count (no time
# limit). See the module docstring for the total-runtime estimate
# these two feed into.
inner_search_trial_count = 300
final_model_search_trial_count = 300
inner_cv_folds = 4  # k-fold CV splits used to score each Optuna trial

random_seed = 42


# ======================================================================
# Search space + one reusable Optuna study runner (used for both the
# per-outer-fold nested searches and the final-model search).
# ======================================================================

def sample_random_forest_params(trial: optuna.trial.Trial) -> dict:
    # max_features needs a CONDITIONAL suggestion ("fraction" -> also
    # suggest_float the actual fraction), since Optuna records whatever
    # trial.suggest_* calls actually ran under raw names (max_features_
    # kind / max_features_fraction) -- resolve_rf_params() below turns
    # that back into the single max_features value RandomForestClassifier
    # actually accepts. sqrt/log2 alone (~7-10 features/split here) left
    # a real gap before "all features"; a bounded 0.1-0.6 fraction fills
    # it without reintroducing max_features=None's 5-9x slowdown (see
    # this script's git history for the timing that ruled that out).
    max_features_kind = trial.suggest_categorical(
        "max_features_kind", ["sqrt", "log2", "fraction"])
    if max_features_kind == "fraction":
        max_features = trial.suggest_float("max_features_fraction", 0.1, 0.6)
    else:
        max_features = max_features_kind

    return dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
        max_depth=trial.suggest_categorical(
            "max_depth", [None, 5, 10, 15, 20, 30]),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=max_features,
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        class_weight=trial.suggest_categorical(
            "class_weight", [None, "balanced"]),
        bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
    )


def resolve_rf_params(raw_params: dict) -> dict:
    """study.best_params (and every trial's own suggested-parameter
    dict) carries max_features as the raw max_features_kind/
    max_features_fraction pair sample_random_forest_params() actually
    suggested, not a max_features key -- this converts that pair back
    into the single value RandomForestClassifier(**...) accepts, so
    every caller downstream can just use the result directly."""
    params = dict(raw_params)
    kind = params.pop("max_features_kind")
    fraction = params.pop("max_features_fraction", None)
    params["max_features"] = fraction if kind == "fraction" else kind
    return params


def run_inner_search(
        X: np.ndarray, y: np.ndarray, n_trials: int, cv_folds: int,
        seed: int, study_name: str) -> tuple:
    """K-fold CV (macro-F1) on (X, y) only -- caller is responsible for
    (X, y) not containing whatever it later evaluates on. Returns
    (best_params, best_cv_macro_f1, trial_count) with best_params
    already resolved (see resolve_rf_params) and ready to pass straight
    into RandomForestClassifier(**best_params, ...)."""

    def objective(trial: optuna.trial.Trial) -> float:
        params = resolve_rf_params(sample_random_forest_params(trial))
        splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed)
        fold_scores = []
        for train_idx, val_idx in splitter.split(X, y):
            model = RandomForestClassifier(
                **params, random_state=seed, n_jobs=-1)
            model.fit(X[train_idx], y[train_idx])
            predicted = model.predict(X[val_idx])
            fold_scores.append(
                f1_score(y[val_idx], predicted, average="macro"))
        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
        study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return resolve_rf_params(study.best_params), study.best_value, len(study.trials)


# ======================================================================
# Part 1: nested LOSO -- unbiased tuned-vs-baseline comparison.
# ======================================================================

def nested_loso_tune_modality(modality: str) -> dict:
    train_csv = pd.read_csv(tmc.modality_csv_paths[modality]["train"])
    test_csv = pd.read_csv(tmc.modality_csv_paths[modality]["test"])
    pooled_table = pd.concat([train_csv, test_csv], ignore_index=True)
    feature_columns = [
        c for c in pooled_table.columns
        if c not in tmc.metadata_columns and c != tmc.target_column]
    class_order = sorted(pooled_table[tmc.target_column].unique())
    subjects = sorted(int(s) for s in pooled_table["subject"].unique())

    print(f"\n{'=' * 70}\n{modality.upper()} -- nested LOSO tuning "
          f"({len(subjects)} outer folds, {inner_search_trial_count} inner "
          f"trials each)\n{'=' * 70}")

    fold_rows = []
    all_true, all_predicted = [], []
    for outer_subject in subjects:
        fold_start_s = time.time()
        outer_test = pooled_table[pooled_table["subject"] == outer_subject]
        remaining = pooled_table[pooled_table["subject"] != outer_subject]

        X_inner = remaining[feature_columns].to_numpy(dtype=float)
        y_inner = remaining[tmc.target_column].to_numpy()
        best_params, best_cv_score, n_trials = run_inner_search(
            X_inner, y_inner, inner_search_trial_count, inner_cv_folds,
            random_seed, f"{modality}_outer_s{outer_subject}")

        # Final refit for THIS outer fold: the same balanced-draw
        # training construction the untuned baseline uses, so tuning is
        # the only thing that differs from loso_metrics_by_modality.csv.
        train_table = loso.draw_balanced_train(remaining, random_seed)
        X_train = train_table[feature_columns].to_numpy(dtype=float)
        y_train = train_table[tmc.target_column].to_numpy()
        X_test = outer_test[feature_columns].to_numpy(dtype=float)
        y_test = outer_test[tmc.target_column].to_numpy()

        model = RandomForestClassifier(
            **best_params, random_state=random_seed, n_jobs=-1)
        model.fit(X_train, y_train)
        y_predicted = model.predict(X_test)
        fold_accuracy = float(np.mean(y_predicted == y_test))
        elapsed_s = time.time() - fold_start_s

        print(f"  outer s{outer_subject}: {n_trials} inner trials (CV "
              f"macro-F1={best_cv_score:.3f}) -> held-out accuracy="
              f"{fold_accuracy:.3f}  [{elapsed_s:.0f}s]")

        row = {
            "subject": f"s{outer_subject}", "accuracy": fold_accuracy,
            "inner_cv_macro_f1": best_cv_score, "inner_trials": n_trials,
            **best_params,
        }
        fold_rows.append(row)
        all_true.append(y_test)
        all_predicted.append(y_predicted)

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(
        PROCESSED_DATA_DIRECTORY /
        f"loso_tuned_fold_results_{modality}.csv", index=False)

    all_true = np.concatenate(all_true)
    all_predicted = np.concatenate(all_predicted)
    pooled_report = classification_report(
        all_true, all_predicted, labels=class_order, output_dict=True)

    loso.plot_loso_results(
        modality, fold_table["subject"], fold_table["accuracy"],
        all_true, all_predicted, class_order, pooled_report,
        filename_suffix="_tuned", title_suffix=" (tuned)")

    return tmc.flatten_classification_report(modality, pooled_report, class_order)


# ======================================================================
# Part 2: final deployable model -- one hyperparameter set per
# modality, tuned/evaluated on the ORIGINAL pooled-window train/test
# split (train_modality_classifiers.py's), never touching LOSO folds.
# ======================================================================

def tune_final_model_for_modality(modality: str) -> dict:
    train_table = pd.read_csv(tmc.modality_csv_paths[modality]["train"])
    test_table = pd.read_csv(tmc.modality_csv_paths[modality]["test"])
    feature_columns = [
        c for c in train_table.columns
        if c not in tmc.metadata_columns and c != tmc.target_column]
    class_order = sorted(train_table[tmc.target_column].unique())

    X_train = train_table[feature_columns].to_numpy(dtype=float)
    y_train = train_table[tmc.target_column].to_numpy()
    X_test = test_table[feature_columns].to_numpy(dtype=float)
    y_test = test_table[tmc.target_column].to_numpy()

    print(f"\n{modality}: final-model search (train-only CV, "
          f"{final_model_search_trial_count} trials)...")
    best_params, best_cv_score, n_trials = run_inner_search(
        X_train, y_train, final_model_search_trial_count, inner_cv_folds,
        random_seed, f"{modality}_final")

    model = RandomForestClassifier(
        **best_params, random_state=random_seed, n_jobs=-1)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    report = classification_report(
        y_test, y_predicted, labels=class_order, output_dict=True)
    print(f"  {n_trials} trials, train-CV macro-F1={best_cv_score:.3f} -> "
          f"test accuracy={report['accuracy']:.3f}")

    joblib.dump(
        {"model": model, "feature_columns": feature_columns,
         "classifier_name": "random_forest_tuned", "modality": modality,
         "hyperparameters": best_params},
        PROCESSED_DATA_DIRECTORY / f"model_{modality}_tuned.joblib")

    row = tmc.flatten_classification_report(modality, report, class_order)
    row["train_cv_macro_f1"] = best_cv_score
    for key, value in best_params.items():
        row[f"param_{key}"] = value
    return row


def main() -> None:
    start_s = time.time()

    print(f"{'#' * 70}\nPart 1: nested LOSO tuning (unbiased tuned-vs-"
          f"baseline comparison)\n{'#' * 70}")
    nested_rows = [
        nested_loso_tune_modality(modality) for modality in tmc.modality_csv_paths]
    nested_table = pd.DataFrame(nested_rows)
    nested_path = PROCESSED_DATA_DIRECTORY / "loso_metrics_by_modality_tuned.csv"
    nested_table.to_csv(nested_path, index=False)

    print(f"\n{'=' * 70}\nNested-LOSO summary (also written to "
          f"{nested_path.name})\n{'=' * 70}")
    print(nested_table.to_string(index=False))

    baseline_loso_path = PROCESSED_DATA_DIRECTORY / "loso_metrics_by_modality.csv"
    if baseline_loso_path.is_file():
        baseline_loso = pd.read_csv(baseline_loso_path)
        comparison = baseline_loso[["modality", "accuracy", "macro_f1"]].merge(
            nested_table[["modality", "accuracy", "macro_f1"]],
            on="modality", suffixes=("_baseline", "_tuned"))
        print(f"\nBaseline LOSO vs. tuned LOSO:")
        print(comparison.to_string(index=False))
    else:
        print(f"\n({baseline_loso_path.name} not found -- run "
              f"train_modality_classifiers_loso.py for a baseline "
              f"comparison too.)")

    print(f"\n{'#' * 70}\nPart 2: final deployable model per modality\n"
          f"{'#' * 70}")
    final_rows = [
        tune_final_model_for_modality(modality)
        for modality in tmc.modality_csv_paths]
    final_table = pd.DataFrame(final_rows)
    final_path = PROCESSED_DATA_DIRECTORY / "final_tuned_model_metrics.csv"
    final_table.to_csv(final_path, index=False)

    print(f"\n{'=' * 70}\nFinal-model summary (also written to "
          f"{final_path.name})\n{'=' * 70}")
    print(final_table.to_string(index=False))

    baseline_path = PROCESSED_DATA_DIRECTORY / "metrics_by_modality.csv"
    if baseline_path.is_file():
        baseline = pd.read_csv(baseline_path)
        comparison = baseline[["modality", "accuracy", "macro_f1"]].merge(
            final_table[["modality", "accuracy", "macro_f1"]],
            on="modality", suffixes=("_baseline", "_tuned"))
        print(f"\nBaseline (default RF) vs. tuned final model "
              f"(pooled-window split):")
        print(comparison.to_string(index=False))
    else:
        print(f"\n({baseline_path.name} not found -- run "
              f"train_modality_classifiers.py for a baseline comparison "
              f"too.)")

    elapsed_minutes = (time.time() - start_s) / 60
    print(f"\nTotal wall-clock time: {elapsed_minutes:.1f} minutes.")


if __name__ == "__main__":
    main()
