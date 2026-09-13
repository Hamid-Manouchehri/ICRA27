"""Regenerate every number quoted in the HRI paper into tidy per-figure and
per-table CSVs, plus a claims file mapping each reported value back to its
source file, column, and row filter.

Inputs:
  - data/processed_data/features_robot_{train,test}_<tag>.csv
  - data/processed_data/loso_feature_set_folds_<tag>.csv
  - data/processed_data/loso_lead_time_sweep_folds_<tag>.csv
  - data/processed_data/losho_fold_results_{robot,all}_<tag>.csv
  - data/raw_data/ur5e_chen_var_adm_2d_*.csv                  (fig5 only)
  - data/processed_data/tobii_chen_var_adm_2d_live_labeled_*.csv (fig5 only)

Outputs:
  - data/processed_data/paper_fig3_accuracy_<tag>.csv
  - data/processed_data/paper_fig4_lead_time_<tag>.csv
  - data/processed_data/paper_fig5_chen_damping_hist_<tag>.csv
  - data/processed_data/paper_fig5_chen_auc_<tag>.csv
  - data/processed_data/paper_tab{1,2,3}_*_<tag>.csv
  - data/processed_data/paper_tab{1,2,3}_*_<tag>.tex
  - data/processed_data/paper_claims_<tag>.csv
  - data/json/build_paper_results_<timestamp>.json

Depends on:
  - scripts/build_modality_feature_datasets.py (writes features_* CSVs)
  - scripts/compare_loso_feature_sets.py (writes loso_* CSVs)
  - scripts/losho_classifier.py (writes losho_fold_results_* CSVs)
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score

# -------------------- User Settings --------------------

# TODO which paper outputs to regenerate. Comment out what you are not
# changing; each entry writes its own CSV and is independent of the rest.
# "fig5" reads ~5M raw samples and is the slow one (roughly a minute).
outputs_to_build = ("tab1", "tab2", "tab3", "fig3", "fig4", "fig5")

# TODO dataset tag, must match output_dataset_name in
# build_modality_feature_datasets.py. Every input and output carries it.
dataset_name = "W_2s"

# TODO subjects expected in every analysis CSV. A mismatch is a hard
# error, not a warning: it means an input was rebuilt with a different
# cohort. s18 was excluded from the study.
expected_subjects = (10, 11, 12, 13, 14, 15, 16, 17)

# TODO window step [s], must match build_modality_feature_datasets.py.
# Converts lead_time_s into an integer window offset.
window_step_s = 2.0

# TODO window size [s], must match build_modality_feature_datasets.py.
# When window_step_s >= window_size_s the windows are contiguous and
# non-overlapping, which forces the persistence baseline to 1.0 at
# offsets 0 and 1 (tautology, then purity-filter artifact).
window_size_s = 2.0

# TODO expected feature count per feature set after confound exclusion,
# guarding against an input rebuilt with different exclusion toggles.
expected_feature_counts = {
    "robot": 57, "gaze": 70, "gaze_pure": 41, "joint": 45,
    "robot_plus_gaze": 127, "robot_plus_gaze_pure": 98, "all": 172,
}

# TODO feature sets carried into the paper, in display order. Dropping
# one here drops it from every table and figure source at once.
feature_sets_for_paper = (
    "robot", "gaze", "gaze_pure", "joint", "robot_plus_gaze",
    "robot_plus_gaze_top20",
)

# TODO damping histogram bin width [N s/m] for fig5, positive.
damping_bin_width_n_s_m = 5.0

# TODO max time gap [s] when matching a label sample to a robot sample
# in fig5, positive. Larger values accept sloppier alignment.
label_match_tolerance_s = 0.05

# TODO decimal places in the generated .tex tables.
decimal_places = 2

# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed_data"
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw_data"
JSON_DIRECTORY = PROJECT_ROOT / "data" / "json"

read_loso_folds_path = (PROCESSED_DATA_DIRECTORY /
                        f"loso_feature_set_folds_{dataset_name}.csv")
read_lead_folds_path = (PROCESSED_DATA_DIRECTORY /
                        f"loso_lead_time_sweep_folds_{dataset_name}.csv")
read_losho_paths = {
    modality: (PROCESSED_DATA_DIRECTORY /
               f"losho_fold_results_{modality}_{dataset_name}.csv")
    for modality in ("robot", "all")
}
read_feature_paths = {
    split: (PROCESSED_DATA_DIRECTORY /
            f"features_robot_{split}_{dataset_name}.csv")
    for split in ("train", "test")
}

write_paths = {
    "fig3": PROCESSED_DATA_DIRECTORY /
            f"paper_fig3_accuracy_{dataset_name}.csv",
    "fig4": PROCESSED_DATA_DIRECTORY /
            f"paper_fig4_lead_time_{dataset_name}.csv",
    "fig5_hist": PROCESSED_DATA_DIRECTORY /
                 f"paper_fig5_chen_damping_hist_{dataset_name}.csv",
    "fig5_auc": PROCESSED_DATA_DIRECTORY /
                f"paper_fig5_chen_auc_{dataset_name}.csv",
    "tab1": PROCESSED_DATA_DIRECTORY /
            f"paper_tab1_participants_{dataset_name}.csv",
    "tab2": PROCESSED_DATA_DIRECTORY /
            f"paper_tab2_main_results_{dataset_name}.csv",
    "tab3": PROCESSED_DATA_DIRECTORY /
            f"paper_tab3_lead_time_stats_{dataset_name}.csv",
    "claims": PROCESSED_DATA_DIRECTORY /
              f"paper_claims_{dataset_name}.csv",
}

METADATA_COLUMNS = [
    "subject", "bandwidth", "shape", "direction", "window_index",
    "window_start_s", "window_end_s", "window_center_s",
    "gaze_sample_count", "robot_sample_count", "trial_class_label",
]
TRIAL_KEY_COLUMNS = ["subject", "bandwidth", "shape", "direction"]
TRIAL_NAME_PATTERN = re.compile(
    r"ur5e_chen_var_adm_2d_(?P<bandwidth>narrow|wide)_(?P<shape>.+)_shape_"
    r"(?P<direction>counter_clockwise|clockwise)_s(?P<subject>\d+)\.csv"
)

claim_rows = []


def check(condition: bool, message: str) -> None:
    """Raise with a readable message when a structural assumption fails.

    Inputs:
      - condition: expression that must hold.
      - message: what was expected, shown verbatim on failure.
    Outputs:
      - None. Raises AssertionError when condition is false.
    """
    if not condition:
        raise AssertionError(f"STRUCTURAL CHECK FAILED: {message}")


def record_claim(claim_id: str, description: str, value, unit: str,
                 source_csv: Path, source_column: str,
                 source_filter: str = "") -> None:
    """Append one paper-quotable number to the traceability table.

    Inputs:
      - claim_id: short stable key cited from the manuscript.
      - description: what the number means, in one line.
      - value, unit: the number as it should be quoted.
      - source_csv, source_column, source_filter: where to re-read it.
    Outputs:
      - None. Appends to the module-level claim_rows list.
    """
    claim_rows.append({
        "claim_id": claim_id,
        "description": description,
        "value": value,
        "unit": unit,
        "source_csv": source_csv.name,
        "source_column": source_column,
        "source_filter": source_filter,
        "script": Path(__file__).name,
    })


def window_offset_for_lead(lead_time_s: float) -> int:
    """Window steps a label is shifted forward for a given lead time.

    Inputs:
      - lead_time_s: prediction horizon [s].
    Outputs:
      - Integer offset, matching compare_loso_feature_sets.py.
    """
    return int(round(lead_time_s / window_step_s))


def persistence_is_degenerate(lead_time_s: float) -> bool:
    """Whether the persistence baseline carries no information here.

    Inputs:
      - lead_time_s: prediction horizon [s].
    Outputs:
      - True at offset 0 (label scored against itself) and at offset 1
        when windows are contiguous and non-overlapping, where the
        purity filter removes every pair that could have disagreed.
    """
    offset = window_offset_for_lead(lead_time_s)
    if offset == 0:
        return True
    return offset == 1 and window_step_s >= window_size_s


def paired_test(first: np.ndarray, second: np.ndarray) -> tuple:
    """Paired Wilcoxon and t-test p-values, NaN when undefined.

    Inputs:
      - first, second: equal-length paired measurement arrays.
    Outputs:
      - (wilcoxon_p, ttest_p), each float or NaN if the test is not
        defined (e.g. every difference is exactly zero).
    """
    try:
        wilcoxon_p = float(wilcoxon(first, second).pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    try:
        ttest_p = float(ttest_rel(first, second).pvalue)
    except ValueError:
        ttest_p = float("nan")
    return wilcoxon_p, ttest_p


def write_latex_table(table: pd.DataFrame, path: Path, caption: str,
                      label: str) -> None:
    """Write a booktabs tabular built from exactly the given DataFrame.

    Inputs:
      - table: the rows and columns to typeset, already ordered.
      - path: .tex file to write.
      - caption, label: LaTeX caption text and \\label key.
    Outputs:
      - Writes path. Numbers use decimal_places; underscores escaped.
    """
    number_format = f"{{:.{decimal_places}f}}"
    alignment = "l" + "r" * (len(table.columns) - 1)
    header = " & ".join(str(c).replace("_", r"\_") for c in table.columns)
    lines = [
        "% Generated by scripts/build_paper_results.py -- do not hand edit.",
        "\\begin{table}[t]", "\\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{alignment}}}", "\\toprule",
        header + r" \\", "\\midrule",
    ]
    for _, row in table.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append("--" if math.isnan(float(value))
                             else number_format.format(float(value)))
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            else:
                cells.append(str(value).replace("_", r"\_"))
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def load_window_metadata() -> pd.DataFrame:
    """Window-level metadata for every built window, train and test.

    Inputs:
      - features_robot_{train,test}_<tag>.csv (metadata columns only).
    Outputs:
      - DataFrame with METADATA_COLUMNS plus a "split" column.
    """
    frames = []
    for split, path in read_feature_paths.items():
        frame = pd.read_csv(path, usecols=METADATA_COLUMNS)
        frame["split"] = split
        frames.append(frame)
    windows = pd.concat(frames, ignore_index=True)
    found = tuple(sorted(windows["subject"].unique()))
    check(found == expected_subjects,
          f"subjects in features_robot_* are {found}, "
          f"expected {expected_subjects}")
    return windows


def build_tab1(windows: pd.DataFrame) -> pd.DataFrame:
    """Per-participant trial, window, and class counts.

    Inputs:
      - windows: output of load_window_metadata().
    Outputs:
      - DataFrame written to paper_tab1_participants_<tag>.csv and .tex.
    """
    rows = []
    for subject, group in windows.groupby("subject"):
        fine = int((group["trial_class_label"] == "Fine").sum())
        gross = int((group["trial_class_label"] == "Gross").sum())
        check(fine + gross == len(group),
              f"subject {subject} has labels outside Fine/Gross")
        rows.append({
            "subject": int(subject),
            "n_trials": int(group[TRIAL_KEY_COLUMNS].drop_duplicates()
                            .shape[0]),
            "n_windows": int(len(group)),
            "fine_windows": fine,
            "gross_windows": gross,
            "fine_fraction": fine / len(group),
        })
    table = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    total_fine = int(table["fine_windows"].sum())
    total_gross = int(table["gross_windows"].sum())
    total_windows = int(table["n_windows"].sum())
    check(total_fine + total_gross == total_windows,
          "tab1 class counts do not sum to the window total")
    check(total_windows == len(windows),
          "tab1 window total does not match the built window count")

    table.to_csv(write_paths["tab1"], index=False)
    write_latex_table(
        table, write_paths["tab1"].with_suffix(".tex"),
        caption="Participants, trials, and labelled windows.",
        label="tab:participants")

    record_claim("n_subjects", "Participants analysed", len(table),
                 "count", write_paths["tab1"], "subject", "all rows")
    record_claim("n_windows_total", "Labelled windows analysed",
                 total_windows, "count", write_paths["tab1"],
                 "n_windows", "sum over rows")
    record_claim("n_windows_fine", "Fine-labelled windows", total_fine,
                 "count", write_paths["tab1"], "fine_windows",
                 "sum over rows")
    record_claim("n_windows_gross", "Gross-labelled windows", total_gross,
                 "count", write_paths["tab1"], "gross_windows",
                 "sum over rows")
    record_claim("majority_class_fraction",
                 "Majority (Gross) share of all windows",
                 100.0 * total_gross / total_windows, "%",
                 write_paths["tab1"], "gross_windows",
                 "sum(gross_windows)/sum(n_windows)")
    return table


def load_loso_folds() -> pd.DataFrame:
    """Per-subject LOSO fold results for every feature set.

    Inputs:
      - loso_feature_set_folds_<tag>.csv.
    Outputs:
      - DataFrame, one row per (feature_set, subject), validated.
    """
    folds = pd.read_csv(read_loso_folds_path)
    found = tuple(sorted(folds["subject"].unique()))
    check(found == expected_subjects,
          f"LOSO folds cover subjects {found}, expected {expected_subjects}")
    counts = folds.groupby("feature_set")["subject"].nunique()
    check((counts == len(expected_subjects)).all(),
          "every feature set must have exactly one fold per subject")
    duplicated = folds.duplicated(["feature_set", "subject"]).sum()
    check(duplicated == 0,
          f"{duplicated} duplicate (feature_set, subject) rows in LOSO folds")
    check(folds["accuracy"].between(0.0, 1.0).all(),
          "LOSO accuracies must lie in [0, 1]")
    for feature_set, expected in expected_feature_counts.items():
        subset = folds[folds["feature_set"] == feature_set]
        if subset.empty:
            continue
        found_counts = sorted(subset["feature_count"].unique())
        check(found_counts == [expected],
              f"{feature_set} has feature_count {found_counts}, "
              f"expected [{expected}] -- confound exclusion may differ")
    return folds


def build_tab2(folds: pd.DataFrame) -> pd.DataFrame:
    """LOSO accuracy summary per feature set.

    Inputs:
      - folds: output of load_loso_folds().
    Outputs:
      - DataFrame written to paper_tab2_main_results_<tag>.csv and .tex.
    """
    rows = []
    for feature_set in feature_sets_for_paper:
        subset = folds[folds["feature_set"] == feature_set]
        check(not subset.empty,
              f"feature set '{feature_set}' is missing from the LOSO folds")
        rows.append({
            "feature_set": feature_set,
            "n_features": int(subset["feature_count"].iloc[0]),
            "loso_accuracy_mean": 100.0 * subset["accuracy"].mean(),
            "loso_accuracy_sd": 100.0 * subset["accuracy"].std(ddof=1),
            "loso_accuracy_worst": 100.0 * subset["accuracy"].min(),
            "majority_baseline_mean":
                100.0 * subset["majority_baseline"].mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(write_paths["tab2"], index=False)
    write_latex_table(
        table, write_paths["tab2"].with_suffix(".tex"),
        caption="Leave-one-subject-out accuracy by feature set (\\%).",
        label="tab:main-results")

    for _, row in table.iterrows():
        record_claim(
            f"loso_{row['feature_set']}_accuracy",
            f"LOSO accuracy, {row['feature_set']} features",
            row["loso_accuracy_mean"], "%", write_paths["tab2"],
            "loso_accuracy_mean", f"feature_set == {row['feature_set']}")
    record_claim("loso_majority_baseline", "Mean per-fold majority baseline",
                 table["majority_baseline_mean"].iloc[0], "%",
                 write_paths["tab2"], "majority_baseline_mean",
                 "any row (identical across feature sets)")
    return table


def build_fig3(folds: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-fold accuracy for LOSO and LOSHO in one long table.

    Inputs:
      - folds: output of load_loso_folds().
      - losho_fold_results_{robot,all}_<tag>.csv.
    Outputs:
      - DataFrame written to paper_fig3_accuracy_<tag>.csv, one row per
        (scheme, feature_set, fold_id).
    """
    rows = []
    for _, row in folds[folds["feature_set"]
                        .isin(feature_sets_for_paper)].iterrows():
        rows.append({
            "scheme": "LOSO", "feature_set": row["feature_set"],
            "fold_id": f"s{int(row['subject'])}",
            "accuracy": 100.0 * row["accuracy"],
            "majority_baseline": 100.0 * row["majority_baseline"],
            "n_features": int(row["feature_count"]),
        })
    for modality, path in read_losho_paths.items():
        losho = pd.read_csv(path)
        shapes = tuple(sorted(losho["shape"].unique()))
        check(shapes == ("ellipse", "fine_gross", "pear"),
              f"LOSHO shapes are {shapes}, expected the three task shapes")
        check(losho["accuracy"].between(0.0, 1.0).all(),
              f"LOSHO accuracies in {path.name} must lie in [0, 1]")
        for _, row in losho.iterrows():
            rows.append({
                "scheme": "LOSHO", "feature_set": modality,
                "fold_id": row["shape"],
                "accuracy": 100.0 * row["accuracy"],
                "majority_baseline": 100.0 * row["majority_baseline"],
                "n_features": expected_feature_counts.get(modality, -1),
            })
    table = pd.DataFrame(rows)
    table.to_csv(write_paths["fig3"], index=False)

    losho_view = table[table["scheme"] == "LOSHO"]
    for _, row in losho_view.iterrows():
        record_claim(
            f"losho_{row['feature_set']}_{row['fold_id']}_accuracy",
            f"LOSHO accuracy, {row['feature_set']} features, "
            f"held-out shape {row['fold_id']}",
            row["accuracy"], "%", write_paths["fig3"], "accuracy",
            f"scheme == LOSHO and feature_set == {row['feature_set']} "
            f"and fold_id == {row['fold_id']}")
    return table


def build_fig4_and_tab3() -> tuple:
    """Lead-time results: tidy per-fold rows plus a per-horizon summary.

    Inputs:
      - loso_lead_time_sweep_folds_<tag>.csv.
    Outputs:
      - (fig4_table, tab3_table), written to paper_fig4_lead_time_<tag>.csv
        and paper_tab3_lead_time_stats_<tag>.csv/.tex. Both carry a
        persistence_degenerate flag so the 0 s and 2 s persistence values
        are never plotted or quoted as a real baseline.
    """
    folds = pd.read_csv(read_lead_folds_path)
    found = tuple(sorted(folds["subject"].unique()))
    check(found == expected_subjects,
          f"lead-time folds cover subjects {found}, "
          f"expected {expected_subjects}")
    check(folds["accuracy"].between(0.0, 1.0).all(),
          "lead-time accuracies must lie in [0, 1]")

    fig4_rows = []
    for _, row in folds.iterrows():
        lead_time_s = float(row["lead_time_s"])
        fig4_rows.append({
            "feature_set": row["base_feature_set"],
            "lead_time_s": lead_time_s,
            "window_offset": window_offset_for_lead(lead_time_s),
            "subject": int(row["subject"]),
            "accuracy": 100.0 * row["accuracy"],
            "majority_baseline": 100.0 * row["majority_baseline"],
            "persistence_baseline": 100.0 * row["persistence_baseline"],
            "persistence_degenerate": persistence_is_degenerate(lead_time_s),
            "test_windows": int(row["test_windows"]),
        })
    fig4_table = pd.DataFrame(fig4_rows)
    fig4_table.to_csv(write_paths["fig4"], index=False)

    degenerate = fig4_table[fig4_table["persistence_degenerate"]]
    check(np.allclose(degenerate["persistence_baseline"], 100.0),
          "a horizon flagged persistence_degenerate did not score exactly "
          "100% -- re-derive persistence_is_degenerate() for these windows")

    summary_rows = []
    for lead_time_s in sorted(fig4_table["lead_time_s"].unique()):
        at_lead = fig4_table[fig4_table["lead_time_s"] == lead_time_s]
        robot = (at_lead[at_lead["feature_set"] == "robot"]
                 .set_index("subject")["accuracy"])
        every = (at_lead[at_lead["feature_set"] == "all"]
                 .set_index("subject")["accuracy"])
        shared = robot.index.intersection(every.index)
        robot, every = robot[shared], every[shared]
        gain = every - robot
        gaze_wilcoxon_p, gaze_ttest_p = paired_test(every.to_numpy(),
                                                    robot.to_numpy())
        is_degenerate = bool(at_lead["persistence_degenerate"].iloc[0])
        row = {
            "lead_time_s": lead_time_s,
            "window_offset": window_offset_for_lead(lead_time_s),
            "n_subjects": int(len(shared)),
            "robot_accuracy_mean": float(robot.mean()),
            "robot_accuracy_sd": float(robot.std(ddof=1)),
            "all_accuracy_mean": float(every.mean()),
            "all_accuracy_sd": float(every.std(ddof=1)),
            "gaze_gain_mean": float(gain.mean()),
            "gaze_gain_sd": float(gain.std(ddof=1)),
            "n_subjects_improved": int((gain > 0).sum()),
            "gaze_wilcoxon_p": gaze_wilcoxon_p,
            "gaze_ttest_p": gaze_ttest_p,
            "persistence_degenerate": is_degenerate,
            "persistence_mean": float(at_lead["persistence_baseline"].mean()),
        }
        for name in ("robot", "all"):
            subset = at_lead[at_lead["feature_set"] == name]
            delta = (subset["accuracy"] -
                     subset["persistence_baseline"]).to_numpy()
            if is_degenerate:
                row[f"{name}_vs_persistence_mean"] = float("nan")
                row[f"{name}_vs_persistence_p"] = float("nan")
            else:
                p_value, _ = paired_test(
                    subset["accuracy"].to_numpy(),
                    subset["persistence_baseline"].to_numpy())
                row[f"{name}_vs_persistence_mean"] = float(delta.mean())
                row[f"{name}_vs_persistence_p"] = p_value
        summary_rows.append(row)

    tab3_table = pd.DataFrame(summary_rows)
    tab3_table.to_csv(write_paths["tab3"], index=False)
    tex_columns = [
        "lead_time_s", "robot_accuracy_mean", "all_accuracy_mean",
        "gaze_gain_mean", "n_subjects_improved", "gaze_wilcoxon_p",
        "robot_vs_persistence_mean", "robot_vs_persistence_p",
    ]
    write_latex_table(
        tab3_table[tex_columns], write_paths["tab3"].with_suffix(".tex"),
        caption="Prediction accuracy (\\%) by lead time, with paired tests "
                "across participants. Persistence comparisons are omitted "
                "where the baseline is degenerate.",
        label="tab:lead-time")

    for _, row in tab3_table.iterrows():
        lead = int(row["lead_time_s"])
        record_claim(f"lead{lead}s_robot_accuracy",
                     f"Accuracy at {lead} s lead, robot features",
                     row["robot_accuracy_mean"], "%", write_paths["tab3"],
                     "robot_accuracy_mean", f"lead_time_s == {lead}")
        record_claim(f"lead{lead}s_all_accuracy",
                     f"Accuracy at {lead} s lead, all features",
                     row["all_accuracy_mean"], "%", write_paths["tab3"],
                     "all_accuracy_mean", f"lead_time_s == {lead}")
        record_claim(f"lead{lead}s_gaze_gain",
                     f"Gaze gain at {lead} s lead (all minus robot)",
                     row["gaze_gain_mean"], "pts", write_paths["tab3"],
                     "gaze_gain_mean", f"lead_time_s == {lead}")
        record_claim(f"lead{lead}s_gaze_wilcoxon_p",
                     f"Paired Wilcoxon p for the gaze gain at {lead} s",
                     row["gaze_wilcoxon_p"], "p", write_paths["tab3"],
                     "gaze_wilcoxon_p", f"lead_time_s == {lead}")
        if not row["persistence_degenerate"]:
            record_claim(
                f"lead{lead}s_robot_vs_persistence",
                f"Robot advantage over persistence at {lead} s lead",
                row["robot_vs_persistence_mean"], "pts",
                write_paths["tab3"], "robot_vs_persistence_mean",
                f"lead_time_s == {lead}")
            record_claim(
                f"lead{lead}s_robot_vs_persistence_p",
                f"Paired Wilcoxon p vs persistence at {lead} s lead",
                row["robot_vs_persistence_p"], "p", write_paths["tab3"],
                "robot_vs_persistence_p", f"lead_time_s == {lead}")
    return fig4_table, tab3_table


def build_fig5() -> tuple:
    """Chen's commanded damping as an implicit fine/gross detector.

    Inputs:
      - data/raw_data/ur5e_chen_var_adm_2d_*.csv (read-only).
      - matching tobii_chen_var_adm_2d_live_labeled_*.csv label files.
    Outputs:
      - (histogram_table, auc_table), written to
        paper_fig5_chen_damping_hist_<tag>.csv (counts per damping bin
        per label) and paper_fig5_chen_auc_<tag>.csv (per-trial AUC).
    """
    bin_edges = np.arange(0.0, 200.0 + damping_bin_width_n_s_m,
                          damping_bin_width_n_s_m)
    bin_centers = bin_edges[:-1] + damping_bin_width_n_s_m / 2.0
    counts = {"Fine": np.zeros(len(bin_centers), dtype=np.int64),
              "Gross": np.zeros(len(bin_centers), dtype=np.int64)}
    auc_rows = []

    for robot_path in sorted(RAW_DATA_DIRECTORY.glob(
            "ur5e_chen_var_adm_2d_*.csv")):
        match = TRIAL_NAME_PATTERN.fullmatch(robot_path.name)
        if match is None:
            continue
        subject = int(match.group("subject"))
        if subject not in expected_subjects:
            continue
        label_path = (PROCESSED_DATA_DIRECTORY / robot_path.name.replace(
            "ur5e_chen_var_adm_2d_",
            "tobii_chen_var_adm_2d_live_labeled_"))
        if not label_path.exists():
            print(f"  no label file for {robot_path.name}, skipping")
            continue

        robot = pd.read_csv(robot_path,
                            usecols=["time_s", "bd_applied_Ns_m"])
        robot = robot.dropna().sort_values("time_s")
        labels = pd.read_csv(label_path, usecols=["time_s", "label"])
        labels = labels.dropna().sort_values("time_s")
        labels = labels[labels["label"].isin(["Fine", "Gross"])]
        if labels.empty:
            continue
        merged = pd.merge_asof(
            robot, labels, on="time_s", direction="nearest",
            tolerance=label_match_tolerance_s).dropna(subset=["label"])
        if merged["label"].nunique() < 2:
            continue

        for label in ("Fine", "Gross"):
            values = merged.loc[merged["label"] == label, "bd_applied_Ns_m"]
            counts[label] += np.histogram(values, bins=bin_edges)[0]

        is_fine = (merged["label"] == "Fine").astype(int)
        auc_rows.append({
            "trial": robot_path.stem,
            "subject": subject,
            "bandwidth": match.group("bandwidth"),
            "shape": match.group("shape"),
            "direction": match.group("direction"),
            "auc": float(roc_auc_score(is_fine,
                                       merged["bd_applied_Ns_m"])),
            "n_samples": int(len(merged)),
            "mean_bd_fine_n_s_m": float(
                merged.loc[merged["label"] == "Fine",
                           "bd_applied_Ns_m"].mean()),
            "mean_bd_gross_n_s_m": float(
                merged.loc[merged["label"] == "Gross",
                           "bd_applied_Ns_m"].mean()),
        })

    histogram_table = pd.DataFrame([
        {"label": label, "bd_bin_center_n_s_m": center, "count": int(count)}
        for label in ("Fine", "Gross")
        for center, count in zip(bin_centers, counts[label])
    ])
    histogram_table.to_csv(write_paths["fig5_hist"], index=False)

    auc_table = pd.DataFrame(auc_rows).sort_values(
        ["subject", "bandwidth", "shape", "direction"]).reset_index(drop=True)
    auc_table.to_csv(write_paths["fig5_auc"], index=False)
    check(auc_table["auc"].between(0.0, 1.0).all(),
          "per-trial AUC values must lie in [0, 1]")

    fine_total = int(histogram_table.loc[histogram_table["label"] == "Fine",
                                         "count"].sum())
    gross_total = int(histogram_table.loc[histogram_table["label"] == "Gross",
                                          "count"].sum())
    weighted_fine = float(
        (auc_table["mean_bd_fine_n_s_m"] * auc_table["n_samples"]).sum() /
        auc_table["n_samples"].sum())
    weighted_gross = float(
        (auc_table["mean_bd_gross_n_s_m"] * auc_table["n_samples"]).sum() /
        auc_table["n_samples"].sum())

    record_claim("chen_auc_mean",
                 "Per-trial AUC of Chen commanded damping vs hand labels",
                 float(auc_table["auc"].mean()), "AUC",
                 write_paths["fig5_auc"], "auc", "mean over rows")
    record_claim("chen_auc_sd", "SD of per-trial Chen damping AUC",
                 float(auc_table["auc"].std(ddof=1)), "AUC",
                 write_paths["fig5_auc"], "auc", "std over rows")
    record_claim("chen_auc_n_trials", "Trials with matched labels for AUC",
                 int(len(auc_table)), "count", write_paths["fig5_auc"],
                 "trial", "row count")
    record_claim("chen_damping_fine_mean",
                 "Mean commanded damping during Fine windows",
                 weighted_fine, "N s/m", write_paths["fig5_auc"],
                 "mean_bd_fine_n_s_m", "sample-weighted mean over rows")
    record_claim("chen_damping_gross_mean",
                 "Mean commanded damping during Gross windows",
                 weighted_gross, "N s/m", write_paths["fig5_auc"],
                 "mean_bd_gross_n_s_m", "sample-weighted mean over rows")
    record_claim("chen_damping_difference",
                 "Fine minus Gross commanded damping",
                 weighted_fine - weighted_gross, "N s/m",
                 write_paths["fig5_auc"],
                 "mean_bd_fine_n_s_m - mean_bd_gross_n_s_m",
                 "sample-weighted means")
    record_claim("chen_damping_fine_samples",
                 "Robot samples labelled Fine in the damping analysis",
                 fine_total, "count", write_paths["fig5_hist"], "count",
                 "label == Fine, summed over bins")
    record_claim("chen_damping_gross_samples",
                 "Robot samples labelled Gross in the damping analysis",
                 gross_total, "count", write_paths["fig5_hist"], "count",
                 "label == Gross, summed over bins")
    return histogram_table, auc_table


def save_run_manifest(built: dict) -> Path:
    """Write a timestamped JSON record of this run's settings and outputs.

    Inputs:
      - built: mapping of output key to the row count written.
    Outputs:
      - Path of the manifest written under data/json/.
    """
    JSON_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = JSON_DIRECTORY / f"build_paper_results_{timestamp}.json"
    path.write_text(json.dumps({
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "outputs_to_build": list(outputs_to_build),
            "dataset_name": dataset_name,
            "expected_subjects": list(expected_subjects),
            "window_size_s": window_size_s,
            "window_step_s": window_step_s,
            "feature_sets_for_paper": list(feature_sets_for_paper),
            "damping_bin_width_n_s_m": damping_bin_width_n_s_m,
            "label_match_tolerance_s": label_match_tolerance_s,
        },
        "rows_written": built,
        "claims_written": len(claim_rows),
    }, indent=2), encoding="utf-8")
    return path


def report_io_paths() -> None:
    """Print every file this run will read and write, before doing work.

    Inputs:
      - None (uses the module-level settings).
    Outputs:
      - None. Prints to stdout.
    """
    print("=" * 68)
    print(f"build_paper_results.py  dataset={dataset_name}")
    print(f"building: {', '.join(outputs_to_build)}")
    print("=" * 68)
    print("READS:")
    for path in [read_loso_folds_path, read_lead_folds_path,
                 *read_losho_paths.values(), *read_feature_paths.values()]:
        print(f"  {path}")
    if "fig5" in outputs_to_build:
        print(f"  {RAW_DATA_DIRECTORY / 'ur5e_chen_var_adm_2d_*.csv'} "
              "(read-only)")
        print(f"  {PROCESSED_DATA_DIRECTORY / 'tobii_*_live_labeled_*.csv'}")
    print("WRITES:")
    for key, path in write_paths.items():
        if key == "claims" or key.split("_")[0] in outputs_to_build:
            print(f"  {path}")
            if key.startswith("tab"):
                print(f"  {path.with_suffix('.tex')}")
    print(f"  {JSON_DIRECTORY / 'build_paper_results_<timestamp>.json'}")
    print("=" * 68)


def main() -> None:
    """Build every requested paper output and the claims traceability file.

    Inputs:
      - None (driven by the User Settings block).
    Outputs:
      - None. Writes the CSV, TeX, claims, and manifest files above.
    """
    report_io_paths()
    built = {}

    windows = load_window_metadata()
    folds = load_loso_folds() if {"tab2", "fig3"} & set(outputs_to_build) \
        else None

    if "tab1" in outputs_to_build:
        built["tab1"] = len(build_tab1(windows))
        print(f"tab1 participants      -> {write_paths['tab1'].name}")
    if "tab2" in outputs_to_build:
        built["tab2"] = len(build_tab2(folds))
        print(f"tab2 main results      -> {write_paths['tab2'].name}")
    if "fig3" in outputs_to_build:
        built["fig3"] = len(build_fig3(folds))
        print(f"fig3 accuracy (tidy)   -> {write_paths['fig3'].name}")
    if {"fig4", "tab3"} & set(outputs_to_build):
        fig4_table, tab3_table = build_fig4_and_tab3()
        built["fig4"] = len(fig4_table)
        built["tab3"] = len(tab3_table)
        print(f"fig4 lead time (tidy)  -> {write_paths['fig4'].name}")
        print(f"tab3 lead-time stats   -> {write_paths['tab3'].name}")
    if "fig5" in outputs_to_build:
        print("fig5: reading raw trials, this is the slow step ...")
        histogram_table, auc_table = build_fig5()
        built["fig5_hist"] = len(histogram_table)
        built["fig5_auc"] = len(auc_table)
        print(f"fig5 damping histogram -> {write_paths['fig5_hist'].name}")
        print(f"fig5 per-trial AUC     -> {write_paths['fig5_auc'].name}")

    claims = pd.DataFrame(claim_rows)
    claims.to_csv(write_paths["claims"], index=False)
    print(f"claims traceability    -> {write_paths['claims'].name} "
          f"({len(claims)} claims)")

    manifest_path = save_run_manifest(built)
    print(f"run manifest           -> {manifest_path.name}")
    print("all structural checks passed.")


if __name__ == "__main__":
    main()
