# Paper Pipeline — Execution Runbook

Step-by-step order to regenerate every number and figure in the HRI paper,
with the inputs, outputs, settings, and a verification check for each step.

Verified against the actual files on 2026-09-13. Anything I could not
verify is marked **[UNVERIFIED]**.

---

## 0. What actually matters

Of the 29 scripts in `scripts/`, **six** are on the paper's critical path.

| Run? | File | Role |
|---|---|---|
| ✅ | `build_modality_feature_datasets.py` | raw+labels → window features |
| ✅ | `compare_loso_feature_sets.py` | LOSO + lead-time sweep |
| ✅ | `losho_classifier.py` | leave-one-shape-out (run **twice**) |
| ✅ | `build_paper_results.py` | → paper CSVs, .tex, claims file |
| ✅ | `matlab/paper_fig{3,4,5}*.m` | → figures |
| ⚠️ | `train_modality_classifiers.py` | **DO NOT RUN** — but its settings are imported by `compare_loso_feature_sets.py`. See §7. |

**Ignore entirely for the paper** (they were exploratory or are hardware):
`find_optimal_feature_count.py`, `check_overfitting_vs_feature_count.py`,
`train_simple_classifier.py`, `tune_random_forest_loso.py`,
`train_modality_classifiers_loso.py`, `train_modality_classifiers_top_n.py`,
`compare_modality_classifiers.py`, `plot_top20_modality_comparison.py`,
`predict_with_saved_model.py`, `test.py`, `test_tobii.py`, and every
`ur5e_*.py` (these drive the real robot — **never run them**).

`classifier_feature_bank.py` is a library, never run directly.

---

## 1. BEFORE YOU START — snapshot

Step 2 overwrites the feature CSVs that every current result was built
from. Copy them first so you can diff old vs new:

```bash
cp -r data/processed_data data/processed_data_backup_20260913
```

**Recommended verification order:** go *backwards* first (Step 5 → 4 → 3),
because those only read. Do Step 2 (rebuild features) last, after the
backup, since it is the only destructive one.

---

## 2. STEP 1 — Labeling  *(already done; re-run only for new trials)*

| | |
|---|---|
| **Script** | `scripts/manual_label_chen_var_adm_2d_by_live_replay.py` |
| **Reads** | `data/raw_data/ur5e_chen_var_adm_2d_*.csv` + matching `tobii_*` |
| **Writes** | `data/processed_data/{ur5e,tobii}_chen_var_adm_2d_live_labeled_*.csv` |
| **Settings** | `robot_csv_filename` — one trial per run, selected by hand |
| **Interactive** | Yes — keyboard-driven replay, one keypress toggles Fine/Gross |

**Do not re-run.** All 8 subjects are labeled. Re-labeling would change
the ground truth every result depends on.

---

## 3. STEP 2 — Build window features  ⚠️ destructive

| | |
|---|---|
| **Script** | `scripts/build_modality_feature_datasets.py` |
| **Reads** | `*_live_labeled_*.csv` (both robot and gaze side) |
| **Writes** | `features_{robot,gaze,joint,all}_{train,test}_W_2s.csv` (8 files) + run manifest in `data/json/` |
| **Runtime** | Minutes (uses a cache — see below) |

### Settings to check (line numbers from the current file)

| Line | Setting | Value | Meaning |
|---|---|---|---|
| 47 | `dataset_name` | `"chen_manual"` | **INPUT selector.** Picks which labeled files to read. **Do not change this to "W_2s"** — it is a registry key, not the output tag. |
| 53 | `output_dataset_name` | `"W_2s"` | **OUTPUT tag.** Goes into every output filename. |
| 61 | `modalities_to_build` | `("robot","gaze","joint","all")` | All four needed |
| 114 | `window_size_s` | `2.0` | |
| 115 | `window_step_s` | `2.0` | = size, so windows do not overlap |
| 124 | `transition_margin_s` | `0.5` | drops windows near a label change |
| 125 | `edge_trim_s` | `1.5` | drops trial start/end |
| 77 | `use_trial_cache` | `True` | reuses per-trial windows; cache key includes the settings above, so it self-invalidates |

> **The single most confusing thing in this repo:** lines 47 and 53 are
> *different* settings. `dataset_name` selects the input;
> `output_dataset_name` names the output. Changing window size means
> changing line 114/115 **and** line 53 (e.g. to `"W_3s"`), not line 47.

### Verify

```
8 CSVs written, all ending _W_2s.csv
robot train+test = 2156 rows total
subjects present = 10..17 (8 subjects)
distinct trials = 86
```

⚠️ **Known issue:** 5 recorded trials produce zero windows (s10
narrow/ellipse/cw, s11 wide/ellipse/cw, s13 wide/fine_gross/cw, s15
narrow/pear/cw and /ccw). Only s13's is explained (no tobii label file).
The other four have valid labels, good gaze, full duration — cause
unknown. This script prints per-trial exclusion counters when it runs;
**watch that output**, it should tell you which gate rejects them.

---

## 4. STEP 3 — LOSO + lead-time sweep

| | |
|---|---|
| **Script** | `scripts/compare_loso_feature_sets.py` |
| **Reads** | `features_*_{train,test}_W_2s.csv` |
| **Writes** | `loso_feature_set_comparison_W_2s.csv`, `loso_feature_set_folds_W_2s.csv`, `loso_lead_time_sweep_W_2s.csv`, `loso_lead_time_sweep_folds_W_2s.csv`, plus confusion-matrix PNGs |

### Settings

| Line | Setting | Value |
|---|---|---|
| 55 | `exclude_positional_features` | `True` |
| 56 | `exclude_force_features` | `True` |
| 59 | `feature_sets_to_run` | the tuple of sets to evaluate |
| 71 | `lead_time_sweep_s` | `(0.0, 2.0, 4.0, 6.0)` |
| 72 | `window_step_s` | `2.0` — **its own copy**, must match Step 2 line 115 |
| 79 | `dataset_name` | `tmc.dataset_name` — **inherited, see §7** |

> ⚠️ Line 72 is a *duplicate* of Step 2's `window_step_s`. If they
> disagree, every lead time is silently wrong (this bug already happened
> once: 0.5 vs 2.0 doubled every horizon). Also keep every entry of
> `lead_time_sweep_s` an exact multiple of `window_step_s`, or sweep
> points collapse onto each other.

### Verify

```
loso_feature_set_comparison_W_2s.csv : robot accuracy = 0.9801 (pooled)
loso_feature_set_folds_W_2s.csv      : 8 rows per feature set
loso_lead_time_sweep_W_2s.csv        : 8 rows (2 feature sets x 4 leads)
```

---

## 5. STEP 4 — LOSHO (run twice)

| | |
|---|---|
| **Script** | `scripts/losho_classifier.py` |
| **Reads** | `features_{modality}_{train,test}_W_2s.csv` |
| **Writes** | `losho_fold_results_{modality}_W_2s.csv` + 3 PNGs |

**Run it twice**, editing one line between runs:

```python
modality = "robot"   # run 1
modality = "all"     # run 2
```

Other settings: `dataset_name = "W_2s"`, `exclude_positional_features = True`,
`exclude_force_features = True`, `random_seed = 42`.

### Verify

```
losho_fold_results_robot_W_2s.csv : 3 rows (ellipse, fine_gross, pear)
  accuracies ~ 0.951, 0.974, 0.976
losho_fold_results_all_W_2s.csv   : 3 rows
  accuracies ~ 0.967, 0.962, 0.980
```

---

## 6. STEP 5 — Build paper numbers

| | |
|---|---|
| **Script** | `scripts/build_paper_results.py` |
| **Reads** | the LOSO/LOSHO/lead-time CSVs above, `features_robot_*` (metadata only), and raw trials (fig5 only, read-only) |
| **Writes** | `paper_fig{3,4,5}_*_W_2s.csv`, `paper_tab{1,2,3}_*_W_2s.csv` + `.tex`, `paper_claims_W_2s.csv`, run manifest |
| **Runtime** | ~1 min (fig5 reads ~5M raw samples) |

Settings: `outputs_to_build` (comment out what you are not changing),
`dataset_name = "W_2s"`, `expected_subjects = (10..17)`,
`window_size_s`/`window_step_s` = 2.0 (must match Step 2).

This script **asserts** its assumptions and crashes loudly if one breaks
(subject set, one fold per subject, feature counts, class totals,
persistence degeneracy). That is intentional.

### Verify

```
prints "all structural checks passed."
paper_claims_W_2s.csv has 46 rows
paper_tab2 : robot = 98.11 +/- 1.26
paper_tab3 : lead 4s robot_vs_persistence_mean = 21.82, p = 0.008
```

`paper_claims_W_2s.csv` is your cross-check table: every number in the
paper, with the file and column it came from.

---

## 7. STEP 6 — Figures (MATLAB)

| | |
|---|---|
| **Scripts** | `matlab/paper_fig3_accuracy.m`, `paper_fig4_lead_time.m`, `paper_fig5_chen_auc.m` |
| **Run all** | `matlab/build_all_paper_figures.m` |
| **Reads** | the `paper_fig*`/`paper_tab3` CSVs from Step 5 |
| **Writes** | `data/images/paper_fig{3,4,5}_*_W_2s.{png,pdf}` |

Each has a `User Settings` block at the top (which series to show, axis
limits, output formats). Requires MATLAB **R2020b+**.

**[UNVERIFIED]** These have never been executed — no MATLAB in the
environment where they were written. Expect possible syntax errors on
first run.

---

## 8. The cross-script trap: `train_modality_classifiers.py`

`compare_loso_feature_sets.py` line 40 does `import train_modality_classifiers as tmc`,
then line 79 `dataset_name = tmc.dataset_name`.

**Consequence:** the dataset that Step 3 analyses is set in a *different
file* than the one you are reading. If you change `dataset_name` in
`train_modality_classifiers.py` (line 75, currently `"W_2s"`), Step 3
silently switches datasets.

Step 3 also inherits from it: `target_column`, `metadata_columns`,
`classifier_name`, `classifier_kwargs`, and the `build_classifier()` and
`flatten_classification_report()` functions.

**You do not need to run `train_modality_classifiers.py`** — it is guarded
by `if __name__ == "__main__"`, so importing it does not train anything.
Just keep line 75 = `"W_2s"`.

(Its own `modalities_to_train = ("gaze",)` on line 70 only matters if you
run it directly, which you should not.)

---

## 9. Settings that must agree across files

| Setting | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 |
|---|---|---|---|---|---|
| dataset tag | `output_dataset_name="W_2s"` (L53) | via `tmc.dataset_name` (L75 of tmc) | `dataset_name="W_2s"` | `dataset_name="W_2s"` | `dataset_name="W_2s"` |
| `window_step_s` | 2.0 (L115) | 2.0 (L72) | — | 2.0 | — |
| `window_size_s` | 2.0 (L114) | — | — | 2.0 | — |
| exclude positional | — | `True` | `True` | — | — |
| exclude force | — | `True` | `True` | — | — |
| `random_seed` | — | 42 | 42 | — | — |

To change window size, you must edit: Step 2 L114+L115+L53, Step 3 L72,
Step 5 `window_size_s`/`window_step_s`/`dataset_name`, Step 4
`dataset_name`, Step 6 `dataset_name`. **Seven places.**

---

## 10. Flags and things I am not sure about

1. **Reproducibility: code is correctly seeded, but not empirically
   confirmed.** I checked every stochastic element in Step 3 and they are
   all seeded — the forest (`classifier_kwargs` has `random_state: 42`),
   the three `np.random.default_rng(0/1/2)` generators, and the balanced
   per-fold draw (`random_state=random_seed`). `n_jobs=-1` does not break
   this: sklearn derives per-tree seeds from `random_state` before
   dispatching in parallel. So re-running Step 3 *should* reproduce the
   CSVs on disk bit-for-bit. **[UNVERIFIED]** — I have not actually
   re-run it to confirm. Still the single most valuable check for you to
   do: re-run Step 3 and diff against your backup. A difference would
   mean a hidden unseeded path.

2. **[UNVERIFIED]** The 4 unexplained dropped trials (§3). Watch the
   exclusion counters when you re-run Step 2.

3. **[UNVERIFIED]** `losho_classifier.py` lists `damping_n_s_m` in its
   `metadata_columns`, but that column does not exist in the feature
   CSVs. It runs without error, so it is presumably tolerated — but I
   have not confirmed how.

4. **Stale docstring:** `all_modality_table()` in
   `compare_loso_feature_sets.py` says the `features_all_*` files are
   "written by build_modality_feature_dataset_all.py". That script was
   merged into `build_modality_feature_datasets.py`. Comment only, no
   effect on results.

5. **Not in the paper path but present on disk:** older `*_chen_manual`
   CSVs from a previous dataset tag. Do not mix them with `*_W_2s`
   results.

6. Steps 3 and 4 each fit their own Random Forests. Re-running them
   overwrites the analysis CSVs but **not** the features, so they are
   safe to re-run without redoing Step 2.
