# Gaze / Fine-Gross Classification Pipeline

Predicts whether a trial-window is "Fine" or "Gross" motor behavior from
the Chen variable-admittance dataset (subjects s10-s18), and tests
whether eye-gaze alone carries information about that regime -- as a
step toward gaze-informed damping control. There is no ground-truth
label for Fine/Gross; it's derived algorithmically (Otsu threshold on
damping) by the first MATLAB script below, and everything downstream
consumes that.

## The controller that generated the data (`scripts/ur5e_chen_var_adm_2d.py`)

Not part of the labeling/training pipeline itself, but everything above
depends on the CSVs it writes, so understanding it matters. It's a
planar variable-admittance controller for the UR5e (`Md*x_ddot +
Bd*x_dot = F_ext`), implementing H. Chen et al.'s RA-L 2024 paper (a
fixed/hardened port of a prior Schunk-Powerball C++ implementation):

1. **Direct intention** (Eq. 7) -- `compute_direct_damping(|F|)` maps
   filtered force magnitude to an isotropic damping `bd`, linearly
   between `force_min_n`/`force_max_n`: small force -> high damping
   (precise motion), large force -> low damping (fast motion). **This
   is the source of the label's circularity** -- see gotcha #1 below.
2. **Indirect intention** (Eqs. 4, 8, 9) -- fits the speed-curvature
   power law `v = K * kappa^-beta` over a sliding window, so `beta`
   characterizes the kind of path the human is drawing.
3. **Intended acceleration** (Eq. 10) -- the fitted power law sets a
   target curvature/turn direction; tangential acceleration is chosen
   to minimize jerk.
4. **Guidance** (Eqs. 11-14) -- builds an anisotropic damping matrix
   `diag(bd/gamma1, gamma2*bd)`, low along the intended direction and
   high perpendicular to it, then rotates it into the base frame and
   blends it continuously toward isotropic `diag(bd, bd)` (a
   `guidance_weight` in [0, 1], never a hard switch, so commanded
   damping never jumps).

Writes one CSV per trial to `data/raw_data/` (force, TCP motion,
power-law fit quality, intended acceleration/force/direction, the full
damping matrix, guidance weight) plus a matching Tobii gaze CSV sharing
the same `time_s` origin. **Never run this script without explicit
permission** -- it drives a real 6-DOF arm.

## Pipeline order

```
data/raw_data/{ur5e,tobii}_chen_var_adm_2d_*.csv + data/calibrations/*.json
        |
        v   compare_plot_save_chen_otsu_labeling_across_subjects.m  (save_labeled_csv=true)
        v
data/processed_data/{ur5e,tobii}_chen_var_adm_2d_otsu_labeled_*.csv
        |
        v   build_modality_feature_datasets.py
        v
data/processed_data/features_{robot,gaze,joint}_{train,test}_chen_otsu.csv
        |
        v   compare_loso_feature_sets.py  /  train_modality_classifiers_loso.py
        v   plot_top20_modality_comparison.py
        v
data/images/*.png, data/processed_data/loso_*.csv
```

Run `compare_chen_var_adm_2d_across_subjects.m` and, for any trial that
looks wrong there, `plot_raw_data_chen_var_adm_2d.m` BEFORE trusting a
labeling run -- they touch no files and exist purely to catch bad data
before it's labeled.

## Files

### MATLAB (`matlab/`) -- data prep / visualization, self-contained scripts

| File | Role | Writes |
|---|---|---|
| `compare_chen_var_adm_2d_across_subjects.m` | Read-only diagnostic: TCP path (colored by damping), force, and damping over time, one figure-row per trial across every subject/shape/bandwidth/direction. Run this first on new data. | nothing |
| `plot_raw_data_chen_var_adm_2d.m` | Deep-dive on ONE selected trial: damping law, curvature fit, anisotropy, velocity tracking, gaze, path accuracy, smoothness, effort, and a real-time replay (12 figures). Use when a trial looks off in the across-subjects comparison. | nothing |
| `compare_plot_save_chen_otsu_labeling_across_subjects.m` | **The labeling script.** Computes a per-trial Otsu threshold on the 1s-smoothed `bd_applied_Ns_m` -> Fine/Gross. Set `save_labeled_csv = true` to write the labeled CSVs everything below depends on. | `data/processed_data/{ur5e,tobii}_chen_var_adm_2d_otsu_labeled_<trial>.csv` |

### Python (`scripts/`) -- feature building / training / evaluation

| File | Role | Reads | Writes |
|---|---|---|---|
| `classifier_feature_bank.py` | Defines the three feature banks (robot 96 / gaze 102 / joint 51 features). Imported only, never run directly. | -- | -- |
| `build_modality_feature_datasets.py` | Discovers labeled trials (`dataset_name = "chen_otsu"` or `"fixed_pilot"`), cuts into 0.5s windows, computes features via the bank above. Resumable per-trial cache under `feature_cache/`. | `*_otsu_labeled_*.csv` | `features_{robot,gaze,joint}_{train,test}_<dataset>.csv` |
| `train_modality_classifiers.py` | Not run directly for chen_otsu -- shared settings module (`classifier_kwargs`, `metadata_columns`, `target_column`, `build_classifier()`, `flatten_classification_report()`) every script below imports. | -- | -- |
| `train_modality_classifiers_loso.py` | Leave-one-subject-out evaluation, one modality's full feature set at a time. | `features_*_<dataset>.csv` | `loso_*_<dataset>.csv/png` |
| `compare_loso_feature_sets.py` | The main comparison tool. Builds 6 feature sets (robot / gaze / **gaze_pure** / joint / robot+gaze / robot+gaze_pure), runs LOSO on each (with top-N variants), plus a lead-time anticipation sweep. | `features_*_<dataset>.csv` | `loso_feature_set_*.csv`, `loso_lead_time_sweep_*.csv`, confusion-matrix PNGs |
| `plot_top20_modality_comparison.py` | Final-figure generator: confusion matrix + metrics, and a separate feature-ranking chart, for Robot-only / Gaze-only / Joint, each on its own top-20 LOSO-selected features. | `features_*_<dataset>.csv` (via `compare_loso_feature_sets.py`) | `top20_confusion_matrix_<dataset>.png`, `top20_feature_ranking_<dataset>.png` |

## First-run checklist

1. `compare_chen_var_adm_2d_across_subjects.m` -- sanity-check raw data.
2. `compare_plot_save_chen_otsu_labeling_across_subjects.m` with `save_labeled_csv = true` -- produce labels.
3. `python build_modality_feature_datasets.py` -- produce feature CSVs.
4. `python compare_loso_feature_sets.py` and/or `python plot_top20_modality_comparison.py` -- results and figures.

## Read this before changing anything

1. **The label is not independent of force.** `compute_direct_damping()` in `ur5e_chen_var_adm_2d.py` makes `bd_applied` a deterministic function of filtered force magnitude alone. Any robot/force feature can "predict" the label almost trivially -- see the `CIRCULARITY WARNING` in `build_modality_feature_datasets.py`'s docstring. This is why gaze-only, not robot-only, is the number that means anything.
2. **`"gaze"` is not gaze-only.** `compare_loso_feature_sets.py`'s 102-feature `"gaze"` bank secretly includes 29 features that also read the robot CSV (e.g. gaze-to-EE distance). Use `"gaze_pure"` (73 features) for strictly Tobii-derived features.
3. **Robot position features can leak shape geometry.** `mean_position_x` etc. tell the classifier *where on the shape* it is, which correlates with curvature by construction (geometry is fixed per shape/bandwidth). Filter by `"position" in feature_name` if you want the classifier to learn behavior, not location.
4. **`otsu_min_class_fraction` (0.20)** exists because one trial's threshold got hijacked by a ~1.2s damping-saturation artifact (subject nearly let go of the end-effector). Don't remove it without reading that function's docstring first.
5. **7 of 103 labeled trials are excluded from training**: 5 for missing gaze/calibration files (`build_modality_feature_datasets.py` skips these automatically and prints why), 2 as explicit judgment calls from a calibration audit (`excluded_trial_stems`).
6. **`dataset_name` / `modalities_to_build` / `modalities_to_train`** are the master switches (`build_modality_feature_datasets.py`, `train_modality_classifiers.py`). Output filenames carry the dataset name so pilot and chen_otsu results never overwrite each other.
7. **`classifier_kwargs` has `n_jobs=-1`** (`train_modality_classifiers.py`) -- remove it if `classifier_name` is ever switched to `"svm"` (`SVC` has no `n_jobs`, and `build_classifier()` passes kwargs through unfiltered).
8. **Never run MATLAB or hardware-control scripts without explicit permission** (this project's standing rule) -- the Python scripts above only read/write CSVs and are safe to run freely; the `.m` script writes real files once `save_labeled_csv = true`.
