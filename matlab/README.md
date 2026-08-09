# MATLAB Analysis Guide

This folder contains the MATLAB analysis and visualization scripts for the 2-D UR5e/Tobii experiments. Each `.m` file is a standalone script: select the matching trial files in its **User Settings** section, then run that script in MATLAB.

## Project flow

```text
scripts/ur5e_fixed_adm_2d.py
    -> data/raw_data/tobii_fixed_adm_2d_*.csv
    -> data/raw_data/ur5e_fixed_adm_2d_*.csv
         |-> plot_tobii_raw_data.m                 (raw-gaze check)
         |-> detect_tobii_fixations_ivt.m          (eye-only fixations)
         |-> gaze_feature_extraction.m             (windowed features)
         `-> compute_plot_similarity_*.m           (joint gaze/robot analysis)

scripts/ur5e_tobii_reference_calibration.py
    -> data/calibrations/*.json
         `-> compute_plot_similarity_*.m
                -> data/processed_data/*_fixation_transitions.csv
                     `-> analyze_gaze_tcp_lead_lag_curvature.m
                          -> *_gaze_tcp_lead_lag.csv
```

## Script guide

### `plot_tobii_raw_data.m`

- **Purpose:** Quick quality check of one raw Tobii recording.
- **Input:** One `tobii_fixed_adm_2d_*.csv` file from `data/raw_data`.
- **Output:** Four figures showing normalized display gaze, x/y gaze over time, pupil diameter and eye-origin motion, and 3-D Tobii User Coordinate System (UCS) data. It also prints sample rate, duration, and validity percentages. It does not write a file.
- **Use it when:** Checking that gaze, pupils, timestamps, and validity look reasonable before further analysis.

### `detect_tobii_fixations_ivt.m`

- **Purpose:** Detect fixations from Tobii data with the I-VT algorithm. It combines valid eyes, computes 3-D angular gaze velocity, labels samples, groups consecutive candidates, and removes short groups. It was just a test script to verify I-VT algorithm!
- **Input:** One raw Tobii CSV containing display gaze, UCS gaze points/origins, validity flags, and `system_time_stamp`.
- **Output:** `fixation_table` and `sample_results` in the workspace, one fixation plot, and a `*_fixations_ivt.csv` event file in `data/processed_data`.
- **Main settings:** `velocity_threshold_deg_s` and `minimum_fixation_duration_s`.
- **Relationship:** This is the standalone, eye-only fixation check. The joint similarity script performs its own I-VT detection separately.

### `compute_plot_similarity_fixation_direction_force_vel.m`

- **Purpose:** Main fixation-transition analysis. It maps Tobii UCS gaze into the CoppeliaSim x-y frame, detects fixations, forms arrows between consecutive fixations, averages robot force and velocity during each transition, and computes cosine similarity.
- **Inputs:** A matching Tobii/UR5e raw CSV pair plus a calibration JSON from `data/calibrations/`. The current analysis expects the synchronized `time_s` column recorded by `scripts/ur5e_fixed_adm_2d.py`.
- **Output:** Five diagnostic figures; `fixation_table`, `gaze_sample_results`, and `fixation_transition_table` in the workspace; and a canonical `*_fixation_transitions.csv` file in `data/processed_data`. Each CSV row represents one transition and includes timing, fixation endpoints/direction, mean force, mean velocity, damping, similarities, angles, sample counts, and validity flags.
- **Main settings:** Trial CSVs, `calibration_id`, spatial scale/axis signs, I-VT thresholds, transition quality thresholds, and plot controls.
- **Relationship:** This processed transition CSV is the best starting point for later fixation-level hypotheses and plots.

### `gaze_feature_extraction.m`

- **Purpose:** Exploratory sliding-window analysis with 102 gaze-only and gaze-versus-end-effector features, including 10x10 and 40x40 stationary/transitional Shannon entropy. It correlates every feature with dimensionless squared jerk (DSJ) and mean planar force.
- **Inputs:** A matching raw Tobii/UR5e CSV pair. Gaze uses normalized display coordinates; CoppeliaSim position is mapped into the same normalized display space.
- **Output:** `window_results`, `correlation_results`, and two absolute-correlation rankings in the workspace; ranked tables in the Command Window; and one normalized comparison plot for gaze/EE speed ratio, DSJ, and mean force. It does not write a file.
- **Main settings:** `window_size_s`, `window_step_s`, sample-count limits, optional time offset, display bounds, and display-y inversion.
- **Relationship:** This is separate from the fixation-transition pipeline and works directly on sliding windows of raw recordings.

### `analyze_gaze_tcp_lead_lag_curvature.m`

- **Purpose:** Determines whether each fixation transition leads the TCP, follows it, or is approximately synchronous by scanning TCP time lag and maximizing directional cosine similarity. It also computes filtered planar TCP curvature.
- **Inputs:** A `*_fixation_transitions.csv` from `compute_plot_similarity_fixation_direction_force_vel.m` and its matching raw UR5e CSV. The raw file is inferred from `trial_id` unless overridden.
- **Output:** `lead_lag_table` and `lag_similarity_matrix` in the workspace, three lead-lag/curvature figures, and a `*_gaze_tcp_lead_lag.csv` summary in `data/processed_data`.
- **Main settings:** Lag vector, TCP displacement-window duration, synchronous-lag tolerance, minimum gaze/TCP displacement, and TCP position/curvature filtering windows.
- **Relationship:** This is a downstream analysis of the canonical fixation-transition CSV; it does not rerun fixation detection or gaze-to-CoppeliaSim calibration.

## Important notes

- Always select a Tobii CSV and UR5e CSV from the **same trial**.
- Raw CSV files are read only. Generated event/transition tables go in `data/processed_data`.
- The MATLAB scripts do not call one another; shared logic such as I-VT detection is currently implemented locally in each relevant script.
- `plot_tobii_raw_data.m`, `detect_tobii_fixations_ivt.m`, and `gaze_feature_extraction.m` derive gaze time from Tobii `system_time_stamp`; the joint similarity script uses the shared synchronized `time_s` field.
- `compute_plot_similarity_fixation_direction_force_vel.asv` is a MATLAB autosave backup, not a script to run.
- Required project-side producers are [`scripts/ur5e_fixed_adm_2d.py`](../scripts/ur5e_fixed_adm_2d.py) for raw recordings and [`scripts/ur5e_tobii_reference_calibration.py`](../scripts/ur5e_tobii_reference_calibration.py) for spatial calibration JSON.
