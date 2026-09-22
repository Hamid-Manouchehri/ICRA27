"""
Chen-style variable admittance controller for planar UR5e pHRI, with the
scalar damping bounds gated by a trained Fine/Gross classifier reading
live Tobii gaze + robot motion.

Same controller as ur5e_chen_var_adm_2d.py (that file's port notes and
deviation list below still apply verbatim) plus one addition: the
force->damping map of paper (7) keeps its shape, but its two endpoints
are interpolated between a Fine and a Gross bound pair by the
classifier's P(Fine). Set enable_classifier_gating = False to get the
unmodified Chen baseline back, so both benchmark arms are this one file.

Paper:
    H. Chen, W. Xu, W. Guo, X. Sheng, "Variable Admittance Control Using
    Velocity-Curvature Patterns to Enhance Physical Human-Robot
    Interaction", IEEE RA-L 9(6), 2024, pp. 5054-5060.
    See doc/Variable_Admittance_Control_Using_Velocity-Curvature_
    Patterns_to_Enhance_Physical_Human-Robot_Interaction.pdf

Inputs:
    UR5e base-frame TCP wrench from RTDE, using only [Fx, Fy] [N].
    UR5e base-frame TCP pose from RTDE, using only [x, y, z] [m].
    UR5e actual TCP velocity from RTDE [vx, vy, vz, wx, wy, wz], with
    linear components [m/s] and angular components [rad/s].
    Live Tobii gaze samples (same SDK callback that already feeds the
    gaze CSV).
    read_model_joblib_path: the scaler + fitted model + feature_columns
    bundle written by train_logistic_regression.py.
    read_calibration_json_path: this session's Tobii-UCS ->
    CoppeliaSim/robot-base mapping, from
    ur5e_tobii_reference_calibration.py.

Outputs:
    Optional UR5e Cartesian velocity command [vx, vy, 0, 0, 0, 0].
    CSV log with force, TCP motion, power-law estimates, intended
    acceleration/force/direction, the full 2x2 damping matrix, the
    guidance blend weight, and the classifier columns (P(Fine), its
    age and status, the interpolated bd_min/bd_max, and every feature
    value the model was fed).
    Optional Tobii gaze CSV sharing the same time_s origin, so trials
    stay comparable with the rest of this project's raw_data.
    Optional CoppeliaSim TCP marker through a separate process.

Controller (paper equation numbers in parentheses):
    Planar mass-damper admittance (2):

        Md * x_ddot + Bd * x_dot = F_ext

    1. Direct intention (7): the force magnitude sets an isotropic
       damping bd. Small force means precise motion, so damping is high;
       large force means fast motion, so damping is low.

    2. Indirect intention (4), (8), (9): curvature of the recent path is
       computed from the measured velocity, and the speed-curvature
       power law

           v = K * kappa^(-beta)

       is fitted by least squares in log-log space over a sliding time
       window. beta in [0, 2/3] characterizes the path the human is
       drawing (beta = 1/3 is the classic two-thirds power law).

    3. Intended acceleration (10): the fitted power law sets the target
       curvature, hence the normal acceleration magnitude. The turn
       direction and the feasible set come from the force direction. The
       tangential acceleration is chosen by a small search minimizing
       jerk relative to the measured previous acceleration.

    4. Guidance (11)-(14): the intended force

           F_intended = md * a_intended + (bd / gamma1) * v

       defines the intended direction. Damping is low along that
       direction and high perpendicular to it,

           B_local = diag(bd / gamma1, gamma2 * bd)
           gamma1  = 1 + g1 * |v|^2,  gamma2 = 1 + g2 * |v|^2   (13)
           Bd      = R * B_local * R^T                          (14)

       where R rotates the intended direction onto the base-frame x
       axis. This implementation blends Bd continuously toward the
       isotropic diag(bd, bd) using a guidance weight in [0, 1] instead
       of switching, so the commanded damping never jumps.

Classifier gating (the only addition to the controller above):
    A background thread recomputes the model's features over the last
    classifier_window_s of buffered gaze + robot samples at
    classifier_update_frequency_hz, and publishes P(Fine) = Pt. The
    bounds of (7) are then interpolated per class,

        bd_min = Pt * Lf + (1 - Pt) * Lg
        bd_max = Pt * Uf + (1 - Pt) * Ug

    with [Lf, Uf] = fine_damping_bounds_n_s_m and [Lg, Ug] =
    gross_damping_bounds_n_s_m. Because (7) is affine in its two
    endpoints, this is algebraically identical to running a Fine-tuned
    and a Gross-tuned Chen controller in parallel and blending their
    commanded damping by Pt -- interpolating bounds and interpolating
    outputs are the same operation, which is the honest way to describe
    it in a write-up. It also means bd_min < bd_max holds for every Pt
    as long as Lf < Uf and Lg < Ug, so the range can never invert.

    The features are not reimplemented here: the exact function objects
    from classifier_feature_bank.py are looked up by the names in the
    bundle's feature_columns, so a feature definition cannot drift
    between training and deployment. Anything in feature_columns that
    cannot be resolved that way is a hard error at startup, not a
    silent NaN at 500 Hz.

    The classifier runs on its own thread rather than inline in the
    control loop so that a slow feature update degrades into a STALE
    probability instead of into control-loop jitter -- the loop only
    ever appends to a deque and reads one float under a lock. Pt is
    also rate-limited downstream for free: damping_rate_max_n_s_m_per_s
    already bounds how fast bd may move, so even a step in Pt reaches
    the operator as a ramp.

Known differences from the offline training pipeline:
    - build_modality_feature_datasets.py rebases gaze time to the FIRST
      TOBII SAMPLE, while robot time starts at controller start; here
      both come off the same trial clock, so the two streams are
      aligned slightly better online than they were in training. The
      offset is one gaze period plus tracker startup, not seconds.
    - Offline, a window with too few valid gaze/robot samples is
      DROPPED. Online nothing can be dropped, so that case holds the
      previous Pt and reports it in classifier_status; past
      max_probability_age_s it falls back to warmup_fine_probability.
    - Offline windows are gated on being free of label transitions
      (transition_margin_s) and trimmed at the trial edges
      (edge_trim_s). Neither applies online, so the model does see the
      transition windows it was never trained on.

Deviations from chen_var_adm.cpp (bug fixes and hardening):
    1. Force deadband was applied to the low-pass filter *output* and
       fed back as the filter state, so the filter state was repeatedly
       zeroed and could not accumulate near the threshold. Deadband is
       now applied to the raw force before filtering, matching
       ur5e_fixed_adm_2d.py.
    2. The per-axis deadband distorted force *direction* near the
       threshold (a force at 45 deg lost both components at different
       times). Replaced by a radial soft deadband on the planar
       magnitude, which preserves direction and is continuous at the
       threshold.
    3. No force bias estimation existed (the Schunk sensor was tared
       over TCP). An unloaded startup bias estimate is now taken, as in
       every other controller in this project.
    4. The power-law window was specified in samples (N = 50), so its
       duration silently changed with the control rate: 0.4 s at the
       paper's 125 Hz but only 0.1 s at the UR5e's 500 Hz, far too short
       to see path curvature. The window is now specified in seconds and
       the estimator runs at its own decimated rate.
    5. Curvature was computed by central-differencing the *unfiltered*
       measured velocity at the control rate, amplifying noise by
       1/(2*dt). Curvature now uses the filtered velocity sampled at the
       decimated estimator rate.
    6. beta and K were used unfiltered even though the paper low-pass
       filters beta. Both are now filtered, K in log space because it
       spans orders of magnitude.
    7. The target curvature used pow(K/v, 1/beta) directly, so a
       multiplicative error e in the fitted gain became e^(1/beta) in
       the commanded curvature. At the C++ beta floor of 0.02 that is
       e^50: a 20% gain error, well within what a 50-sample log-log fit
       on noisy velocity produces, commands 1.8e5 1/m instead of 20
       1/m. The inversion is now done in log space, held within
       target_curvature_max_log_deviation of the measured curvature and
       inside [curvature_min_1_m, curvature_max_1_m].
    8. Fit quality (R^2) was computed and logged but never used, so a
       meaningless fit still drove full-strength guidance. R^2 now feeds
       the guidance weight.
    9. last_turn_sign persisted forever, so after one turn the robot
       kept commanding normal acceleration in that stale direction even
       when the human pushed straight. Replaced by a filtered signed
       turn signal that decays to zero without fresh evidence, and whose
       magnitude feeds the guidance weight.
   10. The damping matrix switched discretely between isotropic and
       anisotropic on the indirect_ready flag, and its frame could
       rotate arbitrarily fast, both of which the operator feels as a
       step in resistance. The matrix is now blended continuously and
       the intended direction is rate-limited.
   11. The minimum-jerk cost used the previous *intended* acceleration,
       but paper equation (10) minimizes jerk against the previous
       *measured* acceleration. Using the intended value made the
       objective self-referential. Now uses the measured value.
   12. The tangential search grid was re-centered on the previous
       tangential acceleration and then clamped, so grid points
       collapsed onto the limit and the search was biased toward never
       changing. The grid now spans the full symmetric range.
   13. intended_acc_prev and intended_force_prev were never reset when
       guidance dropped out, so stale values snapped back on resume.
       Both now decay to zero while guidance is inactive.
   14. Curvature samples outside [kappa_min, kappa_max] were clamped and
       then fitted, piling up artificial points on the bounds and
       biasing beta. They are now excluded from the fit.
   15. Workspace limits were hardcoded absolute Schunk coordinates.
       They are now offsets from the startup TCP, as in this project's
       other controllers.
   16. The loop period came from system_clock (wall clock, NTP-jumpable)
       plus usleep. Replaced by the RTDE initPeriod()/waitPeriod() pair.
   17. The force was rotated by a fixed 90 deg offset and the
       end-effector orientation. getActualTCPForce() already reports the
       wrench in the UR5e base frame, so no rotation is applied here;
       only per-axis signs remain.
   18. Dead code removed: Md_diag/Cd_diag were assigned but never used,
       adm_time was never read, and the z workspace limit acted on a
       velocity that was always zero.

Known behavior worth checking before trusting a trial:
    gamma1/gamma2 (13) grow with the *square* of speed, so at this
    controller's speeds they are nearly inert: at 0.15 m/s the default
    paper gains give gamma1 = 1.045 and gamma2 = 1.09, i.e. only about
    5% anisotropy between the intended and perpendicular directions. The
    paper's design is deliberately isotropic at low speed, so this is
    faithful, not a porting error. If a stronger guidance effect is
    wanted, raise gamma1_speed_gain/gamma2_speed_gain and re-check
    stability. Both are exposed in User Settings.

Main tunable parameters:
    Virtual mass, the damping range [damping_min, damping_max], the
    force range [force_min, force_max], the power-law window and
    estimator rate, the guidance gains, and the workspace corners are
    all in User Settings, sorted most to least important. Check force
    signs, workspace, and the start pose before setting
    send_robot_commands = True.

Dependency:
    pip install --user ur_rtde tobii-research coppeliasim-zmqremoteapi-client

Assumptions:
    Motion is planar in the UR5e base-frame x-y plane at fixed z and
    orientation. The human pushes the tool directly, so the measured TCP
    wrench is the human force. The internal robot controller tracks the
    commanded Cartesian velocity closely enough that the measured TCP
    velocity can be used for the power-law estimate, as assumed in the
    paper.
"""

import csv
import json
import math
import multiprocessing
import queue
import signal
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import joblib
import numpy as np

import classifier_feature_bank as cfb

try:
    import rtde_control
    import rtde_receive
except ImportError:
    rtde_control = None
    rtde_receive = None


# -------------------- User Settings --------------------
robot_ip = "192.168.1.100"  # TODO UR5e IP/hostname [valid robot address]

# TODO bool; True sends physical speedL commands. Keep False until force
# signs, workspace corners, and the start pose have been checked.
send_robot_commands = True

# TODO controller output CSV path [file path]; resolved from this
# script, not from the current working directory.
write_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "ur5e_gaze_assisted_chen_vac_2d_held_out_pear_and_s24_on_narrow_pear_shape.csv"  # 2
    # "ur5e_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s24_on_wide_fine_gross_shape.csv"  # 3
    # "ur5e_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s24_on_narrow_ellipse_shape.csv"  # 4 
    # "ur5e_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s24_on_narrow_fine_gross_shape.csv"  # 5
    # "ur5e_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s24_on_wide_ellipse_shape.csv"  # 1
    # "ur5e_gaze_assisted_chen_vac_2d_held_out_pear_and_s24_on_wide_pear_shape.csv"  # 6
)

# TODO Tobii output CSV path [file path]; one row per Tobii callback.
# Shares the time_s origin of the controller CSV, so the pair matches
# the <ur5e_*, tobii_*> convention used by this project's MATLAB scripts.
tobii_data_csv_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "raw_data" /
    "tobii_gaze_assisted_chen_vac_2d_held_out_pear_and_s23_on_narrow_pear_shape.csv"  # 2
    # "tobii_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s24_on_wide_fine_gross_shape.csv"  # 3
    # "tobii_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s24_on_narrow_ellipse_shape.csv"  # 4
    # "tobii_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s24_on_narrow_fine_gross_shape.csv"  # 5
    # "tobii_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s24_on_wide_ellipse_shape.csv"  # 1 
    # "tobii_gaze_assisted_chen_vac_2d_held_out_pear_and_s24_on_wide_pear_shape.csv"  # 6
)

# --- Classifier-gated damping bounds (the addition over Chen) ---
# TODO bool; False runs the unmodified Chen baseline on
# damping_min_n_s_m/damping_max_n_s_m below -- the benchmark's control
# arm, same binary, same logging.
enable_classifier_gating = True

# TODO the bundle written by train_logistic_regression.py's
# write_model_file (scaler + model + feature_columns). That script
# currently reuses one .joblib name across runs, so the filename does
# NOT record the label lead or window it was trained at -- check its
# settings before trusting a bundle.
read_model_joblib_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "models" /
    # "logistic_regression_held_out_ellipse_shape_and_s14_1win_ahead_W_1s.joblib"  # subject WAS part of training data
    # "logistic_regression_held_out_fine_gross_shape_and_s14_1win_ahead_W_1s.joblib"  # subject WAS part of training data
    # "logistic_regression_held_out_pear_shape_and_s14_1win_ahead_W_1s.joblib"  # subject WAS part of training data

    # "logistic_regression_held_out_ellipse_shape_1win_ahead_W_1s.joblib"  # totally new subject
    "logistic_regression_held_out_pear_shape_1win_ahead_W_1s.joblib"  # totally new subject
    # "logistic_regression_held_out_fine_gross_shape_1win_ahead_W_1s.joblib"  # totally new subject
)

# TODO THIS session's Tobii -> CoppeliaSim/robot-base mapping, from
# ur5e_tobii_reference_calibration.py. Gaze features are computed in
# the CoppeliaSim frame, so a stale calibration silently biases every
# gaze feature instead of failing.
read_calibration_json_path = (
    Path(__file__).resolve().parent.parent /
    "data" /
    "calibrations" /
    # "tobii_coppelia_reference_calibration_wide_ellipse_shape_counter_clockwise_s24.json"
    # "tobii_coppelia_reference_calibration_narrow_ellipse_shape_counter_clockwise_s24.json"
    # "tobii_coppelia_reference_calibration_wide_pear_shape_counter_clockwise_s24.json"
    "tobii_coppelia_reference_calibration_narrow_pear_shape_counter_clockwise_s24.json"
    # "tobii_coppelia_reference_calibration_wide_fine_gross_shape_counter_clockwise_s24.json"
    # "tobii_coppelia_reference_calibration_narrow_fine_gross_shape_counter_clockwise_s24.json"
)

# TODO damping bounds [N s/m] per predicted class, each low < high.
# Fine is precise/slow so it belongs HIGHER than Gross -- paper (7)
# maps a small force onto the upper bound. bd_min/bd_max interpolate
# between these two pairs by P(Fine).
fine_damping_bounds_n_s_m = [80.0, 200.0]
gross_damping_bounds_n_s_m = [10.0, 100.0]

# TODO sham arm: None uses the model; a float in [0, 1] freezes P(Fine)
# there and never loads the model. Re-running at a model run's MEAN
# P(Fine) is the control that separates "the classifier's timing
# helped" from "any two-regime damping range helped".
fixed_fine_probability = None

# TODO feature window [s]. MUST equal build_modality_feature_datasets.
# py's window_size_s for the run that trained the bundle, or every
# feature is computed over the wrong duration.
classifier_window_s = 1.0

# TODO how often the classifier thread recomputes [Hz], 2 to 30. Far
# below the control rate on purpose: features barely move inside one
# window, and each update is a full pass over both buffers.
classifier_update_frequency_hz = 10.0

# TODO P(Fine) before the first successful update and after a stale
# one. 0.5 puts the bounds at the midpoint of the two pairs, i.e. a
# neutral controller rather than a committed one.
warmup_fine_probability = 0.5

# TODO how old P(Fine) may get [s] before falling back to
# warmup_fine_probability -- covers a blink, a look away, or a hung
# feature update.
max_probability_age_s = 2.0

# TODO low-pass alpha on P(Fine) [0, 1]; 1.0 passes it straight
# through. First knob to turn if P(Fine) latches: two of the default
# bundle's four features are robot-motion features and damping shapes
# robot motion, so the classifier sits in a feedback loop that did not
# exist while it was trained.
fine_probability_lpf_alpha = 1.0

# TODO clamp on P(Fine), low < high within [0, 1]. Tightening it (e.g.
# 0.05, 0.95) stops a saturated logit committing fully to one class's
# bounds, at the cost of never reaching either pair exactly.
fine_probability_limits = (0.0, 1.0)

# TODO minimum valid samples in the window before an update is trusted
# [count] -- mirrors build_modality_feature_datasets.py's
# minimum_gaze_samples / minimum_robot_samples.
minimum_classifier_gaze_samples = 6
minimum_classifier_robot_samples = 10

# TODO which of the bundle's classes the gating treats as "Fine".
positive_class_label = "Fine"

# TODO gaze validity + mapping. All four MUST match build_modality_
# feature_datasets.py's settings for the training run.
screen_motion_per_robot_motion = 1.0
tobii_to_coppelia_axis_sign = (1.0, 1.0)
require_gaze_on_display = True
require_both_eyes_valid = True

# TODO classifier_feature_bank.FeatureConfig thresholds, which must
# also match the training run. Unused by the default bundle's four
# features, but required to build the banks at all and live the moment
# a bundle names a threshold-dependent feature.
classifier_workspace_half_extent_m = 0.20
classifier_low_speed_threshold_m_s = 0.05
classifier_high_speed_threshold_m_s = 0.30
classifier_close_thresholds_m = (0.01, 0.02, 0.05)

# --- Admittance model, paper (2) and (7) ---
# TODO virtual mass md [kg], recommended 1 to 20. Paper uses 5 kg. This
# project's other controllers use 10 kg; smaller feels lighter but
# amplifies force noise into commanded acceleration.
virtual_mass_kg = 5.0

# TODO damping range [N s/m], both positive with min < max. Paper uses
# [90, 150] for its handle and robot. This project's fixed-admittance
# baseline runs 10 to 100, so expect to re-tune these on the UR5e.
damping_min_n_s_m = 10.0
damping_max_n_s_m = 200.0

# TODO force range [N] mapping force magnitude to damping in (7), with
# min < max. At or below force_min the damping is damping_max (precise);
# at or above force_max it is damping_min (compliant). Paper uses 1 and
# 11 N, found experimentally with multiple users.
force_min_n = 1.0
force_max_n = 11.0

# TODO damping slew-rate limit [N s/m per s], recommended 100 to 600.
# Bounds how fast the isotropic damping may change so the operator does
# not feel a step in resistance.
damping_rate_max_n_s_m_per_s = 300.0

# --- Power-law estimation, paper (4), (8), (9) ---
# TODO sliding window used for the least-squares fit [s], recommended
# 0.25 to 0.8. The paper's 50 samples at 125 Hz is 0.4 s. Longer windows
# give a steadier beta but lag genuine changes in the drawn path.
power_law_window_s = 0.40

# TODO rate at which curvature samples are taken and the fit is redone
# [Hz], recommended 60 to 200, must be <= control_frequency_hz. Running
# the estimator at the control rate is both noisy (central differences
# over 2 ms) and wasteful; 125 Hz matches the paper's sampling rate.
estimate_frequency_hz = 125.0

# TODO minimum number of valid (curvature, speed) pairs in the window
# before the fit is trusted [count], recommended 10 to 30.
power_law_min_sample_count = 10

# TODO speed below which a sample carries no usable direction or
# curvature information [m/s], recommended 0.003 to 0.02. The paper uses
# 0.005 m/s to separate movement from rest.
minimum_speed_m_s = 0.005

# TODO curvature band kept in the fit [1/m], both positive with
# min < max. Samples outside the band are excluded rather than clamped,
# because clamped samples pile up on the bounds and bias beta.
curvature_min_1_m = 0.05
curvature_max_1_m = 200.0

# TODO power-law exponent bounds [unitless]. The paper derives a
# spectrum of exponents between 0 and 2/3 for simple curved movements,
# with 1/3 the classic two-thirds power law.
beta_min = 0.0
beta_max = 2.0 / 3.0
beta_default = 1.0 / 3.0

# TODO low-pass alpha for the fitted beta and log K [0, 1]; smaller is
# smoother. The paper low-pass filters beta before use. K is filtered in
# log space because it spans orders of magnitude.
power_law_lpf_alpha = 0.05

# --- Guidance strength and gating ---
# TODO gamma speed gains [s^2/m^2] in (13), non-negative. Paper values
# are 2 and 4. Because gamma grows with speed *squared*, these give only
# about 5% anisotropy at 0.15 m/s; raise them for a stronger guidance
# effect and re-check stability.
gamma1_speed_gain = 2.0
gamma2_speed_gain = 4.0

# TODO fit-quality window [unitless R^2, 0 to 1] with lower < upper.
# Guidance stays off at or below the lower value and reaches full
# strength at or above the upper one, ramping linearly between.
fit_quality_r2_lower = 0.20
fit_quality_r2_upper = 0.60

# TODO turn evidence giving full confidence [unitless, 0 to 1]. The
# evidence is |sin| of the angle between velocity and force. For a
# circle traced at 0.15 m/s with these defaults the centripetal term is
# only about 15% of the total force, so 0.15 is a reasonable threshold;
# lower it if guidance feels too weak on gentle curves.
turn_full_confidence = 0.15

# TODO low-pass alpha for the signed turn evidence [0, 1]. Also sets how
# fast guidance decays once the human stops turning.
turn_signal_lpf_alpha = 0.05

# TODO speed at which the speed gate reaches full guidance [m/s], must
# exceed minimum_speed_m_s. Below it the guidance weight ramps down.
guidance_full_speed_m_s = 0.03

# --- Intended acceleration search, paper (10) ---
# TODO tangential acceleration search range [m/s^2], positive.
tangential_acc_limit_m_s2 = 1.0

# TODO cap on the commanded normal acceleration [m/s^2], positive. Also
# bounds how hard the guidance can turn the robot.
max_normal_acc_m_s2 = 2.0

# TODO number of tangential candidates searched [count], odd keeps a
# zero-acceleration candidate in the grid. Recommended 11 to 41.
tangential_search_count = 21

# TODO maximum |log| deviation of the target curvature from the measured
# curvature [unitless], positive. Bounds how far the power-law inversion
# may extrapolate; 0.7 is about a factor of 2 in curvature.
target_curvature_max_log_deviation = 0.7

# TODO beta below which the power-law inversion is too ill-conditioned
# to use [unitless]. Below this the measured curvature is used directly.
minimum_beta_for_inversion = 0.05

# TODO cosine above which the force is treated as aligned with the
# velocity, disabling the "stay between velocity and force" constraint
# of (10) because the enclosed angle is then numerically meaningless.
force_alignment_cosine_cutoff = 0.98

# TODO low-pass alpha for the intended force [0, 1]. The paper filters
# the intended force at 5 Hz, the upper limit of human motion bandwidth.
# For a first-order filter, alpha = 1 - exp(-2*pi*fc*dt).
intended_force_lpf_alpha = 0.06

# TODO maximum rotation rate of the intended direction [rad/s],
# positive. Keeps the damping frame from snapping when the intended
# force passes near zero.
intended_direction_max_rate_rad_s = 6.0

# --- Force conditioning ---
force_lpf_alpha = 0.2  # TODO filter alpha [0, 1]; smaller is smoother
force_deadband_n = 1.0  # TODO radial force deadband [N], 0 to 5
force_bias_samples = 250  # TODO unloaded samples [count], >= 100

# TODO set signs so a positive push produces positive base-frame motion.
force_sign_x = 1.0  # TODO x sign [unitless], valid values: -1 or +1
force_sign_y = 1.0  # TODO y sign [unitless], valid values: -1 or +1

# TODO low-pass alpha for the measured TCP velocity [0, 1]. Used only
# for the power-law estimate and the guidance geometry, never for the
# admittance integration itself.
velocity_lpf_alpha = 0.2

# --- Motion limits and workspace ---
max_cart_speed_m_s = 0.15  # TODO planar speed limit [m/s], 0.01 to 0.15

# Axis-aligned rectangular workspace in the UR5e base-frame x-y plane.
# TODO lower [x, y] corner offsets from the startup TCP [m]. Each value
# must be smaller than the matching upper-corner value.
workspace_lower_corner_offset_m = [-0.2, -0.2]
# TODO upper [x, y] corner offsets from the startup TCP [m].
workspace_upper_corner_offset_m = [0.25, 0.25]

# --- Robot, timing, and startup ---
control_frequency_hz = 500.0  # TODO UR5e native control rate [Hz]
speedl_acceleration_m_s2 = 0.5  # TODO speedL acceleration [m/s^2]
speed_stop_deceleration_m_s2 = 2.0  # TODO stop deceleration [m/s^2]

max_runtime_s = 60.0  # TODO automatic stop time [s], must be positive
subject_preparation_time_s = 1.0  # TODO delay before recording [s], >= 0

move_to_start_configuration = True  # TODO bool; False skips startup move
start_joint_q_deg = [  # TODO joint angles [deg], within robot limits
    -77.64,
    -98.5,
    -95.80,
    -75.75,
    90.3,
    -2.75,
]
start_joint_speed_rad_s = 0.5  # TODO moveJ speed [rad/s], 0.1 to 1
start_joint_acceleration_rad_s2 = 0.5  # TODO moveJ accel [rad/s^2]

# --- Tobii recording ---
enable_tobii_recording = True  # TODO bool; requires a connected tracker
tobii_queue_size = 50000  # TODO callback buffer [samples], >= 10000

# --- CoppeliaSim marker bridge ---
# Runs in a separate process so ZeroMQ latency can never stall the
# control loop.
send_coppeliasim_trajectory = True  # TODO requires CoppeliaSim running
coppeliasim_host = "127.0.0.1"  # TODO ZeroMQ host [hostname or IPv4]
coppeliasim_port = 23000  # TODO ZeroMQ port [integer], default 23000
coppeliasim_object_path = "/Sphere"  # TODO exact marker path in scene
coppeliasim_reference_object_path = ""  # TODO empty means world frame
coppeliasim_update_frequency_hz = 25.0  # TODO marker update rate [Hz]

coppeliasim_position_offset_m = [0.0, 0.0, 0.0]  # TODO offset [m]
coppeliasim_position_signs = [1.0, 1.0, 1.0]  # TODO signs, each -1 or +1


# -------------------- Derived Settings --------------------
dt = 1.0 / control_frequency_hz
start_joint_q = [math.radians(angle_deg) for angle_deg in start_joint_q_deg]

# One estimator update every estimate_period_steps control cycles.
estimate_period_steps = max(1, int(round(
    control_frequency_hz / estimate_frequency_hz)))
estimate_dt = estimate_period_steps * dt

# Window length in decimated samples. Two extra samples are kept so the
# oldest and newest entries can still be central-differenced.
power_law_sample_count = max(3, int(round(
    power_law_window_s / estimate_dt)))
velocity_history_max_len = power_law_sample_count + 2

# One classifier update every classifier_period_steps control cycles.
classifier_period_steps = max(1, int(round(
    control_frequency_hz / classifier_update_frequency_hz)))

# The robot buffer gets one sample per control tick and no eye tracker
# in this setup samples faster, so one length bounds both streams. The
# window itself is trimmed by TIME when a snapshot is taken, not by
# count, so a slower gaze stream still spans the full window.
classifier_history_max_len = int(round(
    classifier_window_s * control_frequency_hz)) + 8

# Largest bd the controller can actually command. With gating on, the
# upper bound is an interpolation between the two pairs' upper values,
# so the reachable maximum is the larger of the two -- the stability
# check below has to use this, not damping_max_n_s_m.
if enable_classifier_gating:
    damping_upper_reachable_n_s_m = max(
        fine_damping_bounds_n_s_m[1], gross_damping_bounds_n_s_m[1])
else:
    damping_upper_reachable_n_s_m = damping_max_n_s_m


# -------------------- Setting Checks --------------------
# Only conditions that would silently produce wrong physics or unsafe
# motion are checked here, per this project's "no defensive scaffolding"
# rule.
if damping_min_n_s_m <= 0.0 or damping_max_n_s_m <= damping_min_n_s_m:
    raise ValueError("Require 0 < damping_min_n_s_m < damping_max_n_s_m.")

if force_max_n <= force_min_n:
    raise ValueError("Require force_min_n < force_max_n.")

if virtual_mass_kg <= 0.0:
    raise ValueError("Require virtual_mass_kg > 0.")

if curvature_min_1_m <= 0.0 or curvature_max_1_m <= curvature_min_1_m:
    raise ValueError(
        "Require 0 < curvature_min_1_m < curvature_max_1_m.")

if estimate_frequency_hz > control_frequency_hz:
    raise ValueError(
        "Require estimate_frequency_hz <= control_frequency_hz.")

if guidance_full_speed_m_s <= minimum_speed_m_s:
    raise ValueError(
        "Require guidance_full_speed_m_s > minimum_speed_m_s.")

if fit_quality_r2_upper <= fit_quality_r2_lower:
    raise ValueError(
        "Require fit_quality_r2_lower < fit_quality_r2_upper.")

if any(
    lower_value >= upper_value
    for lower_value, upper_value in zip(
        workspace_lower_corner_offset_m,
        workspace_upper_corner_offset_m,
    )
):
    raise ValueError(
        "Each workspace lower-corner offset must be less than its "
        "upper-corner offset."
    )

# Explicit discrete-stability check for the admittance integration. The
# forward-Euler update v <- v + (F - B v) / M * dt is stable only while
# B * dt / M < 2; the factor 2 margin keeps it well inside that bound.
if enable_classifier_gating:
    # An inverted pair would make bd_min > bd_max and flip the sense of
    # (7) -- large force would then RAISE damping.
    if fine_damping_bounds_n_s_m[0] <= 0.0 or \
            fine_damping_bounds_n_s_m[1] <= fine_damping_bounds_n_s_m[0]:
        raise ValueError(
            "Require 0 < fine_damping_bounds_n_s_m[0] < "
            "fine_damping_bounds_n_s_m[1].")
    if gross_damping_bounds_n_s_m[0] <= 0.0 or \
            gross_damping_bounds_n_s_m[1] <= gross_damping_bounds_n_s_m[0]:
        raise ValueError(
            "Require 0 < gross_damping_bounds_n_s_m[0] < "
            "gross_damping_bounds_n_s_m[1].")
    if not 0.0 <= fine_probability_limits[0] < fine_probability_limits[1] \
            <= 1.0:
        raise ValueError(
            "Require 0 <= fine_probability_limits[0] < "
            "fine_probability_limits[1] <= 1.")
    if fixed_fine_probability is not None and \
            not 0.0 <= fixed_fine_probability <= 1.0:
        raise ValueError(
            "fixed_fine_probability must be None or within [0, 1].")

_worst_case_damping_n_s_m = damping_upper_reachable_n_s_m * (
    1.0 + gamma2_speed_gain * max_cart_speed_m_s ** 2)
if _worst_case_damping_n_s_m * dt / virtual_mass_kg > 1.0:
    raise ValueError(
        "Admittance integration is too stiff for this control period: "
        "reduce damping_max_n_s_m or gamma2_speed_gain, raise "
        "virtual_mass_kg, or raise control_frequency_hz."
    )


# -------------------- Controller State --------------------
keep_running = True

force_bias_n = [0.0, 0.0]
force_raw_n = [0.0, 0.0]
force_filtered_n = [0.0, 0.0]

# Virtual admittance velocity state; this is what is commanded.
cart_velocity_command_m_s = [0.0, 0.0]

# Measured TCP velocity, raw and low-pass filtered.
velocity_measured_m_s = [0.0, 0.0]
velocity_filtered_m_s = [0.0, 0.0]

start_position_m = [0.0, 0.0]

damping_desired_n_s_m = damping_max_n_s_m
damping_applied_n_s_m = damping_max_n_s_m

# Power-law estimate state.
beta_hat = beta_default
log_gain_hat = 0.0
gain_hat = 1.0
power_law_r2 = 0.0
power_law_rmse = 0.0
power_law_sample_used = 0
power_law_valid = False

curvature_measured_1_m = 0.0
curvature_target_1_m = 0.0

# Measured planar acceleration at the newest usable history sample.
acceleration_measured_m_s2 = [0.0, 0.0]

turn_signal = 0.0
guidance_weight = 0.0
gamma1 = 1.0
gamma2 = 1.0

intended_acc_m_s2 = [0.0, 0.0]
intended_force_n = [0.0, 0.0]
intended_direction_angle_rad = 0.0
intended_direction_xy = [1.0, 0.0]

# Row-major 2x2 planar damping matrix [b11, b12, b21, b22] in N s/m.
damping_matrix_n_s_m = [
    damping_max_n_s_m, 0.0,
    0.0, damping_max_n_s_m,
]

# Decimated history of (time_s, vx, vy) using the filtered velocity.
velocity_history = deque(maxlen=velocity_history_max_len)

# --- Classifier gating state ---
# The control loop and the Tobii callback only ever APPEND to the two
# buffers and read one float back; the classifier thread does the work.
# classifier_lock guards both directions.
classifier_lock = threading.Lock()
classifier_robot_history = deque(maxlen=classifier_history_max_len)
classifier_gaze_history = deque(maxlen=classifier_history_max_len)

fine_probability = warmup_fine_probability
fine_probability_time_s = None
fine_probability_applied = float("nan")
fine_probability_age_s = float("nan")
classifier_status = "warmup"
classifier_compute_ms = 0.0
classifier_update_count = 0
classifier_feature_values = {}

# The (7) endpoints actually in force this tick.
damping_lower_bound_n_s_m = damping_min_n_s_m
damping_upper_bound_n_s_m = damping_max_n_s_m

# Filled by load_classifier() before the robot is touched.
classifier_bundle = None
classifier_feature_functions = []
positive_class_index = 0
tobii_reference_xy_mm = (0.0, 0.0)
coppelia_reference_xy_m = (0.0, 0.0)


def handle_stop_signal(_signal_number, _frame):
    """Stop the control loop when Ctrl+C is pressed."""

    global keep_running
    keep_running = False


# -------------------- Small Math Helpers --------------------
def clamp_value(value, lower_limit, upper_limit):
    """Return a scalar limited to [lower_limit, upper_limit]."""

    if value < lower_limit:
        return lower_limit

    if value > upper_limit:
        return upper_limit

    return value


def vec2_norm(vector):
    """Return the Euclidean norm of a 2-D vector."""

    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])


def vec2_dot(vector_a, vector_b):
    """Return the dot product of two 2-D vectors."""

    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]


def vec2_cross(vector_a, vector_b):
    """Return the scalar cross product a_x*b_y - a_y*b_x."""

    return vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]


def vec2_unit(vector, fallback_unit_vector):
    """Return the unit vector, or the fallback if too close to zero."""

    vector_norm = vec2_norm(vector)

    if vector_norm < 1e-9:
        return list(fallback_unit_vector)

    return [vector[0] / vector_norm, vector[1] / vector_norm]


def vec2_left_normal(vector):
    """Return the vector rotated by +90 deg in the x-y plane."""

    return [-vector[1], vector[0]]


def limit_vector_norm_2d(vector, max_norm):
    """Limit a 2-D vector magnitude while preserving its direction."""

    vector_norm = vec2_norm(vector)

    if vector_norm <= max_norm:
        return list(vector)

    scale = max_norm / vector_norm
    return [scale * vector[0], scale * vector[1]]


def linear_ramp(value, ramp_start, ramp_end):
    """Map value onto [0, 1], ramping linearly between the two bounds.

    Returns 0 at or below ramp_start and 1 at or above ramp_end. Used
    everywhere a continuous gate is needed instead of a hard threshold,
    so no controller quantity can step discontinuously.
    """

    if ramp_end <= ramp_start:
        return 1.0 if value >= ramp_end else 0.0

    return clamp_value(
        (value - ramp_start) / (ramp_end - ramp_start), 0.0, 1.0)


def wrap_angle_rad(angle_rad):
    """Wrap an angle into [-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


# -------------------- Robot I/O --------------------
def read_tcp_force_xy(rtde_receive_interface):
    """Read base-frame TCP force [Fx, Fy] in N from the UR5e.

    getActualTCPForce() already reports the wrench in the robot base
    frame, so unlike the Schunk implementation no rotation by the
    end-effector orientation is applied here.
    """

    tcp_wrench_n = rtde_receive_interface.getActualTCPForce()

    return [
        force_sign_x * tcp_wrench_n[0],
        force_sign_y * tcp_wrench_n[1],
    ]


def read_tcp_position_xyz(rtde_receive_interface):
    """Read base-frame TCP position [x, y, z] in m from the UR5e."""

    tcp_pose = rtde_receive_interface.getActualTCPPose()

    return [tcp_pose[0], tcp_pose[1], tcp_pose[2]]


def read_tcp_actual_velocity(rtde_receive_interface):
    """Read the actual base-frame TCP velocity from the UR5e.

    Returns:
        [vx, vy, vz, wx, wy, wz] with linear components [m/s] and
        angular components [rad/s].
    """

    tcp_actual_speed = rtde_receive_interface.getActualTCPSpeed()

    return [
        tcp_actual_speed[0],
        tcp_actual_speed[1],
        tcp_actual_speed[2],
        tcp_actual_speed[3],
        tcp_actual_speed[4],
        tcp_actual_speed[5],
    ]


def estimate_force_bias(rtde_receive_interface):
    """Average unloaded startup force to estimate the planar bias [N]."""

    global force_bias_n

    force_bias_n = [0.0, 0.0]

    for _sample_idx in range(force_bias_samples):
        measured_force_n = read_tcp_force_xy(rtde_receive_interface)

        force_bias_n[0] += measured_force_n[0]
        force_bias_n[1] += measured_force_n[1]

        time.sleep(dt)

    force_bias_n[0] /= force_bias_samples
    force_bias_n[1] /= force_bias_samples


def update_force_filter(rtde_receive_interface):
    """Bias, radially deadband, then low-pass the planar force [N].

    The deadband is applied to the raw force *before* filtering, and
    acts on the planar magnitude rather than each axis independently:

        f_out = f_raw * max(0, |f_raw| - deadband) / |f_raw|

    This shrinks the vector toward zero without changing its direction
    and is continuous at the threshold, unlike a per-axis hard cutoff.
    """

    measured_force_n = read_tcp_force_xy(rtde_receive_interface)

    force_raw_n[0] = measured_force_n[0] - force_bias_n[0]
    force_raw_n[1] = measured_force_n[1] - force_bias_n[1]

    raw_magnitude_n = vec2_norm(force_raw_n)

    if raw_magnitude_n <= force_deadband_n:
        deadbanded_force_n = [0.0, 0.0]
    else:
        shrink_scale = (
            raw_magnitude_n - force_deadband_n) / raw_magnitude_n
        deadbanded_force_n = [
            shrink_scale * force_raw_n[0],
            shrink_scale * force_raw_n[1],
        ]

    for axis in range(2):
        force_filtered_n[axis] = (
            (1.0 - force_lpf_alpha) * force_filtered_n[axis] +
            force_lpf_alpha * deadbanded_force_n[axis]
        )


def update_velocity_filter(tcp_actual_velocity):
    """Low-pass the measured planar TCP velocity [m/s].

    The filtered velocity feeds the power-law estimate and the guidance
    geometry only. The admittance integration uses its own virtual
    velocity state, so this filter cannot add lag to the commanded
    motion.
    """

    velocity_measured_m_s[0] = tcp_actual_velocity[0]
    velocity_measured_m_s[1] = tcp_actual_velocity[1]

    for axis in range(2):
        velocity_filtered_m_s[axis] = (
            (1.0 - velocity_lpf_alpha) * velocity_filtered_m_s[axis] +
            velocity_lpf_alpha * velocity_measured_m_s[axis]
        )


# -------------------- Indirect Intention (paper 4, 8, 9) -------------
def push_velocity_history(time_s):
    """Append one decimated filtered-velocity sample to the history."""

    velocity_history.append((
        time_s,
        velocity_filtered_m_s[0],
        velocity_filtered_m_s[1],
    ))


def estimate_power_law_from_history():
    """Fit v = K * kappa^(-beta) over the sliding window, paper (9).

    Curvature of each interior history sample comes from paper (8),

        kappa = |vx*ay - vy*ax| / |v|^3

    with acceleration from a central difference over the *decimated*
    timestamps. Taking logarithms of the power law gives

        log(v) = log(K) - beta * log(kappa)

    which is fitted by ordinary least squares. Samples slower than
    minimum_speed_m_s or outside the curvature band are excluded, not
    clamped, so they cannot pile up on a bound and bias the slope.

    Outputs (module state):
        beta_hat, log_gain_hat, gain_hat: filtered power-law parameters.
        power_law_r2, power_law_rmse, power_law_sample_used: fit quality.
        power_law_valid: whether this window produced a usable fit.
        curvature_measured_1_m, acceleration_measured_m_s2: newest valid
            curvature and acceleration, used by the guidance geometry.

    Note on statistics: consecutive samples are strongly autocorrelated,
    so R^2 here indicates how power-law-like the recent path is, not a
    formal goodness of fit with power_law_sample_used degrees of
    freedom.
    """

    global beta_hat, log_gain_hat, gain_hat
    global power_law_r2, power_law_rmse, power_law_sample_used
    global power_law_valid
    global curvature_measured_1_m, acceleration_measured_m_s2

    history = list(velocity_history)
    history_size = len(history)

    log_curvature_values = []
    log_speed_values = []
    newest_curvature_1_m = None
    newest_acceleration_m_s2 = None

    for sample_idx in range(1, history_size - 1):
        previous_time_s, previous_vx, previous_vy = history[sample_idx - 1]
        _current_time_s, current_vx, current_vy = history[sample_idx]
        next_time_s, next_vx, next_vy = history[sample_idx + 1]

        central_interval_s = next_time_s - previous_time_s
        if central_interval_s <= 0.0:
            continue

        acceleration_x_m_s2 = (next_vx - previous_vx) / central_interval_s
        acceleration_y_m_s2 = (next_vy - previous_vy) / central_interval_s

        speed_m_s = math.hypot(current_vx, current_vy)
        if speed_m_s < minimum_speed_m_s:
            continue

        # Newest usable acceleration, kept for the minimum-jerk cost.
        newest_acceleration_m_s2 = [
            acceleration_x_m_s2,
            acceleration_y_m_s2,
        ]

        velocity_cross_acceleration = (
            current_vx * acceleration_y_m_s2 -
            current_vy * acceleration_x_m_s2
        )
        curvature_1_m = abs(velocity_cross_acceleration) / speed_m_s ** 3

        if (curvature_1_m < curvature_min_1_m or
                curvature_1_m > curvature_max_1_m):
            continue

        newest_curvature_1_m = curvature_1_m
        log_curvature_values.append(math.log(curvature_1_m))
        log_speed_values.append(math.log(speed_m_s))

    if newest_acceleration_m_s2 is not None:
        acceleration_measured_m_s2 = newest_acceleration_m_s2

    if newest_curvature_1_m is not None:
        curvature_measured_1_m = newest_curvature_1_m

    power_law_sample_used = len(log_curvature_values)

    if power_law_sample_used < power_law_min_sample_count:
        power_law_r2 = 0.0
        power_law_rmse = 0.0
        power_law_valid = False
        return

    sample_count = float(power_law_sample_used)
    sum_x = math.fsum(log_curvature_values)
    sum_y = math.fsum(log_speed_values)
    sum_xx = math.fsum(x * x for x in log_curvature_values)
    sum_xy = math.fsum(
        x * y for x, y in zip(log_curvature_values, log_speed_values))

    denominator = sample_count * sum_xx - sum_x * sum_x

    # A near-zero denominator means every sample sits at essentially the
    # same curvature, so the slope is unidentifiable from this window.
    if abs(denominator) < 1e-9:
        power_law_r2 = 0.0
        power_law_rmse = 0.0
        power_law_valid = False
        return

    slope = (sample_count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / sample_count
    mean_log_speed = sum_y / sample_count

    residual_square_sum = math.fsum(
        (y - (intercept + slope * x)) ** 2
        for x, y in zip(log_curvature_values, log_speed_values))
    total_square_sum = math.fsum(
        (y - mean_log_speed) ** 2 for y in log_speed_values)

    if total_square_sum > 1e-12:
        power_law_r2 = 1.0 - residual_square_sum / total_square_sum
    else:
        power_law_r2 = 0.0

    power_law_rmse = math.sqrt(residual_square_sum / sample_count)

    # log(v) = log(K) - beta * log(kappa), so beta is the negated slope.
    beta_measured = clamp_value(-slope, beta_min, beta_max)

    beta_hat = (
        (1.0 - power_law_lpf_alpha) * beta_hat +
        power_law_lpf_alpha * beta_measured
    )
    log_gain_hat = (
        (1.0 - power_law_lpf_alpha) * log_gain_hat +
        power_law_lpf_alpha * intercept
    )
    gain_hat = math.exp(log_gain_hat)

    power_law_valid = True


def compute_target_curvature(speed_m_s):
    """Invert the power law for the curvature the human is drawing.

    From v = K * kappa^(-beta),

        log(kappa) = (log(K) - log(v)) / beta

    The inversion is done in log space rather than as pow(K/v, 1/beta),
    because the direct form raises any gain error to the power 1/beta:
    a 20% error at beta = 0.02 becomes a factor of 1.2^50, roughly
    9000. The result is additionally held within
    target_curvature_max_log_deviation of the measured curvature, so a
    momentarily bad fit cannot command an extreme turn.

    Inputs:
        speed_m_s: current planar speed [m/s], must be positive.

    Returns:
        Target curvature [1/m], within [curvature_min_1_m,
        curvature_max_1_m].
    """

    measured_curvature_1_m = clamp_value(
        curvature_measured_1_m, curvature_min_1_m, curvature_max_1_m)

    if (not power_law_valid or
            beta_hat < minimum_beta_for_inversion or
            speed_m_s < minimum_speed_m_s):
        return measured_curvature_1_m

    log_curvature = (log_gain_hat - math.log(speed_m_s)) / beta_hat

    log_measured_curvature = math.log(measured_curvature_1_m)
    log_curvature = clamp_value(
        log_curvature,
        log_measured_curvature - target_curvature_max_log_deviation,
        log_measured_curvature + target_curvature_max_log_deviation,
    )
    log_curvature = clamp_value(
        log_curvature,
        math.log(curvature_min_1_m),
        math.log(curvature_max_1_m),
    )

    return math.exp(log_curvature)


def update_turn_signal():
    """Update the filtered signed turn evidence in [-1, 1].

    The evidence for one sample is

        sin(angle from velocity to force) = (v x F) / (|v| |F|)

    whose sign is the turn direction and whose magnitude says how
    unambiguous that turn is. Filtering it means the signal decays
    toward zero when the human pushes straight along the velocity, so
    unlike the persistent turn sign of the C++ version the controller
    stops commanding normal acceleration once the turn evidence stops.
    """

    global turn_signal

    speed_m_s = vec2_norm(velocity_filtered_m_s)
    force_magnitude_n = vec2_norm(force_filtered_n)

    if speed_m_s < minimum_speed_m_s or force_magnitude_n < 1e-9:
        turn_evidence = 0.0
    else:
        turn_evidence = vec2_cross(
            velocity_filtered_m_s, force_filtered_n) / (
                speed_m_s * force_magnitude_n)
        turn_evidence = clamp_value(turn_evidence, -1.0, 1.0)

    turn_signal = (
        (1.0 - turn_signal_lpf_alpha) * turn_signal +
        turn_signal_lpf_alpha * turn_evidence
    )


def update_guidance_weight():
    """Combine the gates into one continuous guidance weight in [0, 1].

    The weight multiplies the anisotropic damping and the intended
    acceleration, so guidance fades in and out instead of switching.
    All three factors must agree for guidance to reach full strength:
    the power-law fit must be meaningful, the robot must actually be
    moving, and there must be recent unambiguous turn evidence.
    """

    global guidance_weight

    if not power_law_valid:
        guidance_weight = 0.0
        return

    fit_weight = linear_ramp(
        power_law_r2, fit_quality_r2_lower, fit_quality_r2_upper)
    speed_weight = linear_ramp(
        vec2_norm(velocity_filtered_m_s),
        minimum_speed_m_s, guidance_full_speed_m_s)
    turn_weight = linear_ramp(abs(turn_signal), 0.0, turn_full_confidence)

    guidance_weight = fit_weight * speed_weight * turn_weight


# -------------------- Intended Acceleration (paper 10) --------------
def candidate_respects_direction(velocity_m_s, force_n, candidate_m_s2):
    """Check the two direction constraints of paper (10).

    With v(k+1) = v(k) + a * dt from the forward Euler step, the
    constraints require that the next velocity turns the same way the
    force indicates,

        (v x F) * (v x v_next) > 0

    and that the next velocity stays within the angle between the
    current velocity and the force,

        cos(v, v_next) > cos(v, F)

    Inputs:
        velocity_m_s: current filtered planar velocity [m/s].
        force_n: filtered planar human force [N].
        candidate_m_s2: candidate intended acceleration [m/s^2].

    Returns:
        True when the candidate is feasible.
    """

    next_velocity_m_s = [
        velocity_m_s[0] + candidate_m_s2[0] * dt,
        velocity_m_s[1] + candidate_m_s2[1] * dt,
    ]

    speed_m_s = vec2_norm(velocity_m_s)
    next_speed_m_s = vec2_norm(next_velocity_m_s)
    force_magnitude_n = vec2_norm(force_n)

    # With any of these degenerate the enclosed angle is meaningless, so
    # the constraints carry no information and everything is feasible.
    if (speed_m_s < minimum_speed_m_s or
            next_speed_m_s < minimum_speed_m_s or
            force_magnitude_n < 1e-9):
        return True

    force_cross = vec2_cross(velocity_m_s, force_n)
    next_cross = vec2_cross(velocity_m_s, next_velocity_m_s)

    if abs(force_cross) > 1e-9 and force_cross * next_cross <= 0.0:
        return False

    cosine_next = vec2_dot(velocity_m_s, next_velocity_m_s) / (
        speed_m_s * next_speed_m_s)
    cosine_force = vec2_dot(velocity_m_s, force_n) / (
        speed_m_s * force_magnitude_n)

    # When the force is nearly along the velocity the enclosed angle is
    # numerically meaningless, so this constraint is skipped.
    if (cosine_force < force_alignment_cosine_cutoff and
            cosine_next <= cosine_force):
        return False

    return True


def compute_intended_acceleration():
    """Solve the local minimum-jerk problem of paper (10).

    The paper states a constrained optimization without giving a solver.
    The structure is used explicitly instead:
        - the normal acceleration magnitude comes from the power-law
          target curvature, |a_n| = v^2 * kappa;
        - the turn direction comes from the filtered turn signal;
        - the tangential acceleration is picked from a uniform grid by
          the paper's objective, which minimizes jerk against the
          *measured* previous acceleration.

    Outputs (module state):
        intended_acc_m_s2, curvature_target_1_m.
    """

    global intended_acc_m_s2, curvature_target_1_m

    speed_m_s = vec2_norm(velocity_filtered_m_s)

    if guidance_weight <= 0.0 or speed_m_s < minimum_speed_m_s:
        # Decay rather than freeze, so a stale intended acceleration
        # cannot snap back when guidance resumes.
        intended_acc_m_s2 = [
            (1.0 - intended_force_lpf_alpha) * intended_acc_m_s2[0],
            (1.0 - intended_force_lpf_alpha) * intended_acc_m_s2[1],
        ]
        curvature_target_1_m = 0.0
        return

    tangent_xy = vec2_unit(velocity_filtered_m_s, [1.0, 0.0])
    normal_xy = vec2_left_normal(tangent_xy)

    curvature_target_1_m = compute_target_curvature(speed_m_s)

    normal_acc_m_s2 = speed_m_s * speed_m_s * curvature_target_1_m
    normal_acc_m_s2 = clamp_value(
        normal_acc_m_s2, 0.0, max_normal_acc_m_s2)

    turn_sign = 1.0 if turn_signal >= 0.0 else -1.0
    signed_normal_acc_m_s2 = turn_sign * normal_acc_m_s2

    best_cost = None
    best_candidate_m_s2 = None

    for candidate_idx in range(tangential_search_count):
        if tangential_search_count > 1:
            ratio = candidate_idx / float(tangential_search_count - 1)
        else:
            ratio = 0.5

        # Uniform grid over the full symmetric range. The C++ version
        # re-centered this grid on the previous tangential acceleration
        # and then clamped it, which collapsed points onto the limit and
        # biased the search toward never changing.
        tangent_acc_m_s2 = (
            -tangential_acc_limit_m_s2 +
            2.0 * tangential_acc_limit_m_s2 * ratio
        )

        candidate_m_s2 = [
            tangent_acc_m_s2 * tangent_xy[0] +
            signed_normal_acc_m_s2 * normal_xy[0],
            tangent_acc_m_s2 * tangent_xy[1] +
            signed_normal_acc_m_s2 * normal_xy[1],
        ]

        if not candidate_respects_direction(
                velocity_filtered_m_s, force_filtered_n, candidate_m_s2):
            continue

        # Paper (10) objective: squared jerk approximated as
        # (a_candidate - a_measured_previous) / T.
        difference_m_s2 = [
            candidate_m_s2[0] - acceleration_measured_m_s2[0],
            candidate_m_s2[1] - acceleration_measured_m_s2[1],
        ]
        cost = vec2_dot(difference_m_s2, difference_m_s2)

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_candidate_m_s2 = candidate_m_s2

    if best_candidate_m_s2 is not None:
        intended_acc_m_s2 = best_candidate_m_s2
        return

    # No feasible candidate: keep the power-law normal acceleration and
    # carry the measured tangential acceleration through unchanged.
    measured_tangent_acc_m_s2 = vec2_dot(
        acceleration_measured_m_s2, tangent_xy)
    measured_tangent_acc_m_s2 = clamp_value(
        measured_tangent_acc_m_s2,
        -tangential_acc_limit_m_s2,
        tangential_acc_limit_m_s2,
    )

    intended_acc_m_s2 = [
        measured_tangent_acc_m_s2 * tangent_xy[0] +
        signed_normal_acc_m_s2 * normal_xy[0],
        measured_tangent_acc_m_s2 * tangent_xy[1] +
        signed_normal_acc_m_s2 * normal_xy[1],
    ]


# -------------------- Classifier Gating -----------------------------
def resolve_feature_functions(feature_columns, config):
    """Look each of the bundle's feature names up in the real feature
    banks, so nothing is reimplemented here.

    Inputs:
        feature_columns: the bundle's own list, e.g.
            "gaze_gaze_path_efficiency" -- a "<bank>_" prefix followed
            by that bank's own feature name, exactly as
            build_modality_feature_datasets.py prefixes them for the
            "all" table.
        config: cfb.FeatureConfig for the banks.

    Returns:
        list of (column_name, kind, feature_fn), kind naming the call
        signature to use: "gaze", "robot", "joint", or "dsj"/"speed"
        for the two robot values that script adds outside the bank.

    Raises:
        ValueError naming every column that cannot be resolved -- a
        model whose features cannot be computed online must fail here,
        not turn into a silent NaN at 500 Hz.
    """

    banks = {
        "gaze": dict(cfb.build_gaze_feature_bank(config)),
        "robot": dict(cfb.build_robot_feature_bank(config)),
        "joint": dict(cfb.build_joint_feature_bank(config)),
    }
    extras = {"robot_DSJ": "dsj", "robot_mean_ee_speed_m_s": "speed"}

    resolved = []
    unresolved = []
    for column_name in feature_columns:
        if column_name in extras:
            resolved.append((column_name, extras[column_name], None))
            continue

        for bank_name, bank in banks.items():
            prefix = bank_name + "_"
            feature_name = column_name[len(prefix):]
            if column_name.startswith(prefix) and feature_name in bank:
                resolved.append(
                    (column_name, bank_name, bank[feature_name]))
                break
        else:
            unresolved.append(column_name)

    if unresolved:
        raise ValueError(
            f"{read_model_joblib_path.name} needs features that cannot be "
            f"computed online: {unresolved}. Each name must be "
            f"'<gaze|robot|joint>_<feature>' naming a feature that exists "
            f"in classifier_feature_bank.py, or one of {sorted(extras)}.")
    return resolved


def load_classifier():
    """Load the calibration and (unless the sham arm is selected) the
    trained bundle, and resolve its features.

    Called before the robot is touched, so a missing calibration, an
    unknown class, or an uncomputable feature stops the run while the
    arm is still idle.

    Outputs (module state):
        classifier_bundle, classifier_feature_functions,
        positive_class_index, tobii_reference_xy_mm,
        coppelia_reference_xy_m.
    """

    global classifier_bundle, classifier_feature_functions
    global positive_class_index
    global tobii_reference_xy_mm, coppelia_reference_xy_m

    calibration = json.loads(read_calibration_json_path.read_text())
    tobii_reference_xy_mm = tuple(
        calibration["manual_tobii_reference_xy_mm"])
    coppelia_reference_xy_m = tuple(
        calibration["manual_coppelia_reference_xy_m"])
    print("Calibration:", read_calibration_json_path.name)

    if fixed_fine_probability is not None:
        print(
            "Classifier: SHAM arm -- P(Fine) frozen at",
            fixed_fine_probability,
            "(no model loaded)",
        )
        return

    classifier_bundle = joblib.load(read_model_joblib_path)
    class_labels = list(classifier_bundle["classes"])

    if positive_class_label not in class_labels:
        raise ValueError(
            f"positive_class_label '{positive_class_label}' is not one of "
            f"the bundle's classes {class_labels}.")
    positive_class_index = class_labels.index(positive_class_label)

    config = cfb.FeatureConfig(
        workspace_center_xy_m=coppelia_reference_xy_m,
        workspace_half_extent_m=classifier_workspace_half_extent_m,
        low_speed_threshold_m_s=classifier_low_speed_threshold_m_s,
        high_speed_threshold_m_s=classifier_high_speed_threshold_m_s,
        close_thresholds_m=classifier_close_thresholds_m,
    )
    classifier_feature_functions = resolve_feature_functions(
        classifier_bundle["feature_columns"], config)

    print("Classifier:", read_model_joblib_path.name)
    print("  classes:", class_labels, "-> P of", positive_class_label)
    print("  features:", classifier_bundle["feature_columns"])

    # The gaze buffer is filled only by the Tobii callback, so a gaze-
    # dependent bundle without a tracker would sit at
    # warmup_fine_probability for the whole trial without ever saying so.
    needs_gaze = any(
        kind in ("gaze", "joint")
        for _column_name, kind, _feature_fn in classifier_feature_functions)
    if needs_gaze and not enable_tobii_recording:
        raise ValueError(
            "This bundle needs gaze features, but enable_tobii_recording "
            "is False -- gating would silently stay at "
            "warmup_fine_probability for the whole trial.")


def classifier_feature_column_names():
    """The bundle's feature names, or [] on the sham arm."""

    if classifier_bundle is None:
        return []
    return list(classifier_bundle["feature_columns"])


def eye_is_valid(gaze_sample, side):
    """Whether one eye's sample passes the same gates the offline
    pipeline applies before a gaze sample is ever windowed."""

    if gaze_sample.get(f"{side}_gaze_point_validity") != 1:
        return False

    ucs_mm = gaze_sample.get(f"{side}_gaze_point_in_user_coordinate_system")
    if ucs_mm is None or not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in ucs_mm):
        return False

    if require_gaze_on_display:
        display = gaze_sample.get(f"{side}_gaze_point_on_display_area")
        if display is None or not all(
                isinstance(v, (int, float)) and math.isfinite(v) and
                0.0 <= v <= 1.0 for v in display):
            return False

    return True


def extract_gaze_xy_ucs_mm(gaze_sample):
    """One binocular gaze point [mm] in the Tobii user coordinate
    system, or None if this sample fails the validity gates.

    Mirrors build_modality_feature_datasets.py: both eyes averaged when
    require_both_eyes_valid, otherwise whichever eye or eyes are valid.
    """

    left_valid = eye_is_valid(gaze_sample, "left")
    right_valid = eye_is_valid(gaze_sample, "right")
    left_mm = gaze_sample.get("left_gaze_point_in_user_coordinate_system")
    right_mm = gaze_sample.get("right_gaze_point_in_user_coordinate_system")

    if left_valid and right_valid:
        return [0.5 * (left_mm[0] + right_mm[0]),
                0.5 * (left_mm[1] + right_mm[1])]

    if require_both_eyes_valid:
        return None

    if left_valid:
        return [left_mm[0], left_mm[1]]

    if right_valid:
        return [right_mm[0], right_mm[1]]

    return None


def push_classifier_gaze_sample(time_s, gaze_sample):
    """Map one valid gaze sample into the CoppeliaSim frame and buffer
    it. Called from the Tobii SDK callback thread, so it stays to a
    validity check, an affine map, and a deque append."""

    if not enable_classifier_gating:
        return

    gaze_ucs_xy_mm = extract_gaze_xy_ucs_mm(gaze_sample)
    if gaze_ucs_xy_mm is None:
        return

    gaze_xy_m = cfb.map_gaze_ucs_to_coppelia(
        gaze_ucs_xy_mm,
        tobii_reference_xy_mm,
        coppelia_reference_xy_m,
        screen_motion_per_robot_motion,
        tobii_to_coppelia_axis_sign,
    )

    with classifier_lock:
        classifier_gaze_history.append(
            (time_s, float(gaze_xy_m[0]), float(gaze_xy_m[1])))


def push_classifier_robot_sample(
    time_s,
    tcp_position_xyz_m,
    tcp_actual_velocity,
):
    """Buffer one control-tick robot sample for the classifier.

    Holds exactly the quantities build_modality_feature_datasets.py
    reads back out of this controller's own CSV columns, so the feature
    functions see the same streams online that they saw in training.
    """

    if not enable_classifier_gating:
        return

    coppeliasim_position_m = map_tcp_position_to_coppeliasim(
        tcp_position_xyz_m)

    with classifier_lock:
        classifier_robot_history.append((
            time_s,
            coppeliasim_position_m[0],
            coppeliasim_position_m[1],
            tcp_position_xyz_m[0],
            tcp_position_xyz_m[1],
            tcp_actual_velocity[0],
            tcp_actual_velocity[1],
            force_filtered_n[0],
            force_filtered_n[1],
            cart_velocity_command_m_s[0],
            cart_velocity_command_m_s[1],
        ))


def snapshot_classifier_window():
    """Copy both buffers and trim them to the newest
    classifier_window_s, or None while the robot buffer is empty.

    The window is closed on the ROBOT clock (the reliable 500 Hz one)
    and both streams are then selected by time, the same way
    build_modality_feature_datasets.py selects a window.
    """

    with classifier_lock:
        robot_samples = list(classifier_robot_history)
        gaze_samples = list(classifier_gaze_history)

    if not robot_samples:
        return None

    window_end_s = robot_samples[-1][0]
    window_start_s = window_end_s - classifier_window_s

    robot_samples = [
        sample for sample in robot_samples if sample[0] >= window_start_s]
    gaze_samples = [
        sample for sample in gaze_samples
        if window_start_s <= sample[0] <= window_end_s]

    return window_end_s, robot_samples, gaze_samples


def compute_classifier_features(robot_samples, gaze_samples):
    """Every feature the bundle asks for, over one window.

    Wmag is zeros because this controller logs no TCP angular velocity,
    matching what the offline pipeline substitutes for the same missing
    columns (see its optional_numeric_column() note).
    """

    robot = np.asarray(robot_samples, dtype=float)
    gaze = np.asarray(gaze_samples, dtype=float)

    robot_time_s = robot[:, 0]
    Px, Py = robot[:, 1], robot[:, 2]
    base_x_m, base_y_m = robot[:, 3], robot[:, 4]
    Vx, Vy = robot[:, 5], robot[:, 6]
    Fx, Fy = robot[:, 7], robot[:, 8]
    Vcx, Vcy = robot[:, 9], robot[:, 10]
    Wmag = np.zeros_like(robot_time_s)
    Gt, Gx, Gy = gaze[:, 0], gaze[:, 1], gaze[:, 2]

    values = {}
    for column_name, kind, feature_fn in classifier_feature_functions:
        if kind == "gaze":
            values[column_name] = feature_fn(
                Gx, Gy, Gt, Px, Py, robot_time_s)
        elif kind == "robot":
            values[column_name] = feature_fn(
                Px, Py, robot_time_s, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
        elif kind == "joint":
            values[column_name] = feature_fn(
                Gx, Gy, Gt, Px, Py, robot_time_s, Vx, Vy, Fx, Fy, Wmag,
                Vcx, Vcy)
        elif kind == "dsj":
            values[column_name] = cfb.calculate_planar_dsj(
                base_x_m, base_y_m, robot_time_s)
        elif kind == "speed":
            values[column_name] = cfb.calculate_mean_planar_speed(
                Px, Py, robot_time_s)

    return values


def update_fine_probability():
    """One classifier update: snapshot, recompute features, publish.

    Every early return leaves the previous probability in place and
    records why in classifier_status, because a controller cannot drop
    a window the way the offline pipeline does.

    Outputs (module state):
        fine_probability, fine_probability_time_s, classifier_status,
        classifier_feature_values, classifier_compute_ms,
        classifier_update_count.
    """

    global fine_probability, fine_probability_time_s
    global classifier_status, classifier_feature_values
    global classifier_compute_ms, classifier_update_count

    start_time_ns = time.perf_counter_ns()
    snapshot = snapshot_classifier_window()

    if snapshot is None:
        classifier_status = "no_robot_samples"
        return

    window_end_s, robot_samples, gaze_samples = snapshot

    if len(robot_samples) < minimum_classifier_robot_samples:
        classifier_status = "few_robot_samples"
        return

    if len(gaze_samples) < minimum_classifier_gaze_samples:
        classifier_status = "few_gaze_samples"
        return

    feature_values = compute_classifier_features(robot_samples, gaze_samples)
    feature_vector = [feature_values[column_name]
                      for column_name in classifier_bundle["feature_columns"]]

    if not all(math.isfinite(value) for value in feature_vector):
        classifier_feature_values = feature_values
        classifier_status = "nan_feature"
        return

    scaled_features = classifier_bundle["scaler"].transform([feature_vector])
    probability = float(
        classifier_bundle["model"].predict_proba(scaled_features)[
            0, positive_class_index])
    probability = clamp_value(
        probability, fine_probability_limits[0], fine_probability_limits[1])

    with classifier_lock:
        if fine_probability_time_s is None:
            fine_probability = probability
        else:
            fine_probability = (
                (1.0 - fine_probability_lpf_alpha) * fine_probability +
                fine_probability_lpf_alpha * probability
            )
        fine_probability_time_s = window_end_s
        classifier_feature_values = feature_values

    classifier_status = "ok"
    classifier_update_count += 1
    classifier_compute_ms = (
        time.perf_counter_ns() - start_time_ns) * 1e-6


def classifier_thread_main(stop_event):
    """Recompute P(Fine) at classifier_update_frequency_hz until stopped.

    Deliberately off the control loop: a slow or failing update here
    makes the probability STALE (and update_damping_bounds() falls back
    to neutral), where the same work inline would make the 500 Hz loop
    late instead.
    """

    update_period_s = 1.0 / classifier_update_frequency_hz

    while not stop_event.wait(update_period_s):
        try:
            update_fine_probability()
        except Exception as error:
            print("Classifier update failed:", repr(error), flush=True)
            traceback.print_exc()


def start_classifier_thread():
    """Start the classifier thread unless gating or the model is off."""

    if not enable_classifier_gating or fixed_fine_probability is not None:
        return None, None

    stop_event = threading.Event()
    classifier_thread = threading.Thread(
        target=classifier_thread_main,
        args=(stop_event,),
        name="classifier_update",
        daemon=True,
    )
    classifier_thread.start()

    return stop_event, classifier_thread


def stop_classifier_thread(stop_event, classifier_thread):
    """Stop the classifier thread and wait briefly for it to exit."""

    if classifier_thread is None:
        return

    stop_event.set()
    classifier_thread.join(timeout=2.0)


def update_damping_bounds(time_s):
    """Set this tick's (7) endpoints from P(Fine), paper (7) unchanged
    in shape.

        bd_min = Pt * Lf + (1 - Pt) * Lg
        bd_max = Pt * Uf + (1 - Pt) * Ug

    Both are convex combinations of two valid intervals, so
    bd_min < bd_max holds for every Pt. A probability older than
    max_probability_age_s (gaze lost, thread wedged) falls back to
    warmup_fine_probability rather than freezing on a stale class.

    Outputs (module state):
        damping_lower_bound_n_s_m, damping_upper_bound_n_s_m,
        fine_probability_applied, fine_probability_age_s.
    """

    global damping_lower_bound_n_s_m, damping_upper_bound_n_s_m
    global fine_probability_applied, fine_probability_age_s

    if not enable_classifier_gating:
        fine_probability_applied = float("nan")
        fine_probability_age_s = float("nan")
        damping_lower_bound_n_s_m = damping_min_n_s_m
        damping_upper_bound_n_s_m = damping_max_n_s_m
        return

    if fixed_fine_probability is not None:
        probability = fixed_fine_probability
        age_s = 0.0
    else:
        with classifier_lock:
            probability = fine_probability
            published_time_s = fine_probability_time_s

        if published_time_s is None:
            probability = warmup_fine_probability
            age_s = float("nan")
        else:
            age_s = time_s - published_time_s
            if age_s > max_probability_age_s:
                probability = warmup_fine_probability

    fine_probability_applied = probability
    fine_probability_age_s = age_s
    damping_lower_bound_n_s_m = (
        probability * fine_damping_bounds_n_s_m[0] +
        (1.0 - probability) * gross_damping_bounds_n_s_m[0]
    )
    damping_upper_bound_n_s_m = (
        probability * fine_damping_bounds_n_s_m[1] +
        (1.0 - probability) * gross_damping_bounds_n_s_m[1]
    )


# -------------------- Guided Damping (paper 7, 11, 12, 13, 14) ------
def compute_direct_damping(force_magnitude_n, lower_n_s_m, upper_n_s_m):
    """Direct intention, paper (7).

    Inputs:
        force_magnitude_n: planar human force magnitude [N].
        lower_n_s_m, upper_n_s_m: this tick's endpoints, from
            update_damping_bounds() -- the fixed Chen pair with gating
            off, the P(Fine)-interpolated pair with it on. Taking them
            as arguments rather than reading the globals is what lets
            the endpoints move without touching the map's shape.

    Returns:
        Isotropic virtual damping [N s/m] in [lower_n_s_m, upper_n_s_m].

    A larger force means the human wants faster motion, so damping
    decreases linearly from upper_n_s_m at force_min_n to lower_n_s_m
    at force_max_n. The map is affine in (lower, upper) for a fixed
    force, which is why interpolating the endpoints by P(Fine) equals
    interpolating two controllers' commanded damping.
    """

    if force_magnitude_n <= force_min_n:
        return upper_n_s_m

    if force_magnitude_n >= force_max_n:
        return lower_n_s_m

    force_ratio = (force_magnitude_n - force_min_n) / (
        force_max_n - force_min_n)

    return upper_n_s_m - (upper_n_s_m - lower_n_s_m) * force_ratio


def rate_limit_damping(desired_n_s_m, previous_n_s_m):
    """Limit the per-step change of the isotropic damping [N s/m]."""

    max_step_n_s_m = damping_rate_max_n_s_m_per_s * dt
    damping_step = clamp_value(
        desired_n_s_m - previous_n_s_m, -max_step_n_s_m, max_step_n_s_m)

    return previous_n_s_m + damping_step


def update_direct_damping():
    """Update the isotropic damping and the gamma terms, (7) and (13).

    Kept separate from update_damping_matrix() so the data flow follows
    the paper without a stale value anywhere: this sets bd and gamma,
    (11) then uses them to form the intended force and direction, and
    (12)/(14) finally build the matrix from all three.

    Outputs (module state):
        damping_desired_n_s_m, damping_applied_n_s_m, gamma1, gamma2.
    """

    global damping_desired_n_s_m, damping_applied_n_s_m
    global gamma1, gamma2

    force_magnitude_n = vec2_norm(force_filtered_n)
    speed_m_s = vec2_norm(velocity_filtered_m_s)

    damping_desired_n_s_m = compute_direct_damping(
        force_magnitude_n,
        damping_lower_bound_n_s_m,
        damping_upper_bound_n_s_m,
    )
    damping_applied_n_s_m = rate_limit_damping(
        damping_desired_n_s_m, damping_applied_n_s_m)

    gamma1 = 1.0 + gamma1_speed_gain * speed_m_s * speed_m_s
    gamma2 = 1.0 + gamma2_speed_gain * speed_m_s * speed_m_s


def update_intended_direction():
    """Update the intended force and its rate-limited direction.

    Intended force, paper (11):

        F_intended = md * a_intended + (bd / gamma1) * v

    The first term is the force needed to change direction and the
    second is the minimum driving force. The result is low-pass filtered
    at roughly the 5 Hz upper limit of human motion bandwidth, as the
    paper does with a Butterworth filter.

    The direction is then slewed toward the intended force rather than
    set to it, so the damping frame cannot snap around when the intended
    force passes near zero.
    """

    global intended_force_n, intended_direction_angle_rad
    global intended_direction_xy

    intended_force_raw_n = [
        virtual_mass_kg * intended_acc_m_s2[0] +
        (damping_applied_n_s_m / gamma1) * velocity_filtered_m_s[0],
        virtual_mass_kg * intended_acc_m_s2[1] +
        (damping_applied_n_s_m / gamma1) * velocity_filtered_m_s[1],
    ]

    intended_force_n = [
        (1.0 - intended_force_lpf_alpha) * intended_force_n[0] +
        intended_force_lpf_alpha * intended_force_raw_n[0],
        (1.0 - intended_force_lpf_alpha) * intended_force_n[1] +
        intended_force_lpf_alpha * intended_force_raw_n[1],
    ]

    # Target direction: the intended force while guidance is active,
    # otherwise the measured force, otherwise hold the current angle.
    if vec2_norm(intended_force_n) > 1e-9 and guidance_weight > 0.0:
        target_angle_rad = math.atan2(
            intended_force_n[1], intended_force_n[0])
    elif vec2_norm(force_filtered_n) > 1e-9:
        target_angle_rad = math.atan2(
            force_filtered_n[1], force_filtered_n[0])
    else:
        target_angle_rad = intended_direction_angle_rad

    angle_error_rad = wrap_angle_rad(
        target_angle_rad - intended_direction_angle_rad)
    max_angle_step_rad = intended_direction_max_rate_rad_s * dt
    angle_step_rad = clamp_value(
        angle_error_rad, -max_angle_step_rad, max_angle_step_rad)

    intended_direction_angle_rad = wrap_angle_rad(
        intended_direction_angle_rad + angle_step_rad)
    intended_direction_xy = [
        math.cos(intended_direction_angle_rad),
        math.sin(intended_direction_angle_rad),
    ]


def update_damping_matrix():
    """Build the planar damping matrix, paper (12) and (14).

    In the frame whose x axis is the intended direction,

        B_local = diag(bd / gamma1, gamma2 * bd)

    with bd, gamma1 and gamma2 from update_direct_damping().
    Rotating into the base frame with R = [[c, -s], [s, c]] gives
    Bd = R * B_local * R^T, written out entry by entry:

        b11 = b_along*c^2 + b_across*s^2
        b12 = b21 = (b_along - b_across)*c*s
        b22 = b_along*s^2 + b_across*c^2

    The result is blended with the isotropic diag(bd, bd) by the
    guidance weight. Both matrices are symmetric positive definite for
    positive damping, and a convex combination of symmetric positive
    definite matrices is symmetric positive definite, so the blended
    damping always dissipates energy.
    """

    global damping_matrix_n_s_m

    damping_along_n_s_m = damping_applied_n_s_m / gamma1
    damping_across_n_s_m = gamma2 * damping_applied_n_s_m

    cosine = intended_direction_xy[0]
    sine = intended_direction_xy[1]

    guided_b11 = (
        damping_along_n_s_m * cosine * cosine +
        damping_across_n_s_m * sine * sine
    )
    guided_b12 = (
        damping_along_n_s_m - damping_across_n_s_m) * cosine * sine
    guided_b22 = (
        damping_along_n_s_m * sine * sine +
        damping_across_n_s_m * cosine * cosine
    )

    isotropic_weight = 1.0 - guidance_weight

    damping_matrix_n_s_m = [
        isotropic_weight * damping_applied_n_s_m +
        guidance_weight * guided_b11,
        guidance_weight * guided_b12,
        guidance_weight * guided_b12,
        isotropic_weight * damping_applied_n_s_m +
        guidance_weight * guided_b22,
    ]


def integrate_planar_admittance():
    """Integrate Md * x_ddot + Bd * x_dot = F_ext, paper (2).

    Inputs (module state):
        force_filtered_n: human force in the base frame [N].
        damping_matrix_n_s_m: 2x2 damping matrix [N s/m].
        virtual_mass_kg: scalar virtual mass [kg].

    Outputs (module state):
        cart_velocity_command_m_s: commanded planar velocity [m/s].
    """

    global cart_velocity_command_m_s

    damping_force_x_n = (
        damping_matrix_n_s_m[0] * cart_velocity_command_m_s[0] +
        damping_matrix_n_s_m[1] * cart_velocity_command_m_s[1]
    )
    damping_force_y_n = (
        damping_matrix_n_s_m[2] * cart_velocity_command_m_s[0] +
        damping_matrix_n_s_m[3] * cart_velocity_command_m_s[1]
    )

    acceleration_x_m_s2 = (
        force_filtered_n[0] - damping_force_x_n) / virtual_mass_kg
    acceleration_y_m_s2 = (
        force_filtered_n[1] - damping_force_y_n) / virtual_mass_kg

    cart_velocity_command_m_s[0] += acceleration_x_m_s2 * dt
    cart_velocity_command_m_s[1] += acceleration_y_m_s2 * dt

    # Clamping the state itself, not just the output, keeps the virtual
    # model from winding up while the limit is active.
    cart_velocity_command_m_s = limit_vector_norm_2d(
        cart_velocity_command_m_s, max_cart_speed_m_s)


def apply_workspace_limit(tcp_position_xy_m):
    """Remove velocity components pointing outside the x-y rectangle.

    Inputs:
        tcp_position_xy_m: measured [x, y] TCP position [m] in the UR5e
            base frame.

    The rectangle corners are the user-defined offsets from
    start_position_m. Inward and edge-tangential motion stay available.
    """

    workspace_lower_corner_m = [
        start_position_m[0] + workspace_lower_corner_offset_m[0],
        start_position_m[1] + workspace_lower_corner_offset_m[1],
    ]
    workspace_upper_corner_m = [
        start_position_m[0] + workspace_upper_corner_offset_m[0],
        start_position_m[1] + workspace_upper_corner_offset_m[1],
    ]

    for axis in range(2):
        at_lower_limit = (
            tcp_position_xy_m[axis] <= workspace_lower_corner_m[axis])
        at_upper_limit = (
            tcp_position_xy_m[axis] >= workspace_upper_corner_m[axis])

        if at_lower_limit and cart_velocity_command_m_s[axis] < 0.0:
            cart_velocity_command_m_s[axis] = 0.0

        if at_upper_limit and cart_velocity_command_m_s[axis] > 0.0:
            cart_velocity_command_m_s[axis] = 0.0


def send_speed_command(rtde_control_interface):
    """Send the planar TCP velocity [m/s] through speedL()."""

    speed_command = [
        cart_velocity_command_m_s[0],
        cart_velocity_command_m_s[1],
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    if send_robot_commands:
        rtde_control_interface.speedL(
            speed_command, speedl_acceleration_m_s2, dt)


def move_to_start_joint_configuration(rtde_control_interface):
    """Move the UR5e to the user-defined start joint configuration."""

    if not send_robot_commands:
        print("Skipping start joint move because send_robot_commands=False")
        return

    if not move_to_start_configuration:
        print("Skipping start joint move by user setting.")
        return

    print("Moving UR5e to start_joint_q [rad]:")
    print(start_joint_q)

    rtde_control_interface.moveJ(
        start_joint_q,
        start_joint_speed_rad_s,
        start_joint_acceleration_rad_s2,
    )


# -------------------- CoppeliaSim Bridge --------------------
def map_tcp_position_to_coppeliasim(tcp_position_xyz_m):
    """Map UR5e base-frame position [m] to the CoppeliaSim frame [m]."""

    return [
        coppeliasim_position_offset_m[0] +
        coppeliasim_position_signs[0] * tcp_position_xyz_m[0],
        coppeliasim_position_offset_m[1] +
        coppeliasim_position_signs[1] * tcp_position_xyz_m[1],
        coppeliasim_position_offset_m[2] +
        coppeliasim_position_signs[2] * tcp_position_xyz_m[2],
    ]


def coppeliasim_trajectory_process(trajectory_queue, stop_event):
    """Send the latest TCP position to CoppeliaSim from a child process.

    Input:
        trajectory_queue: multiprocessing queue holding
            (controller_time_s, tcp_position_xyz_m) samples.
        stop_event: multiprocessing event set by the robot process.

    Output:
        The configured CoppeliaSim object receives setObjectPosition()
        calls.

    Assumptions:
        CoppeliaSim is already running with its ZeroMQ Remote API
        enabled. Simulator errors terminate only this child process, so
        the control loop is never blocked by simulator latency.
    """

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ImportError as error:
        print(
            "CoppeliaSim bridge disabled: cannot import ZeroMQ client:",
            error,
            flush=True,
        )
        return

    try:
        client = RemoteAPIClient(
            host=coppeliasim_host, port=coppeliasim_port)
        sim = client.require("sim")
        target_object_handle = sim.getObject(coppeliasim_object_path)

        reference_object_handle = -1
        if coppeliasim_reference_object_path:
            reference_object_handle = sim.getObject(
                coppeliasim_reference_object_path)

        update_period_s = 1.0 / coppeliasim_update_frequency_hz
        next_update_time_s = 0.0

        print(
            "CoppeliaSim bridge connected; updating",
            coppeliasim_object_path,
            "at",
            coppeliasim_update_frequency_hz,
            "Hz",
            flush=True,
        )

        while not stop_event.is_set():
            try:
                trajectory_sample = trajectory_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Drop stale samples so simulator lag cannot build a backlog.
            while True:
                try:
                    trajectory_sample = trajectory_queue.get_nowait()
                except queue.Empty:
                    break

            current_time_s = time.monotonic()
            if current_time_s < next_update_time_s:
                continue

            coppeliasim_position_m = map_tcp_position_to_coppeliasim(
                trajectory_sample[1])

            sim.setObjectPosition(
                target_object_handle,
                reference_object_handle,
                coppeliasim_position_m,
            )

            next_update_time_s = current_time_s + update_period_s

    except Exception as error:
        print("CoppeliaSim bridge stopped:", repr(error), flush=True)
        traceback.print_exc()


def start_coppeliasim_trajectory_process():
    """Start the independent CoppeliaSim trajectory process if enabled."""

    if not send_coppeliasim_trajectory:
        return None, None, None

    process_context = multiprocessing.get_context("spawn")
    trajectory_queue = process_context.Queue(maxsize=1)
    stop_event = process_context.Event()
    trajectory_process = process_context.Process(
        target=coppeliasim_trajectory_process,
        args=(trajectory_queue, stop_event),
        name="coppeliasim_trajectory_process",
    )
    trajectory_process.start()

    return trajectory_queue, stop_event, trajectory_process


def publish_coppeliasim_trajectory_sample(
    trajectory_queue,
    time_s,
    tcp_position_xyz_m,
):
    """Enqueue only the newest TCP position without blocking control."""

    if trajectory_queue is None:
        return

    try:
        while True:
            trajectory_queue.get_nowait()
    except queue.Empty:
        pass

    try:
        trajectory_queue.put_nowait((time_s, list(tcp_position_xyz_m)))
    except queue.Full:
        # The child is draining the queue; a later sample will retry.
        pass


def stop_coppeliasim_trajectory_process(
    trajectory_queue,
    stop_event,
    trajectory_process,
):
    """Stop and release the CoppeliaSim trajectory process."""

    if trajectory_process is None:
        return

    stop_event.set()
    trajectory_process.join(timeout=2.0)

    if trajectory_process.is_alive():
        trajectory_process.terminate()
        trajectory_process.join()

    trajectory_queue.close()


# -------------------- Tobii Recording --------------------
def make_tobii_json_safe(value):
    """Convert SDK values into JSON-compatible values without data loss."""

    if isinstance(value, dict):
        return {
            str(key): make_tobii_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_tobii_json_safe(item) for item in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def make_tobii_csv_value(value):
    """Serialize compound SDK values into one CSV cell."""

    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            make_tobii_json_safe(value), separators=(",", ":"))

    return value


def write_tobii_csv(gaze_queue, writer_stop_event):
    """Write synchronized Tobii callback samples to CSV in a thread."""

    with open(tobii_data_csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_field_names = None

        while not writer_stop_event.is_set() or not gaze_queue.empty():
            try:
                (
                    time_s,
                    host_monotonic_time_ns,
                    gaze_sample,
                ) = gaze_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if csv_field_names is None:
                csv_field_names = sorted(gaze_sample.keys())
                csv_writer.writerow(
                    ["time_s", "host_monotonic_time_ns"] +
                    csv_field_names +
                    ["sdk_data_json"]
                )

            csv_row = [time_s, host_monotonic_time_ns] + [
                make_tobii_csv_value(gaze_sample.get(field_name))
                for field_name in csv_field_names
            ]
            csv_row.append(json.dumps(
                make_tobii_json_safe(gaze_sample), separators=(",", ":")))
            csv_writer.writerow(csv_row)


def create_tobii_gaze_callback(
    gaze_queue,
    recording_active_event,
    trial_clock,
):
    """Queue SDK gaze data with the shared host-monotonic trial time."""

    def gaze_callback(data):
        """Timestamp and queue one SDK dictionary without file I/O."""

        if not recording_active_event.is_set():
            return

        host_monotonic_time_ns = time.perf_counter_ns()
        trial_start_time_ns = trial_clock["start_time_ns"]

        if trial_start_time_ns is None:
            return

        gaze_sample = dict(data)
        time_s = (host_monotonic_time_ns - trial_start_time_ns) * 1e-9

        # Re-checked so a callback overlapping shutdown cannot append a
        # post-stop sample.
        if not recording_active_event.is_set():
            return

        push_classifier_gaze_sample(time_s, gaze_sample)

        try:
            gaze_queue.put_nowait(
                (time_s, host_monotonic_time_ns, gaze_sample))
        except queue.Full:
            # Never block the Tobii SDK callback thread.
            pass

    return gaze_callback


def start_tobii_recording(recording_active_event, trial_clock):
    """Arm Tobii acquisition and its synchronized CSV writer."""

    if not enable_tobii_recording:
        print("Tobii recording disabled by user setting.")
        return None

    try:
        import tobii_research as tobii_research_api
    except ImportError:
        print(
            "Tobii recording disabled: install with "
            "`pip install --user tobii-research`"
        )
        return None

    trackers = tobii_research_api.find_all_eyetrackers()

    if not trackers:
        print("No Tobii eye tracker found. Tobii recording disabled.")
        return None

    tracker = trackers[0]
    gaze_queue = queue.Queue(maxsize=tobii_queue_size)
    writer_stop_event = threading.Event()
    writer_thread = threading.Thread(
        target=write_tobii_csv,
        args=(gaze_queue, writer_stop_event),
        name="tobii_csv_writer",
        daemon=True,
    )
    gaze_callback = create_tobii_gaze_callback(
        gaze_queue, recording_active_event, trial_clock)

    writer_thread.start()
    tracker.subscribe_to(
        tobii_research_api.EYETRACKER_GAZE_DATA,
        gaze_callback,
        as_dictionary=True,
    )

    print("Tobii model:", tracker.model)
    print("Tobii serial:", tracker.serial_number)
    print("Tobii CSV:", tobii_data_csv_path)

    return (
        tobii_research_api,
        tracker,
        gaze_callback,
        writer_stop_event,
        writer_thread,
    )


def stop_tobii_recording(tobii_recording):
    """Unsubscribe from Tobii and flush the remaining gaze queue."""

    if tobii_recording is None:
        return

    (
        tobii_research_api,
        tracker,
        gaze_callback,
        writer_stop_event,
        writer_thread,
    ) = tobii_recording

    try:
        tracker.unsubscribe_from(
            tobii_research_api.EYETRACKER_GAZE_DATA, gaze_callback)
    finally:
        writer_stop_event.set()
        writer_thread.join(timeout=2.0)


# -------------------- Logging --------------------
def write_csv_header(csv_writer):
    """Write names and units for the controller CSV columns."""

    csv_writer.writerow([
        "time_s",
        "host_monotonic_time_ns",
        "Fx_raw_N",
        "Fy_raw_N",
        "Fx_filt_N",
        "Fy_filt_N",
        "x_m",
        "y_m",
        "z_m",
        "coppeliasim_x_m",
        "coppeliasim_y_m",
        "coppeliasim_z_m",
        "tcp_vx_actual_m_s",
        "tcp_vy_actual_m_s",
        "tcp_vx_filt_m_s",
        "tcp_vy_filt_m_s",
        "vx_cmd_m_s",
        "vy_cmd_m_s",
        "speed_cmd_m_s",
        "bd_desired_Ns_m",
        "bd_applied_Ns_m",
        "beta_hat",
        "K_hat",
        "power_law_R2",
        "power_law_RMSE",
        "power_law_sample_used",
        "power_law_valid",
        "curvature_measured_1_m",
        "curvature_target_1_m",
        "gamma1",
        "gamma2",
        "turn_signal",
        "guidance_weight",
        "intended_acc_x_m_s2",
        "intended_acc_y_m_s2",
        "intended_force_x_N",
        "intended_force_y_N",
        "intended_dir_x",
        "intended_dir_y",
        "B11_Ns_m",
        "B12_Ns_m",
        "B21_Ns_m",
        "B22_Ns_m",
        "M_kg",
        "send_robot_commands",
        # --- classifier gating; one extra column per model feature, so
        # the header depends on the loaded bundle ---
        "enable_classifier_gating",
        "fine_probability",
        "fine_probability_age_s",
        "classifier_status",
        "classifier_update_count",
        "classifier_compute_ms",
        "bd_min_Ns_m",
        "bd_max_Ns_m",
    ] + classifier_feature_column_names())


def write_csv_row(
    csv_writer,
    time_s,
    host_monotonic_time_ns,
    tcp_position_xyz_m,
    tcp_actual_velocity,
):
    """Write one controller sample with measured TCP state to CSV."""

    coppeliasim_position_m = map_tcp_position_to_coppeliasim(
        tcp_position_xyz_m)

    csv_writer.writerow([
        time_s,
        host_monotonic_time_ns,
        force_raw_n[0],
        force_raw_n[1],
        force_filtered_n[0],
        force_filtered_n[1],
        tcp_position_xyz_m[0],
        tcp_position_xyz_m[1],
        tcp_position_xyz_m[2],
        coppeliasim_position_m[0],
        coppeliasim_position_m[1],
        coppeliasim_position_m[2],
        tcp_actual_velocity[0],
        tcp_actual_velocity[1],
        velocity_filtered_m_s[0],
        velocity_filtered_m_s[1],
        cart_velocity_command_m_s[0],
        cart_velocity_command_m_s[1],
        vec2_norm(cart_velocity_command_m_s),
        damping_desired_n_s_m,
        damping_applied_n_s_m,
        beta_hat,
        gain_hat,
        power_law_r2,
        power_law_rmse,
        power_law_sample_used,
        int(power_law_valid),
        curvature_measured_1_m,
        curvature_target_1_m,
        gamma1,
        gamma2,
        turn_signal,
        guidance_weight,
        intended_acc_m_s2[0],
        intended_acc_m_s2[1],
        intended_force_n[0],
        intended_force_n[1],
        intended_direction_xy[0],
        intended_direction_xy[1],
        damping_matrix_n_s_m[0],
        damping_matrix_n_s_m[1],
        damping_matrix_n_s_m[2],
        damping_matrix_n_s_m[3],
        virtual_mass_kg,
        int(send_robot_commands),
        int(enable_classifier_gating),
        fine_probability_applied,
        fine_probability_age_s,
        classifier_status,
        classifier_update_count,
        classifier_compute_ms,
        damping_lower_bound_n_s_m,
        damping_upper_bound_n_s_m,
    ] + [
        classifier_feature_values.get(column_name, float("nan"))
        for column_name in classifier_feature_column_names()
    ])


# -------------------- Main --------------------
def main():
    """Run the UR5e Chen-style planar variable admittance loop with
    classifier-gated damping bounds."""

    global start_position_m

    print("Output:", write_data_csv_path)
    print("Output:", tobii_data_csv_path)
    if not enable_classifier_gating:
        print("Classifier gating DISABLED -- this is the Chen baseline arm.")
    else:
        print("Input: ", read_calibration_json_path)
        if fixed_fine_probability is None:
            print("Input: ", read_model_joblib_path)
    print()

    if rtde_control is None or rtde_receive is None:
        print("Missing dependency: install with `pip install --user ur_rtde`")
        return

    # Before any hardware: a missing calibration, an unknown class, or a
    # feature that cannot be computed online has to stop the run while
    # the arm is still idle.
    if enable_classifier_gating:
        load_classifier()

    signal.signal(signal.SIGINT, handle_stop_signal)

    print("Connecting to UR5e at", robot_ip)
    rtde_control_interface = rtde_control.RTDEControlInterface(robot_ip)
    rtde_receive_interface = rtde_receive.RTDEReceiveInterface(robot_ip)

    move_to_start_joint_configuration(rtde_control_interface)

    print("Keep the robot unloaded. Estimating planar force bias...")
    estimate_force_bias(rtde_receive_interface)

    tcp_position_xyz_m = read_tcp_position_xyz(rtde_receive_interface)
    start_position_m = [tcp_position_xyz_m[0], tcp_position_xyz_m[1]]

    print(
        "Subject preparation: control and recording start in",
        subject_preparation_time_s,
        "seconds.",
    )
    time.sleep(subject_preparation_time_s)

    (
        trajectory_queue,
        simulator_stop_event,
        trajectory_process,
    ) = start_coppeliasim_trajectory_process()

    recording_active_event = threading.Event()
    trial_clock = {"start_time_ns": None}
    tobii_recording = start_tobii_recording(
        recording_active_event, trial_clock)
    classifier_stop_event, classifier_thread = None, None

    try:
        with open(write_data_csv_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            write_csv_header(csv_writer)

            max_steps = int(max_runtime_s / dt)
            trial_start_time_ns = time.perf_counter_ns()
            trial_clock["start_time_ns"] = trial_start_time_ns
            recording_active_event.set()
            (
                classifier_stop_event,
                classifier_thread,
            ) = start_classifier_thread()

            print("Controller and synchronized recording started.")
            print("Press Ctrl+C to stop.")
            print("send_robot_commands =", send_robot_commands)
            print("virtual_mass_kg =", virtual_mass_kg)
            print(
                "damping range [N s/m] =",
                damping_min_n_s_m,
                damping_max_n_s_m,
            )
            if enable_classifier_gating:
                print(
                    "gated damping bounds [N s/m]: fine =",
                    fine_damping_bounds_n_s_m,
                    " gross =",
                    gross_damping_bounds_n_s_m,
                )
                print(
                    "classifier window:",
                    classifier_window_s,
                    "s, updated at",
                    classifier_update_frequency_hz,
                    "Hz",
                )
            print(
                "power-law window:",
                power_law_window_s,
                "s =",
                power_law_sample_count,
                "samples at",
                control_frequency_hz / estimate_period_steps,
                "Hz",
            )

            for step_idx in range(max_steps):
                if not keep_running:
                    break

                period_start = rtde_control_interface.initPeriod()
                host_monotonic_time_ns = time.perf_counter_ns()
                time_s = (
                    host_monotonic_time_ns - trial_start_time_ns) * 1e-9

                tcp_position_xyz_m = read_tcp_position_xyz(
                    rtde_receive_interface)
                tcp_actual_velocity = read_tcp_actual_velocity(
                    rtde_receive_interface)
                tcp_position_xy_m = [
                    tcp_position_xyz_m[0],
                    tcp_position_xyz_m[1],
                ]

                update_force_filter(rtde_receive_interface)
                update_velocity_filter(tcp_actual_velocity)

                # The estimator runs decimated: central differences over
                # 2 ms of a noisy velocity are meaningless, and refitting
                # the whole window every control cycle would not change
                # the estimate anyway.
                if step_idx % estimate_period_steps == 0:
                    push_velocity_history(time_s)
                    estimate_power_law_from_history()

                # Paper data flow: direct intention (7) sets bd and the
                # gamma terms (13), the intended force (11) then sets
                # the intended direction, and (12)/(14) build the
                # damping matrix from both.
                update_turn_signal()
                update_guidance_weight()
                compute_intended_acceleration()
                # The only step Chen's own flow does not have: P(Fine)
                # sets the endpoints (7) then maps force onto.
                update_damping_bounds(time_s)
                update_direct_damping()
                update_intended_direction()
                update_damping_matrix()

                integrate_planar_admittance()
                apply_workspace_limit(tcp_position_xy_m)
                send_speed_command(rtde_control_interface)

                # Buffered after the workspace limit so the commanded
                # velocity matches the CSV column the offline feature
                # pipeline reads back out of this same file.
                push_classifier_robot_sample(
                    time_s, tcp_position_xyz_m, tcp_actual_velocity)

                write_csv_row(
                    csv_writer,
                    time_s,
                    host_monotonic_time_ns,
                    tcp_position_xyz_m,
                    tcp_actual_velocity,
                )

                # Queue-only: no CoppeliaSim API call happens here.
                publish_coppeliasim_trajectory_sample(
                    trajectory_queue, time_s, tcp_position_xyz_m)

                rtde_control_interface.waitPeriod(period_start)
    finally:
        recording_active_event.clear()
        stop_classifier_thread(classifier_stop_event, classifier_thread)
        stop_tobii_recording(tobii_recording)

        stop_coppeliasim_trajectory_process(
            trajectory_queue, simulator_stop_event, trajectory_process)

        if send_robot_commands:
            rtde_control_interface.speedStop(
                speed_stop_deceleration_m_s2)

        rtde_control_interface.stopScript()

    print("Stopped UR5e Chen-style variable admittance controller.")
    print("CSV saved to:", write_data_csv_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
