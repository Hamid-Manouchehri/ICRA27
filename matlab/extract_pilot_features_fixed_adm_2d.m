%% Windowed feature extraction for ur5e_fixed_adm_2d_labeled.py pilot
%% recordings.
%
% For one recorded trial (robot CSV + gaze CSV + calibration JSON + the
% shape .obj the operator traced), computes the robot, eye-gaze, and
% robot+gaze fusion features listed below over a sliding window with
% adjustable length and overlap, and writes one row per window to an
% output CSV. Intended as training data for a Fine/Gross/Natural
% classifier -- the ground-truth class is NOT a per-file constant.
% robot_csv_file's own per-sample "label" column (written live by
% ur5e_fixed_adm_2d_labeled.py from whichever key the operator is
% holding -- 'f'/'g'/'n', or "None" when no key is held) is read and
% resampled onto Stage 1's uniform grid.
%
% "None" is not a fourth class, just unlabeled time (rest, or a pause
% while switching keys), so it is not considered anywhere in feature
% calculation: a window is skipped outright -- no features computed,
% no output row -- unless every one of its samples already carries the
% same real label (see most_common_label() and the Stage 2 loop for
% why this is a skip rather than a filter applied after the fact).
%
% Inputs:
%   robot_csv_file   - data/raw_data/ur5e_fixed_adm_2d_<trial>.csv
%   gaze_csv_file    - data/raw_data/tobii_fixed_adm_2d_<trial>.csv
%   calibration_json_file - data/calibrations/tobii_coppelia_reference_
%       calibration_<shape>[_s<subject>].json, maps Tobii UCS gaze into
%       the CoppeliaSim frame.
%   shape_obj_file   - data/shapes/<shape>.obj, the reference path the
%       operator traced. Used for the gaze-lookahead and gaze-projected-
%       curvature features; if a trial has no meaningful reference path
%       (e.g. a "Natural" free-motion trial), leave shape_obj_file empty
%       and those features are written as NaN instead of computed.
%
% Output:
%   One CSV, one row per KEPT window: window bounds, trial_class_label,
%   label_purity, then 23 feature columns (11 robot, 6 gaze, 6 fusion --
%   see the three *_window_features() functions below for the exact list
%   and units). A candidate window is dropped -- no row written -- if
%   it is majority "None" (unlabeled) or if it crosses a label change
%   (majority label covers less than 100% of its samples); see the
%   Stage 2 loop. label_purity is kept as an explicit diagnostic even
%   though every surviving row necessarily reads 1.0 by construction --
%   a downstream script can check it rather than take that on faith.
%   The command window reports how many candidate windows were dropped
%   and why.
%
% Two-stage design, for readability and correctness:
%   Stage 1 computes every derived per-sample signal ONCE for the whole
%   trial (velocity smoothing, acceleration, curvature, gaze mapped into
%   the robot frame, gaze projected onto the reference path, ...) on a
%   single uniform time grid. Stage 2 slides a window across that grid
%   and, for each window, mostly just takes a mean/median/count of an
%   already-computed column. Filtering/differentiating separately inside
%   each window would add noisy edge effects at every window boundary
%   and make windows harder to compare; doing it once up front avoids
%   that and keeps the per-window code close to a direct list of
%   one-line feature definitions.
%
% Two different velocity signals are used on purpose:
%   Everything except DSJ is derived from the *measured* TCP velocity
%   (tcp_vx/vy_actual_m_s, lightly smoothed) -- one derivative closer to
%   what was actually measured than starting from position, matching
%   how curvature is already computed elsewhere in this project (e.g.
%   ur5e_chen_var_adm_2d.py's estimator). DSJ specifically is computed
%   from *position* with the exact same paper-Eq.(16) chain
%   (compute_jerk_series/compute_dsj) already used throughout this
%   project's other MATLAB scripts, so DSJ values here stay comparable
%   with DSJ reported by those scripts.
%
% Reference-path direction assumption:
%   d_lookahead and fraction_gaze_ahead depend on the reference path's
%   own point ordering (inherited from the mesh's boundary-edge winding)
%   matching the direction the operator actually traces the shape. This
%   is not verified here -- if "gaze ahead of TCP" comes out backwards
%   for a given shape, negate d_lookahead and use
%   (1 - fraction_gaze_ahead) instead of re-deriving the path.
%
% Method reuse:
%   The shape-band OBJ reader, I-VT fixation detector (with UCS-to-
%   CoppeliaSim mapping), and the DSJ jerk chain are copied verbatim
%   from plot_raw_data_chen_var_adm_2d.m /
%   compare_chen_vs_gaze_assisted_2d.m. Self-contained here per this
%   project's convention that MATLAB scripts do not call one another.
%
% Assumptions:
%   robot_csv_file was written by ur5e_fixed_adm_2d_labeled.py, so
%   column names match its write_csv_header() (x_m/y_m, Fx_filt_N/
%   Fy_filt_N, tcp_vx/vy_actual_m_s) AND it has a "label" column -- a
%   plain ur5e_fixed_adm_2d.py recording (no labeling) will fail here
%   with "Unrecognized table variable name 'label'". gaze_csv_file is
%   the matching Tobii CSV (full SDK callback passthrough) from the same
%   trial, sharing its time_s origin. shape_offset_xy_m is the
%   CoppeliaSim-frame offset already validated for this shape elsewhere
%   in this project.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');
calibration_directory = fullfile( ...
    script_directory, '..', 'data', 'calibrations');
shapes_directory = fullfile(script_directory, '..', 'data', 'shapes');

%% User Settings
% TODO the four input files for this one trial. robot_csv_file must come
% from ur5e_fixed_adm_2d_labeled.py (needs its "label" column) -- plain
% ur5e_fixed_adm_2d.py recordings carry no ground-truth class at all.
robot_csv_file = fullfile(raw_data_directory, ...
    'ur5e_fixed_adm_2d_labeled_d50_s1.csv');
gaze_csv_file = fullfile(raw_data_directory, ...
    'tobii_fixed_adm_2d_labeled_d50_s1.csv');
calibration_json_file = fullfile(calibration_directory, ...
    'tobii_coppelia_reference_calibration_fine_gross_shape_s1.json');
shape_obj_file = fullfile(shapes_directory, 'fine_gross_shape.obj');

% TODO output CSV path [file path].
write_features_csv_path = fullfile(script_directory, '..', 'data', ...
    'processed_data', 'features_fixed_adm_2d_labeled_d50_s1.csv');

% TODO translation from the mesh CAD frame to the CoppeliaSim frame [m].
% Matches the value already validated for this shape elsewhere in this
% project. Ignored if shape_obj_file is empty.
shape_offset_xy_m = [-0.014, -0.650];

% TODO spacing used to resample the reference path [m], positive.
% Ignored if shape_obj_file is empty.
reference_spacing_m = 5e-4;

% --- Windowing ---
% TODO sliding-window length [s], positive. Shorter windows give more
% training rows but noisier per-window statistics (especially the
% power-law fit and fixation-based features, which need several samples
% to be meaningful); longer windows are steadier but blur together
% behaviour that actually changed within the window. Recommended 1 to 5.
window_length_s = 2.0;

% TODO fraction of one window that overlaps the next [0, 1). 0 means
% back-to-back non-overlapping windows; 0.5 means each window shares
% half its samples with its neighbours (more rows, more correlated
% between adjacent rows -- keep this in mind when splitting train/test).
window_overlap_fraction = 0.0;

% --- Shared thresholds ---
% TODO speed below which velocity direction, curvature, and "moving" are
% all considered undefined/at rest [m/s]. Matches this project's other
% controllers and analysis scripts.
minimum_speed_m_s = 0.005;

% TODO minimum planar force magnitude to trust its direction [N], used
% only for angle(d_gaze, F). Below this the force direction is noise.
minimum_force_for_direction_n = 0.5;

% --- Derivative-chain filtering (velocity -> acceleration -> curvature) ---
% TODO smoothing window applied to resampled velocity before
% differentiating it for acceleration/curvature [s]. Matches the
% smoothing windows already used for this project's other derivative
% chains.
velocity_smoothing_window_s = 0.10;

% --- DSJ (paper Eq. 16), computed from position, same chain as
% elsewhere in this project ---
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;

% --- Power-law fit (curvature vs. speed) per window ---
% TODO minimum number of valid (curvature, speed) samples in a window
% before its power-law R^2 is trusted rather than reported as NaN.
power_law_min_sample_count = 10;

% --- I-VT fixation detection (raw gaze -> fixations) ---
% TODO angular-velocity threshold [deg/s] and minimum fixation duration
% [s]. Matches this project's other I-VT settings.
fixation_velocity_threshold_deg_s = 30.0;
minimum_fixation_duration_s = 0.10;
% TODO true excludes gaze estimates outside the normalized display area.
plot_only_on_display_gaze = true;
% TODO physical screen motion per robot motion [unitless], > 0, and the
% Tobii UCS axis signs in CoppeliaSim, each -1 or +1.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

%% ================================================================
% Stage 1: load raw data
% =================================================================

robot_table = readtable(robot_csv_file, 'Delimiter', ',', ...
    'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

robot_time_s = get_numeric_column(robot_table.time_s);
robot_x_m = get_numeric_column(robot_table.x_m);
robot_y_m = get_numeric_column(robot_table.y_m);
robot_force_x_n = get_numeric_column(robot_table.Fx_filt_N);
robot_force_y_n = get_numeric_column(robot_table.Fy_filt_N);
robot_vx_m_s = get_numeric_column(robot_table.tcp_vx_actual_m_s);
robot_vy_m_s = get_numeric_column(robot_table.tcp_vy_actual_m_s);
% Text, not numeric -- already a string array via readtable's
% 'TextType','string' above, so no get_numeric_column() conversion.
robot_label = robot_table.label;

gaze_table = readtable(gaze_csv_file, 'Delimiter', ',', ...
    'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

calibration_reference = jsondecode(fileread(calibration_json_file));
tobii_reference_xy_mm = double( ...
    calibration_reference.manual_tobii_reference_xy_mm(:)');
coppelia_reference_xy_m = double( ...
    calibration_reference.manual_coppelia_reference_xy_m(:)');

has_reference_path = ~isempty(shape_obj_file) && isfile(shape_obj_file);
if has_reference_path
    [~, ~, reference_xy_m] = ...
        read_shape_band_obj(shape_obj_file, reference_spacing_m);
    if isempty(reference_xy_m)
        warning(['Shape mesh did not resolve to a simple two-loop ' ...
            'band; gaze-lookahead and gaze-projected-curvature ' ...
            'features will be NaN.']);
        has_reference_path = false;
    else
        reference_xy_m = reference_xy_m + shape_offset_xy_m;
    end
end

%% ================================================================
% Stage 1: per-sample derived signals, robot side
% =================================================================
% One uniform time grid for the whole trial; see the header comment for
% why this happens once here rather than per window.

motion = compute_motion_derivatives(robot_time_s, robot_vx_m_s, ...
    robot_vy_m_s, robot_x_m, robot_y_m, velocity_smoothing_window_s, ...
    minimum_speed_m_s);

% Force and its DSJ-independent quantities are read off the raw,
% row-aligned CSV directly (no differentiation involved, so no
% smoothing chain is needed) and then interpolated onto motion's
% uniform grid, the "master clock" every other signal is aligned to.
sample.time_s = motion.time_s;
sample.x_m = motion.x_m;
sample.y_m = motion.y_m;
sample.vx_m_s = motion.vx_m_s;
sample.vy_m_s = motion.vy_m_s;
sample.speed_m_s = motion.speed_m_s;
sample.accel_mag_m_s2 = motion.accel_mag_m_s2;
sample.signed_curvature_1_m = motion.signed_curvature_1_m;
sample.curvature_dot_1_m_s = motion.curvature_dot_1_m_s;
sample.trajectory_arclength_m = motion.trajectory_arclength_m;

sample.force_x_n = interp1(robot_time_s, robot_force_x_n, ...
    sample.time_s, 'linear', 'extrap');
sample.force_y_n = interp1(robot_time_s, robot_force_y_n, ...
    sample.time_s, 'linear', 'extrap');
sample.force_mag_n = hypot(sample.force_x_n, sample.force_y_n);
sample.power_w = sample.force_x_n .* sample.vx_m_s + ...
    sample.force_y_n .* sample.vy_m_s;

% Ground-truth class, resampled the same nearest-in-time way as
% fixation position below -- see resample_labels_nearest() for why text
% needs its own resampling helper instead of interp1() directly.
sample.label = resample_labels_nearest(robot_time_s, robot_label, ...
    sample.time_s);

%% ================================================================
% Stage 1: per-sample derived signals, gaze side
% =================================================================

[fixation_xy_m, fixation_time_s, fixation_duration_s, raw_gaze_xy_m, ...
    raw_gaze_time_s] = detect_fixations_coppelia_xy(gaze_table, ...
    tobii_reference_xy_mm, coppelia_reference_xy_m, ...
    screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
    fixation_velocity_threshold_deg_s, minimum_fixation_duration_s, ...
    plot_only_on_display_gaze);

fixation_count = numel(fixation_time_s);
if fixation_count == 0
    warning(['No fixations detected in this trial; every gaze and ' ...
        'fusion feature will be NaN. Check fixation_velocity_' ...
        'threshold_deg_s / minimum_fixation_duration_s against this ' ...
        'gaze recording.']);
end

% Each robot sample inherits the position of whichever fixation is
% nearest in time -- a causal "where was the operator's attention most
% recently directed" signal, resampled onto the robot's own clock so it
% can be combined sample-by-sample with velocity/force/curvature below.
if fixation_count > 0
    sample.fixation_x_m = interp1(fixation_time_s, fixation_xy_m(:, 1), ...
        sample.time_s, 'nearest', 'extrap');
    sample.fixation_y_m = interp1(fixation_time_s, fixation_xy_m(:, 2), ...
        sample.time_s, 'nearest', 'extrap');
else
    sample.fixation_x_m = NaN(size(sample.time_s));
    sample.fixation_y_m = NaN(size(sample.time_s));
end

%% ================================================================
% Stage 1: project TCP and gaze onto the reference path
% =================================================================

sample.tcp_arclength_m = NaN(size(sample.time_s));
sample.gaze_arclength_m = NaN(size(sample.time_s));
sample.delta_s_m = NaN(size(sample.time_s));
sample.gaze_path_curvature_1_m = NaN(size(sample.time_s));

if has_reference_path
    [path_arclength_m, path_curvature_1_m] = ...
        compute_path_geometry(reference_xy_m);
    path_length_m = path_arclength_m(end);

    tcp_nearest_index = nearest_reference_index( ...
        sample.x_m, sample.y_m, reference_xy_m);
    sample.tcp_arclength_m = path_arclength_m(tcp_nearest_index);

    if fixation_count > 0
        gaze_nearest_index = nearest_reference_index( ...
            sample.fixation_x_m, sample.fixation_y_m, reference_xy_m);
        sample.gaze_arclength_m = path_arclength_m(gaze_nearest_index);
        sample.gaze_path_curvature_1_m = ...
            path_curvature_1_m(gaze_nearest_index);

        sample.delta_s_m = wrap_signed_arclength( ...
            sample.gaze_arclength_m - sample.tcp_arclength_m, ...
            path_length_m);
    end
end

sample.angle_gaze_v_deg = NaN(size(sample.time_s));
sample.angle_gaze_f_deg = NaN(size(sample.time_s));
sample.lookahead_over_speed_s = NaN(size(sample.time_s));
sample.lookahead_times_curvature = NaN(size(sample.time_s));

for sample_idx = 1:numel(sample.time_s)
    gaze_direction_xy = [ ...
        sample.fixation_x_m(sample_idx) - sample.x_m(sample_idx), ...
        sample.fixation_y_m(sample_idx) - sample.y_m(sample_idx)];

    if sample.speed_m_s(sample_idx) >= minimum_speed_m_s
        velocity_xy = [sample.vx_m_s(sample_idx), sample.vy_m_s(sample_idx)];
        sample.angle_gaze_v_deg(sample_idx) = ...
            angle_between_vectors_deg(gaze_direction_xy, velocity_xy);

        if isfinite(sample.delta_s_m(sample_idx))
            sample.lookahead_over_speed_s(sample_idx) = ...
                sample.delta_s_m(sample_idx) / sample.speed_m_s(sample_idx);
        end
    end

    if sample.force_mag_n(sample_idx) >= minimum_force_for_direction_n
        force_xy = [sample.force_x_n(sample_idx), sample.force_y_n(sample_idx)];
        sample.angle_gaze_f_deg(sample_idx) = ...
            angle_between_vectors_deg(gaze_direction_xy, force_xy);
    end

    if isfinite(sample.delta_s_m(sample_idx))
        sample.lookahead_times_curvature(sample_idx) = ...
            sample.delta_s_m(sample_idx) * ...
            sample.signed_curvature_1_m(sample_idx);
    end
end

%% ================================================================
% Stage 2: windows and per-window features
% =================================================================

window_bounds = compute_window_bounds( ...
    sample.time_s, window_length_s, window_overlap_fraction);
window_count = size(window_bounds, 1);

feature_rows = cell(window_count, 1);

% A window is skipped entirely -- no features computed, no output row
% -- unless every one of its samples already carries the same real
% (non-"None") label. This is stricter than just marking the row, and
% deliberately so: "None" data must not be considered in feature
% calculation at all, and the only way to guarantee that for EVERY
% feature (not just the mean/median ones) is to never hand a mixed
% window to the feature functions in the first place. DSJ and
% curvature_sign_reversal_rate_per_m specifically differentiate/compare
% consecutive samples across the window's full time span; surgically
% removing just the "None"-labeled samples from the middle of an
% otherwise-kept window would leave a time gap that interp1() inside
% compute_jerk_series() would silently bridge with a straight line,
% fabricating motion (and jerk) that never happened. Skipping the whole
% window sidesteps that rather than risk it.
excluded_none_count = 0;
excluded_mixed_count = 0;

for window_idx = 1:window_count
    first_index = window_bounds(window_idx, 1);
    last_index = window_bounds(window_idx, 2);
    window_range = first_index:last_index;

    [majority_label, purity] = most_common_label(sample.label(window_range));

    if majority_label == "None"
        excluded_none_count = excluded_none_count + 1;
        continue;
    elseif purity < 1
        excluded_mixed_count = excluded_mixed_count + 1;
        continue;
    end

    robot_features = compute_robot_window_features(sample, window_range, ...
        minimum_speed_m_s, power_law_min_sample_count, ...
        jerk_outlier_window_s, jerk_smoothing_window_s);

    gaze_features = compute_gaze_window_features( ...
        sample.time_s(first_index), sample.time_s(last_index), ...
        sample.delta_s_m(window_range), fixation_time_s, ...
        fixation_duration_s, fixation_xy_m, raw_gaze_time_s, ...
        raw_gaze_xy_m);

    fusion_features = compute_fusion_window_features(sample, window_range);

    feature_row = struct();
    feature_row.window_index = window_idx;
    feature_row.window_start_s = sample.time_s(first_index);
    feature_row.window_end_s = sample.time_s(last_index);
    feature_row.window_center_s = 0.5 * ( ...
        sample.time_s(first_index) + sample.time_s(last_index));
    feature_row.trial_class_label = majority_label;
    feature_row.label_purity = purity;

    feature_rows{window_idx} = merge_structs( ...
        feature_row, robot_features, gaze_features, fusion_features);
end

is_kept_window = ~cellfun(@isempty, feature_rows);
feature_table = struct2table([feature_rows{is_kept_window}]);
writetable(feature_table, write_features_csv_path);

fprintf(['\nWrote %d of %d candidate windows (%.1f s each, %.0f%% ' ...
    'overlap) to %s\n(excluded: %d unlabeled "None", %d crossed a ' ...
    'label change).\n'], height(feature_table), window_count, ...
    window_length_s, 100 * window_overlap_fraction, ...
    write_features_csv_path, excluded_none_count, excluded_mixed_count);


%% ================================================================
%% Local functions -- Stage 1 helpers
%% ================================================================

function motion = compute_motion_derivatives(raw_time_s, raw_vx_m_s, ...
        raw_vy_m_s, raw_x_m, raw_y_m, smoothing_window_s, ...
        minimum_speed_m_s)
%COMPUTE_MOTION_DERIVATIVES Whole-trial velocity, acceleration, and
% signed curvature on one uniform time grid.
%
% Velocity is the *measured* TCP velocity (raw_vx_m_s, raw_vy_m_s),
% resampled and lightly smoothed, then differentiated once for
% acceleration -- one derivative closer to what was actually measured
% than differentiating position twice. Position is resampled the same
% way, unfiltered beyond that resampling, purely to build
% trajectory_arclength_m (cumulative distance actually travelled).
%
% Signed curvature (paper convention used throughout this project):
%
%   kappa = (vx*ay - vy*ax) / |v|^3
%
% set to NaN wherever |v| < minimum_speed_m_s, where the direction (and
% therefore the curvature) is undefined rather than just small.
%
% Output: struct with fields time_s, x_m, y_m, vx_m_s, vy_m_s, speed_m_s,
%   accel_mag_m_s2, signed_curvature_1_m, curvature_dot_1_m_s,
%   trajectory_arclength_m, all the same length, on one uniform grid.

    valid_rows = isfinite(raw_time_s) & isfinite(raw_x_m) & ...
        isfinite(raw_y_m) & isfinite(raw_vx_m_s) & isfinite(raw_vy_m_s);
    clean_time_s = raw_time_s(valid_rows);
    clean_x_m = raw_x_m(valid_rows);
    clean_y_m = raw_y_m(valid_rows);
    clean_vx_m_s = raw_vx_m_s(valid_rows);
    clean_vy_m_s = raw_vy_m_s(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_x_m = clean_x_m(sort_order);
    clean_y_m = clean_y_m(sort_order);
    clean_vx_m_s = clean_vx_m_s(sort_order);
    clean_vy_m_s = clean_vy_m_s(sort_order);

    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_x_m = clean_x_m(unique_indices);
    clean_y_m = clean_y_m(unique_indices);
    clean_vx_m_s = clean_vx_m_s(unique_indices);
    clean_vy_m_s = clean_vy_m_s(unique_indices);

    sample_intervals_s = diff(clean_time_s);
    median_interval_s = median(sample_intervals_s(sample_intervals_s > 0));

    uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';
    uniform_x_m = interp1(clean_time_s, clean_x_m, uniform_time_s, 'linear');
    uniform_y_m = interp1(clean_time_s, clean_y_m, uniform_time_s, 'linear');
    uniform_vx_m_s = interp1( ...
        clean_time_s, clean_vx_m_s, uniform_time_s, 'linear');
    uniform_vy_m_s = interp1( ...
        clean_time_s, clean_vy_m_s, uniform_time_s, 'linear');

    smoothing_samples = window_to_odd_sample_count( ...
        smoothing_window_s, median_interval_s, numel(uniform_time_s));
    filtered_vx_m_s = movmean(uniform_vx_m_s, smoothing_samples, ...
        'Endpoints', 'shrink');
    filtered_vy_m_s = movmean(uniform_vy_m_s, smoothing_samples, ...
        'Endpoints', 'shrink');

    filtered_ax_m_s2 = gradient(filtered_vx_m_s, uniform_time_s);
    filtered_ay_m_s2 = gradient(filtered_vy_m_s, uniform_time_s);

    speed_m_s = hypot(filtered_vx_m_s, filtered_vy_m_s);
    signed_curvature_1_m = (filtered_vx_m_s .* filtered_ay_m_s2 - ...
        filtered_vy_m_s .* filtered_ax_m_s2) ./ speed_m_s .^ 3;
    signed_curvature_1_m(speed_m_s < minimum_speed_m_s) = NaN;

    curvature_dot_1_m_s = gradient(signed_curvature_1_m, uniform_time_s);

    trajectory_arclength_m = [0; cumsum(hypot( ...
        diff(uniform_x_m), diff(uniform_y_m)))];

    motion.time_s = uniform_time_s;
    motion.x_m = uniform_x_m;
    motion.y_m = uniform_y_m;
    motion.vx_m_s = filtered_vx_m_s;
    motion.vy_m_s = filtered_vy_m_s;
    motion.speed_m_s = speed_m_s;
    motion.accel_mag_m_s2 = hypot(filtered_ax_m_s2, filtered_ay_m_s2);
    motion.signed_curvature_1_m = signed_curvature_1_m;
    motion.curvature_dot_1_m_s = curvature_dot_1_m_s;
    motion.trajectory_arclength_m = trajectory_arclength_m;
end


function [arclength_m, curvature_1_m] = compute_path_geometry(path_xy_m)
%COMPUTE_PATH_GEOMETRY Cumulative arc length and signed curvature of a
% closed reference-path polyline (geometry of the SHAPE, not of the
% robot's motion -- see compute_motion_derivatives for that).
%
% Curvature at point i is the turning angle between the incoming and
% outgoing segments, divided by the average of their lengths -- the
% standard discrete estimate of dtheta/ds for a polyline. The turning
% angle's sign follows atan2(cross, dot), so it is positive for a
% left (counterclockwise) turn and negative for a right turn, using
% whatever direction path_xy_m's points happen to be ordered in (see
% the "Reference-path direction assumption" note in the file header).
%
% Input:
%   path_xy_m - N-by-2 closed-loop points, evenly spaced in arc length
%       (as read_shape_band_obj()'s centre_xy_m already is).
%
% Output:
%   arclength_m - N-by-1 cumulative distance from path_xy_m(1, :),
%       ending at the total closed-loop length.
%   curvature_1_m - N-by-1 signed curvature at each point [1/m].

    point_count = size(path_xy_m, 1);
    segment_vectors_m = diff([path_xy_m; path_xy_m(1, :)]);
    segment_lengths_m = hypot(segment_vectors_m(:, 1), segment_vectors_m(:, 2));

    arclength_m = [0; cumsum(segment_lengths_m(1:end - 1))];

    curvature_1_m = zeros(point_count, 1);
    for point_idx = 1:point_count
        previous_idx = mod(point_idx - 2, point_count) + 1;
        incoming_segment = segment_vectors_m(previous_idx, :);
        outgoing_segment = segment_vectors_m(point_idx, :);

        turning_angle_rad = atan2( ...
            incoming_segment(1) * outgoing_segment(2) - ...
            incoming_segment(2) * outgoing_segment(1), ...
            incoming_segment(1) * outgoing_segment(1) + ...
            incoming_segment(2) * outgoing_segment(2));

        average_spacing_m = 0.5 * (segment_lengths_m(previous_idx) + ...
            segment_lengths_m(point_idx));
        curvature_1_m(point_idx) = turning_angle_rad / average_spacing_m;
    end
end


function nearest_index = nearest_reference_index( ...
        path_x_m, path_y_m, reference_xy_m)
%NEAREST_REFERENCE_INDEX Index of the nearest reference sample, per
% point. Copied verbatim from analyze_fine_gross_regions_chen_2d.m.

    sample_count = numel(path_x_m);
    reference_count = size(reference_xy_m, 1);

    best_distance_m = inf(sample_count, 1);
    nearest_index = ones(sample_count, 1);

    for reference_idx = 1:reference_count
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        closer = candidate_m < best_distance_m;
        best_distance_m(closer) = candidate_m(closer);
        nearest_index(closer) = reference_idx;
    end
end


function wrapped_delta_m = wrap_signed_arclength(raw_delta_m, path_length_m)
%WRAP_SIGNED_ARCLENGTH Shortest signed distance around a closed loop.
%
% raw_delta_m (e.g. gaze arc length minus TCP arc length) can be as
% large as +/- path_length_m simply because the path wraps around; this
% rewraps it into (-path_length_m/2, +path_length_m/2], the same trick
% used to wrap an angle into (-pi, pi].

    wrapped_delta_m = mod(raw_delta_m + 0.5 * path_length_m, ...
        path_length_m) - 0.5 * path_length_m;
end


function angle_deg = angle_between_vectors_deg(vector_a, vector_b)
%ANGLE_BETWEEN_VECTORS_DEG Signed angle from vector_a to vector_b [deg].
%
% Uses atan2(cross, dot), so the result is in (-180, 180]: positive
% means vector_b is counterclockwise from vector_a. NaN if either
% vector is (numerically) zero-length, since direction is undefined.

    norm_a = hypot(vector_a(1), vector_a(2));
    norm_b = hypot(vector_b(1), vector_b(2));

    if norm_a < 1e-9 || norm_b < 1e-9
        angle_deg = NaN;
        return;
    end

    cross_value = vector_a(1) * vector_b(2) - vector_a(2) * vector_b(1);
    dot_value = vector_a(1) * vector_b(1) + vector_a(2) * vector_b(2);
    angle_deg = atan2d(cross_value, dot_value);
end


function resampled_label = resample_labels_nearest( ...
        raw_time_s, raw_label, query_time_s)
%RESAMPLE_LABELS_NEAREST Nearest-in-time per-sample class label,
% resampled onto query_time_s (Stage 1's uniform grid).
%
% Labels are text ("None"/"Fine"/"Gross"/"Natural", from
% ur5e_fixed_adm_2d_labeled.py's keyboard-driven "label" column), so
% this cannot use interp1() directly the way the numeric signals above
% are resampled. Instead the nearest RAW SAMPLE INDEX is found first
% (interp1 on plain sample indices, which is numeric) and that index is
% used to look the label up directly -- the same "index, then index
% into the real array" trick nearest_reference_index() uses for the
% reference path.
%
% raw_time_s can have duplicate or out-of-order timestamps (the raw CSV
% is not guaranteed clean), so it is sorted and deduplicated first, the
% same way compute_motion_derivatives() cleans its own input.

    valid_rows = isfinite(raw_time_s);
    clean_time_s = raw_time_s(valid_rows);
    clean_label = raw_label(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_label = clean_label(sort_order);

    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_label = clean_label(unique_indices);

    nearest_index = round(interp1(clean_time_s, ...
        (1:numel(clean_time_s))', query_time_s, 'nearest', 'extrap'));
    resampled_label = clean_label(nearest_index);
end


%% ================================================================
%% Local functions -- Stage 2 helpers (windowing and features)
%% ================================================================

function window_bounds = compute_window_bounds( ...
        time_s, window_length_s, overlap_fraction)
%COMPUTE_WINDOW_BOUNDS Start/end sample indices for a sliding window.
%
% Windows are placed at a fixed time step, not a fixed sample count, so
% window_length_s means the same thing regardless of the trial's sample
% rate. The final partial window (shorter than window_length_s, when
% the trial length is not an exact multiple of the step) is dropped
% rather than padded, so every window is a full, comparable length.
%
% Output:
%   window_bounds - M-by-2, [first_sample_index, last_sample_index] per
%       window, both inclusive indices into time_s.

    step_s = window_length_s * (1 - overlap_fraction);
    trial_duration_s = time_s(end) - time_s(1);

    window_start_times_s = (time_s(1): step_s: ...
        time_s(1) + trial_duration_s - window_length_s)';

    window_bounds = zeros(numel(window_start_times_s), 2);
    for window_idx = 1:numel(window_start_times_s)
        window_start_s = window_start_times_s(window_idx);
        window_end_s = window_start_s + window_length_s;

        window_bounds(window_idx, 1) = find(time_s >= window_start_s, 1);
        window_bounds(window_idx, 2) = find(time_s <= window_end_s, 1, 'last');
    end
end


function robot_features = compute_robot_window_features(sample, ...
        window_range, minimum_speed_m_s, power_law_min_sample_count, ...
        jerk_outlier_window_s, jerk_smoothing_window_s)
%COMPUTE_ROBOT_WINDOW_FEATURES The 11 robot-only features for one window.

    window_speed_m_s = sample.speed_m_s(window_range);
    window_curvature_1_m = sample.signed_curvature_1_m(window_range);
    moving_mask = window_speed_m_s > minimum_speed_m_s;

    robot_features.mean_force_n = mean( ...
        sample.force_mag_n(window_range), 'omitnan');
    robot_features.mean_speed_m_s = mean(window_speed_m_s, 'omitnan');
    robot_features.mean_curvature_1_m = mean( ...
        window_curvature_1_m, 'omitnan');
    robot_features.mean_curvature_dot_1_m_s = mean( ...
        sample.curvature_dot_1_m_s(window_range), 'omitnan');
    robot_features.mean_acceleration_m_s2 = mean( ...
        sample.accel_mag_m_s2(window_range), 'omitnan');
    robot_features.mean_power_w = mean(sample.power_w(window_range), 'omitnan');
    robot_features.median_speed_moving_m_s = median( ...
        window_speed_m_s(moving_mask), 'omitnan');
    robot_features.median_abs_curvature_1_m = median( ...
        abs(window_curvature_1_m), 'omitnan');

    robot_features.dsj = compute_window_dsj( ...
        sample.time_s(window_range), sample.x_m(window_range), ...
        sample.y_m(window_range), jerk_outlier_window_s, ...
        jerk_smoothing_window_s);

    robot_features.power_law_r2 = fit_power_law_r2( ...
        window_speed_m_s, window_curvature_1_m, minimum_speed_m_s, ...
        power_law_min_sample_count);

    robot_features.curvature_sign_reversal_rate_per_m = ...
        compute_sign_reversal_rate(window_curvature_1_m, ...
        sample.trajectory_arclength_m(window_range));
end


function gaze_features = compute_gaze_window_features(window_start_s, ...
        window_end_s, window_delta_s_m, fixation_time_s, ...
        fixation_duration_s, fixation_xy_m, raw_gaze_time_s, raw_gaze_xy_m)
%COMPUTE_GAZE_WINDOW_FEATURES The 6 eye-gaze-only features for one
% window. Unlike the robot features, these read the fixation list and
% the raw-gaze list directly (filtered to this window's time range)
% rather than a per-sample column, since they describe properties of
% fixations/saccades themselves, not a continuously-varying quantity.

    in_window_fixations = fixation_time_s >= window_start_s & ...
        fixation_time_s <= window_end_s;
    window_fixation_count = sum(in_window_fixations);

    gaze_features.fixation_duration_s = mean( ...
        fixation_duration_s(in_window_fixations), 'omitnan');
    gaze_features.fixation_rate_per_s = window_fixation_count / ...
        (window_end_s - window_start_s);
    gaze_features.d_lookahead_m = mean(window_delta_s_m, 'omitnan');

    if window_fixation_count >= 2
        window_fixation_xy_m = fixation_xy_m(in_window_fixations, :);
        transition_vectors_m = diff(window_fixation_xy_m);
        transition_lengths_m = hypot( ...
            transition_vectors_m(:, 1), transition_vectors_m(:, 2));
        gaze_features.fixation_transition_m = mean(transition_lengths_m);
    else
        gaze_features.fixation_transition_m = NaN;
    end

    if window_fixation_count >= 3
        transition_angles_deg = zeros(size(transition_vectors_m, 1) - 1, 1);
        for transition_idx = 1:numel(transition_angles_deg)
            transition_angles_deg(transition_idx) = ...
                angle_between_vectors_deg( ...
                transition_vectors_m(transition_idx, :), ...
                transition_vectors_m(transition_idx + 1, :));
        end
        gaze_features.fixation_transition_angle_deg = ...
            mean(transition_angles_deg, 'omitnan');
    else
        gaze_features.fixation_transition_angle_deg = NaN;
    end

    in_window_raw_gaze = raw_gaze_time_s >= window_start_s & ...
        raw_gaze_time_s <= window_end_s;
    if any(in_window_raw_gaze)
        window_raw_gaze_xy_m = raw_gaze_xy_m(in_window_raw_gaze, :);
        centroid_xy_m = mean(window_raw_gaze_xy_m, 1);
        distances_to_centroid_m = hypot( ...
            window_raw_gaze_xy_m(:, 1) - centroid_xy_m(1), ...
            window_raw_gaze_xy_m(:, 2) - centroid_xy_m(2));
        gaze_features.gaze_dispersion_m = mean(distances_to_centroid_m);
    else
        gaze_features.gaze_dispersion_m = NaN;
    end
end


function fusion_features = compute_fusion_window_features(sample, window_range)
%COMPUTE_FUSION_WINDOW_FEATURES The 6 robot+gaze fusion features for one
% window. Every one of these is a mean of a per-sample column already
% built in Stage 1, since building them there (once, for the whole
% trial) rather than per window keeps this function a plain list of
% one-line definitions.

    fusion_features.angle_gaze_v_deg = mean( ...
        sample.angle_gaze_v_deg(window_range), 'omitnan');
    fusion_features.angle_gaze_f_deg = mean( ...
        sample.angle_gaze_f_deg(window_range), 'omitnan');
    fusion_features.d_lookahead_over_v_s = mean( ...
        sample.lookahead_over_speed_s(window_range), 'omitnan');
    fusion_features.d_lookahead_times_curvature = mean( ...
        sample.lookahead_times_curvature(window_range), 'omitnan');
    fusion_features.curvature_at_gaze_point_1_m = mean( ...
        sample.gaze_path_curvature_1_m(window_range), 'omitnan');

    window_delta_s_m = sample.delta_s_m(window_range);
    valid_delta_s = isfinite(window_delta_s_m);
    if any(valid_delta_s)
        fusion_features.fraction_gaze_ahead = ...
            mean(window_delta_s_m(valid_delta_s) > 0);
    else
        fusion_features.fraction_gaze_ahead = NaN;
    end
end


function dsj = compute_window_dsj(window_time_s, window_x_m, window_y_m, ...
        jerk_outlier_window_s, jerk_smoothing_window_s)
%COMPUTE_WINDOW_DSJ Dimensionless squared jerk (paper Eq. 16) for one
% window's position slice, using the exact chain
% (compute_jerk_series/compute_dsj) already used throughout this
% project, so these DSJ values are comparable with DSJ reported
% elsewhere.

    if numel(window_time_s) < 5
        dsj = NaN;
        return;
    end

    jerk = compute_jerk_series(window_time_s, window_x_m, window_y_m, ...
        jerk_outlier_window_s, jerk_smoothing_window_s);
    dsj = compute_dsj(jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m);
end


function r_squared = fit_power_law_r2(window_speed_m_s, ...
        window_curvature_1_m, minimum_speed_m_s, minimum_sample_count)
%FIT_POWER_LAW_R2 Goodness of fit of v = K*kappa^(-beta) within one
% window, via ordinary least squares on log(v) vs. log(kappa) -- the
% same regression ur5e_chen_var_adm_2d.py's power-law estimator uses,
% applied here as a one-shot batch fit over a single window instead of
% a running online fit. NaN if too few valid samples fall in this
% window to fit a meaningful line.

    valid_mask = window_speed_m_s > minimum_speed_m_s & ...
        isfinite(window_curvature_1_m) & window_curvature_1_m ~= 0;

    log_speed = log(window_speed_m_s(valid_mask));
    log_curvature = log(abs(window_curvature_1_m(valid_mask)));

    sample_count = numel(log_speed);
    if sample_count < minimum_sample_count
        r_squared = NaN;
        return;
    end

    sum_x = sum(log_curvature);
    sum_y = sum(log_speed);
    sum_xx = sum(log_curvature .^ 2);
    sum_xy = sum(log_curvature .* log_speed);

    denominator = sample_count * sum_xx - sum_x ^ 2;
    if abs(denominator) < 1e-9
        r_squared = NaN;
        return;
    end

    slope = (sample_count * sum_xy - sum_x * sum_y) / denominator;
    intercept = (sum_y - slope * sum_x) / sample_count;

    predicted_log_speed = intercept + slope * log_curvature;
    residual_sum_squares = sum((log_speed - predicted_log_speed) .^ 2);
    total_sum_squares = sum((log_speed - mean(log_speed)) .^ 2);

    if total_sum_squares < 1e-12
        r_squared = NaN;
    else
        r_squared = 1 - residual_sum_squares / total_sum_squares;
    end
end


function reversal_rate_per_m = compute_sign_reversal_rate( ...
        window_curvature_1_m, window_trajectory_arclength_m)
%COMPUTE_SIGN_REVERSAL_RATE How often signed curvature changes sign,
% per metre actually travelled within this window. NaN samples (from
% compute_motion_derivatives' minimum_speed_m_s gate) are dropped before
% counting sign changes, since a "reversal" spanning a stretch where
% direction was undefined is not a meaningful reversal.

    valid_mask = isfinite(window_curvature_1_m);
    valid_curvature = window_curvature_1_m(valid_mask);

    distance_travelled_m = window_trajectory_arclength_m(end) - ...
        window_trajectory_arclength_m(1);

    if numel(valid_curvature) < 2 || distance_travelled_m <= 0
        reversal_rate_per_m = NaN;
        return;
    end

    sign_changes = sum(diff(sign(valid_curvature)) ~= 0);
    reversal_rate_per_m = sign_changes / distance_travelled_m;
end


function merged = merge_structs(varargin)
%MERGE_STRUCTS Combine several scalar structs into one, field by field.

    merged = struct();
    for struct_idx = 1:numel(varargin)
        field_names = fieldnames(varargin{struct_idx});
        for field_idx = 1:numel(field_names)
            merged.(field_names{field_idx}) = ...
                varargin{struct_idx}.(field_names{field_idx});
        end
    end
end


function [majority_label, purity] = most_common_label(label_slice)
%MOST_COMMON_LABEL Majority-vote class label for one window, plus the
% fraction of the window's samples that agree with it.
%
% purity is in [0, 1]: 1.0 means every sample in the window carried the
% same label; lower values flag a window whose time span crossed a
% label change while the operator was recording (see the file header
% and the Output doc comment).
%
% Ties are broken alphabetically -- unique()'s default sort order for
% string arrays -- so the result never depends on sample order or on an
% unspecified/version-dependent tie-break inside a built-in like
% mode(categorical(...)).

    unique_labels = unique(label_slice);  % ascending, i.e. alphabetical
    label_counts = zeros(size(unique_labels));
    for label_idx = 1:numel(unique_labels)
        label_counts(label_idx) = sum(label_slice == unique_labels(label_idx));
    end

    [best_count, best_idx] = max(label_counts);
    majority_label = unique_labels(best_idx);
    purity = double(best_count) / numel(label_slice);
end


%% ================================================================
%% Local functions -- copied verbatim from this project's other scripts
%% ================================================================

function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
        obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh. Copied
% verbatim from plot_raw_data_fixed_adm_2d.m.
%
% Input:
%   obj_file - Wavefront OBJ of a thin flat band, as exported by the CAD
%       tool used for this project's shapes.
%   resample_spacing_m - arclength spacing of the returned loops [m].
%
% Output:
%   outer_xy_m, inner_xy_m - resampled outer and inner edge loops, each
%       N-by-2 in the mesh CAD frame [m].
%   centre_xy_m - the reference path, midway between the two edges.
%       Empty when the mesh does not resolve into exactly two loops.
%
% Method:
%   The band is extruded in z, so its bottom cap carries the planar
%   outline. Cap triangles are collected, and the outline is the set of
%   edges belonging to exactly one triangle: interior triangulation
%   diagonals are shared by two and drop out. Those boundary edges chain
%   into closed loops; the loop enclosing the larger area is the outer
%   edge. The centre line takes, for each outer sample, the nearest
%   inner sample and averages the two, which is what an offset band
%   gives.

    outer_xy_m = [];
    inner_xy_m = [];
    centre_xy_m = [];

    file_id = fopen(obj_file, 'r');
    if file_id < 0
        return;
    end
    file_text = textscan(file_id, '%s', 'Delimiter', '\n', ...
        'WhiteSpace', '');
    fclose(file_id);
    file_lines = file_text{1};

    vertices = zeros(0, 3);
    faces = {};

    for line_idx = 1:numel(file_lines)
        this_line = strtrim(file_lines{line_idx});
        if startsWith(this_line, 'v ')
            values = sscanf(this_line(2:end), '%f');
            vertices(end + 1, 1:3) = values(1:3)'; %#ok<AGROW>
        elseif startsWith(this_line, 'f ')
            tokens = strsplit(strtrim(this_line(2:end)));
            face_indices = zeros(1, numel(tokens));
            for token_idx = 1:numel(tokens)
                slash_parts = strsplit(tokens{token_idx}, '/');
                face_indices(token_idx) = str2double(slash_parts{1});
            end
            faces{end + 1} = face_indices; %#ok<AGROW>
        end
    end

    if isempty(vertices) || isempty(faces)
        return;
    end

    z_bottom = min(vertices(:, 3));
    z_tolerance = 1e-9 + 1e-6 * (max(vertices(:, 3)) - z_bottom);

    edge_list = zeros(0, 2);
    for face_idx = 1:numel(faces)
        face_indices = faces{face_idx};
        if ~all(abs(vertices(face_indices, 3) - z_bottom) <= z_tolerance)
            continue;
        end
        for corner_idx = 1:numel(face_indices)
            next_corner = mod(corner_idx, numel(face_indices)) + 1;
            edge_list(end + 1, :) = sort([ ...
                face_indices(corner_idx), ...
                face_indices(next_corner)]); %#ok<AGROW>
        end
    end

    if isempty(edge_list)
        return;
    end

    [unique_edges, ~, edge_group] = unique(edge_list, 'rows');
    edge_use_count = accumarray(edge_group, 1);
    boundary_edges = unique_edges(edge_use_count == 1, :);

    if isempty(boundary_edges)
        return;
    end

    remaining_edges = boundary_edges;
    loops = {};
    while ~isempty(remaining_edges)
        loop_indices = remaining_edges(1, 1);
        current_vertex = remaining_edges(1, 2);
        remaining_edges(1, :) = [];

        while true
            loop_indices(end + 1) = current_vertex; %#ok<AGROW>
            next_row = find(any(remaining_edges == current_vertex, 2), 1);
            if isempty(next_row)
                break;
            end
            edge_pair = remaining_edges(next_row, :);
            remaining_edges(next_row, :) = [];
            if edge_pair(1) == current_vertex
                current_vertex = edge_pair(2);
            else
                current_vertex = edge_pair(1);
            end
            if current_vertex == loop_indices(1)
                break;
            end
        end

        if numel(loop_indices) > 3
            loops{end + 1} = vertices(loop_indices, 1:2); %#ok<AGROW>
        end
    end

    if numel(loops) ~= 2
        return;
    end

    loop_areas = zeros(1, 2);
    for loop_idx = 1:2
        loop_xy = loops{loop_idx};
        loop_areas(loop_idx) = abs(0.5 * sum( ...
            loop_xy(:, 1) .* circshift(loop_xy(:, 2), -1) - ...
            circshift(loop_xy(:, 1), -1) .* loop_xy(:, 2)));
    end

    [~, outer_index] = max(loop_areas);
    inner_index = 3 - outer_index;

    outer_xy_m = resample_closed_loop( ...
        loops{outer_index}, resample_spacing_m);
    inner_xy_m = resample_closed_loop( ...
        loops{inner_index}, resample_spacing_m);

    centre_xy_m = zeros(size(outer_xy_m));
    for sample_idx = 1:size(outer_xy_m, 1)
        distances_m = hypot( ...
            inner_xy_m(:, 1) - outer_xy_m(sample_idx, 1), ...
            inner_xy_m(:, 2) - outer_xy_m(sample_idx, 2));
        [~, nearest_index] = min(distances_m);
        centre_xy_m(sample_idx, :) = 0.5 * ( ...
            outer_xy_m(sample_idx, :) + inner_xy_m(nearest_index, :));
    end
end


function resampled_xy_m = resample_closed_loop(loop_xy_m, spacing_m)
%RESAMPLE_CLOSED_LOOP Resample a closed polygon at uniform arclength.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.

    closed_xy_m = [loop_xy_m; loop_xy_m(1, :)];
    segment_lengths_m = hypot( ...
        diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
    arclength_m = [0; cumsum(segment_lengths_m)];

    keep_mask = [true; segment_lengths_m > 0];
    arclength_m = arclength_m(keep_mask);
    closed_xy_m = closed_xy_m(keep_mask, :);

    total_length_m = arclength_m(end);
    sample_count = max(4, round(total_length_m / spacing_m));
    query_m = linspace(0, total_length_m, sample_count)';

    resampled_xy_m = [ ...
        interp1(arclength_m, closed_xy_m(:, 1), query_m, 'linear'), ...
        interp1(arclength_m, closed_xy_m(:, 2), query_m, 'linear')];
end


function jerk = compute_jerk_series( ...
        raw_time_s, raw_x_m, raw_y_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.
%
% Jerk is a third derivative, so noise is amplified three times. The
% chain resamples onto a uniform grid, rejects outliers, smooths, and
% then smooths again after every differentiation.

    valid_rows = isfinite(raw_time_s) & isfinite(raw_x_m) & ...
        isfinite(raw_y_m);
    clean_time_s = raw_time_s(valid_rows);
    clean_x_m = raw_x_m(valid_rows);
    clean_y_m = raw_y_m(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_x_m = clean_x_m(sort_order);
    clean_y_m = clean_y_m(sort_order);

    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_x_m = clean_x_m(unique_indices);
    clean_y_m = clean_y_m(unique_indices);

    sample_intervals_s = diff(clean_time_s);
    median_interval_s = median(sample_intervals_s(sample_intervals_s > 0));

    uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';
    uniform_x_m = interp1(clean_time_s, clean_x_m, uniform_time_s, 'linear');
    uniform_y_m = interp1(clean_time_s, clean_y_m, uniform_time_s, 'linear');

    outlier_samples = window_to_odd_sample_count( ...
        outlier_window_s, median_interval_s, numel(uniform_time_s));
    smoothing_samples = window_to_odd_sample_count( ...
        smoothing_window_s, median_interval_s, numel(uniform_time_s));

    if outlier_samples > 1
        uniform_x_m = movmedian(uniform_x_m, outlier_samples, ...
            'Endpoints', 'shrink');
        uniform_y_m = movmedian(uniform_y_m, outlier_samples, ...
            'Endpoints', 'shrink');
    end

    filtered_x_m = movmean(uniform_x_m, smoothing_samples, ...
        'Endpoints', 'shrink');
    filtered_y_m = movmean(uniform_y_m, smoothing_samples, ...
        'Endpoints', 'shrink');

    velocity_x_m_s = movmean(gradient(filtered_x_m, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');
    velocity_y_m_s = movmean(gradient(filtered_y_m, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');

    acceleration_x_m_s2 = movmean( ...
        gradient(velocity_x_m_s, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');
    acceleration_y_m_s2 = movmean( ...
        gradient(velocity_y_m_s, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');

    jerk_x_m_s3 = gradient(acceleration_x_m_s2, uniform_time_s);
    jerk_y_m_s3 = gradient(acceleration_y_m_s2, uniform_time_s);

    jerk.time_s = uniform_time_s;
    jerk.squared_jerk_m2_s6 = jerk_x_m_s3 .^ 2 + jerk_y_m_s3 .^ 2;
    jerk.path_length_m = sum(hypot( ...
        diff(filtered_x_m), diff(filtered_y_m)));
end


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample count.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.

    if window_s <= 0
        sample_count = 1;
        return;
    end

    sample_count = max(3, round(window_s / sample_interval_s));
    if mod(sample_count, 2) == 0
        sample_count = sample_count + 1;
    end

    largest_odd_count = maximum_sample_count;
    if mod(largest_odd_count, 2) == 0
        largest_odd_count = largest_odd_count - 1;
    end
    sample_count = min(sample_count, largest_odd_count);
end


function dsj = compute_dsj(time_s, squared_jerk_m2_s6, path_length_m)
%COMPUTE_DSJ Dimensionless squared jerk, paper Eq. (16). Copied verbatim
% from plot_raw_data_fixed_adm_2d.m.
%
%   DSJ = integral( jerk^2 ) dt * (t2 - t1)^5 / A^2
%
% with A the path length.

    if path_length_m <= 0 || numel(time_s) < 2
        dsj = NaN;
        return;
    end

    movement_duration_s = time_s(end) - time_s(1);
    squared_jerk_integral = trapz(time_s, squared_jerk_m2_s6);

    dsj = squared_jerk_integral * movement_duration_s ^ 5 / ...
        path_length_m ^ 2;
end


function [fixation_xy_m, fixation_time_s, fixation_duration_s, ...
        raw_gaze_xy_m, raw_gaze_time_s] = detect_fixations_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres in CoppeliaSim [m].
% Copied verbatim (plus fixation_duration_s, added to the output list)
% from plot_raw_data_chen_var_adm_2d.m / compare_chen_vs_gaze_assisted_
% 2d.m, which reuse the I-VT and UCS-to-CoppeliaSim formulas from
% compute_plot_similarity_fixation_direction_force_vel.m.
%
% fixation_xy_m is N-by-2 and empty when no fixation meets the minimum
% duration. raw_gaze_xy_m is every sample where BOTH eyes are valid,
% strictly averaged (left + right) / 2, mapped the same way;
% raw_gaze_time_s is its matching trial time [s]. fixation_time_s is
% the mid-time of each fixation, on the shared trial clock.

    left_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);
    left_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_origin_in_user_coordinate_system, 3);
    right_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_origin_in_user_coordinate_system, 3);
    left_display_gaze = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display_gaze = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1;
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1;
    left_origin_valid = get_numeric_column( ...
        gaze_table.left_gaze_origin_validity) == 1;
    right_origin_valid = get_numeric_column( ...
        gaze_table.right_gaze_origin_validity) == 1;

    gaze_time_s = get_numeric_column(gaze_table.time_s);

    left_point_valid = left_point_valid & finite_rows(left_gaze_ucs_mm);
    right_point_valid = right_point_valid & finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end

    [left_gaze_direction, left_eye_valid] = calculate_eye_direction( ...
        left_gaze_origin_ucs_mm, left_gaze_ucs_mm, ...
        left_origin_valid, left_point_valid);
    [right_gaze_direction, right_eye_valid] = calculate_eye_direction( ...
        right_gaze_origin_ucs_mm, right_gaze_ucs_mm, ...
        right_origin_valid, right_point_valid);

    [combined_gaze_ucs_mm, combined_point_valid] = combine_valid_eye_points( ...
        left_gaze_ucs_mm, right_gaze_ucs_mm, left_eye_valid, right_eye_valid);
    [combined_gaze_direction, combined_direction_valid] = ...
        combine_valid_eye_directions( ...
        left_gaze_direction, right_gaze_direction, ...
        left_eye_valid, right_eye_valid);
    combined_gaze_valid = combined_point_valid & combined_direction_valid;
    combined_gaze_ucs_mm(~combined_gaze_valid, :) = NaN;

    angular_velocity_deg_s = calculate_angular_velocity( ...
        combined_gaze_direction, gaze_time_s, combined_gaze_valid);
    fixation_candidate = combined_gaze_valid & ...
        isfinite(angular_velocity_deg_s) & ...
        angular_velocity_deg_s <= velocity_threshold_deg_s;

    candidate_changes = diff(double([false; fixation_candidate; false]));
    candidate_start_indices = find(candidate_changes == 1);
    candidate_end_indices = find(candidate_changes == -1) - 1;

    fixation_mean_ucs_x_mm = zeros(0, 1);
    fixation_mean_ucs_y_mm = zeros(0, 1);
    fixation_time_s = zeros(0, 1);
    fixation_duration_s = zeros(0, 1);

    for candidate_idx = 1:numel(candidate_start_indices)
        start_idx = candidate_start_indices(candidate_idx);
        end_idx = candidate_end_indices(candidate_idx);
        candidate_duration_s = gaze_time_s(end_idx) - gaze_time_s(start_idx);

        if candidate_duration_s < minimum_fixation_duration_s
            continue;
        end

        fixation_rows = start_idx:end_idx;
        mean_ucs_mm = mean(combined_gaze_ucs_mm(fixation_rows, :), 1);
        fixation_mean_ucs_x_mm(end + 1, 1) = mean_ucs_mm(1); %#ok<AGROW>
        fixation_mean_ucs_y_mm(end + 1, 1) = mean_ucs_mm(2); %#ok<AGROW>

        fixation_time_s(end + 1, 1) = 0.5 * ( ...
            gaze_time_s(start_idx) + gaze_time_s(end_idx)); %#ok<AGROW>
        fixation_duration_s(end + 1, 1) = candidate_duration_s; %#ok<AGROW>
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    fixation_ucs_xy_mm = [fixation_mean_ucs_x_mm, fixation_mean_ucs_y_mm];
    fixation_displacement_ucs_mm = fixation_ucs_xy_mm - tobii_reference_xy_mm;
    fixation_displacement_coppelia_m = fixation_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    fixation_xy_m = coppelia_reference_xy_m + fixation_displacement_coppelia_m;

    both_eyes_point_valid = left_point_valid & right_point_valid;
    raw_gaze_ucs_xy_mm = 0.5 * ( ...
        left_gaze_ucs_mm(both_eyes_point_valid, 1:2) + ...
        right_gaze_ucs_mm(both_eyes_point_valid, 1:2));
    raw_gaze_displacement_ucs_mm = raw_gaze_ucs_xy_mm - tobii_reference_xy_mm;
    raw_gaze_displacement_coppelia_m = raw_gaze_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    raw_gaze_xy_m = coppelia_reference_xy_m + raw_gaze_displacement_coppelia_m;
    raw_gaze_time_s = gaze_time_s(both_eyes_point_valid);
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV
% cells. Copied verbatim from compute_plot_similarity_fixation_
% direction_force_vel.m.

    if ischar(raw_column)
        raw_column = cellstr(raw_column);
    end

    sample_count = numel(raw_column);
    vector_values = NaN(sample_count, vector_size);
    for sample_idx = 1:sample_count
        if iscell(raw_column)
            raw_value = raw_column{sample_idx};
        else
            raw_value = raw_column(sample_idx);
        end

        if isstring(raw_value)
            raw_value = char(raw_value);
        end
        if isempty(raw_value)
            continue;
        end

        raw_value = strrep(raw_value, '[', '');
        raw_value = strrep(raw_value, ']', '');
        raw_value = strrep(raw_value, ',', ' ');
        parsed_values = sscanf(raw_value, '%f');
        value_count = min(numel(parsed_values), vector_size);
        vector_values(sample_idx, 1:value_count) = ...
            parsed_values(1:value_count)';
    end
end


function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.
% Copied verbatim from compute_plot_similarity_fixation_direction_
% force_vel.m.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display
% bounds. Copied verbatim from compute_plot_similarity_fixation_
% direction_force_vel.m.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end


function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes. Copied
% verbatim from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = size(left_point, 1);
    valid_eye_count = double(left_valid) + double(right_valid);
    combined_valid = valid_eye_count > 0;

    point_sum = zeros(sample_count, 3);
    point_sum(left_valid, :) = left_point(left_valid, :);
    point_sum(right_valid, :) = point_sum(right_valid, :) + ...
        right_point(right_valid, :);

    combined_point = NaN(sample_count, 3);
    combined_point(combined_valid, :) = ...
        point_sum(combined_valid, :) ./ valid_eye_count(combined_valid);
end


function [unit_direction, eye_valid] = calculate_eye_direction( ...
        gaze_origin_ucs_mm, gaze_point_ucs_mm, ...
        origin_validity, point_validity)
%CALCULATE_EYE_DIRECTION Build one eye's unit direction in Tobii UCS.
% Copied verbatim from compute_plot_similarity_fixation_direction_
% force_vel.m.

    direction = gaze_point_ucs_mm - gaze_origin_ucs_mm;
    direction_norm = sqrt(sum(direction .^ 2, 2));
    eye_valid = origin_validity & point_validity & ...
        finite_rows(gaze_origin_ucs_mm) & finite_rows(gaze_point_ucs_mm) & ...
        isfinite(direction_norm) & direction_norm > 0;

    unit_direction = NaN(size(direction));
    unit_direction(eye_valid, :) = direction(eye_valid, :) ./ ...
        direction_norm(eye_valid);
end


function [combined_direction, combined_valid] = ...
        combine_valid_eye_directions( ...
        left_direction, right_direction, left_valid, right_valid)
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye
% directions. Copied verbatim from compute_plot_similarity_fixation_
% direction_force_vel.m.

    sample_count = size(left_direction, 1);
    combined_valid = left_valid | right_valid;

    direction_sum = zeros(sample_count, 3);
    direction_sum(left_valid, :) = left_direction(left_valid, :);
    direction_sum(right_valid, :) = direction_sum(right_valid, :) + ...
        right_direction(right_valid, :);

    direction_norm = sqrt(sum(direction_sum .^ 2, 2));
    combined_valid = combined_valid & isfinite(direction_norm) & ...
        direction_norm > 0;

    combined_direction = NaN(sample_count, 3);
    combined_direction(combined_valid, :) = ...
        direction_sum(combined_valid, :) ./ direction_norm(combined_valid);
end


function angular_velocity_deg_s = calculate_angular_velocity( ...
        gaze_direction, time_s, gaze_valid)
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s]. Copied
% verbatim from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = numel(time_s);
    angular_velocity_deg_s = NaN(sample_count, 1);
    sample_interval_s = diff(time_s);
    valid_pair = gaze_valid(1:end-1) & gaze_valid(2:end) & ...
        isfinite(sample_interval_s) & sample_interval_s > 0;

    direction_dot_product = sum( ...
        gaze_direction(1:end-1, :) .* gaze_direction(2:end, :), 2);
    direction_dot_product = min(max(direction_dot_product, -1.0), 1.0);
    angular_step_deg = acosd(direction_dot_product);

    interval_velocity_deg_s = NaN(sample_count - 1, 1);
    interval_velocity_deg_s(valid_pair) = ...
        angular_step_deg(valid_pair) ./ sample_interval_s(valid_pair);
    angular_velocity_deg_s(2:end) = interval_velocity_deg_s;

    valid_changes = diff(double([false; gaze_valid; false]));
    run_starts = find(valid_changes == 1);
    run_ends = find(valid_changes == -1) - 1;
    for run_idx = 1:numel(run_starts)
        start_idx = run_starts(run_idx);
        end_idx = run_ends(run_idx);

        if end_idx <= start_idx
            continue;
        end

        if isfinite(angular_velocity_deg_s(start_idx + 1))
            angular_velocity_deg_s(start_idx) = ...
                angular_velocity_deg_s(start_idx + 1);
        end
    end
end
