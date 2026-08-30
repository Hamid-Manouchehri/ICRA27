%% AWESOME_SCRIPT: Extract windowed JOINT (gaze+robot) features and
%% correlations.
%
% Third of a matched trio with gaze_feature_extraction.m (gaze-only, 102
% features) and robot_feature_extraction.m (robot-only, 96 features):
% this one is every feature that genuinely needs BOTH signals -- remove
% either one and the feature is undefined, not just unused. Built so a
% robot-only vs. gaze-only vs. joint classifier comparison has a real,
% comparably-sized third leg instead of the 30 gaze-relative features
% that were previously only available bundled inside the gaze bank.
%
% Frame, and why it differs from gaze_feature_extraction.m:
%   A "distance between gaze and hand" only means something if both are
%   in the SAME frame. gaze_feature_extraction.m pairs raw normalized
%   display-area gaze with a separately min-max-normalized end effector
%   -- an internally consistent scheme, but not physical units, and not
%   the same frame robot_feature_extraction.m's position features use.
%   This script instead maps gaze through the CALIBRATED Tobii-UCS ->
%   CoppeliaSim mapping this project uses elsewhere (extract_pilot_
%   features_fixed_adm_2d.m's detect_fixations_coppelia_xy(), also
%   ported to Python as gaze_features.map_gaze_ucs_to_coppelia()), so
%   gaze position and robot position are both in real metres, directly
%   comparable. That is why this script needs a THIRD input file
%   (calibration_json_file) neither of the other two needs.
%
% Inputs:
%   Tobii CSV with UCS gaze [mm] and system timestamps [us].
%   UR5e CSV with position [m], measured/commanded velocity [m/s],
%   angular velocity [rad/s], filtered planar force [N], time [s].
%   Calibration JSON (manual_tobii_reference_xy_mm, manual_coppelia_
%   reference_xy_m) from the same trial's calibration step.
%
% Outputs retained in the MATLAB workspace:
%   window_results      - all features, DSJ, and mean force per window.
%   correlation_results, correlation_ranked_by_dsj,
%   correlation_ranked_by_mean_force - same as the other two scripts.
%
% Shared feature-function interface:
%   value = feature_name(Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%   Gx, Gy, Gt   - calibrated gaze position [m] and its own time [s].
%   Px, Py, Pt   - robot position (CoppeliaSim frame) [m] and time [s].
%   Vx, Vy       - MEASURED TCP velocity [m/s].
%   Fx, Fy       - filtered planar force [N].
%   Wmag         - angular speed magnitude [rad/s].
%   Vcx, Vcy     - commanded planar velocity [m/s].
%   Every feature that pairs a per-sample gaze quantity with a per-
%   sample robot quantity does so through synchronize_robot_onto_gaze()
%   below, which interpolates robot signals (already differentiated at
%   the robot's own native, much higher sample rate -- never the other
%   way around) onto gaze timestamps, restricted to the overlap of the
%   two clocks.
%
% Feature groups (see the "Feature functions" section for the exact
% list):
%   001-030 gaze-vs-robot position/speed relationship -- distance,
%           offset statistics, correlation, cross-correlation/lag,
%           velocity alignment, speed ratio. Direct adaptation of
%           gaze_feature_extraction.m's own 30 gaze-relative features
%           (071-100 there), now standalone and using the calibrated
%           frame instead of the normalized display.
%   031-036 gaze-vs-curvature -- does gaze behave differently when the
%           robot's path is tightly curved vs. nearly straight.
%   037-042 gaze-vs-force -- including angle_gaze_offset_to_force, a
%           direct, path-free port of extract_pilot_features_fixed_
%           adm_2d.m's angle_gaze_f_deg. DAMPING-CONFOUNDED (see below).
%   043-044 gaze-vs-angular-speed (wrist rotation).
%   045-046 gaze-vs-command-tracking-error. DAMPING-CONFOUNDED.
%   047-051 gaze lookahead relative to robot motion direction --
%           path-free ports of extract_pilot_features_fixed_adm_2d.m's
%           angle_gaze_v_deg / d_lookahead_over_v_s / d_lookahead_
%           times_curvature / fraction_gaze_ahead, using the straight-
%           line projection of (gaze - robot) onto the velocity
%           direction instead of arc-length along a reference path
%           (this script has no shape .obj to project onto).
%
% Same caveat as robot_feature_extraction.m: every FORCE-derived and
% TRACKING-ERROR-derived feature here is partly a function of the
% CURRENT admittance damping, not just behaviour -- see that script's
% header for the full reasoning. Treat their ranking with that in mind.

clear;
close all
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
gaze_data_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'tobii_fixed_adm_2d_labeled_fine_gross_shape_d100_s2.csv');
robot_data_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'ur5e_fixed_adm_2d_labeled_fine_gross_shape_d100_s2.csv');
calibration_json_file = fullfile(script_directory, ...
    '..', 'data', 'calibrations', ...
    'tobii_coppelia_reference_calibration_fine_gross_shape_s2.json');

window_size_s = 3.5;  % TODO window duration [s], recommended 0.25 to 2
window_step_s = 0.02; % TODO step [s]; default matches the other two
                       % scripts in this trio -- see their headers for
                       % why this is fine for a correlation study but
                       % not classifier training data.
minimum_gaze_samples = 6;  % TODO valid gaze samples/window, recommended >= 6
minimum_robot_samples = 10;  % TODO robot samples/window, recommended >= 10

% TODO true excludes gaze estimates outside the normalized display area
% (a data-quality gate on the RAW Tobii signal, unrelated to the
% calibrated-mapping frame).
require_gaze_on_display = true;
% TODO true requires BOTH eyes valid for a gaze sample to be used
% (matches extract_pilot_features_fixed_adm_2d.m's raw-gaze branch).
require_both_eyes_valid = true;
% TODO calibrated-mapping settings, matching this project's other
% calibrated-mapping scripts.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

if window_size_s <= 0 || window_step_s <= 0
    error('window_size_s and window_step_s must be positive.');
end

%% Load raw recordings and calibration
gaze_table = readtable(gaze_data_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
robot_table = readtable(robot_data_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
calibration_reference = jsondecode(fileread(calibration_json_file));
tobii_reference_xy_mm = double( ...
    calibration_reference.manual_tobii_reference_xy_mm(:)');
coppelia_reference_xy_m = double( ...
    calibration_reference.manual_coppelia_reference_xy_m(:)');

required_gaze_variables = {
    'system_time_stamp'
    'left_gaze_point_in_user_coordinate_system'
    'right_gaze_point_in_user_coordinate_system'
    'left_gaze_point_on_display_area'
    'right_gaze_point_on_display_area'
    'left_gaze_point_validity'
    'right_gaze_point_validity'
    };
required_robot_variables = {
    'time_s'
    'coppeliasim_x_m'
    'coppeliasim_y_m'
    'tcp_vx_actual_m_s'
    'tcp_vy_actual_m_s'
    'tcp_wx_actual_rad_s'
    'tcp_wy_actual_rad_s'
    'tcp_wz_actual_rad_s'
    'vx_cmd_m_s'
    'vy_cmd_m_s'
    'Fx_filt_N'
    'Fy_filt_N'
    'x_m'
    'y_m'
    };
require_table_variables(gaze_table, required_gaze_variables, 'Tobii CSV');
require_table_variables(robot_table, required_robot_variables, 'UR5e CSV');

%% Decode robot signals
Pt_all = get_numeric_column(robot_table.time_s);
end_effector_x_m_all = get_numeric_column(robot_table.x_m);
end_effector_y_m_all = get_numeric_column(robot_table.y_m);
Px_all = get_numeric_column(robot_table.coppeliasim_x_m);
Py_all = get_numeric_column(robot_table.coppeliasim_y_m);
Vx_all = get_numeric_column(robot_table.tcp_vx_actual_m_s);
Vy_all = get_numeric_column(robot_table.tcp_vy_actual_m_s);
Fx_all = get_numeric_column(robot_table.Fx_filt_N);
Fy_all = get_numeric_column(robot_table.Fy_filt_N);
Vcx_all = get_numeric_column(robot_table.vx_cmd_m_s);
Vcy_all = get_numeric_column(robot_table.vy_cmd_m_s);

wx_all = get_numeric_column(robot_table.tcp_wx_actual_rad_s);
wy_all = get_numeric_column(robot_table.tcp_wy_actual_rad_s);
wz_all = get_numeric_column(robot_table.tcp_wz_actual_rad_s);
Wmag_all = sqrt(wx_all.^2 + wy_all.^2 + wz_all.^2);

robot_valid = isfinite(Pt_all) & isfinite(Px_all) & isfinite(Py_all) & ...
    isfinite(end_effector_x_m_all) & isfinite(end_effector_y_m_all) & ...
    isfinite(Vx_all) & isfinite(Vy_all);

Pt = Pt_all(robot_valid);
end_effector_x_m = end_effector_x_m_all(robot_valid);
end_effector_y_m = end_effector_y_m_all(robot_valid);
Px = Px_all(robot_valid);
Py = Py_all(robot_valid);
Vx = Vx_all(robot_valid);
Vy = Vy_all(robot_valid);
Fx = Fx_all(robot_valid);
Fy = Fy_all(robot_valid);
Vcx = Vcx_all(robot_valid);
Vcy = Vcy_all(robot_valid);
Wmag = Wmag_all(robot_valid);

%% Decode and calibrate gaze
left_gaze_ucs_mm = parse_vector_column( ...
    gaze_table.left_gaze_point_in_user_coordinate_system, 3);
right_gaze_ucs_mm = parse_vector_column( ...
    gaze_table.right_gaze_point_in_user_coordinate_system, 3);
left_display_gaze = parse_vector_column( ...
    gaze_table.left_gaze_point_on_display_area, 2);
right_display_gaze = parse_vector_column( ...
    gaze_table.right_gaze_point_on_display_area, 2);

left_point_valid = get_numeric_column( ...
    gaze_table.left_gaze_point_validity) == 1;
right_point_valid = get_numeric_column( ...
    gaze_table.right_gaze_point_validity) == 1;
left_point_valid = left_point_valid & finite_rows(left_gaze_ucs_mm);
right_point_valid = right_point_valid & finite_rows(right_gaze_ucs_mm);

if require_gaze_on_display
    left_point_valid = left_point_valid & ...
        point_is_on_display(left_display_gaze);
    right_point_valid = right_point_valid & ...
        point_is_on_display(right_display_gaze);
end

gaze_system_time_us = get_numeric_column(gaze_table.system_time_stamp);
first_gaze_timestamp_idx = find(isfinite(gaze_system_time_us), 1, 'first');
if isempty(first_gaze_timestamp_idx)
    error('The Tobii CSV contains no finite system timestamps.');
end
gaze_start_time_us = gaze_system_time_us(first_gaze_timestamp_idx);
Gt_all = (gaze_system_time_us - gaze_start_time_us) * 1e-6;

if require_both_eyes_valid
    gaze_point_valid = left_point_valid & right_point_valid;
    gaze_ucs_xy_mm_all = 0.5 * ( ...
        left_gaze_ucs_mm(:, 1:2) + right_gaze_ucs_mm(:, 1:2));
else
    eye_count = double(left_point_valid) + double(right_point_valid);
    gaze_point_valid = eye_count > 0;
    gaze_sum_xy = zeros(size(left_gaze_ucs_mm, 1), 2);
    gaze_sum_xy(left_point_valid, :) = ...
        gaze_sum_xy(left_point_valid, :) + ...
        left_gaze_ucs_mm(left_point_valid, 1:2);
    gaze_sum_xy(right_point_valid, :) = ...
        gaze_sum_xy(right_point_valid, :) + ...
        right_gaze_ucs_mm(right_point_valid, 1:2);
    gaze_ucs_xy_mm_all = NaN(size(gaze_sum_xy));
    gaze_ucs_xy_mm_all(gaze_point_valid, :) = ...
        gaze_sum_xy(gaze_point_valid, :) ./ eye_count(gaze_point_valid);
end

gaze_valid = gaze_point_valid & isfinite(Gt_all);
Gt = Gt_all(gaze_valid);
gaze_ucs_xy_mm = gaze_ucs_xy_mm_all(gaze_valid, :);

coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
gaze_displacement_ucs_mm = gaze_ucs_xy_mm - tobii_reference_xy_mm;
gaze_displacement_coppelia_m = gaze_displacement_ucs_mm .* ...
    tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
gaze_xy_m = coppelia_reference_xy_m + gaze_displacement_coppelia_m;
Gx = gaze_xy_m(:, 1);
Gy = gaze_xy_m(:, 2);

%% Define feature functions
feature_functions = {
    @feature_001_mean_gaze_ee_distance
    @feature_002_std_gaze_ee_distance
    @feature_003_median_gaze_ee_distance
    @feature_004_max_gaze_ee_distance
    @feature_005_rms_gaze_ee_distance
    @feature_006_p90_gaze_ee_distance
    @feature_007_min_gaze_ee_distance
    @feature_008_final_gaze_ee_distance
    @feature_009_gaze_ee_distance_slope
    @feature_010_mean_relative_x
    @feature_011_mean_relative_y
    @feature_012_std_relative_x
    @feature_013_std_relative_y
    @feature_014_relative_path_length
    @feature_015_mean_relative_speed
    @feature_016_std_relative_speed
    @feature_017_fraction_within_0p01
    @feature_018_fraction_within_0p02
    @feature_019_fraction_within_0p05
    @feature_020_gaze_ee_x_correlation
    @feature_021_gaze_ee_y_correlation
    @feature_022_gaze_ee_speed_correlation
    @feature_023_mean_velocity_cosine_similarity
    @feature_024_aligned_velocity_fraction
    @feature_025_gaze_to_ee_speed_ratio
    @feature_026_max_x_cross_correlation
    @feature_027_x_lead_at_max_correlation_s
    @feature_028_max_y_cross_correlation
    @feature_029_y_lead_at_max_correlation_s
    @feature_030_mean_gaze_lead_along_ee_motion
    @feature_031_correlation_gaze_ee_distance_curvature
    @feature_032_correlation_gaze_speed_absolute_curvature
    @feature_033_mean_curvature_when_gaze_close
    @feature_034_mean_curvature_when_gaze_far
    @feature_035_fraction_gaze_close_during_high_curvature
    @feature_036_correlation_gaze_offset_angle_curvature
    @feature_037_angle_gaze_offset_to_force
    @feature_038_correlation_gaze_ee_distance_force
    @feature_039_correlation_gaze_speed_force
    @feature_040_mean_gaze_ee_distance_high_force
    @feature_041_mean_gaze_ee_distance_low_force
    @feature_042_fraction_gaze_close_during_high_force
    @feature_043_correlation_gaze_speed_angular_speed
    @feature_044_correlation_gaze_ee_distance_angular_speed
    @feature_045_correlation_gaze_speed_tracking_error
    @feature_046_correlation_gaze_ee_distance_tracking_error
    @feature_047_angle_gaze_offset_to_velocity
    @feature_048_mean_lookahead_distance
    @feature_049_fraction_gaze_ahead
    @feature_050_lookahead_distance_over_speed
    @feature_051_lookahead_times_curvature
    };

feature_count = numel(feature_functions);
assert(feature_count == 51, 'Exactly 51 feature functions are required.');
feature_names = cellfun(@func2str, feature_functions, ...
    'UniformOutput', false);
feature_names = regexprep(feature_names, '^feature_[0-9]{3}_', '');

%% Calculate features and outcomes over sliding windows
analysis_start_s = max(min(Gt), min(Pt));
analysis_end_s = min(max(Gt), max(Pt));
if analysis_end_s - analysis_start_s < window_size_s
    error('The synchronized recordings are shorter than one window.');
end

window_start_s = (analysis_start_s:window_step_s: ...
    (analysis_end_s - window_size_s))';
window_end_s = window_start_s + window_size_s;
window_count = numel(window_start_s);

feature_values = NaN(window_count, feature_count);
dsj = NaN(window_count, 1);
mean_force_n = NaN(window_count, 1);
gaze_sample_count = zeros(window_count, 1);
robot_sample_count = zeros(window_count, 1);

for window_idx = 1:window_count
    is_last_window = window_idx == window_count;
    if is_last_window
        gaze_window = Gt >= window_start_s(window_idx) & ...
            Gt <= window_end_s(window_idx);
        robot_window = Pt >= window_start_s(window_idx) & ...
            Pt <= window_end_s(window_idx);
    else
        gaze_window = Gt >= window_start_s(window_idx) & ...
            Gt < window_end_s(window_idx);
        robot_window = Pt >= window_start_s(window_idx) & ...
            Pt < window_end_s(window_idx);
    end

    gaze_sample_count(window_idx) = nnz(gaze_window);
    robot_sample_count(window_idx) = nnz(robot_window);
    if gaze_sample_count(window_idx) < minimum_gaze_samples || ...
            robot_sample_count(window_idx) < minimum_robot_samples
        continue;
    end

    window_Gx = Gx(gaze_window);
    window_Gy = Gy(gaze_window);
    window_Gt = Gt(gaze_window);
    window_Px = Px(robot_window);
    window_Py = Py(robot_window);
    window_Pt = Pt(robot_window);
    window_Vx = Vx(robot_window);
    window_Vy = Vy(robot_window);
    window_Fx = Fx(robot_window);
    window_Fy = Fy(robot_window);
    window_Wmag = Wmag(robot_window);
    window_Vcx = Vcx(robot_window);
    window_Vcy = Vcy(robot_window);

    for feature_idx = 1:feature_count
        feature_values(window_idx, feature_idx) = ...
            feature_functions{feature_idx}( ...
            window_Gx, window_Gy, window_Gt, ...
            window_Px, window_Py, window_Pt, ...
            window_Vx, window_Vy, window_Fx, window_Fy, ...
            window_Wmag, window_Vcx, window_Vcy);
    end

    dsj(window_idx) = calculate_planar_dsj( ...
        end_effector_x_m(robot_window), end_effector_y_m(robot_window), ...
        Pt(robot_window));
    mean_force_n(window_idx) = calculate_mean_planar_force( ...
        window_Fx, window_Fy);
end

feature_table = array2table(feature_values, ...
    'VariableNames', feature_names);
window_metadata = table( ...
    window_start_s, window_end_s, gaze_sample_count, robot_sample_count);
outcome_table = table(dsj, mean_force_n, ...
    'VariableNames', {'DSJ', 'mean_force_N'});
window_results = [window_metadata, feature_table, outcome_table];

%% Correlate every feature with DSJ and mean force
correlation_with_dsj = NaN(feature_count, 1);
correlation_with_mean_force = NaN(feature_count, 1);
for feature_idx = 1:feature_count
    correlation_with_dsj(feature_idx) = pearson_correlation( ...
        feature_values(:, feature_idx), dsj);
    correlation_with_mean_force(feature_idx) = pearson_correlation( ...
        feature_values(:, feature_idx), mean_force_n);
end

correlation_results = table( ...
    string(feature_names(:)), ...
    correlation_with_dsj, ...
    abs(correlation_with_dsj), ...
    correlation_with_mean_force, ...
    abs(correlation_with_mean_force), ...
    'VariableNames', { ...
    'feature', ...
    'correlation_with_DSJ', ...
    'absolute_correlation_with_DSJ', ...
    'correlation_with_mean_force', ...
    'absolute_correlation_with_mean_force'});

dsj_sort_values = abs(correlation_with_dsj);
dsj_sort_values(~isfinite(dsj_sort_values)) = -Inf;
[~, dsj_sort_order] = sort(dsj_sort_values, 'descend');

mean_force_sort_values = abs(correlation_with_mean_force);
mean_force_sort_values(~isfinite(mean_force_sort_values)) = -Inf;
[~, mean_force_sort_order] = sort(mean_force_sort_values, 'descend');

rank = (1:feature_count)';
correlation_ranked_by_dsj = [ ...
    table(rank), correlation_results(dsj_sort_order, :)];
correlation_ranked_by_mean_force = [ ...
    table(rank), correlation_results(mean_force_sort_order, :)];

fprintf('Gaze file: %s\n', gaze_data_file);
fprintf('Robot file: %s\n', robot_data_file);
fprintf('Window size/step: %.3f s / %.3f s\n', ...
    window_size_s, window_step_s);
fprintf('Analyzed windows: %d\n', window_count);
fprintf('\nFeatures ranked by absolute correlation with DSJ:\n');
disp(correlation_ranked_by_dsj);
fprintf('\nFeatures ranked by absolute correlation with mean force ');
fprintf('(damping-confounded -- see file header):\n');
disp(correlation_ranked_by_mean_force);

%% Plot normalized gaze-ee distance, DSJ, and force
window_center_s = (window_start_s + window_end_s) / 2.0;
gaze_ee_distance = window_results.mean_gaze_ee_distance;

gaze_ee_distance_normalized = normalize_finite_to_unit_interval( ...
    gaze_ee_distance);
dsj_normalized = normalize_finite_to_unit_interval(dsj);
mean_force_normalized = normalize_finite_to_unit_interval(mean_force_n);

figure('Name', 'Normalized gaze-EE distance, DSJ, and force', ...
    'Color', 'w');
hold on;
plot(window_center_s, gaze_ee_distance_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Mean gaze-EE distance');
plot(window_center_s, dsj_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'DSJ');
plot(window_center_s, mean_force_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Mean planar force');
grid on;
xlabel('Window center time [s]');
ylabel('Min-max normalized value [0, 1]');
title('Mean gaze-EE distance compared with DSJ and mean force');
legend('Location', 'best');
ylim([0, 1]);

%% Feature functions 001-051
% Every feature has the required (Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy,
% Wmag, Vcx, Vcy) interface and returns one scalar for one sliding window.

% --- 001-030: gaze-vs-robot position/speed relationship (adapted from
% gaze_feature_extraction.m's own 30 gaze-relative features, now using
% the calibrated frame instead of the normalized display) ---

function value = feature_001_mean_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_001_MEAN_GAZE_EE_DISTANCE Mean gaze-to-hand distance [m].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_mean(distance);
end


function value = feature_002_std_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_002_STD_GAZE_EE_DISTANCE Gaze-to-hand distance deviation.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_std(distance);
end


function value = feature_003_median_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_003_MEDIAN_GAZE_EE_DISTANCE Median gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_median(distance);
end


function value = feature_004_max_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_004_MAX_GAZE_EE_DISTANCE Maximum gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_maximum(distance);
end


function value = feature_005_rms_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_005_RMS_GAZE_EE_DISTANCE RMS gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_root_mean_square(distance);
end


function value = feature_006_p90_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_006_P90_GAZE_EE_DISTANCE 90th percentile gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = percentile_value(distance, 90.0);
end


function value = feature_007_min_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_007_MIN_GAZE_EE_DISTANCE Minimum gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_minimum(distance);
end


function value = feature_008_final_gaze_ee_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_008_FINAL_GAZE_EE_DISTANCE Last synchronized gaze-to-hand distance.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    if isempty(distance)
        value = NaN;
    else
        value = distance(end);
    end
end


function value = feature_009_gaze_ee_distance_slope( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_009_GAZE_EE_DISTANCE_SLOPE Linear distance trend [m/s].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = linear_slope(synced.time_s, distance);
end


function value = feature_010_mean_relative_x( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_010_MEAN_RELATIVE_X Mean signed gaze-minus-hand x offset [m].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(synced.gaze_x - synced.robot_x);
end


function value = feature_011_mean_relative_y( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_011_MEAN_RELATIVE_Y Mean signed gaze-minus-hand y offset [m].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(synced.gaze_y - synced.robot_y);
end


function value = feature_012_std_relative_x( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_012_STD_RELATIVE_X Standard deviation of gaze-minus-hand x.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_std(synced.gaze_x - synced.robot_x);
end


function value = feature_013_std_relative_y( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_013_STD_RELATIVE_Y Standard deviation of gaze-minus-hand y.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_std(synced.gaze_y - synced.robot_y);
end


function value = feature_014_relative_path_length( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_014_RELATIVE_PATH_LENGTH Path length of the gaze-minus-hand
% offset trajectory [m].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = planar_path_length( ...
        synced.gaze_x - synced.robot_x, synced.gaze_y - synced.robot_y);
end


function value = feature_015_mean_relative_speed( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_015_MEAN_RELATIVE_SPEED Mean speed of the gaze-minus-hand offset.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    relative_motion = calculate_robot_motion( ...
        gradient(synced.gaze_x - synced.robot_x, synced.time_s), ...
        gradient(synced.gaze_y - synced.robot_y, synced.time_s), ...
        synced.time_s);
    value = safe_mean(relative_motion.speed);
end


function value = feature_016_std_relative_speed( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_016_STD_RELATIVE_SPEED Relative-speed standard deviation.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    relative_motion = calculate_robot_motion( ...
        gradient(synced.gaze_x - synced.robot_x, synced.time_s), ...
        gradient(synced.gaze_y - synced.robot_y, synced.time_s), ...
        synced.time_s);
    value = safe_std(relative_motion.speed);
end


function value = feature_017_fraction_within_0p01( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_017_FRACTION_WITHIN_0P01 Fraction of time gaze is within 0.01 m
% of the hand.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_mean(double(distance <= 0.01));
end


function value = feature_018_fraction_within_0p02( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_018_FRACTION_WITHIN_0P02 Fraction of time gaze is within 0.02 m
% of the hand.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_mean(double(distance <= 0.02));
end


function value = feature_019_fraction_within_0p05( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_019_FRACTION_WITHIN_0P05 Fraction of time gaze is within 0.05 m
% of the hand.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = safe_mean(double(distance <= 0.05));
end


function value = feature_020_gaze_ee_x_correlation( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_020_GAZE_EE_X_CORRELATION Synchronized gaze/hand x correlation.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = pearson_correlation(synced.gaze_x, synced.robot_x);
end


function value = feature_021_gaze_ee_y_correlation( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_021_GAZE_EE_Y_CORRELATION Synchronized gaze/hand y correlation.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = pearson_correlation(synced.gaze_y, synced.robot_y);
end


function value = feature_022_gaze_ee_speed_correlation( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_022_GAZE_EE_SPEED_CORRELATION Gaze/hand speed correlation.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = pearson_correlation(gaze_motion.speed, synced.speed);
end


function value = feature_023_mean_velocity_cosine_similarity( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_023_MEAN_VELOCITY_COSINE_SIMILARITY Mean motion alignment [-1,1]
% between gaze velocity and hand velocity.
    cosine_similarity = gaze_ee_velocity_cosine( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(cosine_similarity);
end


function value = feature_024_aligned_velocity_fraction( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_024_ALIGNED_VELOCITY_FRACTION Fraction with positive alignment.
    cosine_similarity = gaze_ee_velocity_cosine( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(double(cosine_similarity > 0));
end


function value = feature_025_gaze_to_ee_speed_ratio( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_025_GAZE_TO_EE_SPEED_RATIO Mean gaze speed / mean hand speed.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = safe_ratio( ...
        safe_mean(gaze_motion.speed), safe_mean(synced.speed));
end


function value = feature_026_max_x_cross_correlation( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_026_MAX_X_CROSS_CORRELATION Best x correlation within +/-0.25 s.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    [value, ~] = best_lagged_correlation( ...
        synced.gaze_x, synced.robot_x, synced.time_s, 0.25);
end


function value = feature_027_x_lead_at_max_correlation_s( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_027_X_LEAD_AT_MAX_CORRELATION_S Positive means gaze leads the
% hand [s].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    [~, value] = best_lagged_correlation( ...
        synced.gaze_x, synced.robot_x, synced.time_s, 0.25);
end


function value = feature_028_max_y_cross_correlation( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_028_MAX_Y_CROSS_CORRELATION Best y correlation within +/-0.25 s.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    [value, ~] = best_lagged_correlation( ...
        synced.gaze_y, synced.robot_y, synced.time_s, 0.25);
end


function value = feature_029_y_lead_at_max_correlation_s( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_029_Y_LEAD_AT_MAX_CORRELATION_S Positive means gaze leads the
% hand [s].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    [~, value] = best_lagged_correlation( ...
        synced.gaze_y, synced.robot_y, synced.time_s, 0.25);
end


function value = feature_030_mean_gaze_lead_along_ee_motion( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_030_MEAN_GAZE_LEAD_ALONG_EE_MOTION Signed along-path gaze lead.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    moving = synced.speed > 1e-9;
    along_path_lead = NaN(size(synced.speed));
    along_path_lead(moving) = ( ...
        (synced.gaze_x(moving) - synced.robot_x(moving)) .* ...
        synced.velocity_x(moving) + ...
        (synced.gaze_y(moving) - synced.robot_y(moving)) .* ...
        synced.velocity_y(moving)) ./ synced.speed(moving);
    value = safe_mean(along_path_lead);
end


% --- 031-036: gaze vs. robot path curvature -- does gaze behave
% differently when the robot's path is tightly curved vs. nearly
% straight ---

function value = feature_031_correlation_gaze_ee_distance_curvature( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_031_CORRELATION_GAZE_EE_DISTANCE_CURVATURE Correlation between
% gaze-to-hand distance and |curvature|.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = pearson_correlation(distance, abs(synced.curvature));
end


function value = feature_032_correlation_gaze_speed_absolute_curvature( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_032_CORRELATION_GAZE_SPEED_ABSOLUTE_CURVATURE Correlation
% between gaze speed and |curvature|.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = pearson_correlation(gaze_motion.speed, abs(synced.curvature));
end


function value = feature_033_mean_curvature_when_gaze_close( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_033_MEAN_CURVATURE_WHEN_GAZE_CLOSE Mean |curvature| restricted
% to samples where gaze-to-hand distance is below this window's own
% median -- "how tight are the turns when the operator is looking at
% their hand."
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = curvature_conditioned_on_distance( ...
        distance, synced.curvature, true);
end


function value = feature_034_mean_curvature_when_gaze_far( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_034_MEAN_CURVATURE_WHEN_GAZE_FAR Mean |curvature| restricted to
% samples where gaze-to-hand distance is above this window's own median.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = curvature_conditioned_on_distance( ...
        distance, synced.curvature, false);
end


function value = feature_035_fraction_gaze_close_during_high_curvature( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_035_FRACTION_GAZE_CLOSE_DURING_HIGH_CURVATURE Of the samples in
% the tighter (above-median |curvature|) half of this window, the
% fraction where gaze was also within 0.02 m of the hand.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = fraction_close_during_high_reference( ...
        distance, abs(synced.curvature), 0.02);
end


function value = feature_036_correlation_gaze_offset_angle_curvature( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_036_CORRELATION_GAZE_OFFSET_ANGLE_CURVATURE Correlation between
% signed curvature and the signed angle of the (gaze - hand) offset
% relative to hand velocity -- does the operator's gaze swing to one
% side more as the path turns more sharply that way.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    offset_angle_deg = signed_angle_series( ...
        synced.velocity_x, synced.velocity_y, ...
        synced.gaze_x - synced.robot_x, synced.gaze_y - synced.robot_y);
    value = pearson_correlation(synced.curvature, offset_angle_deg);
end


% --- 037-042: gaze vs. robot force. DAMPING-CONFOUNDED -- see file
% header. ---

function value = feature_037_angle_gaze_offset_to_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_037_ANGLE_GAZE_OFFSET_TO_FORCE Mean signed angle between the
% (gaze - hand) offset and the applied force direction [deg]. Direct,
% path-free port of extract_pilot_features_fixed_adm_2d.m's
% angle_gaze_f_deg. Only computed where force exceeds a 0.5 N direction-
% trust floor, matching that script's minimum_force_for_direction_n.
    MINIMUM_FORCE_FOR_DIRECTION_N = 0.5;
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    offset_x = synced.gaze_x - synced.robot_x;
    offset_y = synced.gaze_y - synced.robot_y;
    angle_deg = signed_angle_series( ...
        synced.force_x, synced.force_y, offset_x, offset_y);
    angle_deg(synced.force_magnitude < MINIMUM_FORCE_FOR_DIRECTION_N) = NaN;
    value = safe_mean(angle_deg);
end


function value = feature_038_correlation_gaze_ee_distance_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_038_CORRELATION_GAZE_EE_DISTANCE_FORCE Correlation between
% gaze-to-hand distance and force magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = pearson_correlation(distance, synced.force_magnitude);
end


function value = feature_039_correlation_gaze_speed_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_039_CORRELATION_GAZE_SPEED_FORCE Correlation between gaze speed
% and force magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = pearson_correlation(gaze_motion.speed, synced.force_magnitude);
end


function value = feature_040_mean_gaze_ee_distance_high_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_040_MEAN_GAZE_EE_DISTANCE_HIGH_FORCE Mean gaze-to-hand distance
% restricted to samples above this window's own median force magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = value_conditioned_on_reference( ...
        distance, synced.force_magnitude, false);
end


function value = feature_041_mean_gaze_ee_distance_low_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_041_MEAN_GAZE_EE_DISTANCE_LOW_FORCE Mean gaze-to-hand distance
% restricted to samples below this window's own median force magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = value_conditioned_on_reference( ...
        distance, synced.force_magnitude, true);
end


function value = feature_042_fraction_gaze_close_during_high_force( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_042_FRACTION_GAZE_CLOSE_DURING_HIGH_FORCE Of the samples above
% this window's own median force magnitude, the fraction where gaze was
% also within 0.02 m of the hand.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = fraction_close_during_high_reference( ...
        distance, synced.force_magnitude, 0.02);
end


% --- 043-044: gaze vs. robot angular speed (wrist rotation) ---

function value = feature_043_correlation_gaze_speed_angular_speed( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_043_CORRELATION_GAZE_SPEED_ANGULAR_SPEED Correlation between
% gaze speed and the wrist's angular speed magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = pearson_correlation(gaze_motion.speed, synced.angular_speed);
end


function value = feature_044_correlation_gaze_ee_distance_angular_speed( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_044_CORRELATION_GAZE_EE_DISTANCE_ANGULAR_SPEED Correlation
% between gaze-to-hand distance and the wrist's angular speed magnitude.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = pearson_correlation(distance, synced.angular_speed);
end


% --- 045-046: gaze vs. robot command-tracking error. DAMPING-CONFOUNDED
% -- see file header. ---

function value = feature_045_correlation_gaze_speed_tracking_error( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_045_CORRELATION_GAZE_SPEED_TRACKING_ERROR Correlation between
% gaze speed and |commanded - actual velocity|.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);
    value = pearson_correlation(gaze_motion.speed, synced.tracking_error);
end


function value = feature_046_correlation_gaze_ee_distance_tracking_error( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_046_CORRELATION_GAZE_EE_DISTANCE_TRACKING_ERROR Correlation
% between gaze-to-hand distance and |commanded - actual velocity|.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    distance = hypot(synced.gaze_x - synced.robot_x, ...
        synced.gaze_y - synced.robot_y);
    value = pearson_correlation(distance, synced.tracking_error);
end


% --- 047-051: gaze lookahead relative to robot motion direction --
% path-free ports of extract_pilot_features_fixed_adm_2d.m's
% angle_gaze_v_deg / d_lookahead_over_v_s / d_lookahead_times_curvature /
% fraction_gaze_ahead, using the straight-line projection of
% (gaze - hand) onto the velocity direction instead of arc-length along
% a reference path (this script has no shape .obj to project onto). ---

function value = feature_047_angle_gaze_offset_to_velocity( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_047_ANGLE_GAZE_OFFSET_TO_VELOCITY Mean signed angle between the
% (gaze - hand) offset and the hand's velocity direction [deg]. Only
% computed where speed exceeds this project's usual 0.005 m/s at-rest
% floor, matching MINIMUM_SPEED_M_S in calculate_robot_motion().
    MINIMUM_SPEED_M_S = 0.005;
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    offset_x = synced.gaze_x - synced.robot_x;
    offset_y = synced.gaze_y - synced.robot_y;
    angle_deg = signed_angle_series( ...
        synced.velocity_x, synced.velocity_y, offset_x, offset_y);
    angle_deg(synced.speed < MINIMUM_SPEED_M_S) = NaN;
    value = safe_mean(angle_deg);
end


function value = feature_048_mean_lookahead_distance( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_048_MEAN_LOOKAHEAD_DISTANCE Mean signed projection of
% (gaze - hand) onto the hand's own velocity direction [m] -- positive
% means gaze sits ahead of the hand along its current heading, negative
% means behind. The straight-line analogue of extract_pilot_features_
% fixed_adm_2d.m's delta_s_m (which projects along a reference path
% instead).
    lookahead = lookahead_distance_series( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(lookahead);
end


function value = feature_049_fraction_gaze_ahead( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_049_FRACTION_GAZE_AHEAD Fraction of (valid) samples with a
% positive lookahead distance.
    lookahead = lookahead_distance_series( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    valid_lookahead = lookahead(isfinite(lookahead));
    if isempty(valid_lookahead)
        value = NaN;
    else
        value = mean(valid_lookahead > 0);
    end
end


function value = feature_050_lookahead_distance_over_speed( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_050_LOOKAHEAD_DISTANCE_OVER_SPEED Mean lookahead distance
% divided by hand speed [s] -- a lookahead TIME, path-free analogue of
% extract_pilot_features_fixed_adm_2d.m's d_lookahead_over_v_s.
    MINIMUM_SPEED_M_S = 0.005;
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    lookahead = lookahead_distance_series( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    ratio = NaN(size(lookahead));
    moving = synced.speed >= MINIMUM_SPEED_M_S;
    ratio(moving) = lookahead(moving) ./ synced.speed(moving);
    value = safe_mean(ratio);
end


function value = feature_051_lookahead_times_curvature( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_051_LOOKAHEAD_TIMES_CURVATURE Mean lookahead distance times
% signed curvature -- path-free analogue of extract_pilot_features_
% fixed_adm_2d.m's d_lookahead_times_curvature.
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    lookahead = lookahead_distance_series( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    value = safe_mean(lookahead .* synced.curvature);
end

%% Joint-specific helpers (new -- not in the other two scripts)

function synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%SYNCHRONIZE_ROBOT_ONTO_GAZE Interpolate every robot-side raw and
% derived signal onto gaze timestamps, restricted to the time overlap
% of the two clocks -- generalizes gaze_feature_extraction.m's
% synchronize_gaze_and_end_effector() (which only carries position) to
% also carry speed, curvature, force, angular speed, and tracking
% error. Speed/curvature are differentiated from Vx/Vy at the robot's
% OWN native (much higher) sample rate first (via calculate_robot_
% motion), and only interpolated onto the sparser gaze clock afterward
% -- never the other way around, so they stay as well-resolved as
% robot_feature_extraction.m's own versions of the same quantities.

    [gx, gy, gaze_time_s] = clean_planar_time_series(Gx, Gy, Gt);

    robot_index = clean_time_indices(Pt);
    if numel(robot_index) < 2 || isempty(gaze_time_s)
        synced = empty_synchronized_bundle();
        return;
    end
    robot_time_s = Pt(robot_index);

    motion = calculate_robot_motion(Vx, Vy, Pt);
    % motion's own internal cleaning (inside calculate_robot_motion, via
    % clean_planar_time_series on Vx, Vy, Pt) applies the IDENTICAL
    % sort/dedupe rule as clean_time_indices(Pt) above, so motion.
    % time_s and robot_time_s coincide sample-for-sample -- interpolate
    % speed/curvature against motion.time_s directly rather than
    % re-deriving that alignment.

    in_overlap = gaze_time_s >= motion.time_s(1) & ...
        gaze_time_s <= motion.time_s(end);
    gt = gaze_time_s(in_overlap);
    gx = gx(in_overlap);
    gy = gy(in_overlap);

    if isempty(gt)
        synced = empty_synchronized_bundle();
        return;
    end

    Px_clean = Px(robot_index);
    Py_clean = Py(robot_index);
    Fx_clean = Fx(robot_index);
    Fy_clean = Fy(robot_index);
    Wmag_clean = Wmag(robot_index);
    Vcx_clean = Vcx(robot_index);
    Vcy_clean = Vcy(robot_index);

    px_i = interp1(robot_time_s, Px_clean, gt, 'linear');
    py_i = interp1(robot_time_s, Py_clean, gt, 'linear');
    speed_i = interp1(motion.time_s, motion.speed, gt, 'linear');
    curvature_i = interp1(motion.time_s, motion.curvature, gt, 'linear');
    velocity_x_i = interp1(motion.time_s, motion.velocity_x, gt, 'linear');
    velocity_y_i = interp1(motion.time_s, motion.velocity_y, gt, 'linear');
    force_x_i = interp1(robot_time_s, Fx_clean, gt, 'linear');
    force_y_i = interp1(robot_time_s, Fy_clean, gt, 'linear');
    angular_speed_i = interp1(robot_time_s, Wmag_clean, gt, 'linear');
    commanded_x_i = interp1(robot_time_s, Vcx_clean, gt, 'linear');
    commanded_y_i = interp1(robot_time_s, Vcy_clean, gt, 'linear');

    valid = isfinite(gx) & isfinite(gy) & isfinite(px_i) & isfinite(py_i);

    synced.gaze_x = gx(valid);
    synced.gaze_y = gy(valid);
    synced.time_s = gt(valid);
    synced.robot_x = px_i(valid);
    synced.robot_y = py_i(valid);
    synced.speed = speed_i(valid);
    synced.curvature = curvature_i(valid);
    synced.velocity_x = velocity_x_i(valid);
    synced.velocity_y = velocity_y_i(valid);
    synced.force_x = force_x_i(valid);
    synced.force_y = force_y_i(valid);
    synced.force_magnitude = hypot(force_x_i(valid), force_y_i(valid));
    synced.angular_speed = angular_speed_i(valid);
    synced.tracking_error = hypot( ...
        commanded_x_i(valid) - velocity_x_i(valid), ...
        commanded_y_i(valid) - velocity_y_i(valid));
end


function synced = empty_synchronized_bundle()
%EMPTY_SYNCHRONIZED_BUNDLE All-empty synced struct, same fields as
% synchronize_robot_onto_gaze(), for its too-little-data early return.
    field_names = {'gaze_x', 'gaze_y', 'time_s', 'robot_x', 'robot_y', ...
        'speed', 'curvature', 'velocity_x', 'velocity_y', 'force_x', ...
        'force_y', 'force_magnitude', 'angular_speed', 'tracking_error'};
    for field_idx = 1:numel(field_names)
        synced.(field_names{field_idx}) = [];
    end
end


function final_index = clean_time_indices(raw_time_s)
%CLEAN_TIME_INDICES Index bookkeeping for turning a possibly-duplicate,
% out-of-order, non-finite-containing raw_time_s into a sorted,
% duplicate-free time series: raw_time_s(final_index) is sorted,
% duplicate-free, and finite. Apply the SAME final_index to any value
% array paired with raw_time_s (e.g. Fx(final_index)) to keep it
% aligned -- the multi-array generalization of clean_planar_time_
% series() below, which only cleans exactly two paired value arrays.
    valid_idx = find(isfinite(raw_time_s));
    [~, sort_order] = sort(raw_time_s(valid_idx));
    sorted_idx = valid_idx(sort_order);
    [~, unique_pos] = unique(raw_time_s(sorted_idx), 'stable');
    final_index = sorted_idx(unique_pos);
end


function value = curvature_conditioned_on_distance( ...
        distance, curvature, want_close)
%CURVATURE_CONDITIONED_ON_DISTANCE Mean |curvature| restricted to the
% half of the (paired) samples either below (want_close=true) or above
% (false) this window's own median gaze-to-hand distance.
    valid = isfinite(distance) & isfinite(curvature);
    distance = distance(valid);
    curvature = curvature(valid);
    if numel(distance) < 2
        value = NaN;
        return;
    end

    median_distance = median(distance);
    if want_close
        selected = distance <= median_distance;
    else
        selected = distance > median_distance;
    end

    if ~any(selected)
        value = NaN;
    else
        value = mean(abs(curvature(selected)));
    end
end


function value = value_conditioned_on_reference( ...
        values, reference, want_low_reference)
%VALUE_CONDITIONED_ON_REFERENCE Mean of values restricted to the half of
% the (paired) samples either below (want_low_reference=true) or above
% (false) this window's own median of reference.
    valid = isfinite(values) & isfinite(reference);
    values = values(valid);
    reference = reference(valid);
    if numel(values) < 2
        value = NaN;
        return;
    end

    median_reference = median(reference);
    if want_low_reference
        selected = reference <= median_reference;
    else
        selected = reference > median_reference;
    end

    if ~any(selected)
        value = NaN;
    else
        value = mean(values(selected));
    end
end


function value = fraction_close_during_high_reference( ...
        distance, reference, close_threshold_m)
%FRACTION_CLOSE_DURING_HIGH_REFERENCE Of the (paired) samples above this
% window's own median of reference, the fraction where distance was
% also <= close_threshold_m.
    valid = isfinite(distance) & isfinite(reference);
    distance = distance(valid);
    reference = reference(valid);
    if numel(distance) < 2
        value = NaN;
        return;
    end

    high_reference = reference > median(reference);
    if ~any(high_reference)
        value = NaN;
    else
        value = mean(distance(high_reference) <= close_threshold_m);
    end
end


function angle_deg = signed_angle_series( ...
        reference_x, reference_y, target_x, target_y)
%SIGNED_ANGLE_SERIES Per-sample signed angle from (reference_x,
% reference_y) to (target_x, target_y) [deg], via atan2(cross, dot) --
% the vectorized, per-sample analogue of extract_pilot_features_fixed_
% adm_2d.m's angle_between_vectors_deg(). NaN wherever either vector is
% (numerically) zero-length, where direction is undefined.
    reference_norm = hypot(reference_x, reference_y);
    target_norm = hypot(target_x, target_y);

    cross_value = reference_x .* target_y - reference_y .* target_x;
    dot_value = reference_x .* target_x + reference_y .* target_y;
    angle_deg = atan2d(cross_value, dot_value);
    angle_deg(reference_norm < 1e-9 | target_norm < 1e-9) = NaN;
end


function lookahead = lookahead_distance_series( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%LOOKAHEAD_DISTANCE_SERIES Signed projection of (gaze - hand) onto the
% hand's own (unit) velocity direction [m], per synchronized sample --
% positive means gaze sits ahead of the hand along its current heading.
% NaN wherever speed is below this project's usual 0.005 m/s at-rest
% floor (calculate_robot_motion()'s own MINIMUM_SPEED_M_S), where
% direction is undefined.
    MINIMUM_SPEED_M_S = 0.005;
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    offset_x = synced.gaze_x - synced.robot_x;
    offset_y = synced.gaze_y - synced.robot_y;

    lookahead = NaN(size(synced.speed));
    moving = synced.speed >= MINIMUM_SPEED_M_S;
    unit_velocity_x = synced.velocity_x(moving) ./ synced.speed(moving);
    unit_velocity_y = synced.velocity_y(moving) ./ synced.speed(moving);
    lookahead(moving) = offset_x(moving) .* unit_velocity_x + ...
        offset_y(moving) .* unit_velocity_y;
end


function cosine_similarity = gaze_ee_velocity_cosine( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%GAZE_EE_VELOCITY_COSINE Cosine similarity of gaze velocity and hand
% velocity, in [-1, 1].
    synced = synchronize_robot_onto_gaze( ...
        Gx, Gy, Gt, Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy);
    gaze_motion = calculate_robot_motion( ...
        synced.gaze_x, synced.gaze_y, synced.time_s);

    speed_product = gaze_motion.speed .* synced.speed;
    valid = isfinite(speed_product) & speed_product > 1e-12;
    cosine_similarity = ( ...
        gaze_motion.velocity_x(valid) .* synced.velocity_x(valid) + ...
        gaze_motion.velocity_y(valid) .* synced.velocity_y(valid)) ./ ...
        speed_product(valid);
    cosine_similarity = min(max(cosine_similarity, -1.0), 1.0);
end


function [best_correlation, best_lag_s] = best_lagged_correlation( ...
        gaze_values, ee_values, time_s, maximum_lag_s)
%BEST_LAGGED_CORRELATION Find best correlation over positive/negative
% lags. Copied verbatim from gaze_feature_extraction.m.
%
% A positive returned lag means an earlier gaze sample best matches a
% later end-effector sample; therefore, positive lag indicates gaze
% leading the hand.
    gaze_values = double(gaze_values(:));
    ee_values = double(ee_values(:));
    time_s = double(time_s(:));
    valid = isfinite(gaze_values) & isfinite(ee_values) & isfinite(time_s);
    gaze_values = gaze_values(valid);
    ee_values = ee_values(valid);
    time_s = time_s(valid);

    if numel(time_s) < 4
        best_correlation = NaN;
        best_lag_s = NaN;
        return;
    end

    nominal_interval_s = median(diff(time_s));
    if ~isfinite(nominal_interval_s) || nominal_interval_s <= 0
        best_correlation = NaN;
        best_lag_s = NaN;
        return;
    end

    maximum_lag_samples = min( ...
        floor(maximum_lag_s / nominal_interval_s), numel(time_s) - 3);
    candidate_lags = (-maximum_lag_samples:maximum_lag_samples)';
    candidate_correlations = NaN(size(candidate_lags));

    for lag_idx = 1:numel(candidate_lags)
        lag_samples = candidate_lags(lag_idx);
        if lag_samples >= 0
            gaze_segment = gaze_values(1:end-lag_samples);
            ee_segment = ee_values(1 + lag_samples:end);
        else
            shift = -lag_samples;
            gaze_segment = gaze_values(1 + shift:end);
            ee_segment = ee_values(1:end-shift);
        end
        candidate_correlations(lag_idx) = pearson_correlation( ...
            gaze_segment, ee_segment);
    end

    finite_candidates = find(isfinite(candidate_correlations));
    if isempty(finite_candidates)
        best_correlation = NaN;
        best_lag_s = NaN;
        return;
    end

    [best_correlation, local_best_idx] = max( ...
        candidate_correlations(finite_candidates));
    best_idx = finite_candidates(local_best_idx);
    best_lag_s = candidate_lags(best_idx) * nominal_interval_s;
end


function value = linear_slope(time_s, values)
%LINEAR_SLOPE Least-squares signal slope with respect to time. Copied
% verbatim from gaze_feature_extraction.m.
    time_s = double(time_s(:));
    values = double(values(:));
    valid = isfinite(time_s) & isfinite(values);
    time_s = time_s(valid);
    values = values(valid);
    if numel(time_s) < 2
        value = NaN;
        return;
    end

    centered_time_s = time_s - mean(time_s);
    denominator = sum(centered_time_s.^2);
    if denominator <= 0
        value = NaN;
    else
        value = sum(centered_time_s .* (values - mean(values))) / ...
            denominator;
    end
end


function motion = calculate_robot_motion(Vx, Vy, Pt)
%CALCULATE_ROBOT_MOTION Speed, acceleration, jerk, direction, and signed
% curvature from MEASURED velocity. Copied verbatim from robot_feature_
% extraction.m -- see that script's own copy for the full explanation.
    MINIMUM_SPEED_M_S = 0.005;

    [Vx, Vy, Pt] = clean_planar_time_series(Vx, Vy, Pt);
    sample_count = numel(Pt);

    motion.time_s = Pt;
    motion.velocity_x = Vx;
    motion.velocity_y = Vy;
    motion.speed = hypot(Vx, Vy);
    motion.acceleration = NaN(sample_count, 1);
    motion.jerk = NaN(sample_count, 1);
    motion.direction = NaN(sample_count, 1);
    motion.curvature = NaN(sample_count, 1);
    motion.curvature_dot = NaN(sample_count, 1);

    moving = motion.speed > 1e-12;
    motion.direction(moving) = atan2(Vy(moving), Vx(moving));

    if sample_count < 3
        return;
    end

    acceleration_x = gradient(Vx, Pt);
    acceleration_y = gradient(Vy, Pt);
    motion.acceleration = hypot(acceleration_x, acceleration_y);

    curvature = (Vx .* acceleration_y - Vy .* acceleration_x) ./ ...
        motion.speed .^ 3;
    curvature(motion.speed < MINIMUM_SPEED_M_S) = NaN;
    motion.curvature = curvature;
    motion.curvature_dot = gradient(curvature, Pt);

    if sample_count < 4
        return;
    end
    jerk_x = gradient(acceleration_x, Pt);
    jerk_y = gradient(acceleration_y, Pt);
    motion.jerk = hypot(jerk_x, jerk_y);
end


%% Shared helpers, copied verbatim from gaze_feature_extraction.m and/or
% robot_feature_extraction.m (generic utilities with no gaze- or robot-
% specific content).

function require_table_variables(data_table, required_names, table_label)
%REQUIRE_TABLE_VARIABLES Confirm that a loaded table has required columns.
    imported_names = data_table.Properties.VariableNames;
    missing_names = setdiff(required_names, imported_names);
    if ~isempty(missing_names)
        error('%s is missing columns: %s', ...
            table_label, strjoin(missing_names, ', '));
    end
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like numeric vectors stored in CSV cells.
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
    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display
% bounds.
    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end


function dsj = calculate_planar_dsj(x_m, y_m, time_s)
%CALCULATE_PLANAR_DSJ Dimensionless squared jerk over one window, from
% x_m/y_m (raw robot-base-frame position) -- matches this project's
% usual DSJ convention, same as robot_feature_extraction.m's copy.
    [x, y, t] = clean_planar_time_series(x_m, y_m, time_s);
    sample_count = numel(t);

    if sample_count < 4
        dsj = NaN;
        return;
    end

    velocity_x = gradient(x, t);
    velocity_y = gradient(y, t);
    acceleration_x = gradient(velocity_x, t);
    acceleration_y = gradient(velocity_y, t);
    jerk_x = gradient(acceleration_x, t);
    jerk_y = gradient(acceleration_y, t);
    squared_jerk = jerk_x.^2 + jerk_y.^2;

    duration_s = t(end) - t(1);
    path_length_m = planar_path_length(x, y);

    if ~isfinite(duration_s) || duration_s <= 0 || ...
            ~isfinite(path_length_m) || path_length_m <= 1e-12 || ...
            any(~isfinite(squared_jerk))
        dsj = NaN;
        return;
    end

    integrated_squared_jerk = trapz(t, squared_jerk);
    dsj = duration_s^5 * integrated_squared_jerk / path_length_m^2;
end


function mean_force_n = calculate_mean_planar_force(force_x_n, force_y_n)
%CALCULATE_MEAN_PLANAR_FORCE Mean filtered planar force magnitude [N].
    force_magnitude_n = hypot(force_x_n, force_y_n);
    mean_force_n = safe_mean(force_magnitude_n);
end


function values = finite_values(values)
%FINITE_VALUES Return finite values as a column vector.
    values = double(values(:));
    values = values(isfinite(values));
end


function value = safe_mean(values)
%SAFE_MEAN Arithmetic mean of finite values.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end


function value = safe_std(values)
%SAFE_STD Sample standard deviation of finite values.
    values = finite_values(values);
    if numel(values) < 2
        value = NaN;
    else
        value = std(values, 0);
    end
end


function value = safe_minimum(values)
%SAFE_MINIMUM Minimum finite value.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
    else
        value = min(values);
    end
end


function value = safe_maximum(values)
%SAFE_MAXIMUM Maximum finite value.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
    else
        value = max(values);
    end
end


function value = safe_median(values)
%SAFE_MEDIAN Median of finite values.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
    else
        value = median(values);
    end
end


function value = pearson_correlation(first_values, second_values)
%PEARSON_CORRELATION Toolbox-free Pearson correlation of finite pairs.
    first_values = double(first_values(:));
    second_values = double(second_values(:));
    valid = isfinite(first_values) & isfinite(second_values);
    first_values = first_values(valid);
    second_values = second_values(valid);

    if numel(first_values) < 3
        value = NaN;
        return;
    end

    first_centered = first_values - mean(first_values);
    second_centered = second_values - mean(second_values);
    denominator = sqrt( ...
        sum(first_centered.^2) * sum(second_centered.^2));
    if denominator <= 0
        value = NaN;
    else
        value = sum(first_centered .* second_centered) / denominator;
    end
end


function value = safe_root_mean_square(values)
%SAFE_ROOT_MEAN_SQUARE RMS of finite values.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
    else
        value = sqrt(mean(values.^2));
    end
end


function value = safe_ratio(numerator, denominator)
%SAFE_RATIO Return a finite ratio or NaN for a zero/invalid denominator.
    if ~isfinite(numerator) || ~isfinite(denominator) || ...
            abs(denominator) <= 1e-12
        value = NaN;
    else
        value = numerator / denominator;
    end
end


function value = percentile_value(values, percentile)
%PERCENTILE_VALUE Linearly interpolated percentile without a toolbox.
    values = sort(finite_values(values));
    if isempty(values)
        value = NaN;
        return;
    end

    percentile = min(max(percentile, 0.0), 100.0);
    rank = 1.0 + (numel(values) - 1.0) * percentile / 100.0;
    lower_idx = floor(rank);
    upper_idx = ceil(rank);
    interpolation_weight = rank - lower_idx;
    value = values(lower_idx) * (1.0 - interpolation_weight) + ...
        values(upper_idx) * interpolation_weight;
end


function [x, y, time_s] = clean_planar_time_series(x, y, time_s)
%CLEAN_PLANAR_TIME_SERIES Remove invalid/duplicate samples and sort by time.
    x = double(x(:));
    y = double(y(:));
    time_s = double(time_s(:));
    valid = isfinite(x) & isfinite(y) & isfinite(time_s);
    x = x(valid);
    y = y(valid);
    time_s = time_s(valid);

    [time_s, sort_order] = sort(time_s);
    x = x(sort_order);
    y = y(sort_order);
    [time_s, unique_indices] = unique(time_s, 'stable');
    x = x(unique_indices);
    y = y(unique_indices);
end


function value = planar_path_length(x, y)
%PLANAR_PATH_LENGTH Sum Euclidean distance between consecutive samples.
    x = double(x(:));
    y = double(y(:));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 2
        value = NaN;
    else
        value = sum(hypot(diff(x), diff(y)));
    end
end


function normalized_values = normalize_finite_to_unit_interval(values)
%NORMALIZE_FINITE_TO_UNIT_INTERVAL Min-max normalize finite values to [0,1].
%
% NaN and Inf remain NaN. A constant finite signal is plotted at zero.
    values = double(values(:));
    normalized_values = NaN(size(values));
    finite_rows_mask = isfinite(values);
    finite_values_in_signal = values(finite_rows_mask);
    if isempty(finite_values_in_signal)
        return;
    end

    minimum_value = min(finite_values_in_signal);
    maximum_value = max(finite_values_in_signal);
    value_range = maximum_value - minimum_value;
    if value_range <= 0
        normalized_values(finite_rows_mask) = 0.0;
    else
        normalized_values(finite_rows_mask) = ...
            (finite_values_in_signal - minimum_value) / value_range;
    end
end
