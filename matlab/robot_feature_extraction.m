%% AWESOME_SCRIPT: Extract windowed robot-only features and correlations.
%
% Robot-side counterpart to gaze_feature_extraction.m, same structure and
% (roughly) same scale -- built specifically so a later robot-only vs.
% gaze-only vs. joint classifier comparison isn't unfair to the robot side
% by only giving it two features (DSJ, mean speed) while gaze gets 102.
% Everything here comes from the UR5e CSV alone; no eye tracker involved.
%
% Inputs:
%   UR5e CSV with position [m], measured/commanded velocity [m/s],
%   angular velocity [rad/s], filtered planar force [N], time [s].
%
% Outputs retained in the MATLAB workspace:
%   window_results      - all features, DSJ, and mean force per window.
%   correlation_results - Pearson correlation of every feature with DSJ and
%                         mean planar force.
%   correlation_ranked_by_dsj and correlation_ranked_by_mean_force - the
%                         same correlations ranked by absolute magnitude.
%
% Shared feature-function interface:
%   value = feature_name(Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%   Px, Py   - position in the CoppeliaSim/robot-base frame [m].
%   Pt       - position/velocity/force time [s].
%   Vx, Vy   - MEASURED TCP velocity [m/s] (not re-differentiated from
%              position -- one derivative closer to what was actually
%              measured, matching this project's other derivative chains).
%   Fx, Fy   - filtered planar force [N].
%   Wmag     - angular speed magnitude sqrt(wx^2+wy^2+wz^2) [rad/s].
%   Vcx, Vcy - COMMANDED planar velocity [m/s] (the admittance
%              controller's own setpoint, before the robot's dynamics
%              caught up to it).
%
% A caveat worth knowing before trusting any ranking this script produces:
%   Every FORCE-derived feature (see the "Force" section below) and every
%   COMMAND-TRACKING feature (see "Command tracking") is partly a function
%   of the CURRENT admittance damping (B_x_Ns_m/B_y_Ns_m), not just of the
%   person's behaviour -- higher damping needs more force and lags the
%   command more for the same intended motion. If damping varies across
%   the trials you pool together (as it does in this project's fine/gross
%   dataset: d10/d100/d200), these two feature groups partly just encode
%   which damping condition a window came from. They are still computed
%   here (this script tries to extract "whatever can be gotten" from the
%   raw data, per its own brief), but treat their ranking with that in
%   mind, and consider excluding them downstream the same way mean force
%   was already excluded from the fine/gross classifier's feature set.
%
% Important assumptions:
%   Position/curvature features use coppeliasim_x_m/y_m (the same frame
%   gaze_feature_extraction.m's Ex/Ey use, and detect_fixations_coppelia_
%   xy's calibrated gaze mapping targets), so a robot-position feature and
%   a gaze-position feature mean "the same place" if ever compared
%   directly. DSJ specifically uses x_m/y_m (raw robot-base-frame
%   position) instead, matching every other DSJ chain in this project, so
%   DSJ values here stay comparable with DSJ reported elsewhere -- this
%   split mirrors gaze_feature_extraction.m's own end_effector_x_m (for
%   DSJ) vs. Ex (for its 102 features) distinction exactly.

clear;
close all
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
robot_data_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'ur5e_fixed_adm_2d_labeled_fine_gross_shape_d100_s2.csv');

window_size_s = 3.5;  % TODO window duration [s], recommended 0.25 to 2
window_step_s = 0.02; % TODO step [s]; default is 94% overlap, matching
                       % gaze_feature_extraction.m -- see that script's
                       % own header for why this is fine for a
                       % correlation study but not for classifier
                       % training data (build_feature_dataset.py-style
                       % scripts use a much coarser, non-overlapping step).
minimum_robot_samples = 10;  % TODO valid samples/window, recommended >= 10

% Two thresholds used only inside specific feature functions
% (calculate_robot_motion's curvature gate, fit_power_law's minimum
% sample count) are NOT listed here even though they're logically
% "settings" -- local functions in this script can't see this
% workspace, and the shared feature-function interface is fixed at 10
% arguments the same way gaze_feature_extraction.m's is fixed at 6, so
% there's no clean way to thread them through. They're hardcoded as
% named constants at the top of those specific functions instead,
% exactly how gaze_feature_extraction.m hardcodes its own 0.10/0.50
% speed thresholds directly inside feature_051/052 rather than passing
% them in -- edit them there if you need to change them.

if window_size_s <= 0 || window_step_s <= 0
    error('window_size_s and window_step_s must be positive.');
end

%% Load raw recording
robot_table = readtable(robot_data_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_robot_variables = {
    'time_s'
    'x_m'
    'y_m'
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
    };
require_table_variables(robot_table, required_robot_variables, 'UR5e CSV');

%% Decode position, velocity, force, angular speed, and commanded velocity
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

valid_rows = isfinite(Pt_all) & isfinite(Px_all) & isfinite(Py_all) & ...
    isfinite(end_effector_x_m_all) & isfinite(end_effector_y_m_all) & ...
    isfinite(Vx_all) & isfinite(Vy_all);

Pt = Pt_all(valid_rows);
end_effector_x_m = end_effector_x_m_all(valid_rows);
end_effector_y_m = end_effector_y_m_all(valid_rows);
Px = Px_all(valid_rows);
Py = Py_all(valid_rows);
Vx = Vx_all(valid_rows);
Vy = Vy_all(valid_rows);
Fx = Fx_all(valid_rows);
Fy = Fy_all(valid_rows);
Vcx = Vcx_all(valid_rows);
Vcy = Vcy_all(valid_rows);
Wmag = Wmag_all(valid_rows);

%% Define feature functions
feature_functions = {
    @feature_001_mean_position_x
    @feature_002_mean_position_y
    @feature_003_std_position_x
    @feature_004_std_position_y
    @feature_005_variance_position_x
    @feature_006_variance_position_y
    @feature_007_minimum_position_x
    @feature_008_maximum_position_x
    @feature_009_range_position_x
    @feature_010_minimum_position_y
    @feature_011_maximum_position_y
    @feature_012_range_position_y
    @feature_013_median_position_x
    @feature_014_median_position_y
    @feature_015_iqr_position_x
    @feature_016_iqr_position_y
    @feature_017_mad_position_x
    @feature_018_mad_position_y
    @feature_019_skewness_position_x
    @feature_020_skewness_position_y
    @feature_021_kurtosis_position_x
    @feature_022_kurtosis_position_y
    @feature_023_covariance_position_xy
    @feature_024_correlation_position_xy
    @feature_025_position_dispersion_rms
    @feature_026_position_bcea_68
    @feature_027_path_length
    @feature_028_net_displacement
    @feature_029_path_efficiency
    @feature_030_mean_speed
    @feature_031_std_speed
    @feature_032_max_speed
    @feature_033_median_speed
    @feature_034_p90_speed
    @feature_035_rms_speed
    @feature_036_mean_acceleration
    @feature_037_std_acceleration
    @feature_038_max_acceleration
    @feature_039_rms_acceleration
    @feature_040_mean_jerk
    @feature_041_std_jerk
    @feature_042_max_jerk
    @feature_043_rms_jerk
    @feature_044_mean_velocity_x
    @feature_045_mean_velocity_y
    @feature_046_std_velocity_x
    @feature_047_std_velocity_y
    @feature_048_mean_direction
    @feature_049_direction_concentration
    @feature_050_mean_absolute_direction_change
    @feature_051_low_speed_fraction
    @feature_052_high_speed_fraction
    @feature_053_high_speed_event_count
    @feature_054_low_speed_event_count
    @feature_055_missing_time_fraction
    @feature_056_gap_count
    @feature_057_sample_rate
    @feature_058_position_x_shannon_entropy
    @feature_059_position_y_shannon_entropy
    @feature_060_stationary_shannon_entropy
    @feature_061_transitional_shannon_entropy
    @feature_062_position_x_mean_crossing_count
    @feature_063_position_y_mean_crossing_count
    @feature_064_position_x_lag1_autocorrelation
    @feature_065_position_y_lag1_autocorrelation
    @feature_066_speed_lag1_autocorrelation
    @feature_067_mean_curvature
    @feature_068_std_curvature
    @feature_069_median_absolute_curvature
    @feature_070_max_absolute_curvature
    @feature_071_mean_curvature_dot
    @feature_072_std_curvature_dot
    @feature_073_curvature_sign_reversal_rate
    @feature_074_power_law_r2
    @feature_075_power_law_exponent
    @feature_076_mean_angular_speed
    @feature_077_std_angular_speed
    @feature_078_max_angular_speed
    @feature_079_rms_angular_speed
    @feature_080_angular_speed_lag1_autocorrelation
    @feature_081_stationary_shannon_entropy_40x40
    @feature_082_transitional_shannon_entropy_40x40
    @feature_083_mean_force_magnitude
    @feature_084_std_force_magnitude
    @feature_085_max_force_magnitude
    @feature_086_median_force_magnitude
    @feature_087_rms_force_magnitude
    @feature_088_mean_force_x
    @feature_089_mean_force_y
    @feature_090_std_force_x
    @feature_091_std_force_y
    @feature_092_mean_power
    @feature_093_std_power
    @feature_094_mean_tracking_error
    @feature_095_std_tracking_error
    @feature_096_commanded_actual_speed_correlation
    };

feature_count = numel(feature_functions);
assert(feature_count == 96, 'Exactly 96 feature functions are required.');
feature_names = cellfun(@func2str, feature_functions, ...
    'UniformOutput', false);
feature_names = regexprep(feature_names, '^feature_[0-9]{3}_', '');

%% Calculate features and outcomes over sliding windows
analysis_start_s = min(Pt);
analysis_end_s = max(Pt);
if analysis_end_s - analysis_start_s < window_size_s
    error('The recording is shorter than one window.');
end

window_start_s = (analysis_start_s:window_step_s: ...
    (analysis_end_s - window_size_s))';
window_end_s = window_start_s + window_size_s;
window_count = numel(window_start_s);

feature_values = NaN(window_count, feature_count);
dsj = NaN(window_count, 1);
mean_force_n = NaN(window_count, 1);
robot_sample_count = zeros(window_count, 1);

for window_idx = 1:window_count
    is_last_window = window_idx == window_count;
    if is_last_window
        robot_window = Pt >= window_start_s(window_idx) & ...
            Pt <= window_end_s(window_idx);
    else
        robot_window = Pt >= window_start_s(window_idx) & ...
            Pt < window_end_s(window_idx);
    end

    robot_sample_count(window_idx) = nnz(robot_window);
    if robot_sample_count(window_idx) < minimum_robot_samples
        continue;
    end

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
            window_Px, window_Py, window_Pt, window_Vx, window_Vy, ...
            window_Fx, window_Fy, window_Wmag, window_Vcx, window_Vcy);
    end

    dsj(window_idx) = calculate_planar_dsj( ...
        end_effector_x_m(robot_window), end_effector_y_m(robot_window), ...
        Pt(robot_window));
    mean_force_n(window_idx) = calculate_mean_planar_force( ...
        window_Fx, window_Fy);
end

feature_table = array2table(feature_values, ...
    'VariableNames', feature_names);
window_metadata = table(window_start_s, window_end_s, robot_sample_count);
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

fprintf('Robot file: %s\n', robot_data_file);
fprintf('Window size/step: %.3f s / %.3f s\n', ...
    window_size_s, window_step_s);
fprintf('Analyzed windows: %d\n', window_count);
fprintf('\nFeatures ranked by absolute correlation with DSJ:\n');
disp(correlation_ranked_by_dsj);
fprintf('\nFeatures ranked by absolute correlation with mean force ');
fprintf('(damping-confounded -- see file header):\n');
disp(correlation_ranked_by_mean_force);

%% Plot normalized speed, DSJ, and mean force over the trial
window_center_s = (window_start_s + window_end_s) / 2.0;
mean_speed = window_results.mean_speed;

mean_speed_normalized = normalize_finite_to_unit_interval(mean_speed);
dsj_normalized = normalize_finite_to_unit_interval(dsj);
mean_force_normalized = normalize_finite_to_unit_interval(mean_force_n);

figure('Name', 'Normalized speed, DSJ, and force', 'Color', 'w');
hold on;
plot(window_center_s, mean_speed_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Mean speed');
plot(window_center_s, dsj_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'DSJ');
plot(window_center_s, mean_force_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Mean planar force');
grid on;
xlabel('Window center time [s]');
ylabel('Min-max normalized value [0, 1]');
title('Mean speed compared with DSJ and mean force');
legend('Location', 'best');
ylim([0, 1]);

%% Feature functions 001-096
% Every feature has the required (Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx,
% Vcy) interface and returns one scalar for one sliding window.

function value = feature_001_mean_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_001_MEAN_POSITION_X Mean horizontal position [m].
    value = safe_mean(Px);
end


function value = feature_002_mean_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_002_MEAN_POSITION_Y Mean vertical position [m].
    value = safe_mean(Py);
end


function value = feature_003_std_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_003_STD_POSITION_X Horizontal position standard deviation [m].
    value = safe_std(Px);
end


function value = feature_004_std_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_004_STD_POSITION_Y Vertical position standard deviation [m].
    value = safe_std(Py);
end


function value = feature_005_variance_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_005_VARIANCE_POSITION_X Horizontal position variance [m^2].
    value = safe_variance(Px);
end


function value = feature_006_variance_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_006_VARIANCE_POSITION_Y Vertical position variance [m^2].
    value = safe_variance(Py);
end


function value = feature_007_minimum_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_007_MINIMUM_POSITION_X Minimum horizontal position.
    value = safe_minimum(Px);
end


function value = feature_008_maximum_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_008_MAXIMUM_POSITION_X Maximum horizontal position.
    value = safe_maximum(Px);
end


function value = feature_009_range_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_009_RANGE_POSITION_X Horizontal position range.
    value = safe_range(Px);
end


function value = feature_010_minimum_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_010_MINIMUM_POSITION_Y Minimum vertical position.
    value = safe_minimum(Py);
end


function value = feature_011_maximum_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_011_MAXIMUM_POSITION_Y Maximum vertical position.
    value = safe_maximum(Py);
end


function value = feature_012_range_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_012_RANGE_POSITION_Y Vertical position range.
    value = safe_range(Py);
end


function value = feature_013_median_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_013_MEDIAN_POSITION_X Median horizontal position.
    value = safe_median(Px);
end


function value = feature_014_median_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_014_MEDIAN_POSITION_Y Median vertical position.
    value = safe_median(Py);
end


function value = feature_015_iqr_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_015_IQR_POSITION_X Horizontal position interquartile range.
    value = safe_interquartile_range(Px);
end


function value = feature_016_iqr_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_016_IQR_POSITION_Y Vertical position interquartile range.
    value = safe_interquartile_range(Py);
end


function value = feature_017_mad_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_017_MAD_POSITION_X Median absolute horizontal position deviation.
    value = safe_median_absolute_deviation(Px);
end


function value = feature_018_mad_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_018_MAD_POSITION_Y Median absolute vertical position deviation.
    value = safe_median_absolute_deviation(Py);
end


function value = feature_019_skewness_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_019_SKEWNESS_POSITION_X Horizontal position distribution skewness.
    value = safe_skewness(Px);
end


function value = feature_020_skewness_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_020_SKEWNESS_POSITION_Y Vertical position distribution skewness.
    value = safe_skewness(Py);
end


function value = feature_021_kurtosis_position_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_021_KURTOSIS_POSITION_X Horizontal position Pearson kurtosis.
    value = safe_kurtosis(Px);
end


function value = feature_022_kurtosis_position_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_022_KURTOSIS_POSITION_Y Vertical position Pearson kurtosis.
    value = safe_kurtosis(Py);
end


function value = feature_023_covariance_position_xy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_023_COVARIANCE_POSITION_XY Sample covariance of position x and y.
    value = safe_covariance(Px, Py);
end


function value = feature_024_correlation_position_xy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_024_CORRELATION_POSITION_XY Pearson correlation of position x/y.
    value = pearson_correlation(Px, Py);
end


function value = feature_025_position_dispersion_rms( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_025_POSITION_DISPERSION_RMS RMS distance from the position centroid.
    center_x = safe_mean(Px);
    center_y = safe_mean(Py);
    radius = hypot(Px - center_x, Py - center_y);
    value = safe_root_mean_square(radius);
end


function value = feature_026_position_bcea_68( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_026_POSITION_BCEA_68 Area of the 68% bivariate position ellipse.
    sigma_x = safe_std(Px);
    sigma_y = safe_std(Py);
    rho_xy = pearson_correlation(Px, Py);
    probability = 0.68;
    ellipse_factor = -log(1.0 - probability);

    if ~isfinite(sigma_x) || ~isfinite(sigma_y) || ~isfinite(rho_xy)
        value = NaN;
        return;
    end

    value = 2.0 * pi * ellipse_factor * sigma_x * sigma_y * ...
        sqrt(max(0.0, 1.0 - rho_xy^2));
end


function value = feature_027_path_length( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_027_PATH_LENGTH Total path length [m].
    [x, y, ~] = clean_planar_time_series(Px, Py, Pt);
    value = planar_path_length(x, y);
end


function value = feature_028_net_displacement( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_028_NET_DISPLACEMENT Start-to-end displacement [m].
    [x, y, ~] = clean_planar_time_series(Px, Py, Pt);
    value = planar_net_displacement(x, y);
end


function value = feature_029_path_efficiency( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_029_PATH_EFFICIENCY Net displacement divided by path length.
    [x, y, ~] = clean_planar_time_series(Px, Py, Pt);
    path_length = planar_path_length(x, y);
    net_displacement = planar_net_displacement(x, y);
    value = safe_ratio(net_displacement, path_length);
end


function value = feature_030_mean_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_030_MEAN_SPEED Mean planar speed [m/s].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(motion.speed);
end


function value = feature_031_std_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_031_STD_SPEED Speed standard deviation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_std(motion.speed);
end


function value = feature_032_max_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_032_MAX_SPEED Maximum speed.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_maximum(motion.speed);
end


function value = feature_033_median_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_033_MEDIAN_SPEED Median speed.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_median(motion.speed);
end


function value = feature_034_p90_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_034_P90_SPEED 90th percentile of speed.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = percentile_value(motion.speed, 90.0);
end


function value = feature_035_rms_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_035_RMS_SPEED Root-mean-square speed.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_root_mean_square(motion.speed);
end


function value = feature_036_mean_acceleration( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_036_MEAN_ACCELERATION Mean acceleration magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(motion.acceleration);
end


function value = feature_037_std_acceleration( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_037_STD_ACCELERATION Acceleration standard deviation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_std(motion.acceleration);
end


function value = feature_038_max_acceleration( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_038_MAX_ACCELERATION Maximum acceleration magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_maximum(motion.acceleration);
end


function value = feature_039_rms_acceleration( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_039_RMS_ACCELERATION RMS acceleration magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_root_mean_square(motion.acceleration);
end


function value = feature_040_mean_jerk( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_040_MEAN_JERK Mean jerk magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(motion.jerk);
end


function value = feature_041_std_jerk( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_041_STD_JERK Jerk standard deviation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_std(motion.jerk);
end


function value = feature_042_max_jerk( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_042_MAX_JERK Maximum jerk magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_maximum(motion.jerk);
end


function value = feature_043_rms_jerk( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_043_RMS_JERK RMS jerk magnitude.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_root_mean_square(motion.jerk);
end


function value = feature_044_mean_velocity_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_044_MEAN_VELOCITY_X Mean horizontal velocity.
    value = safe_mean(Vx);
end


function value = feature_045_mean_velocity_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_045_MEAN_VELOCITY_Y Mean vertical velocity.
    value = safe_mean(Vy);
end


function value = feature_046_std_velocity_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_046_STD_VELOCITY_X Horizontal velocity standard deviation.
    value = safe_std(Vx);
end


function value = feature_047_std_velocity_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_047_STD_VELOCITY_Y Vertical velocity standard deviation.
    value = safe_std(Vy);
end


function value = feature_048_mean_direction( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_048_MEAN_DIRECTION Circular mean motion direction [rad].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = circular_mean(motion.direction);
end


function value = feature_049_direction_concentration( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_049_DIRECTION_CONCENTRATION Circular resultant in [0, 1].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = circular_concentration(motion.direction);
end


function value = feature_050_mean_absolute_direction_change( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_050_MEAN_ABSOLUTE_DIRECTION_CHANGE Mean turn angle [rad].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    direction_change = atan2( ...
        sin(diff(motion.direction)), cos(diff(motion.direction)));
    value = safe_mean(abs(direction_change));
end


function value = feature_051_low_speed_fraction( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_051_LOW_SPEED_FRACTION Fraction below 0.02 m/s.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(double(motion.speed < 0.02));
end


function value = feature_052_high_speed_fraction( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_052_HIGH_SPEED_FRACTION Fraction above 0.08 m/s.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(double(motion.speed > 0.08));
end


function value = feature_053_high_speed_event_count( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_053_HIGH_SPEED_EVENT_COUNT Number of high-speed episodes.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = count_true_runs(motion.speed > 0.08);
end


function value = feature_054_low_speed_event_count( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_054_LOW_SPEED_EVENT_COUNT Number of low-speed episodes.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = count_true_runs(motion.speed < 0.02);
end


function value = feature_055_missing_time_fraction( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_055_MISSING_TIME_FRACTION Excess time in sample gaps.
    [~, ~, time_s] = clean_planar_time_series(Px, Py, Pt);
    value = missing_time_fraction(time_s);
end


function value = feature_056_gap_count( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_056_GAP_COUNT Count intervals above 1.5 nominal periods.
    [~, ~, time_s] = clean_planar_time_series(Px, Py, Pt);
    value = sample_gap_count(time_s);
end


function value = feature_057_sample_rate( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_057_SAMPLE_RATE Effective sample rate [Hz].
    [~, ~, time_s] = clean_planar_time_series(Px, Py, Pt);
    if numel(time_s) < 2 || time_s(end) <= time_s(1)
        value = NaN;
        return;
    end
    value = (numel(time_s) - 1) / (time_s(end) - time_s(1));
end


function value = feature_058_position_x_shannon_entropy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_058_POSITION_X_SHANNON_ENTROPY Entropy of 10 horizontal bins
% spanning this window's own x range [bits]. Unlike gaze_feature_
% extraction.m's display-based entropy, there is no fixed physical
% "screen" for robot position to bin against, so the bin edges are
% this WINDOW's own min/max -- see one_dimensional_entropy_over_range().
    value = one_dimensional_entropy_over_range(Px, 10);
end


function value = feature_059_position_y_shannon_entropy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_059_POSITION_Y_SHANNON_ENTROPY Entropy of 10 vertical bins [bits].
    value = one_dimensional_entropy_over_range(Py, 10);
end


function value = feature_060_stationary_shannon_entropy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_060_STATIONARY_SHANNON_ENTROPY 10x10 occupancy entropy [bits].
    states = spatial_states_over_range(Px, Py, 10);
    value = shannon_entropy_from_states(states, 100);
end


function value = feature_061_transitional_shannon_entropy( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_061_TRANSITIONAL_SHANNON_ENTROPY Conditional transition entropy.
    states = spatial_states_over_range(Px, Py, 10);
    value = transition_shannon_entropy(states, 100);
end


function value = feature_062_position_x_mean_crossing_count( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_062_POSITION_X_MEAN_CROSSING_COUNT Horizontal mean crossings.
    value = mean_crossing_count(Px);
end


function value = feature_063_position_y_mean_crossing_count( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_063_POSITION_Y_MEAN_CROSSING_COUNT Vertical mean crossings.
    value = mean_crossing_count(Py);
end


function value = feature_064_position_x_lag1_autocorrelation( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_064_POSITION_X_LAG1_AUTOCORRELATION Horizontal lag-one correlation.
    value = lag_one_autocorrelation(Px);
end


function value = feature_065_position_y_lag1_autocorrelation( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_065_POSITION_Y_LAG1_AUTOCORRELATION Vertical lag-one correlation.
    value = lag_one_autocorrelation(Py);
end


function value = feature_066_speed_lag1_autocorrelation( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_066_SPEED_LAG1_AUTOCORRELATION Speed lag-one correlation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = lag_one_autocorrelation(motion.speed);
end


function value = feature_067_mean_curvature( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_067_MEAN_CURVATURE Mean signed path curvature [1/m].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(motion.curvature);
end


function value = feature_068_std_curvature( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_068_STD_CURVATURE Signed curvature standard deviation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_std(motion.curvature);
end


function value = feature_069_median_absolute_curvature( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_069_MEDIAN_ABSOLUTE_CURVATURE Median |curvature| [1/m].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_median(abs(motion.curvature));
end


function value = feature_070_max_absolute_curvature( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_070_MAX_ABSOLUTE_CURVATURE Maximum |curvature| [1/m].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_maximum(abs(motion.curvature));
end


function value = feature_071_mean_curvature_dot( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_071_MEAN_CURVATURE_DOT Mean rate of curvature change [1/(m s)].
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_mean(motion.curvature_dot);
end


function value = feature_072_std_curvature_dot( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_072_STD_CURVATURE_DOT Curvature-rate standard deviation.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    value = safe_std(motion.curvature_dot);
end


function value = feature_073_curvature_sign_reversal_rate( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_073_CURVATURE_SIGN_REVERSAL_RATE How often signed curvature
% changes sign, per metre actually travelled in this window.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    valid_curvature = motion.curvature(isfinite(motion.curvature));

    [x, y, ~] = clean_planar_time_series(Px, Py, Pt);
    distance_travelled_m = planar_path_length(x, y);

    if numel(valid_curvature) < 2 || ~isfinite(distance_travelled_m) || ...
            distance_travelled_m <= 0
        value = NaN;
        return;
    end
    sign_changes = sum(diff(sign(valid_curvature)) ~= 0);
    value = sign_changes / distance_travelled_m;
end


function value = feature_074_power_law_r2( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_074_POWER_LAW_R2 Goodness of fit of v = K*kappa^(-beta) in this
% window (same regression this project's power-law estimators use
% elsewhere, e.g. ur5e_chen_var_adm_2d.py and extract_pilot_features_
% fixed_adm_2d.m's fit_power_law_r2).
    motion = calculate_robot_motion(Vx, Vy, Pt);
    [r_squared, ~] = fit_power_law(motion.speed, motion.curvature);
    value = r_squared;
end


function value = feature_075_power_law_exponent( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_075_POWER_LAW_EXPONENT Fitted beta in v = K*kappa^(-beta) --
% how strongly speed drops as curvature tightens in this window.
    motion = calculate_robot_motion(Vx, Vy, Pt);
    [~, beta] = fit_power_law(motion.speed, motion.curvature);
    value = beta;
end


function value = feature_076_mean_angular_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_076_MEAN_ANGULAR_SPEED Mean 3-axis angular speed magnitude [rad/s].
    value = safe_mean(Wmag);
end


function value = feature_077_std_angular_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_077_STD_ANGULAR_SPEED Angular speed standard deviation.
    value = safe_std(Wmag);
end


function value = feature_078_max_angular_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_078_MAX_ANGULAR_SPEED Maximum angular speed.
    value = safe_maximum(Wmag);
end


function value = feature_079_rms_angular_speed( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_079_RMS_ANGULAR_SPEED RMS angular speed.
    value = safe_root_mean_square(Wmag);
end


function value = feature_080_angular_speed_lag1_autocorrelation( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_080_ANGULAR_SPEED_LAG1_AUTOCORRELATION Angular-speed lag-one
% correlation.
    value = lag_one_autocorrelation(Wmag);
end


function value = feature_081_stationary_shannon_entropy_40x40( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_081_STATIONARY_SHANNON_ENTROPY_40X40 Occupancy entropy [bits].
    states = spatial_states_over_range(Px, Py, 40);
    value = shannon_entropy_from_states(states, 1600);
end


function value = feature_082_transitional_shannon_entropy_40x40( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_082_TRANSITIONAL_SHANNON_ENTROPY_40X40 Transition entropy [bits].
    states = spatial_states_over_range(Px, Py, 40);
    value = transition_shannon_entropy(states, 1600);
end


function value = feature_083_mean_force_magnitude( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_083_MEAN_FORCE_MAGNITUDE Mean filtered planar force [N].
% DAMPING-CONFOUNDED -- see file header.
    value = safe_mean(hypot(Fx, Fy));
end


function value = feature_084_std_force_magnitude( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_084_STD_FORCE_MAGNITUDE Force magnitude standard deviation.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_std(hypot(Fx, Fy));
end


function value = feature_085_max_force_magnitude( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_085_MAX_FORCE_MAGNITUDE Maximum force magnitude.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_maximum(hypot(Fx, Fy));
end


function value = feature_086_median_force_magnitude( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_086_MEDIAN_FORCE_MAGNITUDE Median force magnitude.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_median(hypot(Fx, Fy));
end


function value = feature_087_rms_force_magnitude( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_087_RMS_FORCE_MAGNITUDE RMS force magnitude.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_root_mean_square(hypot(Fx, Fy));
end


function value = feature_088_mean_force_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_088_MEAN_FORCE_X Mean signed horizontal force [N].
% DAMPING-CONFOUNDED -- see file header.
    value = safe_mean(Fx);
end


function value = feature_089_mean_force_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_089_MEAN_FORCE_Y Mean signed vertical force [N].
% DAMPING-CONFOUNDED -- see file header.
    value = safe_mean(Fy);
end


function value = feature_090_std_force_x( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_090_STD_FORCE_X Horizontal force standard deviation.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_std(Fx);
end


function value = feature_091_std_force_y( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_091_STD_FORCE_Y Vertical force standard deviation.
% DAMPING-CONFOUNDED -- see file header.
    value = safe_std(Fy);
end


function value = feature_092_mean_power( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_092_MEAN_POWER Mean interaction power F.v [W].
% DAMPING-CONFOUNDED -- see file header (inherits force's confound).
    power_w = Fx .* Vx + Fy .* Vy;
    value = safe_mean(power_w);
end


function value = feature_093_std_power( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_093_STD_POWER Interaction power standard deviation.
% DAMPING-CONFOUNDED -- see file header (inherits force's confound).
    power_w = Fx .* Vx + Fy .* Vy;
    value = safe_std(power_w);
end


function value = feature_094_mean_tracking_error( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_094_MEAN_TRACKING_ERROR Mean |commanded - actual velocity| [m/s]
% -- how far the admittance controller's own commanded velocity was from
% what the robot actually achieved. DAMPING-CONFOUNDED -- see file
% header: higher damping lags the command more for the same intent.
    tracking_error = hypot(Vcx - Vx, Vcy - Vy);
    value = safe_mean(tracking_error);
end


function value = feature_095_std_tracking_error( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_095_STD_TRACKING_ERROR Tracking-error standard deviation.
% DAMPING-CONFOUNDED -- see file header.
    tracking_error = hypot(Vcx - Vx, Vcy - Vy);
    value = safe_std(tracking_error);
end


function value = feature_096_commanded_actual_speed_correlation( ...
        Px, Py, Pt, Vx, Vy, Fx, Fy, Wmag, Vcx, Vcy)
%FEATURE_096_COMMANDED_ACTUAL_SPEED_CORRELATION Correlation between
% commanded and actual speed magnitude. DAMPING-CONFOUNDED -- see file
% header.
    commanded_speed = hypot(Vcx, Vcy);
    actual_speed = hypot(Vx, Vy);
    value = pearson_correlation(commanded_speed, actual_speed);
end

%% Robot-specific helpers (new -- not in gaze_feature_extraction.m)

function motion = calculate_robot_motion(Vx, Vy, Pt)
%CALCULATE_ROBOT_MOTION Speed, acceleration, jerk, direction, and signed
% curvature from MEASURED velocity (not re-differentiated from
% position) -- one derivative closer to what was actually measured,
% matching this project's other robot derivative chains
% (compute_motion_derivatives in extract_pilot_features_fixed_adm_2d.m).
%
% Signed curvature (same convention used throughout this project):
%
%   kappa = (vx*ay - vy*ax) / |v|^3
%
% set to NaN wherever |v| < MINIMUM_SPEED_M_S, where direction (and
% therefore curvature) is undefined rather than just small. Hardcoded
% here (not a User Setting) -- see this script's User Settings section
% for why.

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


function [r_squared, beta] = fit_power_law( ...
        window_speed_m_s, window_curvature_1_m)
%FIT_POWER_LAW Least-squares fit of v = K*kappa^(-beta) within one
% window, via linear regression of log(v) on log(kappa) -- same
% regression this project's other power-law estimators use (e.g.
% extract_pilot_features_fixed_adm_2d.m's fit_power_law_r2(),
% ur5e_chen_var_adm_2d.py's online estimator). Returns both the
% goodness of fit (r_squared) and the fitted exponent (beta); [NaN, NaN]
% if too few valid samples fall in this window to fit a meaningful line.

    MINIMUM_SPEED_M_S = 0.005;
    MINIMUM_SAMPLE_COUNT = 10;

    valid_mask = window_speed_m_s > MINIMUM_SPEED_M_S & ...
        isfinite(window_curvature_1_m) & window_curvature_1_m ~= 0;

    log_speed = log(window_speed_m_s(valid_mask));
    log_curvature = log(abs(window_curvature_1_m(valid_mask)));

    sample_count = numel(log_speed);
    if sample_count < MINIMUM_SAMPLE_COUNT
        r_squared = NaN;
        beta = NaN;
        return;
    end

    sum_x = sum(log_curvature);
    sum_y = sum(log_speed);
    sum_xx = sum(log_curvature .^ 2);
    sum_xy = sum(log_curvature .* log_speed);

    denominator = sample_count * sum_xx - sum_x ^ 2;
    if abs(denominator) < 1e-9
        r_squared = NaN;
        beta = NaN;
        return;
    end

    slope = (sample_count * sum_xy - sum_x * sum_y) / denominator;
    intercept = (sum_y - slope * sum_x) / sample_count;
    beta = -slope;

    predicted_log_speed = intercept + slope * log_curvature;
    residual_sum_squares = sum((log_speed - predicted_log_speed) .^ 2);
    total_sum_squares = sum((log_speed - mean(log_speed)) .^ 2);

    if total_sum_squares < 1e-12
        r_squared = NaN;
    else
        r_squared = 1 - residual_sum_squares / total_sum_squares;
    end
end


function value = one_dimensional_entropy_over_range(values, bin_count)
%ONE_DIMENSIONAL_ENTROPY_OVER_RANGE Shannon entropy of bin_count equal-
% width bins spanning THIS WINDOW's own [min, max] range [bits]. Unlike
% gaze_feature_extraction.m's one_dimensional_display_entropy(), robot
% position has no fixed physical box to bin against, so the range is
% taken from the data itself -- a window that barely moved shows near-
% zero entropy almost by construction (nearly everything in one bin),
% which is the intended behaviour (low spatial spread reads as low
% entropy) rather than a defect.

    values = finite_values(values);
    if numel(values) < 2
        value = NaN;
        return;
    end

    minimum_value = min(values);
    value_range = max(values) - minimum_value;
    if value_range <= 0
        value = 0.0;  % every sample in exactly one bin -> zero entropy
        return;
    end

    clipped_values = min(max( ...
        (values - minimum_value) / value_range, 0.0), 1.0 - 1e-12);
    states = floor(clipped_values * bin_count) + 1;
    value = shannon_entropy_from_states(states, bin_count);
end


function states = spatial_states_over_range(x, y, bin_count_per_axis)
%SPATIAL_STATES_OVER_RANGE Convert position into 2D grid state IDs,
% binning each axis over THIS WINDOW's own [min, max] range -- the
% robot-position analogue of gaze_feature_extraction.m's
% spatial_gaze_states(), which bins over a fixed [0, 1] display instead
% (robot position has no equivalent fixed box -- see
% one_dimensional_entropy_over_range() above for the same reasoning).

    x = double(x(:));
    y = double(y(:));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);

    if numel(x) < 2
        states = [];
        return;
    end

    x_range = max(x) - min(x);
    y_range = max(y) - min(y);

    if x_range > 0
        nx = min(max((x - min(x)) / x_range, 0.0), 1.0 - 1e-12);
    else
        nx = zeros(size(x));
    end
    if y_range > 0
        ny = min(max((y - min(y)) / y_range, 0.0), 1.0 - 1e-12);
    else
        ny = zeros(size(y));
    end

    x_bin = floor(nx * bin_count_per_axis) + 1;
    y_bin = floor(ny * bin_count_per_axis) + 1;
    states = x_bin + (y_bin - 1) * bin_count_per_axis;
end


%% Shared helpers, copied verbatim from gaze_feature_extraction.m
% (generic statistics/signal-processing utilities with no gaze-specific
% content -- identical behaviour is exactly the point, so a feature that
% happens to be computed the same way on both sides, e.g. entropy of a
% fixed-box signal, is genuinely comparable).

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


function dsj = calculate_planar_dsj(x_m, y_m, time_s)
%CALCULATE_PLANAR_DSJ Dimensionless squared jerk over one window.
%
% DSJ = T^5 / L^2 * integral(||jerk(t)||^2 dt), where T is window
% duration and L is planar path length. Lower DSJ indicates smoother
% motion. Uses x_m/y_m (raw robot-base-frame position, differentiated
% three times from position) rather than the calculate_robot_motion()
% chain above (which differentiates measured velocity only twice) --
% see file header for why DSJ specifically keeps this project's usual
% position-based convention instead.
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

%% Shared feature helpers
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


function value = safe_variance(values)
%SAFE_VARIANCE Sample variance of finite values.
    standard_deviation = safe_std(values);
    value = standard_deviation^2;
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


function value = safe_range(values)
%SAFE_RANGE Maximum minus minimum of finite values.
    minimum_value = safe_minimum(values);
    maximum_value = safe_maximum(values);
    if ~isfinite(minimum_value) || ~isfinite(maximum_value)
        value = NaN;
    else
        value = maximum_value - minimum_value;
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


function value = safe_interquartile_range(values)
%SAFE_INTERQUARTILE_RANGE Difference between 75th and 25th percentiles.
    value = percentile_value(values, 75.0) - ...
        percentile_value(values, 25.0);
end


function value = safe_median_absolute_deviation(values)
%SAFE_MEDIAN_ABSOLUTE_DEVIATION Median absolute deviation from median.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
        return;
    end
    value = median(abs(values - median(values)));
end


function value = safe_skewness(values)
%SAFE_SKEWNESS Population-moment skewness of finite samples.
    values = finite_values(values);
    if numel(values) < 3
        value = NaN;
        return;
    end

    centered = values - mean(values);
    second_moment = mean(centered.^2);
    if second_moment <= 0
        value = NaN;
    else
        value = mean(centered.^3) / second_moment^(3 / 2);
    end
end


function value = safe_kurtosis(values)
%SAFE_KURTOSIS Population-moment Pearson kurtosis of finite samples.
    values = finite_values(values);
    if numel(values) < 4
        value = NaN;
        return;
    end

    centered = values - mean(values);
    second_moment = mean(centered.^2);
    if second_moment <= 0
        value = NaN;
    else
        value = mean(centered.^4) / second_moment^2;
    end
end


function value = safe_covariance(first_values, second_values)
%SAFE_COVARIANCE Sample covariance of paired finite values.
    first_values = double(first_values(:));
    second_values = double(second_values(:));
    valid = isfinite(first_values) & isfinite(second_values);
    first_values = first_values(valid);
    second_values = second_values(valid);

    if numel(first_values) < 2
        value = NaN;
        return;
    end

    first_centered = first_values - mean(first_values);
    second_centered = second_values - mean(second_values);
    value = sum(first_centered .* second_centered) / ...
        (numel(first_values) - 1);
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


function value = planar_net_displacement(x, y)
%PLANAR_NET_DISPLACEMENT Euclidean distance from first to last sample.
    x = double(x(:));
    y = double(y(:));
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    if numel(x) < 2
        value = NaN;
    else
        value = hypot(x(end) - x(1), y(end) - y(1));
    end
end


function value = circular_mean(angles_rad)
%CIRCULAR_MEAN Circular mean angle [rad].
    angles_rad = finite_values(angles_rad);
    if isempty(angles_rad)
        value = NaN;
    else
        value = atan2(mean(sin(angles_rad)), mean(cos(angles_rad)));
    end
end


function value = circular_concentration(angles_rad)
%CIRCULAR_CONCENTRATION Mean resultant length in [0, 1].
    angles_rad = finite_values(angles_rad);
    if isempty(angles_rad)
        value = NaN;
    else
        value = hypot( ...
            mean(cos(angles_rad)), mean(sin(angles_rad)));
    end
end


function count = count_true_runs(logical_values)
%COUNT_TRUE_RUNS Count contiguous true episodes in a logical vector.
    logical_values = logical(logical_values(:));
    if isempty(logical_values)
        count = NaN;
    else
        count = double(logical_values(1)) + ...
            nnz(~logical_values(1:end-1) & logical_values(2:end));
    end
end


function value = missing_time_fraction(time_s)
%MISSING_TIME_FRACTION Fraction of duration beyond nominal sample periods.
    time_s = sort(finite_values(time_s));
    if numel(time_s) < 3 || time_s(end) <= time_s(1)
        value = NaN;
        return;
    end

    sample_intervals_s = diff(time_s);
    nominal_interval_s = median(sample_intervals_s);
    is_gap = sample_intervals_s > 1.5 * nominal_interval_s;
    missing_time_s = sum( ...
        sample_intervals_s(is_gap) - nominal_interval_s);
    value = missing_time_s / (time_s(end) - time_s(1));
end


function value = sample_gap_count(time_s)
%SAMPLE_GAP_COUNT Count intervals above 1.5 nominal sample periods.
    time_s = sort(finite_values(time_s));
    if numel(time_s) < 3
        value = NaN;
        return;
    end
    sample_intervals_s = diff(time_s);
    nominal_interval_s = median(sample_intervals_s);
    value = nnz(sample_intervals_s > 1.5 * nominal_interval_s);
end


function value = shannon_entropy_from_states(states, state_count)
%SHANNON_ENTROPY_FROM_STATES Shannon entropy of state occupancy [bits].
    states = double(states(:));
    valid = isfinite(states) & states >= 1 & states <= state_count;
    states = round(states(valid));
    if isempty(states)
        value = NaN;
        return;
    end

    counts = accumarray(states, 1, [state_count, 1]);
    probabilities = counts / sum(counts);
    value = entropy_from_probabilities(probabilities);
end


function value = transition_shannon_entropy(states, state_count)
%TRANSITION_SHANNON_ENTROPY Occupancy-weighted next-state entropy [bits].
    states = double(states(:));
    valid = isfinite(states) & states >= 1 & states <= state_count;
    states = round(states(valid));
    if numel(states) < 2
        value = NaN;
        return;
    end

    from_states = states(1:end-1);
    to_states = states(2:end);
    transition_counts = sparse( ...
        from_states, to_states, 1, state_count, state_count);
    from_counts = full(sum(transition_counts, 2));
    total_transitions = sum(from_counts);

    conditional_entropy = zeros(state_count, 1);
    occupied_from_states = find(from_counts > 0);
    for occupied_idx = 1:numel(occupied_from_states)
        state_idx = occupied_from_states(occupied_idx);
        transition_probabilities = nonzeros( ...
            transition_counts(state_idx, :)) / from_counts(state_idx);
        conditional_entropy(state_idx) = ...
            entropy_from_probabilities(transition_probabilities);
    end

    state_weights = from_counts / total_transitions;
    value = sum(state_weights .* conditional_entropy);
end


function value = entropy_from_probabilities(probabilities)
%ENTROPY_FROM_PROBABILITIES Shannon entropy in bits.
    probabilities = double(probabilities(:));
    probabilities = probabilities(isfinite(probabilities) & ...
        probabilities > 0);
    if isempty(probabilities)
        value = 0.0;
    else
        value = -sum(probabilities .* log2(probabilities));
    end
end


function value = mean_crossing_count(values)
%MEAN_CROSSING_COUNT Count crossings of a signal's own mean.
    values = finite_values(values);
    if numel(values) < 2
        value = NaN;
        return;
    end

    centered_sign = sign(values - mean(values));
    centered_sign = centered_sign(centered_sign ~= 0);
    if numel(centered_sign) < 2
        value = 0;
    else
        value = nnz(centered_sign(1:end-1) .* centered_sign(2:end) < 0);
    end
end


function value = lag_one_autocorrelation(values)
%LAG_ONE_AUTOCORRELATION Correlation between consecutive signal samples.
    values = finite_values(values);
    if numel(values) < 4
        value = NaN;
    else
        value = pearson_correlation(values(1:end-1), values(2:end));
    end
end


function normalized_values = normalize_finite_to_unit_interval(values)
%NORMALIZE_FINITE_TO_UNIT_INTERVAL Min-max normalize finite values to [0,1].
%
% NaN and Inf remain NaN. A constant finite signal is plotted at zero.
    values = double(values(:));
    normalized_values = NaN(size(values));
    finite_rows = isfinite(values);
    finite_values_in_signal = values(finite_rows);
    if isempty(finite_values_in_signal)
        return;
    end

    minimum_value = min(finite_values_in_signal);
    maximum_value = max(finite_values_in_signal);
    value_range = maximum_value - minimum_value;
    if value_range <= 0
        normalized_values(finite_rows) = 0.0;
    else
        normalized_values(finite_rows) = ...
            (finite_values_in_signal - minimum_value) / value_range;
    end
end
