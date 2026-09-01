%% AWESOME_SCRIPT: Extract 102 windowed gaze features and correlations.
%
% Inputs:
%   Tobii CSV with normalized display gaze and system timestamps [us].
%   UR5e CSV with TCP/CoppeliaSim position [m], time [s], and force [N].
%
% Outputs retained in the MATLAB workspace:
%   window_results      - 102 features, DSJ, and mean force per window.
%   correlation_results - Pearson correlation of every feature with DSJ and
%                         mean planar force.
%   correlation_ranked_by_dsj and correlation_ranked_by_mean_force - the
%                         same correlations ranked by absolute magnitude.
%
% Shared feature-function interface:
%   value = feature_name(Gx, Gy, Gt, Ex, Ey, Et)
%   Gx, Gy - binocular gaze in normalized display coordinates [unitless].
%   Gt     - gaze time [s].
%   Ex, Ey - end effector mapped to normalized display coordinates.
%   Et     - end-effector time [s].
%
% Important assumptions:
%   Tobii and UR5e recordings start at approximately the same time. A manual
%   gaze_time_offset_s setting is provided for synchronization correction.
%   Relative features require gaze and end effector in the same display
%   frame. By default, CoppeliaSim x/y are min-max mapped over this trial and
%   display y is inverted. Replace the limits with calibrated values when
%   available. Stationary and transitional Shannon entropy are calculated
%   on both 10-by-10 and 40-by-40 display grids and reported in bits.

clear;
close all
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
gaze_data_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', 'old_data/', ...
    'tobii_fixed_adm_2d_s2_irregular_shape_d50.csv');
robot_data_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', 'old_data/', ...
    'ur5e_fixed_adm_2d_s2_irregular_shape_d50.csv');

window_size_s = 3.5;  % TODO window duration [s], recommended 0.25 to 2
window_step_s = 0.02; % window_size_s / 2;  % TODO step [s]; default is 50% overlap
gaze_time_offset_s = 0.0;  % TODO add to gaze time [s] for synchronization
minimum_gaze_samples = 6;  % TODO valid gaze samples/window, recommended >= 6
minimum_robot_samples = 10;  % TODO robot samples/window, recommended >= 10

% TODO calibrated CoppeliaSim display bounds [m], [minimum, maximum].
% Leave empty to use this trial's minimum and maximum values.
end_effector_x_display_limits_m = [];
end_effector_y_display_limits_m = [];
invert_end_effector_display_y = true;  % TODO true for top-left display origin

if window_size_s <= 0 || window_step_s <= 0
    error('window_size_s and window_step_s must be positive.');
end

%% Load raw recordings
gaze_table = readtable(gaze_data_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');
robot_table = readtable(robot_data_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_gaze_variables = {
    'system_time_stamp'
    'left_gaze_point_on_display_area'
    'right_gaze_point_on_display_area'
    'left_gaze_point_validity'
    'right_gaze_point_validity'
    };
required_robot_variables = {
    'time_s'
    'Fx_filt_N'
    'Fy_filt_N'
    'x_m'
    'y_m'
    'coppeliasim_x_m'
    'coppeliasim_y_m'
    };
require_table_variables(gaze_table, required_gaze_variables, 'Tobii CSV');
require_table_variables(robot_table, required_robot_variables, 'UR5e CSV');

%% Decode and combine binocular gaze
left_gaze = parse_vector_column( ...
    gaze_table.left_gaze_point_on_display_area, 2);
right_gaze = parse_vector_column( ...
    gaze_table.right_gaze_point_on_display_area, 2);
left_valid = get_numeric_column(gaze_table.left_gaze_point_validity) == 1;
right_valid = get_numeric_column(gaze_table.right_gaze_point_validity) == 1;

[Gx_all, Gy_all] = combine_binocular_gaze( ...
    left_gaze, right_gaze, left_valid, right_valid);
gaze_system_time_us = get_numeric_column(gaze_table.system_time_stamp);
first_gaze_timestamp_idx = find(isfinite(gaze_system_time_us), 1, 'first');
if isempty(first_gaze_timestamp_idx)
    error('The Tobii CSV contains no finite system timestamps.');
end

gaze_start_time_us = gaze_system_time_us(first_gaze_timestamp_idx);
Gt_all = (gaze_system_time_us - gaze_start_time_us) * 1e-6;
Gt_all = Gt_all + gaze_time_offset_s;

valid_gaze_rows = isfinite(Gx_all) & isfinite(Gy_all) & isfinite(Gt_all);
Gx = Gx_all(valid_gaze_rows);
Gy = Gy_all(valid_gaze_rows);
Gt = Gt_all(valid_gaze_rows);

%% Decode end-effector trajectory and planar force
Et_all = get_numeric_column(robot_table.time_s);
end_effector_x_m_all = get_numeric_column(robot_table.x_m);
end_effector_y_m_all = get_numeric_column(robot_table.y_m);
coppeliasim_x_m_all = get_numeric_column(robot_table.coppeliasim_x_m);
coppeliasim_y_m_all = get_numeric_column(robot_table.coppeliasim_y_m);
force_x_n_all = get_numeric_column(robot_table.Fx_filt_N);
force_y_n_all = get_numeric_column(robot_table.Fy_filt_N);

valid_robot_rows = isfinite(Et_all) & ...
    isfinite(end_effector_x_m_all) & isfinite(end_effector_y_m_all) & ...
    isfinite(coppeliasim_x_m_all) & isfinite(coppeliasim_y_m_all);

Et = Et_all(valid_robot_rows);
end_effector_x_m = end_effector_x_m_all(valid_robot_rows);
end_effector_y_m = end_effector_y_m_all(valid_robot_rows);
coppeliasim_x_m = coppeliasim_x_m_all(valid_robot_rows);
coppeliasim_y_m = coppeliasim_y_m_all(valid_robot_rows);
force_x_n = force_x_n_all(valid_robot_rows);
force_y_n = force_y_n_all(valid_robot_rows);

[Ex, end_effector_x_limits_used_m] = map_to_unit_interval( ...
    coppeliasim_x_m, end_effector_x_display_limits_m);
[Ey, end_effector_y_limits_used_m] = map_to_unit_interval( ...
    coppeliasim_y_m, end_effector_y_display_limits_m);
if invert_end_effector_display_y
    Ey = 1.0 - Ey;
end

%% Define exactly 102 feature functions
feature_functions = {
    @feature_001_mean_gaze_x
    @feature_002_mean_gaze_y
    @feature_003_std_gaze_x
    @feature_004_std_gaze_y
    @feature_005_variance_gaze_x
    @feature_006_variance_gaze_y
    @feature_007_minimum_gaze_x
    @feature_008_maximum_gaze_x
    @feature_009_range_gaze_x
    @feature_010_minimum_gaze_y
    @feature_011_maximum_gaze_y
    @feature_012_range_gaze_y
    @feature_013_median_gaze_x
    @feature_014_median_gaze_y
    @feature_015_iqr_gaze_x
    @feature_016_iqr_gaze_y
    @feature_017_mad_gaze_x
    @feature_018_mad_gaze_y
    @feature_019_skewness_gaze_x
    @feature_020_skewness_gaze_y
    @feature_021_kurtosis_gaze_x
    @feature_022_kurtosis_gaze_y
    @feature_023_covariance_gaze_xy
    @feature_024_correlation_gaze_xy
    @feature_025_gaze_dispersion_rms
    @feature_026_bcea_68
    @feature_027_gaze_path_length
    @feature_028_gaze_net_displacement
    @feature_029_gaze_path_efficiency
    @feature_030_mean_gaze_speed
    @feature_031_std_gaze_speed
    @feature_032_max_gaze_speed
    @feature_033_median_gaze_speed
    @feature_034_p90_gaze_speed
    @feature_035_rms_gaze_speed
    @feature_036_mean_gaze_acceleration
    @feature_037_std_gaze_acceleration
    @feature_038_max_gaze_acceleration
    @feature_039_rms_gaze_acceleration
    @feature_040_mean_gaze_jerk
    @feature_041_std_gaze_jerk
    @feature_042_max_gaze_jerk
    @feature_043_rms_gaze_jerk
    @feature_044_mean_gaze_velocity_x
    @feature_045_mean_gaze_velocity_y
    @feature_046_std_gaze_velocity_x
    @feature_047_std_gaze_velocity_y
    @feature_048_mean_gaze_direction
    @feature_049_gaze_direction_concentration
    @feature_050_mean_absolute_direction_change
    @feature_051_low_speed_fraction
    @feature_052_high_speed_fraction
    @feature_053_high_speed_event_count
    @feature_054_low_speed_event_count
    @feature_055_gaze_missing_time_fraction
    @feature_056_gaze_gap_count
    @feature_057_gaze_sample_rate
    @feature_058_gaze_x_shannon_entropy
    @feature_059_gaze_y_shannon_entropy
    @feature_060_stationary_shannon_entropy
    @feature_061_transitional_shannon_entropy
    @feature_062_mean_distance_from_display_center
    @feature_063_std_distance_from_display_center
    @feature_064_display_center_fraction
    @feature_065_outside_display_fraction
    @feature_066_gaze_x_mean_crossing_count
    @feature_067_gaze_y_mean_crossing_count
    @feature_068_gaze_x_lag1_autocorrelation
    @feature_069_gaze_y_lag1_autocorrelation
    @feature_070_gaze_speed_lag1_autocorrelation
    @feature_071_mean_relative_x
    @feature_072_mean_relative_y
    @feature_073_std_relative_x
    @feature_074_std_relative_y
    @feature_075_mean_gaze_ee_distance
    @feature_076_std_gaze_ee_distance
    @feature_077_median_gaze_ee_distance
    @feature_078_max_gaze_ee_distance
    @feature_079_rms_gaze_ee_distance
    @feature_080_p90_gaze_ee_distance
    @feature_081_min_gaze_ee_distance
    @feature_082_final_gaze_ee_distance
    @feature_083_gaze_ee_distance_slope
    @feature_084_fraction_within_0p05
    @feature_085_fraction_within_0p10
    @feature_086_fraction_within_0p20
    @feature_087_gaze_ee_x_correlation
    @feature_088_gaze_ee_y_correlation
    @feature_089_gaze_ee_speed_correlation
    @feature_090_relative_path_length
    @feature_091_mean_relative_speed
    @feature_092_std_relative_speed
    @feature_093_mean_velocity_cosine_similarity
    @feature_094_aligned_velocity_fraction
    @feature_095_gaze_to_ee_speed_ratio
    @feature_096_max_x_cross_correlation
    @feature_097_x_lead_at_max_correlation_s
    @feature_098_max_y_cross_correlation
    @feature_099_y_lead_at_max_correlation_s
    @feature_100_mean_gaze_lead_along_ee_motion
    @feature_101_stationary_shannon_entropy_40x40
    @feature_102_transitional_shannon_entropy_40x40
    };

feature_count = numel(feature_functions);
assert(feature_count == 102, 'Exactly 102 feature functions are required.');
feature_names = cellfun(@func2str, feature_functions, ...
    'UniformOutput', false);
feature_names = regexprep(feature_names, '^feature_[0-9]{3}_', '');

%% Calculate features and outcomes over sliding windows
analysis_start_s = max(min(Gt), min(Et));
analysis_end_s = min(max(Gt), max(Et));
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
        robot_window = Et >= window_start_s(window_idx) & ...
            Et <= window_end_s(window_idx);
    else
        gaze_window = Gt >= window_start_s(window_idx) & ...
            Gt < window_end_s(window_idx);
        robot_window = Et >= window_start_s(window_idx) & ...
            Et < window_end_s(window_idx);
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
    window_Ex = Ex(robot_window);
    window_Ey = Ey(robot_window);
    window_Et = Et(robot_window);

    for feature_idx = 1:feature_count
        feature_values(window_idx, feature_idx) = ...
            feature_functions{feature_idx}( ...
            window_Gx, window_Gy, window_Gt, ...
            window_Ex, window_Ey, window_Et);
    end

    dsj(window_idx) = calculate_planar_dsj( ...
        end_effector_x_m(robot_window), ...
        end_effector_y_m(robot_window), ...
        Et(robot_window));
    mean_force_n(window_idx) = calculate_mean_planar_force( ...
        force_x_n(robot_window), force_y_n(robot_window));
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
fprintf('End-effector display x limits used [m]: [%.6f, %.6f]\n', ...
    end_effector_x_limits_used_m);
fprintf('End-effector display y limits used [m]: [%.6f, %.6f]\n', ...
    end_effector_y_limits_used_m);
fprintf('\nFeatures ranked by absolute correlation with DSJ:\n');
disp(correlation_ranked_by_dsj);
fprintf('\nFeatures ranked by absolute correlation with mean force:\n');
disp(correlation_ranked_by_mean_force);

%% Plot normalized gaze/EE speed ratio, DSJ, and mean force
window_center_s = (window_start_s + window_end_s) / 2.0;
gaze_ee_speed_ratio = window_results.gaze_to_ee_speed_ratio;

gaze_ee_speed_ratio_normalized = normalize_finite_to_unit_interval( ...
    gaze_ee_speed_ratio);
dsj_normalized = normalize_finite_to_unit_interval(dsj);
mean_force_normalized = normalize_finite_to_unit_interval(mean_force_n);

figure('Name', 'Normalized gaze-speed ratio, DSJ, and force', ...
    'Color', 'w');
hold on;
plot(window_center_s, gaze_ee_speed_ratio_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Gaze/EE speed ratio');
plot(window_center_s, dsj_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'DSJ');
plot(window_center_s, mean_force_normalized, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Mean planar force');
grid on;
xlabel('Window center time [s]');
ylabel('Min-max normalized value [0, 1]');
title('Gaze/EE speed ratio compared with DSJ and mean force');
legend('Location', 'best');
ylim([0, 1]);

%% Feature functions 001-102
% Every feature has the required (Gx, Gy, Gt, Ex, Ey, Et) interface and
% returns one scalar for one sliding window.

function value = feature_001_mean_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_001_MEAN_GAZE_X Mean horizontal gaze position [normalized].
    value = safe_mean(Gx);
end


function value = feature_002_mean_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_002_MEAN_GAZE_Y Mean vertical gaze position [normalized].
    value = safe_mean(Gy);
end


function value = feature_003_std_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_003_STD_GAZE_X Horizontal gaze standard deviation [normalized].
    value = safe_std(Gx);
end


function value = feature_004_std_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_004_STD_GAZE_Y Vertical gaze standard deviation [normalized].
    value = safe_std(Gy);
end


function value = feature_005_variance_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_005_VARIANCE_GAZE_X Horizontal gaze variance [normalized^2].
    value = safe_variance(Gx);
end


function value = feature_006_variance_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_006_VARIANCE_GAZE_Y Vertical gaze variance [normalized^2].
    value = safe_variance(Gy);
end


function value = feature_007_minimum_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_007_MINIMUM_GAZE_X Minimum horizontal gaze position.
    value = safe_minimum(Gx);
end


function value = feature_008_maximum_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_008_MAXIMUM_GAZE_X Maximum horizontal gaze position.
    value = safe_maximum(Gx);
end


function value = feature_009_range_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_009_RANGE_GAZE_X Horizontal gaze range.
    value = safe_range(Gx);
end


function value = feature_010_minimum_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_010_MINIMUM_GAZE_Y Minimum vertical gaze position.
    value = safe_minimum(Gy);
end


function value = feature_011_maximum_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_011_MAXIMUM_GAZE_Y Maximum vertical gaze position.
    value = safe_maximum(Gy);
end


function value = feature_012_range_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_012_RANGE_GAZE_Y Vertical gaze range.
    value = safe_range(Gy);
end


function value = feature_013_median_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_013_MEDIAN_GAZE_X Median horizontal gaze position.
    value = safe_median(Gx);
end


function value = feature_014_median_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_014_MEDIAN_GAZE_Y Median vertical gaze position.
    value = safe_median(Gy);
end


function value = feature_015_iqr_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_015_IQR_GAZE_X Horizontal gaze interquartile range.
    value = safe_interquartile_range(Gx);
end


function value = feature_016_iqr_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_016_IQR_GAZE_Y Vertical gaze interquartile range.
    value = safe_interquartile_range(Gy);
end


function value = feature_017_mad_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_017_MAD_GAZE_X Median absolute horizontal gaze deviation.
    value = safe_median_absolute_deviation(Gx);
end


function value = feature_018_mad_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_018_MAD_GAZE_Y Median absolute vertical gaze deviation.
    value = safe_median_absolute_deviation(Gy);
end


function value = feature_019_skewness_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_019_SKEWNESS_GAZE_X Horizontal gaze distribution skewness.
    value = safe_skewness(Gx);
end


function value = feature_020_skewness_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_020_SKEWNESS_GAZE_Y Vertical gaze distribution skewness.
    value = safe_skewness(Gy);
end


function value = feature_021_kurtosis_gaze_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_021_KURTOSIS_GAZE_X Horizontal gaze Pearson kurtosis.
    value = safe_kurtosis(Gx);
end


function value = feature_022_kurtosis_gaze_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_022_KURTOSIS_GAZE_Y Vertical gaze Pearson kurtosis.
    value = safe_kurtosis(Gy);
end


function value = feature_023_covariance_gaze_xy(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_023_COVARIANCE_GAZE_XY Sample covariance of gaze x and y.
    value = safe_covariance(Gx, Gy);
end


function value = feature_024_correlation_gaze_xy(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_024_CORRELATION_GAZE_XY Pearson correlation of gaze x and y.
    value = pearson_correlation(Gx, Gy);
end


function value = feature_025_gaze_dispersion_rms(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_025_GAZE_DISPERSION_RMS RMS distance from the gaze centroid.
    center_x = safe_mean(Gx);
    center_y = safe_mean(Gy);
    radius = hypot(Gx - center_x, Gy - center_y);
    value = safe_root_mean_square(radius);
end


function value = feature_026_bcea_68(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_026_BCEA_68 Area of the 68% bivariate gaze ellipse.
    sigma_x = safe_std(Gx);
    sigma_y = safe_std(Gy);
    rho_xy = pearson_correlation(Gx, Gy);
    probability = 0.68;
    ellipse_factor = -log(1.0 - probability);

    if ~isfinite(sigma_x) || ~isfinite(sigma_y) || ~isfinite(rho_xy)
        value = NaN;
        return;
    end

    value = 2.0 * pi * ellipse_factor * sigma_x * sigma_y * ...
        sqrt(max(0.0, 1.0 - rho_xy^2));
end


function value = feature_027_gaze_path_length(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_027_GAZE_PATH_LENGTH Total gaze path length [normalized].
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    value = planar_path_length(x, y);
end


function value = feature_028_gaze_net_displacement(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_028_GAZE_NET_DISPLACEMENT Start-to-end gaze displacement.
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    value = planar_net_displacement(x, y);
end


function value = feature_029_gaze_path_efficiency(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_029_GAZE_PATH_EFFICIENCY Net displacement divided by path length.
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    path_length = planar_path_length(x, y);
    net_displacement = planar_net_displacement(x, y);
    value = safe_ratio(net_displacement, path_length);
end


function value = feature_030_mean_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_030_MEAN_GAZE_SPEED Mean gaze speed [normalized/s].
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(motion.speed);
end


function value = feature_031_std_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_031_STD_GAZE_SPEED Gaze speed standard deviation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_std(motion.speed);
end


function value = feature_032_max_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_032_MAX_GAZE_SPEED Maximum gaze speed.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_maximum(motion.speed);
end


function value = feature_033_median_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_033_MEDIAN_GAZE_SPEED Median gaze speed.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_median(motion.speed);
end


function value = feature_034_p90_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_034_P90_GAZE_SPEED 90th percentile of gaze speed.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = percentile_value(motion.speed, 90.0);
end


function value = feature_035_rms_gaze_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_035_RMS_GAZE_SPEED Root-mean-square gaze speed.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_root_mean_square(motion.speed);
end


function value = feature_036_mean_gaze_acceleration(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_036_MEAN_GAZE_ACCELERATION Mean gaze acceleration magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(motion.acceleration);
end


function value = feature_037_std_gaze_acceleration(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_037_STD_GAZE_ACCELERATION Gaze acceleration standard deviation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_std(motion.acceleration);
end


function value = feature_038_max_gaze_acceleration(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_038_MAX_GAZE_ACCELERATION Maximum gaze acceleration magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_maximum(motion.acceleration);
end


function value = feature_039_rms_gaze_acceleration(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_039_RMS_GAZE_ACCELERATION RMS gaze acceleration magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_root_mean_square(motion.acceleration);
end


function value = feature_040_mean_gaze_jerk(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_040_MEAN_GAZE_JERK Mean gaze jerk magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(motion.jerk);
end


function value = feature_041_std_gaze_jerk(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_041_STD_GAZE_JERK Gaze jerk standard deviation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_std(motion.jerk);
end


function value = feature_042_max_gaze_jerk(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_042_MAX_GAZE_JERK Maximum gaze jerk magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_maximum(motion.jerk);
end


function value = feature_043_rms_gaze_jerk(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_043_RMS_GAZE_JERK RMS gaze jerk magnitude.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_root_mean_square(motion.jerk);
end


function value = feature_044_mean_gaze_velocity_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_044_MEAN_GAZE_VELOCITY_X Mean horizontal gaze velocity.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(motion.velocity_x);
end


function value = feature_045_mean_gaze_velocity_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_045_MEAN_GAZE_VELOCITY_Y Mean vertical gaze velocity.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(motion.velocity_y);
end


function value = feature_046_std_gaze_velocity_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_046_STD_GAZE_VELOCITY_X Horizontal velocity standard deviation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_std(motion.velocity_x);
end


function value = feature_047_std_gaze_velocity_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_047_STD_GAZE_VELOCITY_Y Vertical velocity standard deviation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_std(motion.velocity_y);
end


function value = feature_048_mean_gaze_direction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_048_MEAN_GAZE_DIRECTION Circular mean motion direction [rad].
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = circular_mean(motion.direction);
end


function value = feature_049_gaze_direction_concentration(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_049_GAZE_DIRECTION_CONCENTRATION Circular resultant in [0, 1].
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = circular_concentration(motion.direction);
end


function value = feature_050_mean_absolute_direction_change( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_050_MEAN_ABSOLUTE_DIRECTION_CHANGE Mean turn angle [rad].
    motion = calculate_planar_motion(Gx, Gy, Gt);
    direction_change = atan2( ...
        sin(diff(motion.direction)), cos(diff(motion.direction)));
    value = safe_mean(abs(direction_change));
end


function value = feature_051_low_speed_fraction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_051_LOW_SPEED_FRACTION Fraction below 0.10 normalized/s.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(double(motion.speed < 0.10));
end


function value = feature_052_high_speed_fraction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_052_HIGH_SPEED_FRACTION Fraction above 0.50 normalized/s.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = safe_mean(double(motion.speed > 0.50));
end


function value = feature_053_high_speed_event_count(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_053_HIGH_SPEED_EVENT_COUNT Number of high-speed gaze episodes.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = count_true_runs(motion.speed > 0.50);
end


function value = feature_054_low_speed_event_count(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_054_LOW_SPEED_EVENT_COUNT Number of low-speed gaze episodes.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = count_true_runs(motion.speed < 0.10);
end


function value = feature_055_gaze_missing_time_fraction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_055_GAZE_MISSING_TIME_FRACTION Excess time in gaze-sample gaps.
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    value = missing_time_fraction(time_s);
end


function value = feature_056_gaze_gap_count(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_056_GAZE_GAP_COUNT Count intervals above 1.5 nominal periods.
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    value = sample_gap_count(time_s);
end


function value = feature_057_gaze_sample_rate(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_057_GAZE_SAMPLE_RATE Effective valid-gaze sample rate [Hz].
    [x, y, time_s] = clean_planar_time_series(Gx, Gy, Gt);
    if numel(time_s) < 2 || time_s(end) <= time_s(1)
        value = NaN;
        return;
    end
    value = (numel(time_s) - 1) / (time_s(end) - time_s(1));
end


function value = feature_058_gaze_x_shannon_entropy(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_058_GAZE_X_SHANNON_ENTROPY Entropy of 10 horizontal bins [bits].
    value = one_dimensional_display_entropy(Gx, 10);
end


function value = feature_059_gaze_y_shannon_entropy(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_059_GAZE_Y_SHANNON_ENTROPY Entropy of 10 vertical bins [bits].
    value = one_dimensional_display_entropy(Gy, 10);
end


function value = feature_060_stationary_shannon_entropy( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_060_STATIONARY_SHANNON_ENTROPY 10x10 occupancy entropy [bits].
    states = spatial_gaze_states(Gx, Gy, 10);
    value = shannon_entropy_from_states(states, 100);
end


function value = feature_061_transitional_shannon_entropy( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_061_TRANSITIONAL_SHANNON_ENTROPY Conditional transition entropy.
    states = spatial_gaze_states(Gx, Gy, 10);
    value = transition_shannon_entropy(states, 100);
end


function value = feature_062_mean_distance_from_display_center( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_062_MEAN_DISTANCE_FROM_DISPLAY_CENTER Mean distance from (0.5,0.5).
    center_distance = hypot(Gx - 0.5, Gy - 0.5);
    value = safe_mean(center_distance);
end


function value = feature_063_std_distance_from_display_center( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_063_STD_DISTANCE_FROM_DISPLAY_CENTER Center-distance variation.
    center_distance = hypot(Gx - 0.5, Gy - 0.5);
    value = safe_std(center_distance);
end


function value = feature_064_display_center_fraction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_064_DISPLAY_CENTER_FRACTION Fraction within radius 0.25 of center.
    center_distance = hypot(Gx - 0.5, Gy - 0.5);
    value = safe_mean(double(center_distance <= 0.25));
end


function value = feature_065_outside_display_fraction(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_065_OUTSIDE_DISPLAY_FRACTION Fraction outside normalized display.
    outside = Gx < 0 | Gx > 1 | Gy < 0 | Gy > 1;
    value = safe_mean(double(outside));
end


function value = feature_066_gaze_x_mean_crossing_count( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_066_GAZE_X_MEAN_CROSSING_COUNT Horizontal mean crossings.
    value = mean_crossing_count(Gx);
end


function value = feature_067_gaze_y_mean_crossing_count( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_067_GAZE_Y_MEAN_CROSSING_COUNT Vertical mean crossings.
    value = mean_crossing_count(Gy);
end


function value = feature_068_gaze_x_lag1_autocorrelation( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_068_GAZE_X_LAG1_AUTOCORRELATION Horizontal lag-one correlation.
    value = lag_one_autocorrelation(Gx);
end


function value = feature_069_gaze_y_lag1_autocorrelation( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_069_GAZE_Y_LAG1_AUTOCORRELATION Vertical lag-one correlation.
    value = lag_one_autocorrelation(Gy);
end


function value = feature_070_gaze_speed_lag1_autocorrelation( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_070_GAZE_SPEED_LAG1_AUTOCORRELATION Speed lag-one correlation.
    motion = calculate_planar_motion(Gx, Gy, Gt);
    value = lag_one_autocorrelation(motion.speed);
end


function value = feature_071_mean_relative_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_071_MEAN_RELATIVE_X Mean signed gaze-minus-EE x offset.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(relative_x);
end


function value = feature_072_mean_relative_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_072_MEAN_RELATIVE_Y Mean signed gaze-minus-EE y offset.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(relative_y);
end


function value = feature_073_std_relative_x(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_073_STD_RELATIVE_X Standard deviation of gaze-minus-EE x.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_std(relative_x);
end


function value = feature_074_std_relative_y(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_074_STD_RELATIVE_Y Standard deviation of gaze-minus-EE y.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_std(relative_y);
end


function value = feature_075_mean_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_075_MEAN_GAZE_EE_DISTANCE Mean gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(distance);
end


function value = feature_076_std_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_076_STD_GAZE_EE_DISTANCE Gaze-to-EE distance deviation.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_std(distance);
end


function value = feature_077_median_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_077_MEDIAN_GAZE_EE_DISTANCE Median gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_median(distance);
end


function value = feature_078_max_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_078_MAX_GAZE_EE_DISTANCE Maximum gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_maximum(distance);
end


function value = feature_079_rms_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_079_RMS_GAZE_EE_DISTANCE RMS gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_root_mean_square(distance);
end


function value = feature_080_p90_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_080_P90_GAZE_EE_DISTANCE 90th percentile gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = percentile_value(distance, 90.0);
end


function value = feature_081_min_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_081_MIN_GAZE_EE_DISTANCE Minimum gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_minimum(distance);
end


function value = feature_082_final_gaze_ee_distance(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_082_FINAL_GAZE_EE_DISTANCE Last synchronized gaze-to-EE distance.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    if isempty(distance)
        value = NaN;
    else
        value = distance(end);
    end
end


function value = feature_083_gaze_ee_distance_slope(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_083_GAZE_EE_DISTANCE_SLOPE Linear distance trend [normalized/s].
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    distance = hypot(relative_x, relative_y);
    value = linear_slope(time_s, distance);
end


function value = feature_084_fraction_within_0p05(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_084_FRACTION_WITHIN_0P05 Fraction within 0.05 display units.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(double(distance <= 0.05));
end


function value = feature_085_fraction_within_0p10(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_085_FRACTION_WITHIN_0P10 Fraction within 0.10 display units.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(double(distance <= 0.10));
end


function value = feature_086_fraction_within_0p20(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_086_FRACTION_WITHIN_0P20 Fraction within 0.20 display units.
    distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(double(distance <= 0.20));
end


function value = feature_087_gaze_ee_x_correlation(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_087_GAZE_EE_X_CORRELATION Synchronized gaze/EE x correlation.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = pearson_correlation(gx, ex);
end


function value = feature_088_gaze_ee_y_correlation(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_088_GAZE_EE_Y_CORRELATION Synchronized gaze/EE y correlation.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = pearson_correlation(gy, ey);
end


function value = feature_089_gaze_ee_speed_correlation( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_089_GAZE_EE_SPEED_CORRELATION Gaze/EE speed correlation.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    gaze_motion = calculate_planar_motion(gx, gy, time_s);
    ee_motion = calculate_planar_motion(ex, ey, time_s);
    value = pearson_correlation(gaze_motion.speed, ee_motion.speed);
end


function value = feature_090_relative_path_length(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_090_RELATIVE_PATH_LENGTH Path length of gaze-minus-EE offset.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    value = planar_path_length(relative_x, relative_y);
end


function value = feature_091_mean_relative_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_091_MEAN_RELATIVE_SPEED Mean speed of gaze-minus-EE offset.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    motion = calculate_planar_motion(relative_x, relative_y, time_s);
    value = safe_mean(motion.speed);
end


function value = feature_092_std_relative_speed(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_092_STD_RELATIVE_SPEED Relative-speed standard deviation.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    motion = calculate_planar_motion(relative_x, relative_y, time_s);
    value = safe_std(motion.speed);
end


function value = feature_093_mean_velocity_cosine_similarity( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_093_MEAN_VELOCITY_COSINE_SIMILARITY Mean motion alignment [-1,1].
    cosine_similarity = gaze_ee_velocity_cosine(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(cosine_similarity);
end


function value = feature_094_aligned_velocity_fraction( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_094_ALIGNED_VELOCITY_FRACTION Fraction with positive alignment.
    cosine_similarity = gaze_ee_velocity_cosine(Gx, Gy, Gt, Ex, Ey, Et);
    value = safe_mean(double(cosine_similarity > 0));
end


function value = feature_095_gaze_to_ee_speed_ratio(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_095_GAZE_TO_EE_SPEED_RATIO Mean gaze speed divided by mean EE speed.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    gaze_motion = calculate_planar_motion(gx, gy, time_s);
    ee_motion = calculate_planar_motion(ex, ey, time_s);
    value = safe_ratio( ...
        safe_mean(gaze_motion.speed), safe_mean(ee_motion.speed));
end


function value = feature_096_max_x_cross_correlation(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_096_MAX_X_CROSS_CORRELATION Best x correlation within +/-0.25 s.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    [value, best_lag_s] = best_lagged_correlation(gx, ex, time_s, 0.25);
end


function value = feature_097_x_lead_at_max_correlation_s( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_097_X_LEAD_AT_MAX_CORRELATION_S Positive means gaze leads EE [s].
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    [best_correlation, value] = best_lagged_correlation( ...
        gx, ex, time_s, 0.25);
end


function value = feature_098_max_y_cross_correlation(Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_098_MAX_Y_CROSS_CORRELATION Best y correlation within +/-0.25 s.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    [value, best_lag_s] = best_lagged_correlation(gy, ey, time_s, 0.25);
end


function value = feature_099_y_lead_at_max_correlation_s( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_099_Y_LEAD_AT_MAX_CORRELATION_S Positive means gaze leads EE [s].
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    [best_correlation, value] = best_lagged_correlation( ...
        gy, ey, time_s, 0.25);
end


function value = feature_100_mean_gaze_lead_along_ee_motion( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_100_MEAN_GAZE_LEAD_ALONG_EE_MOTION Signed along-path gaze lead.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    ee_motion = calculate_planar_motion(ex, ey, time_s);
    ee_speed = ee_motion.speed;
    moving = ee_speed > 1e-9;

    along_path_lead = NaN(size(ee_speed));
    along_path_lead(moving) = ( ...
        (gx(moving) - ex(moving)) .* ee_motion.velocity_x(moving) + ...
        (gy(moving) - ey(moving)) .* ee_motion.velocity_y(moving)) ./ ...
        ee_speed(moving);
    value = safe_mean(along_path_lead);
end


function value = feature_101_stationary_shannon_entropy_40x40( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_101_STATIONARY_SHANNON_ENTROPY_40X40 Occupancy entropy [bits].
    states = spatial_gaze_states(Gx, Gy, 40);
    value = shannon_entropy_from_states(states, 1600);
end


function value = feature_102_transitional_shannon_entropy_40x40( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%FEATURE_102_TRANSITIONAL_SHANNON_ENTROPY_40X40 Transition entropy [bits].
    states = spatial_gaze_states(Gx, Gy, 40);
    value = transition_shannon_entropy(states, 1600);
end

%% Data preparation and outcome helpers
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


function [Gx, Gy] = combine_binocular_gaze( ...
        left_gaze, right_gaze, left_valid, right_valid)
%COMBINE_BINOCULAR_GAZE Average all valid eyes at each Tobii timestamp.
    left_usable = left_valid & all(isfinite(left_gaze), 2);
    right_usable = right_valid & all(isfinite(right_gaze), 2);
    eye_count = double(left_usable) + double(right_usable);

    gaze_sum_x = zeros(size(eye_count));
    gaze_sum_y = zeros(size(eye_count));
    gaze_sum_x(left_usable) = left_gaze(left_usable, 1);
    gaze_sum_y(left_usable) = left_gaze(left_usable, 2);
    gaze_sum_x(right_usable) = gaze_sum_x(right_usable) + ...
        right_gaze(right_usable, 1);
    gaze_sum_y(right_usable) = gaze_sum_y(right_usable) + ...
        right_gaze(right_usable, 2);

    Gx = NaN(size(eye_count));
    Gy = NaN(size(eye_count));
    at_least_one_eye = eye_count > 0;
    Gx(at_least_one_eye) = gaze_sum_x(at_least_one_eye) ./ ...
        eye_count(at_least_one_eye);
    Gy(at_least_one_eye) = gaze_sum_y(at_least_one_eye) ./ ...
        eye_count(at_least_one_eye);
end


function [normalized_values, limits_used] = map_to_unit_interval( ...
        values, requested_limits)
%MAP_TO_UNIT_INTERVAL Affinely map physical coordinates to [0, 1].
    finite_data = values(isfinite(values));
    if isempty(finite_data)
        error('Cannot map an empty coordinate vector to display space.');
    end

    if isempty(requested_limits)
        limits_used = [min(finite_data), max(finite_data)];
    else
        limits_used = double(requested_limits(:))';
    end

    if numel(limits_used) ~= 2 || any(~isfinite(limits_used)) || ...
            limits_used(2) <= limits_used(1)
        error('Display limits must be finite [minimum, maximum] values.');
    end

    coordinate_range = limits_used(2) - limits_used(1);
    normalized_values = (values - limits_used(1)) / coordinate_range;
end


function dsj = calculate_planar_dsj(x_m, y_m, time_s)
%CALCULATE_PLANAR_DSJ Dimensionless squared jerk over one window.
%
% DSJ = T^5 / L^2 * integral(||jerk(t)||^2 dt), where T is window
% duration and L is planar path length. Lower DSJ indicates smoother motion.
    motion = calculate_planar_motion(x_m, y_m, time_s);
    duration_s = safe_range(motion.time_s);
    path_length_m = planar_path_length(motion.x, motion.y);

    if numel(motion.time_s) < 4 || ~isfinite(duration_s) || ...
            duration_s <= 0 || ~isfinite(path_length_m) || ...
            path_length_m <= 1e-12 || any(~isfinite(motion.jerk))
        dsj = NaN;
        return;
    end

    integrated_squared_jerk = trapz( ...
        motion.time_s, motion.jerk.^2);
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


function motion = calculate_planar_motion(x, y, time_s)
%CALCULATE_PLANAR_MOTION Differentiate a planar trajectory in time.
    [x, y, time_s] = clean_planar_time_series(x, y, time_s);
    sample_count = numel(time_s);

    motion.x = x;
    motion.y = y;
    motion.time_s = time_s;
    motion.velocity_x = NaN(sample_count, 1);
    motion.velocity_y = NaN(sample_count, 1);
    motion.speed = NaN(sample_count, 1);
    motion.acceleration = NaN(sample_count, 1);
    motion.jerk = NaN(sample_count, 1);
    motion.direction = NaN(sample_count, 1);

    if sample_count < 2
        return;
    end

    velocity_x = gradient(x, time_s);
    velocity_y = gradient(y, time_s);
    speed = hypot(velocity_x, velocity_y);
    motion.velocity_x = velocity_x;
    motion.velocity_y = velocity_y;
    motion.speed = speed;
    moving = speed > 1e-12;
    motion.direction(moving) = atan2(velocity_y(moving), velocity_x(moving));

    if sample_count < 3
        return;
    end

    acceleration_x = gradient(velocity_x, time_s);
    acceleration_y = gradient(velocity_y, time_s);
    motion.acceleration = hypot(acceleration_x, acceleration_y);

    if sample_count < 4
        return;
    end

    jerk_x = gradient(acceleration_x, time_s);
    jerk_y = gradient(acceleration_y, time_s);
    motion.jerk = hypot(jerk_x, jerk_y);
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


function value = one_dimensional_display_entropy(values, bin_count)
%ONE_DIMENSIONAL_DISPLAY_ENTROPY Shannon entropy of fixed display bins.
    values = finite_values(values);
    if isempty(values)
        value = NaN;
        return;
    end

    clipped_values = min(max(values, 0.0), 1.0 - 1e-12);
    states = floor(clipped_values * bin_count) + 1;
    value = shannon_entropy_from_states(states, bin_count);
end


function states = spatial_gaze_states(Gx, Gy, bin_count_per_axis)
%SPATIAL_GAZE_STATES Convert normalized gaze into display-grid state IDs.
    Gx = double(Gx(:));
    Gy = double(Gy(:));
    valid = isfinite(Gx) & isfinite(Gy);
    Gx = min(max(Gx(valid), 0.0), 1.0 - 1e-12);
    Gy = min(max(Gy(valid), 0.0), 1.0 - 1e-12);

    x_bin = floor(Gx * bin_count_per_axis) + 1;
    y_bin = floor(Gy * bin_count_per_axis) + 1;
    states = x_bin + (y_bin - 1) * bin_count_per_axis;
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


function [gx, gy, ex, ey, time_s] = ...
        synchronize_gaze_and_end_effector(Gx, Gy, Gt, Ex, Ey, Et)
%SYNCHRONIZE_GAZE_AND_END_EFFECTOR Interpolate EE position at gaze times.
    [gx, gy, gaze_time_s] = clean_planar_time_series(Gx, Gy, Gt);
    [ex_source, ey_source, ee_time_s] = ...
        clean_planar_time_series(Ex, Ey, Et);

    if isempty(gaze_time_s) || numel(ee_time_s) < 2
        gx = [];
        gy = [];
        ex = [];
        ey = [];
        time_s = [];
        return;
    end

    in_overlap = gaze_time_s >= ee_time_s(1) & ...
        gaze_time_s <= ee_time_s(end);
    gx = gx(in_overlap);
    gy = gy(in_overlap);
    time_s = gaze_time_s(in_overlap);
    ex = interp1(ee_time_s, ex_source, time_s, 'linear');
    ey = interp1(ee_time_s, ey_source, time_s, 'linear');

    valid = isfinite(gx) & isfinite(gy) & ...
        isfinite(ex) & isfinite(ey) & isfinite(time_s);
    gx = gx(valid);
    gy = gy(valid);
    ex = ex(valid);
    ey = ey(valid);
    time_s = time_s(valid);
end


function [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%RELATIVE_OFFSETS Synchronized gaze-minus-end-effector displacement.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    relative_x = gx - ex;
    relative_y = gy - ey;
end


function distance = gaze_end_effector_distance(Gx, Gy, Gt, Ex, Ey, Et)
%GAZE_END_EFFECTOR_DISTANCE Synchronized gaze-to-EE Euclidean distance.
    [relative_x, relative_y, time_s] = relative_offsets( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    distance = hypot(relative_x, relative_y);
end


function value = linear_slope(time_s, values)
%LINEAR_SLOPE Least-squares signal slope with respect to time.
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


function cosine_similarity = gaze_ee_velocity_cosine( ...
        Gx, Gy, Gt, Ex, Ey, Et)
%GAZE_EE_VELOCITY_COSINE Cosine similarity of gaze and EE velocities.
    [gx, gy, ex, ey, time_s] = synchronize_gaze_and_end_effector( ...
        Gx, Gy, Gt, Ex, Ey, Et);
    gaze_motion = calculate_planar_motion(gx, gy, time_s);
    ee_motion = calculate_planar_motion(ex, ey, time_s);

    speed_product = gaze_motion.speed .* ee_motion.speed;
    valid = isfinite(speed_product) & speed_product > 1e-12;
    cosine_similarity = ( ...
        gaze_motion.velocity_x(valid) .* ee_motion.velocity_x(valid) + ...
        gaze_motion.velocity_y(valid) .* ee_motion.velocity_y(valid)) ./ ...
        speed_product(valid);
    cosine_similarity = min(max(cosine_similarity, -1.0), 1.0);
end


function [best_correlation, best_lag_s] = best_lagged_correlation( ...
        gaze_values, ee_values, time_s, maximum_lag_s)
%BEST_LAGGED_CORRELATION Find best correlation over positive/negative lags.
%
% A positive returned lag means an earlier gaze sample best matches a later
% end-effector sample; therefore, positive lag indicates gaze leading EE.
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
