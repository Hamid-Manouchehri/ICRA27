%% Compare a gaze-assisted trial against the Chen-benchmark trial.
%
% Compares scripts/ur5e_gaze_assisted_adm_2d.py against
% scripts/ur5e_chen_var_adm_2d.py (Chen et al. 2024) on the same
% "chen_shape" task, using the SAME paper-metric formulas (Eqs 15-17)
% computed identically for both, so the numbers are directly comparable.
%
% Inputs:
%   data/raw_data/ur5e_gaze_assisted_adm_2d_<trial>.csv   one CSV only:
%     this controller has no separate Tobii log -- it detects fixations
%     in real time and writes only the derived fixation/intent columns.
%   data/raw_data/ur5e_chen_var_adm_2d_<trial>.csv        controller log
%   data/raw_data/tobii_chen_var_adm_2d_<trial>.csv       matching gaze
%     log; Chen's fixations are I-VT-detected here from raw gaze, since
%     that controller does not do gaze-based intent estimation at all.
%   data/calibrations/<file>.json    Chen's Tobii-to-CoppeliaSim mapping.
%   data/shapes/<shape>.obj          optional, accuracy metric.
%   All five are selected in User Settings.
%
% Output:
%   Figure 1: TCP path in the CoppeliaSim x-y frame for each controller,
%   side by side, colored by interaction force magnitude, with each
%   controller's own detected fixations overlaid.
%   Figure 2: effort, paper Eq. (17). Force magnitude over time for both
%   (overlaid), and a bar comparison of mean force.
%   Figure 3: smoothness, paper Eq. (16). Jerk magnitude over time for
%   both (overlaid), and a bar comparison of dimensionless squared jerk
%   at the same smoothing window (only DSJ values computed with the same
%   window are comparable).
%   Figure 4: activity of each controller's own adaptive-damping signal
%   over time -- gaze-assisted's total_influence versus Chen's
%   guidance_weight. Both are dimensionless [0, 1] "how much is the
%   anisotropic-damping mechanism currently trusted" signals, but they
%   are computed from entirely different inputs (real eye-gaze fixation
%   confidence and force-intent alignment, versus a velocity-curvature
%   power-law fit quality and turn evidence) -- equal values mean
%   "equally confident by that method's own criteria," not "equally
%   correct." Do not read this as a like-for-like control signal.
%   Accuracy, paper Eq. (15), is attempted from the shape mesh but is
%   not available for chen_shape.obj: it resolves to three boundary
%   loops (one outer, two separate inner cutouts), not the two-loop
%   single-cutout band this project's shape-mesh reader assumes -- this
%   most likely reflects the paper's own task of a circle segment and a
%   square segment joined by straight lines, each with its own inner
%   edge, rather than a mesh export error. read_shape_band_obj() already
%   degrades gracefully (skips with a warning) when it does not find
%   exactly two loops, so this script still runs; it is unaffected for
%   any future shape mesh that IS a simple single-cutout band.
%   A numeric side-by-side summary is printed to the command window.
%   No file is written; this script only reads raw CSVs, JSON and OBJ.
%
% Method reuse:
%   I-VT fixation detection, eye combination, UCS-to-CoppeliaSim
%   mapping, the shape-band OBJ reader, and the jerk/DSJ chain are
%   copied verbatim from plot_raw_data_chen_var_adm_2d.m, self-contained
%   here per this project's convention that MATLAB scripts do not call
%   one another.
%
% Assumptions:
%   Both CSVs were written by their respective scripts, so column names
%   match those scripts' write_csv_header(). scripts/ur5e_gaze_assisted_
%   adm_2d.py logs TCP position in the robot base frame only
%   (tcp_x_m/tcp_y_m), not CoppeliaSim; it is mapped here with the same
%   coppeliasim_axis_from_robot/coppeliasim_sign_from_robot/
%   coppeliasim_offset_xy_m transform that script itself uses (default
%   identity, matching its current settings) -- VERIFY this matches
%   whatever that script's settings were at the time chen_shape_test was
%   recorded. Fixation-detection thresholds differ between the two
%   controllers (this script's Chen I-VT settings versus that script's
%   own fixation_velocity_threshold_deg_s/minimum_fixation_duration_s,
%   30 deg/s and 0.10 s by its current defaults), so fixation COUNTS are
%   not directly comparable even though both represent "detected
%   fixations" -- noted again in the printed summary.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO gaze-assisted controller log (single CSV, no separate gaze log).
gaze_assisted_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'ur5e_gaze_assisted_adm_2d_chen_shape_test.csv');

% TODO Chen controller log and its matching raw gaze log.
chen_robot_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'ur5e_chen_var_adm_2d_chen_shape_test.csv');
chen_gaze_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'tobii_chen_var_adm_2d_chen_shape_test.csv');

% TODO calibration JSON for Chen's gaze mapping (Tobii UCS to
% CoppeliaSim). The gaze-assisted controller needs no calibration here:
% its CSV already carries fixation_x_robot_m/fixation_y_robot_m, mapped
% by that script itself at record time.
calibration_json_file = fullfile( ...
    script_directory, '..', 'data', 'calibrations', ...
    'tobii_coppelia_reference_calibration_chen_shape.json');

% TODO shape mesh both operators traced, used only for the accuracy
% metric (attempted, but unavailable for chen_shape.obj -- see the
% header comment above).
shape_obj_file = fullfile( ...
    script_directory, '..', 'data', 'shapes', 'chen_shape.obj');

% TODO translation from the mesh CAD frame to the CoppeliaSim frame [m].
% Only used if accuracy becomes available (see above).
shape_offset_xy_m = [0.0, -0.65];

% TODO spacing used to resample the reference path [m], positive.
reference_spacing_m = 5e-4;

% TODO I-VT angular-velocity threshold [deg/s] and minimum retained
% fixation duration [s], used ONLY for Chen's raw-gaze fixation
% detection. Matches plot_raw_data_chen_var_adm_2d.m /
% compute_plot_similarity_fixation_direction_force_vel.m.
velocity_threshold_deg_s = 20.0;
minimum_fixation_duration_s = 0.060;

% TODO true excludes gaze estimates outside the normalized display,
% used only for Chen's raw-gaze fixation detection.
plot_only_on_display_gaze = true;

% TODO physical screen motion / robot motion [unitless], must be > 0,
% and the Tobii UCS axis signs in CoppeliaSim, each -1 or +1. Used only
% for Chen's raw-gaze fixation detection.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO gaze-assisted TCP-to-CoppeliaSim mapping, matching
% scripts/ur5e_gaze_assisted_adm_2d.py's coppeliasim_axis_from_robot /
% coppeliasim_sign_from_robot / coppeliasim_offset_xy_m. Given here
% 1-based (MATLAB), so that script's 0-based [0, 1] identity is [1, 2]
% here. Default is that script's current identity settings.
gaze_assisted_axis_from_robot = [1, 2];
gaze_assisted_sign_from_robot = [1.0, 1.0];
gaze_assisted_offset_xy_m = [0.0, 0.0];

% TODO speed separating movement from rest [m/s]. The paper uses 0.005
% and evaluates every performance metric on movement samples only.
moving_speed_threshold_m_s = 0.005;

% TODO filter windows for the jerk chain used by the smoothness metric
% [s]. Matches plot_raw_data_chen_var_adm_2d.m /
% plot_raw_data_fixed_adm_2d.m so DSJ values are comparable with those
% scripts' output, not just between the two trials here.
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;
jerk_sensitivity_windows_s = [0.05, 0.10, 0.20];

% TODO marker sizes [points^2] and colors for the spatial panels.
trajectory_marker_size = 10;
fixation_marker_size = 40;
gaze_assisted_color = [0.00, 0.45, 0.74];
chen_color = [0.85, 0.20, 0.10];

%% Load the gaze-assisted trial
ga_table = readtable(gaze_assisted_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

ga_time_s = ga_table.time_s;
ga_force_x_n = ga_table.Fx_filtered_N;
ga_force_y_n = ga_table.Fy_filtered_N;
ga_force_magnitude_n = hypot(ga_force_x_n, ga_force_y_n);
ga_speed_m_s = hypot(ga_table.tcp_vx_actual_m_s, ga_table.tcp_vy_actual_m_s);
ga_total_influence = ga_table.total_influence;

ga_tcp_xy_robot_m = [ga_table.tcp_x_m, ga_table.tcp_y_m];
ga_tcp_xy_m = map_robot_xy_to_coppeliasim( ...
    ga_tcp_xy_robot_m, gaze_assisted_axis_from_robot, ...
    gaze_assisted_sign_from_robot, gaze_assisted_offset_xy_m);

[ga_fixation_xy_robot_m, ga_fixation_time_s, ga_fixation_duration_s] = ...
    extract_distinct_gaze_assisted_fixations( ...
    ga_time_s, get_numeric_column(ga_table.fixation_x_robot_m), ...
    get_numeric_column(ga_table.fixation_y_robot_m), ...
    get_numeric_column(ga_table.fixation_duration_s), ...
    get_numeric_column(ga_table.fixation_valid) == 1);
ga_fixation_xy_m = map_robot_xy_to_coppeliasim( ...
    ga_fixation_xy_robot_m, gaze_assisted_axis_from_robot, ...
    gaze_assisted_sign_from_robot, gaze_assisted_offset_xy_m);

%% Load the Chen trial
chen_table = readtable(chen_robot_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

chen_time_s = chen_table.time_s;
chen_force_x_n = chen_table.Fx_filt_N;
chen_force_y_n = chen_table.Fy_filt_N;
chen_force_magnitude_n = hypot(chen_force_x_n, chen_force_y_n);
chen_speed_m_s = hypot( ...
    chen_table.tcp_vx_actual_m_s, chen_table.tcp_vy_actual_m_s);
chen_guidance_weight = chen_table.guidance_weight;
chen_tcp_xy_m = [chen_table.coppeliasim_x_m, chen_table.coppeliasim_y_m];

chen_gaze_table = readtable(chen_gaze_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
calibration_reference = jsondecode(fileread(calibration_json_file));
tobii_reference_xy_mm = double( ...
    calibration_reference.manual_tobii_reference_xy_mm(:)');
coppelia_reference_xy_m = double( ...
    calibration_reference.manual_coppelia_reference_xy_m(:)');

[chen_fixation_xy_m, ~, chen_fixation_time_s] = ...
    detect_fixations_coppelia_xy( ...
    chen_gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
    screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
    velocity_threshold_deg_s, minimum_fixation_duration_s, ...
    plot_only_on_display_gaze);

%% Shared metrics: paper performance metrics, Section III.C
% Movement samples only, as the paper does.

ga_moving_mask = ga_speed_m_s > moving_speed_threshold_m_s;
chen_moving_mask = chen_speed_m_s > moving_speed_threshold_m_s;

ga_mean_force_n = mean(ga_force_magnitude_n(ga_moving_mask));
chen_mean_force_n = mean(chen_force_magnitude_n(chen_moving_mask));

ga_jerk = compute_jerk_series(ga_time_s, ga_tcp_xy_m(:, 1), ...
    ga_tcp_xy_m(:, 2), jerk_outlier_window_s, jerk_smoothing_window_s);
chen_jerk = compute_jerk_series(chen_time_s, chen_tcp_xy_m(:, 1), ...
    chen_tcp_xy_m(:, 2), jerk_outlier_window_s, jerk_smoothing_window_s);

ga_dsj = compute_dsj( ...
    ga_jerk.time_s, ga_jerk.squared_jerk_m2_s6, ga_jerk.path_length_m);
chen_dsj = compute_dsj(chen_jerk.time_s, ...
    chen_jerk.squared_jerk_m2_s6, chen_jerk.path_length_m);

ga_dsj_sensitivity = zeros(numel(jerk_sensitivity_windows_s), 1);
chen_dsj_sensitivity = zeros(numel(jerk_sensitivity_windows_s), 1);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    ga_sensitivity_jerk = compute_jerk_series(ga_time_s, ...
        ga_tcp_xy_m(:, 1), ga_tcp_xy_m(:, 2), jerk_outlier_window_s, ...
        jerk_sensitivity_windows_s(window_idx));
    ga_dsj_sensitivity(window_idx) = compute_dsj(ga_sensitivity_jerk.time_s, ...
        ga_sensitivity_jerk.squared_jerk_m2_s6, ...
        ga_sensitivity_jerk.path_length_m);

    chen_sensitivity_jerk = compute_jerk_series(chen_time_s, ...
        chen_tcp_xy_m(:, 1), chen_tcp_xy_m(:, 2), jerk_outlier_window_s, ...
        jerk_sensitivity_windows_s(window_idx));
    chen_dsj_sensitivity(window_idx) = compute_dsj( ...
        chen_sensitivity_jerk.time_s, ...
        chen_sensitivity_jerk.squared_jerk_m2_s6, ...
        chen_sensitivity_jerk.path_length_m);
end

%% Accuracy, paper Eq. (15) -- attempted, see header comment
accuracy_available = false;

if isfile(shape_obj_file)
    [~, ~, reference_xy_m] = ...
        read_shape_band_obj(shape_obj_file, reference_spacing_m);

    if isempty(reference_xy_m)
        warning(['Shape mesh did not yield a simple two-loop band ' ...
            '(chen_shape.obj has an outer boundary plus two separate ' ...
            'inner cutouts, likely the circle and square task ' ...
            'segments), so accuracy is skipped. See the header ' ...
            'comment for detail.']);
    else
        reference_xy_m = reference_xy_m + shape_offset_xy_m;
        ga_tracking_error_m = distance_to_reference_path( ...
            ga_tcp_xy_m(:, 1), ga_tcp_xy_m(:, 2), reference_xy_m);
        chen_tracking_error_m = distance_to_reference_path( ...
            chen_tcp_xy_m(:, 1), chen_tcp_xy_m(:, 2), reference_xy_m);
        accuracy_available = true;
    end
else
    warning('Shape mesh not found; accuracy is skipped.');
end

%% ================================================================
% Figure 1: TCP path comparison, colored by interaction force
% =================================================================

figure('Name', 'TCP path comparison', 'Color', 'w');
tile_layout = tiledlayout(1, 2, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

force_color_limits_n = [0, max( ...
    max(ga_force_magnitude_n), max(chen_force_magnitude_n))];

nexttile;
hold on;
scatter(ga_tcp_xy_m(:, 1), ga_tcp_xy_m(:, 2), trajectory_marker_size, ...
    ga_force_magnitude_n, 'filled', 'DisplayName', 'TCP path');
scatter(ga_fixation_xy_m(:, 1), ga_fixation_xy_m(:, 2), ...
    fixation_marker_size, gaze_assisted_color, 'filled', ...
    'MarkerFaceAlpha', 0.7, 'DisplayName', 'Fixations');
axis equal;
grid on;
clim(force_color_limits_n);
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title(sprintf('Gaze-assisted (%d fixations)', numel(ga_fixation_time_s)));
legend('Location', 'best', 'FontSize', 8);

nexttile;
hold on;
scatter(chen_tcp_xy_m(:, 1), chen_tcp_xy_m(:, 2), trajectory_marker_size, ...
    chen_force_magnitude_n, 'filled', 'DisplayName', 'TCP path');
scatter(chen_fixation_xy_m(:, 1), chen_fixation_xy_m(:, 2), ...
    fixation_marker_size, chen_color, 'filled', 'MarkerFaceAlpha', 0.7, ...
    'DisplayName', 'Fixations');
axis equal;
grid on;
clim(force_color_limits_n);
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title(sprintf('Chen benchmark (%d fixations)', numel(chen_fixation_time_s)));
legend('Location', 'best', 'FontSize', 8);

force_colorbar = colorbar;
force_colorbar.Layout.Tile = 'east';
force_colorbar.Label.String = 'Interaction force |F| [N]';
colormap;
sgtitle(['TCP path colored by interaction force, both controllers ' ...
    '(shared color axis)']);

%% ================================================================
% Figure 2: Effort comparison, paper Eq. (17)
% =================================================================

figure('Name', 'Effort comparison', 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plot(ga_time_s, ga_force_magnitude_n, '-', 'Color', gaze_assisted_color, ...
    'LineWidth', 0.9, 'DisplayName', 'Gaze-assisted');
plot(chen_time_s, chen_force_magnitude_n, '-', 'Color', chen_color, ...
    'LineWidth', 0.9, 'DisplayName', 'Chen benchmark');
grid on;
xlabel('Time [s]');
ylabel('|F| [N]');
title('Interaction force magnitude over time');
legend('Location', 'best');

nexttile;
bar_labels = categorical({'Gaze-assisted', 'Chen benchmark'});
bar_labels = reordercats(bar_labels, {'Gaze-assisted', 'Chen benchmark'});
bar(bar_labels, [ga_mean_force_n, chen_mean_force_n], ...
    'FaceColor', 'flat', 'CData', [gaze_assisted_color; chen_color]);
grid on;
ylabel('Mean |F| [N]');
title('Mean force, Eq. (17) (moving samples only)');

sgtitle('Effort, paper Eq. (17)');

%% ================================================================
% Figure 3: Smoothness comparison, paper Eq. (16)
% =================================================================

figure('Name', 'Smoothness comparison', 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plot(ga_jerk.time_s, sqrt(ga_jerk.squared_jerk_m2_s6), '-', ...
    'Color', gaze_assisted_color, 'LineWidth', 0.9, ...
    'DisplayName', 'Gaze-assisted');
plot(chen_jerk.time_s, sqrt(chen_jerk.squared_jerk_m2_s6), '-', ...
    'Color', chen_color, 'LineWidth', 0.9, 'DisplayName', 'Chen benchmark');
grid on;
xlabel('Time [s]');
ylabel('|jerk| [m/s^3]');
title(sprintf('Jerk magnitude (smoothing window %.0f ms)', ...
    1e3 * jerk_smoothing_window_s));
legend('Location', 'best');

nexttile;
bar(bar_labels, [ga_dsj, chen_dsj], 'FaceColor', 'flat', ...
    'CData', [gaze_assisted_color; chen_color]);
grid on;
ylabel('DSJ [-]');
title(sprintf('DSJ, Eq. (16), at %.0f ms window', ...
    1e3 * jerk_smoothing_window_s));

sgtitle(['Smoothness, paper Eq. (16) (DSJ is filter-dependent; only ' ...
    'compare trials at the same window)']);

%% ================================================================
% Figure 4: adaptive-damping-signal activity comparison
% =================================================================
% NOT a like-for-like control signal -- see the header comment. Shown
% together only because both are dimensionless [0, 1] "how confident is
% this controller in its own anisotropic-damping trigger" signals.

figure('Name', 'Adaptive-damping activity comparison', 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(ga_time_s, ga_total_influence, '-', 'Color', gaze_assisted_color, ...
    'LineWidth', 1.0);
grid on;
ylim([-0.05, 1.05]);
ylabel('total\_influence [-]');
title(sprintf(['Gaze-assisted: gaze-intent confidence x force-' ...
    'alignment weight x distance weight (mean %.3f)'], ...
    mean(ga_total_influence)));

nexttile;
plot(chen_time_s, chen_guidance_weight, '-', 'Color', chen_color, ...
    'LineWidth', 1.0);
grid on;
ylim([-0.05, 1.05]);
xlabel('Time [s]');
ylabel('guidance\_weight [-]');
title(sprintf(['Chen: power-law fit quality x speed gate x turn ' ...
    'confidence (mean %.3f)'], mean(chen_guidance_weight)));

sgtitle(['Adaptive-damping signal activity (different underlying ' ...
    'computations -- see header comment)']);

%% ================================================================
% Summary
% =================================================================

ga_duration_s = ga_time_s(end) - ga_time_s(1);
chen_duration_s = chen_time_s(end) - chen_time_s(1);
ga_path_length_m = sum(hypot( ...
    diff(ga_tcp_xy_m(:, 1)), diff(ga_tcp_xy_m(:, 2))));
chen_path_length_m = sum(hypot( ...
    diff(chen_tcp_xy_m(:, 1)), diff(chen_tcp_xy_m(:, 2))));

fprintf('\n--- Gaze-assisted vs. Chen benchmark, chen_shape task ---\n');
fprintf('%-28s %14s %14s\n', '', 'Gaze-assisted', 'Chen benchmark');
fprintf('%-28s %14.2f %14.2f\n', 'Duration [s]', ...
    ga_duration_s, chen_duration_s);
fprintf('%-28s %14d %14d\n', 'Samples', height(ga_table), height(chen_table));
fprintf('%-28s %14.1f %14.1f\n', 'Mean rate [Hz]', ...
    (height(ga_table) - 1) / ga_duration_s, ...
    (height(chen_table) - 1) / chen_duration_s);
fprintf('%-28s %14.4f %14.4f\n', 'Path length [m]', ...
    ga_path_length_m, chen_path_length_m);
fprintf('%-28s %14.1f %14.1f\n', 'Moving samples [%%]', ...
    100 * mean(ga_moving_mask), 100 * mean(chen_moving_mask));
fprintf('%-28s %14.4f %14.4f\n', 'Mean speed [m/s]', ...
    mean(ga_speed_m_s), mean(chen_speed_m_s));
fprintf('\n');
fprintf('%-28s %14.3f %14.3f\n', 'Mean force Eq.(17) [N]', ...
    ga_mean_force_n, chen_mean_force_n);
fprintf('%-28s %14.4g %14.4g\n', 'DSJ Eq.(16) [-]', ga_dsj, chen_dsj);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    fprintf('  at %3.0f ms window %14.4g %14.4g\n', ...
        1e3 * jerk_sensitivity_windows_s(window_idx), ...
        ga_dsj_sensitivity(window_idx), chen_dsj_sensitivity(window_idx));
end
fprintf(['  DSJ is filter-dependent; only compare at the same ' ...
    'window.\n']);

if accuracy_available
    fprintf('%-28s %14.2f %14.2f\n', 'Mean error Eq.(15) [mm]', ...
        1e3 * mean(ga_tracking_error_m(ga_moving_mask)), ...
        1e3 * mean(chen_tracking_error_m(chen_moving_mask)));
else
    fprintf(['%-28s %14s %14s\n'], 'Accuracy Eq.(15)', ...
        'unavailable', 'unavailable');
    fprintf(['  chen_shape.obj has 3 boundary loops (outer + 2 ' ...
        'separate inner cutouts), not the single-cutout band this ' ...
        'reader assumes -- see header comment.\n']);
end

fprintf('\n');
fprintf('%-28s %14.3f %14.3f\n', 'Mean adaptive signal [-]', ...
    mean(ga_total_influence), mean(chen_guidance_weight));
fprintf('%-28s %14.1f %14.1f\n', 'Signal > 0.5 [%%]', ...
    100 * mean(ga_total_influence > 0.5), ...
    100 * mean(chen_guidance_weight > 0.5));
fprintf(['  Different underlying computations (real gaze vs. power-' ...
    'law fit); not a like-for-like comparison, see header comment.\n']);

fprintf('\n');
fprintf('%-28s %14d %14d\n', 'Detected fixations', ...
    numel(ga_fixation_time_s), numel(chen_fixation_time_s));
fprintf(['  Different fixation-detection thresholds between the two ' ...
    'controllers -- counts are not directly comparable, see header ' ...
    'comment.\n']);

fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m (itself from
% compute_plot_similarity_fixation_direction_force_vel.m), per this
% project's convention that MATLAB scripts do not call one another,
% plus two new functions specific to comparing these two controllers'
% differently-shaped CSVs.

function coppelia_xy_m = map_robot_xy_to_coppeliasim( ...
        robot_xy_m, axis_from_robot_1based, sign_from_robot, offset_xy_m)
%MAP_ROBOT_XY_TO_COPPELIASIM Port of scripts/ur5e_gaze_assisted_adm_2d.py's
% map_tcp_position_to_coppeliasim(), planar only.
%
% axis_from_robot_1based(c) names which robot column (1=x, 2=y) becomes
% CoppeliaSim column c -- that script's 0-based coppeliasim_axis_from_robot
% shifted by one for MATLAB indexing. [1, 2] (the default here) is the
% identity permutation matching that script's current settings.

    sample_count = size(robot_xy_m, 1);
    coppelia_xy_m = zeros(sample_count, 2);

    for coppelia_axis = 1:2
        robot_axis = axis_from_robot_1based(coppelia_axis);
        coppelia_xy_m(:, coppelia_axis) = offset_xy_m(coppelia_axis) + ...
            sign_from_robot(coppelia_axis) * robot_xy_m(:, robot_axis);
    end
end


function [fixation_xy_m, fixation_time_s, fixation_duration_s] = ...
        extract_distinct_gaze_assisted_fixations( ...
        time_s, fixation_x_m, fixation_y_m, fixation_duration_s_column, ...
        fixation_valid)
%EXTRACT_DISTINCT_GAZE_ASSISTED_FIXATIONS Collapse a continuously
% refining running-mean fixation stream into one row per fixation.
%
% scripts/ur5e_gaze_assisted_adm_2d.py updates fixation_x/y_m and
% fixation_duration_s on every new Tobii sample (~60 Hz) for as long as
% gaze stays on the same fixation candidate: fixation_duration_s grows
% and the position is the running mean over all samples so far, so it
% differs (usually trivially) on almost every valid row -- NOT a "one
% row per completed fixation" log the way Chen's batch I-VT output is.
% A new fixation is marked by fixation_duration_s DROPPING relative to
% the previous valid row (it restarts counting from that fixation's own
% start), or by validity resuming after a gap. This returns each
% fixation's LAST (most-refined) sample, taken immediately before the
% next fixation's first sample.
%
% Verified against data/raw_data/ur5e_gaze_assisted_adm_2d_
% chen_shape_test.csv: gives 153 fixations over 72.5 s (2.1/s, a
% physiologically ordinary rate), versus 2870 "fixations" from naively
% treating every position change as a new fixation.

    sample_count = numel(time_s);
    is_new_fixation = false(sample_count, 1);
    previous_duration_s = NaN;

    for sample_idx = 1:sample_count
        if ~fixation_valid(sample_idx)
            previous_duration_s = NaN;
            continue;
        end

        is_new_fixation(sample_idx) = isnan(previous_duration_s) || ...
            fixation_duration_s_column(sample_idx) < ...
            previous_duration_s - 1e-9;

        previous_duration_s = fixation_duration_s_column(sample_idx);
    end

    new_fixation_indices = find(is_new_fixation);
    fixation_count = numel(new_fixation_indices);

    fixation_x_m_out = zeros(fixation_count, 1);
    fixation_y_m_out = zeros(fixation_count, 1);
    fixation_time_s = zeros(fixation_count, 1);
    fixation_duration_s = zeros(fixation_count, 1);

    last_valid_index = find(fixation_valid, 1, 'last');

    for fixation_idx = 1:fixation_count
        if fixation_idx < fixation_count
            representative_index = new_fixation_indices(fixation_idx + 1) - 1;
        else
            representative_index = last_valid_index;
        end

        fixation_x_m_out(fixation_idx) = fixation_x_m(representative_index);
        fixation_y_m_out(fixation_idx) = fixation_y_m(representative_index);
        fixation_time_s(fixation_idx) = time_s(representative_index);
        fixation_duration_s(fixation_idx) = ...
            fixation_duration_s_column(representative_index);
    end

    fixation_xy_m = [fixation_x_m_out, fixation_y_m_out];
end


function [fixation_xy_m, raw_gaze_xy_m, fixation_time_s, ...
        raw_gaze_time_s] = detect_fixations_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres in CoppeliaSim [m].
%
% Reuses the I-VT and UCS-to-CoppeliaSim formulas from
% compute_plot_similarity_fixation_direction_force_vel.m. fixation_xy_m
% is N-by-2 and empty when no fixation meets the minimum duration.
% raw_gaze_xy_m is every sample where BOTH eyes are valid, strictly
% averaged (left + right) / 2, mapped the same way; raw_gaze_time_s is
% its matching trial time [s]. fixation_time_s is the mid-time of each
% fixation, on the shared trial clock.

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


function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
        obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh.
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


function distance_m = distance_to_reference_path( ...
        path_x_m, path_y_m, reference_xy_m)
%DISTANCE_TO_REFERENCE_PATH Nearest-reference-sample distance [m].

    distance_m = inf(numel(path_x_m), 1);

    for reference_idx = 1:size(reference_xy_m, 1)
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        distance_m = min(distance_m, candidate_m);
    end
end


function jerk = compute_jerk_series( ...
        raw_time_s, raw_x_m, raw_y_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
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
%COMPUTE_DSJ Dimensionless squared jerk, paper Eq. (16).
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
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV cells.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
%POINT_IS_ON_DISPLAY Identify normalized points inside active display bounds.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end


function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye directions.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s].
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
