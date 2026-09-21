%% Diagnostics for one classifier-gated Chen variable admittance trial.
%
% Inputs:
%   data/raw_data/<read_robot_csv_file>    controller log, written by
%     scripts/ur5e_gaze_assisted_chen_vac_2d.py -- every column
%     plot_raw_data_chen_vac_2d.m reads, plus the classifier-gating
%     columns (present whether enable_classifier_gating was True or
%     False for that run).
%   data/raw_data/<read_gaze_csv_file>   matching gaze log.
%   data/calibrations/<read_calibration_json_file>   optional, only for
%     the gaze overlay in Figure 1 and the P(Fine) path in Figure 8.
%   data/shapes/<read_shape_obj_file>          optional, only for the
%     accuracy metric in Figure 4.
%   All four, plus the output filename, are DERIVED from the "Trial
%   identity" block in User Settings (bandwidth_label, shape_name,
%   subject_id, calibration_direction_label) -- to switch trials, edit
%   that block only, not the paths themselves.
%
% Output:
%   Figure 1: TCP path colored by applied damping, with gaze overlay and
%     intended-direction arrows.
%   Figure 2: direct intention, paper Eq. (7) -- force against its band,
%     and desired/applied damping against the GATED [bd_min, bd_max]
%     band (which MOVES with P(Fine), unlike the fixed band in
%     plot_raw_data_chen_vac_2d.m), with the fixed Fine/Gross bound
%     pairs drawn for reference.
%   Figure 3: admittance execution -- commanded vs. measured velocity.
%   Figure 4: accuracy, paper Eq. (15).
%   Figure 5: smoothness, paper Eq. (16), DSJ.
%   Figure 6: effort, paper Eq. (17), mean force.
%   Figure 7: velocity/force/damping relationships.
%   Figure 8 (NEW): TCP path colored by P(Fine) instead of damping --
%     where the classifier thought the task was Fine vs. Gross.
%   Figure 9 (NEW): P(Fine) over time (with non-"ok" classifier_status
%     ticks marked) and probability age against the staleness threshold.
%   Figure 10 (NEW): model performance -- P(Fine) histogram split by the
%     model's own predicted class, and, ONLY if the CSV carries a human
%     "label" column (e.g. a *_live_labeled_*.csv from manual_label_
%     chen_var_adm_2d_by_live_replay.py), a live confusion matrix of
%     predicted vs. human label for this specific trial.
%   A numeric summary printed to the command window.
%   All ten figures (Figure 4 only if the shape mesh loads) are also
%   saved TOGETHER in one native MATLAB .fig file under data/images/
%   figures/plot_raw_data_gaze_assisted_chen_vac_2d/ (see
%   write_combined_fig in User Settings), so plot_saved_raw_data_gaze_
%   assisted_chen_vac_2d.m can reopen all of them at once without
%   rereading any input.
%
%   NOT included, by request (see plot_raw_data_chen_vac_2d.m for these):
%     fitted power law, speed-curvature relation, turn evidence and
%     damping anisotropy, pupil diameter and gaze validity, and the
%     animated gaze/TCP/damping replay.
%
% Method reuse:
%   I-VT fixation detection, eye combination, the UCS-to-CoppeliaSim
%   mapping, the shape-band mesh reader, the jerk/DSJ chain, and the
%   Figure 8/9 P(Fine) machinery are copied from plot_raw_data_chen_
%   vac_2d.m / plot_gaze_assisted_chen_vac_gating.m, self-contained here
%   per this project's convention that MATLAB scripts do not call one
%   another.
%
% Assumptions:
%   The CSV was written by scripts/ur5e_gaze_assisted_chen_vac_2d.py, so
%   column names match that script's write_csv_header(). The gaze CSV
%   shares its time_s origin with the controller CSV.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));

%% User Settings
% TODO explicit directory where this script's figures are saved, one
% subfolder per script for organization (see AI_GUIDELINES.md).
write_image_directory = fullfile(script_directory, '..', 'data', ...
    'images', 'figures', 'plot_raw_data_gaze_assisted_chen_vac_2d');

% TODO -------------------- Trial identity --------------------
% All four input paths, plus the output filename, are DERIVED from
% these few tags -- switch trials by editing only this block, never the
% paths further down. Verified against every file actually present
% under data/raw_data/test_subjects/, data/calibrations/, and
% data/shapes/ as of this writing.
bandwidth_label = "narrow";    % TODO 'narrow' or 'wide'
shape_name = "ellipse";        % TODO 'ellipse', 'pear', or 'fine_gross'
subject_id = "s11";            % TODO trial subject, e.g. "s11"

% Calibration filenames don't all carry a direction token (a handful of
% older ones have none at all), but every trial this script has been
% run on so far does -- 'clockwise' or 'counter_clockwise'.
calibration_direction_label = "clockwise"; % TODO 'clockwise' or 'counter_clockwise'

% The robot/gaze CSV filenames also encode which shape+subject the
% model bundle was held out of training on (leave-one-out
% generalization test). That pair equals the trial's own shape+subject
% in every run so far, so these default to the values above -- override
% only if a specific run's held-out pair truly differs from its own
% trial identity.
held_out_shape_name = shape_name;
held_out_subject_id = subject_id;

shape_tag = bandwidth_label + "_" + shape_name;
% TODO -------------------- End trial identity --------------------

% Output filename (native MATLAB .fig; every figure below is bundled
% into this ONE file via savefig()'s multi-handle form, not one file
% per figure -- see plot_saved_raw_data_gaze_assisted_chen_vac_2d.m,
% which reopens all of them at once with a single openfig()). Figure 4
% (accuracy) is only produced when the shape mesh loads (see
% accuracy_available below); when it is skipped, this file simply holds
% the other nine figures.
write_combined_fig = fullfile(write_image_directory, sprintf( ...
    'raw_data_gaze_assisted_chen_vac_2d_%s_%s.fig', shape_tag, subject_id));

% Figure window title -- same as the saved file's own name (no
% directory, no extension), so a reopened window is traceable to its
% .fig file at a glance.
[~, write_combined_fig_name, ~] = fileparts(write_combined_fig);

% Controller log written by scripts/ur5e_gaze_assisted_chen_vac_2d.py.
read_robot_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'test_subjects', sprintf( ...
    'ur5e_gaze_assisted_chen_vac_2d_held_out_%s_and_%s_on_%s_shape.csv', ...
    held_out_shape_name, held_out_subject_id, shape_tag));

% Matching gaze log from the same trial (same time_s origin).
read_gaze_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'test_subjects', sprintf( ...
    'tobii_gaze_assisted_chen_vac_2d_held_out_%s_and_%s_on_%s_shape.csv', ...
    held_out_shape_name, held_out_subject_id, shape_tag));

% TODO true overlays gaze on the TCP path (Figure 1). Set false to skip
% it, e.g. for a trial recorded without the eye tracker.
plot_gaze_overlay = true;

% Calibration JSON mapping Tobii UCS to the CoppeliaSim frame. If
% missing, the script still runs and only gaze-position panels skip,
% because the controller diagnostics need no calibration.
read_calibration_json_file = fullfile( ...
    script_directory, '..', 'data', 'calibrations', sprintf( ...
    'tobii_coppelia_reference_calibration_%s_shape_%s_%s.json', ...
    shape_tag, calibration_direction_label, subject_id));

% TODO must match the Python controller's User Settings. Force band is
% drawn as reference lines on Figure 2; the script warns if the logged
% damping ever leaves BOTH the Fine and Gross bounds' outer envelope.
force_min_n = 1.0;
force_max_n = 11.0;

% TODO damping bounds [N s/m] per predicted class, must match the Python
% controller's fine_damping_bounds_n_s_m / gross_damping_bounds_n_s_m --
% drawn as reference lines showing the two pairs bd_min/bd_max
% interpolate between.
fine_damping_bounds_n_s_m = [80.0, 200.0];
gross_damping_bounds_n_s_m = [10.0, 100.0];

% TODO must match the Python controller's max_probability_age_s -- drawn
% as the staleness threshold on Figure 9's probability-age panel.
max_probability_age_s = 2.0;

% TODO which class counts as "positive" for Figure 10's confusion
% matrix -- must be one of the two values in the CSV's "label" column,
% when present.
positive_class_label = "Fine";

% Shape mesh the operator traced, used only for the accuracy metric
% (Figure 4).
read_shape_obj_file = fullfile( ...
    script_directory, '..', 'data', 'shapes', ...
    sprintf('%s_shape.obj', shape_tag));

% Translation from the mesh CAD frame to the CoppeliaSim frame [m], one
% entry per shape_name. VERIFY THIS before trusting the accuracy number
% -- the summary prints a registration-shift diagnostic so a bad offset
% is visible rather than silent. TODO add a case here before using a
% new shape_name.
switch shape_name
    case "ellipse"
        shape_offset_xy_m = [-0.015, -0.650];
    case "pear"
        shape_offset_xy_m = [0.025, -0.660];
    case "fine_gross"
        shape_offset_xy_m = [-0.0, -0.650];
    otherwise
        error(['No shape_offset_xy_m entry for shape_name "%s" -- add ' ...
            'one to the switch above.'], shape_name);
end

% TODO spacing used to resample the reference path [m], positive.
reference_spacing_m = 5e-4;

% TODO speed separating movement from rest [m/s]. The paper uses 0.005
% and evaluates every performance metric on movement samples only.
moving_speed_threshold_m_s = 0.005;

% TODO filter windows for the jerk chain used by the smoothness metric
% [s].
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;

% TODO extra smoothing windows [s] used only to report DSJ's filter
% sensitivity.
jerk_sensitivity_windows_s = [0.05, 0.10, 0.20];

% TODO I-VT angular-velocity threshold [deg/s] and minimum retained
% fixation duration [s] for the Figure 1 gaze overlay.
velocity_threshold_deg_s = 20.0;
minimum_fixation_duration_s = 0.060;

% TODO true excludes gaze estimates outside the normalized display.
plot_only_on_display_gaze = true;

% TODO physical screen motion / robot motion [unitless], must be > 0,
% and the Tobii UCS axis signs in CoppeliaSim, each -1 or +1. Must match
% the values used elsewhere in this project.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO plot every Nth intended-direction arrow to avoid clutter, and
% their display length [m].
intended_arrow_stride = 60;
intended_arrow_length_m = 0.012;

% TODO marker sizes [points^2] and colors for the spatial panels.
trajectory_marker_size = 10;
raw_gaze_marker_size = 4;
raw_gaze_marker_color = "r";
fixation_marker_size = 18;
fixation_marker_color = 'g';

% TODO Figure 7 (velocity/force/damping relationships) marker size
% [points^2] and transparency [0, 1].
relationship_marker_size = 6;
relationship_marker_alpha = 0.15;

% TODO Figure 10 probability histogram bin count.
probability_histogram_bin_count = 30;

fprintf('Input:  %s\n', read_robot_csv_file);
fprintf('Input:  %s\n', read_gaze_csv_file);
fprintf('Input:  %s (optional)\n', read_calibration_json_file);
fprintf('Input:  %s (optional)\n', read_shape_obj_file);
fprintf('Output: %s\n\n', write_combined_fig);

if ~isfolder(write_image_directory)
    mkdir(write_image_directory);
end
figure_handles = gobjects(0);

%% Load the controller log
robot_table = readtable(read_robot_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

time_s = robot_table.time_s;

force_x_n = robot_table.Fx_filt_N;
force_y_n = robot_table.Fy_filt_N;
force_magnitude_n = hypot(force_x_n, force_y_n);
force_raw_magnitude_n = hypot(robot_table.Fx_raw_N, robot_table.Fy_raw_N);

coppeliasim_x_m = robot_table.coppeliasim_x_m;
coppeliasim_y_m = robot_table.coppeliasim_y_m;

velocity_command_x_m_s = robot_table.vx_cmd_m_s;
velocity_command_y_m_s = robot_table.vy_cmd_m_s;
speed_command_m_s = robot_table.speed_cmd_m_s;
speed_actual_m_s = hypot( ...
    robot_table.tcp_vx_actual_m_s, robot_table.tcp_vy_actual_m_s);

damping_desired_n_s_m = robot_table.bd_desired_Ns_m;
damping_applied_n_s_m = robot_table.bd_applied_Ns_m;

intended_direction_x = robot_table.intended_dir_x;
intended_direction_y = robot_table.intended_dir_y;

virtual_mass_kg = robot_table.M_kg(1);

% --- Classifier-gating columns ---
bd_min_n_s_m = robot_table.bd_min_Ns_m;
bd_max_n_s_m = robot_table.bd_max_Ns_m;
fine_probability = robot_table.fine_probability;
probability_age_s = robot_table.fine_probability_age_s;
classifier_status = robot_table.classifier_status;
have_probability = any(~isnan(fine_probability));
have_label = ismember('label', robot_table.Properties.VariableNames);
if have_label
    true_label = robot_table.label;
end

%% Load and map gaze, if requested and possible
gaze_available = false;
gaze_calibrated = false;
raw_gaze_xy_m = [];
raw_gaze_time_s = [];
fixation_xy_m = [];
fixation_time_s = [];

if plot_gaze_overlay && isfile(read_gaze_csv_file)
    gaze_table = readtable(read_gaze_csv_file, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
    gaze_available = true;

    if isfile(read_calibration_json_file)
        calibration_reference = jsondecode(fileread(read_calibration_json_file));
        tobii_reference_xy_mm = double( ...
            calibration_reference.manual_tobii_reference_xy_mm(:)');
        coppelia_reference_xy_m = double( ...
            calibration_reference.manual_coppelia_reference_xy_m(:)');

        [fixation_xy_m, raw_gaze_xy_m, fixation_time_s, raw_gaze_time_s] = ...
            detect_fixations_coppelia_xy( ...
            gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
            screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
            velocity_threshold_deg_s, minimum_fixation_duration_s, ...
            plot_only_on_display_gaze);
        gaze_calibrated = true;
    else
        warning(['Calibration JSON not found, so gaze positions are ' ...
            'skipped. Controller diagnostics are unaffected.']);
    end
elseif plot_gaze_overlay
    warning('Gaze CSV not found; skipping all gaze panels.');
end

%% ================================================================
% Figure 1: TCP path colored by the applied isotropic damping
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
hold on;

if gaze_calibrated && ~isempty(raw_gaze_xy_m)
    scatter(raw_gaze_xy_m(:, 1), raw_gaze_xy_m(:, 2), ...
        raw_gaze_marker_size, raw_gaze_marker_color, 'filled', ...
        'MarkerFaceAlpha', 0.25, 'DisplayName', 'Raw gaze (both eyes)');
end

scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
    damping_applied_n_s_m, 'filled', 'DisplayName', 'TCP path');

if gaze_calibrated && ~isempty(fixation_xy_m)
    scatter(fixation_xy_m(:, 1), fixation_xy_m(:, 2), ...
        fixation_marker_size, fixation_marker_color, 'filled', ...
        'MarkerFaceAlpha', 0.7, 'DisplayName', 'I-VT fixation');
end

% arrow_indices = 1:intended_arrow_stride:numel(coppeliasim_x_m);
% quiver(coppeliasim_x_m(arrow_indices), coppeliasim_y_m(arrow_indices), ...
%     intended_arrow_length_m .* intended_direction_x(arrow_indices), ...
%     intended_arrow_length_m .* intended_direction_y(arrow_indices), ...
%     0, 'Color', [0.85, 0.20, 0.10], 'LineWidth', 1.0, ...
%     'DisplayName', 'Intended direction');

axis equal;
grid on;
clim([min(gross_damping_bounds_n_s_m), max(fine_damping_bounds_n_s_m)]);
damping_colorbar = colorbar;
damping_colorbar.Label.String = 'Applied damping b_d [N s/m]';
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title('TCP path colored by applied damping, with intended direction');
legend('Location', 'best', 'FontSize', 8);

%% ================================================================
% Figure 2: Direct intention, paper Eq. (7), through the gated bounds
% =================================================================

gross_color = [0.20, 0.50, 0.80];
fine_color = [0.85, 0.20, 0.20];

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, force_raw_magnitude_n, '-', 'LineWidth', 0.6, ...
    'Color', [0.7, 0.7, 0.7]);
hold on;
plot(time_s, force_magnitude_n, '-', 'LineWidth', 1.2);
yline(force_min_n, 'r--', 'f_{min}');
yline(force_max_n, 'r--', 'f_{max}');
grid on;
ylabel('|F| [N]');
title(['Human force magnitude against the direct-intention band. ' ...
    'Below f_{min} damping is maximal, above f_{max} it is minimal.']);
legend('Raw', 'Filtered', 'Location', 'best');

nexttile;
hold on;
fill([time_s; flipud(time_s)], [bd_min_n_s_m; flipud(bd_max_n_s_m)], ...
    [0.85, 0.85, 0.85], 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
    'DisplayName', 'Gated [b_{min}, b_{max}]');
plot(time_s, damping_desired_n_s_m, '-', 'LineWidth', 1.0, ...
    'DisplayName', 'Desired');
plot(time_s, damping_applied_n_s_m, '-', 'LineWidth', 1.3, ...
    'DisplayName', 'Applied (rate limited)');
yline(fine_damping_bounds_n_s_m(1), '--', 'Color', fine_color, ...
    'HandleVisibility', 'off');
yline(fine_damping_bounds_n_s_m(2), '--', 'Fine [L_f, U_f]', ...
    'Color', fine_color, 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 7, 'HandleVisibility', 'off');
yline(gross_damping_bounds_n_s_m(1), '--', 'Gross [L_g, U_g]', ...
    'Color', gross_color, 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 7, 'HandleVisibility', 'off');
yline(gross_damping_bounds_n_s_m(2), '--', 'Color', gross_color, ...
    'HandleVisibility', 'off');
grid on;
xlabel('Time [s]');
ylabel('b_d [N s/m]');
title('Isotropic damping from Eq. (7), against the MOVING gated band');
legend('Location', 'best', 'FontSize', 7);

%% ================================================================
% Figure 3: Admittance execution
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, velocity_command_x_m_s, '-', 'LineWidth', 1.0);
hold on;
plot(time_s, robot_table.tcp_vx_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
ylabel('v_x [m/s]');
title('Commanded versus measured TCP velocity, x');
legend('Commanded', 'Measured', 'Location', 'best');

nexttile;
plot(time_s, velocity_command_y_m_s, '-', 'LineWidth', 1.0);
hold on;
plot(time_s, robot_table.tcp_vy_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
ylabel('v_y [m/s]');
title('Commanded versus measured TCP velocity, y');
legend('Commanded', 'Measured', 'Location', 'best');

nexttile;
plot(time_s, speed_command_m_s, '-', 'LineWidth', 1.2);
hold on;
plot(time_s, speed_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Commanded versus measured planar speed');
legend('Commanded', 'Measured', 'Location', 'best');

%% ================================================================
% Figures 4 to 6: paper performance metrics, Section III.C
% =================================================================

moving_mask = speed_actual_m_s > moving_speed_threshold_m_s;

%% ---- Figure 4: Accuracy, paper Eq. (15)

accuracy_available = false;

if isfile(read_shape_obj_file)
    [outer_edge_xy_m, inner_edge_xy_m, reference_xy_m] = ...
        read_shape_band_obj(read_shape_obj_file, reference_spacing_m);

    if isempty(reference_xy_m)
        warning(['Shape mesh did not yield at least two edge loops, so ' ...
            'the reference path is unavailable and accuracy is ' ...
            'skipped.']);
    else
        outer_edge_xy_m = outer_edge_xy_m + shape_offset_xy_m;
        inner_edge_xy_m = inner_edge_xy_m + shape_offset_xy_m;
        reference_xy_m = reference_xy_m + shape_offset_xy_m;

        tracking_error_m = distance_to_reference_path( ...
            coppeliasim_x_m, coppeliasim_y_m, reference_xy_m);
        accuracy_available = true;
    end
else
    warning(['Shape mesh not found, so accuracy is skipped. Set ' ...
        'read_shape_obj_file to the mesh the operator traced.']);
end

if accuracy_available
    mean_error_m = mean(tracking_error_m(moving_mask));

    figure_handles(end + 1) = figure('Name', ...
        write_combined_fig_name, 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(outer_edge_xy_m(:, 1), outer_edge_xy_m(:, 2), '-', ...
        'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.0, ...
        'DisplayName', 'Band edge');
    hold on;
    plot(inner_edge_xy_m(:, 1), inner_edge_xy_m(:, 2), '-', ...
        'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    plot(reference_xy_m(:, 1), reference_xy_m(:, 2), '--', ...
        'Color', [0.1, 0.1, 0.1], 'LineWidth', 1.0, ...
        'DisplayName', 'Reference centre line');
    scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
        1e3 * tracking_error_m, 'filled', 'DisplayName', 'TCP path');
    axis equal;
    grid on;
    error_colorbar = colorbar;
    error_colorbar.Label.String = 'Error [mm]';
    xlabel('CoppeliaSim x [m]');
    ylabel('CoppeliaSim y [m]');
    title('TCP path colored by distance to the reference');
    legend('Location', 'best', 'FontSize', 8);

    nexttile;
    plot(time_s, 1e3 * tracking_error_m, '-', 'LineWidth', 1.0);
    hold on;
    yline(1e3 * mean_error_m, 'r--', ...
        sprintf('Mean %.2f mm', 1e3 * mean_error_m));
    grid on;
    xlabel('Time [s]');
    ylabel('Error [mm]');
    title('Distance to the reference path over time');
end

%% ---- Figure 5: Smoothness, dimensionless squared jerk, paper Eq. (16)

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

jerk = compute_jerk_series(time_s, coppeliasim_x_m, coppeliasim_y_m, ...
    jerk_outlier_window_s, jerk_smoothing_window_s);

dimensionless_squared_jerk = compute_dsj( ...
    jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m);

nexttile;
plot(jerk.time_s, sqrt(jerk.squared_jerk_m2_s6), '-', 'LineWidth', 1.0);
grid on;
ylabel('|jerk| [m/s^3]');
title(sprintf(['Jerk magnitude (smoothing window %.0f ms). ' ...
    'DSJ = %.3g'], 1e3 * jerk_smoothing_window_s, ...
    dimensionless_squared_jerk));

nexttile;
plot(jerk.time_s, cumtrapz(jerk.time_s, jerk.squared_jerk_m2_s6), ...
    '-', 'LineWidth', 1.2);
grid on;
xlabel('Time [s]');
ylabel('\int |jerk|^2 dt [m^2/s^5]');
title(['Cumulative squared jerk: steep sections are where smoothness ' ...
    'was lost']);

dsj_sensitivity = zeros(numel(jerk_sensitivity_windows_s), 1);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    sensitivity_jerk = compute_jerk_series( ...
        time_s, coppeliasim_x_m, coppeliasim_y_m, ...
        jerk_outlier_window_s, jerk_sensitivity_windows_s(window_idx));
    dsj_sensitivity(window_idx) = compute_dsj(sensitivity_jerk.time_s, ...
        sensitivity_jerk.squared_jerk_m2_s6, ...
        sensitivity_jerk.path_length_m);
end

%% ---- Figure 6: Effort, mean force, paper Eq. (17)

mean_force_n = mean(force_magnitude_n(moving_mask));

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, force_magnitude_n, '-', 'LineWidth', 1.0);
hold on;
yline(mean_force_n, 'r--', sprintf('Mean %.3f N', mean_force_n));
grid on;
xlabel('Time [s]');
ylabel('|F| [N]');
title('Interaction force magnitude over time');

nexttile;
histogram(force_magnitude_n(moving_mask), 40, ...
    'Normalization', 'probability');
hold on;
xline(mean_force_n, 'r--', sprintf('Mean %.3f N', mean_force_n));
grid on;
xlabel('|F| [N]');
ylabel('Fraction of moving samples [-]');
title('Force distribution while moving');

%% ================================================================
% Figure 7: velocity, force, and damping relationships
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
scatter(force_magnitude_n, speed_actual_m_s, relationship_marker_size, ...
    time_s, 'filled', 'MarkerFaceAlpha', relationship_marker_alpha);
grid on;
speed_force_colorbar = colorbar;
speed_force_colorbar.Label.String = 'Time [s]';
xlabel('|F| [N]');
ylabel('Speed [m/s]');
title('Speed vs. force');

nexttile;
scatter(speed_actual_m_s, damping_applied_n_s_m, relationship_marker_size, ...
    time_s, 'filled', 'MarkerFaceAlpha', relationship_marker_alpha);
grid on;
damping_speed_colorbar = colorbar;
damping_speed_colorbar.Label.String = 'Time [s]';
xlabel('Speed [m/s]');
ylabel('Damping [N s/m]');
title('Damping vs. speed');

nexttile;
scatter(force_magnitude_n, damping_applied_n_s_m, relationship_marker_size, ...
    time_s, 'filled', 'MarkerFaceAlpha', relationship_marker_alpha);
grid on;
damping_force_colorbar = colorbar;
damping_force_colorbar.Label.String = 'Time [s]';
xlabel('|F| [N]');
ylabel('Damping [N s/m]');
title('Damping vs. force');

%% ================================================================
% Figure 8 (NEW): TCP path colored by P(Fine)
% =================================================================

probability_colormap = [ ...
    linspace(gross_color(1), fine_color(1), 256)', ...
    linspace(gross_color(2), fine_color(2), 256)', ...
    linspace(gross_color(3), fine_color(3), 256)'];

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
hold on;

if have_probability
    scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
        fine_probability, 'filled', 'DisplayName', 'TCP path');
    colormap(gca, probability_colormap);
    clim([0, 1]);
    probability_colorbar = colorbar;
    probability_colorbar.Label.String = 'P(Fine)  [blue = Gross, red = Fine]';
    title('TCP path colored by predicted P(Fine)');
else
    scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
        damping_applied_n_s_m, 'filled', 'DisplayName', 'TCP path');
    damping_colorbar2 = colorbar;
    damping_colorbar2.Label.String = 'Applied damping b_d [N s/m]';
    title(['TCP path colored by applied damping (gating was off or ' ...
        'every probability is NaN, so P(Fine) is unavailable)']);
end

axis equal;
grid on;
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
legend('Location', 'best', 'FontSize', 8);

%% ================================================================
% Figure 9 (NEW): P(Fine) over time and probability age
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

probability_axes = nexttile;
hold on;
if have_probability
    plot(time_s, fine_probability, '-', 'LineWidth', 1.0, ...
        'DisplayName', 'P(Fine)');
    yline(0.5, 'k--', 'HandleVisibility', 'off');

    is_not_ok = classifier_status ~= "ok";
    if any(is_not_ok)
        scatter(time_s(is_not_ok), fine_probability(is_not_ok), 10, ...
            [0.6, 0.6, 0.6], 'filled', 'MarkerFaceAlpha', 0.5, ...
            'DisplayName', 'Non-"ok" status');
    end
    legend('Location', 'best', 'FontSize', 7);
else
    text(0.5, 0.5, 'Classifier gating was off for this trial', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
end
grid on;
ylim([-0.05, 1.05]);
ylabel('P(Fine) [-]');
title('Predicted probability, with non-"ok" status ticks marked');

age_axes = nexttile;
if have_probability
    plot(time_s, probability_age_s, '-', 'LineWidth', 1.0);
    hold on;
    yline(max_probability_age_s, 'r--', 'Fallback threshold');
    ylabel('Age [s]');
else
    text(0.5, 0.5, 'Classifier gating was off for this trial', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
end
grid on;
xlabel('Time [s]');
title('P(Fine) age (time since the last successful classifier update)');

linkaxes([probability_axes, age_axes], 'x');

%% ================================================================
% Figure 10 (NEW): model performance -- probability profile, and a
% live confusion matrix if this trial has human labels
% =================================================================

% Eligibility is computed FIRST and used for both the tiledlayout size
% and the draw below, so a have_label=true trial with a degenerate
% label column (not exactly {positive_class_label, one other value})
% cannot reserve a second tile that then never gets drawn into.
confusion_available = false;
if have_probability && have_label
    label_values = sort(unique(true_label(true_label ~= "None")));
    confusion_available = numel(label_values) == 2 && ...
        any(label_values == positive_class_label);
end

figure_handles(end + 1) = figure('Name', write_combined_fig_name, ...
    'Color', 'w');
if confusion_available
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
else
    tiledlayout(1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
end

nexttile;
if have_probability
    predicted_is_fine = fine_probability > 0.5;
    histogram_edges = linspace(0, 1, probability_histogram_bin_count + 1);
    histogram(fine_probability(predicted_is_fine), histogram_edges, ...
        'Normalization', 'probability', 'FaceColor', fine_color, ...
        'FaceAlpha', 0.6, 'DisplayName', 'Predicted Fine');
    hold on;
    histogram(fine_probability(~predicted_is_fine), histogram_edges, ...
        'Normalization', 'probability', 'FaceColor', gross_color, ...
        'FaceAlpha', 0.6, 'DisplayName', 'Predicted Gross');
    grid on;
    xlabel('P(Fine)');
    ylabel('Fraction of ticks');
    title('Predicted-probability profile');
    legend('Location', 'best', 'FontSize', 7);
else
    text(0.5, 0.5, 'Classifier gating was off for this trial', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
end

if confusion_available
    negative_class_label = label_values(label_values ~= positive_class_label);
    negative_class_label = negative_class_label(1);
    labelled_mask = true_label ~= "None";
    predicted_label = repmat(negative_class_label, height(robot_table), 1);
    predicted_label(fine_probability > 0.5) = positive_class_label;

    confusion_counts = zeros(2, 2);
    class_order = [positive_class_label, negative_class_label];
    for true_idx = 1:2
        for predicted_idx = 1:2
            confusion_counts(true_idx, predicted_idx) = sum( ...
                labelled_mask & true_label == class_order(true_idx) & ...
                predicted_label == class_order(predicted_idx));
        end
    end
    live_accuracy = sum(diag(confusion_counts)) / sum(labelled_mask);

    nexttile;
    imagesc(confusion_counts);
    colormap(gca, 'parula');
    colorbar;
    set(gca, 'XTick', 1:2, 'XTickLabel', cellstr(class_order), ...
        'YTick', 1:2, 'YTickLabel', cellstr(class_order));
    xlabel('Predicted');
    ylabel('True (human label)');
    title(sprintf('Live confusion matrix (acc=%.3f, n=%d)', ...
        live_accuracy, sum(labelled_mask)));
    for true_idx = 1:2
        for predicted_idx = 1:2
            text(predicted_idx, true_idx, ...
                num2str(confusion_counts(true_idx, predicted_idx)), ...
                'HorizontalAlignment', 'center', 'Color', 'k', ...
                'FontWeight', 'bold');
        end
    end
end

savefig(figure_handles, write_combined_fig);

%% ================================================================
% Summary
% =================================================================

trial_duration_s = time_s(end) - time_s(1);
sample_interval_s = diff(time_s);
path_length_m = sum(hypot(diff(coppeliasim_x_m), diff(coppeliasim_y_m)));

fprintf('\n--- Gaze-assisted Chen trial summary ---\n');
fprintf('File: %s\n', read_robot_csv_file);
fprintf('Duration: %.2f s over %d samples (%.1f Hz mean)\n', ...
    trial_duration_s, height(robot_table), ...
    (height(robot_table) - 1) / trial_duration_s);
fprintf('Control period: median %.3f ms, max %.3f ms\n', ...
    1e3 * median(sample_interval_s), 1e3 * max(sample_interval_s));
fprintf('Virtual mass: %.2f kg\n', virtual_mass_kg);
fprintf('Path length: %.4f m, moving %.1f %% of samples\n', ...
    path_length_m, 100 * mean(moving_mask));

fprintf('\nEffort and motion:\n');
fprintf('  Mean force: %.3f N, peak %.3f N\n', ...
    mean(force_magnitude_n), max(force_magnitude_n));
fprintf('  Mean speed: %.4f m/s, peak %.4f m/s\n', ...
    mean(speed_actual_m_s), max(speed_actual_m_s));

fprintf('\nDirect intention, Eq. (7), gated:\n');
fprintf('  Applied damping: %.2f to %.2f N s/m, mean %.2f\n', ...
    min(damping_applied_n_s_m), max(damping_applied_n_s_m), ...
    mean(damping_applied_n_s_m));
if min(damping_applied_n_s_m) < min(gross_damping_bounds_n_s_m) - 1e-6 || ...
        max(damping_applied_n_s_m) > max(fine_damping_bounds_n_s_m) + 1e-6
    warning(['Logged damping leaves the union of the Fine/Gross bound ' ...
        'pairs -- these User Settings probably do not match the ' ...
        'controller run.']);
end

fprintf('\nPerformance metrics, paper Section III.C ');
fprintf('(movement samples only, speed > %.3f m/s):\n', ...
    moving_speed_threshold_m_s);

if accuracy_available
    moving_error_m = tracking_error_m(moving_mask);
    fprintf('  Accuracy  Eq. (15) mean error: %.2f mm ', ...
        1e3 * mean_error_m);
    fprintf('(median %.2f, max %.2f)\n', ...
        1e3 * median(moving_error_m), 1e3 * max(moving_error_m));

    [best_shift_xy_m, best_error_m] = estimate_reference_shift( ...
        coppeliasim_x_m(moving_mask), coppeliasim_y_m(moving_mask), ...
        reference_xy_m, reference_spacing_m);
    fprintf('    registration check: a further shift of [%+.1f, %+.1f] mm ', ...
        1e3 * best_shift_xy_m(1), 1e3 * best_shift_xy_m(2));
    fprintf('would give %.2f mm\n', 1e3 * best_error_m);
    if norm(best_shift_xy_m) > 0.005
        fprintf(['    NOTE: that is over 5 mm, so verify ' ...
            'shape_offset_xy_m before quoting this accuracy.\n']);
    end
else
    fprintf('  Accuracy  Eq. (15): skipped, no reference path.\n');
end

fprintf('  Smooth.   Eq. (16) DSJ: %.4g ', dimensionless_squared_jerk);
fprintf('(smoothing window %.0f ms)\n', 1e3 * jerk_smoothing_window_s);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    fprintf('    at %3.0f ms window: DSJ = %.4g\n', ...
        1e3 * jerk_sensitivity_windows_s(window_idx), ...
        dsj_sensitivity(window_idx));
end
fprintf(['    DSJ is filter-dependent; only compare trials computed ' ...
    'with the same window.\n']);

fprintf('  Effort    Eq. (17) mean force: %.3f N\n', mean_force_n);

fprintf('\nClassifier:\n');
if have_probability
    valid_probability = fine_probability(~isnan(fine_probability));
    fprintf('  P(Fine): mean %.3f, median %.3f, range [%.3f, %.3f]\n', ...
        mean(valid_probability), median(valid_probability), ...
        min(valid_probability), max(valid_probability));
    unique_statuses = unique(classifier_status);
    fprintf('  Status breakdown:\n');
    for status_idx = 1:numel(unique_statuses)
        this_status = unique_statuses(status_idx);
        status_count = sum(classifier_status == this_status);
        fprintf('    %-20s %6d  (%.1f %%)\n', this_status, status_count, ...
            100 * status_count / height(robot_table));
    end
    valid_age_s = probability_age_s(~isnan(probability_age_s));
    if ~isempty(valid_age_s)
        stale_count = sum(probability_age_s > max_probability_age_s);
        fprintf(['  Probability age: mean %.3f s, max %.3f s -- exceeded ' ...
            '%.1f s threshold on %d samples (%.1f %%)\n'], ...
            mean(valid_age_s), max(valid_age_s), max_probability_age_s, ...
            stale_count, 100 * stale_count / height(robot_table));
    end
else
    fprintf('  Gating was off for this trial.\n');
end
if confusion_available
    fprintf('  Live accuracy vs. human label: %.3f\n', live_accuracy);
elseif have_probability && ~have_label
    fprintf(['  No "label" column in this CSV -- live confusion matrix ' ...
        'skipped. Use a *_live_labeled_*.csv to get one.\n']);
end

if gaze_available
    fprintf('\nGaze:\n');
    if gaze_calibrated
        fprintf('  I-VT fixations: %d\n', size(fixation_xy_m, 1));
    end
end

fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================

function [best_shift_xy_m, best_error_m] = estimate_reference_shift( ...
        path_x_m, path_y_m, reference_xy_m, search_spacing_m)
%ESTIMATE_REFERENCE_SHIFT Extra translation that would minimize error.
% Copied from plot_raw_data_chen_vac_2d.m. Diagnostic only, never
% applied to the reported accuracy.

    best_shift_xy_m = [0, 0];
    best_error_m = mean(distance_to_reference_path( ...
        path_x_m, path_y_m, reference_xy_m));

    search_radius_m = 0.02;
    for refine_step = 1:2
        step_m = search_radius_m / 5;
        centre_shift_m = best_shift_xy_m;
        for shift_x_m = centre_shift_m(1) + (-5:5) * step_m
            for shift_y_m = centre_shift_m(2) + (-5:5) * step_m
                shifted_reference_m = reference_xy_m + ...
                    [shift_x_m, shift_y_m];
                trial_error_m = mean(distance_to_reference_path( ...
                    path_x_m, path_y_m, shifted_reference_m));
                if trial_error_m < best_error_m
                    best_error_m = trial_error_m;
                    best_shift_xy_m = [shift_x_m, shift_y_m];
                end
            end
        end
        search_radius_m = max(step_m, search_spacing_m * 10);
    end
end


function jerk = compute_jerk_series( ...
        raw_time_s, raw_x_m, raw_y_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
% Copied from plot_raw_data_chen_vac_2d.m.

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
