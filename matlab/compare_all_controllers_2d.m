%% Compare all six controller benchmarks on the same fine_gross_shape task.
%
% Compares, on the same "fine_gross_shape" task, subject s0:
%   1. Fixed admittance, low damping  (scripts/ur5e_fixed_adm_2d.py, d=10)
%   2. Fixed admittance, high damping (scripts/ur5e_fixed_adm_2d.py, d=100)
%   3. Kang    (scripts/ur5e_kang_var_adm_2d.py)
%   4. Chen    (scripts/ur5e_chen_var_adm_2d.py)
%   5. Ying    (scripts/ur5e_ying_var_adm_2d.py)
%   6. Gaze-assisted (scripts/ur5e_gaze_assisted_adm_2d.py)
%
% using the same paper-metric formulas (Eqs 15-17, this project's
% convention throughout) computed identically for every condition, so
% the numbers are directly comparable across all six.
%
% Inputs:
%   data/raw_data/ur5e_<controller>_..._s0.csv   one CSV per condition,
%     selected in User Settings. No Tobii/gaze CSVs are read: this
%     comparison covers pose, force, smoothness, accuracy, time, and
%     path length only, not gaze/fixation metrics.
%   data/shapes/fine_gross_shape.obj             accuracy reference.
%
% Output:
%   Figure 1: TCP path per condition (2x3 tiles), colored by
%   interaction force magnitude, shared color axis across all six.
%   Figure 2: force and jerk magnitude over time, all six overlaid.
%   Figure 3: summary bar charts (2x3 tiles) -- duration, path length,
%   mean force Eq.(17), DSJ Eq.(16), mean error Eq.(15), moving-time %.
%   A numeric summary table is printed to the command window.
%   No file is written; this script only reads raw CSVs and one OBJ.
%
% Method reuse:
%   The shape-band OBJ reader, jerk/DSJ chain, and reference-shift
%   diagnostic are copied verbatim from plot_raw_data_fixed_adm_2d.m;
%   the robot-frame-to-CoppeliaSim mapping for the gaze-assisted
%   condition is copied verbatim from compare_chen_vs_gaze_assisted_2d.m.
%   Self-contained here per this project's convention that MATLAB
%   scripts do not call one another.
%
% Assumptions:
%   All CSVs were written by their respective scripts' write_csv_header(),
%   so column names match. Five of six conditions (both fixed-damping
%   trials, Kang, Chen, Ying) log coppeliasim_x_m/coppeliasim_y_m
%   directly. The gaze-assisted condition logs only robot-base-frame
%   x_m/y_m (no coppeliasim_* columns) and is mapped here with the same
%   coppeliasim_axis_from_robot/coppeliasim_sign_from_robot/
%   coppeliasim_offset_xy_m transform that script itself uses (default
%   identity, matching its current settings) -- VERIFY this matches
%   whatever that script's settings were at the time its trial was
%   recorded. shape_offset_xy_m is the value already validated for
%   fine_gross_shape.obj in plot_raw_data_fixed_adm_2d.m /
%   plot_raw_data_chen_var_adm_2d.m; estimate_reference_shift() flags
%   any condition where it does not line up well (most likely to matter
%   for the gaze-assisted condition, the only one reaching this frame
%   through an extra transform rather than being logged directly).

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');

%% User Settings
% TODO one CSV per condition, in the display order used throughout this
% script's figures and summary table.
csv_fixed_low_damping = fullfile( ...
    raw_data_directory, 'ur5e_fixed_adm_2d_fine_gross_shape_d10_s0.csv');
csv_fixed_high_damping = fullfile( ...
    raw_data_directory, 'ur5e_fixed_adm_2d_fine_gross_shape_d100_s0.csv');
csv_kang = fullfile( ...
    raw_data_directory, 'ur5e_kang_var_adm_2d_fine_gross_shape_s0.csv');
csv_chen = fullfile( ...
    raw_data_directory, 'ur5e_chen_var_adm_2d_fine_gross_shape_s0.csv');
csv_ying = fullfile( ...
    raw_data_directory, 'ur5e_ying_var_adm_2d_fine_gross_shape_s0.csv');
csv_gaze_assisted = fullfile( ...
    raw_data_directory, 'ur5e_gaze_assisted_adm_2d_fine_gross_shape_s0.csv');

% TODO shape mesh both operators traced, used for the accuracy metric.
shape_obj_file = fullfile( ...
    script_directory, '..', 'data', 'shapes', 'fine_gross_shape.obj');

% TODO translation from the mesh CAD frame to the CoppeliaSim frame [m].
% Matches the value already validated for this shape in
% plot_raw_data_fixed_adm_2d.m / plot_raw_data_chen_var_adm_2d.m.
shape_offset_xy_m = [0.0, -0.645];

% TODO spacing used to resample the reference path [m], positive.
reference_spacing_m = 5e-4;

% TODO grid spacing for the reference-shift diagnostic [m], positive.
% Only affects how finely estimate_reference_shift() searches, never the
% reported accuracy itself.
reference_shift_search_spacing_m = 1e-4;

% TODO gaze-assisted TCP-to-CoppeliaSim mapping, matching
% scripts/ur5e_gaze_assisted_adm_2d.py's coppeliasim_axis_from_robot /
% coppeliasim_sign_from_robot / coppeliasim_offset_xy_m. Given here
% 1-based (MATLAB), so that script's 0-based [0, 1] identity is [1, 2]
% here. Default is that script's current identity settings. The other
% five conditions need no such mapping; they log coppeliasim_x_m/y_m
% directly.
gaze_assisted_axis_from_robot = [1, 2];
gaze_assisted_sign_from_robot = [1.0, 1.0];
gaze_assisted_offset_xy_m = [0.0, 0.0];

% TODO speed separating movement from rest [m/s]. The paper uses 0.005
% and evaluates every performance metric on movement samples only.
moving_speed_threshold_m_s = 0.005;

% TODO filter windows for the jerk chain used by the smoothness metric
% [s]. Matches plot_raw_data_fixed_adm_2d.m / plot_raw_data_chen_var_
% adm_2d.m so DSJ values are comparable with those scripts' own output,
% not just between the conditions here.
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;
jerk_sensitivity_windows_s = [0.05, 0.10, 0.20];

% TODO marker size [points^2] for the spatial panels.
trajectory_marker_size = 8;

%% Conditions table
% One row per condition. position_mode is 'coppeliasim' for the five
% conditions that log coppeliasim_x_m/coppeliasim_y_m directly, or
% 'robot_transform' for the one condition (gaze-assisted) that must be
% mapped from robot-base-frame x_m/y_m at load time.

condition_names = { ...
    'Fixed (d=10)', 'Fixed (d=100)', 'Kang', 'Chen', 'Ying', ...
    'Gaze-assisted'};
condition_csv_files = { ...
    csv_fixed_low_damping, csv_fixed_high_damping, csv_kang, csv_chen, ...
    csv_ying, csv_gaze_assisted};
condition_position_modes = { ...
    'coppeliasim', 'coppeliasim', 'coppeliasim', 'coppeliasim', ...
    'coppeliasim', 'robot_transform'};
condition_colors = [ ...
    0.55, 0.55, 0.55;   % Fixed (d=10)  - mid gray
    0.15, 0.15, 0.15;   % Fixed (d=100) - dark gray
    0.30, 0.60, 0.20;   % Kang          - green
    0.85, 0.20, 0.10;   % Chen          - red-orange
    0.90, 0.60, 0.00;   % Ying          - amber
    0.00, 0.45, 0.74];  % Gaze-assisted - blue

condition_count = numel(condition_names);

%% Load every condition and compute the shared metrics
% Row cell arrays throughout, matching condition_names' row orientation,
% so cellfun(...) below returns row vectors matching bar_labels'
% categorical row vector -- avoids relying on bar() to reconcile a
% row/column mismatch between x and y.
trials = cell(1, condition_count);
metrics = cell(1, condition_count);
jerks = cell(1, condition_count);

reference_xy_m = [];
if isfile(shape_obj_file)
    [~, ~, reference_xy_m] = ...
        read_shape_band_obj(shape_obj_file, reference_spacing_m);
    if isempty(reference_xy_m)
        warning(['Shape mesh did not yield a simple two-loop band; ' ...
            'accuracy is skipped for all conditions.']);
    else
        reference_xy_m = reference_xy_m + shape_offset_xy_m;
    end
else
    warning('Shape mesh not found; accuracy is skipped.');
end

for condition_idx = 1:condition_count
    trials{condition_idx} = load_controller_trial( ...
        condition_csv_files{condition_idx}, ...
        condition_position_modes{condition_idx}, ...
        gaze_assisted_axis_from_robot, gaze_assisted_sign_from_robot, ...
        gaze_assisted_offset_xy_m);

    [metrics{condition_idx}, jerks{condition_idx}] = ...
        compute_condition_metrics(trials{condition_idx}, ...
        reference_xy_m, moving_speed_threshold_m_s, ...
        jerk_outlier_window_s, jerk_smoothing_window_s, ...
        jerk_sensitivity_windows_s, reference_shift_search_spacing_m);
end

%% ================================================================
% Figure 1: TCP path per condition, colored by interaction force
% =================================================================

figure('Name', 'TCP path comparison (all six)', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

force_color_max_n = 0;
for condition_idx = 1:condition_count
    force_color_max_n = max(force_color_max_n, ...
        max(hypot(trials{condition_idx}.force_x_n, ...
        trials{condition_idx}.force_y_n)));
end
force_color_limits_n = [0, force_color_max_n];

for condition_idx = 1:condition_count
    nexttile;
    trial = trials{condition_idx};
    force_magnitude_n = hypot(trial.force_x_n, trial.force_y_n);

    scatter(trial.xy_m(:, 1), trial.xy_m(:, 2), trajectory_marker_size, ...
        force_magnitude_n, 'filled');
    if ~isempty(reference_xy_m)
        hold on;
        plot(reference_xy_m(:, 1), reference_xy_m(:, 2), 'k--', ...
            'LineWidth', 0.75);
        hold off;
    end
    axis equal;
    grid on;
    clim(force_color_limits_n);
    title(condition_names{condition_idx});
    xlabel('CoppeliaSim x [m]');
    ylabel('CoppeliaSim y [m]');
end

force_colorbar = colorbar;
force_colorbar.Layout.Tile = 'east';
force_colorbar.Label.String = 'Interaction force |F| [N]';
sgtitle(['TCP path colored by interaction force, all six conditions ' ...
    '(shared color axis; dashed line is the reference centerline)']);

%% ================================================================
% Figure 2: force and jerk magnitude over time, all six overlaid
% =================================================================

% figure('Name', 'Force and jerk over time (all six)', 'Color', 'w');
% tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
% nexttile;
% hold on;
% for condition_idx = 1:condition_count
%     trial = trials{condition_idx};
%     plot(trial.time_s, hypot(trial.force_x_n, trial.force_y_n), '-', ...
%         'Color', condition_colors(condition_idx, :), 'LineWidth', 0.8, ...
%         'DisplayName', condition_names{condition_idx});
% end
% grid on;
% xlabel('Time [s]');
% ylabel('|F| [N]');
% title('Interaction force magnitude over time');
% legend('Location', 'eastoutside');
% 
% nexttile;
% hold on;
% for condition_idx = 1:condition_count
%     jerk = jerks{condition_idx};
%     plot(jerk.time_s, sqrt(jerk.squared_jerk_m2_s6), '-', ...
%         'Color', condition_colors(condition_idx, :), 'LineWidth', 0.8, ...
%         'DisplayName', condition_names{condition_idx});
% end
% grid on;
% xlabel('Time [s]');
% ylabel('|jerk| [m/s^3]');
% title(sprintf('Jerk magnitude (smoothing window %.0f ms)', ...
%     1e3 * jerk_smoothing_window_s));
% legend('Location', 'eastoutside');
% 
% sgtitle('Force and jerk over time, all six conditions');

%% ================================================================
% Figure 3: summary bar charts
% =================================================================

duration_s = cellfun(@(m) m.duration_s, metrics);
path_length_m = cellfun(@(m) m.path_length_m, metrics);
mean_force_n = cellfun(@(m) m.mean_force_n, metrics);
dsj = cellfun(@(m) m.dsj, metrics);
mean_error_mm = cellfun(@(m) m.mean_error_mm, metrics);
moving_percent = 100 * cellfun(@(m) m.moving_fraction, metrics);

bar_labels = categorical(condition_names);
bar_labels = reordercats(bar_labels, condition_names);

figure('Name', 'Summary metrics (all six)', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
bar(bar_labels, duration_s, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('Duration [s]');
title('Execution time');

nexttile;
bar(bar_labels, path_length_m, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('Path length [m]');
title('TCP path length');

nexttile;
bar(bar_labels, mean_force_n, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('Mean |F| [N]');
title('Mean force, Eq. (17) (moving samples only)');

nexttile;
bar(bar_labels, dsj, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('DSJ [-]');
title(sprintf('DSJ, Eq. (16), at %.0f ms window', ...
    1e3 * jerk_smoothing_window_s));

nexttile;
bar(bar_labels, mean_error_mm, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('Mean error [mm]');
if isempty(reference_xy_m)
    title('Accuracy, Eq. (15) -- unavailable (see command window)');
else
    title('Accuracy, Eq. (15) (moving samples only)');
end

nexttile;
bar(bar_labels, moving_percent, 'FaceColor', 'flat', 'CData', condition_colors);
grid on;
ylabel('Moving samples [%]');
title(sprintf('Time above %.0f mm/s', 1e3 * moving_speed_threshold_m_s));

sgtitle('Summary metrics, all six conditions, fine_gross_shape task');

%% ================================================================
% Summary table (command window)
% =================================================================

fprintf('\n--- All six controllers, fine_gross_shape task, s0 ---\n');
fprintf('%-26s', '');
for condition_idx = 1:condition_count
    fprintf('%16s', condition_names{condition_idx});
end
fprintf('\n');

print_row('Duration [s]', '%16.2f', duration_s);
print_row('Samples', '%16d', cellfun(@(m) m.sample_count, metrics));
print_row('Mean rate [Hz]', '%16.1f', cellfun(@(m) m.mean_rate_hz, metrics));
print_row('Path length [m]', '%16.4f', path_length_m);
print_row('Moving samples [%]', '%16.1f', moving_percent);
print_row('Mean speed [m/s]', '%16.4f', ...
    cellfun(@(m) m.mean_speed_m_s, metrics));
fprintf('\n');
print_row('Mean force Eq.(17) [N]', '%16.3f', mean_force_n);
print_row('DSJ Eq.(16) [-]', '%16.4g', dsj);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    sensitivity_values = cellfun( ...
        @(m) m.dsj_sensitivity(window_idx), metrics);
    print_row(sprintf('  at %3.0f ms window', ...
        1e3 * jerk_sensitivity_windows_s(window_idx)), '%16.4g', ...
        sensitivity_values);
end
fprintf('  DSJ is filter-dependent; only compare at the same window.\n');
fprintf('\n');

if isempty(reference_xy_m)
    fprintf(['Accuracy Eq.(15): unavailable for all conditions (see ' ...
        'warning above).\n']);
else
    print_row('Mean error Eq.(15) [mm]', '%16.2f', mean_error_mm);
    print_row('Reference-shift residual [mm]', '%16.2f', ...
        1e3 * cellfun(@(m) m.reference_shift_error_m, metrics));
    fprintf(['  Reference-shift residual is a diagnostic only (never ' ...
        'applied to the reported error): a large value for one ' ...
        'condition (most likely gaze-assisted, the only condition ' ...
        'reached through an extra frame transform) means shape_offset_' ...
        'xy_m or the transform settings may not match that trial.\n']);
end
fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================

function trial = load_controller_trial(csv_file, position_mode, ...
        axis_from_robot_1based, sign_from_robot, offset_xy_m)
%LOAD_CONTROLLER_TRIAL Read one controller CSV into a uniform struct.
%
% Every controller in this project logs time_s, Fx_filt_N/Fy_filt_N,
% and tcp_vx_actual_m_s/tcp_vy_actual_m_s under identical names. TCP
% position is either read directly from coppeliasim_x_m/coppeliasim_y_m
% (position_mode = 'coppeliasim') or mapped from robot-base-frame
% x_m/y_m (position_mode = 'robot_transform', currently only the
% gaze-assisted controller, which logs no coppeliasim_* columns).

    data_table = readtable(csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');

    trial.time_s = get_numeric_column(data_table.time_s);
    trial.force_x_n = get_numeric_column(data_table.Fx_filt_N);
    trial.force_y_n = get_numeric_column(data_table.Fy_filt_N);
    trial.speed_m_s = hypot( ...
        get_numeric_column(data_table.tcp_vx_actual_m_s), ...
        get_numeric_column(data_table.tcp_vy_actual_m_s));
    trial.sample_count = height(data_table);

    if strcmp(position_mode, 'coppeliasim')
        trial.xy_m = [get_numeric_column(data_table.coppeliasim_x_m), ...
            get_numeric_column(data_table.coppeliasim_y_m)];
    else
        robot_xy_m = [get_numeric_column(data_table.x_m), ...
            get_numeric_column(data_table.y_m)];
        trial.xy_m = map_robot_xy_to_coppeliasim(robot_xy_m, ...
            axis_from_robot_1based, sign_from_robot, offset_xy_m);
    end
end


function [condition_metrics, jerk] = compute_condition_metrics( ...
        trial, reference_xy_m, moving_speed_threshold_m_s, ...
        jerk_outlier_window_s, jerk_smoothing_window_s, ...
        jerk_sensitivity_windows_s, reference_shift_search_spacing_m)
%COMPUTE_CONDITION_METRICS Paper performance metrics for one condition.
% Movement samples only (speed > moving_speed_threshold_m_s), as the
% paper does throughout this project's scripts.

    moving_mask = trial.speed_m_s > moving_speed_threshold_m_s;
    force_magnitude_n = hypot(trial.force_x_n, trial.force_y_n);

    condition_metrics.duration_s = trial.time_s(end) - trial.time_s(1);
    condition_metrics.sample_count = trial.sample_count;
    condition_metrics.mean_rate_hz = ...
        (trial.sample_count - 1) / condition_metrics.duration_s;
    condition_metrics.path_length_m = sum(hypot( ...
        diff(trial.xy_m(:, 1)), diff(trial.xy_m(:, 2))));
    condition_metrics.moving_fraction = mean(moving_mask);
    condition_metrics.mean_speed_m_s = mean(trial.speed_m_s);
    condition_metrics.mean_force_n = mean(force_magnitude_n(moving_mask));

    jerk = compute_jerk_series(trial.time_s, trial.xy_m(:, 1), ...
        trial.xy_m(:, 2), jerk_outlier_window_s, jerk_smoothing_window_s);
    condition_metrics.dsj = compute_dsj( ...
        jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m);

    condition_metrics.dsj_sensitivity = ...
        zeros(numel(jerk_sensitivity_windows_s), 1);
    for window_idx = 1:numel(jerk_sensitivity_windows_s)
        sensitivity_jerk = compute_jerk_series(trial.time_s, ...
            trial.xy_m(:, 1), trial.xy_m(:, 2), jerk_outlier_window_s, ...
            jerk_sensitivity_windows_s(window_idx));
        condition_metrics.dsj_sensitivity(window_idx) = compute_dsj( ...
            sensitivity_jerk.time_s, sensitivity_jerk.squared_jerk_m2_s6, ...
            sensitivity_jerk.path_length_m);
    end

    if isempty(reference_xy_m)
        condition_metrics.mean_error_mm = NaN;
        condition_metrics.reference_shift_error_m = NaN;
    else
        tracking_error_m = distance_to_reference_path( ...
            trial.xy_m(:, 1), trial.xy_m(:, 2), reference_xy_m);
        condition_metrics.mean_error_mm = ...
            1e3 * mean(tracking_error_m(moving_mask));

        [~, condition_metrics.reference_shift_error_m] = ...
            estimate_reference_shift(trial.xy_m(:, 1), trial.xy_m(:, 2), ...
            reference_xy_m, reference_shift_search_spacing_m);
    end
end


function print_row(row_label, value_format, values)
%PRINT_ROW Print one summary-table row across all conditions.

    fprintf('%-26s', row_label);
    for value_idx = 1:numel(values)
        fprintf(value_format, values(value_idx));
    end
    fprintf('\n');
end


function coppelia_xy_m = map_robot_xy_to_coppeliasim( ...
        robot_xy_m, axis_from_robot_1based, sign_from_robot, offset_xy_m)
%MAP_ROBOT_XY_TO_COPPELIASIM Port of scripts/ur5e_gaze_assisted_adm_2d.py's
% map_tcp_position_to_coppeliasim(), planar only. Copied verbatim from
% compare_chen_vs_gaze_assisted_2d.m.
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


function distance_m = distance_to_reference_path( ...
        path_x_m, path_y_m, reference_xy_m)
%DISTANCE_TO_REFERENCE_PATH Nearest-reference-sample distance [m]. Copied
% verbatim from plot_raw_data_fixed_adm_2d.m.

    distance_m = inf(numel(path_x_m), 1);

    for reference_idx = 1:size(reference_xy_m, 1)
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        distance_m = min(distance_m, candidate_m);
    end
end


function [best_shift_xy_m, best_error_m] = estimate_reference_shift( ...
        path_x_m, path_y_m, reference_xy_m, search_spacing_m)
%ESTIMATE_REFERENCE_SHIFT Extra translation that would minimize error.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.
%
% Diagnostic only, never applied to the reported accuracy: fitting the
% reference to the data would bias the metric. It exists so a bad
% shape_offset_xy_m (or, here, a bad gaze-assisted frame transform)
% shows up as a large residual shift instead of silently inflating the
% error.
%
% Coarse grid followed by one refinement, both centred on zero shift.

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
