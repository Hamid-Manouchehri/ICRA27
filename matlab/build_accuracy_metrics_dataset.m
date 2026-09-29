clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));

data_directory = fullfile(script_directory, '..', 'data');

%% Settings
manifest_file = fullfile(data_directory, 'trial_manifest.csv');

raw_data_directory = fullfile( ...
    data_directory, 'raw_data', 'test_subjects');

shape_directory = fullfile( ...
    data_directory, 'shapes');

output_csv = fullfile( ...
    data_directory, 'processed_data', ...
    'tracking_metrics_all_trials.csv');

output_mat = fullfile( ...
    data_directory, 'processed_data', ...
    'tracking_metrics_all_trials.mat');

minimum_speed_m_s = 0.005;

%% Load manifest
M = readtable(manifest_file, ...
    'TextType', 'string');

n_trials = height(M);

results = cell(n_trials, 1);

%% ================================================================
% Process every trial
% =================================================================
for i = 1:n_trials

    fprintf('\nProcessing trial %d/%d\n', i, n_trials);

    robot_file = fullfile( ...
        raw_data_directory, M.robot_csv(i));

    shape_file = fullfile( ...
        shape_directory, M.shape_obj(i));

    T = readtable(robot_file);

    %% ------------------------------------------------------------
    % Reference geometry
    % -------------------------------------------------------------
    [outer, inner, center] = ...
        read_shape_band_obj(shape_file, 5e-4);

    offset = [M.offset_x(i), M.offset_y(i)];

    center = center + offset;
    outer  = outer  + offset;
    inner  = inner  + offset;

    half_width = hypot( ...
        outer(:,1) - center(:,1), ...
        outer(:,2) - center(:,2));

    %% ------------------------------------------------------------
    % Robot signals
    % -------------------------------------------------------------
    t  = T.time_s;

    x  = T.coppeliasim_x_m;
    y  = T.coppeliasim_y_m;

    vx = T.tcp_vx_actual_m_s;
    vy = T.tcp_vy_actual_m_s;

    speed = hypot(vx, vy);

    %% Optional movement gate
    valid = ...
        isfinite(t) & ...
        isfinite(x) & ...
        isfinite(y) & ...
        isfinite(vx) & ...
        isfinite(vy) & ...
        speed >= minimum_speed_m_s;

    t  = t(valid);
    x  = x(valid);
    y  = y(valid);
    vx = vx(valid);
    vy = vy(valid);
    speed = speed(valid);

    %% ------------------------------------------------------------
    % Centerline tracking
    % -------------------------------------------------------------
    [distance_m, path_angle, centerline_position] = ...
        nearest_centerline_distance_and_heading( ...
        x, y, center);

    tcp_angle = atan2(vy, vx);

    angular_error_rad = ...
        signed_angle_difference( ...
        tcp_angle, path_angle);

    angular_error_deg = ...
        abs(rad2deg(angular_error_rad));

    %% ------------------------------------------------------------
    % Corridor-normalized accuracy
    % -------------------------------------------------------------
    d_norm = normalized_corridor_distance( ...
        distance_m, centerline_position, half_width);

    in_band_percent = ...
        100 * mean(d_norm <= 1);

    violation = max(d_norm - 1, 0);

    outside = d_norm > 1;

    if any(outside)
        mean_violation_when_outside = ...
            mean(violation(outside));

        max_violation = ...
            max(violation(outside));
    else
        mean_violation_when_outside = 0;
        max_violation = 0;
    end

    % Useful because it includes BOTH frequency and severity
    mean_violation_overall = mean(violation);

    %% ------------------------------------------------------------
    % Basic trajectory quantities
    % -------------------------------------------------------------
    duration_s = t(end) - t(1);

    path_length_m = sum( ...
        hypot(diff(x), diff(y)));

    mean_speed_m_s = mean(speed);

    %% ------------------------------------------------------------
    % Tracking error statistics
    % -------------------------------------------------------------
    mean_error_mm = ...
        1e3 * mean(distance_m);

    median_error_mm = ...
        1e3 * median(distance_m);

    p90_error_mm = ...
        1e3 * prctile(distance_m, 90);

    max_error_mm = ...
        1e3 * max(distance_m);

    mean_norm_distance = mean(d_norm);
    p90_norm_distance  = prctile(d_norm, 90);
    max_norm_distance  = max(d_norm);

    %% ------------------------------------------------------------
    % Angular tracking
    % -------------------------------------------------------------
    mean_heading_error_deg = ...
        mean(angular_error_deg);

    p90_heading_error_deg = ...
        prctile(angular_error_deg, 90);

    %% ------------------------------------------------------------
    % DSJ
    % -------------------------------------------------------------
    % Full, continuous trial -- NOT the speed-masked x/y/t above. DSJ
    % triple-differentiates position; removing low-speed samples first
    % opens time gaps, and differentiating across a gap produces a
    % spurious jerk spike there (matches compare_chen_vs_gaze_assisted_
    % performance.m's compute_jerk_series, which is likewise called on
    % the unmasked series). calculate_planar_dsj_local does its own
    % isfinite() filtering internally, so this still drops real NaNs.
    dsj = calculate_planar_dsj_local( ...
        T.coppeliasim_x_m, T.coppeliasim_y_m, T.time_s);

    %% ------------------------------------------------------------
    % Interaction force
    % -------------------------------------------------------------
    mean_force_N = NaN;

    if all(ismember( ...
            {'Fx_filt_N','Fy_filt_N'}, ...
            T.Properties.VariableNames))

        Fx = T.Fx_filt_N(valid);
        Fy = T.Fy_filt_N(valid);

        mean_force_N = ...
            mean(hypot(Fx, Fy), 'omitnan');
    end

    %% ------------------------------------------------------------
    % One ROW = one trial
    % -------------------------------------------------------------
    results{i} = table( ...
        M.subject(i), ...
        M.controller(i), ...
        M.shape(i), ...
        M.bandwidth(i), ...
        M.direction(i), ...
        M.trial(i), ...
        duration_s, ...
        path_length_m, ...
        mean_speed_m_s, ...
        mean_error_mm, ...
        median_error_mm, ...
        p90_error_mm, ...
        max_error_mm, ...
        mean_norm_distance, ...
        p90_norm_distance, ...
        max_norm_distance, ...
        in_band_percent, ...
        mean_violation_overall, ...
        mean_violation_when_outside, ...
        max_violation, ...
        mean_heading_error_deg, ...
        p90_heading_error_deg, ...
        dsj, ...
        mean_force_N, ...
        'VariableNames', { ...
        'subject', ...
        'controller', ...
        'shape', ...
        'bandwidth', ...
        'direction', ...
        'trial', ...
        'duration_s', ...
        'path_length_m', ...
        'mean_speed_m_s', ...
        'mean_error_mm', ...
        'median_error_mm', ...
        'p90_error_mm', ...
        'max_error_mm', ...
        'mean_norm_distance', ...
        'p90_norm_distance', ...
        'max_norm_distance', ...
        'in_band_percent', ...
        'mean_violation_overall', ...
        'mean_violation_when_outside', ...
        'max_violation', ...
        'mean_heading_error_deg', ...
        'p90_heading_error_deg', ...
        'DSJ', ...
        'mean_force_N'});

end

%% ================================================================
% Combine all trials
% =================================================================
trial_metrics = vertcat(results{:});

writetable(trial_metrics, output_csv);

save(output_mat, 'trial_metrics');

fprintf('\nSaved %d trials:\n%s\n', ...
    height(trial_metrics), output_csv);


%% ================================================================
% Subject-condition averages
%
% Multiple repeated trials from the same subject/condition become
% ONE subject-level observation.
% =================================================================

metric_names = { ...
    'duration_s', ...
    'mean_error_mm', ...
    'mean_norm_distance', ...
    'in_band_percent', ...
    'mean_violation_overall', ...
    'mean_heading_error_deg', ...
    'DSJ', ...
    'mean_force_N'};

subject_condition_metrics = groupsummary( ...
    trial_metrics, ...
    {'subject','controller','shape','bandwidth','direction'}, ...
    'mean', ...
    metric_names);

subject_output = fullfile( ...
    data_directory, 'processed_data', ...
    'tracking_metrics_subject_condition.csv');

writetable(subject_condition_metrics, ...
    subject_output);

fprintf('Saved subject-condition dataset:\n%s\n', ...
    subject_output);


%% ================================================================
% Simple population visualization
% =================================================================

metrics_to_plot = { ...
    'mean_error_mm', ...
    'in_band_percent', ...
    'mean_violation_overall', ...
    'DSJ', ...
    'mean_force_N'};

for m = 1:numel(metrics_to_plot)

    metric = metrics_to_plot{m};

    figure('Color','w');

    boxchart( ...
        categorical(trial_metrics.controller), ...
        trial_metrics.(metric));

    ylabel(strrep(metric, '_', ' '));
    xlabel('Controller');

    title(strrep(metric, '_', ' '));

    grid on;

end


%% ================================================================
% Local DSJ
% =================================================================
function dsj = calculate_planar_dsj_local(x, y, t)

    x = double(x(:));
    y = double(y(:));
    t = double(t(:));

    valid = isfinite(x) & ...
            isfinite(y) & ...
            isfinite(t);

    x = x(valid);
    y = y(valid);
    t = t(valid);

    if numel(t) < 4
        dsj = NaN;
        return;
    end

    vx = gradient(x, t);
    vy = gradient(y, t);

    ax = gradient(vx, t);
    ay = gradient(vy, t);

    jx = gradient(ax, t);
    jy = gradient(ay, t);

    jerk_squared = jx.^2 + jy.^2;

    duration = t(end) - t(1);

    path_length = sum( ...
        hypot(diff(x), diff(y)));

    if duration <= 0 || ...
       path_length <= 1e-12

        dsj = NaN;
        return;
    end

    dsj = duration^5 / path_length^2 * ...
        trapz(t, jerk_squared);

end