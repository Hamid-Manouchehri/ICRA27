%% Performance features per trial, Chen vs Gated, saved to a .mat.
% One row per manifest trial. Features are added one at a time and cross-checked.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (per manifest row)
%
% Outputs:
%   - data/processed_data/performance_features_all_trials.mat: trial_features
%     (table, one row per trial) plus the four *_label_L code lists
%
% Depends on:
%   - functions/sparc_fun.m
%   - functions/fine_mask_fun.m
%   - functions/sparc_fine_fun.m
%   - functions/jerk_series_fun.m
%   - functions/compute_dsj.m
%   - functions/dsj_fine_fun.m
%   - functions/mean_force_fun.m
%   - functions/path_efficiency_fun.m
%   - functions/traveled_path_fun.m
%   - functions/boundary_violations_fun.m
%   - functions/angular_divergence_fun.m
%   - functions/signed_angle_difference.m
%   - functions/nearest_centerline_distance_and_heading.m
%   - functions/normalized_corridor_distance.m
%   - functions/read_shape_band_obj.m

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
raw_data_directory = fullfile(data_directory, 'raw_data', 'test_subjects');
shape_directory = fullfile(data_directory, 'shapes');

% Code lists: a trial is saved as its 1-based index into these.
controller_label_L = {"Chen", "Gated"};  % Chen:1, Gated: 2
shape_name_L = {"fine_gross", "pear", "ellipse"};  % fine_gross:1, pear: 2, ellipse: 3
bandwidth_label_L = {"wide", "narrow"};  % wide: 1, narrow: 2
direction_label_L = {"clockwise", "counter_clockwise"};  % clockwise: 1, counter_clockwise: 2

sparc_padlevel = 4;  % zero-padding level
sparc_cutoff_hz = 10.0;  % max cutoff frequency [Hz]
sparc_amp_threshold = 0.05;  % adaptive cutoff amplitude threshold
sparc_min_pass_s = 1.0;  % shortest fine-area pass used for SPARC [s]

jerk_outlier_window_s = 0.02;  % jerk median-filter window [s] (as in ET.m)
jerk_smoothing_window_s = 0.10;  % jerk moving-mean window [s] (as in ET.m)
dsj_min_pass_s = 1.0;  % shortest fine-area pass used for DSJ [s]

moving_speed_threshold_m_s = 0.005;  % mean force uses samples faster than this [m/s]
reference_spacing_m = 5e-4;  % centerline resample spacing [m]
violation_min_s = 0.05;  % shortest out-of-band excursion counted [s]
theta_station_spacing_m = 5e-3;  % centerline distance between consecutive points for angular divergence [m]

write_features_mat = fullfile(data_directory, 'processed_data', ...
    'performance_features_all_trials.mat');

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\\*.csv\n', manifest_file, raw_data_directory);
fprintf('writing:\n  %s\n', write_features_mat);

M = readtable(manifest_file, 'TextType', 'string');
n_trials = height(M);

% Identifiers, encoded with numbers.
subject_id = M.subject;
trial = M.trial;
[~, controller_index] = ismember(M.controller, string(controller_label_L));
[~, shape_index] = ismember(M.shape, string(shape_name_L));
[~, bandwidth_index] = ismember(M.bandwidth, string(bandwidth_label_L));
[~, direction_index] = ismember(M.direction, string(direction_label_L));
assert(all([controller_index; shape_index; bandwidth_index; direction_index] > 0), ...
    'A manifest label is missing from its code list in User Setting.');

time_s = cell(n_trials, 1);
tcp_xy_m = cell(n_trials, 1);
tcp_v_xy_m_s = cell(n_trials, 1);
sparc = nan(n_trials, 1);
sparc_fine = nan(n_trials, 1);
sparc_fine_passes = cell(n_trials, 1);
fine_pass_duration_s = cell(n_trials, 1);
dsj = nan(n_trials, 1);
dsj_fine = nan(n_trials, 1);
dsj_fine_passes = cell(n_trials, 1);
mean_force_n = nan(n_trials, 1);
path_efficiency = nan(n_trials, 1);
traveled_path_m = nan(n_trials, 1);
n_boundary_violations = nan(n_trials, 1);
violation_duration_s = cell(n_trials, 1);
angular_divergence_deg = nan(n_trials, 1);
angular_divergence_sq_deg2 = nan(n_trials, 1);
theta_deg = cell(n_trials, 1);
positive_progress = nan(n_trials, 1);
total_progress_m = nan(n_trials, 1);
horizon_angle_deg = nan(n_trials, 1);
horizon_angle_sq_deg2 = nan(n_trials, 1);
tcp_heading_deg = cell(n_trials, 1);

% Centerline and corridor half-width of each shape file, in the CoppeliaSim
% frame (read once per file).
shape_geometry_cache = containers.Map();

for i = 1:n_trials
    fprintf('Processing trial %d/%d\n', i, n_trials);

    T = readtable(fullfile(raw_data_directory, M.robot_csv(i)));

    % Recording time and robot TCP data (CoppeliaSim frame).
    time_s{i} = T.time_s;
    tcp_xy_m{i} = [T.coppeliasim_x_m, T.coppeliasim_y_m];
    tcp_v_xy_m_s{i} = [T.tcp_vx_actual_m_s, T.tcp_vy_actual_m_s];

    % Smoothness of the TCP speed profile.
    speed_m_s = hypot(T.tcp_vx_actual_m_s, T.tcp_vy_actual_m_s);
    fs_hz = 1 / median(diff(T.time_s));
    sparc(i) = sparc_fun(speed_m_s, fs_hz, sparc_padlevel, ...
        sparc_cutoff_hz, sparc_amp_threshold);

    % Smoothness inside the fine area only, one SPARC per pass.
    fine_mask = fine_mask_fun(T.coppeliasim_x_m, shape_index(i));
    [sparc_fine(i), sparc_fine_passes{i}, fine_pass_duration_s{i}] = ...
        sparc_fine_fun(speed_m_s, fine_mask, fs_hz, sparc_min_pass_s, ...
        sparc_padlevel, sparc_cutoff_hz, sparc_amp_threshold);

    % Dimensionless squared jerk, whole trial and inside the fine area.
    [time_u_s, x_u_m, y_u_m, squared_jerk_m2_s6] = jerk_series_fun( ...
        T.time_s, T.coppeliasim_x_m, T.coppeliasim_y_m, ...
        T.tcp_vx_actual_m_s, T.tcp_vy_actual_m_s, ...
        jerk_outlier_window_s, jerk_smoothing_window_s);
    dsj(i) = compute_dsj(time_u_s, squared_jerk_m2_s6, ...
        sum(hypot(diff(x_u_m), diff(y_u_m))));
    fine_mask_u = fine_mask_fun(x_u_m, shape_index(i));
    [dsj_fine(i), dsj_fine_passes{i}] = dsj_fine_fun(time_u_s, x_u_m, y_u_m, ...
        squared_jerk_m2_s6, fine_mask_u, dsj_min_pass_s);

    % Mean interaction force while moving.
    mean_force_n(i) = mean_force_fun(T.Fx_filt_N, T.Fy_filt_N, speed_m_s, ...
        moving_speed_threshold_m_s);

    % Shape geometry of this trial.
    shape_key = char(M.shape_obj(i));
    if ~isKey(shape_geometry_cache, shape_key)
        [outer_xy_m, ~, center_xy_m] = read_shape_band_obj( ...
            fullfile(shape_directory, M.shape_obj(i)), reference_spacing_m);
        offset_xy_m = [M.offset_x(i), M.offset_y(i)];
        center_xy_m = center_xy_m + offset_xy_m;
        outer_xy_m = outer_xy_m + offset_xy_m;
        half_width_m = hypot(outer_xy_m(:, 1) - center_xy_m(:, 1), ...
            outer_xy_m(:, 2) - center_xy_m(:, 2));
        shape_geometry_cache(shape_key) = struct('center', center_xy_m, ...
            'half_width', half_width_m);
    end
    geometry = shape_geometry_cache(shape_key);

    % Net progress along the centerline per metre of TCP path.
    path_efficiency(i) = path_efficiency_fun(T.coppeliasim_x_m, ...
        T.coppeliasim_y_m, geometry.center);

    % Distance the TCP travelled.
    traveled_path_m(i) = traveled_path_fun(T.coppeliasim_x_m, T.coppeliasim_y_m);

    % Times the TCP left the band and came back.
    [distance_m, ~, centerline_position] = nearest_centerline_distance_and_heading( ...
        T.coppeliasim_x_m, T.coppeliasim_y_m, geometry.center);
    normalized_distance = normalized_corridor_distance(distance_m, ...
        centerline_position, geometry.half_width);
    [n_boundary_violations(i), violation_duration_s{i}] = boundary_violations_fun( ...
        normalized_distance, T.time_s, violation_min_s);

    % Angle between the TCP path and the centerline over consecutive points.
    [theta_deg{i}, tcp_chord_length_m, center_chord_length_m, tcp_heading_deg{i}] = ...
        angular_divergence_fun(T.coppeliasim_x_m, T.coppeliasim_y_m, ...
        centerline_position, geometry.center, theta_station_spacing_m);
    angular_divergence_deg(i) = mean(abs(theta_deg{i}));
    angular_divergence_sq_deg2(i) = mean(theta_deg{i} .^ 2);

    % Progress per interval: TCP displacement times cos|theta|.
    progress_m = tcp_chord_length_m .* cosd(abs(theta_deg{i}));
    positive_progress(i) = mean(progress_m ./ center_chord_length_m);
    total_progress_m(i) = sum(progress_m);

    % Angle of the TCP motion with the horizontal axis, same intervals.
    horizon_angle_deg(i) = mean(abs(tcp_heading_deg{i}));
    horizon_angle_sq_deg2(i) = mean(tcp_heading_deg{i} .^ 2);
end

trial_features = table(subject_id, trial, controller_index, shape_index, ...
    bandwidth_index, direction_index, time_s, tcp_xy_m, tcp_v_xy_m_s, sparc, ...
    sparc_fine, sparc_fine_passes, fine_pass_duration_s, dsj, dsj_fine, ...
    dsj_fine_passes, mean_force_n, path_efficiency, traveled_path_m, ...
    n_boundary_violations, violation_duration_s, angular_divergence_deg, ...
    angular_divergence_sq_deg2, theta_deg, positive_progress, total_progress_m, ...
    horizon_angle_deg, horizon_angle_sq_deg2, tcp_heading_deg);

save(write_features_mat, 'trial_features', 'controller_label_L', ...
    'shape_name_L', 'bandwidth_label_L', 'direction_label_L');

fprintf('\nSaved %d trials:\n%s\n', n_trials, write_features_mat);
