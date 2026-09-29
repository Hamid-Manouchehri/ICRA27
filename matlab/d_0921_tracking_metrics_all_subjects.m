%% Per-trial tracking time series for every trial in the manifest, saved to a .mat.
% Same geometry and metric functions as d_0921_tracking_metrics_chen_vs_gated.m, no plots.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (per manifest row)
%   - data/shapes/<shape_obj> (per manifest row)
%
% Outputs:
%   - data/processed_data/tracking_metrics_all_subjects.mat, N trials:
%     subject_id, trial, controller_index, shape_index, bandwidth_index,
%     direction_index (N-by-1) plus the four *_label_L code lists;
%     time_s, tcp_x_m, tcp_y_m, tcp_vx_m_s, tcp_vy_m_s,
%     dist_to_centerline_m, dist_to_centerline_norm,
%     angular_divergence_rad, dist_rate_m_s,
%     angular_divergence_rate_rad_s (N-by-samples_per_trial);
%     in_band_percent (N-by-1); boundary_violation (N-by-1 struct)
%
% Depends on:
%   - trial_manifest.csv
%   - functions/read_shape_band_obj.m
%   - functions/nearest_centerline_distance_and_heading.m
%   - functions/signed_angle_difference.m
%   - functions/normalized_corridor_distance.m
%   - functions/rate_of_change.m
%   - functions/in_band_accuracy.m
%   - functions/boundary_violation_magnitude.m

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

reference_spacing_m = 5e-4;  % TODO centerline resample spacing [m]

samples_per_trial = 30000;  % TODO rows in every robot CSV; another length errors when stored

% TODO code lists: each trial is saved as its 1-based index into these.
% shape/bandwidth order matches ET.m, so codes line up with ET_data.mat.
controller_label_L = {"Chen", "Gated"};
shape_name_L = {"fine_gross", "pear", "ellipse"};
bandwidth_label_L = {"wide", "narrow"};
direction_label_L = {"clockwise", "counter_clockwise"};

write_metrics_mat = fullfile(data_directory, 'processed_data', ...
    'accuracy_metrics_exploration_all_subjects.mat');

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\\*.csv\n  %s\\*.obj\n', ...
    manifest_file, raw_data_directory, shape_directory);
fprintf('writing:\n  %s\n', write_metrics_mat);

M = readtable(manifest_file, 'TextType', 'string');
n_trials = height(M);

subject_id = M.subject;
trial = M.trial;
[~, controller_index] = ismember(M.controller, string(controller_label_L));
[~, shape_index] = ismember(M.shape, string(shape_name_L));
[~, bandwidth_index] = ismember(M.bandwidth, string(bandwidth_label_L));
[~, direction_index] = ismember(M.direction, string(direction_label_L));
assert(all([controller_index; shape_index; bandwidth_index; direction_index] > 0), ...
    'A manifest label is missing from its code list in User Setting.');

[time_s, tcp_x_m, tcp_y_m, tcp_vx_m_s, tcp_vy_m_s, ...
    dist_to_centerline_m, dist_to_centerline_norm, angular_divergence_rad, ...
    dist_rate_m_s, angular_divergence_rate_rad_s] = ...
    deal(nan(n_trials, samples_per_trial));
in_band_percent = nan(n_trials, 1);

% Shape geometry (centerline + local corridor half-width) depends only
% on shape_obj, not on subject/controller -- cached so each distinct
% shape file is resampled once instead of once per trial.
shape_geometry_cache = containers.Map();

for i = 1:n_trials
    fprintf('Processing trial %d/%d\n', i, n_trials);

    robot_file = fullfile(raw_data_directory, M.robot_csv(i));
    shape_key = char(M.shape_obj(i));

    if ~isKey(shape_geometry_cache, shape_key)
        shape_file = fullfile(shape_directory, M.shape_obj(i));
        [outer, ~, center] = read_shape_band_obj(shape_file, reference_spacing_m);
        offset = [M.offset_x(i), M.offset_y(i)];
        center = center + offset;
        outer  = outer  + offset;
        half_w = hypot(outer(:,1)-center(:,1), outer(:,2)-center(:,2));
        shape_geometry_cache(shape_key) = struct('center', center, 'half_w', half_w);
    end
    geometry = shape_geometry_cache(shape_key);
    center = geometry.center;
    half_w = geometry.half_w;

    T = readtable(robot_file);
    x  = T.coppeliasim_x_m;  % CoppeliaSim frame, same as the shape mesh
    y  = T.coppeliasim_y_m;
    vx = T.tcp_vx_actual_m_s;
    vy = T.tcp_vy_actual_m_s;
    t  = T.time_s;

    [d, path_angle, nearest_index] = nearest_centerline_distance_and_heading(x, y, center);

    tcp_angle = atan2(vy, vx);  % TCP pose taken as direction of travel (no orientation column logged)
    ang_div = signed_angle_difference(tcp_angle, path_angle);

    d_norm = normalized_corridor_distance(d, nearest_index, half_w);

    time_s(i, :) = t';
    tcp_x_m(i, :) = x';
    tcp_y_m(i, :) = y';
    tcp_vx_m_s(i, :) = vx';
    tcp_vy_m_s(i, :) = vy';
    dist_to_centerline_m(i, :) = d';
    dist_to_centerline_norm(i, :) = d_norm';
    angular_divergence_rad(i, :) = ang_div';
    dist_rate_m_s(i, :) = rate_of_change(d, t)';
    angular_divergence_rate_rad_s(i, :) = rate_of_change(ang_div, t)';
    in_band_percent(i) = in_band_accuracy(d_norm);
    boundary_violation(i) = boundary_violation_magnitude(d_norm);
end

save(write_metrics_mat, ...
    'subject_id', 'trial', 'controller_index', 'shape_index', ...
    'bandwidth_index', 'direction_index', ...
    'controller_label_L', 'shape_name_L', 'bandwidth_label_L', 'direction_label_L', ...
    'time_s', 'tcp_x_m', 'tcp_y_m', 'tcp_vx_m_s', 'tcp_vy_m_s', ...
    'dist_to_centerline_m', 'dist_to_centerline_norm', 'angular_divergence_rad', ...
    'dist_rate_m_s', 'angular_divergence_rate_rad_s', ...
    'in_band_percent', 'boundary_violation');

fprintf('\nSaved %d trials:\n%s\n', n_trials, write_metrics_mat);
