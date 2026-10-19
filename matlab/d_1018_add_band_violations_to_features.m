%% Boundary violations against the drawn band, added to trial_features.
% A sample is outside when the TCP is outside the outer edge or inside the
% inner edge of the shape mesh (no centerline). Excursions are counted with
% boundary_violations_fun, the same rule as the replay's red numbers.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (all trials)
%   - data/shapes/<shape_obj> (band edges)
%   - data/processed_data/performance_features_all_trials.mat
%
% Outputs:
%   - same .mat, trial_features gets n_band_violations and
%     band_violation_duration_s (other variables and columns unchanged)
%
% Depends on:
%   - d_0929_performance_feature_extraction_all_trials.m (writes the .mat)
%   - functions/read_shape_band_obj.m
%   - functions/boundary_violations_fun.m

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
read_features_mat = fullfile(data_directory, 'processed_data', ...
    'performance_features_all_trials.mat');
write_features_mat = read_features_mat;  % columns added in place

edge_spacing_m = 5e-4;  % band edge resample spacing [m], as in the replay
violation_min_s = 0.05;  % shortest excursion counted [s], as in d_0929

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\\*.csv\n  %s\\*.obj\n  %s\n', ...
    manifest_file, raw_data_directory, shape_directory, read_features_mat);
fprintf('writing:\n  %s (trial_features, columns added)\n', ...
    write_features_mat);

M = readtable(manifest_file, 'TextType', 'string');
features = load(read_features_mat);
F = features.trial_features;
n_trials = height(M);

% trial_features rows follow the manifest (d_0929 loops over it in order).
[~, controller_index] = ismember(M.controller, ...
    string(features.controller_label_L));
[~, shape_index] = ismember(M.shape, string(features.shape_name_L));
[~, bandwidth_index] = ismember(M.bandwidth, ...
    string(features.bandwidth_label_L));
assert(isequal([F.subject_id, F.controller_index, F.shape_index, ...
    F.bandwidth_index], [M.subject, controller_index, shape_index, ...
    bandwidth_index]), 'trial_features rows do not follow the manifest');

n_band_violations = nan(n_trials, 1);
band_violation_duration_s = cell(n_trials, 1);

shape_objs = unique(M.shape_obj);
for s = 1:numel(shape_objs)
    [outer_xy_m, inner_xy_m] = read_shape_band_obj( ...
        fullfile(shape_directory, shape_objs(s)), edge_spacing_m);

    for i = find(M.shape_obj == shape_objs(s))'
        fprintf('Trial %d/%d: %s\n', i, n_trials, M.robot_csv(i));
        T = readtable(fullfile(raw_data_directory, M.robot_csv(i)));
        offset_xy_m = [M.offset_x(i), M.offset_y(i)];
        outer_trial_xy_m = outer_xy_m + offset_xy_m;
        inner_trial_xy_m = inner_xy_m + offset_xy_m;

        % Outside: outside the outer edge or inside the inner edge.
        outside_band = ~(inpolygon(T.coppeliasim_x_m, T.coppeliasim_y_m, ...
            outer_trial_xy_m(:, 1), outer_trial_xy_m(:, 2)) & ...
            ~inpolygon(T.coppeliasim_x_m, T.coppeliasim_y_m, ...
            inner_trial_xy_m(:, 1), inner_trial_xy_m(:, 2)));

        % boundary_violations_fun counts values > 1: 2 outside, 1 inside.
        [n_band_violations(i), band_violation_duration_s{i}] = ...
            boundary_violations_fun(1 + outside_band, T.time_s, ...
            violation_min_s);
    end
end

F.n_band_violations = n_band_violations;
F.band_violation_duration_s = band_violation_duration_s;
trial_features = F;
save(write_features_mat, 'trial_features', '-append');

fprintf('\nSaved n_band_violations (total %d, centerline method %d):\n%s\n', ...
    sum(n_band_violations), sum(F.n_boundary_violations), write_features_mat);
