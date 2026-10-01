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

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
raw_data_directory = fullfile(data_directory, 'raw_data', 'test_subjects');

% Code lists: a trial is saved as its 1-based index into these.
controller_label_L = {"Chen", "Gated"};  % Chen:1, Gated: 2
shape_name_L = {"fine_gross", "pear", "ellipse"};  % fine_gross:1, pear: 2, ellipse: 3
bandwidth_label_L = {"wide", "narrow"};  % wide: 1, narrow: 2
direction_label_L = {"clockwise", "counter_clockwise"};  % clockwise: 1, counter_clockwise: 2

sparc_padlevel = 4;  % zero-padding level
sparc_cutoff_hz = 10.0;  % max cutoff frequency [Hz]
sparc_amp_threshold = 0.05;  % adaptive cutoff amplitude threshold
sparc_min_pass_s = 1.0;  % shortest fine-area pass used for SPARC [s]

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
    fine_mask = fine_mask_tfun(T.coppeliasim_x_m, shape_index(i));
    [sparc_fine(i), sparc_fine_passes{i}, fine_pass_duration_s{i}] = ...
        sparc_fine_fun(speed_m_s, fine_mask, fs_hz, sparc_min_pass_s, ...
        sparc_padlevel, sparc_cutoff_hz, sparc_amp_threshold);
end

trial_features = table(subject_id, trial, controller_index, shape_index, ...
    bandwidth_index, direction_index, time_s, tcp_xy_m, tcp_v_xy_m_s, sparc, ...
    sparc_fine, sparc_fine_passes, fine_pass_duration_s);

save(write_features_mat, 'trial_features', 'controller_label_L', ...
    'shape_name_L', 'bandwidth_label_L', 'direction_label_L');

fprintf('\nSaved %d trials:\n%s\n', n_trials, write_features_mat);
