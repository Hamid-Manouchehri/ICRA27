%% Gated damping law vs. force (paper Fig. 3): loads all Gated benchmark trials.
% Pools |F_filt|, the commanded damping and P(Fine) over every Gated trial.
% The law curves and the overlay are the next step.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (Gated rows only)
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - none (reads the manifest and the Gated raw CSVs directly)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
raw_data_directory = fullfile(data_directory, 'raw_data', 'test_subjects');

gated_controller_label = "Gated";
subsample_step = 50;  % plot every Nth sample
colormap_name = "parula";  % any built-in colormap name

% Constants of ur5e_gaze_assisted_chen_vac_2d.py, for the analytic curves.
force_min_n = 1.0;  % damping is at its upper bound at or below this [N]
force_max_n = 11.0;  % damping is at its lower bound at or above this [N]
fine_bounds_n_s_m = [80, 200];  % [lower, upper], P(Fine) = 1
gross_bounds_n_s_m = [10, 100];  % [lower, upper], P(Fine) = 0
baseline_bounds_n_s_m = [10, 200];  % [lower, upper], Chen arm

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\\<robot_csv of %s rows>\n', manifest_file, ...
    raw_data_directory, gated_controller_label);
fprintf('writing:\n  nothing (figure only)\n');

M = readtable(manifest_file, 'TextType', 'string');
M = M(M.controller == gated_controller_label, :);
n_trials = height(M);

force_filt_n = cell(n_trials, 1);
bd_desired_n_s_m = cell(n_trials, 1);
bd_min_n_s_m = cell(n_trials, 1);
bd_max_n_s_m = cell(n_trials, 1);
fine_probability = cell(n_trials, 1);
trial_index = cell(n_trials, 1);  % row of M, to recover subject/shape/width

for i = 1:n_trials
% for i = 21:21
    fprintf('Reading trial %d/%d\n', i, n_trials);

    T = readtable(fullfile(raw_data_directory, M.robot_csv(i)));

    force_filt_n{i} = hypot(T.Fx_filt_N, T.Fy_filt_N);
    bd_desired_n_s_m{i} = T.bd_desired_Ns_m;
    bd_min_n_s_m{i} = T.bd_min_Ns_m;
    bd_max_n_s_m{i} = T.bd_max_Ns_m;
    fine_probability{i} = T.fine_probability;
    trial_index{i} = i * ones(height(T), 1);
end

% One long column per quantity, all trials stacked.
force_filt_n = vertcat(force_filt_n{:});
bd_desired_n_s_m = vertcat(bd_desired_n_s_m{:});
bd_min_n_s_m = vertcat(bd_min_n_s_m{:});
bd_max_n_s_m = vertcat(bd_max_n_s_m{:});
fine_probability = vertcat(fine_probability{:});
trial_index = vertcat(trial_index{:});

fprintf('%d %s trials, %d samples\n', n_trials, gated_controller_label, ...
    numel(force_filt_n));

%% Plot
figure('Name', 'damping_law_vs_force');
hold on;

keep = 1:subsample_step:numel(force_filt_n);
scatter(force_filt_n(keep), bd_desired_n_s_m(keep), 4, fine_probability(keep), ...
    'filled');
colormap(gca, colormap_name);
cb = colorbar;
cb.Label.String = 'P(Fine)';
cb.FontSize = 14;
clim([0, 1]);

% TODO: analytic curves (P = 0, 0.5, 1 and the baseline) on top.

xlabel('|F| [N]');
ylabel('b_d [N s/m]');
% title(sprintf(['Gated, %d trials, every %dth sample, ' ...
%     'color = P(Fine), colormap: %s'], n_trials, subsample_step, ...
%     colormap_name));
set(gca, 'FontSize', 14);
xlim([0 12]);
ylim([10 210]);
grid on;
