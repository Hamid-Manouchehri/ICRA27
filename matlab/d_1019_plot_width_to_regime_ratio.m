%% Width-to-regime ratio of the four classifier features.
% Per participant: mean |narrow - wide| over shapes and regimes, divided by
% mean |Fine - Gross| over shapes (signed within a shape, averaged over the
% two widths). Below 1: width moves the feature less than regime does.
%
% Inputs:
%   - data/processed_data/features_all_train_W_1s.csv
%   - data/processed_data/features_all_test_W_1s.csv
%
% Outputs:
%   - printed summary and one figure, nothing written to disk
%
% Depends on:
%   - scripts/build_modality_feature_datasets.py (writes the inputs)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_train_csv = fullfile(data_directory, 'processed_data', ...
    'features_all_train_W_1s.csv');
read_test_csv = fullfile(data_directory, 'processed_data', ...
    'features_all_test_W_1s.csv');

feature_names = ["gaze_gaze_path_efficiency", "gaze_p90_gaze_ee_distance", ...
    "robot_direction_concentration", "robot_std_curvature"];
feature_labels = ["path efficiency (gaze)", "p90 gaze-EE distance (gaze)", ...
    "direction concentration (robot)", "std curvature (robot)"];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\n', read_train_csv, read_test_csv);
fprintf('writing:\n  nothing (printout and figure only)\n');

% Train and test are one set of 1 s windows split by window: use both.
T = [readtable(read_train_csv, 'TextType', 'string'); ...
    readtable(read_test_csv, 'TextType', 'string')];

subjects = unique(T.subject);
n_subjects = numel(subjects);
n_features = numel(feature_names);
shapes = ["fine_gross", "pear", "ellipse"];
bandwidths = ["wide", "narrow"];
regimes = ["Fine", "Gross"];

% Median per participant, shape, width and regime (directions pooled).
G = groupsummary(T, {'subject', 'shape', 'bandwidth', ...
    'trial_class_label'}, 'median', feature_names);

% M(subject, shape, width, regime, feature), NaN for a missing cell.
M = nan(n_subjects, 3, 2, 2, n_features);
for g = 1:height(G)
    i = find(subjects == G.subject(g));
    s = find(shapes == G.shape(g));
    w = find(bandwidths == G.bandwidth(g));
    r = find(regimes == G.trial_class_label(g));
    M(i, s, w, r, :) = G{g, "median_" + feature_names};
end

ratio = nan(n_subjects, n_features);
fprintf('\nwidth-to-regime ratio, %d participants\n', n_subjects);
for k = 1:n_features
    % |narrow - wide| per regime, averaged over regimes: subject x shape.
    width_shift = mean(abs(M(:, :, 2, :, k) - M(:, :, 1, :, k)), 4, ...
        'omitnan');
    % Fine - Gross per width, averaged over widths, then magnitude, so a
    % sign flip between widths shrinks the gap.
    regime_gap = abs(mean(M(:, :, :, 1, k) - M(:, :, :, 2, k), 3, ...
        'omitnan'));
    ratio(:, k) = mean(width_shift, 2, 'omitnan') ./ ...
        mean(regime_gap, 2, 'omitnan');

    fprintf(['%-30s median %.2f (range %.2f to %.2f) | ' ...
        '|narrow - wide| %.4g, |Fine - Gross| %.4g\n'], feature_names(k), ...
        median(ratio(:, k)), min(ratio(:, k)), max(ratio(:, k)), ...
        median(width_shift(:), 'omitnan'), median(regime_gap(:), 'omitnan'));
end

%% Plot
figure('Name', 'width_to_regime_ratio_classifier_features', 'Color', 'w');
hold on;
for k = 1:n_features
    subject_dots = swarmchart(k * ones(n_subjects, 1), ratio(:, k), 60, ...
        'k', 'filled', 'XJitterWidth', 0.5);
    median_line = plot(k + [-0.3, 0.3], median(ratio(:, k)) * [1, 1], ...
        'm', 'LineWidth', 4);
end
yline(1, 'k--');
xlim([0.5, n_features + 0.5]);
xticks(1:n_features);
xticklabels(feature_labels);
ylabel('|narrow - wide| / |Fine - Gross|');
grid on;
legend([subject_dots, median_line], {'subject', 'median'});
title({sprintf('width-to-regime ratio, %d participants', n_subjects), ...
    '1 s windows, Chen VAC trials, manual Fine/Gross labels'});
