%% Accuracy metrics per subject, Chen vs gated, from the saved .mat.
% One Chen/Gated pair per subject: that subject's trials averaged over the
% selected conditions. Only conditions where BOTH controllers exist for the
% subject are used, so a missing trial cannot tilt the comparison.
%
% Inputs:
%   - data/processed_data/accuracy_metrics_exploration_all_subjects.mat
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - d_0921_tracking_metrics_all_subjects.m (writes the input .mat)
%   - functions/iif.m
%   - functions/accuracy_metric_values.m
%   - functions/paired_subject_means.m

clc; clear; close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_metrics_mat = fullfile(data_directory, 'processed_data', ...
    'accuracy_metrics_exploration_all_subjects.mat');

plot_shape_name = "";       % TODO "" pools all shapes, or one of shape_name_L
plot_bandwidth_label = "";  % TODO "" pools both bandwidths, or "wide" / "narrow"

chen_color = [0.20 0.40 0.80];
gated_color = [0.85 0.35 0.10];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_metrics_mat);
fprintf('writing:\n  nothing (figure only)\n');

L = load(read_metrics_mat, 'trial_metrics', 'controller_label_L', ...
    'shape_name_L', 'bandwidth_label_L');
R = L.trial_metrics;  % one row per trial

[metric_values, metric_titles, metric_units] = accuracy_metric_values(R);

% Rows of the selected shape / bandwidth.
is_selected = true(height(R), 1);
if plot_shape_name ~= ""
    is_selected = is_selected & ...
        R.shape_index == find(string(L.shape_name_L) == plot_shape_name);
end
if plot_bandwidth_label ~= ""
    is_selected = is_selected & ...
        R.bandwidth_index == find(string(L.bandwidth_label_L) == plot_bandwidth_label);
end

chen_code = find(string(L.controller_label_L) == "Chen");
gated_code = find(string(L.controller_label_L) == "Gated");
[subjects, chen_mean, gated_mean] = paired_subject_means( ...
    R, metric_values, is_selected, chen_code, gated_code);
subject_count = numel(subjects);

figure('Name', 'accuracy_metrics_by_subject', 'Color', 'w');
layout = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
title(layout, sprintf('Chen vs Gated per subject: shape = %s, bandwidth = %s (%d subjects)', ...
    iif(plot_shape_name == "", "all pooled", plot_shape_name), ...
    iif(plot_bandwidth_label == "", "both pooled", plot_bandwidth_label), subject_count), ...
    'Interpreter', 'none');

for m = 1:size(metric_values, 2)
    nexttile;
    hold on;
    plot([1:subject_count; 1:subject_count], [chen_mean(:, m)'; gated_mean(:, m)'], ...
        '-', 'Color', [0.7 0.7 0.7], 'HandleVisibility', 'off');
    plot(1:subject_count, chen_mean(:, m), 'o', 'MarkerFaceColor', chen_color, ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'Chen');
    plot(1:subject_count, gated_mean(:, m), 's', 'MarkerFaceColor', gated_color, ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'Gated');
    set(gca, 'XTick', 1:subject_count, 'XTickLabel', compose('s%d', subjects));
    xlim([0.5, subject_count + 0.5]);
    grid on;
    ylabel(metric_units{m});
    title(metric_titles{m}, 'FontSize', 9);
    if m == 1
        legend('Location', 'best');
    end
end
