%% Accuracy metrics per subject, Chen vs gated, one column per shape.
% Rows are metrics, columns are shapes. Each metric row shares one y-axis
% across shapes, so shape differences in difficulty show up directly.
% Per shape, a subject's trials are averaged over the selected bandwidths,
% using only conditions where BOTH controllers exist for that subject.
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
metric_count = size(metric_values, 2);
shape_count = numel(L.shape_name_L);
all_subjects = unique(R.subject_id);  % common x axis for every column

chen_code = find(string(L.controller_label_L) == "Chen");
gated_code = find(string(L.controller_label_L) == "Gated");

is_bandwidth = true(height(R), 1);
if plot_bandwidth_label ~= ""
    is_bandwidth = R.bandwidth_index == ...
        find(string(L.bandwidth_label_L) == plot_bandwidth_label);
end

figure('Name', 'accuracy_metrics_by_subject_per_shape', 'Color', 'w');
layout = tiledlayout(metric_count, shape_count, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
title(layout, sprintf('Chen vs Gated per subject, per shape: bandwidth = %s', ...
    iif(plot_bandwidth_label == "", "both pooled", plot_bandwidth_label)));

ax = gobjects(metric_count, shape_count);
for s = 1:shape_count
    is_selected = is_bandwidth & R.shape_index == s;
    [subjects, chen_mean, gated_mean] = paired_subject_means( ...
        R, metric_values, is_selected, chen_code, gated_code);
    [~, x_position] = ismember(subjects, all_subjects);

    for m = 1:metric_count
        ax(m, s) = nexttile((m - 1) * shape_count + s);
        hold on;
        plot([x_position'; x_position'], [chen_mean(:, m)'; gated_mean(:, m)'], ...
            '-', 'Color', [0.7 0.7 0.7], 'HandleVisibility', 'off');
        plot(x_position, chen_mean(:, m), 'o', 'MarkerFaceColor', chen_color, ...
            'MarkerEdgeColor', 'k', 'DisplayName', 'Chen');
        plot(x_position, gated_mean(:, m), 's', 'MarkerFaceColor', gated_color, ...
            'MarkerEdgeColor', 'k', 'DisplayName', 'Gated');
        set(gca, 'XTick', 1:numel(all_subjects), ...
            'XTickLabel', compose('s%d', all_subjects), 'FontSize', 8);
        xlim([0.5, numel(all_subjects) + 0.5]);
        grid on;
        if m == 1
            title(sprintf('%s (%d subjects)', L.shape_name_L{s}, numel(subjects)), ...
                'FontSize', 10, 'Interpreter', 'none');
        end
        if s == 1
            ylabel({metric_titles{m}, metric_units{m}}, 'FontSize', 8);
        end
        if m == 1 && s == 1
            legend('Location', 'best', 'FontSize', 7);
        end
        if m < metric_count
            set(gca, 'XTickLabel', []);
        end
    end
end

% One y range per metric row, wide enough for every shape, with a 5%
% margin so markers at the extremes are not clipped by the axis edge.
for m = 1:metric_count
    y_limits = vertcat(ax(m, :).YLim);
    y_low = min(y_limits(:, 1));
    y_high = max(y_limits(:, 2));
    y_margin = 0.05 * (y_high - y_low);
    set(ax(m, :), 'YLim', [y_low - y_margin, y_high + y_margin]);
end
