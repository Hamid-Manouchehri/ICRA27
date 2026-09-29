%% Accuracy metrics per subject, Chen vs gated, from the saved .mat.
% One Chen/Gated pair per subject: that subject's trials averaged over the
% selected conditions. Only conditions where BOTH controllers exist for the
% subject are used, so a missing trial cannot tilt the comparison.
% One figure per group in the metric table below, one tile per metric.
%
% Inputs:
%   - data/processed_data/accuracy_metrics_exploration_all_subjects.mat
%
% Outputs:
%   - figures only, nothing written to disk
%
% Depends on:
%   - d_0921_tracking_metrics_all_subjects.m (writes the input .mat)
%   - functions/paired_subject_means.m

clc; clear; close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_metrics_mat = fullfile(data_directory, 'processed_data', ...
    'accuracy_metrics_exploration_all_subjects.mat');

plot_shape_name = "all";       % TODO "all" pools all shapes, or one of shape_name_L
plot_bandwidth_label = "all";  % TODO "all" pools both bandwidths, or "wide" / "narrow"

chen_color = [0.20 0.40 0.80];
gated_color = [0.85 0.35 0.10];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_metrics_mat);
fprintf('writing:\n  nothing (figures only)\n');

%% 1. Load. R has one row per trial (see d_0921_tracking_metrics_all_subjects.m).
L = load(read_metrics_mat, 'trial_metrics', 'controller_label_L', ...
    'shape_name_L', 'bandwidth_label_L');
R = L.trial_metrics;

%% 2. Metrics. One row = one tile: figure name, title, unit, one number per trial.
% To add a metric, add a row. Rows with the same figure name share a figure.
% NOTE: the angular-divergence rows are wrong for shapes whose centerline is
% ordered opposite to the travel direction (narrow_fine_gross, wide_pear).
metrics = {
    'band accuracy and distance', 'in-band accuracy', '[%]', ...
        R.in_band_percent;
    'band accuracy and distance', 'mean normalized distance', '[frac of half-width]', ...
        cellfun(@mean, R.dist_to_centerline_norm);
    'band accuracy and distance', 'mean normalized boundary violation', '[frac of half-width]', ...
        R.violation_mean_excess_overall;
    'band accuracy and distance', 'mean distance to centerline', '[mm]', ...
        1e3 * cellfun(@mean, R.dist_to_centerline_m);

    'heading error and rates', 'mean |angular divergence|', '[deg]', ...
        cellfun(@(a) mean(abs(rad2deg(a))), R.angular_divergence_rad);
    'heading error and rates', 'median |rate of distance to centerline|', '[mm/s]', ...
        1e3 * cellfun(@(r) median(abs(r)), R.dist_rate_m_s);
    'heading error and rates', 'median |rate of angular divergence|', '[deg/s]', ...
        rad2deg(cellfun(@(r) median(abs(r)), R.angular_divergence_rate_rad_s))};

%% 3. Choose which trials to use.
is_selected = true(height(R), 1);
if plot_shape_name ~= "all"
    is_selected = is_selected & ...
        R.shape_index == find(string(L.shape_name_L) == plot_shape_name);
end
if plot_bandwidth_label ~= "all"
    is_selected = is_selected & ...
        R.bandwidth_index == find(string(L.bandwidth_label_L) == plot_bandwidth_label);
end

chen_code = find(string(L.controller_label_L) == "Chen");
gated_code = find(string(L.controller_label_L) == "Gated");

%% 4. Plot: one figure per group, one tile per metric.
for figure_name = unique(metrics(:, 1), 'stable')'
    figure('Name', figure_name{1}, 'Color', 'w');
    layout = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

    for k = find(strcmp(metrics(:, 1), figure_name{1}))'
        metric_title = metrics{k, 2};
        metric_unit = metrics{k, 3};
        per_trial_values = metrics{k, 4};

        % Average each subject's paired trials, separately per controller.
        [subjects, chen_mean, gated_mean] = paired_subject_means( ...
            R, per_trial_values, is_selected, chen_code, gated_code);
        x = 1:numel(subjects);

        nexttile;
        hold on;

        plot([x; x], [chen_mean'; gated_mean'], '-', 'Color', [0.7 0.7 0.7], ...
            'HandleVisibility', 'off');

        plot(x, chen_mean, 'o', 'MarkerFaceColor', chen_color, ...
            'MarkerEdgeColor', 'k', 'DisplayName', 'Chen');

        plot(x, gated_mean, 's', 'MarkerFaceColor', gated_color, ...
            'MarkerEdgeColor', 'k', 'DisplayName', 'Gated');

        set(gca, 'XTick', x, 'XTickLabel', compose('s%d', subjects));
        xlim([0.5, numel(subjects) + 0.5]);
        grid on;
        ylabel(metric_unit);
        title(metric_title, 'FontSize', 9);
    end

    key = legend(gca);  % Chen / Gated key, one for the whole figure
    key.Layout.Tile = 'east';
    title(layout, sprintf('Chen vs Gated per subject: shape = %s, bandwidth = %s (%d subjects)', ...
        plot_shape_name, plot_bandwidth_label, numel(subjects)), 'Interpreter', 'none');
end
