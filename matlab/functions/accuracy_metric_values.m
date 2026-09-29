function [metric_values, metric_titles, metric_units] = accuracy_metric_values(trial_metrics)
%ACCURACY_METRIC_VALUES One accuracy number per trial, one column per metric.
%
% Reduces the saved per-sample series to the four per-trial accuracy
% metrics used in the per-subject plots. Column order is fixed:
%   1 in-band accuracy [%]
%   2 mean normalized distance [frac of half-width]
%   3 mean normalized boundary violation [frac of half-width]
%   4 mean distance to centerline [mm]
%
% Inputs:
%   trial_metrics: N-row table written by d_0921_tracking_metrics_all_subjects.m
%
% Outputs:
%   metric_values: [N x 4] one row per trial
%   metric_titles: 1x4 cell of column names
%   metric_units: 1x4 cell of column units

metric_values = [ ...
    trial_metrics.in_band_percent, ...
    cellfun(@mean, trial_metrics.dist_to_centerline_norm), ...
    trial_metrics.violation_mean_excess_overall, ...
    1e3 * cellfun(@mean, trial_metrics.dist_to_centerline_m)];
metric_titles = {'in-band accuracy', 'mean normalized distance', ...
    'mean normalized boundary violation', 'mean distance to centerline'};
metric_units = {'[%]', '[frac of half-width]', '[frac of half-width]', '[mm]'};
end
