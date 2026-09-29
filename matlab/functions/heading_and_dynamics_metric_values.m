function [metric_values, metric_titles, metric_units] = heading_and_dynamics_metric_values(trial_metrics)
%HEADING_AND_DYNAMICS_METRIC_VALUES Heading-error and error-rate numbers, one per trial.
%
% Reduces the saved per-sample series to three per-trial metrics. Column
% order is fixed:
%   1 mean |angular divergence| [deg]
%   2 median |rate of distance to centerline| [mm/s]
%   3 median |rate of angular divergence| [deg/s]
%
% The two rates use the median, not the mean: the angular-divergence rate
% has isolated +-180 deg wrap spikes (thousands of deg/s) that would
% dominate a mean, while the median tracks the typical rate.
%
% Math (per trial):
%   col 1 = mean(|rad2deg(angular_divergence_rad)|)
%   col 2 = 1e3 * median(|dist_rate_m_s|)
%   col 3 = rad2deg(median(|angular_divergence_rate_rad_s|))
%
% Inputs:
%   trial_metrics: N-row table written by d_0921_tracking_metrics_all_subjects.m
%
% Outputs:
%   metric_values: [N x 3] one row per trial
%   metric_titles: 1x3 cell of column names
%   metric_units: 1x3 cell of column units

metric_values = [ ...
    cellfun(@(a) mean(abs(rad2deg(a))), trial_metrics.angular_divergence_rad), ...
    1e3 * cellfun(@(r) median(abs(r)), trial_metrics.dist_rate_m_s), ...
    rad2deg(cellfun(@(r) median(abs(r)), trial_metrics.angular_divergence_rate_rad_s))];
metric_titles = {'mean |angular divergence|', ...
    'median |rate of distance to centerline|', ...
    'median |rate of angular divergence|'};
metric_units = {'[deg]', '[mm/s]', '[deg/s]'};
end
