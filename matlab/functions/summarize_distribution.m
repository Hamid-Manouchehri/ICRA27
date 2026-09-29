function stats = summarize_distribution(values)
%SUMMARIZE_DISTRIBUTION Mean/std/median/p90/max of a point-wise signal.
%
% Generic distributional summary, reused for any per-sample metric
% (distance to centerline, |angular divergence|, ...) instead of a
% one-off fprintf/mean call per metric.
%
% Math (NaNs excluded from every statistic):
%   mean   = (1/n) * sum(values)
%   std    = sqrt( (1/(n-1)) * sum((values - mean)^2) )
%   median = middle value of the sorted values
%   p90    = 90th percentile (linear interpolation, MATLAB prctile default)
%   max    = max(values)
%
% Inputs:
%   values: point-wise numeric signal
%
% Outputs:
%   stats: struct with fields mean, std, median, p90, max (same units as values)

stats.mean = mean(values, 'omitnan');
stats.std = std(values, 'omitnan');
stats.median = median(values, 'omitnan');
stats.p90 = prctile(values, 90);
stats.max = max(values, [], 'omitnan');
end
