function [subjects, chen_mean, gated_mean] = paired_subject_means( ...
    trial_metrics, metric_values, is_selected, chen_code, gated_code)
%PAIRED_SUBJECT_MEANS Per-subject Chen and Gated means over paired conditions.
%
% A condition is one (subject, shape, bandwidth). Among the selected rows,
% only conditions that have BOTH a Chen and a Gated trial are kept, so a
% missing trial cannot tilt the comparison. Each subject's kept trials are
% then averaged separately per controller.
%
% Math (per subject s, per metric column m):
%   chen_mean(s, m)  = mean of metric_values(kept Chen rows of s, m)
%   gated_mean(s, m) = mean of metric_values(kept Gated rows of s, m)
%
% Inputs:
%   trial_metrics: N-row table (subject_id, shape_index, bandwidth_index,
%     controller_index columns)
%   metric_values: [N x M] one row per trial
%   is_selected: [N x 1] logical, rows to consider
%   chen_code, gated_code: controller_index values of Chen and Gated
%
% Outputs:
%   subjects: [S x 1] subject ids with at least one paired condition
%   chen_mean, gated_mean: [S x M] per-subject means

condition_key = trial_metrics.subject_id * 100 + ...
    trial_metrics.shape_index * 10 + trial_metrics.bandwidth_index;
is_chen = is_selected & trial_metrics.controller_index == chen_code;
is_gated = is_selected & trial_metrics.controller_index == gated_code;
paired_keys = intersect(condition_key(is_chen), condition_key(is_gated));
is_paired = is_selected & ismember(condition_key, paired_keys);

subjects = unique(trial_metrics.subject_id(is_paired));
chen_mean = nan(numel(subjects), size(metric_values, 2));
gated_mean = nan(numel(subjects), size(metric_values, 2));
for s = 1:numel(subjects)
    of_subject = is_paired & trial_metrics.subject_id == subjects(s);
    chen_mean(s, :) = mean(metric_values(of_subject & is_chen, :), 1);
    gated_mean(s, :) = mean(metric_values(of_subject & is_gated, :), 1);
end
end
