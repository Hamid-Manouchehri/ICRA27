%% Significance tests of performance metrics, Gated vs Chen.
% Per metric: paired t-test and Wilcoxon signed-rank test on per-participant
% means. p-values are also Holm-adjusted across the metrics.
%
% Inputs:
%   - data/processed_data/performance_features_all_trials.csv
%
% Outputs:
%   - results printed and one figure, nothing written to disk
%
% Depends on:
%   - none

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_features_csv = fullfile(data_directory, 'processed_data', ...
    'performance_features_all_trials.csv');

% Metrics tested, in Table 1 order; violations_per_m is derived below.
metric_names = ["mean_force", "total_progress_m", "violations_per_m", ...
    "sparc"];
metric_labels = ["mean force [N] (effort)", ...
    "centerline progress [m] (pace)", ...
    "boundary violations per m of path (accuracy)", "SPARC (smoothness)"];
significance_level = 0.05;  % two-sided; CIs are 1 - level

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_features_csv);
fprintf('writing:\n  nothing (printout and figure only)\n');

F = readtable(read_features_csv);

% Violations per meter of TCP path.
F.violations_per_m = F.n_boundary_violation ./ F.travelled_path_m;

T = F(:, {'subject_id', 'shape_index', 'bandwidth_index'});
T.controller = categorical(F.controller_index, [2, 1], {'Gated', 'Chen'});

n_metrics = numel(metric_names);
p_ttest = nan(n_metrics, 1);
p_signrank = nan(n_metrics, 1);
ci_percent = 100 * (1 - significance_level);

figure('Name', 'significance_tests_gated_vs_chen', 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:n_metrics
    T.metric = F.(metric_names(k));

    % One mean per participant and controller, over the conditions where
    % both trials exist.
    P = unstack(T, 'metric', 'controller');
    P = rmmissing(P);
    G = groupsummary(P, 'subject_id', 'mean', {'Gated', 'Chen'});
    gated = G.mean_Gated;
    chen = G.mean_Chen;
    difference = gated - chen;
    n_subjects = numel(difference);

    [~, p_ttest(k), ttest_ci, ttest_stats] = ttest(gated, chen, ...
        'Alpha', significance_level);
    p_signrank(k) = signrank(gated, chen);

    fprintf('\n%s, Gated - Chen, %d participants\n', metric_names(k), ...
        n_subjects);
    fprintf('participant means (mean +- SD): Chen %.3g +- %.3g, ', ...
        mean(chen), std(chen));
    fprintf('Gated %.3g +- %.3g\n', mean(gated), std(gated));
    fprintf(['paired t-test: difference %.3g, %g%% CI [%.3g, %.3g], ' ...
        't(%d) = %.2f, p = %.4f\n'], mean(difference), ci_percent, ...
        ttest_ci, ttest_stats.df, ttest_stats.tstat, p_ttest(k));
    fprintf(['Wilcoxon signed-rank: p = %.4f, lower under Gated for ' ...
        '%d of %d\n'], p_signrank(k), sum(difference < 0), n_subjects);

    nexttile;
    hold on;
    plot([1, 2], [chen, gated]', 'Color', [0.7, 0.7, 0.7]);
    plot(ones(n_subjects, 1), chen, 'bo', 'MarkerFaceColor', 'b');
    plot(2 * ones(n_subjects, 1), gated, 'ro', 'MarkerFaceColor', 'r');
    xlim([0.5, 2.5]);
    xticks([1, 2]);
    xticklabels({'Chen', 'Gated'});
    ylabel(metric_labels(k));
    grid on;
    title({sprintf('Gated - Chen = %.3g, %g%% CI [%.3g, %.3g]', ...
        mean(difference), ci_percent, ttest_ci), sprintf(['unadjusted p: ' ...
        't %.4f, Wilcoxon %.4f'], p_ttest(k), p_signrank(k))});
end

% Holm-Bonferroni across metrics, separately for each test.
p_raw = [p_ttest, p_signrank];
[p_sorted, order] = sort(p_raw);
p_holm_sorted = min(1, cummax((n_metrics:-1:1)' .* p_sorted));
p_holm = nan(size(p_raw));
for c = 1:2
    p_holm(order(:, c), c) = p_holm_sorted(:, c);
end

fprintf('\np-values, raw and Holm-adjusted across %d metrics\n', n_metrics);
disp(table(metric_names', p_ttest, p_holm(:, 1), p_signrank, ...
    p_holm(:, 2), 'VariableNames', {'metric', 'p_ttest', 'p_ttest_holm', ...
    'p_wilcoxon', 'p_wilcoxon_holm'}));
