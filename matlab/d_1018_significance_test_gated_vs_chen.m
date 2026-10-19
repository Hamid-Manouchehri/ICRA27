%% Significance tests of performance metrics, Gated vs Chen.
% Per metric: paired t-test and Wilcoxon signed-rank test on per-participant
% means, and the controller effect from a linear mixed-effects model on all
% trials. p-values are also Holm-adjusted across the metrics.
%
% Inputs:
%   - data/processed_data/performance_features_all_trials.mat
%
% Outputs:
%   - results printed and one figure, nothing written to disk
%
% Depends on:
%   - d_0929_performance_feature_extraction_all_trials.m (writes the input)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_features_mat = fullfile(data_directory, 'processed_data', ...
    'performance_features_all_trials.mat');

% Metrics tested; violations_per_m and centerline_pace_m_s are derived below.
metric_names = ["sparc", "mean_force_n", "violations_per_m", ...
    "centerline_pace_m_s"];
metric_labels = ["SPARC (smoothness)", "mean force [N] (effort)", ...
    "boundary violations per m of path (accuracy)", ...
    "centerline progress [m/s] (pace)"];
significance_level = 0.05;  % two-sided; CIs are 1 - level

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_features_mat);
fprintf('writing:\n  nothing (printout and figure only)\n');

features = load(read_features_mat, 'trial_features');
F = features.trial_features;

% Violations per meter of TCP path; net centerline progress per second.
trial_duration_s = cellfun(@(t) t(end) - t(1), F.time_s);
F.violations_per_m = F.n_boundary_violations ./ F.traveled_path_m;
F.centerline_pace_m_s = F.total_progress_m ./ trial_duration_s;

T = F(:, {'subject_id', 'shape_index', 'bandwidth_index'});
T.controller = categorical(F.controller_index, [2, 1], {'Gated', 'Chen'});
T.shape = categorical(T.shape_index, [1, 2, 3], {'paw', 'pear', 'ellipse'});
T.width = categorical(T.bandwidth_index, [1, 2], {'wide', 'narrow'});
T.subject = categorical(T.subject_id);

n_metrics = numel(metric_names);
p_ttest = nan(n_metrics, 1);
p_signrank = nan(n_metrics, 1);
p_lme = nan(n_metrics, 1);
ci_percent = 100 * (1 - significance_level);

figure('Name', 'significance_tests_gated_vs_chen', 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

for k = 1:n_metrics
    T.metric = F.(metric_names(k));

    % One mean per participant and controller, over the conditions where
    % both trials exist.
    P = unstack(T(:, {'subject_id', 'shape_index', 'bandwidth_index', ...
        'controller', 'metric'}), 'metric', 'controller');
    P = rmmissing(P);
    G = groupsummary(P, 'subject_id', 'mean', {'Gated', 'Chen'});
    gated = G.mean_Gated;
    chen = G.mean_Chen;
    difference = gated - chen;
    n_subjects = numel(difference);

    [~, p_ttest(k), ttest_ci, ttest_stats] = ttest(gated, chen, ...
        'Alpha', significance_level);
    p_signrank(k) = signrank(gated, chen);
    effect_size_dz = mean(difference) / std(difference);

    % Mixed model, effects coding (Gated: +1, Chen: -1), so Gated - Chen
    % averaged over shapes and widths is 2 * the controller_Gated coefficient.
    lme = fitlme(T, ['metric ~ controller*shape*width + ' ...
        '(1 + controller | subject)'], 'DummyVarCoding', 'effects', ...
        'FitMethod', 'REML');
    is_controller = strcmp(lme.CoefficientNames, 'controller_Gated');
    lme_ci = coefCI(lme, 'Alpha', significance_level, ...
        'DFMethod', 'satterthwaite');
    lme_ci = 2 * lme_ci(is_controller, :);
    lme_difference = 2 * lme.Coefficients.Estimate(is_controller);
    A = anova(lme, 'DFMethod', 'satterthwaite');
    p_lme(k) = A.pValue(strcmp(A.Term, 'controller'));

    fprintf('\n%s, Gated - Chen, %d participants\n', metric_names(k), ...
        n_subjects);
    fprintf('participant means: Chen %.4g, Gated %.4g\n', mean(chen), ...
        mean(gated));
    fprintf(['paired t-test: difference %.3g, %g%% CI [%.3g, %.3g], ' ...
        't(%d) = %.2f, p = %.4f, d_z = %.2f\n'], mean(difference), ...
        ci_percent, ttest_ci, ttest_stats.df, ttest_stats.tstat, ...
        p_ttest(k), effect_size_dz);
    fprintf(['Wilcoxon signed-rank: p = %.4f, lower under Gated for ' ...
        '%d of %d\n'], p_signrank(k), sum(difference < 0), n_subjects);
    fprintf('LMM: difference %.3g, %g%% CI [%.3g, %.3g], p = %.4f\n', ...
        lme_difference, ci_percent, lme_ci, p_lme(k));
    fprintf('LMM ANOVA (controller:shape, controller:width = interactions)\n');
    disp(A);

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
        't %.4f, Wilcoxon %.4f, LMM %.4f'], p_ttest(k), p_signrank(k), ...
        p_lme(k))});
end

% Holm-Bonferroni across metrics, separately for each test.
p_raw = [p_ttest, p_signrank, p_lme];
[p_sorted, order] = sort(p_raw);
p_holm_sorted = min(1, cummax((n_metrics:-1:1)' .* p_sorted));
p_holm = nan(size(p_raw));
for c = 1:3
    p_holm(order(:, c), c) = p_holm_sorted(:, c);
end

fprintf('\np-values, raw and Holm-adjusted across %d metrics\n', n_metrics);
disp(table(metric_names', p_ttest, p_holm(:, 1), p_signrank, ...
    p_holm(:, 2), p_lme, p_holm(:, 3), 'VariableNames', {'metric', ...
    'p_ttest', 'p_ttest_holm', 'p_wilcoxon', 'p_wilcoxon_holm', 'p_lmm', ...
    'p_lmm_holm'}));
