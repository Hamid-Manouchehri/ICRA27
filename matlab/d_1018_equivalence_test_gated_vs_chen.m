%% Equivalence test of one performance metric, Gated vs Chen.
% Paired TOST and JZS Bayes factor on per-participant mean differences, and
% the CI of the controller effect from a linear mixed-effects model.
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

metric_name = "sparc";  % sparc, sparc_fine, dsj, mean_force_n, ...
equivalence_margin = NaN;  % TODO: smallest meaningful |Gated - Chen|
bayes_prior_scale = sqrt(2) / 2;  % Cauchy prior width r on d_z
significance_level = 0.05;  % TOST level; CIs are 1 - 2 * level

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_features_mat);
fprintf('writing:\n  nothing (printout and figure only)\n');

features = load(read_features_mat, 'trial_features');
T = features.trial_features(:, {'subject_id', 'shape_index', ...
    'bandwidth_index'});
T.controller = categorical(features.trial_features.controller_index, ...
    [2, 1], {'Gated', 'Chen'});
T.metric = features.trial_features.(metric_name);
T.shape = categorical(T.shape_index, [1, 2, 3], {'paw', 'pear', 'ellipse'});
T.width = categorical(T.bandwidth_index, [1, 2], {'wide', 'narrow'});
T.subject = categorical(T.subject_id);

% One Gated - Chen difference per participant: mean over the conditions
% where both trials exist.
P = unstack(T(:, {'subject_id', 'shape_index', 'bandwidth_index', ...
    'controller', 'metric'}), 'metric', 'controller');
P = rmmissing(P);
P.difference = P.Gated - P.Chen;
G = groupsummary(P, 'subject_id', 'mean', 'difference');
difference = G.mean_difference;

n_subjects = numel(difference);
dof = n_subjects - 1;
mean_difference = mean(difference);
se_difference = std(difference) / sqrt(n_subjects);
t_value = mean_difference / se_difference;
p_difference = 2 * tcdf(-abs(t_value), dof);
paired_ci = mean_difference + [-1, 1] * ...
    tinv(1 - significance_level, dof) * se_difference;

% TOST: equivalent if both one-sided tests (against -margin and +margin)
% reject, i.e. the 1 - 2 * level CI lies inside +/- margin.
p_tost = max(tcdf(-(mean_difference + equivalence_margin) / ...
    se_difference, dof), tcdf((mean_difference - equivalence_margin) / ...
    se_difference, dof));

% JZS Bayes factor of the paired t (Rouder et al., 2009, Eq. 1, width r).
r = bayes_prior_scale;
integrand = @(g) (1 + n_subjects * g * r^2) .^ -0.5 .* (1 + t_value^2 ./ ...
    ((1 + n_subjects * g * r^2) * dof)) .^ (-(dof + 1) / 2) .* ...
    (2 * pi) ^ -0.5 .* g .^ -1.5 .* exp(-1 ./ (2 * g));
bf01 = (1 + t_value^2 / dof) ^ (-(dof + 1) / 2) / ...
    integral(integrand, 0, Inf);

% Mixed model, effects coding (Gated: +1, Chen: -1), so Gated - Chen
% averaged over shapes and widths is 2 * the controller_Gated coefficient.
lme = fitlme(T, ['metric ~ controller*shape*width + ' ...
    '(1 + controller | subject)'], 'DummyVarCoding', 'effects', ...
    'FitMethod', 'REML');
is_controller = strcmp(lme.CoefficientNames, 'controller_Gated');
lme_ci = coefCI(lme, 'Alpha', 2 * significance_level, ...
    'DFMethod', 'satterthwaite');
lme_ci = 2 * lme_ci(is_controller, :);
lme_difference = 2 * lme.Coefficients.Estimate(is_controller);
A = anova(lme, 'DFMethod', 'satterthwaite');
lme_p_difference = A.pValue(strcmp(A.Term, 'controller'));

ci_percent = 100 * (1 - 2 * significance_level);
fprintf('\n%s, Gated - Chen, %d participants\n', metric_name, n_subjects);
fprintf(['paired: mean %.3f, %g%% CI [%.3f, %.3f], t(%d) = %.2f, ' ...
    'p = %.4f\n'], mean_difference, ci_percent, paired_ci, dof, t_value, ...
    p_difference);
fprintf('LMM: %.3f, %g%% CI [%.3f, %.3f], p = %.4f\n', lme_difference, ...
    ci_percent, lme_ci, lme_p_difference);
fprintf('CI fits inside +/- %.3f (paired), +/- %.3f (LMM)\n', ...
    max(abs(paired_ci)), max(abs(lme_ci)));
fprintf('paired TOST, margin %g: p = %.4f\n', equivalence_margin, p_tost);
fprintf(['BF01 = %.3f (r = %.3f); above 3: evidence for no difference, ' ...
    'below 1/3: for a difference\n'], bf01, r);

%% Plot
figure('Name', sprintf('equivalence_test_%s_gated_vs_chen', metric_name), ...
    'Color', 'w');
hold on;
plot(ones(n_subjects, 1), difference, 'ko');
errorbar(2, mean_difference, mean_difference - paired_ci(1), ...
    paired_ci(2) - mean_difference, 'bs', 'MarkerFaceColor', 'b');
errorbar(3, lme_difference, lme_difference - lme_ci(1), ...
    lme_ci(2) - lme_difference, 'rs', 'MarkerFaceColor', 'r');
plot([0.5, 3.5], [0, 0], 'k');
plot([0.5, 3.5], equivalence_margin * [1, 1], 'k--');
plot([0.5, 3.5], -equivalence_margin * [1, 1], 'k--');
xlim([0.5, 3.5]);
xticks(1:3);
xticklabels({'participants', sprintf('paired mean, %g%% CI', ci_percent), ...
    sprintf('LMM, %g%% CI', ci_percent)});
ylabel(sprintf('%s: Gated - Chen', metric_name), 'Interpreter', 'none');
grid on;
title({sprintf('%s, Gated - Chen, %d participants, margin +/- %g', ...
    metric_name, n_subjects, equivalence_margin), ...
    sprintf('TOST p = %.4f, BF01 = %.3f (r = %.3f)', p_tost, bf01, r)}, ...
    'Interpreter', 'none');
