%% Fine/Gross separation of the four classifier features, wide vs narrow.
% Per participant and width: AUC of each feature for Fine vs Gross 1 s
% windows (shapes and directions pooled), oriented on all windows so that
% AUC < 0.5 means the separation reversed. Paired Wilcoxon test of wide vs
% narrow across participants, Holm-adjusted across the four features.
%
% Inputs:
%   - data/processed_data/features_all_train_W_1s.csv
%   - data/processed_data/features_all_test_W_1s.csv
%
% Outputs:
%   - printed summary and one figure, nothing written to disk
%
% Depends on:
%   - scripts/build_modality_feature_datasets.py (writes the inputs)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_train_csv = fullfile(data_directory, 'processed_data', ...
    'features_all_train_W_1s.csv');
read_test_csv = fullfile(data_directory, 'processed_data', ...
    'features_all_test_W_1s.csv');

feature_names = ["gaze_gaze_path_efficiency", "gaze_p90_gaze_ee_distance", ...
    "robot_direction_concentration", "robot_std_curvature"];
feature_labels = ["path efficiency (gaze)", "p90 gaze-EE distance (gaze)", ...
    "direction concentration (robot)", "std curvature (robot)"];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\n', read_train_csv, read_test_csv);
fprintf('writing:\n  nothing (printout and figure only)\n');

% Train and test are one set of 1 s windows split by window: use both.
T = [readtable(read_train_csv, 'TextType', 'string'); ...
    readtable(read_test_csv, 'TextType', 'string')];

subjects = unique(T.subject);
n_subjects = numel(subjects);
n_features = numel(feature_names);
bandwidths = ["wide", "narrow"];
is_fine = T.trial_class_label == "Fine";

auc = nan(n_subjects, 2, n_features);  % subject x width x feature
feature_sign = ones(n_features, 1);
p_signrank = nan(n_features, 1);

for k = 1:n_features
    score = T.(feature_names(k));
    valid = ~isnan(score);

    % Orient the feature so that Fine scores higher over all windows.
    [~, ~, ~, pooled_auc] = perfcurve(is_fine(valid), score(valid), true);
    if pooled_auc < 0.5
        feature_sign(k) = -1;
    end

    for i = 1:n_subjects
        for w = 1:2
            rows = valid & T.subject == subjects(i) & ...
                T.bandwidth == bandwidths(w);
            [~, ~, ~, auc(i, w, k)] = perfcurve(is_fine(rows), ...
                feature_sign(k) * score(rows), true);
        end
    end
    p_signrank(k) = signrank(auc(:, 1, k), auc(:, 2, k));
end

% Holm-Bonferroni across the four features.
[p_sorted, order] = sort(p_signrank);
p_holm = nan(n_features, 1);
p_holm(order) = min(1, cummax((n_features:-1:1)' .* p_sorted));

fprintf('\nAUC Fine vs Gross per participant, %d participants\n', n_subjects);
for k = 1:n_features
    drop = auc(:, 1, k) - auc(:, 2, k);
    fprintf(['%-30s sign %+d | median AUC wide %.3f, narrow %.3f | ' ...
        'median drop %+.3f, lower in narrow %d of %d | Wilcoxon p = ' ...
        '%.4f (Holm %.4f)\n'], feature_names(k), feature_sign(k), ...
        median(auc(:, 1, k)), median(auc(:, 2, k)), median(drop), ...
        sum(drop > 0), n_subjects, p_signrank(k), p_holm(k));
end

%% Plot
figure('Name', 'auc_wide_vs_narrow_classifier_features', 'Color', 'w');
tiledlayout(1, n_features, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:n_features
    nexttile;
    hold on;
    plot([1, 2], auc(:, :, k)', 'Color', [0.6, 0.8, 1]);
    plot(ones(n_subjects, 1), auc(:, 1, k), 'ko', 'MarkerFaceColor', 'k');
    plot(2 * ones(n_subjects, 1), auc(:, 2, k), 'ko', 'MarkerFaceColor', 'k');
    yline(0.5, 'k--');
    xlim([0.5, 2.5]);
    xticks([1, 2]);
    xticklabels(bandwidths);
    ylim([0.4, 1]);
    ylabel('AUC, Fine vs Gross');
    grid on;
    title({feature_labels(k), sprintf('Wilcoxon p = %.3f (Holm %.3f)', ...
        p_signrank(k), p_holm(k))});
end
