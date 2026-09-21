%% Standard performance plots for one trained logistic regression fit.
%
% Reads train_logistic_regression.py's two output CSVs (already on
% disk, no retraining) and produces the plots people typically check
% for a probabilistic binary classifier.
%
% Inputs:
%   data/processed_data/<read_predictions_csv_file>   test-set true label,
%     predicted label, per-class predicted probability (one row per
%     window) -- written by train_logistic_regression.py.
%   data/processed_data/<read_coefficients_csv_file>   one row per feature,
%     standardized coefficient -- written by train_logistic_regression.py.
%   Both, plus the output filename, are DERIVED from the "Trial
%   identity" block in User Settings (shape_name, held_out_subject_id,
%   prediction_horizon_label, window_size_label) -- to switch trained
%   models, edit that block only, not the paths themselves.
%
% Output:
%   Figure 1 (2x3 tiles):
%     1. Confusion matrix at the model's own predicted_label (threshold
%        0.5, as sklearn's .predict() uses internally).
%     2. ROC curve (TPR vs FPR swept over threshold) + AUC.
%     3. Precision-Recall curve + AUC.
%     4. Calibration (reliability) curve -- mean predicted probability
%        vs. empirical positive fraction, per bin, against the y = x
%        reference. Relevant specifically if predicted probability will
%        be used as a continuous signal (e.g. a damping blend weight),
%        not just thresholded.
%     5. Standardized coefficient bar chart (from the coefficients CSV).
%     6. Predicted-probability histogram, split by true class.
%   Figure 2: stem3 plot, predicted probability as the 3rd dimension
%     over each window's (x, y) position, FILTERED to shape_name (see
%     User Settings) -- held_out_criteria can hold out more than one
%     shape's worth of rows (e.g. a subject criterion pulls in that
%     subject's other-shape trials too), so this filter is what keeps
%     the plot to one physical shape. Falls back to plotting every row
%     unfiltered, with a warning, if the predictions CSV predates the
%     "shape" identifying column (rerun train_logistic_regression.py to
%     add it). Requires read_predictions_csv_file to carry train_
%     logistic_regression.py's position_columns -- an older predictions
%     CSV without them will error here with a clear message.
%   A numeric summary (accuracy, ROC-AUC, PR-AUC, per-bin calibration
%   counts) printed to the command window.
%   Both figures are also saved TOGETHER in one native MATLAB .fig file
%   under data/images/figures/plot_logistic_regression_performance/
%   (see write_combined_fig in User Settings), so plot_saved_logistic_
%   regression_performance.m can reopen both at once without rereading
%   either CSV.
%
% Positive-class convention:
%   ROC/PR/calibration all treat positive_class_label (see User
%   Settings, default "Fine") as the positive class, matching
%   feature_analysis_ET.m's own Y = strcmp(label,'Fine') convention
%   elsewhere in this project.
%
% Threshold sweep, not perfcurve():
%   ROC/PR are computed by a manual threshold sweep rather than the
%   Statistics and Machine Learning Toolbox's perfcurve(), so this
%   script has no toolbox dependency beyond what readtable itself needs
%   -- consistent with this project's convention of self-contained
%   MATLAB scripts (see e.g. analyze_fine_gross_regions_chen_2d.m's
%   two_means_threshold(), which reimplements a 2-means split the same
%   way rather than calling kmeans()).
%
% Method reuse:
%   Self-contained here per this project's convention that MATLAB
%   scripts do not call one another.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
processed_data_directory = fullfile(script_directory, '..', 'data', 'processed_data');

%% User Settings
% TODO explicit directory where this script's figures are saved, one
% subfolder per script for organization (see AI_GUIDELINES.md).
write_image_directory = fullfile(script_directory, '..', 'data', ...
    'images', 'figures', 'plot_logistic_regression_performance');

% TODO -------------------- Trial identity --------------------
% Both input CSVs and the output filename are DERIVED from these few
% tags -- switch trained models by editing only this block. Verified
% against every file actually present under data/processed_data/ as of
% this writing (the "_and_s<id>_" leave-one-subject-out CSVs; a few
% differently-named bundles also exist there, e.g. "..._shape_robot_
% W_1s_...", and are NOT reachable through this block).
shape_name = "fine_gross";           % TODO 'ellipse', 'pear', or 'fine_gross'
held_out_subject_id = "s13";         % TODO subject held out of training
prediction_horizon_label = "1win_ahead";  % TODO must match train_
                                           % logistic_regression.py's
                                           % own filename tokens
window_size_label = "W_1s";          % TODO must match train_logistic_
                                      % regression.py's own filename token
% TODO -------------------- End trial identity --------------------

% Output filename (native MATLAB .fig; BOTH figures below are bundled
% into this ONE file via savefig()'s multi-handle form, not one file
% per figure -- see plot_saved_logistic_regression_performance.m, which
% reopens both at once with a single openfig()).
write_combined_fig = fullfile(write_image_directory, sprintf( ...
    'logistic_regression_performance_%s_%s.fig', ...
    shape_name, held_out_subject_id));

% Figure window title -- same as the saved file's own name (no
% directory, no extension), so a reopened window is traceable to its
% .fig file at a glance.
[~, write_combined_fig_name, ~] = fileparts(write_combined_fig);

% The two CSVs written by train_logistic_regression.py -- must match
% its write_predictions_csv / write_coefficients_csv settings.
read_predictions_csv_file = fullfile(processed_data_directory, sprintf( ...
    'logistic_regression_held_out_%s_shape_and_%s_%s_%s_predictions.csv', ...
    shape_name, held_out_subject_id, prediction_horizon_label, ...
    window_size_label));
read_coefficients_csv_file = fullfile(processed_data_directory, sprintf( ...
    'logistic_regression_held_out_%s_shape_and_%s_%s_%s_coefficients.csv', ...
    shape_name, held_out_subject_id, prediction_horizon_label, ...
    window_size_label));

% TODO must match train_logistic_regression.py's target_column setting.
target_column_name = 'trial_class_label';

% TODO which class counts as "positive" for ROC/PR/calibration (see
% header comment's Positive-class convention note). Must be one of the
% two values actually present in target_column_name, and a
% "probability_<this>" column must exist in predictions_csv_file.
positive_class_label = "Fine";

% TODO threshold sweep resolution for the ROC/PR curves.
roc_threshold_count = 200;

% TODO number of equal-width bins [0,1] for the calibration curve. A
% bin's point is only as reliable as its sample count -- printed to the
% command window per bin, so a thin bin can be spotted rather than
% trusted blindly.
calibration_bin_count = 10;

% TODO histogram bin count for the predicted-probability-by-class plot.
probability_histogram_bin_count = 30;

% TODO Figure 2's (x, y) axes -- must match a pair of position_columns
% train_logistic_regression.py wrote into predictions_csv_file.
position_x_column_name = "robot_mean_position_x";
position_y_column_name = "robot_mean_position_y";
%% End User Settings

fprintf('Input:  %s\n', read_predictions_csv_file);
fprintf('Input:  %s\n', read_coefficients_csv_file);
fprintf('Output: %s\n\n', write_combined_fig);

if ~isfolder(write_image_directory)
    mkdir(write_image_directory);
end
figure_handles = gobjects(0);

%% Load
predictions_table = readtable(read_predictions_csv_file, 'Delimiter', ',', ...
    'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
    'TextType', 'string');
coefficients_table = readtable(read_coefficients_csv_file, 'Delimiter', ',', ...
    'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

true_label = predictions_table.(target_column_name);
predicted_label = predictions_table.predicted_label;
class_order = sort(unique(true_label));
if ~any(class_order == positive_class_label)
    error(['positive_class_label "%s" not found in %s -- present ' ...
        'values are: %s'], positive_class_label, target_column_name, ...
        strjoin(cellstr(class_order), ', '));
end
negative_class_label = class_order(class_order ~= positive_class_label);
negative_class_label = negative_class_label(1);

probability_column_name = "probability_" + positive_class_label;
if ~ismember(probability_column_name, predictions_table.Properties.VariableNames)
    error(['Column "%s" not found in %s -- check positive_class_label ' ...
        'matches a probability_<class> column in the predictions CSV.'], ...
        probability_column_name, read_predictions_csv_file);
end
score = predictions_table.(probability_column_name);
is_true_positive = (true_label == positive_class_label);

position_check_columns = [position_x_column_name, position_y_column_name];
predictions_columns = predictions_table.Properties.VariableNames;
missing_position_columns = position_check_columns( ...
    ~ismember(position_check_columns, predictions_columns));
if ~isempty(missing_position_columns)
    error(['Figure 2 needs %s in %s -- rerun train_logistic_regression.py ' ...
        '(its position_columns setting) to regenerate this CSV with ' ...
        'them.'], strjoin(cellstr(missing_position_columns), ', '), ...
        read_predictions_csv_file);
end

sample_count = height(predictions_table);
overall_accuracy = sum(predicted_label == true_label) / sample_count;

%% ================================================================
% ROC and Precision-Recall: manual threshold sweep
% =================================================================

thresholds = linspace(0, 1, roc_threshold_count);
true_positive_rate = zeros(size(thresholds));
false_positive_rate = zeros(size(thresholds));
precision = nan(size(thresholds));

positive_count = sum(is_true_positive);
negative_count = sum(~is_true_positive);

for threshold_idx = 1:numel(thresholds)
    predicted_positive = score >= thresholds(threshold_idx);
    true_positive_count = sum(predicted_positive & is_true_positive);
    false_positive_count = sum(predicted_positive & ~is_true_positive);

    true_positive_rate(threshold_idx) = true_positive_count / positive_count;
    false_positive_rate(threshold_idx) = false_positive_count / negative_count;
    predicted_positive_count = true_positive_count + false_positive_count;
    if predicted_positive_count > 0
        precision(threshold_idx) = true_positive_count / predicted_positive_count;
    end
end
recall = true_positive_rate;

% trapz needs FPR ascending; threshold ascending gives FPR descending
% (higher threshold -> fewer predicted positive -> lower FPR), so abs()
% the result instead of re-sorting.
roc_auc = abs(trapz(false_positive_rate, true_positive_rate));

% The highest thresholds predict nobody positive (0/0 = NaN precision)
% and get dropped -- but this model is confident enough that recall is
% already well above 0 at the last non-degenerate threshold, so
% dropping those NaNs without replacement silently truncates the
% integral before it reaches recall=0, undercounting pr_auc by the
% area under that missing (high-precision) tail. Fix: append the
% boundary point (recall=0, precision=1), same convention sklearn's
% precision_recall_curve() uses -- zero predictions means zero false
% positives, so precision there is defined as 1.
pr_valid = ~isnan(precision);
recall_for_pr = [recall(pr_valid), 0];
precision_for_pr = [precision(pr_valid), 1];
pr_auc = abs(trapz(recall_for_pr, precision_for_pr));

%% ================================================================
% Calibration curve: equal-width probability bins
% =================================================================

bin_edges = linspace(0, 1, calibration_bin_count + 1);
bin_mean_predicted = nan(1, calibration_bin_count);
bin_empirical_fraction = nan(1, calibration_bin_count);
bin_sample_count = zeros(1, calibration_bin_count);

fprintf('Calibration bins (positive class = %s):\n', positive_class_label);
for bin_idx = 1:calibration_bin_count
    if bin_idx == calibration_bin_count
        in_bin = score >= bin_edges(bin_idx) & score <= bin_edges(bin_idx + 1);
    else
        in_bin = score >= bin_edges(bin_idx) & score < bin_edges(bin_idx + 1);
    end
    bin_sample_count(bin_idx) = sum(in_bin);
    if bin_sample_count(bin_idx) > 0
        bin_mean_predicted(bin_idx) = mean(score(in_bin));
        bin_empirical_fraction(bin_idx) = mean(is_true_positive(in_bin));
    end
    fprintf('  [%.2f, %.2f): n=%4d  mean predicted=%s  empirical fraction=%s\n', ...
        bin_edges(bin_idx), bin_edges(bin_idx + 1), bin_sample_count(bin_idx), ...
        format_or_dash(bin_mean_predicted(bin_idx)), ...
        format_or_dash(bin_empirical_fraction(bin_idx)));
end

%% ================================================================
% Figure 1: six diagnostic tiles
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- Tile 1: confusion matrix ---
nexttile;
confusion_counts = zeros(2, 2);
for true_idx = 1:2
    for predicted_idx = 1:2
        confusion_counts(true_idx, predicted_idx) = sum( ...
            true_label == class_order(true_idx) & ...
            predicted_label == class_order(predicted_idx));
    end
end
imagesc(confusion_counts);
colormap(gca, 'parula');
colorbar;
set(gca, 'XTick', 1:2, 'XTickLabel', cellstr(class_order), ...
    'YTick', 1:2, 'YTickLabel', cellstr(class_order));
xlabel('Predicted');
ylabel('True');
title(sprintf('Confusion matrix (acc=%.3f)', overall_accuracy));
for true_idx = 1:2
    for predicted_idx = 1:2
        text(predicted_idx, true_idx, num2str(confusion_counts(true_idx, predicted_idx)), ...
            'HorizontalAlignment', 'center', 'Color', 'k', 'FontWeight', 'bold');
    end
end

% --- Tile 2: ROC curve ---
nexttile;
plot(false_positive_rate, true_positive_rate, '-', 'LineWidth', 1.5);
hold on;
plot([0, 1], [0, 1], 'k--');
grid on;
axis square;
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('ROC (AUC=%.3f)', roc_auc));

% --- Tile 3: Precision-Recall curve ---
nexttile;
plot(recall_for_pr, precision_for_pr, '-', 'LineWidth', 1.5, ...
    'DisplayName', 'Precision-Recall');
hold on;
yline(positive_count / sample_count, 'k--', 'DisplayName', 'Chance (class prevalence)');
grid on;
axis square;
ylim([0, 1]);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Precision-Recall (AUC=%.3f)', pr_auc));
legend('Location', 'best', 'FontSize', 7);

% --- Tile 4: calibration curve ---
nexttile;
plot([0, 1], [0, 1], 'k--');
hold on;
plot(bin_mean_predicted, bin_empirical_fraction, 'o-', 'LineWidth', 1.5, ...
    'MarkerFaceColor', 'auto');
grid on;
axis square;
xlim([0, 1]);
ylim([0, 1]);
xlabel('Mean predicted probability');
ylabel('Empirical positive fraction');
title('Calibration');

% --- Tile 5: standardized coefficients ---
nexttile;
[~, sort_idx] = sort(abs(coefficients_table.standardized_coefficient));
sorted_features = coefficients_table.feature(sort_idx);
sorted_coefficients = coefficients_table.standardized_coefficient(sort_idx);
bar_colors = repmat([0.20, 0.50, 0.80], numel(sorted_coefficients), 1);
bar_colors(sorted_coefficients < 0, :) = repmat([0.85, 0.20, 0.20], ...
    sum(sorted_coefficients < 0), 1);
barh(sorted_coefficients, 'FaceColor', 'flat', 'CData', bar_colors);
set(gca, 'YTick', 1:numel(sorted_features), 'YTickLabel', ...
    cellstr(sorted_features), 'TickLabelInterpreter', 'none', 'FontSize', 7);
grid on;
xlabel('Standardized coefficient');
title(sprintf('Coefficients (+ pushes toward %s)', negative_class_label));

% --- Tile 6: predicted-probability histogram by true class ---
nexttile;
histogram_edges = linspace(0, 1, probability_histogram_bin_count + 1);
histogram(score(is_true_positive), histogram_edges, ...
    'Normalization', 'probability', 'FaceColor', [0.85, 0.20, 0.20], ...
    'FaceAlpha', 0.6, 'DisplayName', char(positive_class_label));
hold on;
histogram(score(~is_true_positive), histogram_edges, ...
    'Normalization', 'probability', 'FaceColor', [0.20, 0.50, 0.80], ...
    'FaceAlpha', 0.6, 'DisplayName', char(negative_class_label));
grid on;
xlabel(sprintf('Predicted probability of %s', positive_class_label));
ylabel('Fraction of windows');
title('Predicted probability by true class');
legend('Location', 'best', 'FontSize', 7);

sgtitle(sprintf('Logistic regression test-set performance (n=%d)', sample_count));

%% ================================================================
% Figure 2: predicted probability as a 3rd dimension, stem3 plot over
% each window's (x, y) position on the held-out shape.
% =================================================================

have_shape_column = ismember('shape', predictions_table.Properties.VariableNames);
if have_shape_column
    figure2_mask = predictions_table.shape == shape_name;
    distinct_shapes = unique(predictions_table.shape);
    fprintf(['\nFigure 2 filtered to shape="%s": %d of %d rows kept ' ...
        '(held-out set spans %d shape(s): %s).\n'], shape_name, ...
        sum(figure2_mask), sample_count, numel(distinct_shapes), ...
        strjoin(cellstr(distinct_shapes), ', '));
    if ~any(figure2_mask)
        error(['No rows with shape="%s" in %s -- check shape_name.'], ...
            shape_name, read_predictions_csv_file);
    end
else
    figure2_mask = true(sample_count, 1);
    warning(['predictions CSV has no "shape" column -- Figure 2 pools ' ...
        'EVERY held-out row regardless of shape. Rerun train_logistic_' ...
        'regression.py (identifying_columns now includes "shape") to fix.']);
end

position_x = predictions_table.(position_x_column_name)(figure2_mask);
position_y = predictions_table.(position_y_column_name)(figure2_mask);
is_fine_true = predictions_table.(target_column_name)(figure2_mask) == ...
    positive_class_label;
figure2_score = score(figure2_mask);
figure2_sample_count = sum(figure2_mask);

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
hold on;
stem3(position_x(is_fine_true), position_y(is_fine_true), ...
    figure2_score(is_fine_true), 'Color', [0.85, 0.20, 0.20], ...
    'Marker', '.', 'MarkerSize', 8, ...
    'DisplayName', sprintf('True %s', positive_class_label));
stem3(position_x(~is_fine_true), position_y(~is_fine_true), ...
    figure2_score(~is_fine_true), 'Color', [0.20, 0.50, 0.80], ...
    'Marker', '.', 'MarkerSize', 8, ...
    'DisplayName', sprintf('True %s', negative_class_label));
grid on;
view(-35, 25);
zlim([0, 1]);
xlabel('Robot mean position x [m]');
ylabel('Robot mean position y [m]');
zlabel(sprintf('Predicted probability of %s', positive_class_label));
title(sprintf(['Predicted P(%s) over shape="%s", per window (n=%d)'], ...
    positive_class_label, shape_name, figure2_sample_count));
legend('Location', 'best', 'FontSize', 7);

savefig(figure_handles, write_combined_fig);

%% ================================================================
% Command-window summary
% =================================================================

fprintf('\n--- Logistic regression performance summary ---\n');
fprintf('  n = %d (%d %s, %d %s)\n', sample_count, positive_count, ...
    positive_class_label, negative_count, negative_class_label);
fprintf('  Accuracy:  %.4f\n', overall_accuracy);
fprintf('  ROC AUC:   %.4f\n', roc_auc);
fprintf('  PR AUC:    %.4f\n', pr_auc);
fprintf('\n');


%% ================================================================
%% Local functions
