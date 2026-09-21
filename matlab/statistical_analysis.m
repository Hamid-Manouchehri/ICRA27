%% Paired t-tests, Chen vs. gated, on ET.m's batch-sweep output.
%
% Inputs:
%   data/processed_data/<read_data_mat>   X (N-by-9), x_column_names,
%     bandwidth_label_L, shape_name_L, sID -- written by ET.m. Rows
%     ET.m skipped (missing CSV) are NaN in every metric column and are
%     dropped here before any test.
%
% Output:
%   Printed paired t-tests (mean_error_mm, dsj, path_length_m; Chen vs
%   gated), pooling both bandwidths and then one bandwidth at a time, a
%   subject-level gated-minus-Chen difference table, a leave-N-subjects-
%   out sensitivity check, and a z-score outlier check -- all
%   to the command window, nothing written to disk. Both bandwidths are
%   always included; analysis_shape_name restricts every section to one
%   shape (or pools all shapes, the original behavior, if left "").
%   analysis_exclude_subject_ids drops one or more subjects from every
%   section (empty = exclude no one).

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));

%% User Settings
% TODO .mat written by ET.m.
read_data_mat = fullfile(script_directory, '..', 'data', ...
    'processed_data', 'ET_data.mat');

% TODO subject ids (sID entries, e.g. {"s21", "s20"}) excluded in the
% sensitivity check below -- one or several. Empty cell = skip that
% check entirely (exclude no one).
sensitivity_exclude_subject_ids = {};

% TODO restrict every analysis below to one shape (a shape_name_L
% entry, e.g. "pear") -- all subjects and both bandwidths are still
% included. Set to "" to pool all shapes (the original behavior).
analysis_shape_name = "ellipse";

% TODO subject ids (sID entries) dropped from EVERY section below, not
% just the sensitivity check above -- e.g. {"s22", "s23"}. Empty cell =
% exclude no one.
analysis_exclude_subject_ids = {};

fprintf('Input: %s\n\n', read_data_mat);

%% Load and unpack
loaded = load(read_data_mat);
X = loaded.X;
x_column_names = loaded.x_column_names;
bandwidth_label_L = loaded.bandwidth_label_L;
shape_name_L = loaded.shape_name_L;
sID = loaded.sID;

valid_rows = ~isnan(X(:, strcmp(x_column_names, 'chen_mean_error_mm')));
fprintf('%d of %d rows kept (%d skipped by ET.m, missing CSV).\n', ...
    sum(valid_rows), numel(valid_rows), sum(~valid_rows));
X = X(valid_rows, :);

if strlength(analysis_shape_name) > 0
    % string(shape_name_L), not strcmp -- see the sID/string comment below.
    analysis_shape_index = find(string(shape_name_L) == analysis_shape_name);
    if isempty(analysis_shape_index)
        error('analysis_shape_name "%s" not found in shape_name_L.', ...
            analysis_shape_name);
    end
    shape_index = X(:, strcmp(x_column_names, 'shape_index'));
    X = X(shape_index == analysis_shape_index, :);
    fprintf('Restricted to shape = "%s": %d rows remain.\n', ...
        analysis_shape_name, size(X, 1));
    shape_label_text = sprintf('shape=%s', analysis_shape_name);
else
    fprintf('All shapes pooled (analysis_shape_name is "").\n');
    shape_label_text = 'all shapes pooled';
end

if ~isempty(analysis_exclude_subject_ids)
    exclude_subject_indices = resolve_subject_indices( ...
        sID, analysis_exclude_subject_ids);
    subject_index_all = X(:, strcmp(x_column_names, 'subject_index'));
    X = X(~ismember(subject_index_all, exclude_subject_indices), :);
    fprintf('Excluded subjects {%s}: %d rows remain.\n', ...
        strjoin(cellstr(string(analysis_exclude_subject_ids)), ', '), ...
        size(X, 1));
end

bandwidth_index = X(:, strcmp(x_column_names, 'bandwidth_index'));
subject_index = X(:, strcmp(x_column_names, 'subject_index'));

% Column lookup, one pair per metric -- errors clearly if ET.m's
% x_column_names ever drifts from what this script expects.
metric_names = {'mean_error_mm', 'dsj', 'f_dsj','path_length_m', 'mean_force_n'};
metric_count = numel(metric_names);
chen_columns = zeros(1, metric_count);
gated_columns = zeros(1, metric_count);
for metric_idx = 1:metric_count
    chen_columns(metric_idx) = find_column(x_column_names, ...
        ['chen_' metric_names{metric_idx}]);
    gated_columns(metric_idx) = find_column(x_column_names, ...
        ['gated_' metric_names{metric_idx}]);
end

subjects = unique(subject_index);
subject_count = numel(subjects);
subject_labels = sID(subjects)';

%% ============================================================
% 1. Chen vs gated, pooling both bandwidths (average every bandwidth x
%    shape combo -- one shape if analysis_shape_name is set -- within
%    each subject)
% ============================================================

mean_chen_all = nan(subject_count, metric_count);
mean_gated_all = nan(subject_count, metric_count);
for subject_idx = 1:subject_count
    row_mask = subject_index == subjects(subject_idx);
    mean_chen_all(subject_idx, :) = mean(X(row_mask, chen_columns), 1);
    mean_gated_all(subject_idx, :) = mean(X(row_mask, gated_columns), 1);
end

fprintf('\nBOTH BANDWIDTHS, %s (n = %d subjects)\n', shape_label_text, ...
    subject_count);
for metric_idx = 1:metric_count
    %[~, p, ~, stats] = ttest(mean_chen_all(:, metric_idx), mean_gated_all(:, metric_idx));
    %fprintf('%-16s p = %.4f, t = %.3f\n', metric_names{metric_idx},p, stats.tstat);
    [p, h, stats] = signrank(mean_chen_all(:, metric_idx), mean_gated_all(:, metric_idx));
    fprintf('%-16s p = %.4f, t = %.3f\n', metric_names{metric_idx},p, stats.signedrank);
end

%% ============================================================
% 2. Chen vs gated, one bandwidth at a time
% ============================================================

for bandwidth_idx = 1:numel(bandwidth_label_L)
    mean_chen_bw = nan(subject_count, metric_count);
    mean_gated_bw = nan(subject_count, metric_count);
    for subject_idx = 1:subject_count
        row_mask = subject_index == subjects(subject_idx) & ...
            bandwidth_index == bandwidth_idx;
        if any(row_mask)
            mean_chen_bw(subject_idx, :) = mean(X(row_mask, chen_columns), 1);
            mean_gated_bw(subject_idx, :) = mean(X(row_mask, gated_columns), 1);
        end
    end

    valid_subjects = all(~isnan(mean_chen_bw), 2);
    fprintf('\n%s ONLY, %s (n = %d of %d subjects)\n', ...
        upper(bandwidth_label_L{bandwidth_idx}), shape_label_text, ...
        sum(valid_subjects), subject_count);
    for metric_idx = 1:metric_count
        %[~, p] = ttest(mean_chen_bw(valid_subjects, metric_idx), mean_gated_bw(valid_subjects, metric_idx));
        [p,h] = signrank(mean_chen_bw(valid_subjects, metric_idx), mean_gated_bw(valid_subjects, metric_idx));
        fprintf('%-16s p = %.4f\n', metric_names{metric_idx}, p);
    end
end

%% ============================================================
% 3. Subject-level gated-minus-Chen differences (both bandwidths pooled)
% ============================================================

diff_all = mean_gated_all - mean_chen_all;

T = array2table(diff_all, 'VariableNames', ...
    strcat('gatedMinusChen_', metric_names));
T = addvars(T, subject_labels, 'Before', 1, 'NewVariableNames', {'Subject'});
disp(T)

%% ============================================================
% 4. Sensitivity check: does excluding one or more subjects change the
%    result?
% ============================================================

if isempty(sensitivity_exclude_subject_ids)
    fprintf(['\nSensitivity check skipped (sensitivity_exclude_subject_ids ' ...
        'is empty).\n']);
else
    sensitivity_exclude_indices = resolve_subject_indices( ...
        sID, sensitivity_exclude_subject_ids);
    keep = ~ismember(subjects, sensitivity_exclude_indices);

    fprintf('\nWITHOUT {%s}\n', ...
        strjoin(cellstr(string(sensitivity_exclude_subject_ids)), ', '));
    for metric_idx = 1:metric_count
        %[~, p] = ttest(mean_chen_all(keep, metric_idx), mean_gated_all(keep, metric_idx));
        [p, h] = signrank(mean_chen_all(keep, metric_idx), mean_gated_all(keep, metric_idx));
        fprintf('%-16s p = %.4f\n', metric_names{metric_idx}, p);
    end
end

%% ============================================================
% 5. Simple subject-level outlier check (z-score of the gated-minus-
%    Chen difference, both bandwidths pooled)
% ============================================================

z_scores = (diff_all - mean(diff_all, 1)) ./ std(diff_all, 0, 1);

Toutlier = array2table([diff_all, z_scores], 'VariableNames', ...
    [strcat('Diff_', metric_names), strcat('Z_', metric_names)]);
Toutlier = addvars(Toutlier, subject_labels, 'Before', 1, ...
    'NewVariableNames', {'Subject'});
disp(Toutlier)


%% ================================================================
%% Local functions
