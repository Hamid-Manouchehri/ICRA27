%% Gaze behavior comparison: Chen baseline vs. classifier-gated.
%
% Inputs:
%   data/raw_data/test_subjects/<chen_trials(i).gaze_csv_file>    one
%     Tobii CSV per (shape, bandwidth) cell for the baseline arm.
%   data/raw_data/test_subjects/<gated_trials(i).gaze_csv_file>   the
%     matching cell for the gated arm -- same subject, same shape, same
%     bandwidth as chen_trials(i), in the SAME ORDER (validated at
%     startup, not just assumed).
%
% Outputs:
%   Figure 1 (2x2 tiles), one grouped bar chart per metric, one group
%     per (shape, bandwidth) cell, two bars per group (Chen vs. gated):
%     valid-gaze fraction, mean pupil diameter, gaze dispersion, mean
%     gaze speed. Also saved as data/images/figures/<write_comparison_
%     fig> (a native MATLAB .fig, fully interactive when reopened), so
%     matlab/plot_saved_gaze_behavior_comparison.m can display it later
%     without re-reading any Tobii CSV.
%   A numeric summary printed to the command window.
%
% Method:
%   Every metric comes directly from the raw Tobii CSV's normalized
%   display-area gaze point ([0, 1] x [0, 1], both eyes averaged when
%   both are valid) and pupil diameter -- no calibration JSON and no
%   robot CSV needed, so this stays independent of which controller (or
%   which session's calibration) produced the trial and of the trial's
%   trace direction (not recorded in these filenames). Gaze-to-end-
%   effector distance would need calibration and the robot CSV, and is
%   deliberately left out of this "simple" version.
%
% Assumptions:
%   chen_trials and gated_trials are matched pairs by (shape,
%   bandwidth), in identical order -- checked explicitly at startup, not
%   just assumed.
%
% Method reuse:
%   None; every function here is new, not copied from another script.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'test_subjects');

%% User Settings
% TODO baseline-arm trials -- one entry per (shape, bandwidth) cell.
% Subject fixed at s14 for every cell by default, so this is a same-
% subject comparison (a different subject per cell would confound the
% controller effect with between-subject variability).
chen_trials = struct( ...
    'shape', {'ellipse', 'ellipse', 'pear', 'pear', 'fine_gross', 'fine_gross'}, ...
    'bandwidth', {'narrow', 'wide', 'narrow', 'wide', 'narrow', 'wide'}, ...
    'gaze_csv_file', { ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_narrow_ellipse_shape_s17.csv'), ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_wide_ellipse_shape_s17.csv'), ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_narrow_pear_shape_s17.csv'), ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_wide_pear_shape_s17.csv'), ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_narrow_fine_gross_shape_s17.csv'), ...
        fullfile(raw_data_directory, 'tobii_chen_vac_2d_wide_fine_gross_shape_s17.csv')});
chen_label = "Chen (baseline)";
chen_color = [0.45, 0.45, 0.45];

% TODO gated-arm trials -- SAME (shape, bandwidth) order as chen_trials.
gated_trials = struct( ...
    'shape', {'ellipse', 'ellipse', 'pear', 'pear', 'fine_gross', 'fine_gross'}, ...
    'bandwidth', {'narrow', 'wide', 'narrow', 'wide', 'narrow', 'wide'}, ...
    'gaze_csv_file', { ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s17_on_narrow_ellipse_shape.csv'), ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s17_on_wide_ellipse_shape.csv'), ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_pear_and_s17_on_narrow_pear_shape.csv'), ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_pear_and_s17_on_wide_pear_shape.csv'), ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s17_on_narrow_fine_gross_shape.csv'), ...
        fullfile(raw_data_directory, 'tobii_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s17_on_wide_fine_gross_shape.csv')});
gated_label = "Gaze-assisted (gated)";
gated_color = [0.00, 0.45, 0.74];

% TODO explicit output filename (data/images/figures/) for the Figure 1
% comparison, saved as a native MATLAB .fig (via savefig(), not
% exportgraphics() -- .fig is not one of exportgraphics' supported
% formats). Rename this yourself when chen_trials/gated_trials change
% subject, so a new run doesn't silently overwrite a previous one (no
% auto-generated tag in the filename, per this project's convention).
write_comparison_fig = fullfile(script_directory, '..', 'data', ...
    'images', 'figures', 'compare_gaze_behavior_chen_vs_gated', ...
    'gaze_behavior_chen_vs_gated_s17.fig');

% TODO gaze validity gates, matching build_modality_feature_datasets.
% py's own require_gaze_on_display / require_both_eyes_valid settings.
require_gaze_on_display = true;
require_both_eyes_valid = true;

% TODO gap [s] above which a consecutive valid-sample pair is excluded
% from the gaze-speed metric -- avoids a blink/dropout's edges being
% counted as one very large (or, divided by the long gap, misleadingly
% slow) eye movement.
max_gaze_gap_for_speed_s = 0.1;

fprintf('Output: %s\n\n', write_comparison_fig);

%% Validate the two trial lists describe the same cells, same order
if numel(chen_trials) ~= numel(gated_trials)
    error('chen_trials has %d entries, gated_trials has %d -- must match.', ...
        numel(chen_trials), numel(gated_trials));
end
for trial_idx = 1:numel(chen_trials)
    if ~strcmp(chen_trials(trial_idx).shape, gated_trials(trial_idx).shape) || ...
            ~strcmp(chen_trials(trial_idx).bandwidth, gated_trials(trial_idx).bandwidth)
        error(['chen_trials(%d) = %s/%s but gated_trials(%d) = %s/%s -- ' ...
            'the two lists must describe the same (shape, bandwidth) ' ...
            'cells in the same order.'], trial_idx, ...
            chen_trials(trial_idx).shape, chen_trials(trial_idx).bandwidth, ...
            trial_idx, gated_trials(trial_idx).shape, ...
            gated_trials(trial_idx).bandwidth);
    end
end

%% Compute gaze metrics for every cell, both arms
trial_count = numel(chen_trials);
group_labels = strings(1, trial_count);
chen_valid_fraction = nan(1, trial_count);
chen_pupil_mm = nan(1, trial_count);
chen_dispersion = nan(1, trial_count);
chen_speed = nan(1, trial_count);
gated_valid_fraction = nan(1, trial_count);
gated_pupil_mm = nan(1, trial_count);
gated_dispersion = nan(1, trial_count);
gated_speed = nan(1, trial_count);

fprintf('Gaze behavior comparison -- %d (shape, bandwidth) cell(s)\n\n', ...
    trial_count);

for trial_idx = 1:trial_count
    group_labels(trial_idx) = sprintf('%s %s', ...
        chen_trials(trial_idx).bandwidth, chen_trials(trial_idx).shape);

    fprintf('Input:  %s\n', chen_trials(trial_idx).gaze_csv_file);
    fprintf('Input:  %s\n', gated_trials(trial_idx).gaze_csv_file);

    chen_metrics = compute_gaze_metrics(chen_trials(trial_idx).gaze_csv_file, ...
        require_both_eyes_valid, require_gaze_on_display, ...
        max_gaze_gap_for_speed_s);
    gated_metrics = compute_gaze_metrics(gated_trials(trial_idx).gaze_csv_file, ...
        require_both_eyes_valid, require_gaze_on_display, ...
        max_gaze_gap_for_speed_s);

    chen_valid_fraction(trial_idx) = chen_metrics.valid_fraction;
    chen_pupil_mm(trial_idx) = chen_metrics.mean_pupil_diameter_mm;
    chen_dispersion(trial_idx) = chen_metrics.gaze_dispersion;
    chen_speed(trial_idx) = chen_metrics.mean_gaze_speed_per_s;

    gated_valid_fraction(trial_idx) = gated_metrics.valid_fraction;
    gated_pupil_mm(trial_idx) = gated_metrics.mean_pupil_diameter_mm;
    gated_dispersion(trial_idx) = gated_metrics.gaze_dispersion;
    gated_speed(trial_idx) = gated_metrics.mean_gaze_speed_per_s;
end
fprintf('\n');

%% ================================================================
% Figure 1: 4-panel grouped-bar comparison
% =================================================================

figure('Name', 'Gaze behavior: Chen vs. gated', 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_grouped_metric_bars(nexttile, chen_valid_fraction, gated_valid_fraction, ...
    group_labels, chen_label, gated_label, chen_color, gated_color, ...
    'Valid fraction [-]', 'Valid-gaze fraction', '%.3f');

plot_grouped_metric_bars(nexttile, chen_pupil_mm, gated_pupil_mm, ...
    group_labels, chen_label, gated_label, chen_color, gated_color, ...
    'Pupil diameter [mm]', 'Mean pupil diameter (both eyes)', '%.2f');

plot_grouped_metric_bars(nexttile, chen_dispersion, gated_dispersion, ...
    group_labels, chen_label, gated_label, chen_color, gated_color, ...
    'Dispersion [norm.]', 'Gaze dispersion (display-area units)', '%.3f');

plot_grouped_metric_bars(nexttile, chen_speed, gated_speed, ...
    group_labels, chen_label, gated_label, chen_color, gated_color, ...
    'Speed [norm./s]', 'Mean gaze speed (display-area units/s)', '%.2f');

sgtitle('Gaze behavior by (shape, bandwidth): Chen vs. gated');

savefig(gcf, write_comparison_fig);
fprintf('Wrote: %s\n\n', write_comparison_fig);

%% ================================================================
% Summary
% =================================================================

fprintf('%-24s %14s %14s %10s\n', '', char(chen_label), char(gated_label), 'Delta');
print_gaze_row('Valid fraction [-]', chen_valid_fraction, ...
    gated_valid_fraction, group_labels, '%14.3f');
print_gaze_row('Pupil diameter [mm]', chen_pupil_mm, gated_pupil_mm, ...
    group_labels, '%14.2f');
print_gaze_row('Dispersion [norm.]', chen_dispersion, gated_dispersion, ...
    group_labels, '%14.3f');
print_gaze_row('Speed [norm./s]', chen_speed, gated_speed, group_labels, ...
    '%14.2f');
fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================

function xy = parse_display_area_column(text_column)
%PARSE_DISPLAY_AREA_COLUMN Parse a column of "[x,y]" strings (Tobii's
% *_gaze_point_on_display_area cells) into an N-by-2 array, NaN rows
% where the cell is unparseable (e.g. "[NaN,NaN]" during a dropout).

    row_count = numel(text_column);
    xy = nan(row_count, 2);
    for row_idx = 1:row_count
        text = strtrim(char(text_column(row_idx)));
        text = strrep(strrep(text, '[', ''), ']', '');
        parts = strsplit(text, ',');
        if numel(parts) == 2
            xy(row_idx, 1) = str2double(parts{1});
            xy(row_idx, 2) = str2double(parts{2});
        end
    end
end


function metrics = compute_gaze_metrics(gaze_csv_file, ...
        require_both_eyes_valid, require_gaze_on_display, ...
        max_gaze_gap_for_speed_s)
%COMPUTE_GAZE_METRICS Simple gaze-behavior summary for one trial, from
% the raw Tobii CSV alone -- normalized display-area coordinates
% ([0, 1] x [0, 1]), both eyes combined per require_both_eyes_valid.
%
% Output: scalar struct -- sample_count, valid_fraction,
%   mean_pupil_diameter_mm, gaze_dispersion, mean_gaze_speed_per_s.

    gaze_table = readtable(gaze_csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');

    time_s = gaze_table.time_s;
    left_xy = parse_display_area_column( ...
        gaze_table.left_gaze_point_on_display_area);
    right_xy = parse_display_area_column( ...
        gaze_table.right_gaze_point_on_display_area);
    left_point_valid = gaze_table.left_gaze_point_validity == 1;
    right_point_valid = gaze_table.right_gaze_point_validity == 1;

    if require_gaze_on_display
        left_on_display = all( ...
            isfinite(left_xy) & left_xy >= 0 & left_xy <= 1, 2);
        right_on_display = all( ...
            isfinite(right_xy) & right_xy >= 0 & right_xy <= 1, 2);
        left_point_valid = left_point_valid & left_on_display;
        right_point_valid = right_point_valid & right_on_display;
    end

    row_count = height(gaze_table);
    gaze_xy = nan(row_count, 2);
    if require_both_eyes_valid
        gaze_valid = left_point_valid & right_point_valid;
        gaze_xy(gaze_valid, :) = 0.5 * ( ...
            left_xy(gaze_valid, :) + right_xy(gaze_valid, :));
    else
        gaze_valid = left_point_valid | right_point_valid;
        both = left_point_valid & right_point_valid;
        only_left = left_point_valid & ~right_point_valid;
        only_right = right_point_valid & ~left_point_valid;
        gaze_xy(both, :) = 0.5 * (left_xy(both, :) + right_xy(both, :));
        gaze_xy(only_left, :) = left_xy(only_left, :);
        gaze_xy(only_right, :) = right_xy(only_right, :);
    end

    metrics.sample_count = row_count;
    metrics.valid_fraction = mean(gaze_valid);

    valid_xy = gaze_xy(gaze_valid, :);
    valid_time_s = time_s(gaze_valid);
    metrics.gaze_dispersion = mean(std(valid_xy, 0, 1));

    if numel(valid_time_s) > 1
        gaze_step = hypot(diff(valid_xy(:, 1)), diff(valid_xy(:, 2)));
        time_step_s = diff(valid_time_s);
        step_ok = time_step_s > 0 & time_step_s <= max_gaze_gap_for_speed_s;
        metrics.mean_gaze_speed_per_s = ...
            mean(gaze_step(step_ok) ./ time_step_s(step_ok));
    else
        metrics.mean_gaze_speed_per_s = NaN;
    end

    left_pupil_valid = gaze_table.left_pupil_validity == 1;
    right_pupil_valid = gaze_table.right_pupil_validity == 1;
    both_pupil_valid = left_pupil_valid & right_pupil_valid;
    metrics.mean_pupil_diameter_mm = mean(0.5 * ( ...
        gaze_table.left_pupil_diameter(both_pupil_valid) + ...
        gaze_table.right_pupil_diameter(both_pupil_valid)));
end


function plot_grouped_metric_bars(ax, chen_values, gated_values, ...
        group_labels, chen_label, gated_label, chen_color, gated_color, ...
        y_label_text, title_text, value_format)
%PLOT_GROUPED_METRIC_BARS One metric, one tile: N (shape, bandwidth)
% groups, two colored bars per group (Chen vs. gated), each bar's mean
% value printed above it. NaN entries (a missing cell) are simply not
% drawn -- bar() skips them.

    group_count = numel(chen_values);
    values = [chen_values(:), gated_values(:)];

    hold(ax, 'on');
    bar_handle = bar(ax, values);
    bar_handle(1).FaceColor = chen_color;
    bar_handle(2).FaceColor = gated_color;
    bar_handle(1).DisplayName = char(chen_label);
    bar_handle(2).DisplayName = char(gated_label);

    for group_idx = 1:group_count
        for series_idx = 1:2
            value = values(group_idx, series_idx);
            if isnan(value)
                continue;
            end
            text(ax, bar_handle(series_idx).XEndPoints(group_idx), ...
                bar_handle(series_idx).YEndPoints(group_idx), ...
                sprintf(value_format, value), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 6);
        end
    end

    set(ax, 'XTick', 1:group_count, 'XTickLabel', cellstr(group_labels), ...
        'XTickLabelRotation', 20);
    grid(ax, 'on');
    ylabel(ax, y_label_text);
    title(ax, title_text, 'FontSize', 9);
    legend(ax, 'Location', 'best', 'FontSize', 7);
end


function print_gaze_row(name, chen_values, gated_values, group_labels, ...
        value_format)
%PRINT_GAZE_ROW One metric's rows: every group's Chen/gated values and
% their difference.

    fprintf('%s:\n', name);
    for group_idx = 1:numel(group_labels)
        fprintf(['  %-22s ' value_format ' ' value_format ' %+10.4g\n'], ...
            char(group_labels(group_idx)), chen_values(group_idx), ...
            gated_values(group_idx), ...
            gated_values(group_idx) - chen_values(group_idx));
    end
end
