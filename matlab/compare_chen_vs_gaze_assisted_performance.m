%% Chen baseline vs. classifier-gated: performance metric comparison.
%
% Inputs:
%   data/raw_data/<read_chen_robot_csv_files>    one or more ur5e_chen_var_
%     adm_2d.py trial CSVs (the baseline arm).
%   data/raw_data/<read_gated_robot_csv_files>   one or more ur5e_gaze_
%     assisted_chen_vac_2d.py trial CSVs (the gated arm), same shape.
%   data/shapes/<read_shape_obj_file>            reference band mesh for the
%     traced shape, shared by both arms (they must trace the same shape
%     for the comparison to mean anything).
%   All three of the above, plus the output filename, are DERIVED from
%   the "Trial identity" block in User Settings (bandwidth_label,
%   shape_name, chen_subject_id, gated_subject_id) -- to switch trials,
%   edit that block only, not the paths themselves.
%
% Outputs:  
%   Figure 1: 4-panel bar comparison, one panel per metric -- mean
%     tracking error (paper Eq. 15), mean interaction force (Eq. 17),
%     DSJ smoothness (Eq. 16), and traversed path length. Each bar is the mean
%     across that arm's trials, with a std error bar (zero-length when
%     an arm has only one trial).
%   Figure 2: control-loop period distribution for both arms, log-scale
%     y -- tests whether the gated arm's classifier thread measurably
%     degraded loop timing relative to the untouched baseline.
%   Figure 3: the same metrics split by the HUMAN Fine/Gross labels, if
%     both arms' CSVs carry a "label" column (otherwise skipped with a
%     note). Trial-pooled means cancel a redistribution effect -- both
%     arms sweep a similar damping range and merely choose different
%     moments to be stiff -- so this is where such an effect can
%     actually be seen. NB a geometric proxy for the regions (mesh band
%     width) was tried and rejected: it agrees with the human labels
%     only ~49% of the time on this shape, i.e. not at all.
%   A numeric summary printed to the command window, including each
%     arm's own reference-registration check (see shape_offset_xy_m
%     below) so a bad offset is visible rather than silently inflating
%     the tracking-error comparison.
%   Figures 1 and 3 (the latter only when produced) are also saved
%   TOGETHER in one native MATLAB .fig file under data/images/figures/
%   compare_chen_vs_gaze_assisted_performance/ (see write_combined_fig
%   in User Settings), so plot_saved_chen_vs_gaze_assisted_performance.m
%   can reopen whichever figures were produced without rereading any
%   CSV or the mesh.
%
% Assumptions:
%   Both arms' CSVs were written by their respective controller scripts,
%   so column names match those scripts' write_csv_header(). Both arms
%   traced the SAME physical shape -- comparing different shapes would
%   make every metric here meaningless, not just the tracking error.
%   shape_offset_xy_m below (one entry per shape_name) is copied from
%   plot_raw_data_chen_var_adm_2d.m's own per-shape comments (not
%   independently re-derived here) -- verify it via this script's own
%   printed registration-shift diagnostic before trusting the error
%   numbers. The gated arm's CSV also encodes which shape+subject its
%   model was held out of training on; every trial run so far has that
%   pair equal to its own trial identity (see held_out_shape_name /
%   held_out_subject_id in User Settings) -- verified against every CSV
%   under data/raw_data/test_subjects/ as of this writing.
%
% Method reuse:
%   read_shape_band_obj, resample_closed_loop, distance_to_reference_
%   path, estimate_reference_shift, compute_jerk_series,
%   window_to_odd_sample_count, and compute_dsj are copied verbatim from
%   plot_raw_data_chen_var_adm_2d.m, self-contained here per this
%   project's convention that MATLAB scripts do not call one another.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data', ...
'test_subjects');

%% User Settings
% TODO explicit directory where this script's figures are saved, one
% subfolder per script for organization (see AI_GUIDELINES.md).
write_image_directory = fullfile(script_directory, '..', 'data', ...
    'images', 'figures', 'compare_chen_vs_gaze_assisted_performance');

% TODO -------------------- Trial identity --------------------
% Everything below this block (CSV filenames, shape mesh, registration
% offset, output figure name) is DERIVED from these few tags -- switch
% trials by editing only this block, never the paths further down.
% Verified against every file actually present under data/raw_data/
% test_subjects/ and data/shapes/ as of this writing.
bandwidth_label = "wide";      % TODO 'narrow' or 'wide'
shape_name = "fine_gross";        % TODO 'ellipse', 'pear', or 'fine_gross'
chen_subject_id = "s19";       % TODO baseline-arm subject, e.g. "s11"
gated_subject_id = "s19";      % TODO gated-arm subject -- same trial
                                % pair as chen_subject_id unless
                                % deliberately comparing across subjects

% The gated arm's CSV filename also encodes which shape+subject its
% model bundle was held out of training on (leave-one-out
% generalization test). That pair equals the trial's own shape+subject
% in every run so far, so these default to the values above -- override
% only if a specific run's held-out pair truly differs from its own
% trial identity.
held_out_shape_name = shape_name;
held_out_subject_id = gated_subject_id;

shape_tag = bandwidth_label + "_" + shape_name;
% TODO -------------------- End trial identity --------------------

% Output filename (native MATLAB .fig; every figure below is bundled
% into this ONE file via savefig()'s multi-handle form, not one file
% per figure -- see plot_saved_chen_vs_gaze_assisted_performance.m,
% which reopens all of them at once with a single openfig()). Figure 3
% (per-region comparison) is only produced when both arms' CSVs carry a
% "label" column (see have_regions below); when it is skipped, this
% file simply holds Figure 1 alone.
write_combined_fig = fullfile(write_image_directory, sprintf( ...
    'chen_vs_gaze_assisted_performance_%s_%s.fig', ...
    shape_tag, chen_subject_id));

% Figure window title -- same as the saved file's own name (no
% directory, no extension), so a reopened window is traceable to its
% .fig file at a glance.
[~, write_combined_fig_name, ~] = fileparts(write_combined_fig);

% Baseline-arm trial CSV. To add more matched trial pairs, append more
% cells here by hand -- the bar/error-bar plotting below already
% handles any trial count.
% For the PER-REGION comparison this must be a *_live_labeled_*.csv
% output of manual_label_chen_var_adm_2d_by_live_replay.py (it carries
% a "label" column on top of every controller column). A raw CSV still
% works for the whole-trial metrics; the per-region figure is skipped.
read_chen_robot_csv_files = {fullfile(raw_data_directory, sprintf( ...
    'ur5e_chen_vac_2d_%s_shape_%s.csv', shape_tag, chen_subject_id))};

chen_label = "Chen (baseline)";

% Gated-arm trial CSV, same shape as read_chen_robot_csv_files above.
read_gated_robot_csv_files = {fullfile(raw_data_directory, sprintf( ...
    'ur5e_gaze_assisted_chen_vac_2d_held_out_%s_and_%s_on_%s_shape.csv', ...
    held_out_shape_name, held_out_subject_id, shape_tag))};

gated_label = "Gaze-assisted (gated)";

% Shape mesh both arms traced, used for the tracking-error metric.
read_shape_obj_file = fullfile(script_directory, '..', 'data', 'shapes', ...
    sprintf('%s_shape.obj', shape_tag));

% Translation from the mesh CAD frame to the CoppeliaSim frame [m], one
% entry per shape_name. Copied from plot_raw_data_chen_var_adm_2d.m's
% own per-shape comments (not independently re-derived here) -- VERIFY
% against this script's printed registration-shift diagnostic before
% trusting the error numbers.
switch shape_name
    case "ellipse"
        shape_offset_xy_m = [-0.015, -0.650];
    case "pear"
        shape_offset_xy_m = [0.025, -0.660];
    case "fine_gross"
        shape_offset_xy_m = [-0.0, -0.650];
    otherwise
        % TODO add a case above before using a new shape_name.
        error(['No shape_offset_xy_m entry for shape_name "%s" -- add ' ...
            'one to the switch above.'], shape_name);
end

% TODO spacing used to resample the reference path [m], positive --
% sets the quantization floor of the tracking-error metric.
reference_spacing_m = 5e-4;

% TODO speed separating movement from rest [m/s]. The paper uses 0.005
% and evaluates tracking error and mean force on movement samples only.
moving_speed_threshold_m_s = 0.005;

% TODO filter windows for the jerk chain feeding DSJ [s]. DSJ values are
% comparable between arms only when computed with the same windows,
% which this script guarantees by using one setting for both.
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;

% TODO nominal control-loop period [s] -- must match both controllers'
% 1 / control_frequency_hz (500 Hz for both as of this writing).
control_period_nominal_s = 1.0 / 500.0;

% TODO a tick slower than this many times the nominal period counts as
% an "overrun" in the printed summary and the Figure 2 threshold line.
control_period_overrun_factor = 1.5;

% TODO cap on samples fed to the registration-shift diagnostic, which
% re-scores the path against the reference 242 times -- the full trial
% would take minutes. It is a diagnostic, not a reported metric, so a
% subsample is enough.
registration_check_max_samples = 2000;

chen_color = [0.45, 0.45, 0.45];
gated_color = [0.00, 0.45, 0.74];

fprintf('Input:  %s\n', read_chen_robot_csv_files{1});
fprintf('Input:  %s\n', read_gated_robot_csv_files{1});
fprintf('Input:  %s\n', read_shape_obj_file);
fprintf('Output: %s\n\n', write_combined_fig);

if ~isfolder(write_image_directory)
    mkdir(write_image_directory);
end
figure_handles = gobjects(0);

%% Load the shared reference path
[~, ~, reference_xy_m] = read_shape_band_obj( ...
    read_shape_obj_file, reference_spacing_m);
if isempty(reference_xy_m)
    error(['Shape mesh "%s" did not yield two edge loops -- tracking ' ...
        'error cannot be computed. Check read_shape_obj_file.'], read_shape_obj_file);
end
reference_xy_m = reference_xy_m + shape_offset_xy_m;

%% Compute metrics for both arms
chen_metrics = compute_arm_metrics(read_chen_robot_csv_files, reference_xy_m, ...
    moving_speed_threshold_m_s, jerk_outlier_window_s, ...
    jerk_smoothing_window_s, control_period_nominal_s, ...
    control_period_overrun_factor, registration_check_max_samples);
gated_metrics = compute_arm_metrics(read_gated_robot_csv_files, reference_xy_m, ...
    moving_speed_threshold_m_s, jerk_outlier_window_s, ...
    jerk_smoothing_window_s, control_period_nominal_s, ...
    control_period_overrun_factor, registration_check_max_samples);

have_regions = chen_metrics.has_labels && gated_metrics.has_labels;
if ~have_regions
    fprintf(['\nNOTE: per-region comparison SKIPPED -- it needs a "label" ' ...
        'column (Fine/Gross)\nin both arms'' CSVs. Label the trials with ' ...
        'scripts/manual_label_chen_var_adm_2d_by_live_replay.py\nand point ' ...
        'the settings above at the *_live_labeled_*.csv outputs.\n']);
end

%% ================================================================
% Figure 1: the 4 requested metrics, one bar pair each
% =================================================================

figure_handles(end + 1) = figure('Name', ...
    write_combined_fig_name, 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

plot_metric_bars(nexttile, chen_metrics.mean_error_mm, ...
    gated_metrics.mean_error_mm, chen_label, gated_label, chen_color, ...
    gated_color, 'Error [mm]', 'Tracking error (Eq. 15)', '%.2f mm');

plot_metric_bars(nexttile, chen_metrics.mean_force_n, ...
    gated_metrics.mean_force_n, chen_label, gated_label, chen_color, ...
    gated_color, 'Force [N]', 'Mean interaction force (Eq. 17)', '%.3f N');

plot_metric_bars(nexttile, chen_metrics.dsj, gated_metrics.dsj, ...
    chen_label, gated_label, chen_color, gated_color, 'DSJ [-]', ...
    'Smoothness, dimensionless squared jerk (Eq. 16)', '%.3g');

plot_metric_bars(nexttile, chen_metrics.path_length_m, ...
    gated_metrics.path_length_m, chen_label, gated_label, chen_color, ...
    gated_color, 'Path length [m]', 'Traversed path length', '%.3f m');

sgtitle('Chen baseline vs. classifier-gated controller');

%% ================================================================
% Figure 2: control-loop period distribution (both arms)
% =================================================================
% Tests, on the actual recorded trials rather than a synthetic bench
% test, whether the gated arm's classifier thread measurably slowed the
% 500 Hz loop relative to the untouched baseline.

% all_chen_period_ms = vertcat(chen_metrics.period_ms{:});
% all_gated_period_ms = vertcat(gated_metrics.period_ms{:});
% 
% figure('Name', 'Control loop timing', 'Color', 'w');
% hold on;
% histogram(all_chen_period_ms, 'BinWidth', 0.1, ...
%     'Normalization', 'probability', 'FaceColor', chen_color, ...
%     'FaceAlpha', 0.5, 'DisplayName', chen_label);
% histogram(all_gated_period_ms, 'BinWidth', 0.1, ...
%     'Normalization', 'probability', 'FaceColor', gated_color, ...
%     'FaceAlpha', 0.5, 'DisplayName', gated_label);
% xline(1e3 * control_period_nominal_s, 'k--', 'Nominal', ...
%     'HandleVisibility', 'off');
% xline(1e3 * control_period_nominal_s * control_period_overrun_factor, ...
%     'r--', 'Overrun threshold', 'HandleVisibility', 'off');
% set(gca, 'YScale', 'log');
% grid on;
% xlabel('Control loop period [ms]');
% ylabel('Fraction of ticks (log scale)');
% title('Control loop period distribution');
% legend('Location', 'best');

%% ================================================================
% Figure 3: the same metrics, split by the shape's fine/gross regions
% =================================================================
% This is where a redistribution effect can actually show up. Both arms
% sweep a similar damping range over a whole trial, so trial-pooled
% means cancel out "stiff at the right moments" against "stiff at
% force-determined moments". Splitting by region does not.

if have_regions
    figure_handles(end + 1) = figure('Name', ...
        write_combined_fig_name, 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    plot_region_bars(nexttile, chen_metrics.fine_error_mm, ...
        gated_metrics.fine_error_mm, chen_metrics.gross_error_mm, ...
        gated_metrics.gross_error_mm, chen_label, gated_label, chen_color, ...
        gated_color, 'Error [mm]', 'Tracking error by region', '%.2f');

    plot_region_bars(nexttile, chen_metrics.fine_force_n, ...
        gated_metrics.fine_force_n, chen_metrics.gross_force_n, ...
        gated_metrics.gross_force_n, chen_label, gated_label, chen_color, ...
        gated_color, 'Force [N]', 'Mean force by region', '%.2f');

    plot_region_bars(nexttile, chen_metrics.fine_damping_n_s_m, ...
        gated_metrics.fine_damping_n_s_m, chen_metrics.gross_damping_n_s_m, ...
        gated_metrics.gross_damping_n_s_m, chen_label, gated_label, ...
        chen_color, gated_color, 'b_d [N s/m]', ...
        'Applied damping by region (did the arms differ here?)', '%.0f');

    plot_region_bars(nexttile, chen_metrics.fine_time_s, ...
        gated_metrics.fine_time_s, chen_metrics.gross_time_s, ...
        gated_metrics.gross_time_s, chen_label, gated_label, chen_color, ...
        gated_color, 'Time [s]', 'Moving time by region', '%.1f');

    sgtitle('Per-region comparison (human Fine/Gross labels)');
end

savefig(figure_handles, write_combined_fig);

%% ================================================================
% Summary
% =================================================================

fprintf('\n--- Chen vs. gated performance comparison ---\n');
fprintf('Chen arm:  %d trial(s)\n', numel(read_chen_robot_csv_files));
for trial_idx = 1:numel(read_chen_robot_csv_files)
    fprintf('  %s\n', read_chen_robot_csv_files{trial_idx});
end
fprintf('Gated arm: %d trial(s)\n', numel(read_gated_robot_csv_files));
for trial_idx = 1:numel(read_gated_robot_csv_files)
    fprintf('  %s\n', read_gated_robot_csv_files{trial_idx});
end

fprintf('\nReference registration check (shape_offset_xy_m = [%.4f, %.4f]):\n', ...
    shape_offset_xy_m(1), shape_offset_xy_m(2));
print_registration_check('Chen', chen_metrics);
print_registration_check('Gated', gated_metrics);

fprintf('\n%-28s %14s %14s %10s\n', 'Metric', char(chen_label), ...
    char(gated_label), 'Delta');
print_metric_row('Tracking error [mm]', chen_metrics.mean_error_mm, ...
    gated_metrics.mean_error_mm, '%14.3f');
print_metric_row('Mean force [N]', chen_metrics.mean_force_n, ...
    gated_metrics.mean_force_n, '%14.4f');
print_metric_row('DSJ [-]', chen_metrics.dsj, gated_metrics.dsj, '%14.4g');
print_metric_row('Path length [m]', chen_metrics.path_length_m, ...
    gated_metrics.path_length_m, '%14.4f');

if have_regions
    fprintf(['\nBy human-labelled region (moving samples only; DSJ is ' ...
        'whole-trial by nature):\n']);
    fprintf('%-22s %10s %10s %10s   %10s %10s %10s\n', '', ...
        'Fine Chen', 'Fine Gate', 'Fine d', ...
        'Gross Chen', 'Gross Gate', 'Gross d');
    print_region_row('Tracking error [mm]', chen_metrics.fine_error_mm, ...
        gated_metrics.fine_error_mm, chen_metrics.gross_error_mm, ...
        gated_metrics.gross_error_mm, '%10.3f');
    print_region_row('Mean force [N]', chen_metrics.fine_force_n, ...
        gated_metrics.fine_force_n, chen_metrics.gross_force_n, ...
        gated_metrics.gross_force_n, '%10.3f');
    print_region_row('Applied damping [Ns/m]', ...
        chen_metrics.fine_damping_n_s_m, gated_metrics.fine_damping_n_s_m, ...
        chen_metrics.gross_damping_n_s_m, ...
        gated_metrics.gross_damping_n_s_m, '%10.1f');
    print_region_row('Moving time [s]', chen_metrics.fine_time_s, ...
        gated_metrics.fine_time_s, chen_metrics.gross_time_s, ...
        gated_metrics.gross_time_s, '%10.2f');
end

fprintf('\nControl loop timing:\n');
print_timing_row('Chen', chen_metrics);
print_timing_row('Gated', gated_metrics);
fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================

function metrics = compute_arm_metrics(csv_files, reference_xy_m, ...
        moving_speed_threshold_m_s, jerk_outlier_window_s, ...
        jerk_smoothing_window_s, control_period_nominal_s, ...
        control_period_overrun_factor, registration_check_max_samples)
%COMPUTE_ARM_METRICS Paper Eq. (15)/(16)/(17), loop timing, and the same
% metrics split by the shape's own fine/gross regions, per trial.
%
% Input:
%   csv_files - cell array of controller CSV paths, one per trial.
%   reference_xy_m - shape reference centre line, already offset into
%       the CoppeliaSim frame.
%
% The fine/gross split comes from each CSV's own "label" column, i.e.
% the HUMAN labels from manual_label_chen_var_adm_2d_by_live_replay.py
% -- that is the operative definition of fine/gross in this project and
% the one the classifier was trained on. Rows labelled "None" belong to
% neither region and are excluded from both. metrics.has_labels reports
% whether every trial in this arm actually had the column.
%
% Output: struct of 1-by-trial_count arrays -- mean_error_mm,
%   mean_force_n, dsj, path_length_m, median_period_ms,
%   max_period_ms, overrun_fraction, reference_shift_mm, and the
%   per-region fine_*/gross_* variants of error, force, damping and
%   moving time -- plus period_ms, a 1-by-trial_count cell of each
%   trial's raw per-tick periods [ms].
%
% DSJ is deliberately NOT split by region: it is a whole-trajectory
% integral scaled by duration^5, so a value computed over interleaved
% fragments of a path would not be comparable to anything.

    trial_count = numel(csv_files);
    metrics.mean_error_mm = zeros(1, trial_count);
    metrics.mean_force_n = zeros(1, trial_count);
    metrics.dsj = zeros(1, trial_count);
    metrics.path_length_m = zeros(1, trial_count);
    metrics.median_period_ms = zeros(1, trial_count);
    metrics.max_period_ms = zeros(1, trial_count);
    metrics.overrun_fraction = zeros(1, trial_count);
    metrics.period_ms = cell(1, trial_count);
    metrics.reference_shift_mm = zeros(1, trial_count);
    metrics.fine_error_mm = zeros(1, trial_count);
    metrics.gross_error_mm = zeros(1, trial_count);
    metrics.fine_force_n = zeros(1, trial_count);
    metrics.gross_force_n = zeros(1, trial_count);
    metrics.fine_damping_n_s_m = zeros(1, trial_count);
    metrics.gross_damping_n_s_m = zeros(1, trial_count);
    metrics.fine_time_s = zeros(1, trial_count);
    metrics.gross_time_s = zeros(1, trial_count);
    metrics.has_labels = true;

    for trial_idx = 1:trial_count
        robot_table = readtable(csv_files{trial_idx}, ...
            'Delimiter', ',', 'ReadVariableNames', true, ...
            'VariableNamingRule', 'preserve', 'TextType', 'string');

        time_s = robot_table.time_s;
        coppeliasim_x_m = robot_table.coppeliasim_x_m;
        coppeliasim_y_m = robot_table.coppeliasim_y_m;
        force_magnitude_n = hypot( ...
            robot_table.Fx_filt_N, robot_table.Fy_filt_N);
        speed_actual_m_s = hypot( ...
            robot_table.tcp_vx_actual_m_s, robot_table.tcp_vy_actual_m_s);
        damping_applied_n_s_m = robot_table.bd_applied_Ns_m;
        moving_mask = speed_actual_m_s > moving_speed_threshold_m_s;

        tracking_error_m = distance_to_reference_path( ...
            coppeliasim_x_m, coppeliasim_y_m, reference_xy_m);
        metrics.mean_error_mm(trial_idx) = ...
            1e3 * mean(tracking_error_m(moving_mask));
        metrics.mean_force_n(trial_idx) = ...
            mean(force_magnitude_n(moving_mask));

        % Regime from the human label column. "Fine" and "Gross" are
        % tested separately rather than as each other's complement, so
        % "None"-labelled rows (transitions) fall into neither.
        has_label_column = ismember( ...
            'label', robot_table.Properties.VariableNames);
        metrics.has_labels = metrics.has_labels && has_label_column;
        if has_label_column
            sample_is_fine = robot_table.label == "Fine";
            sample_is_gross = robot_table.label == "Gross";
        else
            sample_is_fine = false(height(robot_table), 1);
            sample_is_gross = false(height(robot_table), 1);
        end
        sample_period_s = median(diff(time_s));

        fine_mask = moving_mask & sample_is_fine;
        gross_mask = moving_mask & sample_is_gross;
        metrics.fine_error_mm(trial_idx) = ...
            1e3 * mean(tracking_error_m(fine_mask));
        metrics.gross_error_mm(trial_idx) = ...
            1e3 * mean(tracking_error_m(gross_mask));
        metrics.fine_force_n(trial_idx) = mean(force_magnitude_n(fine_mask));
        metrics.gross_force_n(trial_idx) = mean(force_magnitude_n(gross_mask));
        metrics.fine_damping_n_s_m(trial_idx) = ...
            mean(damping_applied_n_s_m(fine_mask));
        metrics.gross_damping_n_s_m(trial_idx) = ...
            mean(damping_applied_n_s_m(gross_mask));
        metrics.fine_time_s(trial_idx) = sum(fine_mask) * sample_period_s;
        metrics.gross_time_s(trial_idx) = sum(gross_mask) * sample_period_s;

        % Same moving_mask-restricted samples as the headline metric
        % above, so this is a fair "would a better offset help" check
        % rather than comparing means over two different sample sets.
        % Subsampled because this re-scores the path 242 times.
        moving_indices = find(moving_mask);
        check_stride = max(1, ceil( ...
            numel(moving_indices) / registration_check_max_samples));
        check_indices = moving_indices(1:check_stride:end);
        [~, best_error_m] = estimate_reference_shift( ...
            coppeliasim_x_m(check_indices), coppeliasim_y_m(check_indices), ...
            reference_xy_m, 5e-3);
        metrics.reference_shift_mm(trial_idx) = ...
            1e3 * (mean(tracking_error_m(check_indices)) - best_error_m);

        jerk = compute_jerk_series(time_s, coppeliasim_x_m, ...
            coppeliasim_y_m, jerk_outlier_window_s, jerk_smoothing_window_s);
        metrics.dsj(trial_idx) = compute_dsj( ...
            jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m);
        metrics.path_length_m(trial_idx) = jerk.path_length_m;

        period_s = diff(time_s);
        metrics.period_ms{trial_idx} = 1e3 * period_s;
        metrics.median_period_ms(trial_idx) = 1e3 * median(period_s);
        metrics.max_period_ms(trial_idx) = 1e3 * max(period_s);
        metrics.overrun_fraction(trial_idx) = mean(period_s > ...
            control_period_overrun_factor * control_period_nominal_s);
    end
end


function [best_shift_xy_m, best_error_m] = estimate_reference_shift( ...
        path_x_m, path_y_m, reference_xy_m, search_spacing_m)
%ESTIMATE_REFERENCE_SHIFT Extra translation that would minimize error.
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m. Diagnostic only,
% never applied to the reported accuracy: fitting the reference to the
% data would bias the metric.

    best_shift_xy_m = [0, 0];
    best_error_m = mean(distance_to_reference_path( ...
        path_x_m, path_y_m, reference_xy_m));

    search_radius_m = 0.02;
    for refine_step = 1:2
        step_m = search_radius_m / 5;
        centre_shift_m = best_shift_xy_m;
        for shift_x_m = centre_shift_m(1) + (-5:5) * step_m
            for shift_y_m = centre_shift_m(2) + (-5:5) * step_m
                shifted_reference_m = reference_xy_m + ...
                    [shift_x_m, shift_y_m];
                trial_error_m = mean(distance_to_reference_path( ...
                    path_x_m, path_y_m, shifted_reference_m));
                if trial_error_m < best_error_m
                    best_error_m = trial_error_m;
                    best_shift_xy_m = [shift_x_m, shift_y_m];
                end
            end
        end
        search_radius_m = max(step_m, search_spacing_m * 10);
    end
end


function jerk = compute_jerk_series( ...
        raw_time_s, raw_x_m, raw_y_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m.

    valid_rows = isfinite(raw_time_s) & isfinite(raw_x_m) & ...
        isfinite(raw_y_m);
    clean_time_s = raw_time_s(valid_rows);
    clean_x_m = raw_x_m(valid_rows);
    clean_y_m = raw_y_m(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_x_m = clean_x_m(sort_order);
    clean_y_m = clean_y_m(sort_order);

    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_x_m = clean_x_m(unique_indices);
    clean_y_m = clean_y_m(unique_indices);

    sample_intervals_s = diff(clean_time_s);
    median_interval_s = median(sample_intervals_s(sample_intervals_s > 0));

    uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';
    uniform_x_m = interp1(clean_time_s, clean_x_m, uniform_time_s, 'linear');
    uniform_y_m = interp1(clean_time_s, clean_y_m, uniform_time_s, 'linear');

    outlier_samples = window_to_odd_sample_count( ...
        outlier_window_s, median_interval_s, numel(uniform_time_s));
    smoothing_samples = 3;
    %smoothing_samples = window_to_odd_sample_count(smoothing_window_s, median_interval_s, numel(uniform_time_s));

    if outlier_samples > 1
        uniform_x_m = movmedian(uniform_x_m, outlier_samples, ...
            'Endpoints', 'shrink');
        uniform_y_m = movmedian(uniform_y_m, outlier_samples, ...
            'Endpoints', 'shrink');
    end

    filtered_x_m = movmean(uniform_x_m, smoothing_samples, ...
        'Endpoints', 'shrink');
    filtered_y_m = movmean(uniform_y_m, smoothing_samples, ...
        'Endpoints', 'shrink');

    velocity_x_m_s = movmean(gradient(filtered_x_m, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');
    velocity_y_m_s = movmean(gradient(filtered_y_m, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');

    acceleration_x_m_s2 = movmean( ...
        gradient(velocity_x_m_s, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');
    acceleration_y_m_s2 = movmean( ...
        gradient(velocity_y_m_s, uniform_time_s), ...
        smoothing_samples, 'Endpoints', 'shrink');

    jerk_x_m_s3 = gradient(acceleration_x_m_s2, uniform_time_s);
    jerk_y_m_s3 = gradient(acceleration_y_m_s2, uniform_time_s);

    jerk.time_s = uniform_time_s;
    jerk.squared_jerk_m2_s6 = jerk_x_m_s3 .^ 2 + jerk_y_m_s3 .^ 2;
    jerk.path_length_m = sum(hypot( ...
        diff(filtered_x_m), diff(filtered_y_m)));
end
