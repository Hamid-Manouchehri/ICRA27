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


function plot_metric_bars(ax, chen_values, gated_values, chen_label, ...
        gated_label, chen_color, gated_color, y_label_text, title_text, ...
        value_format)
%PLOT_METRIC_BARS One metric, one tile: two colored bars, a std error
% bar (zero-length with only one trial per arm), and the mean value
% printed above each bar.

    means = [mean(chen_values), mean(gated_values)];
    stds = [0, 0];
    if numel(chen_values) > 1
        stds(1) = std(chen_values);
    end
    if numel(gated_values) > 1
        stds(2) = std(gated_values);
    end

    hold(ax, 'on');
    bar_handle = bar(ax, 1:2, means, 'FaceColor', 'flat');
    bar_handle.CData(1, :) = chen_color;
    bar_handle.CData(2, :) = gated_color;
    errorbar(ax, 1:2, means, stds, 'LineStyle', 'none', 'Color', 'k', ...
        'LineWidth', 1.2, 'CapSize', 8, 'HandleVisibility', 'off');
    for bar_idx = 1:2
        text(ax, bar_idx, means(bar_idx) + stds(bar_idx), ...
            sprintf(value_format, means(bar_idx)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 8);
    end
    set(ax, 'XTick', 1:2, 'XTickLabel', {char(chen_label), char(gated_label)});
    grid(ax, 'on');
    ylabel(ax, y_label_text);
    title(ax, title_text, 'FontSize', 9);
end


function plot_region_bars(ax, chen_fine, gated_fine, chen_gross, ...
        gated_gross, chen_label, gated_label, chen_color, gated_color, ...
        y_label_text, title_text, value_format)
%PLOT_REGION_BARS One metric as grouped bars: two regions x two arms,
% each bar the mean across that arm's trials.

    values = [mean(chen_fine), mean(gated_fine); ...
        mean(chen_gross), mean(gated_gross)];

    hold(ax, 'on');
    bar_handle = bar(ax, values);
    bar_handle(1).FaceColor = chen_color;
    bar_handle(2).FaceColor = gated_color;
    bar_handle(1).DisplayName = char(chen_label);
    bar_handle(2).DisplayName = char(gated_label);

    for group_idx = 1:2
        for series_idx = 1:2
            text(ax, bar_handle(series_idx).XEndPoints(group_idx), ...
                bar_handle(series_idx).YEndPoints(group_idx), ...
                sprintf(value_format, values(group_idx, series_idx)), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 7);
        end
    end

    set(ax, 'XTick', 1:2, 'XTickLabel', {'Fine region', 'Gross region'});
    grid(ax, 'on');
    ylabel(ax, y_label_text);
    title(ax, title_text, 'FontSize', 9);
    legend(ax, 'Location', 'best', 'FontSize', 7);
end


function print_region_row(name, chen_fine, gated_fine, chen_gross, ...
        gated_gross, number_format)
%PRINT_REGION_ROW One per-region row: both arms and the gated-minus-Chen
% difference, for each region.

    fine_chen = mean(chen_fine);
    fine_gated = mean(gated_fine);
    gross_chen = mean(chen_gross);
    gross_gated = mean(gated_gross);

    fprintf(['%-22s ' number_format ' ' number_format ' %+10.3g   ' ...
        number_format ' ' number_format ' %+10.3g\n'], ...
        name, fine_chen, fine_gated, fine_gated - fine_chen, ...
        gross_chen, gross_gated, gross_gated - gross_chen);
end


function print_registration_check(arm_name, metrics)
%PRINT_REGISTRATION_CHECK How much a per-trial best-fit shift would have
% reduced tracking error -- large values mean shape_offset_xy_m is
% probably wrong for that trial, not that the controller tracked poorly.

    for trial_idx = 1:numel(metrics.reference_shift_mm)
        fprintf(['  %-6s trial %d: best-fit shift would reduce error by ' ...
            '%.3f mm%s\n'], arm_name, trial_idx, ...
            metrics.reference_shift_mm(trial_idx), ...
            iif(metrics.reference_shift_mm(trial_idx) > 1.0, ...
            '  <-- check shape_offset_xy_m', ''));
    end
end


function result = iif(condition, true_value, false_value)
%IIF Inline conditional, since MATLAB has no ternary operator.

    if condition
        result = true_value;
    else
        result = false_value;
    end
end


function print_metric_row(name, chen_values, gated_values, number_format)
%PRINT_METRIC_ROW One summary-table row: arm means and their difference.

    chen_mean = mean(chen_values);
    gated_mean = mean(gated_values);
    fprintf(['%-28s ' number_format ' ' number_format ' %+10.3g\n'], ...
        name, chen_mean, gated_mean, gated_mean - chen_mean);
end


function print_timing_row(arm_name, metrics)
%PRINT_TIMING_ROW Control-loop period stats, pooled across an arm's trials.

    all_period_ms = vertcat(metrics.period_ms{:});
    fprintf(['  %-6s median %.3f ms, max %.3f ms, mean overrun fraction ' ...
        '%.3f %%\n'], arm_name, median(all_period_ms), ...
        max(all_period_ms), 100 * mean(metrics.overrun_fraction));
end


function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
        obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh.
%
% Output centre_xy_m is empty when fewer than two boundary loops are
% found. More than two loops are handled too -- a band that pinches to
% a thin neck (e.g. the pear shape's tip) can detach a small extra loop
% there, so the two loops with the LARGEST area are taken as outer/
% inner and any smaller ones are discarded with a warning, rather than
% failing the whole shape.
%
% Same fix applied to plot_raw_data_chen_var_adm_2d.m's own copy, so
% both stay identical (see that file's Method reuse note).

    outer_xy_m = [];
    inner_xy_m = [];
    centre_xy_m = [];

    file_id = fopen(obj_file, 'r');
    if file_id < 0
        return;
    end
    file_text = textscan(file_id, '%s', 'Delimiter', '\n', ...
        'WhiteSpace', '');
    fclose(file_id);
    file_lines = file_text{1};

    vertices = zeros(0, 3);
    faces = {};

    for line_idx = 1:numel(file_lines)
        this_line = strtrim(file_lines{line_idx});
        if startsWith(this_line, 'v ')
            values = sscanf(this_line(2:end), '%f');
            vertices(end + 1, 1:3) = values(1:3)'; %#ok<AGROW>
        elseif startsWith(this_line, 'f ')
            tokens = strsplit(strtrim(this_line(2:end)));
            face_indices = zeros(1, numel(tokens));
            for token_idx = 1:numel(tokens)
                slash_parts = strsplit(tokens{token_idx}, '/');
                face_indices(token_idx) = str2double(slash_parts{1});
            end
            faces{end + 1} = face_indices; %#ok<AGROW>
        end
    end

    if isempty(vertices) || isempty(faces)
        return;
    end

    z_bottom = min(vertices(:, 3));
    z_tolerance = 1e-9 + 1e-6 * (max(vertices(:, 3)) - z_bottom);

    edge_list = zeros(0, 2);
    for face_idx = 1:numel(faces)
        face_indices = faces{face_idx};
        if ~all(abs(vertices(face_indices, 3) - z_bottom) <= z_tolerance)
            continue;
        end
        for corner_idx = 1:numel(face_indices)
            next_corner = mod(corner_idx, numel(face_indices)) + 1;
            edge_list(end + 1, :) = sort([ ...
                face_indices(corner_idx), ...
                face_indices(next_corner)]); %#ok<AGROW>
        end
    end

    if isempty(edge_list)
        return;
    end

    [unique_edges, ~, edge_group] = unique(edge_list, 'rows');
    edge_use_count = accumarray(edge_group, 1);
    boundary_edges = unique_edges(edge_use_count == 1, :);

    if isempty(boundary_edges)
        return;
    end

    remaining_edges = boundary_edges;
    loops = {};
    while ~isempty(remaining_edges)
        loop_indices = remaining_edges(1, 1);
        current_vertex = remaining_edges(1, 2);
        remaining_edges(1, :) = [];

        while true
            loop_indices(end + 1) = current_vertex; %#ok<AGROW>
            next_row = find(any(remaining_edges == current_vertex, 2), 1);
            if isempty(next_row)
                break;
            end
            edge_pair = remaining_edges(next_row, :);
            remaining_edges(next_row, :) = [];
            if edge_pair(1) == current_vertex
                current_vertex = edge_pair(2);
            else
                current_vertex = edge_pair(1);
            end
            if current_vertex == loop_indices(1)
                break;
            end
        end

        if numel(loop_indices) > 3
            loops{end + 1} = vertices(loop_indices, 1:2); %#ok<AGROW>
        end
    end

    if numel(loops) < 2
        return;
    end

    loop_areas_m2 = zeros(1, numel(loops));
    for loop_idx = 1:numel(loops)
        loop_xy = loops{loop_idx};
        loop_areas_m2(loop_idx) = abs(0.5 * sum( ...
            loop_xy(:, 1) .* circshift(loop_xy(:, 2), -1) - ...
            circshift(loop_xy(:, 1), -1) .* loop_xy(:, 2)));
    end

    [sorted_areas_m2, sorted_order] = sort(loop_areas_m2, 'descend');
    outer_index = sorted_order(1);
    inner_index = sorted_order(2);

    if numel(loops) > 2
        warning(['%s: found %d boundary loop(s) beyond the outer/inner ' ...
            'pair (discarded area(s) %s mm^2, likely a meshing artifact ' ...
            'at a thin pinch point in the band) -- using only the two ' ...
            'largest loops by area.'], obj_file, numel(loops) - 2, ...
            mat2str(round(1e6 * sorted_areas_m2(3:end), 1)));
    end

    outer_xy_m = resample_closed_loop( ...
        loops{outer_index}, resample_spacing_m);
    inner_xy_m = resample_closed_loop( ...
        loops{inner_index}, resample_spacing_m);

    centre_xy_m = zeros(size(outer_xy_m));
    for sample_idx = 1:size(outer_xy_m, 1)
        distances_m = hypot( ...
            inner_xy_m(:, 1) - outer_xy_m(sample_idx, 1), ...
            inner_xy_m(:, 2) - outer_xy_m(sample_idx, 2));
        [~, nearest_index] = min(distances_m);
        centre_xy_m(sample_idx, :) = 0.5 * ( ...
            outer_xy_m(sample_idx, :) + inner_xy_m(nearest_index, :));
    end
end


function resampled_xy_m = resample_closed_loop(loop_xy_m, spacing_m)
%RESAMPLE_CLOSED_LOOP Resample a closed polygon at uniform arclength.
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m.

    closed_xy_m = [loop_xy_m; loop_xy_m(1, :)];
    segment_lengths_m = hypot( ...
        diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
    arclength_m = [0; cumsum(segment_lengths_m)];

    keep_mask = [true; segment_lengths_m > 0];
    arclength_m = arclength_m(keep_mask);
    closed_xy_m = closed_xy_m(keep_mask, :);

    total_length_m = arclength_m(end);
    sample_count = max(4, round(total_length_m / spacing_m));
    query_m = linspace(0, total_length_m, sample_count)';

    resampled_xy_m = [ ...
        interp1(arclength_m, closed_xy_m(:, 1), query_m, 'linear'), ...
        interp1(arclength_m, closed_xy_m(:, 2), query_m, 'linear')];
end


function distance_m = distance_to_reference_path( ...
        path_x_m, path_y_m, reference_xy_m)
%DISTANCE_TO_REFERENCE_PATH Nearest-reference-sample distance [m].
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m.

    distance_m = inf(numel(path_x_m), 1);

    for reference_idx = 1:size(reference_xy_m, 1)
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        distance_m = min(distance_m, candidate_m);
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


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample count.
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m.

    if window_s <= 0
        sample_count = 1;
        return;
    end

    sample_count = max(3, round(window_s / sample_interval_s));
    if mod(sample_count, 2) == 0
        sample_count = sample_count + 1;
    end

    largest_odd_count = maximum_sample_count;
    if mod(largest_odd_count, 2) == 0
        largest_odd_count = largest_odd_count - 1;
    end
    sample_count = min(sample_count, largest_odd_count);
end


function dsj = compute_dsj(time_s, squared_jerk_m2_s6, path_length_m)
%COMPUTE_DSJ Dimensionless squared jerk, paper Eq. (16).
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m.

    if path_length_m <= 0 || numel(time_s) < 2
        dsj = NaN;
        return;
    end

    movement_duration_s = time_s(end) - time_s(1);
    squared_jerk_integral = trapz(time_s, squared_jerk_m2_s6);

    dsj = squared_jerk_integral * movement_duration_s ^ 5 / ...
        path_length_m ^ 2;
end
