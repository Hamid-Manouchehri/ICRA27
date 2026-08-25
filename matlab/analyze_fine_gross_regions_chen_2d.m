%% Fine vs. gross path-region performance, Chen controller, damping sweep.
%
% For scripts/ur5e_chen_var_adm_2d.py on the fine_gross_shape task,
% subject s0, across five configured damping-range trials
% (d10-50, d50-100, d100-200, d200-300, d10-300): splits every recorded
% sample into a "fine" (narrow-band) or "gross" (wide-band) path region,
% detected from the shape mesh's own geometry, and compares performance
% and the controller's adaptive damping between regions and across the
% five configured ranges.
%
% Inputs:
%   data/raw_data/ur5e_chen_var_adm_2d_fine_gross_shape_d<range>_s0.csv
%     one CSV per damping-range trial, selected in User Settings.
%   data/shapes/fine_gross_shape.obj   band mesh; region detection and
%     the accuracy metric both come from it.
%
% Output:
%   Figure 1: TCP path per trial (2x3 tiles), each recorded sample
%   colored by its detected region (fine/gross), reference centerline
%   underneath for context and a dashed vertical line at the region
%   boundary so the split can be checked by eye.
%   Figure 2: grouped bar charts (2x2 tiles) -- mean tracking error,
%   mean force, mean speed, and time spent in region -- x-axis is the
%   configured damping range, bars grouped by region.
%   Figure 3: boxplot of the controller's own adaptive damping
%   (bd_applied_Ns_m), grouped by region within each configured range.
%   A numeric summary table is printed to the command window.
%   No file is written; this script only reads raw CSVs and one OBJ.
%
% Region detection method:
%   The two regions of fine_gross_shape.obj separate along the
%   CoppeliaSim x axis: the zig-zag (fine) teeth occupy the +x side of
%   the shape and the smooth rounded track (gross) occupies the rest,
%   so a single x threshold splits them. Every recorded TCP sample is
%   labelled directly from its own coppeliasim_x_m against
%   region_boundary_coppeliasim_x_m -- fine above it, gross at or below
%   it. The reference centerline is labelled by the same rule, so the
%   printed reference split and the per-sample split always agree.
%   An earlier revision inferred the split from the mesh's local band
%   width via two_means_threshold(), then gave each sample its nearest
%   centerline point's label. That mislabelled samples around the
%   transition, where the band width varies continuously instead of
%   separating into two clean clusters.
%
% Transition-band exclusion:
%   Samples whose coppeliasim_x_m falls inside
%   region_transition_x_range_m are dropped from every metric. Around
%   the boundary the operator is neither tracing the zig-zag nor the
%   smooth track but changing between the two, and a Chen-style
%   controller is still re-converging its curvature estimate there, so
%   those samples describe the transition rather than either region.
%   They are still drawn in Figure 1, in a neutral colour, so the
%   excluded stretch stays visible instead of silently vanishing.
%
% Trial trimming:
%   Every CSV starts recording before the operator touches the tool and
%   keeps recording after the trace is finished, so both ends carry a
%   stationary stretch that is not part of the experiment. Each trial is
%   cut to its motion span before anything is measured:
%   trim_to_motion_span() keeps only the samples from the start of the
%   first sustained above-threshold speed run to the end of the last
%   one. Untrimmed, those idle stretches would be counted as dwell time
%   in whichever region the TCP happened to be parked in, and their
%   near-zero force would be averaged into that region's force
%   statistics. A pause partway through the trace is deliberately kept:
%   it is part of the trial, unlike the stationary stretches outside it.
%
% Method reuse:
%   The shape-band OBJ reader is copied verbatim from
%   plot_raw_data_fixed_adm_2d.m. Self-contained here per this project's
%   convention that MATLAB scripts do not call one another.
%
% Metric conventions:
%   Tracking error, mean force, and mean speed are evaluated on moving
%   samples only (speed > moving_speed_threshold_m_s) within each
%   region, matching this project's Eq. (15)-(17) convention elsewhere.
%   Time spent in a region is a dwell-time measure, deliberately NOT
%   moving-gated (a pause inside a region is still time spent there),
%   so it is not directly comparable to the moving-gated metrics. It is
%   measured over the trimmed motion span only, so it no longer counts
%   the stationary lead-in and tail of the recording.
%   Smoothness (DSJ) is intentionally not computed per region: DSJ's
%   jerk chain needs a continuous time window, and splitting it at
%   region boundaries mid-trial would introduce filter edge artifacts
%   that overrepresent short regions -- a segmented DSJ was left out
%   rather than reported unreliably.
%
% Assumptions:
%   All five CSVs were written by ur5e_chen_var_adm_2d.py, so column
%   names match its write_csv_header() (coppeliasim_x_m/y_m,
%   Fx_filt_N/Fy_filt_N, bd_applied_Ns_m, tcp_vx_actual_m_s/
%   tcp_vy_actual_m_s). shape_offset_xy_m is the value already
%   validated for fine_gross_shape.obj in plot_raw_data_fixed_adm_2d.m /
%   plot_raw_data_chen_var_adm_2d.m.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');

%% User Settings
% TODO the five configured-damping-range trials, in display order.
trial_csv_files = { ...
    fullfile(raw_data_directory, ...
        'ur5e_chen_var_adm_2d_fine_gross_shape_d10-50_s0.csv'), ...
    fullfile(raw_data_directory, ...
        'ur5e_chen_var_adm_2d_fine_gross_shape_d50-100_s0.csv'), ...
    fullfile(raw_data_directory, ...
        'ur5e_chen_var_adm_2d_fine_gross_shape_d100-200_s0.csv'), ...
    fullfile(raw_data_directory, ...
        'ur5e_chen_var_adm_2d_fine_gross_shape_d200-300_s0.csv'), ...
    fullfile(raw_data_directory, ...
        'ur5e_chen_var_adm_2d_fine_gross_shape_d10-300_s0.csv')};
trial_labels = {'10-50', '50-100', '100-200', '200-300', '10-300'};

% TODO shape mesh traced by the operator; also the source of the region
% detection (band width along the path).
shape_obj_file = fullfile( ...
    script_directory, '..', 'data', 'shapes', 'fine_gross_shape.obj');

% TODO translation from the mesh CAD frame to the CoppeliaSim frame [m].
% Matches the value already validated for this shape in
% plot_raw_data_fixed_adm_2d.m / plot_raw_data_chen_var_adm_2d.m.
shape_offset_xy_m = [0.0, -0.645];

% TODO CoppeliaSim x coordinate separating the two path regions [m].
% Samples with coppeliasim_x_m above this are labelled fine (the
% zig-zag teeth on the +x side of fine_gross_shape.obj); samples at or
% below it are labelled gross. fine_gross_shape.obj spans roughly
% -0.098 to 0.088 m in x and its teeth begin near 0.05 m. Check the
% dashed boundary line drawn in Figure 1 against the traced path
% before trusting the split.
region_boundary_coppeliasim_x_m = 0.05;

% TODO CoppeliaSim x range excluded from every metric [m], as
% [lower, upper] with lower < upper. Covers the stretch where the
% operator is transitioning between the two regions rather than tracing
% either one. Widen it if the region bar charts still look contaminated
% by transition behaviour; set both entries equal to disable the
% exclusion.
region_transition_x_range_m = [0.03, 0.05];

% TODO spacing used to resample the reference path [m], positive.
reference_spacing_m = 5e-4;

% TODO speed separating movement from rest [m/s], used for error/force/
% speed (not for time-in-region, which is a dwell-time measure -- see
% header comment) and for finding each trial's motion span.
moving_speed_threshold_m_s = 0.005;

% TODO shortest continuous above-threshold speed run that counts as the
% experiment starting or ending [s], positive. Everything before the
% first such run and after the last one is discarded as the stationary
% lead-in and tail of the recording. Raise it if a noise spike in the
% stationary stretch is being taken as motion onset; lower it if a
% genuinely short first stroke is being cut off.
minimum_motion_run_s = 0.25;

% TODO colors for the two detected regions and marker size [points^2].
fine_color = [0.80, 0.20, 0.20];
gross_color = [0.20, 0.50, 0.80];
excluded_color = [0.65, 0.65, 0.65];
trajectory_marker_size = 6;

% TODO one color per damping-range trial, for the bar/boxplot panels.
trial_colors = [ ...
    0.40, 0.40, 0.40;   % 10-50
    0.30, 0.60, 0.20;   % 50-100
    0.90, 0.60, 0.00;   % 100-200
    0.85, 0.20, 0.10;   % 200-300
    0.00, 0.45, 0.74];  % 10-300 (full range, shown last as a reference)

trial_count = numel(trial_csv_files);

%% Region detection: CoppeliaSim x threshold
[outer_xy_m, inner_xy_m, centre_xy_m] = ...
    read_shape_band_obj(shape_obj_file, reference_spacing_m);

if isempty(centre_xy_m)
    error(['fine_gross_shape.obj did not resolve to a simple two-loop ' ...
        'band; region detection and accuracy both depend on it.']);
end

if region_transition_x_range_m(1) > region_transition_x_range_m(2)
    error(['region_transition_x_range_m must be [lower, upper] with ' ...
        'lower <= upper; a reversed range would silently exclude ' ...
        'nothing.']);
end

reference_xy_m = centre_xy_m + shape_offset_xy_m;
region_is_fine = reference_xy_m(:, 1) > region_boundary_coppeliasim_x_m;
reference_is_excluded = in_transition_band( ...
    reference_xy_m(:, 1), region_transition_x_range_m);

fprintf('\nRegion detection: boundary at CoppeliaSim x = %.3f m; ', ...
    region_boundary_coppeliasim_x_m);
fprintf('reference x spans %.3f to %.3f m ', ...
    min(reference_xy_m(:, 1)), max(reference_xy_m(:, 1)));
fprintf('(%d/%d reference points fine/gross, ', ...
    sum(region_is_fine & ~reference_is_excluded), ...
    sum(~region_is_fine & ~reference_is_excluded));
fprintf('%d excluded in x = %.3f to %.3f m).\n', ...
    sum(reference_is_excluded), region_transition_x_range_m(1), ...
    region_transition_x_range_m(2));

%% Load every trial and split by region
trials = cell(1, trial_count);
sample_is_fine = cell(1, trial_count);
sample_is_excluded = cell(1, trial_count);
trial_metrics = cell(1, trial_count);

fprintf('\nTrial trimming to motion span:\n');

for trial_idx = 1:trial_count
    trials{trial_idx} = load_chen_trial(trial_csv_files{trial_idx});
    [trials{trial_idx}, trim_info] = trim_to_motion_span( ...
        trials{trial_idx}, moving_speed_threshold_m_s, ...
        minimum_motion_run_s);

    fprintf(['  %-8s kept %6.2f s of %6.2f s (cut %.2f s lead-in, ' ...
        '%.2f s tail), samples %d-%d of %d.\n'], ...
        trial_labels{trial_idx}, trim_info.kept_duration_s, ...
        trim_info.original_duration_s, trim_info.lead_in_s, ...
        trim_info.tail_s, trim_info.first_index, ...
        trim_info.last_index, trim_info.original_sample_count);

    sample_is_fine{trial_idx} = ...
        trials{trial_idx}.xy_m(:, 1) > region_boundary_coppeliasim_x_m;
    sample_is_excluded{trial_idx} = in_transition_band( ...
        trials{trial_idx}.xy_m(:, 1), region_transition_x_range_m);

    trial_metrics{trial_idx} = compute_trial_region_metrics( ...
        trials{trial_idx}, reference_xy_m, sample_is_fine{trial_idx}, ...
        sample_is_excluded{trial_idx}, moving_speed_threshold_m_s);
end

%% ================================================================
% Figure 1: TCP path per trial, colored by detected region
% =================================================================

figure('Name', 'TCP path by fine/gross region', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for trial_idx = 1:trial_count
    nexttile;
    hold on;
    plot(reference_xy_m(:, 1), reference_xy_m(:, 2), '-', ...
        'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.75, ...
        'DisplayName', 'Reference centerline');
    xline(region_transition_x_range_m(1), '--', ...
        'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.0, ...
        'DisplayName', 'Excluded transition band');
    xline(region_transition_x_range_m(2), '--', ...
        'Color', [0.35, 0.35, 0.35], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');

    trial = trials{trial_idx};
    is_excluded = sample_is_excluded{trial_idx};
    is_fine = sample_is_fine{trial_idx} & ~is_excluded;
    is_gross = ~sample_is_fine{trial_idx} & ~is_excluded;
    scatter(trial.xy_m(is_fine, 1), trial.xy_m(is_fine, 2), ...
        trajectory_marker_size, fine_color, 'filled', ...
        'DisplayName', 'Fine region');
    scatter(trial.xy_m(is_gross, 1), trial.xy_m(is_gross, 2), ...
        trajectory_marker_size, gross_color, 'filled', ...
        'DisplayName', 'Gross region');
    scatter(trial.xy_m(is_excluded, 1), trial.xy_m(is_excluded, 2), ...
        trajectory_marker_size, excluded_color, 'filled', ...
        'DisplayName', 'Excluded (transition)');

    axis equal;
    grid on;
    title(sprintf('d = %s N s/m', trial_labels{trial_idx}));
    xlabel('CoppeliaSim x [m]');
    ylabel('CoppeliaSim y [m]');
    if trial_idx == 1
        legend('Location', 'best', 'FontSize', 7);
    end
end

sgtitle(sprintf(['TCP path colored by detected path region ' ...
    '(fine = CoppeliaSim x > %.3f m, gross = x <= %.3f m; ' ...
    'x = %.3f to %.3f m excluded from all metrics)'], ...
    region_boundary_coppeliasim_x_m, region_boundary_coppeliasim_x_m, ...
    region_transition_x_range_m(1), region_transition_x_range_m(2)));

%% ================================================================
% Figure 2: performance metrics, grouped by region
% =================================================================

mean_error_mm = zeros(trial_count, 2);
mean_force_n = zeros(trial_count, 2);
mean_speed_m_s = zeros(trial_count, 2);
time_in_region_s = zeros(trial_count, 2);

for trial_idx = 1:trial_count
    mean_error_mm(trial_idx, :) = trial_metrics{trial_idx}.mean_error_mm;
    mean_force_n(trial_idx, :) = trial_metrics{trial_idx}.mean_force_n;
    mean_speed_m_s(trial_idx, :) = trial_metrics{trial_idx}.mean_speed_m_s;
    time_in_region_s(trial_idx, :) = trial_metrics{trial_idx}.time_in_region_s;
end

region_legend = {'Fine', 'Gross'};

figure('Name', 'Performance by region and damping range', 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
bar(mean_error_mm, 'grouped');
set(gca, 'XTickLabel', trial_labels);
grid on;
xlabel('Configured damping range [N s/m]');
ylabel('Mean error [mm]');
title('Accuracy, Eq. (15) (moving samples only)');
legend(region_legend, 'Location', 'best');

% nexttile;
% bar(mean_force_n, 'grouped');
% set(gca, 'XTickLabel', trial_labels);
% grid on;
% xlabel('Configured damping range [N s/m]');
% ylabel('Mean |F| [N]');
% title('Effort, Eq. (17) (moving samples only)');
% legend(region_legend, 'Location', 'best');

nexttile;
bar(mean_force_n, 'grouped');
set(gca, 'XTickLabel', trial_labels);
grid on;
xlabel('Configured damping range [N s/m]');
ylabel('Mean |F| [N]');
title('Effort, Eq. (17) (moving samples only)');
legend(region_legend, 'Location', 'best');

nexttile;
bar(mean_speed_m_s, 'grouped');
set(gca, 'XTickLabel', trial_labels);
grid on;
xlabel('Configured damping range [N s/m]');
ylabel('Mean speed [m/s]');
title('Speed (moving samples only)');
legend(region_legend, 'Location', 'best');

nexttile;
bar(time_in_region_s, 'grouped');
set(gca, 'XTickLabel', trial_labels);
grid on;
xlabel('Configured damping range [N s/m]');
ylabel('Time in region [s]');
title('Dwell time (all samples, not moving-gated)');
legend(region_legend, 'Location', 'best');

sgtitle('Performance by path region and configured damping range');

%% ================================================================
% Figure 3: adaptive damping distribution, grouped by region x range
% =================================================================

damping_values_n_s_m = [];
damping_group_range = {};
damping_group_region = {};

for trial_idx = 1:trial_count
    for region_idx = 1:2
        values = trial_metrics{trial_idx}.damping_n_s_m{region_idx};
        damping_values_n_s_m = [damping_values_n_s_m; values]; %#ok<AGROW>
        damping_group_range = [damping_group_range; ...
            repmat(trial_labels(trial_idx), numel(values), 1)]; %#ok<AGROW>
        damping_group_region = [damping_group_region; ...
            repmat(region_legend(region_idx), numel(values), 1)]; %#ok<AGROW>
    end
end

figure('Name', 'Adaptive damping by region and damping range', 'Color', 'w');
boxplot(damping_values_n_s_m, {damping_group_range, damping_group_region}, ...
    'FactorGap', [10, 2], 'LabelOrientation', 'inline');
grid on;
ylabel('bd\_applied [N s/m]');
title(['Chen controller''s adaptive damping, grouped by configured ' ...
    'range and detected region']);

%% ================================================================
% Summary table (command window)
% =================================================================

fprintf('\n--- Chen, fine_gross_shape, s0, by region and damping range ---\n');
fprintf('%-24s', '');
for trial_idx = 1:trial_count
    fprintf('%22s', ['d = ', trial_labels{trial_idx}]);
end
fprintf('\n');

print_region_row('Mean error, fine [mm]', '%22.2f', mean_error_mm(:, 1));
print_region_row('Mean error, gross [mm]', '%22.2f', mean_error_mm(:, 2));
fprintf('\n');
print_region_row('Mean force, fine [N]', '%22.3f', mean_force_n(:, 1));
print_region_row('Mean force, gross [N]', '%22.3f', mean_force_n(:, 2));
fprintf('\n');
print_region_row('Mean speed, fine [m/s]', '%22.4f', mean_speed_m_s(:, 1));
print_region_row('Mean speed, gross [m/s]', '%22.4f', mean_speed_m_s(:, 2));
fprintf('\n');
print_region_row('Time in fine [s]', '%22.2f', time_in_region_s(:, 1));
print_region_row('Time in gross [s]', '%22.2f', time_in_region_s(:, 2));
fprintf(['  Dwell time is not moving-gated; the four metrics above it ' ...
    'are (moving samples only).\n']);
fprintf('\n');
for trial_idx = 1:trial_count
    for region_idx = 1:2
        values = trial_metrics{trial_idx}.damping_n_s_m{region_idx};
        fprintf(['  d=%-10s %-6s bd_applied: mean %6.1f, median %6.1f ' ...
            'N s/m (n=%d)\n'], trial_labels{trial_idx}, ...
            region_legend{region_idx}, mean(values), median(values), ...
            numel(values));
    end
end
fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================

function trial = load_chen_trial(csv_file)
%LOAD_CHEN_TRIAL Read one ur5e_chen_var_adm_2d.py CSV into a plain struct.

    data_table = readtable(csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');

    trial.time_s = get_numeric_column(data_table.time_s);
    trial.xy_m = [get_numeric_column(data_table.coppeliasim_x_m), ...
        get_numeric_column(data_table.coppeliasim_y_m)];
    trial.force_x_n = get_numeric_column(data_table.Fx_filt_N);
    trial.force_y_n = get_numeric_column(data_table.Fy_filt_N);
    trial.speed_m_s = hypot( ...
        get_numeric_column(data_table.tcp_vx_actual_m_s), ...
        get_numeric_column(data_table.tcp_vy_actual_m_s));
    trial.damping_n_s_m = get_numeric_column(data_table.bd_applied_Ns_m);
end


function is_excluded = in_transition_band(x_m, transition_x_range_m)
%IN_TRANSITION_BAND True for x values inside the excluded band.
%
% Closed on both ends, so the band edges themselves are excluded too.
% Setting both entries of transition_x_range_m equal reduces this to a
% single x value, which effectively disables the exclusion.

    is_excluded = x_m >= transition_x_range_m(1) & ...
        x_m <= transition_x_range_m(2);
end


function [trial, trim_info] = trim_to_motion_span( ...
        trial, moving_speed_threshold_m_s, minimum_motion_run_s)
%TRIM_TO_MOTION_SPAN Keep only the samples between motion onset and offset.
%
% Onset is the start of the first continuous run of above-threshold
% speed lasting at least minimum_motion_run_s; offset is the end of the
% last such run. Requiring a sustained run rather than a single sample
% keeps one noise spike in the stationary lead-in from defining the
% start of the experiment. Everything in between is kept, pauses
% included -- see the Trial trimming note in the header comment.
%
% time_s is re-zeroed at onset, so a trimmed trial reads as its own
% recording. Every per-sample field is sliced with the same index range,
% so sample_is_fine and the metrics stay aligned with it.

    moving_mask = trial.speed_m_s > moving_speed_threshold_m_s;

    mask_edges = diff([false; moving_mask(:); false]);
    run_first_index = find(mask_edges == 1);
    run_last_index = find(mask_edges == -1) - 1;
    run_duration_s = trial.time_s(run_last_index) - ...
        trial.time_s(run_first_index);

    is_sustained_run = run_duration_s >= minimum_motion_run_s;

    if ~any(is_sustained_run)
        error(['No motion run of at least %.3f s above %.4f m/s was ' ...
            'found: this trial is either idle or the thresholds are ' ...
            'wrong for this recording.'], ...
            minimum_motion_run_s, moving_speed_threshold_m_s);
    end

    sustained_first_index = run_first_index(is_sustained_run);
    sustained_last_index = run_last_index(is_sustained_run);
    first_index = sustained_first_index(1);
    last_index = sustained_last_index(end);

    trim_info.first_index = first_index;
    trim_info.last_index = last_index;
    trim_info.original_sample_count = numel(trial.time_s);
    trim_info.original_duration_s = trial.time_s(end) - trial.time_s(1);
    trim_info.lead_in_s = trial.time_s(first_index) - trial.time_s(1);
    trim_info.tail_s = trial.time_s(end) - trial.time_s(last_index);
    trim_info.kept_duration_s = ...
        trial.time_s(last_index) - trial.time_s(first_index);

    keep_range = first_index:last_index;
    trial.time_s = trial.time_s(keep_range) - trial.time_s(first_index);
    trial.xy_m = trial.xy_m(keep_range, :);
    trial.force_x_n = trial.force_x_n(keep_range);
    trial.force_y_n = trial.force_y_n(keep_range);
    trial.speed_m_s = trial.speed_m_s(keep_range);
    trial.damping_n_s_m = trial.damping_n_s_m(keep_range);
end


function trial_metrics = compute_trial_region_metrics( ...
        trial, reference_xy_m, sample_is_fine, sample_is_excluded, ...
        moving_speed_threshold_m_s)
%COMPUTE_TRIAL_REGION_METRICS Region-split performance for one trial.
%
% Column 1 of every output is the fine region, column 2 is gross.
% Tracking error, force, and speed use moving samples only within each
% region; time_in_region_s does not (dwell time, see header comment).
% Samples flagged by sample_is_excluded are removed from both, so the
% transition band contributes to neither region's statistics nor to
% either region's dwell time.

    tracking_error_m = distance_to_reference_path( ...
        trial.xy_m(:, 1), trial.xy_m(:, 2), reference_xy_m);
    force_magnitude_n = hypot(trial.force_x_n, trial.force_y_n);
    moving_mask = trial.speed_m_s > moving_speed_threshold_m_s;

    sample_dt_s = [diff(trial.time_s); median(diff(trial.time_s))];

    trial_metrics.mean_error_mm = zeros(1, 2);
    trial_metrics.mean_force_n = zeros(1, 2);
    trial_metrics.mean_speed_m_s = zeros(1, 2);
    trial_metrics.time_in_region_s = zeros(1, 2);
    trial_metrics.damping_n_s_m = cell(1, 2);

    for region_idx = 1:2
        if region_idx == 1
            region_mask = sample_is_fine;
        else
            region_mask = ~sample_is_fine;
        end
        region_mask = region_mask & ~sample_is_excluded;
        eval_mask = region_mask & moving_mask;

        trial_metrics.mean_error_mm(region_idx) = ...
            1e3 * mean(tracking_error_m(eval_mask));
        trial_metrics.mean_force_n(region_idx) = ...
            mean(force_magnitude_n(eval_mask));
        trial_metrics.mean_speed_m_s(region_idx) = ...
            mean(trial.speed_m_s(eval_mask));
        trial_metrics.time_in_region_s(region_idx) = ...
            sum(sample_dt_s(region_mask));
        trial_metrics.damping_n_s_m{region_idx} = ...
            trial.damping_n_s_m(eval_mask);
    end
end


function width_m = compute_band_width_along_centerline( ...
        outer_xy_m, inner_xy_m)
%COMPUTE_BAND_WIDTH_ALONG_CENTERLINE Local band width at each outer point.
%
% Same nearest-inner-sample search read_shape_band_obj() uses internally
% to build its centerline, so width_m(i) is defined at the same point as
% that function's centre_xy_m(i, :).

    sample_count = size(outer_xy_m, 1);
    width_m = zeros(sample_count, 1);

    for sample_idx = 1:sample_count
        distances_m = hypot( ...
            inner_xy_m(:, 1) - outer_xy_m(sample_idx, 1), ...
            inner_xy_m(:, 2) - outer_xy_m(sample_idx, 2));
        width_m(sample_idx) = min(distances_m);
    end
end


function [threshold_value, is_low_cluster] = two_means_threshold(values)
%TWO_MEANS_THRESHOLD Split a 1-D distribution into two clusters.
%
% Simple iterative 1-D k-means (k=2), initialized at the data extremes.
% Returns the midpoint between the two converged cluster centers as a
% threshold, and a logical mask marking membership in the lower-valued
% cluster. Used here instead of a hardcoded width value so this keeps
% working if the shape mesh ever changes -- see header comment.

    values = values(:);
    centers = [min(values), max(values)];

    for iteration = 1:50
        distance_to_low = abs(values - centers(1));
        distance_to_high = abs(values - centers(2));
        is_low_cluster = distance_to_low <= distance_to_high;

        if any(is_low_cluster)
            centers(1) = mean(values(is_low_cluster));
        end
        if any(~is_low_cluster)
            centers(2) = mean(values(~is_low_cluster));
        end
    end

    threshold_value = mean(centers);
end


function nearest_index = nearest_reference_index( ...
        path_x_m, path_y_m, reference_xy_m)
%NEAREST_REFERENCE_INDEX Index of the nearest reference sample, per point.
%
% Same computational pattern as distance_to_reference_path(), but keeps
% the arg-min index instead of the min distance, so each recorded sample
% can inherit a per-reference-point property (here, the region label).

    sample_count = numel(path_x_m);
    reference_count = size(reference_xy_m, 1);

    best_distance_m = inf(sample_count, 1);
    nearest_index = ones(sample_count, 1);

    for reference_idx = 1:reference_count
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        closer = candidate_m < best_distance_m;
        best_distance_m(closer) = candidate_m(closer);
        nearest_index(closer) = reference_idx;
    end
end


function print_region_row(row_label, value_format, values)
%PRINT_REGION_ROW Print one summary-table row across all trials.

    fprintf('%-24s', row_label);
    for value_idx = 1:numel(values)
        fprintf(value_format, values(value_idx));
    end
    fprintf('\n');
end


function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
        obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh. Copied
% verbatim from plot_raw_data_fixed_adm_2d.m.
%
% Input:
%   obj_file - Wavefront OBJ of a thin flat band, as exported by the CAD
%       tool used for this project's shapes.
%   resample_spacing_m - arclength spacing of the returned loops [m].
%
% Output:
%   outer_xy_m, inner_xy_m - resampled outer and inner edge loops, each
%       N-by-2 in the mesh CAD frame [m].
%   centre_xy_m - the reference path, midway between the two edges.
%       Empty when the mesh does not resolve into exactly two loops.
%
% Method:
%   The band is extruded in z, so its bottom cap carries the planar
%   outline. Cap triangles are collected, and the outline is the set of
%   edges belonging to exactly one triangle: interior triangulation
%   diagonals are shared by two and drop out. Those boundary edges chain
%   into closed loops; the loop enclosing the larger area is the outer
%   edge. The centre line takes, for each outer sample, the nearest
%   inner sample and averages the two, which is what an offset band
%   gives.

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

    if numel(loops) ~= 2
        return;
    end

    loop_areas = zeros(1, 2);
    for loop_idx = 1:2
        loop_xy = loops{loop_idx};
        loop_areas(loop_idx) = abs(0.5 * sum( ...
            loop_xy(:, 1) .* circshift(loop_xy(:, 2), -1) - ...
            circshift(loop_xy(:, 1), -1) .* loop_xy(:, 2)));
    end

    [~, outer_index] = max(loop_areas);
    inner_index = 3 - outer_index;

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
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.

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
%DISTANCE_TO_REFERENCE_PATH Nearest-reference-sample distance [m]. Copied
% verbatim from plot_raw_data_fixed_adm_2d.m.

    distance_m = inf(numel(path_x_m), 1);

    for reference_idx = 1:size(reference_xy_m, 1)
        candidate_m = hypot( ...
            path_x_m - reference_xy_m(reference_idx, 1), ...
            path_y_m - reference_xy_m(reference_idx, 2));
        distance_m = min(distance_m, candidate_m);
    end
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied verbatim from plot_raw_data_fixed_adm_2d.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end
