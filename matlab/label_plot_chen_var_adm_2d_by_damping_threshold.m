%% Label a Chen variable-admittance trial by thresholding applied damping.
%
% Unlike ur5e_fixed_adm_2d_labeled.py (a live, keyboard-driven "label"
% column recorded DURING a trial), this labels an ALREADY-RECORDED
% ur5e_chen_var_adm_2d.py trial AFTER THE FACT, purely from its own
% logged applied damping b_d. This matches the controller's own
% physical convention (scripts/ur5e_chen_var_adm_2d.py,
% compute_desired_damping): high damping is the STIFF/PRECISE regime
% (b_d -> damping_max_n_s_m when human force is low), low damping is
% the COMPLIANT/FAST regime (b_d -> damping_min_n_s_m when force is
% high) -- see also Figure 1's docstring in
% plot_raw_data_chen_var_adm_2d.m ("low b_d ... fast motion... high
% b_d ... precision").
%
% LABELED FROM THE MOVING-AVERAGE PROFILE, not the raw per-sample value:
% b_d is first smoothed by a damping_moving_average_window_s [s]
% centered moving average (see that setting below), and every step from
% there on -- thresholding, the sigmoid, fine_score, sample_weight --
% acts on THAT smoothed profile, not the noisy instantaneous b_d. This
% matches "Fine"/"Gross" as a sustained REGIME rather than a label that
% flips on a single noisy sample the rate-limited controller only
% briefly passed through. Setting damping_moving_average_window_s to 0
% degenerates the smoothed profile back to the raw signal exactly
% (movmean with a 1-sample window is the identity), recovering the
% original per-sample thresholding behaviour.
%
% AUTOMATIC, PER-TRIAL THRESHOLD (Otsu's method): damping_threshold_n_s_m
% is NOT a fixed constant -- how hard a given subject pushes (and so how
% high their damping actually climbs, per the controller's own Eq. 7)
% varies by person and by trial, so one hand-picked N s/m value doesn't
% transfer across subjects/trials. Instead the threshold is computed
% from THIS trial's own smoothed-damping histogram, choosing whichever
% value maximizes between-class variance (Otsu, 1979) -- i.e. the
% valley between two modes if the trial's damping is genuinely
% bimodal (alternating fast/compliant and slow/precise regimes), and
% something close to the mean if it isn't. See otsu_threshold() below;
% otsu_bin_count controls its histogram resolution. Check the printed
% sample_weight statistics for a given trial before trusting its
% Otsu-picked threshold -- a high ambiguous-sample fraction there means
% that trial's damping wasn't cleanly bimodal to begin with.
%
% otsu_min_class_fraction guards against a specific failure mode: a
% brief saturation/clamp event (damping pinned near its max for ~1s+)
% shows up as a small, extreme-mean clump in the histogram, and Otsu's
% between-class-variance objective will happily isolate that tiny clump
% as its own "class" instead of splitting the trial's real, larger
% bimodal structure -- since a huge (mean0-mean1)^2 wins even when one
% side's weight is tiny. Raising this floor above that clump's sample
% fraction forces Otsu to only consider splits with a reasonably
% balanced class size on both sides.
%
% SOFT labeling: rather than a bare step function at
% damping_threshold_n_s_m, the smoothed damping profile is passed
% through a sigmoid CENTERED on that threshold, giving a continuous
% "fine_score" in (0, 1) -- smoothed b_d far above the threshold scores
% near 1 (confidently Fine), far below scores near 0 (confidently
% Gross), and smoothed b_d close to the threshold scores near 0.5
% (genuinely ambiguous, rather than being forced into one bucket by an
% arbitrary cliff). The HARD "label" column used throughout this
% project ("Fine"/"Gross") is recovered exactly as fine_score > 0.5,
% which is mathematically identical to a bare "smoothed b_d >
% damping_threshold_n_s_m" rule (sigmoid(0) = 0.5) -- so every
% downstream script that only reads "label" sees no change at all. What
% is new is a "sample_weight" column, 2*|fine_score - 0.5| in [0, 1]:
% 0 at the threshold itself (fully ambiguous, should barely count
% during training), rising to 1 far from it (fully confident). Passing
% this as scikit-learn's sample_weight when fitting a classifier lets
% ambiguous near-threshold windows contribute less to training than
% confidently-labeled ones, instead of being forced to count exactly as
% much as a b_d = 200 sample the same as the "Fine" class.
%
% Inputs:
%   data/raw_data/ur5e_chen_var_adm_2d_<trial>.csv   controller log
%   data/raw_data/tobii_chen_var_adm_2d_<trial>.csv  matching gaze log
%   data/calibrations/<file>.json  Tobii UCS -> CoppeliaSim calibration,
%     for the gaze overlay on the TCP-path plot only (labeling itself
%     needs no calibration).
%   All selected in User Settings. The gaze CSV's time_s shares the
%   same trial-clock origin as the robot CSV's time_s (both written
%   from the same trial clock -- see plot_raw_data_chen_var_adm_2d.m's
%   own header), so labels are transferred onto gaze samples by nearest
%   robot-sample time, with no calibration/UCS mapping needed for that
%   part.
%
% Output:
%   data/processed_data/ur5e_chen_var_adm_2d_damping_labeled_<trial>.csv
%     -- the robot table, unchanged, plus three new columns: "label"
%     ("Fine"/"Gross", one per robot sample), "fine_score" (the
%     continuous sigmoid confidence in (0, 1)), and "sample_weight"
%     (2*|fine_score - 0.5|, for weighted classifier training).
%   data/processed_data/tobii_chen_var_adm_2d_damping_labeled_<trial>.csv
%     -- the gaze table, unchanged, plus the same three columns,
%     nearest-in-time from the robot samples.
%   Figure 1: applied damping over time colored by label, with a
%   tunable moving-average overlay (top-left, spanning 2 columns; the
%   window is damping_moving_average_window_s [s], skipped when <= 0),
%   the sigmoid fine_score curve against damping (top-right), TCP path
%   colored by label with a gaze overlay (bottom-left, skipped with a
%   warning if the calibration JSON is unavailable), Fine/Gross sample
%   counts (bottom-middle), and the sample_weight distribution
%   (bottom-right).
%   Figure 2: instantaneous (not smoothed) velocity/force/damping
%   relationships, colored by label -- speed vs. force, damping vs.
%   speed, damping vs. force.
%   A numeric summary is printed to the command window.
%
% Assumptions:
%   Both CSVs were written by scripts/ur5e_chen_var_adm_2d.py (or its
%   matching Tobii logger for the gaze one), so column names match
%   that script's write_csv_header().

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO controller log to label, written by scripts/ur5e_chen_var_adm_2d.py.
read_robot_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'old_data', ...
    'ur5e_chen_var_adm_2d_fine_gross_shape_counter_clockwise_s0.csv');

% TODO matching gaze log from the same trial (same time_s origin).
read_gaze_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'old_data', ...
    'tobii_chen_var_adm_2d_fine_gross_shape_counter_clockwise_s0.csv');

% TODO calibration JSON mapping Tobii UCS to the CoppeliaSim frame for
% this subject/shape, used only to overlay gaze on the TCP path plot
% (does not affect labeling, which comes entirely from the robot log).
read_calibration_json_file = fullfile( ...
    script_directory, '..', 'data', 'calibrations', ...
    'tobii_coppelia_reference_calibration_narrow_ellipse_shape_counter_clockwise_s17.json');

% TODO output directory and file names. Written under processed_data/,
% not raw_data/, since this is a DERIVED labeling of an already-recorded
% trial, not the controller script's own output (see this project's
% CLAUDE.md: raw_data/ is only ever written by the controller script
% that produced it).
write_output_directory = fullfile(script_directory, '..', 'data', 'processed_data');
write_robot_labeled_csv_file = fullfile(write_output_directory, ...
    'ur5e_chen_var_adm_2d_damping_labeled_narrow_pear_shape_counter_clockwise_s17.csv');
write_gaze_labeled_csv_file = fullfile(write_output_directory, ...
    'tobii_chen_var_adm_2d_damping_labeled_narrow_pear_shape_counter_clockwise_s17.csv');

% TODO histogram resolution for the automatic Otsu threshold [count],
% > 0 (see the module docstring's "AUTOMATIC, PER-TRIAL THRESHOLD"
% section -- damping_threshold_n_s_m is COMPUTED below, not a manual
% constant). Too few bins under-resolves where the valley between two
% modes actually sits; too many makes the histogram noisy enough that
% Otsu can lock onto a spurious single-sample spike instead of the real
% valley. 100 bins over this controller's [10, 200] N s/m band is
% ~1.9 N s/m/bin, fine-grained without being noise-sensitive.
otsu_bin_count = 50;

% TODO min sample fraction [0-0.5] Otsu must put on each side of the
% split -- blocks a rare saturation clump from hijacking the threshold.
otsu_min_class_fraction = 0.10;

% TODO physical screen motion / robot motion [unitless], must be > 0,
% and the Tobii UCS axis signs in CoppeliaSim, each -1 or +1. Must match
% the values used elsewhere in this project.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO true excludes gaze estimates outside the normalized [0, 1]
% display area; true requires both eyes valid for a gaze sample to
% count (false falls back to averaging whichever eye/eyes are valid).
% Matches this project's other calibrated-mapping scripts' defaults.
require_gaze_on_display = true;
require_both_eyes_valid = true;

% TODO which logged damping column to threshold [N s/m]. bd_applied_Ns_m
% (the rate-limited value actually commanded, matching what Figure 1 of
% plot_raw_data_chen_var_adm_2d.m colors the TCP path by) is the default;
% bd_desired_Ns_m is the un-rate-limited alternative, in case the applied
% column's rate limiting is not wanted here.
damping_column_name = "bd_applied_Ns_m";

% TODO sigmoid width [N s/m], > 0: how gradually fine_score transitions
% from 0 to 1 around damping_threshold_n_s_m. At +-1 scale from the
% threshold, fine_score is about 0.27/0.73; at +-2 scale, about
% 0.12/0.88 -- so this sets how wide the genuinely-ambiguous "gray
% zone" is. Default is roughly a tenth of the controller's full
% [damping_min_n_s_m, damping_max_n_s_m] = [10, 200] span.
sigmoid_scale_n_s_m = 20.0;

% TODO label text for each side of the threshold. STRICTLY greater than
% the threshold (fine_score > 0.5) is fine_label_text; everything else
% (including exactly at the threshold) is gross_label_text.
fine_label_text = "Fine";
gross_label_text = "Gross";

% TODO marker sizes [points^2] and colors for the Fine/Gross plots.
fine_marker_size = 6;
fine_color = [0.00, 0.45, 0.74];

gross_marker_size = 6;
gross_color = [0.85, 0.33, 0.10];

% TODO gaze overlay on the TCP path [points^2], color, and marker
% transparency -- gaze samples are far denser than the robot path in
% that region, so a lower alpha keeps the underlying path readable.
gaze_marker_size = 4;
gaze_color = 'g';
gaze_marker_alpha = 1.0;

% TODO marker transparency [0, 1] for the velocity/force/damping
% relationship scatter plots -- these plot every raw robot sample
% (tens of thousands per trial) against each other, so a low alpha is
% what actually makes point DENSITY visible instead of one solid blob.
relationship_marker_alpha = 0.12;

% TODO tunable moving-average window [s], >= 0, THIS DRIVES LABELING
% (see the module docstring) -- not just a plot cosmetic. Converted to
% an odd sample count using this trial's own median sample interval
% (same "time window, not sample count" pattern as plot_raw_data_chen_
% var_adm_2d.m's jerk_smoothing_window_s), so it means the same span
% regardless of the controller's actual sample rate. 0 degenerates to
% labeling from the raw per-sample b_d directly (no smoothing), and
% also skips drawing the separate moving-average line in the top-left
% plot (which would otherwise just retrace the raw scatter exactly).
damping_moving_average_window_s = 1.0;
damping_moving_average_color = 'g';
damping_moving_average_line_width = 1.6;

%% Load the controller log and label it by thresholding damping
robot_table = readtable(read_robot_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

if ~ismember(damping_column_name, string(robot_table.Properties.VariableNames))
    error(['damping_column_name "%s" is not a column of %s. Available ' ...
        'damping-like columns: bd_desired_Ns_m, bd_applied_Ns_m, ' ...
        'B11_Ns_m, B22_Ns_m.'], damping_column_name, read_robot_csv_file);
end

robot_time_s = robot_table.time_s;
damping_n_s_m = robot_table.(damping_column_name);

% Raw per-sample force magnitude and TCP speed, for the velocity/force/
% damping relationship figure below only (not used by labeling itself).
force_magnitude_n = hypot(robot_table.Fx_filt_N, robot_table.Fy_filt_N);
speed_m_s = hypot( ...
    robot_table.tcp_vx_actual_m_s, robot_table.tcp_vy_actual_m_s);

% Moving average of damping over time -- THIS, not the raw per-sample
% value, is what labeling below acts on (see the module docstring's
% "LABELED FROM THE MOVING-AVERAGE PROFILE" section). Window converted
% from seconds to an odd sample count using this trial's own median
% sample interval, so damping_moving_average_window_s means the same
% span regardless of the actual control-loop rate. Always computed
% (movmean with the 1-sample window damping_moving_average_window_s<=0
% produces is the identity, so this degenerates to the raw signal
% exactly rather than needing a separate skip branch).
robot_sample_interval_s = median(diff(robot_time_s));
damping_moving_average_sample_count = window_to_odd_sample_count( ...
    damping_moving_average_window_s, robot_sample_interval_s, ...
    numel(damping_n_s_m));
damping_moving_average_n_s_m = movmean( ...
    damping_n_s_m, damping_moving_average_sample_count);

% Automatic, per-trial threshold (Otsu's method) on the SMOOTHED
% damping profile -- see the module docstring's "AUTOMATIC, PER-TRIAL
% THRESHOLD" section for the reasoning.
damping_threshold_n_s_m = otsu_threshold( ...
    damping_moving_average_n_s_m, otsu_bin_count, otsu_min_class_fraction);

% Soft label: sigmoid centered on damping_threshold_n_s_m, applied to
% the SMOOTHED damping profile above -- see the module docstring's
% "SOFT labeling" section for the reasoning.
robot_fine_score = 1 ./ (1 + exp( ...
    -(damping_moving_average_n_s_m - damping_threshold_n_s_m) / ...
    sigmoid_scale_n_s_m));
robot_sample_weight = 2 * abs(robot_fine_score - 0.5);

% Hard label, recovered from the soft score at its 0.5 midpoint --
% identical to a bare "damping_moving_average_n_s_m >
% damping_threshold_n_s_m" rule (sigmoid(0) = 0.5), so every script
% that only reads "label" sees no change.
robot_is_fine = robot_fine_score > 0.5;
robot_label = repmat(gross_label_text, height(robot_table), 1);
robot_label(robot_is_fine) = fine_label_text;
robot_table.label = robot_label;
robot_table.fine_score = robot_fine_score;
robot_table.sample_weight = robot_sample_weight;

%% Load the gaze log and transfer label/fine_score/sample_weight by
% nearest robot sample time (gaze has no damping of its own).
gaze_table = readtable(read_gaze_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

resampled = resample_nearest_multi(robot_time_s, ...
    {robot_label, robot_fine_score, robot_sample_weight}, gaze_table.time_s);
gaze_table.label = resampled{1};
gaze_table.fine_score = resampled{2};
gaze_table.sample_weight = resampled{3};

%% Map gaze onto the CoppeliaSim frame, for the TCP-path overlay only
% (does not affect labeling above, and is skipped -- with a warning --
% if the calibration JSON is unavailable).
gaze_overlay_available = false;
if isfile(read_calibration_json_file)
    calibration = jsondecode(fileread(read_calibration_json_file));
    tobii_reference_xy_mm = double(calibration.manual_tobii_reference_xy_mm(:)');
    coppelia_reference_xy_m = double(calibration.manual_coppelia_reference_xy_m(:)');

    [gaze_xy_m, gaze_point_valid] = map_gaze_ucs_to_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        require_gaze_on_display, require_both_eyes_valid);
    gaze_overlay_available = true;
else
    warning(['Calibration JSON not found, so the gaze overlay on the ' ...
        'TCP path is skipped. Labeling above is unaffected.']);
end

%% Write both labeled CSVs
if ~isfolder(write_output_directory)
    mkdir(write_output_directory);
end
writetable(robot_table, write_robot_labeled_csv_file);
writetable(gaze_table, write_gaze_labeled_csv_file);

%% ================================================================
% Plot: damping over time, the soft-labeling sigmoid, TCP path, label
% counts, and the sample_weight distribution.
% =================================================================

figure('Name', 'Chen trial labeled by damping threshold', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- Top-left (spans 2 columns): applied damping over time ----
nexttile(1, [1, 2]);
hold on;
scatter(robot_time_s(robot_is_fine), damping_n_s_m(robot_is_fine), ...
    fine_marker_size, fine_color, 'filled', 'DisplayName', ...
    sprintf('%s (n=%d)', fine_label_text, nnz(robot_is_fine)));
scatter(robot_time_s(~robot_is_fine), damping_n_s_m(~robot_is_fine), ...
    gross_marker_size, gross_color, 'filled', 'DisplayName', ...
    sprintf('%s (n=%d)', gross_label_text, nnz(~robot_is_fine)));
if damping_moving_average_window_s > 0
    plot(robot_time_s, damping_moving_average_n_s_m, '-', ...
        'Color', damping_moving_average_color, ...
        'LineWidth', damping_moving_average_line_width, 'DisplayName', ...
        sprintf('Moving average (%.2f s window)', ...
        damping_moving_average_window_s));
end
yline(damping_threshold_n_s_m, 'k--', ...
    sprintf('Otsu threshold = %.1f N s/m', damping_threshold_n_s_m));
grid on;
xlabel('Time [s]');
ylabel('Damping [N s/m]');
title(sprintf('%s, colored by hard threshold label', damping_column_name));
legend('Location', 'best');

% ---- Top-right: the soft-labeling sigmoid, fine_score vs the SMOOTHED
% damping profile (not raw b_d -- fine_score is a deterministic
% function of the smoothed signal, so plotting it against raw b_d would
% no longer trace a clean curve). Every real sample plotted where it
% actually falls on the curve (not just an idealized line), so this
% also shows how much of the trial's own smoothed-damping distribution
% sits in the ambiguous middle versus the confident tails.
nexttile;
hold on;
scatter(damping_moving_average_n_s_m(robot_is_fine), ...
    robot_fine_score(robot_is_fine), fine_marker_size, fine_color, ...
    'filled', 'DisplayName', fine_label_text);
scatter(damping_moving_average_n_s_m(~robot_is_fine), ...
    robot_fine_score(~robot_is_fine), gross_marker_size, gross_color, ...
    'filled', 'DisplayName', gross_label_text);
xline(damping_threshold_n_s_m, 'k--', 'HandleVisibility', 'off');
yline(0.5, 'k:', 'HandleVisibility', 'off');
grid on;
ylim([0, 1]);
xlabel('Smoothed damping [N s/m]');
ylabel('fine\_score [-]');
title(sprintf('Soft label: sigmoid scale = %.1f N s/m', sigmoid_scale_n_s_m));
legend('Location', 'best');

% ---- Bottom-left: TCP path colored by label ----
nexttile;
hold on;
scatter(robot_table.coppeliasim_x_m(robot_is_fine), ...
    robot_table.coppeliasim_y_m(robot_is_fine), fine_marker_size, ...
    fine_color, 'filled', 'DisplayName', fine_label_text);
scatter(robot_table.coppeliasim_x_m(~robot_is_fine), ...
    robot_table.coppeliasim_y_m(~robot_is_fine), gross_marker_size, ...
    gross_color, 'filled', 'DisplayName', gross_label_text);
if gaze_overlay_available
    scatter(gaze_xy_m(gaze_point_valid, 1), gaze_xy_m(gaze_point_valid, 2), ...
        gaze_marker_size, gaze_color, 'filled', ...
        'MarkerFaceAlpha', gaze_marker_alpha, 'DisplayName', ...
        sprintf('Gaze (n=%d)', nnz(gaze_point_valid)));
end
axis equal;
grid on;
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title('TCP path, colored by threshold label, with gaze overlay');
legend('Location', 'best');

% ---- Bottom-middle: label counts ----
nexttile;
label_counts = [nnz(robot_is_fine), nnz(~robot_is_fine)];
bar_handle = bar(categorical({char(fine_label_text), char(gross_label_text)}), ...
    label_counts);
bar_handle.FaceColor = 'flat';
bar_handle.CData(1, :) = fine_color;
bar_handle.CData(2, :) = gross_color;
grid on;
ylabel('Robot sample count [-]');
title(sprintf('Label split: %.1f%% %s, %.1f%% %s', ...
    100 * label_counts(1) / sum(label_counts), fine_label_text, ...
    100 * label_counts(2) / sum(label_counts), gross_label_text));

% ---- Bottom-right: sample_weight (label confidence) distribution ----
% Low weight = near the ambiguous threshold, should barely count during
% weighted classifier training; high weight = confidently labeled.
nexttile;
histogram(robot_sample_weight, 30);
hold on;
xline(mean(robot_sample_weight), 'r--', ...
    sprintf('mean = %.2f', mean(robot_sample_weight)));
grid on;
xlim([0, 1]);
xlabel('sample\_weight [-]');
ylabel('Robot sample count [-]');
title('Label confidence distribution');

%% ================================================================
% Figure 2: velocity/force/damping relationships, colored by label.
% Every raw robot sample (not smoothed), since these are meant to show
% the instantaneous physical relationships the controller itself
% implements (Eq. 7: damping as a function of force; Eqs. 4/8: speed as
% a function of curvature/force chain) -- not the smoothed labeling
% profile.
% =================================================================

figure('Name', 'Velocity, force, and damping relationships', 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- Speed vs. force ----
nexttile;
hold on;
scatter(force_magnitude_n(robot_is_fine), speed_m_s(robot_is_fine), ...
    fine_marker_size, fine_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    fine_label_text);
scatter(force_magnitude_n(~robot_is_fine), speed_m_s(~robot_is_fine), ...
    gross_marker_size, gross_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    gross_label_text);
grid on;
xlabel('|F| [N]');
ylabel('Speed [m/s]');
title('Speed vs. force');
legend('Location', 'best');

% ---- Damping vs. speed ----
nexttile;
hold on;
scatter(speed_m_s(robot_is_fine), damping_n_s_m(robot_is_fine), ...
    fine_marker_size, fine_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    fine_label_text);
scatter(speed_m_s(~robot_is_fine), damping_n_s_m(~robot_is_fine), ...
    gross_marker_size, gross_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    gross_label_text);
grid on;
xlabel('Speed [m/s]');
ylabel('Damping [N s/m]');
title('Damping vs. speed');
legend('Location', 'best');

% ---- Damping vs. force ----
nexttile;
hold on;
scatter(force_magnitude_n(robot_is_fine), damping_n_s_m(robot_is_fine), ...
    fine_marker_size, fine_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    fine_label_text);
scatter(force_magnitude_n(~robot_is_fine), damping_n_s_m(~robot_is_fine), ...
    gross_marker_size, gross_color, 'filled', ...
    'MarkerFaceAlpha', relationship_marker_alpha, 'DisplayName', ...
    gross_label_text);
grid on;
xlabel('|F| [N]');
ylabel('Damping [N s/m]');
title('Damping vs. force');
legend('Location', 'best');

%% ================================================================
% Summary
% =================================================================

fprintf('\n--- Chen trial labeled by damping threshold (soft label) ---\n');
fprintf('Robot log:   %s\n', read_robot_csv_file);
fprintf('Gaze log:    %s\n', read_gaze_csv_file);
fprintf(['Thresholded column: %s, smoothed by a %.2f s moving average ' ...
    '(%d samples)\n'], damping_column_name, damping_moving_average_window_s, ...
    damping_moving_average_sample_count);
fprintf(['Threshold: %.2f N s/m, auto-picked by Otsu''s method for ' ...
    'THIS trial (%d-bin histogram) -- not a fixed constant, see the ' ...
    'module docstring\n'], damping_threshold_n_s_m, otsu_bin_count);
fprintf(['Sigmoid: centered at that threshold, scale %.2f N s/m -> ' ...
    'hard label "%s" where fine_score > 0.5, otherwise "%s"\n'], ...
    sigmoid_scale_n_s_m, fine_label_text, gross_label_text);
fprintf('Raw damping range in this trial: %.2f to %.2f N s/m (mean %.2f)\n', ...
    min(damping_n_s_m), max(damping_n_s_m), mean(damping_n_s_m));
fprintf(['Smoothed damping range (what labeling actually thresholds): ' ...
    '%.2f to %.2f N s/m (mean %.2f)\n'], min(damping_moving_average_n_s_m), ...
    max(damping_moving_average_n_s_m), mean(damping_moving_average_n_s_m));
fprintf('Robot samples: %d total -> %d %s (%.1f%%), %d %s (%.1f%%)\n', ...
    height(robot_table), label_counts(1), fine_label_text, ...
    100 * label_counts(1) / sum(label_counts), label_counts(2), ...
    gross_label_text, 100 * label_counts(2) / sum(label_counts));
fprintf(['sample_weight: mean %.3f, median %.3f -- %.1f%% of samples ' ...
    'are low-confidence (weight < 0.2, near the threshold), %.1f%% ' ...
    'are high-confidence (weight > 0.8)\n'], mean(robot_sample_weight), ...
    median(robot_sample_weight), ...
    100 * mean(robot_sample_weight < 0.2), ...
    100 * mean(robot_sample_weight > 0.8));

gaze_is_fine = gaze_table.label == fine_label_text;
fprintf('Gaze samples:  %d total -> %d %s (%.1f%%), %d %s (%.1f%%)\n', ...
    height(gaze_table), nnz(gaze_is_fine), fine_label_text, ...
    100 * nnz(gaze_is_fine) / height(gaze_table), nnz(~gaze_is_fine), ...
    gross_label_text, 100 * nnz(~gaze_is_fine) / height(gaze_table));

fprintf('\nWrote:\n  %s\n  %s\n', write_robot_labeled_csv_file, ...
    write_gaze_labeled_csv_file);


%% ================================================================
% Local functions
% =================================================================

function threshold = otsu_threshold(data, bin_count, min_class_fraction)
%OTSU_THRESHOLD Data-driven threshold that best separates data into two
% groups, by maximizing between-class variance (Otsu, 1979) -- the
% standard automatic-thresholding method for a (possibly) bimodal
% distribution. Implemented directly here, no Image Processing Toolbox
% dependency, per this project's convention of self-contained MATLAB
% scripts (that toolbox's otsuthresh() would otherwise do the same
% thing off a normalized histogram).
%
% Degrades gracefully on a non-bimodal (e.g. unimodal) distribution:
% still returns SOME threshold (whichever split maximizes between-class
% variance), just not as meaningful a "valley between two peaks" one --
% see the module docstring's note on checking sample_weight afterward.
%
% min_class_fraction excludes splits that put fewer than that fraction
% of samples on either side, so a small extreme-mean clump (e.g. a
% damping-saturation event) can't win by isolating itself -- see the
% module docstring's note on otsu_min_class_fraction.

    finite_data = data(isfinite(data));
    [counts, edges] = histcounts(finite_data, bin_count);
    bin_centers = 0.5 * (edges(1:end-1) + edges(2:end));

    total_count = sum(counts);
    cumulative_count = cumsum(counts);
    cumulative_weighted_sum = cumsum(bin_centers .* counts);

    % Class 0 = bins up to and including this one; class 1 = the rest.
    weight0 = cumulative_count / total_count;
    weight1 = 1 - weight0;
    mean0 = cumulative_weighted_sum ./ max(cumulative_count, 1);
    mean1 = (cumulative_weighted_sum(end) - cumulative_weighted_sum) ./ ...
        max(total_count - cumulative_count, 1);

    between_class_variance = weight0 .* weight1 .* (mean0 - mean1) .^ 2;
    % A split with an empty side (weight0 or weight1 == 0) is
    % meaningless, and one below min_class_fraction is too unbalanced
    % to trust -- both excluded rather than left to win on a small,
    % extreme-mean clump.
    between_class_variance(weight0 < min_class_fraction | ...
        weight1 < min_class_fraction) = -Inf;

    [~, best_bin_idx] = max(between_class_variance);
    threshold = bin_centers(best_bin_idx);
end


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample
% count. Copied from plot_raw_data_chen_var_adm_2d.m (self-contained
% here per this project's convention that MATLAB scripts do not call
% one another) -- odd, so a centered moving average (movmean) has an
% unambiguous center sample.

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


function resampled_columns = resample_nearest_multi( ...
        raw_time_s, raw_columns, query_time_s)
%RESAMPLE_NEAREST_MULTI Nearest-in-time resampling of MULTIPLE Nx1
% columns (any mix of text and numeric -- e.g. a string label column
% alongside numeric fine_score/sample_weight columns) onto
% query_time_s, sharing ONE sort/dedupe/nearest-index computation
% across every column so they all resample from the EXACT SAME nearest
% raw sample (rather than three separate calls risking drift).
% Generalizes extract_pilot_features_fixed_adm_2d.m's
% resample_labels_nearest to arbitrary column types (self-contained
% here per this project's convention that MATLAB scripts do not call
% one another): text columns cannot use interp1() directly the way
% numeric signals are, so the nearest RAW SAMPLE INDEX is found first
% (interp1 on plain sample indices, which is numeric) and that index is
% used to look every column up directly.
%
% raw_columns is a cell array of Nx1 columns, each the same length as
% raw_time_s. Returns a cell array the same size, each resampled onto
% query_time_s. raw_time_s can have duplicate or out-of-order
% timestamps, so it is sorted and deduplicated first.

    valid_rows = isfinite(raw_time_s);
    clean_time_s = raw_time_s(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');

    nearest_index = round(interp1(clean_time_s, ...
        (1:numel(clean_time_s))', query_time_s, 'nearest', 'extrap'));

    resampled_columns = cell(size(raw_columns));
    for column_idx = 1:numel(raw_columns)
        clean_column = raw_columns{column_idx}(valid_rows);
        clean_column = clean_column(sort_order);
        clean_column = clean_column(unique_indices);
        resampled_columns{column_idx} = clean_column(nearest_index);
    end
end


function [gaze_xy_m, gaze_valid] = map_gaze_ucs_to_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        require_gaze_on_display, require_both_eyes_valid)
%MAP_GAZE_UCS_TO_COPPELIA_XY Raw (not fixation-filtered) gaze position in
% the CoppeliaSim frame [m], one row per gaze sample (NaN where
% invalid). Reuses this project's standard calibrated-mapping formula
% (matches the raw_gaze_xy_m branch of detect_fixations_coppelia_xy in
% plot_raw_data_chen_var_adm_2d.m, and build_feature_dataset.py's
% load_calibrated_streams), with the require_both_eyes_valid /
% require_gaze_on_display toggle already used by this project's other
% calibrated-mapping scripts -- self-contained here per this project's
% convention that MATLAB scripts do not call one another.

    left_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);
    left_display = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1 & finite_rows(left_ucs_mm);
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1 & finite_rows(right_ucs_mm);

    if require_gaze_on_display
        left_point_valid = left_point_valid & point_is_on_display(left_display);
        right_point_valid = right_point_valid & point_is_on_display(right_display);
    end

    sample_count = height(gaze_table);
    if require_both_eyes_valid
        gaze_valid = left_point_valid & right_point_valid;
        gaze_ucs_xy_mm = 0.5 * ( ...
            left_ucs_mm(:, 1:2) + right_ucs_mm(:, 1:2));
    else
        eye_count = double(left_point_valid) + double(right_point_valid);
        gaze_valid = eye_count > 0;
        gaze_sum_mm = zeros(sample_count, 2);
        gaze_sum_mm(left_point_valid, :) = ...
            gaze_sum_mm(left_point_valid, :) + left_ucs_mm(left_point_valid, 1:2);
        gaze_sum_mm(right_point_valid, :) = ...
            gaze_sum_mm(right_point_valid, :) + right_ucs_mm(right_point_valid, 1:2);
        gaze_ucs_xy_mm = NaN(sample_count, 2);
        gaze_ucs_xy_mm(gaze_valid, :) = ...
            gaze_sum_mm(gaze_valid, :) ./ eye_count(gaze_valid);
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    gaze_displacement_ucs_mm = gaze_ucs_xy_mm - tobii_reference_xy_mm;
    gaze_displacement_coppelia_m = gaze_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    gaze_xy_m = coppelia_reference_xy_m + gaze_displacement_coppelia_m;
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied from plot_raw_data_chen_var_adm_2d.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV
% cells. Copied from plot_raw_data_chen_var_adm_2d.m.

    if ischar(raw_column)
        raw_column = cellstr(raw_column);
    end

    sample_count = numel(raw_column);
    vector_values = NaN(sample_count, vector_size);
    for sample_idx = 1:sample_count
        if iscell(raw_column)
            raw_value = raw_column{sample_idx};
        else
            raw_value = raw_column(sample_idx);
        end

        if isstring(raw_value)
            raw_value = char(raw_value);
        end
        if isempty(raw_value)
            continue;
        end

        raw_value = strrep(raw_value, '[', '');
        raw_value = strrep(raw_value, ']', '');
        raw_value = strrep(raw_value, ',', ' ');
        parsed_values = sscanf(raw_value, '%f');
        value_count = min(numel(parsed_values), vector_size);
        vector_values(sample_idx, 1:value_count) = parsed_values(1:value_count)';
    end
end


function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.
% Copied from plot_raw_data_chen_var_adm_2d.m.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display
% bounds. Copied from plot_raw_data_chen_var_adm_2d.m.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end
