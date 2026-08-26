%% Diagnostics for one fixed-admittance trial (scripts/ur5e_fixed_adm_2d.py).
%
% Sibling of plot_raw_data_chen_var_adm_2d.m for the simpler controller:
% B_x/B_y here are constants set once per trial (Eq (7)-style direct
% intention, power-law indirect intention, and the anisotropic damping
% matrix in that script all have no equivalent here), so the figures
% specific to that internal state are dropped. Everything that is a
% general property of a pHRI trial rather than of Chen's controller
% (paper performance metrics, execution check, gaze, replay) carries
% over with the same formulas. Every figure carries the trial's B_x/B_y
% as a sgtitle (or, for Figure 1's single axes, appended to its title),
% since damping does not vary within a trial here the way Chen's does.
%
% Inputs:
%   data/raw_data/ur5e_fixed_adm_2d_<trial>.csv   controller log
%   data/raw_data/tobii_fixed_adm_2d_<trial>.csv  matching gaze log
%   data/calibrations/<file>.json                 optional, gaze overlay
%   data/shapes/<shape>.obj                       optional, accuracy metric
%   All four are selected in User Settings.
%
% Output:
%   Figure 1: TCP path in the CoppeliaSim x-y frame, colored by
%   interaction force magnitude (the spatial-effort analog of Chen's
%   damping map -- damping itself is a flat constant here, so coloring
%   by it would be a single uniform color), with a raw-gaze overlay.
%   Figure 2: admittance execution. Commanded versus measured TCP
%   velocity, to confirm the robot tracked the virtual admittance model.
%   Figure 3: accuracy, paper Eq. (15). Mean distance from the traced
%   path to the reference centre line of the band, recovered from the
%   shape mesh. Skipped if the mesh is unavailable.
%   Figure 4: smoothness, paper Eq. (16). Dimensionless squared jerk,
%   reported at several smoothing windows because the value depends on
%   the filter.
%   Figure 5: effort, paper Eq. (17). Mean interaction force.
%   Figures 3 to 5 use movement samples only (speed above
%   moving_speed_threshold_m_s), as the paper does.
%   Figure 6: gaze. Both-eye-averaged pupil diameter and per-eye
%   validity.
%   Figure 7: replay of the trial. Left tile animates the TCP path, the
%   real-time raw gaze point (small green dot), and the I-VT fixations
%   detected so far (larger red markers); top-right shows interaction
%   force magnitude at the same instant (the analog of Chen's damping
%   panel -- force is what actually varies here); bottom-right shows
%   gaze/fixation distance from the TCP over time. All three gaze/force
%   layers are placed by their own timestamps on the shared trial clock,
%   so they stay synchronized despite different sample rates (gaze near
%   60 Hz, controller at 500 Hz). Set animate_evolution false to skip it.
%   A numeric summary is printed to the command window.
%   No file is written; this script only reads raw CSVs, JSON and OBJ.
%
% Method reuse:
%   I-VT fixation detection, eye combination, UCS-to-CoppeliaSim
%   mapping, the shape-band OBJ reader, and the jerk/DSJ chain are
%   copied verbatim from plot_raw_data_chen_var_adm_2d.m (itself from
%   compute_plot_similarity_fixation_direction_force_vel.m), self-
%   contained here per this project's convention that MATLAB scripts do
%   not call one another.
%
% Assumptions:
%   The CSV was written by scripts/ur5e_fixed_adm_2d.py, so the column
%   names match that script's write_csv_header(). The gaze CSV shares
%   its time_s origin with the controller CSV, as both are written from
%   the same trial clock.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO controller log written by scripts/ur5e_fixed_adm_2d.py.
robot_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'ur5e_fixed_adm_2d_fine_gross_shape_d100_s0.csv');

% TODO matching gaze log from the same trial (same time_s origin).
gaze_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'tobii_fixed_adm_2d_fine_gross_shape_d100_s0.csv');

% TODO true overlays gaze on the TCP path and draws the gaze figure.
% Set false to skip every gaze panel, e.g. for a trial recorded without
% the eye tracker.
plot_gaze_overlay = true;

% TODO calibration JSON mapping Tobii UCS to the CoppeliaSim frame. Only
% used when plot_gaze_overlay is true. If the file is missing the script
% still runs and only the gaze-position panels are skipped, because the
% other diagnostics need no calibration.
calibration_json_file = fullfile( ...
    script_directory, '..', 'data', 'calibrations', ...
    'tobii_coppelia_reference_calibration_fine_gross_shape_s0.json');

% TODO I-VT angular-velocity threshold [deg/s] and minimum retained
% fixation duration [s] for the gaze panel. Matches
% compute_plot_similarity_fixation_direction_force_vel.m.
velocity_threshold_deg_s = 20.0;
minimum_fixation_duration_s = 0.060;

% TODO true excludes gaze estimates outside the normalized display.
plot_only_on_display_gaze = true;

% TODO physical screen motion / robot motion [unitless], must be > 0,
% and the Tobii UCS axis signs in CoppeliaSim, each -1 or +1. Must match
% the values used elsewhere in this project.
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO shape mesh the operator traced, used only for the accuracy metric
% (Figure 3). The mesh must be a flat band: an outer and an inner edge
% loop, whose midpoint is taken as the reference path. Accuracy is
% skipped, with a warning, if this file is missing.
shape_obj_file = fullfile( ...
    script_directory, '..', 'data', 'shapes', 'fine_gross_shape.obj');

% TODO translation from the mesh CAD frame to the CoppeliaSim frame [m].
% VERIFY THIS before trusting the accuracy number: it sets where the
% reference path sits, so a wrong offset inflates the error directly.
% Figure 3 shows the reference against the traced path, and the summary
% prints the offset that would minimize the error, so a bad registration
% is visible rather than silent.
% current shapes offsets:
% fine_gross_shape: [0.0, -0.645]
% wide_curved_shape: [0.025, -0.650]
% narrow_curved_shape: [0.025, -0.638]
% wide_edge_shape: [0.0, -0.625]
% narrow_edge_shape: [0.0, -0.633]
% wide_circle_shape: [-0.01, -0.640]
% narrow_circle_shape: [-0.01, -0.658]
% protractor_shape: [-0.025, -0.638]
shape_offset_xy_m = [0.0, -0.645];

% TODO spacing used to resample the reference path [m], positive. The
% error is the distance to the nearest reference sample, so this sets
% the quantization floor of the accuracy metric; 5e-4 keeps it well
% below a typical tracking error.
reference_spacing_m = 5e-4;

% TODO speed separating movement from rest [m/s]. The paper uses 0.005
% and evaluates every performance metric on movement samples only.
moving_speed_threshold_m_s = 0.005;

% TODO filter windows for the jerk chain used by the smoothness metric
% [s]. Jerk is a third derivative, so it is dominated by noise unless
% the position trace is filtered first. The outlier window is a moving
% median; the smoothing window is a moving mean applied to position and
% again after each differentiation.
jerk_outlier_window_s = 0.02;
jerk_smoothing_window_s = 0.10;

% TODO extra smoothing windows [s] used only to report how sensitive DSJ
% is to this choice. DSJ values are comparable between trials only when
% computed with the same window.
jerk_sensitivity_windows_s = [0.05, 0.10, 0.20];

% TODO true plays the Figure 7 replay of the trial. Set false to skip
% it, since it runs in real user time rather than drawing at once.
animate_evolution = true;

% TODO replay every Nth control sample [count], positive. At 500 Hz a
% stride of 10 replays 50 frames per trial second. Larger is faster and
% coarser.
animation_stride = 10;

% TODO pause between replay frames [s], >= 0. Set 0 to replay as fast as
% the figure can redraw.
animation_pause_s = 0.001;

% TODO trailing window of TCP path kept bright during the replay [s],
% positive. The rest of the travelled path stays visible but faded, so
% both the recent motion and the whole history are readable.
animation_trail_s = 1.5;

% TODO Figure 7 real-time gaze point marker size [points^2] and color.
% Deliberately smaller than replay_fixation_marker_size below, so the
% raw, jittery gaze position reads as a transient dot while a detected
% fixation reads as the more significant event.
replay_gaze_point_marker_size = 10;
replay_gaze_point_color = 'g';

% TODO Figure 7 oldest a raw gaze sample may be and still be shown as
% "current" [s], positive. Prevents the real-time gaze dot from freezing
% on a stale position during a gaze-tracking dropout; it disappears
% instead once no sample this recent exists.
replay_gaze_point_max_age_s = 0.15;

% TODO Figure 7 fixation marker size [points^2] and color.
replay_fixation_marker_size = 60;
replay_fixation_marker_color = 'r';

% TODO marker sizes [points^2] and colors for the spatial panels.
trajectory_marker_size = 10;
raw_gaze_marker_size = 4;
raw_gaze_marker_color = "r";

%% Load the controller log
robot_table = readtable(robot_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

time_s = robot_table.time_s;

force_x_n = robot_table.Fx_filt_N;
force_y_n = robot_table.Fy_filt_N;
force_magnitude_n = hypot(force_x_n, force_y_n);
force_raw_magnitude_n = hypot(robot_table.Fx_raw_N, robot_table.Fy_raw_N);

coppeliasim_x_m = robot_table.coppeliasim_x_m;
coppeliasim_y_m = robot_table.coppeliasim_y_m;

velocity_command_x_m_s = robot_table.vx_cmd_m_s;
velocity_command_y_m_s = robot_table.vy_cmd_m_s;
speed_command_m_s = robot_table.speed_cmd_m_s;
speed_actual_m_s = hypot( ...
    robot_table.tcp_vx_actual_m_s, robot_table.tcp_vy_actual_m_s);

virtual_mass_x_kg = robot_table.M_x_kg(1);
virtual_mass_y_kg = robot_table.M_y_kg(1);
damping_x_n_s_m = robot_table.B_x_Ns_m(1);
damping_y_n_s_m = robot_table.B_y_Ns_m(1);

% B is a flat constant for this controller (unlike Chen's time-varying
% anisotropic matrix), so one label suffices for every figure below.
damping_info_str = sprintf( ...
    'B_x = %.0f, B_y = %.0f N s/m', damping_x_n_s_m, damping_y_n_s_m);

%% Load and map gaze, if requested and possible
gaze_available = false;
gaze_calibrated = false;
raw_gaze_xy_m = [];
raw_gaze_time_s = [];
fixation_xy_m = [];
fixation_time_s = [];
pupil_time_s = [];
pupil_diameter_mm = [];
gaze_time_s = [];
left_point_valid = [];
right_point_valid = [];

if plot_gaze_overlay && isfile(gaze_csv_file)
    gaze_table = readtable(gaze_csv_file, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
    gaze_available = true;

    [gaze_time_s, pupil_time_s, pupil_diameter_mm, ...
        left_point_valid, right_point_valid] = ...
        read_gaze_quality_series(gaze_table, plot_only_on_display_gaze);

    if isfile(calibration_json_file)
        calibration_reference = jsondecode(fileread(calibration_json_file));
        tobii_reference_xy_mm = double( ...
            calibration_reference.manual_tobii_reference_xy_mm(:)');
        coppelia_reference_xy_m = double( ...
            calibration_reference.manual_coppelia_reference_xy_m(:)');

        [fixation_xy_m, raw_gaze_xy_m, fixation_time_s, raw_gaze_time_s] = ...
            detect_fixations_coppelia_xy( ...
            gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
            screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
            velocity_threshold_deg_s, minimum_fixation_duration_s, ...
            plot_only_on_display_gaze);
        gaze_calibrated = true;
    else
        warning(['Calibration JSON not found, so gaze positions are ' ...
            'skipped. Other diagnostics are unaffected.']);
    end
elseif plot_gaze_overlay
    warning('Gaze CSV not found; skipping all gaze panels.');
end

%% ================================================================
% Figure 1: TCP path colored by interaction force magnitude
% =================================================================

figure('Name', 'TCP path and interaction force', 'Color', 'w');
hold on;

if gaze_calibrated && ~isempty(raw_gaze_xy_m)
    scatter(raw_gaze_xy_m(:, 1), raw_gaze_xy_m(:, 2), ...
        raw_gaze_marker_size, raw_gaze_marker_color, 'filled', ...
        'MarkerFaceAlpha', 0.25, 'DisplayName', 'Raw gaze (both eyes)');
end

scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
    force_magnitude_n, 'filled', 'DisplayName', 'TCP path');

if gaze_calibrated && ~isempty(fixation_xy_m)
    scatter(fixation_xy_m(:, 1), fixation_xy_m(:, 2), ...
        replay_fixation_marker_size, replay_fixation_marker_color, ...
        'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', 'I-VT fixation');
end

axis equal;
grid on;
force_colorbar = colorbar;
force_colorbar.Label.String = 'Interaction force |F| [N]';
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title(sprintf('TCP path colored by interaction force magnitude (%s)', ...
    damping_info_str));
legend('Location', 'best', 'FontSize', 8);

%% ================================================================
% Figure 2: Admittance execution
% =================================================================
% Confirms the robot tracked the virtual admittance model: the
% controller commands a Cartesian velocity from M*x_ddot + B*x_dot = F,
% and this checks how closely the measured TCP velocity followed it.

figure('Name', 'Admittance execution', 'Color', 'w');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, velocity_command_x_m_s, '-', 'LineWidth', 1.0);
hold on;
plot(time_s, robot_table.tcp_vx_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
ylabel('v_x [m/s]');
title('Commanded versus measured TCP velocity, x');
legend('Commanded', 'Measured', 'Location', 'best');

nexttile;
plot(time_s, velocity_command_y_m_s, '-', 'LineWidth', 1.0);
hold on;
plot(time_s, robot_table.tcp_vy_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
ylabel('v_y [m/s]');
title('Commanded versus measured TCP velocity, y');
legend('Commanded', 'Measured', 'Location', 'best');

nexttile;
plot(time_s, speed_command_m_s, '-', 'LineWidth', 1.2);
hold on;
plot(time_s, speed_actual_m_s, '-', 'LineWidth', 0.8);
grid on;
xlabel('Time [s]');
ylabel('Speed [m/s]');
title('Commanded versus measured planar speed');
legend('Commanded', 'Measured', 'Location', 'best');

sgtitle(damping_info_str);

%% ================================================================
% Figures 3 to 5: paper performance metrics, Section III.C
% =================================================================
% All three are evaluated on movement samples only, as the paper does:
% "A threshold of 0.005 m/s is set to differentiate the movement data
% from the whole trial data." Standing-still samples would otherwise
% dilute the mean error and mean force toward whatever the robot did
% while parked.

moving_mask = speed_actual_m_s > moving_speed_threshold_m_s;

%% ---- Figure 3: Accuracy, paper Eq. (15)
% Error = (1/M) * sum || X(i) - X_reference ||, the mean distance from
% the traced path to the reference path. The reference is the centre
% line of the band the operator was asked to trace, recovered from the
% shape mesh: the mesh gives an outer and an inner edge loop, and the
% centre line is the midpoint between them.

accuracy_available = false;

if isfile(shape_obj_file)
    [outer_edge_xy_m, inner_edge_xy_m, reference_xy_m] = ...
        read_shape_band_obj(shape_obj_file, reference_spacing_m);

    if isempty(reference_xy_m)
        warning(['Shape mesh did not yield two edge loops, so the ' ...
            'reference path is unavailable and accuracy is skipped.']);
    else
        % Mesh is in the CAD frame; move it into the CoppeliaSim frame.
        outer_edge_xy_m = outer_edge_xy_m + shape_offset_xy_m;
        inner_edge_xy_m = inner_edge_xy_m + shape_offset_xy_m;
        reference_xy_m = reference_xy_m + shape_offset_xy_m;

        tracking_error_m = distance_to_reference_path( ...
            coppeliasim_x_m, coppeliasim_y_m, reference_xy_m);
        accuracy_available = true;
    end
else
    warning(['Shape mesh not found, so accuracy is skipped. Set ' ...
        'shape_obj_file to the mesh the operator traced.']);
end

if accuracy_available
    mean_error_m = mean(tracking_error_m(moving_mask));

    figure('Name', 'Accuracy: tracking error', 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(outer_edge_xy_m(:, 1), outer_edge_xy_m(:, 2), '-', ...
        'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.0, ...
        'DisplayName', 'Band edge');
    hold on;
    plot(inner_edge_xy_m(:, 1), inner_edge_xy_m(:, 2), '-', ...
        'Color', [0.6, 0.6, 0.6], 'LineWidth', 1.0, ...
        'HandleVisibility', 'off');
    plot(reference_xy_m(:, 1), reference_xy_m(:, 2), '--', ...
        'Color', [0.1, 0.1, 0.1], 'LineWidth', 1.0, ...
        'DisplayName', 'Reference centre line');
    scatter(coppeliasim_x_m, coppeliasim_y_m, trajectory_marker_size, ...
        1e3 * tracking_error_m, 'filled', 'DisplayName', 'TCP path');
    axis equal;
    grid on;
    error_colorbar = colorbar;
    error_colorbar.Label.String = 'Error [mm]';
    xlabel('CoppeliaSim x [m]');
    ylabel('CoppeliaSim y [m]');
    title('TCP path colored by distance to the reference');
    legend('Location', 'best', 'FontSize', 8);

    nexttile;
    plot(time_s, 1e3 * tracking_error_m, '-', 'LineWidth', 1.0);
    hold on;
    yline(1e3 * mean_error_m, 'r--', ...
        sprintf('Mean %.2f mm', 1e3 * mean_error_m));
    grid on;
    xlabel('Time [s]');
    ylabel('Error [mm]');
    title('Distance to the reference path over time');

    sgtitle(damping_info_str);
end

%% ---- Figure 4: Smoothness, dimensionless squared jerk, paper Eq. (16)
% DSJ = integral( jerk_x^2 + jerk_y^2 ) dt * (t2 - t1)^5 / A^2, with A
% the path length. Jerk is the third derivative of position, so it
% amplifies noise three times over; the position trace is filtered and
% each derivative is smoothed again before the next. DSJ is reported at
% several smoothing windows because the absolute value is filter-
% dependent, and only DSJ values computed with the SAME window are
% comparable between trials.

figure('Name', 'Smoothness: dimensionless squared jerk', 'Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

jerk = compute_jerk_series(time_s, coppeliasim_x_m, coppeliasim_y_m, ...
    jerk_outlier_window_s, jerk_smoothing_window_s);

dimensionless_squared_jerk = compute_dsj( ...
    jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m);

nexttile;
plot(jerk.time_s, sqrt(jerk.squared_jerk_m2_s6), '-', 'LineWidth', 1.0);
grid on;
ylabel('|jerk| [m/s^3]');
title(sprintf(['Jerk magnitude (smoothing window %.0f ms). ' ...
    'DSJ = %.3g'], 1e3 * jerk_smoothing_window_s, ...
    dimensionless_squared_jerk));

nexttile;
plot(jerk.time_s, cumtrapz(jerk.time_s, jerk.squared_jerk_m2_s6), ...
    '-', 'LineWidth', 1.2);
grid on;
xlabel('Time [s]');
ylabel('\int |jerk|^2 dt [m^2/s^5]');
title(['Cumulative squared jerk: steep sections are where smoothness ' ...
    'was lost']);

sgtitle(damping_info_str);

% Filter sensitivity, so the number is never read as filter-independent.
dsj_sensitivity = zeros(numel(jerk_sensitivity_windows_s), 1);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    sensitivity_jerk = compute_jerk_series( ...
        time_s, coppeliasim_x_m, coppeliasim_y_m, ...
        jerk_outlier_window_s, jerk_sensitivity_windows_s(window_idx));
    dsj_sensitivity(window_idx) = compute_dsj(sensitivity_jerk.time_s, ...
        sensitivity_jerk.squared_jerk_m2_s6, ...
        sensitivity_jerk.path_length_m);
end

%% ---- Figure 5: Effort, mean force, paper Eq. (17)
% MeanForce = (1/M) * sum || F_ext(i) ||, over movement samples.

mean_force_n = mean(force_magnitude_n(moving_mask));

figure('Name', 'Effort: interaction force', 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, force_magnitude_n, '-', 'LineWidth', 1.0);
hold on;
yline(mean_force_n, 'r--', sprintf('Mean %.3f N', mean_force_n));
grid on;
xlabel('Time [s]');
ylabel('|F| [N]');
title('Interaction force magnitude over time');

nexttile;
histogram(force_magnitude_n(moving_mask), 40, ...
    'Normalization', 'probability');
hold on;
xline(mean_force_n, 'r--', sprintf('Mean %.3f N', mean_force_n));
grid on;
xlabel('|F| [N]');
ylabel('Fraction of moving samples [-]');
title('Force distribution while moving');

sgtitle(damping_info_str);

%% ================================================================
% Figure 6: Gaze quality and pupil diameter
% =================================================================

if gaze_available
    figure('Name', 'Gaze quality and pupil diameter', 'Color', 'w');
    tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    plot(pupil_time_s, pupil_diameter_mm, '-', 'LineWidth', 1.0);
    grid on;
    ylabel('Pupil [mm]');
    title(['Pupil diameter, strictly averaged over both eyes ' ...
        '(samples with one valid eye are excluded)']);

    nexttile;
    stairs(gaze_time_s, double(left_point_valid), '-', 'LineWidth', 1.0);
    hold on;
    stairs(gaze_time_s, 1.1 * double(right_point_valid), '-', ...
        'LineWidth', 1.0);
    grid on;
    ylim([-0.1, 1.3]);
    xlabel('Time [s]');
    ylabel('Valid [-]');
    title('Per-eye gaze-point validity (right offset for visibility)');
    legend('Left', 'Right', 'Location', 'best');

    sgtitle(damping_info_str);
end

%% ================================================================
% Figure 7: replay of gaze, TCP path, and interaction force
% =================================================================
% Left tile replays where the operator was looking against where the TCP
% actually was; top-right shows the interaction force at that same
% instant (the analog of Chen's damping panel -- damping is a flat
% constant here, so force is what actually carries the effort story);
% bottom-right shows gaze/fixation distance from the TCP over time.
%
% Graphics handles are updated in place rather than re-plotting each
% frame, so the replay stays responsive over a full trial instead of
% accumulating thousands of line objects.

if animate_evolution
    figure('Name', 'Replay: gaze, TCP path, and force', 'Color', 'w');
    tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % ---- Left tile: spatial replay (spans both rows)
    path_axes = nexttile(1, [2, 1]);
    hold(path_axes, 'on');

    if accuracy_available
        plot(path_axes, outer_edge_xy_m(:, 1), outer_edge_xy_m(:, 2), ...
            '-', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 1.0, ...
            'DisplayName', 'Band edge');
        plot(path_axes, inner_edge_xy_m(:, 1), inner_edge_xy_m(:, 2), ...
            '-', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 1.0, ...
            'HandleVisibility', 'off');
    end

    % Whole path, faded, so the replay is read against the full shape.
    plot(path_axes, coppeliasim_x_m, coppeliasim_y_m, '-', ...
        'Color', [0.88, 0.88, 0.88], 'LineWidth', 0.8, ...
        'DisplayName', 'Full TCP path');

    travelled_handle = plot(path_axes, NaN, NaN, '-', ...
        'Color', [0.30, 0.30, 0.30], 'LineWidth', 0.8, ...
        'DisplayName', 'Travelled');
    trail_handle = plot(path_axes, NaN, NaN, '-', ...
        'Color', [0.85, 0.20, 0.10], 'LineWidth', 1.8, ...
        'DisplayName', 'Recent TCP');
    tcp_marker_handle = plot(path_axes, NaN, NaN, 'o', ...
        'MarkerSize', 7, 'MarkerFaceColor', [0.85, 0.20, 0.10], ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'TCP now');
    gaze_point_handle = scatter(path_axes, NaN, NaN, ...
        replay_gaze_point_marker_size, replay_gaze_point_color, ...
        'filled', 'DisplayName', 'Real-time gaze point');
    fixation_handle = scatter(path_axes, NaN, NaN, ...
        replay_fixation_marker_size, replay_fixation_marker_color, ...
        'filled', 'MarkerFaceAlpha', 0.7, 'DisplayName', ...
        'Fixations so far');

    axis(path_axes, 'equal');
    grid(path_axes, 'on');

    % Limits fixed from the data so the view never jumps mid-replay.
    view_pad_m = 0.01;
    all_x_m = coppeliasim_x_m;
    all_y_m = coppeliasim_y_m;
    if gaze_calibrated && ~isempty(fixation_xy_m)
        all_x_m = [all_x_m; fixation_xy_m(:, 1)];
        all_y_m = [all_y_m; fixation_xy_m(:, 2)];
    end
    if gaze_calibrated && ~isempty(raw_gaze_xy_m)
        all_x_m = [all_x_m; raw_gaze_xy_m(:, 1)];
        all_y_m = [all_y_m; raw_gaze_xy_m(:, 2)];
    end
    xlim(path_axes, [min(all_x_m) - view_pad_m, max(all_x_m) + view_pad_m]);
    ylim(path_axes, [min(all_y_m) - view_pad_m, max(all_y_m) + view_pad_m]);
    xlabel(path_axes, 'CoppeliaSim x [m]');
    ylabel(path_axes, 'CoppeliaSim y [m]');
    legend(path_axes, 'Location', 'best', 'FontSize', 7);

    % ---- Top-right tile: interaction force against time
    force_axes = nexttile;
    hold(force_axes, 'on');

    plot(force_axes, time_s, force_magnitude_n, '-', ...
        'Color', [0.85, 0.85, 0.85], 'LineWidth', 0.8, ...
        'DisplayName', '|F| (whole trial)');
    force_trail_handle = plot(force_axes, NaN, NaN, '-', ...
        'Color', [0.00, 0.45, 0.74], 'LineWidth', 1.4, ...
        'DisplayName', '|F| so far');
    force_marker_handle = plot(force_axes, NaN, NaN, 'o', ...
        'MarkerSize', 7, 'MarkerFaceColor', [0.00, 0.45, 0.74], ...
        'MarkerEdgeColor', 'k', 'DisplayName', '|F| now');

    grid(force_axes, 'on');
    xlim(force_axes, [time_s(1), time_s(end)]);
    xlabel(force_axes, 'Time [s]');
    ylabel(force_axes, '|F| [N]');
    title(force_axes, 'Interaction force');
    legend(force_axes, 'Location', 'best', 'FontSize', 7);

    % ---- Bottom-right tile: gaze and fixation distance from the TCP,
    % over time. Distance from the raw gaze point to the TCP position at
    % the same instant, interpolated onto the gaze sample times since
    % gaze runs slower (~60 Hz) than the controller (500 Hz). This is
    % the temporal companion to the spatial replay: a rise in this
    % distance just before a turn is what "gaze leads the TCP" looks
    % like on a time axis. Fixations are overlaid on the same distance
    % curve, at their own mid-time.
    gaze_temporal_axes = nexttile;
    hold(gaze_temporal_axes, 'on');

    have_gaze_distance = gaze_calibrated && ~isempty(raw_gaze_xy_m);
    if have_gaze_distance
        tcp_x_at_gaze_m = interp1(time_s, coppeliasim_x_m, ...
            raw_gaze_time_s, 'linear', 'extrap');
        tcp_y_at_gaze_m = interp1(time_s, coppeliasim_y_m, ...
            raw_gaze_time_s, 'linear', 'extrap');
        gaze_to_tcp_distance_mm = 1e3 * hypot( ...
            raw_gaze_xy_m(:, 1) - tcp_x_at_gaze_m, ...
            raw_gaze_xy_m(:, 2) - tcp_y_at_gaze_m);

        plot(gaze_temporal_axes, raw_gaze_time_s, ...
            gaze_to_tcp_distance_mm, '-', 'Color', [0.85, 0.85, 0.85], ...
            'LineWidth', 0.8, 'DisplayName', ...
            'Gaze-TCP distance (whole trial)');
    end

    gaze_distance_trail_handle = plot(gaze_temporal_axes, NaN, NaN, ...
        '-', 'Color', [0.10, 0.55, 0.20], 'LineWidth', 1.2, ...
        'DisplayName', 'Gaze-TCP distance so far');
    gaze_distance_marker_handle = plot(gaze_temporal_axes, NaN, NaN, ...
        'o', 'MarkerSize', 7, 'MarkerFaceColor', [0.10, 0.55, 0.20], ...
        'MarkerEdgeColor', 'k', 'DisplayName', 'Gaze-TCP distance now');

    have_fixation_distance = have_gaze_distance && ~isempty(fixation_xy_m);
    if have_fixation_distance
        tcp_x_at_fixation_m = interp1(time_s, coppeliasim_x_m, ...
            fixation_time_s, 'linear', 'extrap');
        tcp_y_at_fixation_m = interp1(time_s, coppeliasim_y_m, ...
            fixation_time_s, 'linear', 'extrap');
        fixation_to_tcp_distance_mm = 1e3 * hypot( ...
            fixation_xy_m(:, 1) - tcp_x_at_fixation_m, ...
            fixation_xy_m(:, 2) - tcp_y_at_fixation_m);
    end
    fixation_distance_handle = scatter(gaze_temporal_axes, NaN, NaN, ...
        replay_fixation_marker_size, replay_fixation_marker_color, ...
        'filled', 'DisplayName', 'Fixations so far');

    grid(gaze_temporal_axes, 'on');
    xlim(gaze_temporal_axes, [time_s(1), time_s(end)]);
    xlabel(gaze_temporal_axes, 'Time [s]');
    ylabel(gaze_temporal_axes, 'Distance from TCP [mm]');
    title(gaze_temporal_axes, 'Gaze and fixation distance from the TCP');
    legend(gaze_temporal_axes, 'Location', 'best', 'FontSize', 7);

    % Shared time axis with the force panel directly above it.
    linkaxes([force_axes, gaze_temporal_axes], 'x');

    sgtitle(damping_info_str);

    % ---- Replay loop
    for sample_idx = 1:animation_stride:numel(time_s)
        now_s = time_s(sample_idx);

        trail_start_idx = find( ...
            time_s >= now_s - animation_trail_s, 1, 'first');

        set(travelled_handle, ...
            'XData', coppeliasim_x_m(1:sample_idx), ...
            'YData', coppeliasim_y_m(1:sample_idx));
        set(trail_handle, ...
            'XData', coppeliasim_x_m(trail_start_idx:sample_idx), ...
            'YData', coppeliasim_y_m(trail_start_idx:sample_idx));
        set(tcp_marker_handle, ...
            'XData', coppeliasim_x_m(sample_idx), ...
            'YData', coppeliasim_y_m(sample_idx));

        set(force_trail_handle, ...
            'XData', time_s(1:sample_idx), ...
            'YData', force_magnitude_n(1:sample_idx));
        set(force_marker_handle, ...
            'XData', now_s, ...
            'YData', force_magnitude_n(sample_idx));

        if gaze_calibrated && ~isempty(raw_gaze_xy_m)
            % Most recent raw gaze sample at or before now_s (causal, no
            % lookahead). Hidden instead of frozen when the newest such
            % sample is older than replay_gaze_point_max_age_s, e.g.
            % during a gaze-tracking dropout.
            gaze_now_idx = find(raw_gaze_time_s <= now_s, 1, 'last');
            gaze_is_current = ~isempty(gaze_now_idx) && ...
                now_s - raw_gaze_time_s(gaze_now_idx) <= ...
                replay_gaze_point_max_age_s;

            if gaze_is_current
                set(gaze_point_handle, ...
                    'XData', raw_gaze_xy_m(gaze_now_idx, 1), ...
                    'YData', raw_gaze_xy_m(gaze_now_idx, 2));
                set(gaze_distance_marker_handle, ...
                    'XData', now_s, ...
                    'YData', gaze_to_tcp_distance_mm(gaze_now_idx));
            else
                set(gaze_point_handle, 'XData', NaN, 'YData', NaN);
                set(gaze_distance_marker_handle, 'XData', NaN, 'YData', NaN);
            end

            gaze_shown_mask = raw_gaze_time_s <= now_s;
            if any(gaze_shown_mask)
                set(gaze_distance_trail_handle, ...
                    'XData', raw_gaze_time_s(gaze_shown_mask), ...
                    'YData', gaze_to_tcp_distance_mm(gaze_shown_mask));
            end
        end

        if gaze_calibrated && ~isempty(fixation_xy_m)
            shown_mask = fixation_time_s <= now_s;
            if any(shown_mask)
                set(fixation_handle, ...
                    'XData', fixation_xy_m(shown_mask, 1), ...
                    'YData', fixation_xy_m(shown_mask, 2));

                % have_fixation_distance additionally requires raw gaze
                % (both-eyes-valid) data, which fixation_xy_m itself does
                % not: fixations tolerate a single-eye fallback, so a
                % trial could have fixations with no both-eyes-valid
                % sample at all, leaving fixation_to_tcp_distance_mm
                % unset.
                if have_fixation_distance
                    set(fixation_distance_handle, ...
                        'XData', fixation_time_s(shown_mask), ...
                        'YData', fixation_to_tcp_distance_mm(shown_mask));
                end
            end
        end

        title(path_axes, sprintf(['Gaze and TCP replay, t = %.2f s, ' ...
            '|F| = %.2f N'], now_s, force_magnitude_n(sample_idx)));

        drawnow limitrate;
        if animation_pause_s > 0
            pause(animation_pause_s);
        end
    end

    % Leave the finished frame showing the whole trial.
    title(path_axes, 'Gaze and TCP replay (finished)');
end

%% ================================================================
% Summary
% =================================================================

trial_duration_s = time_s(end) - time_s(1);
sample_interval_s = diff(time_s);
path_length_m = sum(hypot(diff(coppeliasim_x_m), diff(coppeliasim_y_m)));

fprintf('\n--- Fixed admittance trial summary ---\n');
fprintf('File: %s\n', robot_csv_file);
fprintf('Duration: %.2f s over %d samples (%.1f Hz mean)\n', ...
    trial_duration_s, height(robot_table), ...
    (height(robot_table) - 1) / trial_duration_s);
fprintf('Control period: median %.3f ms, max %.3f ms\n', ...
    1e3 * median(sample_interval_s), 1e3 * max(sample_interval_s));
fprintf('Virtual mass: M_x = %.2f kg, M_y = %.2f kg\n', ...
    virtual_mass_x_kg, virtual_mass_y_kg);
fprintf('Fixed damping: B_x = %.2f, B_y = %.2f N s/m\n', ...
    damping_x_n_s_m, damping_y_n_s_m);
fprintf('Path length: %.4f m, moving %.1f %% of samples\n', ...
    path_length_m, 100 * mean(moving_mask));

fprintf('\nEffort and motion:\n');
fprintf('  Mean force: %.3f N, peak %.3f N\n', ...
    mean(force_magnitude_n), max(force_magnitude_n));
fprintf('  Mean speed: %.4f m/s, peak %.4f m/s\n', ...
    mean(speed_actual_m_s), max(speed_actual_m_s));

fprintf('\nPerformance metrics, paper Section III.C ');
fprintf('(movement samples only, speed > %.3f m/s):\n', ...
    moving_speed_threshold_m_s);

if accuracy_available
    moving_error_m = tracking_error_m(moving_mask);
    fprintf('  Accuracy  Eq. (15) mean error: %.2f mm ', ...
        1e3 * mean_error_m);
    fprintf('(median %.2f, max %.2f)\n', ...
        1e3 * median(moving_error_m), 1e3 * max(moving_error_m));

    % Registration sanity check. This offset is NOT applied; it only
    % says how well the stated shape_offset_xy_m lines the reference up
    % with the traced path. A large residual shift means the accuracy
    % number is measuring misregistration, not operator error.
    [best_shift_xy_m, best_error_m] = estimate_reference_shift( ...
        coppeliasim_x_m(moving_mask), coppeliasim_y_m(moving_mask), ...
        reference_xy_m, reference_spacing_m);
    fprintf('    registration check: a further shift of [%+.1f, %+.1f] mm ', ...
        1e3 * best_shift_xy_m(1), 1e3 * best_shift_xy_m(2));
    fprintf('would give %.2f mm\n', 1e3 * best_error_m);
    if norm(best_shift_xy_m) > 0.005
        fprintf(['    NOTE: that is over 5 mm, so verify ' ...
            'shape_offset_xy_m before quoting this accuracy.\n']);
    end
else
    fprintf('  Accuracy  Eq. (15): skipped, no reference path.\n');
end

fprintf('  Smooth.   Eq. (16) DSJ: %.4g ', dimensionless_squared_jerk);
fprintf('(smoothing window %.0f ms)\n', 1e3 * jerk_smoothing_window_s);
for window_idx = 1:numel(jerk_sensitivity_windows_s)
    fprintf('    at %3.0f ms window: DSJ = %.4g\n', ...
        1e3 * jerk_sensitivity_windows_s(window_idx), ...
        dsj_sensitivity(window_idx));
end
fprintf(['    DSJ is filter-dependent; only compare trials computed ' ...
    'with the same window.\n']);

fprintf('  Effort    Eq. (17) mean force: %.3f N\n', mean_force_n);

if gaze_available
    fprintf('\nGaze:\n');
    fprintf('  Samples: %d at %.1f Hz\n', numel(gaze_time_s), ...
        (numel(gaze_time_s) - 1) / (gaze_time_s(end) - gaze_time_s(1)));
    fprintf('  Left valid %.1f %%, right valid %.1f %%\n', ...
        100 * mean(left_point_valid), 100 * mean(right_point_valid));
    if ~isempty(pupil_diameter_mm)
        fprintf('  Pupil (both eyes): mean %.3f mm, range %.3f to %.3f\n', ...
            mean(pupil_diameter_mm), min(pupil_diameter_mm), ...
            max(pupil_diameter_mm));
    end
    if gaze_calibrated
        fprintf('  I-VT fixations: %d\n', size(fixation_xy_m, 1));
    end
end

fprintf('\n');


%% ================================================================
%% Local functions
%% ================================================================
% Copied verbatim from plot_raw_data_chen_var_adm_2d.m (itself from
% compute_plot_similarity_fixation_direction_force_vel.m), per this
% project's convention that MATLAB scripts do not call one another.
% None of these depend on which controller produced the data.

function [gaze_time_s, pupil_time_s, pupil_diameter_mm, ...
        left_point_valid, right_point_valid] = ...
        read_gaze_quality_series(gaze_table, plot_only_on_display_gaze)
%READ_GAZE_QUALITY_SERIES Per-eye validity and both-eye pupil diameter.
%
% Input:
%   gaze_table - table read from a tobii_*.csv of this project.
%   plot_only_on_display_gaze - true also requires each eye's gaze point
%       to lie inside the normalized display area.
%
% Output:
%   gaze_time_s - trial time of every gaze sample [s].
%   pupil_time_s, pupil_diameter_mm - time and strictly both-eye-averaged
%       pupil diameter [mm]; samples with only one valid pupil are
%       excluded, not substituted, because a single-eye value is not
%       comparable with a two-eye average.
%   left_point_valid, right_point_valid - per-eye gaze-point validity.

    gaze_time_s = get_numeric_column(gaze_table.time_s);

    left_pupil_mm = get_numeric_column(gaze_table.left_pupil_diameter);
    right_pupil_mm = get_numeric_column(gaze_table.right_pupil_diameter);
    left_pupil_valid = get_numeric_column( ...
        gaze_table.left_pupil_validity) == 1 & isfinite(left_pupil_mm);
    right_pupil_valid = get_numeric_column( ...
        gaze_table.right_pupil_validity) == 1 & isfinite(right_pupil_mm);
    both_pupil_valid = left_pupil_valid & right_pupil_valid;

    pupil_time_s = gaze_time_s(both_pupil_valid);
    pupil_diameter_mm = 0.5 * ( ...
        left_pupil_mm(both_pupil_valid) + right_pupil_mm(both_pupil_valid));

    left_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1 & ...
        finite_rows(left_gaze_ucs_mm);
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1 & ...
        finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_display_gaze = parse_vector_column( ...
            gaze_table.left_gaze_point_on_display_area, 2);
        right_display_gaze = parse_vector_column( ...
            gaze_table.right_gaze_point_on_display_area, 2);
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end
end


function [fixation_xy_m, raw_gaze_xy_m, fixation_time_s, ...
        raw_gaze_time_s] = detect_fixations_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres in CoppeliaSim [m].
%
% Reuses the I-VT and UCS-to-CoppeliaSim formulas from
% compute_plot_similarity_fixation_direction_force_vel.m. fixation_xy_m
% is N-by-2 and empty when no fixation meets the minimum duration.
% raw_gaze_xy_m is every sample where BOTH eyes are valid, strictly
% averaged (left + right) / 2, mapped the same way; raw_gaze_time_s is
% its matching trial time [s], so a real-time gaze point can be replayed
% alongside the controller log. fixation_time_s is the mid-time of each
% fixation, on the shared trial clock, for the same purpose.

    left_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);
    left_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_origin_in_user_coordinate_system, 3);
    right_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_origin_in_user_coordinate_system, 3);
    left_display_gaze = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display_gaze = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1;
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1;
    left_origin_valid = get_numeric_column( ...
        gaze_table.left_gaze_origin_validity) == 1;
    right_origin_valid = get_numeric_column( ...
        gaze_table.right_gaze_origin_validity) == 1;

    gaze_time_s = get_numeric_column(gaze_table.time_s);

    left_point_valid = left_point_valid & finite_rows(left_gaze_ucs_mm);
    right_point_valid = right_point_valid & finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end

    [left_gaze_direction, left_eye_valid] = calculate_eye_direction( ...
        left_gaze_origin_ucs_mm, left_gaze_ucs_mm, ...
        left_origin_valid, left_point_valid);
    [right_gaze_direction, right_eye_valid] = calculate_eye_direction( ...
        right_gaze_origin_ucs_mm, right_gaze_ucs_mm, ...
        right_origin_valid, right_point_valid);

    [combined_gaze_ucs_mm, combined_point_valid] = combine_valid_eye_points( ...
        left_gaze_ucs_mm, right_gaze_ucs_mm, left_eye_valid, right_eye_valid);
    [combined_gaze_direction, combined_direction_valid] = ...
        combine_valid_eye_directions( ...
        left_gaze_direction, right_gaze_direction, ...
        left_eye_valid, right_eye_valid);
    combined_gaze_valid = combined_point_valid & combined_direction_valid;
    combined_gaze_ucs_mm(~combined_gaze_valid, :) = NaN;

    angular_velocity_deg_s = calculate_angular_velocity( ...
        combined_gaze_direction, gaze_time_s, combined_gaze_valid);
    fixation_candidate = combined_gaze_valid & ...
        isfinite(angular_velocity_deg_s) & ...
        angular_velocity_deg_s <= velocity_threshold_deg_s;

    candidate_changes = diff(double([false; fixation_candidate; false]));
    candidate_start_indices = find(candidate_changes == 1);
    candidate_end_indices = find(candidate_changes == -1) - 1;

    fixation_mean_ucs_x_mm = zeros(0, 1);
    fixation_mean_ucs_y_mm = zeros(0, 1);
    fixation_time_s = zeros(0, 1);

    for candidate_idx = 1:numel(candidate_start_indices)
        start_idx = candidate_start_indices(candidate_idx);
        end_idx = candidate_end_indices(candidate_idx);
        candidate_duration_s = gaze_time_s(end_idx) - gaze_time_s(start_idx);

        if candidate_duration_s < minimum_fixation_duration_s
            continue;
        end

        fixation_rows = start_idx:end_idx;
        mean_ucs_mm = mean(combined_gaze_ucs_mm(fixation_rows, :), 1);
        fixation_mean_ucs_x_mm(end + 1, 1) = mean_ucs_mm(1); %#ok<AGROW>
        fixation_mean_ucs_y_mm(end + 1, 1) = mean_ucs_mm(2); %#ok<AGROW>

        % Mid-time of the fixation, on the shared trial clock.
        fixation_time_s(end + 1, 1) = 0.5 * ( ...
            gaze_time_s(start_idx) + gaze_time_s(end_idx)); %#ok<AGROW>
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    fixation_ucs_xy_mm = [fixation_mean_ucs_x_mm, fixation_mean_ucs_y_mm];
    fixation_displacement_ucs_mm = fixation_ucs_xy_mm - tobii_reference_xy_mm;
    fixation_displacement_coppelia_m = fixation_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    fixation_xy_m = coppelia_reference_xy_m + fixation_displacement_coppelia_m;

    % Strict two-eye average: unlike combined_gaze_ucs_mm above, this
    % does not fall back to a single eye when only one is valid.
    both_eyes_point_valid = left_point_valid & right_point_valid;
    raw_gaze_ucs_xy_mm = 0.5 * ( ...
        left_gaze_ucs_mm(both_eyes_point_valid, 1:2) + ...
        right_gaze_ucs_mm(both_eyes_point_valid, 1:2));
    raw_gaze_displacement_ucs_mm = raw_gaze_ucs_xy_mm - tobii_reference_xy_mm;
    raw_gaze_displacement_coppelia_m = raw_gaze_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    raw_gaze_xy_m = coppelia_reference_xy_m + raw_gaze_displacement_coppelia_m;
    raw_gaze_time_s = gaze_time_s(both_eyes_point_valid);
end


function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
        obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh.
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

    % Bottom cap: faces whose vertices all lie on the minimum z plane.
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

    % Chain boundary edges into closed loops.
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

    % Centre line: pair each outer sample with its nearest inner sample.
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
%
% Loops over reference samples keeping a running minimum, so memory
% stays O(number of path samples) rather than forming the full pairwise
% distance matrix. Because the reference is densely resampled, the
% nearest-sample distance approximates the true nearest-point distance
% to within half the reference spacing.

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
%
% Diagnostic only, never applied to the reported accuracy: fitting the
% reference to the data would bias the metric. It exists so a bad
% shape_offset_xy_m shows up as a large residual shift instead of
% silently inflating the error.
%
% Coarse grid followed by one refinement, both centred on zero shift.

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
%
% Input:
%   raw_time_s, raw_x_m, raw_y_m - logged TCP trajectory.
%   outlier_window_s - moving-median window applied to position [s].
%   smoothing_window_s - moving-mean window applied to position and
%       again after each differentiation [s].
%
% Output:
%   jerk.time_s, jerk.squared_jerk_m2_s6, jerk.path_length_m.
%
% Jerk is a third derivative, so noise is amplified three times. The
% chain resamples onto a uniform grid, rejects outliers, smooths, and
% then smooths again after every differentiation, matching the approach
% used for curvature elsewhere in this project.

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
    smoothing_samples = window_to_odd_sample_count( ...
        smoothing_window_s, median_interval_s, numel(uniform_time_s));

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
%
%   DSJ = integral( jerk^2 ) dt * (t2 - t1)^5 / A^2
%
% with A the path length. The units cancel: (m^2/s^6)*s * s^5 / m^2 is
% dimensionless, which is what makes DSJ comparable across trials of
% different duration and path length.

    if path_length_m <= 0 || numel(time_s) < 2
        dsj = NaN;
        return;
    end

    movement_duration_s = time_s(end) - time_s(1);
    squared_jerk_integral = trapz(time_s, squared_jerk_m2_s6);

    dsj = squared_jerk_integral * movement_duration_s ^ 5 / ...
        path_length_m ^ 2;
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV cells.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
        vector_values(sample_idx, 1:value_count) = ...
            parsed_values(1:value_count)';
    end
end


function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display bounds.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end


function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = size(left_point, 1);
    valid_eye_count = double(left_valid) + double(right_valid);
    combined_valid = valid_eye_count > 0;

    point_sum = zeros(sample_count, 3);
    point_sum(left_valid, :) = left_point(left_valid, :);
    point_sum(right_valid, :) = point_sum(right_valid, :) + ...
        right_point(right_valid, :);

    combined_point = NaN(sample_count, 3);
    combined_point(combined_valid, :) = ...
        point_sum(combined_valid, :) ./ valid_eye_count(combined_valid);
end


function [unit_direction, eye_valid] = calculate_eye_direction( ...
        gaze_origin_ucs_mm, gaze_point_ucs_mm, ...
        origin_validity, point_validity)
%CALCULATE_EYE_DIRECTION Build one eye's unit direction in Tobii UCS.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    direction = gaze_point_ucs_mm - gaze_origin_ucs_mm;
    direction_norm = sqrt(sum(direction .^ 2, 2));
    eye_valid = origin_validity & point_validity & ...
        finite_rows(gaze_origin_ucs_mm) & finite_rows(gaze_point_ucs_mm) & ...
        isfinite(direction_norm) & direction_norm > 0;

    unit_direction = NaN(size(direction));
    unit_direction(eye_valid, :) = direction(eye_valid, :) ./ ...
        direction_norm(eye_valid);
end


function [combined_direction, combined_valid] = ...
        combine_valid_eye_directions( ...
        left_direction, right_direction, left_valid, right_valid)
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye directions.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = size(left_direction, 1);
    combined_valid = left_valid | right_valid;

    direction_sum = zeros(sample_count, 3);
    direction_sum(left_valid, :) = left_direction(left_valid, :);
    direction_sum(right_valid, :) = direction_sum(right_valid, :) + ...
        right_direction(right_valid, :);

    direction_norm = sqrt(sum(direction_sum .^ 2, 2));
    combined_valid = combined_valid & isfinite(direction_norm) & ...
        direction_norm > 0;

    combined_direction = NaN(sample_count, 3);
    combined_direction(combined_valid, :) = ...
        direction_sum(combined_valid, :) ./ direction_norm(combined_valid);
end


function angular_velocity_deg_s = calculate_angular_velocity( ...
        gaze_direction, time_s, gaze_valid)
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s].
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = numel(time_s);
    angular_velocity_deg_s = NaN(sample_count, 1);
    sample_interval_s = diff(time_s);
    valid_pair = gaze_valid(1:end-1) & gaze_valid(2:end) & ...
        isfinite(sample_interval_s) & sample_interval_s > 0;

    direction_dot_product = sum( ...
        gaze_direction(1:end-1, :) .* gaze_direction(2:end, :), 2);
    direction_dot_product = min(max(direction_dot_product, -1.0), 1.0);
    angular_step_deg = acosd(direction_dot_product);

    interval_velocity_deg_s = NaN(sample_count - 1, 1);
    interval_velocity_deg_s(valid_pair) = ...
        angular_step_deg(valid_pair) ./ sample_interval_s(valid_pair);
    angular_velocity_deg_s(2:end) = interval_velocity_deg_s;

    valid_changes = diff(double([false; gaze_valid; false]));
    run_starts = find(valid_changes == 1);
    run_ends = find(valid_changes == -1) - 1;
    for run_idx = 1:numel(run_starts)
        start_idx = run_starts(run_idx);
        end_idx = run_ends(run_idx);

        % Guarded first: a single-sample run at the very end would put
        % start_idx + 1 past the array, so this must not be folded into
        % one condition that evaluates the index eagerly.
        if end_idx <= start_idx
            continue;
        end

        if isfinite(angular_velocity_deg_s(start_idx + 1))
            angular_velocity_deg_s(start_idx) = ...
                angular_velocity_deg_s(start_idx + 1);
        end
    end
end
