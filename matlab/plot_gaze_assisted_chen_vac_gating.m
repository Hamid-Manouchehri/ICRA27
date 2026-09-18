%% Classifier-gating diagnostics for one ur5e_gaze_assisted_chen_vac_2d.py trial.
%
% Inputs:
%   data/raw_data/<robot_csv_file>   controller log, selected in User
%     Settings. Must have the classifier-gating columns (present whether
%     enable_classifier_gating was True or False for that run).
%
% Outputs:
%   Figure 1: TCP path colored by P(Fine) (falls back to coloring by
%     applied damping if gating was off or every probability is NaN).
%   Figure 2 (4 stacked tiles, shared time axis): force magnitude;
%     P(Fine) with non-"ok" classifier_status ticks marked; the gated
%     [bd_min, bd_max] band with bd_desired/bd_applied overlaid and the
%     fixed Fine/Gross reference lines; probability age against the
%     staleness threshold.
%   A numeric summary printed to the command window: status breakdown,
%     P(Fine)/compute-time/age statistics, how often the age fallback
%     fired, whether bd_applied ever left its own tick's [bd_min, bd_max]
%     band, and mean/std of every logged classifier feature.
%   No file is written; this script only reads the CSV above.
%
% Assumptions:
%   The CSV was written by scripts/ur5e_gaze_assisted_chen_vac_2d.py, so
%   column names match that script's write_csv_header(). Feature columns
%   are whichever ones follow bd_max_Ns_m, since the set depends on
%   which model bundle produced the trial.
%
% Method reuse:
%   Self-contained here per this project's convention that MATLAB
%   scripts do not call one another.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO controller log written by scripts/ur5e_gaze_assisted_chen_vac_2d.py.
robot_csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', 'test_subjects', ...
    'ur5e_gaze_assisted_chen_vac_2d_held_out_pear_and_s19_on_wide_pear_shape.csv');

% TODO damping bounds [N s/m] per class, must match the Python
% controller's fine_damping_bounds_n_s_m / gross_damping_bounds_n_s_m --
% drawn as reference lines showing the two pairs bd_min/bd_max
% interpolate between.
fine_damping_bounds_n_s_m = [80.0, 200.0];
gross_damping_bounds_n_s_m = [10.0, 100.0];

% TODO must match the Python controller's max_probability_age_s -- drawn
% as the staleness threshold on the probability-age panel.
max_probability_age_s = 2.0;

% TODO numeric tolerance [N s/m] for the bd_applied-inside-band check,
% since the rate limiter chasing a MOVING bound can transiently lag
% outside it by a hair even when everything is working correctly.
band_excursion_tolerance_n_s_m = 1e-2;

%% Load
robot_table = readtable(robot_csv_file, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

time_s = robot_table.time_s;
force_magnitude_n = hypot(robot_table.Fx_filt_N, robot_table.Fy_filt_N);
coppeliasim_x_m = robot_table.coppeliasim_x_m;
coppeliasim_y_m = robot_table.coppeliasim_y_m;

bd_desired_n_s_m = robot_table.bd_desired_Ns_m;
bd_applied_n_s_m = robot_table.bd_applied_Ns_m;
bd_min_n_s_m = robot_table.bd_min_Ns_m;
bd_max_n_s_m = robot_table.bd_max_Ns_m;

gating_enabled = robot_table.enable_classifier_gating(1) == 1;
fine_probability = robot_table.fine_probability;
probability_age_s = robot_table.fine_probability_age_s;
classifier_status = robot_table.classifier_status;
classifier_compute_ms = robot_table.classifier_compute_ms;
have_probability = any(~isnan(fine_probability));

% Feature columns are whichever ones follow bd_max_Ns_m -- the set
% depends on which model bundle produced this trial, so it is read from
% the header rather than hardcoded.
all_column_names = robot_table.Properties.VariableNames;
bd_max_column_idx = find(strcmp(all_column_names, 'bd_max_Ns_m'), 1);
feature_column_names = all_column_names(bd_max_column_idx + 1:end);

%% ================================================================
% Figure 1: TCP path colored by P(Fine)
% =================================================================

gross_color = [0.20, 0.50, 0.80];
fine_color = [0.85, 0.20, 0.20];
probability_colormap = [ ...
    linspace(gross_color(1), fine_color(1), 256)', ...
    linspace(gross_color(2), fine_color(2), 256)', ...
    linspace(gross_color(3), fine_color(3), 256)'];

figure('Name', 'TCP path colored by P(Fine)', 'Color', 'w');
hold on;

if have_probability
    scatter(coppeliasim_x_m, coppeliasim_y_m, 10, fine_probability, ...
        'filled', 'DisplayName', 'TCP path');
    colormap(gca, probability_colormap);
    clim([0, 1]);
    probability_colorbar = colorbar;
    probability_colorbar.Label.String = 'P(Fine)  [blue = Gross, red = Fine]';
    title('TCP path colored by predicted P(Fine)');
else
    scatter(coppeliasim_x_m, coppeliasim_y_m, 10, bd_applied_n_s_m, ...
        'filled', 'DisplayName', 'TCP path');
    clim([min(bd_applied_n_s_m), max(bd_applied_n_s_m)]);
    damping_colorbar = colorbar;
    damping_colorbar.Label.String = 'Applied damping b_d [N s/m]';
    title(['TCP path colored by applied damping (gating was off or ' ...
        'every probability is NaN, so P(Fine) is unavailable)']);
end

axis equal;
grid on;
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
legend('Location', 'best', 'FontSize', 8);

%% ================================================================
% Figure 2: force, P(Fine), gated damping bounds, probability age
% =================================================================

figure('Name', 'Classifier gating over time', 'Color', 'w');
tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---- Tile 1: force (context) ----
force_axes = nexttile;
plot(time_s, force_magnitude_n, '-', 'LineWidth', 1.0);
grid on;
ylabel('|F| [N]');
title('Human force magnitude');

% ---- Tile 2: P(Fine), non-"ok" status ticks marked ----
probability_axes = nexttile;
hold on;
if have_probability
    plot(time_s, fine_probability, '-', 'LineWidth', 1.0, ...
        'DisplayName', 'P(Fine)');
    yline(0.5, 'k--', 'HandleVisibility', 'off');

    is_not_ok = classifier_status ~= "ok";
    if any(is_not_ok)
        scatter(time_s(is_not_ok), fine_probability(is_not_ok), 10, ...
            [0.6, 0.6, 0.6], 'filled', 'MarkerFaceAlpha', 0.5, ...
            'DisplayName', 'Non-"ok" status');
    end
    legend('Location', 'best', 'FontSize', 7);
else
    text(0.5, 0.5, 'Classifier gating was off for this trial', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
end
grid on;
ylim([-0.05, 1.05]);
ylabel('P(Fine) [-]');
title('Predicted probability, with non-"ok" status ticks marked');

% ---- Tile 3: gated damping bounds ----
bounds_axes = nexttile;
hold on;
fill([time_s; flipud(time_s)], [bd_min_n_s_m; flipud(bd_max_n_s_m)], ...
    [0.85, 0.85, 0.85], 'FaceAlpha', 0.6, 'EdgeColor', 'none', ...
    'DisplayName', 'Gated [b_{min}, b_{max}]');
plot(time_s, bd_desired_n_s_m, '-', 'LineWidth', 1.0, ...
    'DisplayName', 'b_d desired');
plot(time_s, bd_applied_n_s_m, '-', 'LineWidth', 1.3, ...
    'DisplayName', 'b_d applied');
yline(fine_damping_bounds_n_s_m(1), '--', 'Color', fine_color, ...
    'HandleVisibility', 'off');
yline(fine_damping_bounds_n_s_m(2), '--', 'Fine [L_f, U_f]', ...
    'Color', fine_color, 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 7, 'HandleVisibility', 'off');
yline(gross_damping_bounds_n_s_m(1), '--', 'Gross [L_g, U_g]', ...
    'Color', gross_color, 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 7, 'HandleVisibility', 'off');
yline(gross_damping_bounds_n_s_m(2), '--', 'Color', gross_color, ...
    'HandleVisibility', 'off');
grid on;
ylabel('b_d [N s/m]');
title('Gated bounds vs. desired/applied damping');
legend('Location', 'best', 'FontSize', 7);

% ---- Tile 4: probability age ----
age_axes = nexttile;
%{
if have_probability
    plot(time_s, probability_age_s, '-', 'LineWidth', 1.0);
    hold on;
    yline(max_probability_age_s, 'r--', 'Fallback threshold');
    grid on;
    ylabel('Age [s]');
else
    text(0.5, 0.5, 'Classifier gating was off for this trial', ...
        'Units', 'normalized', 'HorizontalAlignment', 'center');
    grid on;
end
xlabel('Time [s]');
title('P(Fine) age (time since the last successful classifier update)');

linkaxes([force_axes, probability_axes, bounds_axes, age_axes], 'x');
%}
plot(robot_table.x_m,robot_table.bd_applied_Ns_m,'.k')

%{
plot(time_s,robot_table.x_m)
hold on
plot(time_s,0.05+0*robot_table.x_m,'k')

xlim(force_axes, [time_s(1), time_s(end)]);
%}
%% ================================================================
% Summary
% =================================================================

trial_duration_s = time_s(end) - time_s(1);

fprintf('\n--- Classifier gating trial summary ---\n');
fprintf('File: %s\n', robot_csv_file);
fprintf('Duration: %.2f s over %d samples\n', ...
    trial_duration_s, height(robot_table));
if gating_enabled
    fprintf('Classifier gating: true\n');
else
    fprintf('Classifier gating: false\n');
end

if ~have_probability
    fprintf(['No non-NaN P(Fine) in this trial -- gating was off, or ' ...
        'every classifier update failed. Nothing further to summarize.\n']);
else
    fprintf('\nClassifier status breakdown:\n');
    unique_statuses = unique(classifier_status);
    for status_idx = 1:numel(unique_statuses)
        this_status = unique_statuses(status_idx);
        status_count = sum(classifier_status == this_status);
        fprintf('  %-20s %6d  (%.1f %%)\n', this_status, status_count, ...
            100 * status_count / height(robot_table));
    end

    valid_probability = fine_probability(~isnan(fine_probability));
    fprintf('\nP(Fine): mean %.3f, median %.3f, range [%.3f, %.3f]\n', ...
        mean(valid_probability), median(valid_probability), ...
        min(valid_probability), max(valid_probability));

    fprintf('Classifier compute time: mean %.2f ms, median %.2f ms, max %.2f ms\n', ...
        mean(classifier_compute_ms(classifier_compute_ms > 0)), ...
        median(classifier_compute_ms(classifier_compute_ms > 0)), ...
        max(classifier_compute_ms));

    valid_age_s = probability_age_s(~isnan(probability_age_s));
    stale_count = sum(probability_age_s > max_probability_age_s);
    if isempty(valid_age_s)
        fprintf(['Probability age: no successful classifier update in this ' ...
            'trial -- the classifier thread never once published.\n']);
    else
        fprintf(['Probability age: mean %.3f s, median %.3f s, max %.3f s ' ...
            '-- exceeded %.1f s threshold on %d samples (%.1f %%)\n'], ...
            mean(valid_age_s), median(valid_age_s), max(valid_age_s), ...
            max_probability_age_s, stale_count, ...
            100 * stale_count / height(robot_table));
    end

    % bd_applied is checked against THIS TICK's own bd_min/bd_max, since
    % the band itself moves -- a fixed-band check would be meaningless.
    excursion_mask = ...
        bd_applied_n_s_m < bd_min_n_s_m - band_excursion_tolerance_n_s_m | ...
        bd_applied_n_s_m > bd_max_n_s_m + band_excursion_tolerance_n_s_m;
    if any(excursion_mask)
        below_excursion_n_s_m = max( ...
            bd_min_n_s_m(excursion_mask) - bd_applied_n_s_m(excursion_mask));
        above_excursion_n_s_m = max( ...
            bd_applied_n_s_m(excursion_mask) - bd_max_n_s_m(excursion_mask));
        fprintf(['\nb_d applied left its own tick''s [b_min, b_max] band on ' ...
            '%d samples (%.1f %%), worst excursion %.2f N s/m -- expected ' ...
            'transiently while the rate limiter chases a moving band.\n'], ...
            sum(excursion_mask), 100 * mean(excursion_mask), ...
            max(below_excursion_n_s_m, above_excursion_n_s_m));
    else
        fprintf('\nb_d applied stayed inside its own tick''s [b_min, b_max] band throughout.\n');
    end

    fprintf('\nClassifier features (%d):\n', numel(feature_column_names));
    for feature_idx = 1:numel(feature_column_names)
        feature_name = feature_column_names{feature_idx};
        feature_values = robot_table.(feature_name);
        fprintf('  %-32s mean %10.4g  std %10.4g\n', feature_name, ...
            mean(feature_values, 'omitnan'), std(feature_values, 'omitnan'));
    end
end
fprintf('\n');
