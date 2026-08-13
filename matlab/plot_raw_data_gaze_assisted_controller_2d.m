clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO Select the fixation-transition CSV to analyze.

csv_file = fullfile( ...
    script_directory, '..', 'data', 'raw_data', ...
    'ur5e_gaze_assisted_adm_2d_protractor_test.csv');

% Must match Python controller parameters
fixed_damping = 200.0;          % N.s/m
intent_forward_damping = 15.0; % N.s/m

% Must match Python fixation_velocity_threshold_deg_s
fixation_velocity_threshold_deg_s = 30.0; % deg/s

% Plot every Nth active intent arrow to avoid clutter
intent_arrow_stride = 100;

% Display length of intent direction arrows [m]
intent_arrow_length_m = 0.025;

%% Load data
data = readtable(csv_file);

t = data.time_s;

%% Useful logical masks
fixation_valid = data.fixation_valid == 1;
intent_active = data.intent_active == 1;
cbf_active = data.cbf_modified_velocity == 1;

%% ================================================================
% Figure 1: TCP trajectory + fixation points + gaze intent direction
% =================================================================

figure('Name', 'TCP trajectory and gaze intent', 'Color', 'w');
hold on;

% TCP path
plot(data.tcp_x_m, data.tcp_y_m, 'k-', 'LineWidth', 1.5, 'DisplayName', ...
    'TCP trajectory');

% Valid fixation positions
scatter(data.fixation_x_robot_m(fixation_valid), ...
    data.fixation_y_robot_m(fixation_valid), 10, 'filled', 'DisplayName', ...
    'Valid fixation');

% Select only some active intent samples for arrows
active_indices = find(intent_active);

if ~isempty(active_indices)

    arrow_indices = active_indices(1:intent_arrow_stride:end);

    % Fixed-length arrows showing direction only
    arrow_dx = intent_arrow_length_m .* ...
        data.intent_direction_x(arrow_indices);

    arrow_dy = intent_arrow_length_m .* ...
        data.intent_direction_y(arrow_indices);

    quiver(data.tcp_x_m(arrow_indices), data.tcp_y_m(arrow_indices), ...
        arrow_dx, arrow_dy, 0, 'LineWidth', 1.2, 'DisplayName', ...
        'Gaze intent direction');
end

axis equal;
grid on;
xlabel('Robot base x [m]');
ylabel('Robot base y [m]');
title('TCP trajectory, fixation targets, and gaze intent');
legend('Location', 'best');


%% ================================================================
% Figure 2: Fixation confidence, influence, and effective damping
% =================================================================

% Effective damping in the FORWARD gaze-intent direction:
%
% B_eff = (1-influence)*B_fixed ...
%       + influence*B_forward

effective_forward_damping = (1 - data.intent_influence) .* fixed_damping + ...
    data.intent_influence .* intent_forward_damping;

figure('Name', 'Gaze confidence and damping modulation', 'Color', 'w');

tiledlayout(3,1);

% ---- confidence
nexttile;
plot(t, data.intent_confidence, 'LineWidth', 1.2);

hold on;

plot(t, data.intent_influence, 'LineWidth', 1.2);

ylim([-0.05 1.05]);
grid on;

ylabel('Value [-]');
title('Gaze confidence and controller influence');

legend('Intent confidence', 'Intent influence', 'Location', 'best');


% ---- intent active
nexttile;

stairs(t, double(intent_active), 'LineWidth', 1.2);

ylim([-0.1 1.1]);
yticks([0 1]);
yticklabels({'Inactive','Active'});

grid on;

ylabel('Intent');
title('Gaze-intent activation');


% ---- effective damping
nexttile;

plot(t, effective_forward_damping, 'LineWidth', 1.3);

hold on;

yline(fixed_damping, '--', 'Fixed damping');

yline(intent_forward_damping, '--', 'Minimum forward damping');

grid on;

xlabel('Time [s]');
ylabel('B_{forward} [N s/m]');
title('Effective damping along gaze-intent direction');


%% ================================================================
% Figure 3: Human force and TCP velocity along gaze-intent direction
% =================================================================

% Projection of force onto intent direction:
%
% F_parallel = F dot g_hat

force_parallel_n = data.Fx_filtered_N .* data.intent_direction_x + ...
    data.Fy_filtered_N .* data.intent_direction_y;

% Projection of actual TCP velocity onto intent direction:
%
% v_parallel = v dot g_hat

velocity_parallel_m_s = data.tcp_vx_actual_m_s .* data.intent_direction_x + ...
    data.tcp_vy_actual_m_s .* data.intent_direction_y;

% Do not interpret projection when gaze intent is inactive
force_parallel_n(~intent_active) = NaN;
velocity_parallel_m_s(~intent_active) = NaN;

figure('Name', 'Force and velocity along gaze intent', 'Color', 'w');

yyaxis left;

plot(t, force_parallel_n, 'LineWidth', 1.2);

ylabel('F_{||} [N]');

yline(0, ':');

yyaxis right;

plot(t, velocity_parallel_m_s, 'LineWidth', 1.2);

ylabel('v_{||} [m/s]');

yline(0, ':');

grid on;
xlabel('Time [s]');

title('Force and TCP velocity projected onto gaze-intent direction');

legend('F_{\parallel}', 'v_{\parallel}', 'Location', 'best');


%% ================================================================
% Figure 4: CBF intervention
% =================================================================

figure('Name', 'CBF velocity modification', 'Color', 'w');

tiledlayout(3,1);

% ---- X velocity
nexttile;

plot(t, data.vx_nominal_m_s, 'LineWidth', 1.1);

hold on;

plot(t, data.vx_cbf_m_s, '--', 'LineWidth', 1.1);

grid on;

ylabel('v_x [m/s]');
title('Nominal versus CBF-safe x velocity');

legend('Nominal', 'After CBF', 'Location', 'best');


% ---- Y velocity
nexttile;

plot(t, data.vy_nominal_m_s, 'LineWidth', 1.1);

hold on;

plot(t, data.vy_cbf_m_s, '--', 'LineWidth', 1.1);

grid on;

ylabel('v_y [m/s]');
title('Nominal versus CBF-safe y velocity');

legend('Nominal', 'After CBF', 'Location', 'best');


% ---- CBF activation
nexttile;

stairs(t, double(cbf_active), 'LineWidth', 1.2);

ylim([-0.1 1.1]);
yticks([0 1]);
yticklabels({'Inactive','Modified'});

grid on;

xlabel('Time [s]');
ylabel('CBF');
title('CBF intervention');


%% ================================================================
% Figure 5: Spatial map of effective damping along the traced path
% =================================================================
% Tests whether "light" (low damping, gaze-assisted) and "heavy"
% (fixed damping) segments correlate with path geometry, e.g. the
% straight line versus the half-circle in the protractor shape.

figure('Name', 'Spatial damping map', 'Color', 'w');

scatter(data.tcp_x_m, data.tcp_y_m, 10, effective_forward_damping, 'filled');

axis equal;
grid on;
damping_colorbar = colorbar;
damping_colorbar.Label.String = "effecive forward damping";

xlabel('Robot base x [m]');
ylabel('Robot base y [m]');
title('TCP path colored by effective forward damping [N s/m]');


%% ================================================================
% Figure 6: Fixation angular velocity, duration, and I-VT threshold
% =================================================================
% If gaze angular velocity stays above fixation_velocity_threshold_deg_s
% for long stretches (e.g. while visually pursuing the curved arc), the
% I-VT classifier cannot register a fixation there, so intent stays
% inactive and damping stays at the fixed (heavy) value regardless of
% actual user intent. fixation_duration_s resetting every few hundred
% milliseconds (a sawtooth pattern) shows confidence rebuilding from
% zero at the start of every new fixation, even during continuous
% pursuit of the path.

figure('Name', 'Fixation angular velocity, duration, and I-VT threshold', ...
    'Color', 'w');

tiledlayout(3,1);

nexttile;
plot(t, data.fixation_angular_velocity_deg_s, 'LineWidth', 1.0);

hold on;

yline(fixation_velocity_threshold_deg_s, 'r--', 'I-VT threshold');

grid on;
ylabel('Gaze angular velocity [deg/s]');
title('Gaze angular velocity relative to the fixation threshold');

nexttile;
plot(t, data.fixation_duration_s, 'LineWidth', 1.0);

grid on;
ylabel('Fixation duration [s]');
title('Retained fixation duration (resets at each new fixation)');

nexttile;
stairs(t, double(intent_active), 'LineWidth', 1.2);

ylim([-0.1 1.1]);
yticks([0 1]);
yticklabels({'Inactive','Active'});

grid on;
xlabel('Time [s]');
ylabel('Intent');
title('Gaze-intent activation (for comparison)');


%% ================================================================
% Print useful summary
% =================================================================

fprintf('\n--- Gaze-assisted controller summary ---\n');

fprintf('Duration: %.2f s\n', t(end) - t(1));

fprintf('Valid fixation samples: %.1f %%\n', 100 * mean(fixation_valid));

fprintf('Active gaze-intent samples: %.1f %%\n', 100 * mean(intent_active));

fprintf('CBF modified samples: %d / %d\n', nnz(cbf_active), height(data));

fprintf('Minimum effective forward damping: %.2f N.s/m\n', ...
    min(effective_forward_damping));

fprintf('Maximum effective forward damping: %.2f N.s/m\n', ...
    max(effective_forward_damping));

% Duration of each contiguous gaze-intent-active window. Frequent short
% windows (comparable to minimum_fixation_duration_s) indicate the
% confidence-rebuild-per-fixation cycling described in Figure 6; long,
% rare windows separated by long inactive gaps suggest the I-VT
% classifier is not detecting fixations during parts of the path (e.g.
% the arc).

active_run_lengths_s = [];
in_active_run = false;
active_run_start_s = NaN;

for sample_index = 1:height(data)

    if intent_active(sample_index) && ~in_active_run
        in_active_run = true;
        active_run_start_s = t(sample_index);
    elseif ~intent_active(sample_index) && in_active_run
        in_active_run = false;
        run_length_s = t(sample_index) - active_run_start_s;
        active_run_lengths_s(end+1) = run_length_s; %#ok<AGROW>
    end

end

if ~isempty(active_run_lengths_s)
    fprintf(['Gaze-intent active windows: %d, mean %.2f s, ' ...
        'min %.2f s, max %.2f s\n'], ...
        numel(active_run_lengths_s), mean(active_run_lengths_s), ...
        min(active_run_lengths_s), max(active_run_lengths_s));
else
    fprintf('Gaze-intent active windows: 0\n');
end
