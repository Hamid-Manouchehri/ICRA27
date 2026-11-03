%% Paper eq. (10) output over time, Chen vs Gated, one subject/shape/bandwidth.
% Logged intended acceleration (eq. 10) and intended force (eq. 11) magnitudes.
%
% Inputs:
%   - data/raw_data/test_subjects/: one Chen and one Gated robot CSV
%     (names built in User Setting)
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - ur5e_chen_vac_2d.py, ur5e_gaze_assisted_chen_vac_2d.py (write the CSVs)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data', ...
    'test_subjects');

%% User Setting
subject_id = 11;  % 11, 14 to 23
shape = "ellipse";  % ellipse, pear, fine_gross
bandwidth = "narrow";  % narrow, wide

read_chen_csv = fullfile(raw_data_directory, sprintf( ...
    'ur5e_chen_vac_2d_%s_%s_shape_s%d.csv', bandwidth, shape, subject_id));
read_gated_csv = fullfile(raw_data_directory, sprintf( ...
    'ur5e_gaze_assisted_chen_vac_2d_held_out_%s_and_s%d_on_%s_%s_shape.csv', ...
    shape, subject_id, bandwidth, shape));

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\n', read_chen_csv, read_gated_csv);
fprintf('writing:\n  nothing (figure only)\n');

chen = readtable(read_chen_csv);
gated = readtable(read_gated_csv);

figure('Name', sprintf('optimization_output_s%d_%s_%s', subject_id, ...
    bandwidth, shape));

% eq. (10): intended acceleration
nexttile;
hold on;
plot(chen.time_s, hypot(chen.intended_acc_x_m_s2, chen.intended_acc_y_m_s2), ...
    'Color', 'r');
plot(gated.time_s, ...
    hypot(gated.intended_acc_x_m_s2, gated.intended_acc_y_m_s2), 'Color', 'b');
xlabel('time [s]');
ylabel('|a_{int}| [m/s^2]');
title(sprintf('s%d %s %s', subject_id, bandwidth, shape), ...
    'Interpreter', 'none');
legend(["Chen", "Gated"])
grid on;

% eq. (11): intended force = m a_int + (bd / gamma1) v
nexttile;
hold on;
plot(chen.time_s, hypot(chen.intended_force_x_N, chen.intended_force_y_N), ...
    'Color', 'r');
plot(gated.time_s, ...
    hypot(gated.intended_force_x_N, gated.intended_force_y_N), 'Color', 'b');
xlabel('time [s]');
ylabel('|F_{int}| [N]');
title(sprintf('s%d %s %s', subject_id, bandwidth, shape), ...
    'Interpreter', 'none');
legend(["Chen", "Gated"])
grid on;


figure;
plot(chen.time_s, chen.bd_applied_Ns_m);
hold on
plot(gated.time_s, gated.bd_applied_Ns_m);
legend(["Chen", "Gated"])

figure;
subplot(2,1,1)
plot(chen.time_s, chen.x_m)
xlabel("Chen x")
subplot(2,1,2)
plot(gated.time_s, gated.x_m)