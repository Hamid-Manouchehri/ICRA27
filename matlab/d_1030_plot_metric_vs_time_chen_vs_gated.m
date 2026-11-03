%% Any logged metric vs time on one shared time grid, Chen vs Gated.
% Both trials are interpolated onto a common time grid (t_common).
%
% Inputs:
%   - data/raw_data/test_subjects/: one Chen and one Gated robot CSV
%     (names built in User Setting)
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - functions/common_time_interp_fun.m

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data', ...
    'test_subjects');

%% User Setting
subject_id = 11;  % 11, 14 to 23
shape = "ellipse";  % ellipse, pear, fine_gross
bandwidth = "narrow";  % narrow, wide
metric_column = "bd_applied_Ns_m";  % any CSV column, e.g. bd_desired_Ns_m
metric_label = 'b_d [N s/m]';  % y-axis label
dt_common_s = 0.002;  % common grid step [s], 0.002 = 500 Hz

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

[t_common_s, chen_common, gated_common] = common_time_interp_fun( ...
    chen.time_s, chen.(metric_column), gated.time_s, gated.(metric_column), ...
    dt_common_s);

figure('Name', sprintf('%s_vs_time_s%d_%s_%s', metric_column, subject_id, ...
    bandwidth, shape));
hold on;
plot(t_common_s, chen_common, 'Color', 'r');
plot(t_common_s, gated_common, 'Color', 'b');
xlabel('common time [s]');
ylabel(metric_label);
title(sprintf('s%d %s %s, %s', subject_id, bandwidth, shape, metric_column), ...
    'Interpreter', 'none');
legend(["Chen", "Gated"])
grid on;
