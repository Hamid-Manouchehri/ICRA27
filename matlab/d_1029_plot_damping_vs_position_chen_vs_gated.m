%% Damping vs position along the centerline, Chen vs Gated, one subject.
% Same shape file and offset for both trials, so the position axis is shared.
% Raw samples, then each controller's damping at every centerline point
% (median over laps), then damping per lap, then the ROC of the Gated
% trial's P(Fine) against the geometric fine area.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (Chen and Gated trial)
%   - data/shapes/<shape_obj> (centerline)
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - functions/read_shape_band_obj.m
%   - functions/centerline_arclength_fun.m
%   - functions/lap_median_on_centerline_fun.m
%   - functions/lap_times_by_start_crossing_fun.m
%   - functions/fine_mask_fun.m

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
raw_data_directory = fullfile(data_directory, 'raw_data', 'test_subjects');
shape_directory = fullfile(data_directory, 'shapes');

subject_id = 14;  % 11, 14 to 23
shape = "fine_gross";  % ellipse, pear, fine_gross
bandwidth = "narrow";  % narrow, wide
centerline_spacing_m = 5e-4;  % shape resample spacing / sampling resolution [m]

% End User Setting
%% Main Body

fprintf(['reading:\n  %s\n  %s\\<robot_csv of Chen and Gated s%d %s %s>\n' ...
    '  %s\\<shape_obj>\n'], manifest_file, raw_data_directory, ...
    subject_id, bandwidth, shape, shape_directory);
fprintf('writing:\n  nothing (figure only)\n');

M = readtable(manifest_file, 'TextType', 'string');
M = M(M.subject == subject_id & M.shape == shape & M.bandwidth == bandwidth, :);
chen_row = M(M.controller == "Chen", :);
gated_row = M(M.controller == "Gated", :);

% Centerline in the CoppeliaSim frame (same shape_obj and offset for both).
[~, ~, center_xy_m] = read_shape_band_obj( ...
    fullfile(shape_directory, chen_row.shape_obj), centerline_spacing_m);
center_xy_m = center_xy_m + [chen_row.offset_x, chen_row.offset_y];

fprintf('Reading:\n  %s\n  %s\n', chen_row.robot_csv, gated_row.robot_csv);
chen = readtable(fullfile(raw_data_directory, chen_row.robot_csv));
gated = readtable(fullfile(raw_data_directory, gated_row.robot_csv));

chen_position_m = centerline_arclength_fun(chen.coppeliasim_x_m, ...
    chen.coppeliasim_y_m, center_xy_m);
gated_position_m = centerline_arclength_fun(gated.coppeliasim_x_m, ...
    gated.coppeliasim_y_m, center_xy_m);

% Damping of each trial at every centerline point, median over its laps.
[chen_bd_centerline_n_s_m, grid_arclength_m, chen_gap_m] = ...
    lap_median_on_centerline_fun(chen.coppeliasim_x_m, ...
    chen.coppeliasim_y_m, chen.bd_applied_Ns_m, center_xy_m);
[gated_bd_centerline_n_s_m, ~, gated_gap_m] = ...
    lap_median_on_centerline_fun(gated.coppeliasim_x_m, ...
    gated.coppeliasim_y_m, gated.bd_applied_Ns_m, center_xy_m);

% Check: should be a few mm (the TCP's offset from the centerline).
fprintf(['median distance, centerline point to its sample: ' ...
    'Chen %.1f mm, Gated %.1f mm\n'], 1000 * chen_gap_m, 1000 * gated_gap_m);

figure;
plot(center_xy_m(:,1), center_xy_m(:,2))
xlabel('x [m]');
ylabel('y [m]');
legend("centerline")

figure('Name', sprintf('damping_vs_position_s%d_%s_%s', subject_id, ...
    bandwidth, shape));

% raw samples
% nexttile;
% hold on;
% plot(chen_position_m, chen.bd_applied_Ns_m, '.', 'Color', 'r');
% plot(gated_position_m, gated.bd_applied_Ns_m, '.', 'Color', 'b');
% xlabel('position along centerline [m]');
% ylabel('b_d [N s/m]');
% title(sprintf('s%d %s %s, all samples', subject_id, bandwidth, shape), ...
%     'Interpreter', 'none');
% legend(["Chen", "Gated"])
% grid on;

% one value per centerline point and controller
nexttile;
hold on;
plot(grid_arclength_m, chen_bd_centerline_n_s_m, 'Color', 'r');
plot(grid_arclength_m, gated_bd_centerline_n_s_m, 'Color', 'b');
xlabel('position along centerline [m]');
ylabel('median b_d over laps [N s/m]');
title(sprintf('s%d %s %s, at each centerline point', subject_id, ...
    bandwidth, shape), 'Interpreter', 'none');
legend(["Chen", "Gated"])
grid on;

figure;

plot(chen.x_m, chen.bd_applied_Ns_m,'r.')
hold on
plot(gated.x_m, gated.bd_applied_Ns_m,'ko')
xlabel('x [m]');
ylabel('b_d [N s/m]');
legend(["Chen", "Gated"])

chen_id1 = chen.y_m < -0.65;
chen_id2 = chen.y_m > -0.65;
gated_id1 = gated.y_m < -0.65;
gated_id2 = gated.y_m > -0.65;

figure;
subplot(3,1,1)
plot(chen.x_m(chen_id1), chen.bd_applied_Ns_m(chen_id1),'r.')
hold on
plot(gated.x_m(gated_id1), gated.bd_applied_Ns_m(gated_id1),'k.')
xlabel('x [m]');
ylabel('b_d [N s/m]');
legend(["Chen, y < -0.65 m", "Gated, y < -0.65 m"])

subplot(3,1,2)
plot(chen.x_m(chen_id2), chen.bd_applied_Ns_m(chen_id2),'r.')
hold on
plot(gated.x_m(gated_id2), gated.bd_applied_Ns_m(gated_id2),'k.')
xlabel('x [m]');
ylabel('b_d [N s/m]');
legend(["Chen, y > -0.65 m", "Gated, y > -0.65 m"])

subplot(3,1,3)
plot(gated.x_m(gated_id1), gated.fine_probability(gated_id1),'r.')
hold on
plot(gated.x_m(gated_id2), gated.fine_probability(gated_id2),'g.')
xlabel('x [m]');
ylabel('P(Fine)');
legend(["Gated, y < -0.65 m", "Gated, y > -0.65 m"])

% Same rows as the figure above, one column per lap. Laps from crossings of
% the start x.
[chen_lap_start_s, chen_lap_stop_s] = lap_times_by_start_crossing_fun( ...
    chen.time_s, chen.coppeliasim_x_m, chen.coppeliasim_y_m);
[gated_lap_start_s, gated_lap_stop_s] = lap_times_by_start_crossing_fun( ...
    gated.time_s, gated.coppeliasim_x_m, gated.coppeliasim_y_m);

% Lap number of each sample (NaN outside complete laps).
chen_lap = discretize(chen.time_s, [chen_lap_start_s; chen_lap_stop_s(end)]);
gated_lap = discretize(gated.time_s, ...
    [gated_lap_start_s; gated_lap_stop_s(end)]);
n_laps = max(numel(chen_lap_start_s), numel(gated_lap_start_s));

figure('Name', sprintf('damping_per_lap_s%d_%s_%s', subject_id, ...
    bandwidth, shape));
for k = 1:n_laps
    chen_k1 = chen_id1 & chen_lap == k;
    chen_k2 = chen_id2 & chen_lap == k;
    gated_k1 = gated_id1 & gated_lap == k;
    gated_k2 = gated_id2 & gated_lap == k;

    subplot(3, n_laps, k)
    plot(chen.x_m(chen_k1), chen.bd_applied_Ns_m(chen_k1),'r.')
    hold on
    plot(gated.x_m(gated_k1), gated.bd_applied_Ns_m(gated_k1),'k.')
    title(sprintf('lap %d', k));

    subplot(3, n_laps, n_laps + k)
    plot(chen.x_m(chen_k2), chen.bd_applied_Ns_m(chen_k2),'r.')
    hold on
    plot(gated.x_m(gated_k2), gated.bd_applied_Ns_m(gated_k2),'k.')

    subplot(3, n_laps, 2 * n_laps + k)
    plot(gated.x_m(gated_k1), gated.fine_probability(gated_k1),'r.')
    hold on
    plot(gated.x_m(gated_k2), gated.fine_probability(gated_k2),'g.')
    xlabel('x [m]');
end

% y labels and legends on the first column only
subplot(3, n_laps, 1)
ylabel('b_d [N s/m]');
legend(["Chen, y < -0.65 m", "Gated, y < -0.65 m"])
subplot(3, n_laps, n_laps + 1)
ylabel('b_d [N s/m]');
legend(["Chen, y > -0.65 m", "Gated, y > -0.65 m"])
subplot(3, n_laps, 2 * n_laps + 1)
ylabel('P(Fine)');
legend(["Gated, y < -0.65 m", "Gated, y > -0.65 m"])

% ROC of P(Fine) over this trial's samples. The test trials have no hand
% labels, so the reference is the geometric fine area (fine_mask_fun).
gated_is_fine = fine_mask_fun(gated.coppeliasim_x_m, ...
    find(["fine_gross", "pear", "ellipse"] == shape));
[roc_fpr, roc_tpr, ~, roc_auc] = perfcurve(gated_is_fine, ...
    gated.fine_probability, true);

figure('Name', sprintf('roc_p_fine_s%d_%s_%s', subject_id, bandwidth, shape));
plot(roc_fpr, roc_tpr, 'b')
hold on
plot([0 1], [0 1], 'k--')
axis square
xlabel('false positive rate (Gross samples called Fine)');
ylabel('true positive rate (Fine samples called Fine)');
title(sprintf('s%d %s %s, P(Fine) vs fine area, AUC = %.2f', subject_id, ...
    bandwidth, shape, roc_auc), 'Interpreter', 'none');
legend(["Gated", "chance"])
grid on;
