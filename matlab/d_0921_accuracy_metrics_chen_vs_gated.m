clc; 
clear; 
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
% One trial pair, Chen baseline vs gaze-assisted gated. Edit these paths
% to look at a different subject/shape/bandwidth.
chen_csv  = fullfile(data_directory, 'raw_data', 'test_subjects', ...
    'ur5e_chen_vac_2d_wide_fine_gross_shape_s19.csv');
gated_csv = fullfile(data_directory, 'raw_data', 'test_subjects', ...
    'ur5e_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_s19_on_wide_fine_gross_shape.csv');

chen_raw_data = readtable(chen_csv);
gated_raw_data = readtable(gated_csv);

raw_data = {chen_raw_data, gated_raw_data};

shape_obj = fullfile(data_directory, 'shapes', 'wide_fine_gross_shape.obj');
offset = [-0.0, -0.650];  % fine_gross shape, CAD -> CoppeliaSim frame

write_metrics_mat = fullfile(data_directory, 'processed_data', ...
    'tracking_metrics_chen_vs_gated_wide_fine_gross_shape_s19.mat');

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\n  %s\n', chen_csv, gated_csv, shape_obj);
fprintf('writing:\n  %s\n', write_metrics_mat);

[outer, inner, center] = read_shape_band_obj(shape_obj, 5e-4);
center = center + offset;
outer  = outer + offset;
inner  = inner + offset;
half_w = hypot(outer(:,1)-center(:,1), outer(:,2)-center(:,2));  % local corridor half-width

% figure; plot(inner(:,1), inner(:,2))
% hold on;
% plot(outer(:,1), outer(:,2))
% hold on;
% plot(center(:,1), center(:,2), "-.")

names = {'Chen', 'Gated'};
stride = 200;  % plot every Nth point so the quiver plot stays readable
dist_to_centerline_m = cell(1,2);
angular_divergence_rad = cell(1,2);

for k = 1:2
    % load the required raw data:
    T = raw_data{k};
    x = T.coppeliasim_x_m;
    y = T.coppeliasim_y_m;
    vx = T.tcp_vx_actual_m_s;
    vy = T.tcp_vy_actual_m_s;
    t = T.time_s;

    [d, path_angle, centerline_position] = nearest_centerline_distance_and_heading(x, y, center);

    tcp_angle = atan2(vy, vx);  % TCP pose taken as direction of travel (no orientation column logged)
    ang_div = signed_angle_difference(tcp_angle, path_angle);

    dist_to_centerline_m{k} = d;
    angular_divergence_rad{k} = ang_div;

    speed = hypot(vx, vy);
    path_length_m = sum(hypot(diff(x), diff(y)));

    d_stats = summarize_distribution(1e3 * d);  % [mm]
    ang_stats = summarize_distribution(abs(rad2deg(ang_div)));  % [deg]

    d_norm = normalized_corridor_distance(d, centerline_position, half_w);
    d_norm_stats = summarize_distribution(d_norm);  % fraction of corridor half-width

    in_band_percent = in_band_accuracy(d_norm);
    violation = boundary_violation_magnitude(d_norm);

    d_rate_mm_s = 1e3 * rate_of_change(d, t);
    ang_rate_deg_s = rad2deg(rate_of_change(ang_div, t));
    ang_reversal_rate_per_m = sign_reversal_rate_per_meter(ang_div, path_length_m);

    d_speed_corr = pearson_correlation(d, speed);
    ang_speed_corr = pearson_correlation(abs(ang_div), speed);

    ang_bias_deg = angular_divergence_bias(ang_div);

    % Kept per arm so the summary bar figure after the loop can compare
    % every printed metric side by side.
    arm(k).d_stats = d_stats;
    arm(k).ang_stats = ang_stats;
    arm(k).d_norm_stats = d_norm_stats;
    arm(k).in_band_percent = in_band_percent;
    arm(k).violation = violation;
    arm(k).d_rate_mean_abs = mean(abs(d_rate_mm_s));
    arm(k).d_rate_std = std(d_rate_mm_s);
    arm(k).ang_rate_mean_abs = mean(abs(ang_rate_deg_s));
    arm(k).ang_rate_std = std(ang_rate_deg_s);
    arm(k).ang_reversal_rate_per_m = ang_reversal_rate_per_m;
    arm(k).d_speed_corr = d_speed_corr;
    arm(k).ang_speed_corr = ang_speed_corr;
    arm(k).ang_bias_deg = ang_bias_deg;

    % figures
    figure; % TCP + path trajectory
    plot(outer(:,1), outer(:,2), 'k-', 'DisplayName', 'outer'); hold on;
    plot(inner(:,1), inner(:,2), 'k-', 'DisplayName', 'inner');
    plot(center(:,1), center(:,2), '-.', 'DisplayName', 'centerline')
    plot(x, y, '.', 'DisplayName', 'TCP path');
    quiver(x(1:stride:end), y(1:stride:end), cos(tcp_angle(1:stride:end)), sin(tcp_angle(1:stride:end)), 0.5, 'r', 'DisplayName', 'actual heading');
    quiver(x(1:stride:end), y(1:stride:end), cos(path_angle(1:stride:end)), sin(path_angle(1:stride:end)), 0.5, 'b', 'DisplayName', 'desired heading');
    axis equal;
    title(names{k});
    legend;

    figure;  % point-wise dist to center-line
    subplot(1,2,1);
    plot(d);
    legend("point-wise dist to center-line");
    subplot(1,2,2);
    stem3_over_shape(x, y, d, center, stride, 'dist to centerline [m]');

    figure;  % point-wise angular divergence
    subplot(1,2,1);
    plot(ang_div)
    legend("point-wise angular divergence (rad)");
    subplot(1,2,2);
    stem3_over_shape(x, y, ang_div, center, stride, 'angular divergence [rad]');

    figure;  % normalized dist to center-line
    subplot(1,2,1);
    plot(d_norm); yline(1.0, '--', 'corridor boundary');
    legend("normalized dist to centerline (frac of half-width)");
    subplot(1,2,2);
    stem3_over_shape(x, y, d_norm, center, stride, 'normalized dist (frac of half-width)');

    figure;  % time-derivative / rate of distance to center-line
    subplot(1,2,1);
    plot(d_rate_mm_s); ylabel('d(dist)/dt [mm/s]');
    subplot(1,2,2);
    stem3_over_shape(x, y, d_rate_mm_s, center, stride, 'd(dist)/dt [mm/s]');
    
    figure;  % time-derivative / rate of angular divergence
    subplot(1,2,1);
    plot(ang_rate_deg_s); ylabel('d(angular divergence)/dt [deg/s]');
    subplot(1,2,2);
    stem3_over_shape(x, y, ang_rate_deg_s, center, stride, 'd(ang div)/dt [deg/s]');
end

%% Summary: every printed metric as a Chen vs gated bar comparison.
% Grouped by unit so each tile's y-axis stays meaningful -- these
% metrics span [%], [mm], [deg], [1/m] and dimensionless ratios, which
% cannot share one axis. One trial per arm, so there is no variability
% to show; the bars are single observations, not group means.
chen_color = [0.20 0.40 0.80];
gated_color = [0.85 0.35 0.10];

figure('Name', 'tracking_metrics_chen_vs_gated_summary', 'Color', 'w');
summary_layout = tiledlayout(3, 4, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
title(summary_layout, sprintf('%s vs %s, one trial each', names{1}, names{2}));

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).d_stats.mean, arm(1).d_stats.std, arm(1).d_stats.p90, arm(1).d_stats.max], ...
    [arm(2).d_stats.mean, arm(2).d_stats.std, arm(2).d_stats.p90, arm(2).d_stats.max], ...
    ["mean", "std", "p90", "max"], names{1}, names{2}, ...
    chen_color, gated_color, '[mm]', 'dist to centerline', '%.2f');

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).ang_stats.mean, arm(1).ang_stats.std, arm(1).ang_stats.p90, arm(1).ang_stats.max], ...
    [arm(2).ang_stats.mean, arm(2).ang_stats.std, arm(2).ang_stats.p90, arm(2).ang_stats.max], ...
    ["mean", "std", "p90", "max"], names{1}, names{2}, ...
    chen_color, gated_color, '[deg]', '|angular divergence|', '%.2f');

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).d_norm_stats.mean, arm(1).d_norm_stats.std, arm(1).d_norm_stats.p90, arm(1).d_norm_stats.max], ...
    [arm(2).d_norm_stats.mean, arm(2).d_norm_stats.std, arm(2).d_norm_stats.p90, arm(2).d_norm_stats.max], ...
    ["mean", "std", "p90", "max"], names{1}, names{2}, ...
    chen_color, gated_color, '[frac half-width]', 'normalized dist', '%.3f');

ax = nexttile;
plot_metric_bars(ax, arm(1).in_band_percent, arm(2).in_band_percent, ...
    names{1}, names{2}, chen_color, gated_color, '[%]', ...
    'in-band accuracy (zoomed)', '%.3f');
in_band_values = [arm(1).in_band_percent, arm(2).in_band_percent];
ylim(ax, [min(in_band_values) - 0.1 * (100 - min(in_band_values)) - 0.05, 100.1]);

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).violation.mean_excess_when_outside, arm(1).violation.max_excess], ...
    [arm(2).violation.mean_excess_when_outside, arm(2).violation.max_excess], ...
    ["mean|outside", "max"], names{1}, names{2}, ...
    chen_color, gated_color, '[frac half-width]', ...
    'boundary violation excess', '%.3f');

% ax = nexttile;
% plot_metric_bars(ax, arm(1).violation.sample_count, ...
%     arm(2).violation.sample_count, names{1}, names{2}, ...
%     chen_color, gated_color, '[count]', 'violating samples', '%.0f');

% ax = nexttile;
% plot_metric_bars(ax, arm(1).violation.mean_excess_overall, ...
%     arm(2).violation.mean_excess_overall, names{1}, names{2}, ...
%     chen_color, gated_color, '[frac half-width]', ...
%     'mean excess overall (stats-safe)', '%.5f');

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).d_rate_mean_abs, arm(1).d_rate_std], ...
    [arm(2).d_rate_mean_abs, arm(2).d_rate_std], ...
    ["mean|.|", "std"], names{1}, names{2}, ...
    chen_color, gated_color, '[mm/s]', 'd(dist)/dt', '%.2f');

ax = nexttile;
plot_grouped_metric_bars(ax, ...
    [arm(1).ang_rate_mean_abs, arm(1).ang_rate_std], ...
    [arm(2).ang_rate_mean_abs, arm(2).ang_rate_std], ...
    ["mean|.|", "std"], names{1}, names{2}, ...
    chen_color, gated_color, '[deg/s]', 'd(ang div)/dt', '%.1f');

% ax = nexttile;
% plot_metric_bars(ax, arm(1).ang_reversal_rate_per_m, ...
%     arm(2).ang_reversal_rate_per_m, names{1}, names{2}, ...
%     chen_color, gated_color, '[1/m]', 'ang div sign reversals', '%.1f');

% ax = nexttile;
% plot_grouped_metric_bars(ax, ...
%     [arm(1).d_speed_corr, arm(1).ang_speed_corr], ...
%     [arm(2).d_speed_corr, arm(2).ang_speed_corr], ...
%     ["dist", "|ang div|"], names{1}, names{2}, ...
%     chen_color, gated_color, '[r]', 'correlation with speed', '%.3f');

% ax = nexttile;
% plot_metric_bars(ax, arm(1).ang_bias_deg, arm(2).ang_bias_deg, ...
%     names{1}, names{2}, chen_color, gated_color, '[deg]', ...
%     'angular divergence bias', '%.2f');

save(write_metrics_mat, 'names', 'dist_to_centerline_m', 'angular_divergence_rad');

