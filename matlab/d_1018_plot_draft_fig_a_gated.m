%% Draft Fig. A: Gated controller, one participant, one lap.
% Figure 1: P_t, b_d and |F| of that lap as Gaussian-weighted maps, each
% sample's value spread across the corridor. Figure 2: the lap over time.
%
% Inputs:
%   - data/trial_manifest.csv (one row per trial)
%   - data/raw_data/test_subjects/<robot_csv> (Gated trial of the selection)
%   - data/shapes/<shape_obj> (corridor outline)
%
% Outputs:
%   - two figures only, nothing written to disk
%
% Depends on:
%   - functions/read_shape_band_obj.m

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

shape_name = "fine_gross";  % fine_gross (paw), pear, ellipse
bandwidth_name = "wide";  % wide, narrow
subject_id = 18;  % participant shown
lap_number = 2;  % lap shown (1 = first lap from the start)

bin_size_m = 0.0003;  % map grid spacing (pixel size) [m]
smoothing_sigma_m = 0.003;  % Gaussian smoothing width [m]
margin_m = 0.02;  % map extent beyond the corridor's outer edge [m]
force_color_limits_n = [2, 7];  % |F| map color range [N]

% End User Setting
%% Main Body

fprintf(['reading:\n  %s\n  %s\\<robot_csv of Gated s%d %s %s>\n' ...
    '  %s\\<shape_obj>\n'], manifest_file, raw_data_directory, subject_id, ...
    bandwidth_name, shape_name, shape_directory);
fprintf('writing:\n  nothing (figure only)\n');

M = readtable(manifest_file, 'TextType', 'string');
M = M(M.controller == "Gated" & M.shape == shape_name & ...
    M.bandwidth == bandwidth_name & M.subject == subject_id, :);

% Corridor outline in the CoppeliaSim frame.
[outer_xy_m, inner_xy_m] = read_shape_band_obj( ...
    fullfile(shape_directory, M.shape_obj), 5e-4);
outer_xy_m = outer_xy_m + [M.offset_x, M.offset_y];
inner_xy_m = inner_xy_m + [M.offset_x, M.offset_y];

fprintf('Reading s%d: %s\n', subject_id, M.robot_csv);
T = readtable(fullfile(raw_data_directory, M.robot_csv));
time_s = T.time_s - T.time_s(1);

% Laps completed: clockwise turns around the inner edge's center.
center_xy_m = mean(inner_xy_m);
angle_rad = unwrap(atan2(T.coppeliasim_y_m - center_xy_m(2), ...
    T.coppeliasim_x_m - center_xy_m(1)));
laps_done = cummax(-(angle_rad - angle_rad(1)) / (2 * pi));
in_lap = floor(laps_done) == lap_number - 1;
T = T(in_lap, :);
time_s = time_s(in_lap);
fprintf('lap %d: %.2f to %.2f s\n', lap_number, time_s(1), time_s(end));

force_n = hypot(T.Fx_filt_N, T.Fy_filt_N);
start_xy_m = [T.coppeliasim_x_m(1), T.coppeliasim_y_m(1)];
condition_label = bandwidth_name + " " + ...
    replace(shape_name, "fine_gross", "paw");

% Square grid covering the corridor.
x_edges_m = min(outer_xy_m(:, 1)) - margin_m : bin_size_m : ...
    max(outer_xy_m(:, 1)) + margin_m;
y_edges_m = min(outer_xy_m(:, 2)) - margin_m : bin_size_m : ...
    max(outer_xy_m(:, 2)) + margin_m;
x_centers_m = x_edges_m(1:end - 1) + bin_size_m / 2;
y_centers_m = y_edges_m(1:end - 1) + bin_size_m / 2;
grid_size = [numel(y_centers_m), numel(x_centers_m)];

% Gaussian kernel out to 3 sigma, applied along y then x (separable).
kernel_half_width = ceil(3 * smoothing_sigma_m / bin_size_m);
g = exp(-0.5 * ((-kernel_half_width:kernel_half_width) * bin_size_m / ...
    smoothing_sigma_m).^2);

% Grid cells inside the corridor (inside the outer edge, outside the inner).
[x_grid_m, y_grid_m] = meshgrid(x_centers_m, y_centers_m);
in_corridor = inpolygon(x_grid_m, y_grid_m, outer_xy_m(:, 1), ...
    outer_xy_m(:, 2)) & ~inpolygon(x_grid_m, y_grid_m, inner_xy_m(:, 1), ...
    inner_xy_m(:, 2));

% Map samples: in the grid, after the first classifier update (before it,
% P_t is held at 0.5).
ix = discretize(T.coppeliasim_x_m, x_edges_m);
iy = discretize(T.coppeliasim_y_m, y_edges_m);
in_grid = ~isnan(ix) & ~isnan(iy) & T.classifier_update_count > 0;
bin = [iy(in_grid), ix(in_grid)];

% Gaussian-weighted mean: smooth the per-cell sums and counts, divide.
weight = conv2(g, g, accumarray(bin, 1, grid_size), 'same');
pt_map = conv2(g, g, accumarray(bin, T.fine_probability(in_grid), ...
    grid_size), 'same') ./ weight;
bd_map = conv2(g, g, accumarray(bin, T.bd_applied_Ns_m(in_grid), ...
    grid_size), 'same') ./ weight;
force_map = conv2(g, g, accumarray(bin, force_n(in_grid), grid_size), ...
    'same') ./ weight;

% Fill the corridor: each cell takes the value at its nearest lap sample,
% so each sample's color spans the corridor across its direction of travel.
sample_cell = sub2ind(grid_size, bin(:, 1), bin(:, 2));
nearest_cell = sample_cell(knnsearch([T.coppeliasim_x_m(in_grid), ...
    T.coppeliasim_y_m(in_grid)], [x_grid_m(in_corridor), ...
    y_grid_m(in_corridor)]));
pt_map(in_corridor) = pt_map(nearest_cell);
bd_map(in_corridor) = bd_map(nearest_cell);
force_map(in_corridor) = force_map(nearest_cell);
pt_map(~in_corridor) = NaN;
bd_map(~in_corridor) = NaN;
force_map(~in_corridor) = NaN;

%% Plot
% Figure 1: maps of the lap.
figure('Name', sprintf('draft_fig_a_gated_maps_s%d_%s_%s_lap%d', ...
    subject_id, bandwidth_name, shape_name, lap_number), ...
    'WindowStyle', 'docked', 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

maps = {pt_map, bd_map, force_map};
color_limits = {[0, 1], [10, 200], force_color_limits_n};
map_labels = ["P_t", "b_d [N s/m]", "|F| [N]"];

for j = 1:3
    nexttile;
    imagesc(x_centers_m, y_centers_m, maps{j}, 'AlphaData', ~isnan(maps{j}));
    hold on;
    plot(outer_xy_m([1:end, 1], 1), outer_xy_m([1:end, 1], 2), 'k');
    plot(inner_xy_m([1:end, 1], 1), inner_xy_m([1:end, 1], 2), 'k');
    plot(start_xy_m(1), start_xy_m(2), 'ko', 'MarkerFaceColor', 'k');
    axis xy;
    axis image;  % equal scale, tight limits
    axis off;
    clim(color_limits{j});
    cb = colorbar('southoutside');
    cb.Label.String = map_labels(j);
    colormap(gca, 'parula')
    % title({sprintf('%s, Gaussian \\sigma = %g mm', map_labels(j), ...
    %     1000 * smoothing_sigma_m), sprintf('Gated, s%d, %s, lap %d', ...
    %     subject_id, condition_label, lap_number)});
end

% Figure 2: the lap over time.
figure('Name', sprintf('draft_fig_a_gated_time_s%d_%s_%s_lap%d', ...
    subject_id, bandwidth_name, shape_name, lap_number), ...
    'WindowStyle', 'docked', 'Color', 'w');
tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
plot(time_s, T.fine_probability, 'b');
ylabel('P_t');
ylim([0, 1.1]);
xlim(time_s([1, end]));
grid on;
title(sprintf('Gated, s%d, %s, lap %d', subject_id, condition_label, ...
    lap_number));

nexttile;
hold on;
% fill([time_s; flipud(time_s)], ...
%     [T.bd_min_Ns_m; flipud(T.bd_max_Ns_m)], [0.85, 0.85, 0.85], ...
%     'EdgeColor', 'none');
plot(time_s, T.bd_applied_Ns_m, 'b');
ylabel('b_d [N s/m]');
ylim([0, 210]);
xlim(time_s([1, end]));
grid on;
% title('gray: bounds [b_{min}, b_{max}] set by P_t, black: applied b_d');

nexttile;
plot(time_s, force_n, 'b');
ylabel('|F| [N]');
xlabel('time [s]');
xlim(time_s([1, end]));
grid on;
