%% Replay one trial and count boundary violations against the drawn band.
% Samples outside the band are drawn red; each counted violation (same rule
% as boundary_violations_fun) is numbered where it started. Violations
% counted by the centerline method of the feature script are marked blue x.
%
% Inputs:
%   - data/trial_manifest.csv (shape_obj and offset of the trial)
%   - data/raw_data/test_subjects/<robot_csv_name>
%   - data/shapes/<shape_obj>
%
% Outputs:
%   - animation only, nothing written to disk
%
% Depends on:
%   - functions/read_shape_band_obj.m
%   - functions/nearest_centerline_distance_and_heading.m
%   - functions/normalized_corridor_distance.m

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
robot_csv_name = "ur5e_gaze_assisted_chen_vac_2d_held_out_ellipse_and_s11_on_narrow_ellipse_shape.csv";
read_robot_csv = fullfile(data_directory, 'raw_data', 'test_subjects', ...
    robot_csv_name);
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
shape_directory = fullfile(data_directory, 'shapes');

time_window_s = [0, Inf];  % replayed part of the trial [s]
playback_speed = 10;  % 1 = real time
frame_step_samples = 10;  % samples per frame (data at 500 Hz)
follow_half_width_m = 0;  % view half-width around the TCP [m], 0 = whole
violation_min_s = 0.05;  % shortest excursion counted [s], as in d_0929

% End User Setting
%% Main Body

M = readtable(manifest_file, 'TextType', 'string');
trial = M(M.robot_csv == robot_csv_name, :);
read_shape_obj = fullfile(shape_directory, trial.shape_obj);

fprintf('reading:\n  %s\n  %s\n  %s\n', manifest_file, read_robot_csv, ...
    read_shape_obj);
fprintf('writing:\n  nothing (animation only)\n');

T = readtable(read_robot_csv);
time_s = T.time_s - T.time_s(1);
x_m = T.coppeliasim_x_m;
y_m = T.coppeliasim_y_m;
dt_s = median(diff(time_s));

% Drawn band and the centerline of the feature script, CoppeliaSim frame.
[outer_xy_m, inner_xy_m, center_xy_m] = read_shape_band_obj( ...
    read_shape_obj, 5e-4);
offset_xy_m = [trial.offset_x, trial.offset_y];
outer_xy_m = outer_xy_m + offset_xy_m;
inner_xy_m = inner_xy_m + offset_xy_m;
center_xy_m = center_xy_m + offset_xy_m;
half_width_m = hypot(outer_xy_m(:, 1) - center_xy_m(:, 1), ...
    outer_xy_m(:, 2) - center_xy_m(:, 2));

% Outside the drawn band: outside the outer edge or inside the inner edge.
outside_band = ~(inpolygon(x_m, y_m, outer_xy_m(:, 1), outer_xy_m(:, 2)) ...
    & ~inpolygon(x_m, y_m, inner_xy_m(:, 1), inner_xy_m(:, 2)));

% Outside by the centerline method (normalized distance > 1), as in d_0929.
[distance_m, ~, centerline_position] = ...
    nearest_centerline_distance_and_heading(x_m, y_m, center_xy_m);
outside_centerline = normalized_corridor_distance(distance_m, ...
    centerline_position, half_width_m) > 1;

% Counted excursions, rule of boundary_violations_fun: starts after the
% first sample, ends before the last, lasts at least violation_min_s.
change = diff([0; outside_band; 0]);
band_first = find(change == 1);
band_last = find(change == -1) - 1;
counted = band_first > 1 & band_last < numel(x_m) & ...
    (band_last - band_first + 1) * dt_s >= violation_min_s;
band_first = band_first(counted);
band_last = band_last(counted);

change = diff([0; outside_centerline; 0]);
centerline_first = find(change == 1);
centerline_last = find(change == -1) - 1;
counted = centerline_first > 1 & centerline_last < numel(x_m) & ...
    (centerline_last - centerline_first + 1) * dt_s >= violation_min_s;
centerline_first = centerline_first(counted);
centerline_last = centerline_last(counted);

fprintf('\n%s\n  drawn band: %d violations\n', robot_csv_name, ...
    numel(band_last));
fprintf('  centerline method: %d violations\n', numel(centerline_last));

%% Plot
figure('Name', sprintf('replay_%s', erase(robot_csv_name, '.csv')), ...
    'Color', 'w');
hold on;
edge_line = plot(outer_xy_m([1:end, 1], 1), outer_xy_m([1:end, 1], 2), 'k');
plot(inner_xy_m([1:end, 1], 1), inner_xy_m([1:end, 1], 2), 'k');
center_line = plot(center_xy_m(:, 1), center_xy_m(:, 2), 'b:');
trail = plot(NaN, NaN, 'Color', [0.6, 0.6, 0.6]);
outside_dots = plot(NaN, NaN, 'r.');
centerline_marks = plot(NaN, NaN, 'bx');
cursor = plot(x_m(1), y_m(1), 'ko', 'MarkerFaceColor', 'k');
axis equal;
xlim([min(outer_xy_m(:, 1)), max(outer_xy_m(:, 1))] + [-0.01, 0.01]);
ylim([min(outer_xy_m(:, 2)), max(outer_xy_m(:, 2))] + [-0.01, 0.01]);
xlabel('x [m]');
ylabel('y [m]');
grid on;
legend([edge_line, center_line, trail, outside_dots, centerline_marks], ...
    {'band edge', 'centerline of the feature script', 'TCP path', ...
    'outside the drawn band', 'counted by the centerline method'}, ...
    'Location', 'southoutside');

y_outside_m = y_m;
y_outside_m(~outside_band) = NaN;
first_index = find(time_s >= time_window_s(1), 1);
last_index = find(time_s <= time_window_s(2), 1, 'last');
previous_index = first_index - 1;

for i = first_index:frame_step_samples:last_index
    if ~ishandle(cursor)
        break;  % figure closed
    end

    set(trail, 'XData', x_m(first_index:i), 'YData', y_m(first_index:i));
    set(outside_dots, 'XData', x_m(first_index:i), ...
        'YData', y_outside_m(first_index:i));
    set(cursor, 'XData', x_m(i), 'YData', y_m(i), ...
        'MarkerFaceColor', [outside_band(i), 0, 0]);

    % Number each drawn-band violation once it has ended (TCP came back).
    for k = find(band_last > previous_index & band_last <= i)'
        text(x_m(band_first(k)), y_m(band_first(k)), sprintf(' %d', k), ...
            'Color', 'r', 'FontWeight', 'bold');
    end
    new = centerline_last > previous_index & centerline_last <= i;
    set(centerline_marks, ...
        'XData', [centerline_marks.XData, x_m(centerline_first(new))'], ...
        'YData', [centerline_marks.YData, y_m(centerline_first(new))']);

    if follow_half_width_m > 0
        xlim(x_m(i) + [-1, 1] * follow_half_width_m);
        ylim(y_m(i) + [-1, 1] * follow_half_width_m);
    end

    title({sprintf('t = %.1f s', time_s(i)), sprintf(['red numbers, ' ...
        'drawn band: %d of %d | blue x, centerline method: %d of %d'], ...
        sum(band_last <= i), numel(band_last), ...
        sum(centerline_last <= i), numel(centerline_last))});
    drawnow;
    pause(frame_step_samples * dt_s / playback_speed);
    previous_index = i;
end
