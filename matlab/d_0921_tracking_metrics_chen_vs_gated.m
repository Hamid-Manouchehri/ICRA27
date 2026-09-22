clc; clear; close all;

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

% End User Setting
%% Main Body

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

for k = 1:2
    T = raw_data{k};
    x = T.coppeliasim_x_m;
    y = T.coppeliasim_y_m;
    vx = T.tcp_vx_actual_m_s;
    vy = T.tcp_vy_actual_m_s;

    n = length(x);
    d = zeros(n,1);
    path_angle = zeros(n,1);
    for i = 1:n
        dist_to_all = hypot(center(:,1)-x(i), center(:,2)-y(i));
        [d(i), j] = min(dist_to_all);
        j_next = mod(j, size(center,1)) + 1;
        path_angle(i) = atan2(center(j_next,2)-center(j,2), center(j_next,1)-center(j,1));
    end

    tcp_angle = atan2(vy, vx);  % TCP pose taken as direction of travel (no orientation column logged)
    ang_div = atan2(sin(tcp_angle-path_angle), cos(tcp_angle-path_angle));

    fprintf('\n--- %s ---\n', names{k});
    fprintf('mean distance to centerline [mm] : %.2f\n', 1e3*mean(d));
    fprintf('mean |angular divergence| [deg]  : %.2f\n', mean(abs(rad2deg(ang_div))));

    figure;
    plot(outer(:,1), outer(:,2), 'k-', 'DisplayName', 'outer'); hold on;
    plot(inner(:,1), inner(:,2), 'k-', 'DisplayName', 'inner');
    plot(center(:,1), center(:,2), '-.', 'DisplayName', 'centerline')
    plot(x, y, '.', 'DisplayName', 'TCP path');
    quiver(x(1:stride:end), y(1:stride:end), cos(tcp_angle(1:stride:end)), sin(tcp_angle(1:stride:end)), 0.5, 'r', 'DisplayName', 'actual heading');
    quiver(x(1:stride:end), y(1:stride:end), cos(path_angle(1:stride:end)), sin(path_angle(1:stride:end)), 0.5, 'b', 'DisplayName', 'desired heading');
    axis equal;
    title(names{k});
    legend;

    figure;
    plot(d);
    legend("point-wise dist to center-line");

    figure;
    plot(ang_div)
    legend("point-wise angular divergence (rad)");
end

