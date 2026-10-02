%% Manual synthetic unit tests for the performance-feature functions.
% One section per function. Run Setup once, then run any section on its own.
% Each section builds a small synthetic input, calls the function, and prints
% the result next to the expected value. Edit the inputs to try your own cases.
%
% Inputs:
%   - none (all inputs are synthetic)
%
% Outputs:
%   - printed results only, nothing written to disk
%
% Depends on:
%   - functions/fine_mask_fun.m
%   - functions/sparc_fun.m
%   - functions/sparc_fine_fun.m
%   - functions/jerk_series_fun.m
%   - functions/compute_dsj.m
%   - functions/dsj_fine_fun.m
%   - functions/mean_force_fun.m
%   - functions/traveled_path_fun.m
%   - functions/path_efficiency_fun.m
%   - functions/boundary_violations_fun.m
%   - functions/nearest_centerline_distance_and_heading.m
%   - functions/normalized_corridor_distance.m
%   - functions/angular_divergence_fun.m

%% Setup (run once)
clc;
clear;
close all;

functions_directory = "C:\Users\Hamidjoon\projects\ICRA27\matlab\functions";
addpath(functions_directory);

fprintf('reading:\n  nothing (synthetic inputs)\n');
fprintf('writing:\n  nothing (printed results only)\n');

%% fine_mask_fun: fine_gross (1) and pear (2): x > 0.06; ellipse (3): x > 0.08 or x < -0.11
x_m = [-0.2; -0.05; 0.059; 0.061; 0.1];
mask = fine_mask_fun(x_m, 1);
fprintf('fine_gross: got %s, expected [0 0 0 1 1]\n', mat2str(double(mask')));

x_m = [-0.2; -0.12; -0.10; 0; 0.079; 0.081];
mask = fine_mask_fun(x_m, 3);
fprintf('ellipse:    got %s, expected [1 1 0 0 0 1]\n', mat2str(double(mask')));

%% sparc_fun: the authors' documented example and a smoothness ordering
% Example from the reference implementation: expect -1.41403.
t_s = -1:0.01:0.99;
speed_m_s = exp(-5 * t_s .^ 2);
sparc = sparc_fun(speed_m_s, 100, 4, 10.0, 0.05);
fprintf('reference example: got %.5f, expected -1.41403\n', sparc);

% One bell-shaped speed profile vs the same plus a second sub-movement: the
% second must be less smooth (more negative). Earlier run: -1.416 vs -1.607.
t_s = (0:0.002:4)';
one_bell = exp(-((t_s - 2) / 0.4) .^ 2);
two_bells = one_bell + 0.6 * exp(-((t_s - 2.9) / 0.15) .^ 2);
fprintf('one bell %.3f, with a second sub-movement %.3f (second should be lower)\n', ...
    sparc_fun(one_bell, 500, 4, 10, 0.05), sparc_fun(two_bells, 500, 4, 10, 0.05));

%% sparc_fine_fun: SPARC per fine pass, passes under min_pass_s dropped, then the mean
fs_hz = 100;
t_s = (0:1/fs_hz:20 - 1/fs_hz)';  % 2000 samples
speed_m_s = 0.05 + 0.04 * sin(2 * pi * 0.3 * t_s) .^ 2;

fine_mask = false(size(t_s));
fine_mask(101:400) = true;    % pass 1: 300 samples = 3.0 s, kept
fine_mask(701:750) = true;    % pass 2: 50 samples = 0.5 s, dropped (min 1 s)
fine_mask(1001:1400) = true;  % pass 3: 400 samples = 4.0 s, kept

[sparc_mean, sparc_passes, pass_duration_s] = sparc_fine_fun( ...
    speed_m_s, fine_mask, fs_hz, 1.0, 4, 10.0, 0.05);

expected_1 = sparc_fun(speed_m_s(101:400), fs_hz, 4, 10.0, 0.05);
expected_3 = sparc_fun(speed_m_s(1001:1400), fs_hz, 4, 10.0, 0.05);
fprintf('kept passes: got %d, expected 2\n', numel(sparc_passes));
fprintf('durations [s]: got %s, expected [3 4]\n', mat2str(pass_duration_s', 4));
fprintf('per-pass SPARC: got %s, expected %s\n', mat2str(sparc_passes', 5), mat2str([expected_1, expected_3], 5));
fprintf('mean: got %.5f, expected %.5f\n', sparc_mean, mean([expected_1, expected_3]));

%% jerk_series_fun + compute_dsj: a minimum-jerk movement has DSJ = 720
% No filtering (windows 0). Finite differences read low and approach 720 as the
% sampling rate rises: about 694.4 at 500 Hz and 719.4 at 10 kHz.
movement_s = 1;
length_m = 0.1;
for fs_hz = [500 10000]
    t_s = (0:1/fs_hz:movement_s)';
    tau = t_s / movement_s;
    x_m = length_m * (10 * tau .^ 3 - 15 * tau .^ 4 + 6 * tau .^ 5);
    vx_m_s = (length_m / movement_s) * (30 * tau .^ 2 - 60 * tau .^ 3 + 30 * tau .^ 4);
    zeros_m = zeros(size(t_s));

    [time_u_s, x_u_m, y_u_m, squared_jerk_m2_s6] = jerk_series_fun( ...
        t_s, x_m, zeros_m, vx_m_s, zeros_m, 0, 0);
    dsj = compute_dsj(time_u_s, squared_jerk_m2_s6, sum(hypot(diff(x_u_m), diff(y_u_m))));
    fprintf('fs %5d Hz: DSJ got %.1f, expected about 720\n', fs_hz, dsj);
end

%% dsj_fine_fun: DSJ per fine pass, short passes dropped, then the mean
% Constant squared jerk J2 = 4 and constant speed v = 0.1 m/s over a pass of
% duration T: DSJ = (J2 * T) * T^5 / (v * T)^2 = J2 * T^4 / v^2.
fs_hz = 500;
t_s = (0:1/fs_hz:10)';
speed_m_s = 0.1;
x_m = speed_m_s * t_s;
y_m = zeros(size(t_s));
squared_jerk_m2_s6 = 4 * ones(size(t_s));

fine_mask = false(size(t_s));
fine_mask(1:1000) = true;      % pass A: 1.998 s
fine_mask(2001:4000) = true;   % pass B: 3.998 s
fine_mask(4501:4600) = true;   % pass C: 0.198 s, dropped (min 1 s)

[dsj_mean, dsj_passes] = dsj_fine_fun(t_s, x_m, y_m, squared_jerk_m2_s6, fine_mask, 1.0);

expected_A = 4 * 1.998 ^ 4 / speed_m_s ^ 2;
expected_B = 4 * 3.998 ^ 4 / speed_m_s ^ 2;
fprintf('kept passes: got %d, expected 2\n', numel(dsj_passes));
fprintf('per-pass DSJ: got %s, expected %s\n', mat2str(dsj_passes', 6), mat2str([expected_A, expected_B], 6));
fprintf('mean: got %.4f, expected %.4f\n', dsj_mean, mean([expected_A, expected_B]));

%% mean_force_fun: mean force magnitude over the moving samples only
fx_n = [3; 0; 9; 6];
fy_n = [4; 1; 9; 8];
speed_m_s = [0.1; 0.1; 0.001; 0.2];   % sample 3 is below the threshold
mean_force_n = mean_force_fun(fx_n, fy_n, speed_m_s, 0.005);
fprintf('mean force: got %.4f, expected %.4f (mean of 5, 1, 10)\n', mean_force_n, mean([5 1 10]));

%% traveled_path_fun: sum of the step lengths
fprintf('3-4-5 walk: got %.4f, expected 5\n', traveled_path_fun([0; 3], [0; 4]));
fprintf('unit square lap: got %.4f, expected 4\n', traveled_path_fun([0; 1; 1; 0; 0], [0; 0; 1; 1; 0]));

%% nearest_centerline_distance_and_heading: shortest distance to the closed centerline
% Unit square centerline (0,0) -> (1,0) -> (1,1) -> (0,1), closed.
centerline_xy_m = [0 0; 1 0; 1 1; 0 1];
x_m = [0.5; -0.1; 0.5];
y_m = [-0.2; 0.5; 0.3];
[distance_m, heading_rad, position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);
fprintf('distance: got %s, expected [0.2 0.1 0.3]\n', mat2str(distance_m', 4));
fprintf('heading [deg]: got %s, expected [0 -90 0]\n', mat2str(rad2deg(heading_rad'), 4));
fprintf('centerline position: got %s, expected [1.5 4.5 1.5] (index + fraction along the segment)\n', mat2str(position', 4));

%% normalized_corridor_distance: distance as a fraction of the local half-width
% Position 4.5 is half way along the closing segment (point 4 to point 1), so
% the half-width is interpolated between 4 and 1 -> 2.5.
half_width_m = [1; 2; 3; 4];
normalized = normalized_corridor_distance([0.2; 0.5; 0.1], [1.5; 1; 4.5], half_width_m);
fprintf('got %s, expected %s\n', mat2str(normalized', 4), mat2str([0.2 / 1.5, 0.5 / 1, 0.1 / 2.5], 4));
fprintf('position n_ref+1 (end of the closing segment): got %.4f, expected 1 (half-width of point 1)\n', ...
    1 / normalized_corridor_distance(1, 5, half_width_m));

%% path_efficiency_fun: net centerline progress / TCP path length
radius_m = 0.1;
n_ref = 2000;
angle_rad = linspace(0, 2 * pi, n_ref + 1)';
angle_rad(end) = [];
centerline_xy_m = [radius_m * cos(angle_rad), radius_m * sin(angle_rad)];
phi_rad = linspace(0, 3 * 2 * pi, 6000)';   % 3 laps

[efficiency, progress_m] = path_efficiency_fun(radius_m * cos(phi_rad), radius_m * sin(phi_rad), centerline_xy_m);
fprintf('on the centerline: efficiency got %.4f (expected 1), progress %.4f m (expected 3*2*pi*R = %.4f)\n', ...
    efficiency, progress_m, 3 * 2 * pi * radius_m);

wobble_m = 0.003 * sin(40 * phi_rad);
efficiency = path_efficiency_fun((radius_m + wobble_m) .* cos(phi_rad), (radius_m + wobble_m) .* sin(phi_rad), centerline_xy_m);
fprintf('3 mm wobble: efficiency got %.4f, expected about 0.77 (lower than 1)\n', efficiency);

efficiency = path_efficiency_fun(radius_m * cos(-phi_rad), radius_m * sin(-phi_rad), centerline_xy_m);
fprintf('same laps the other way round: got %.4f, expected 1 (direction does not matter)\n', efficiency);

%% boundary_violations_fun: complete out-and-back excursions of at least min_duration_s
% Counted: starts after sample 1, ends before the last sample, lasts >= minimum.
dt_s = 0.002;
normalized_distance = 0.5 * ones(1000, 1);
normalized_distance(1:30) = 1.5;        % starts outside: never counted
normalized_distance(100) = 1.2;         % 1-sample graze: 0.002 s
normalized_distance(200:219) = 1.3;     % 20 samples: 0.04 s
normalized_distance(400:499) = 2.0;     % 100 samples: 0.2 s
normalized_distance(951:1000) = 1.4;    % never returns: never counted
time_s = (0:numel(normalized_distance) - 1)' * dt_s;

[n_050, duration_050_s] = boundary_violations_fun(normalized_distance, time_s, 0.05);
[n_020, ~] = boundary_violations_fun(normalized_distance, time_s, 0.02);
[n_000, ~] = boundary_violations_fun(normalized_distance, time_s, 0);
fprintf('min 0.05 s: got %d (expected 1), duration %s s (expected 0.2)\n', n_050, mat2str(duration_050_s', 3));
fprintf('min 0.02 s: got %d (expected 2)\n', n_020);
fprintf('min 0 s: got %d (expected 3: graze, 0.04 s, 0.2 s)\n', n_000);

%% angular_divergence_fun: angle between TCP chord and centerline chord, per interval
% Circle of radius R, stations every 5 mm, 3 laps.
radius_m = 0.1;
station_spacing_m = 5e-3;
n_ref = 2000;
angle_rad = linspace(0, 2 * pi, n_ref + 1)';
angle_rad(end) = [];
centerline_xy_m = [radius_m * cos(angle_rad), radius_m * sin(angle_rad)];
phi_rad = linspace(0, 3 * 2 * pi, 90000)';

% 1) TCP exactly on the centerline: theta ~ 0.
x_m = radius_m * cos(phi_rad);
y_m = radius_m * sin(phi_rad);
[~, ~, centerline_position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);
theta_deg = angular_divergence_fun(x_m, y_m, centerline_position, centerline_xy_m, station_spacing_m);
fprintf('on centerline: %d intervals (expected about %.0f), mean|theta| %.4f deg (expected ~0)\n', ...
    numel(theta_deg), 3 * 2 * pi * radius_m / station_spacing_m, mean(abs(theta_deg)));

% 2) concentric circle 3 mm outside: same direction as the centerline, theta ~ 0.
x_m = (radius_m + 0.003) * cos(phi_rad);
y_m = (radius_m + 0.003) * sin(phi_rad);
[~, ~, centerline_position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);
theta_deg = angular_divergence_fun(x_m, y_m, centerline_position, centerline_xy_m, station_spacing_m);
fprintf('concentric +3 mm: mean|theta| %.4f deg (expected ~0)\n', mean(abs(theta_deg)));

% 3) sideways wobble: compare with the angle computed from the exact station points.
wobble_amplitude_m = 0.003;
wobble_cycles_per_rad = 20;
radius_tcp_m = radius_m + wobble_amplitude_m * sin(wobble_cycles_per_rad * phi_rad);
x_m = radius_tcp_m .* cos(phi_rad);
y_m = radius_tcp_m .* sin(phi_rad);
[~, ~, centerline_position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);
theta_deg = angular_divergence_fun(x_m, y_m, centerline_position, centerline_xy_m, station_spacing_m);

station_phi_rad = (0:numel(theta_deg))' * station_spacing_m / radius_m;
station_radius_m = radius_m + wobble_amplitude_m * sin(wobble_cycles_per_rad * station_phi_rad);
tcp_chord_m = diff([station_radius_m .* cos(station_phi_rad), station_radius_m .* sin(station_phi_rad)]);
center_chord_m = diff([radius_m * cos(station_phi_rad), radius_m * sin(station_phi_rad)]);
expected_deg = rad2deg(atan2(tcp_chord_m(:, 2), tcp_chord_m(:, 1)) - atan2(center_chord_m(:, 2), center_chord_m(:, 1)));
expected_deg = mod(expected_deg + 180, 360) - 180;
fprintf('wobble: max |function - exact| %.3f deg (expected < 0.2), mean|theta| %.3f vs exact %.3f deg\n', ...
    max(abs(theta_deg - expected_deg)), mean(abs(theta_deg)), mean(abs(expected_deg)));

% 4) travelling the other way round or listing the centerline backwards must not change theta.
[~, ~, position_reversed] = nearest_centerline_distance_and_heading(flipud(x_m), flipud(y_m), centerline_xy_m);
theta_reversed_deg = angular_divergence_fun(flipud(x_m), flipud(y_m), position_reversed, centerline_xy_m, station_spacing_m);
fprintf('reversed travel: mean|theta| %.4f deg (expected %.4f)\n', mean(abs(theta_reversed_deg)), mean(abs(theta_deg)));

%% positive_progress, total_progress, horizon angle: formulas used in the extraction script
% Concentric circle 3 mm outside the centerline, 3 laps, stations every 5 mm.
radius_m = 0.1;
station_spacing_m = 5e-3;
n_ref = 2000;
angle_rad = linspace(0, 2 * pi, n_ref + 1)';
angle_rad(end) = [];
centerline_xy_m = [radius_m * cos(angle_rad), radius_m * sin(angle_rad)];
phi_rad = linspace(0, 3 * 2 * pi, 90000)';
x_m = (radius_m + 0.003) * cos(phi_rad);
y_m = (radius_m + 0.003) * sin(phi_rad);

[~, ~, centerline_position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);
[theta_deg, tcp_chord_length_m, center_chord_length_m, tcp_heading_deg] = ...
    angular_divergence_fun(x_m, y_m, centerline_position, centerline_xy_m, station_spacing_m);

progress_m = tcp_chord_length_m .* cosd(abs(theta_deg));
positive_progress = mean(progress_m ./ center_chord_length_m);
total_progress_m = sum(progress_m);
horizon_angle_deg = mean(abs(tcp_heading_deg));
horizon_angle_sq_deg2 = mean(tcp_heading_deg .^ 2);

fprintf('positive progress: got %.5f, expected (R+d)/R = %.5f\n', positive_progress, (radius_m + 0.003) / radius_m);
fprintf('total progress: got %.5f m, expected about %.5f m (intervals * chord of the offset circle)\n', ...
    total_progress_m, numel(theta_deg) * 2 * (radius_m + 0.003) * sin(station_spacing_m / radius_m / 2));
fprintf('horizon angle: mean|psi| got %.2f deg (expected 90), mean psi^2 got %.0f deg^2 (expected about 10800)\n', ...
    horizon_angle_deg, horizon_angle_sq_deg2);
