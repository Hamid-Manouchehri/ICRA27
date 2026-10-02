function [time_u_s, x_u_m, y_u_m, squared_jerk_m2_s6] = jerk_series_fun( ...
    time_s, x_m, y_m, vx_m_s, vy_m_s, outlier_window_s, smoothing_window_s)
%JERK_SERIES_FUN Squared planar jerk of the TCP on a uniform time grid.
%
% Same filtering as compute_jerk_fine.m: resample to a uniform grid, median
% filter (outlier_window_s), moving-mean smoothing of velocity and
% acceleration (smoothing_window_s), jerk by differentiation.
%
% Math (on the filtered series):
%   a = movmean(dv/dt),  j = da/dt,  squared_jerk = jx^2 + jy^2
%
% Inputs:
%   time_s, x_m, y_m, vx_m_s, vy_m_s: [n x 1] recorded TCP time, position, velocity
%   outlier_window_s: median-filter window [s] (<= 0 turns it off)
%   smoothing_window_s: moving-mean window [s] (<= 0 turns it off)
%
% Outputs:
%   time_u_s, x_u_m, y_u_m: uniform time grid [s] and TCP position on it [m]
%   squared_jerk_m2_s6: squared jerk on that grid [m^2/s^6]

dt_s = median(diff(time_s));
time_u_s = (time_s(1):dt_s:time_s(end))';
x_u_m = interp1(time_s, x_m, time_u_s, 'linear');
y_u_m = interp1(time_s, y_m, time_u_s, 'linear');
vx_u = interp1(time_s, vx_m_s, time_u_s, 'linear');
vy_u = interp1(time_s, vy_m_s, time_u_s, 'linear');

outlier_samples = window_to_odd_sample_count(outlier_window_s, dt_s, numel(time_u_s));
smoothing_samples = window_to_odd_sample_count(smoothing_window_s, dt_s, numel(time_u_s));

if outlier_samples > 1
    vx_u = movmedian(vx_u, outlier_samples, 'Endpoints', 'shrink');
    vy_u = movmedian(vy_u, outlier_samples, 'Endpoints', 'shrink');
    x_u_m = movmedian(x_u_m, outlier_samples, 'Endpoints', 'shrink');
    y_u_m = movmedian(y_u_m, outlier_samples, 'Endpoints', 'shrink');
end

vx_u = movmean(vx_u, smoothing_samples, 'Endpoints', 'shrink');
vy_u = movmean(vy_u, smoothing_samples, 'Endpoints', 'shrink');

ax = movmean(gradient(vx_u, time_u_s), smoothing_samples, 'Endpoints', 'shrink');
ay = movmean(gradient(vy_u, time_u_s), smoothing_samples, 'Endpoints', 'shrink');

jx = gradient(ax, time_u_s);
jy = gradient(ay, time_u_s);
squared_jerk_m2_s6 = jx .^ 2 + jy .^ 2;
end
