function [t_common_s, value_a_common, value_b_common] = ...
    common_time_interp_fun(time_a_s, value_a, time_b_s, value_b, dt_s)
%COMMON_TIME_INTERP_FUN Two signals resampled onto one shared time grid.
%
% The grid covers only the time both signals exist, with step dt_s; each
% signal is linearly interpolated onto it. Works for any logged metric.
%
% Math:
%   t_common = max(t_a(1), t_b(1)) : dt : min(t_a(end), t_b(end))
%   value_common(t) = linear interpolation of value at t
%
% Inputs:
%   time_a_s, time_b_s: [n x 1] strictly increasing sample times [s]
%   value_a, value_b: [n x k] values at those times (any metric, k columns)
%   dt_s: grid step [s]
%
% Outputs:
%   t_common_s: [m x 1] shared time grid [s]
%   value_a_common, value_b_common: [m x k] values on the grid

t_common_s = (max(time_a_s(1), time_b_s(1)):dt_s: ...
    min(time_a_s(end), time_b_s(end)))';
value_a_common = interp1(time_a_s, value_a, t_common_s);
value_b_common = interp1(time_b_s, value_b, t_common_s);
end
