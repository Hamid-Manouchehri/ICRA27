function [lap_start_s, lap_stop_s] = lap_times_by_start_crossing_fun( ...
    time_s, x_m, y_m)
%LAP_TIMES_BY_START_CROSSING_FUN Start and stop time of each complete lap.
%
% A lap ends when the TCP crosses the start x again in the same direction,
% in the start's half of the shape. The TCP must visit the other half
% between laps, so wobble at the start line is not a lap; the last crossing
% of a wobble counts. Time before the first lap and the partial last lap
% are not returned.
%
% Math:
%   y_mid = (max(y) + min(y)) / 2
%   other half: sign(y_0 - y_mid) * (y - y_mid) < 0
%   d = sign(x - x_0) at the first sample in the other half
%   crossing at sample i: d * (x(i-1) - x_0) <= 0 < d * (x(i) - x_0)
%
% Inputs:
%   time_s: [n x 1] sample time [s]
%   x_m, y_m: [n x 1] TCP position [m]
%
% Outputs:
%   lap_start_s, lap_stop_s: [n_laps x 1] start and stop time of each lap [s]

x_start_m = x_m(1);
y_mid_m = (max(y_m) + min(y_m)) / 2;
other_half = sign(y_m(1) - y_mid_m) * (y_m - y_mid_m) < 0;
direction = sign(x_m(find(other_half, 1)) - x_start_m);  % side gone around
past_start = direction * (x_m - x_start_m) > 0;

boundary_index = [];
armed = true;  % false until the TCP visits the other half
for i = 2:numel(x_m)
    if ~past_start(i - 1) && past_start(i) && ~other_half(i)
        if armed
            boundary_index(end + 1) = i;
            armed = false;
        else
            boundary_index(end) = i;  % wobble at the start line: keep the last
        end
    end
    if other_half(i)
        armed = true;
    end
end

lap_start_s = time_s(boundary_index(1:end - 1));
lap_stop_s = time_s(boundary_index(2:end));
end
