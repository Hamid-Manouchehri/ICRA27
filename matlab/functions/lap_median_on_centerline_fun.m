function [value_median, grid_arclength_m, median_gap_m] = ...
    lap_median_on_centerline_fun(x_m, y_m, value, centerline_xy_m)
%LAP_MEDIAN_ON_CENTERLINE_FUN Value at every centerline point, median over laps.
%
% Each centerline point is one grid point. In every lap, the grid point's
% value is taken from the sample nearest to it in progress (same lap, same
% position); laps that never reach the point give NaN. The median over laps
% is returned. Same grid for every trial of a shape, so results line up.
%
% Math (grid point j at arclength s_j, start position s_0, loop length L,
% travel sign d, lap k = 1..ceil(max(progress) / L)):
%   offset_j    = mod(d * (s_j - s_0), L)     distance from the start
%   target_jk   = offset_j + (k - 1) * L      progress of point j in lap k
%   value_jk    = value(i*), i* = argmin_i |progress(i) - target_jk|
%   value_median(j) = median over k of value_jk
%
% Inputs:
%   x_m, y_m: [n x 1] TCP position, same frame as the centerline [m]
%   value: [n x 1] logged value per sample (e.g. damping)
%   centerline_xy_m: [n_ref x 2] ordered closed centerline [m]
%
% Outputs:
%   value_median: [n_ref x 1] median over laps at each centerline point
%   grid_arclength_m: [n_ref x 1] arclength of each centerline point [m]
%   median_gap_m: median x-y distance from a grid point to its sample [m]

[position_m, progress_m, travel_sign] = centerline_arclength_fun(x_m, y_m, ...
    centerline_xy_m);

n_ref = size(centerline_xy_m, 1);
closed_xy_m = centerline_xy_m([1:n_ref, 1], :);
segment_length_m = hypot(diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
node_arclength_m = [0; cumsum(segment_length_m)];
loop_length_m = node_arclength_m(end);
grid_arclength_m = node_arclength_m(1:n_ref);

offset_m = mod(travel_sign * (grid_arclength_m - position_m(1)), loop_length_m);
n_laps = ceil(max(progress_m) / loop_length_m);
target_m = offset_m + loop_length_m * (0:n_laps - 1);  % [n_ref x n_laps]

nearest_index = reshape(knnsearch(progress_m, target_m(:)), size(target_m));
not_reached = target_m > max(progress_m);

value_lap = value(nearest_index);
value_lap(not_reached) = NaN;
value_median = median(value_lap, 2, 'omitnan');

gap_m = hypot(x_m(nearest_index) - centerline_xy_m(:, 1), ...
    y_m(nearest_index) - centerline_xy_m(:, 2));
median_gap_m = median(gap_m(~not_reached));
end
