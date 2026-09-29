function [distance_m, path_heading_rad, centerline_position] = ...
    nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m)
%NEAREST_CENTERLINE_DISTANCE_AND_HEADING Track error vs. a reference path.
%
% Treats the ordered reference centerline as a closed polyline (the last
% point connects back to the first) and finds, for each query point, the
% closest point ON that polyline -- the foot of the shortest perpendicular
% line, or a segment endpoint when the perpendicular falls past the end.
% Returns the distance to it, the heading of the segment it lies on, and
% where along the centerline it lies (to interpolate a per-point corridor
% width elsewhere, via normalized_corridor_distance).
%
% Math (query point p = [x_m(i), y_m(i)], segment k from a = c(k) to
% b = c(k+1), with k+1 wrapping to 1 after the last point):
%   t_k = clamp( ((p - a) . (b - a)) / |b - a|^2 , 0, 1 )
%   f_k = a + t_k (b - a)                 closest point on segment k
%   k*  = argmin_k || p - f_k ||_2
%   distance_m(i)          = || p - f_k* ||_2
%   path_heading_rad(i)    = atan2(b_y - a_y, b_x - a_x) of segment k*
%   centerline_position(i) = k* + t_k*    in [1, n_ref + 1]
%
% Inputs:
%   x_m, y_m: [n x 1] query points, e.g. TCP path samples [m]
%   centerline_xy_m: [n_ref x 2] ordered reference centerline [m]
%
% Outputs:
%   distance_m: [n x 1] shortest distance to the centerline [m]
%   path_heading_rad: [n x 1] heading of the closest segment [rad]
%   centerline_position: [n x 1] fractional index along centerline_xy_m;
%     k* + t_k* means a fraction t_k* of the way from point k* to k*+1

n = numel(x_m);
n_ref = size(centerline_xy_m, 1);

segment_start_xy_m = centerline_xy_m;
segment_vector_xy_m = centerline_xy_m([2:n_ref, 1], :) - centerline_xy_m;
segment_length_sq_m2 = sum(segment_vector_xy_m .^ 2, 2);
segment_heading_rad = atan2(segment_vector_xy_m(:, 2), segment_vector_xy_m(:, 1));

distance_m = zeros(n, 1);
path_heading_rad = zeros(n, 1);
centerline_position = zeros(n, 1);

for i = 1:n
    from_start_x_m = x_m(i) - segment_start_xy_m(:, 1);
    from_start_y_m = y_m(i) - segment_start_xy_m(:, 2);

    t = (from_start_x_m .* segment_vector_xy_m(:, 1) + ...
        from_start_y_m .* segment_vector_xy_m(:, 2)) ./ segment_length_sq_m2;
    t = min(max(t, 0), 1);  % clamp

    dist_to_all = hypot(from_start_x_m - t .* segment_vector_xy_m(:, 1), ...
        from_start_y_m - t .* segment_vector_xy_m(:, 2));
    [distance_m(i), k] = min(dist_to_all);

    path_heading_rad(i) = segment_heading_rad(k);
    centerline_position(i) = k + t(k);
end
end
