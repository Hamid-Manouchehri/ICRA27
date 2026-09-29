function [distance_m, path_heading_rad, nearest_index] = ...
    nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m)
%NEAREST_CENTERLINE_DISTANCE_AND_HEADING Track error vs. a reference path.
%
% For each query point i, finds the nearest sample j on the ordered
% reference centerline and returns the distance to it, the local path
% tangent heading at j, and j itself (e.g. to look up a per-sample
% corridor width elsewhere, via normalized_corridor_distance).
%
% Math (per query point i):
%   j(i)             = argmin_k || [x_m(i), y_m(i)] - centerline_xy_m(k,:) ||_2
%   distance_m(i)     = || [x_m(i), y_m(i)] - centerline_xy_m(j(i),:) ||_2
%   path_heading_rad(i) = atan2(cy(j+1) - cy(j), cx(j+1) - cx(j))
%   (j+1 wraps to 1 after the last centerline sample)
%
% Inputs:
%   x_m, y_m: [n x 1] query points, e.g. TCP path samples [m]
%   centerline_xy_m: [n_ref x 2] ordered reference centerline [m]
%
% Outputs:
%   distance_m: [n x 1] distance to the nearest centerline sample [m]
%   path_heading_rad: [n x 1] local centerline tangent heading [rad]
%   nearest_index: [n x 1] row index j(i) into centerline_xy_m

n = numel(x_m);
distance_m = zeros(n, 1);
path_heading_rad = zeros(n, 1);
nearest_index = zeros(n, 1);
n_ref = size(centerline_xy_m, 1);

for i = 1:n
    dist_to_all = hypot(centerline_xy_m(:,1) - x_m(i), ...
        centerline_xy_m(:,2) - y_m(i));
    [distance_m(i), j] = min(dist_to_all);
    nearest_index(i) = j;
    j_next = mod(j, n_ref) + 1;
    path_heading_rad(i) = atan2( ...
        centerline_xy_m(j_next,2) - centerline_xy_m(j,2), ...
        centerline_xy_m(j_next,1) - centerline_xy_m(j,1));
end
end
