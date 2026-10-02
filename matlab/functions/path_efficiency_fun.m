function [efficiency, progress_m, tcp_length_m] = path_efficiency_fun( ...
    x_m, y_m, centerline_xy_m)
%PATH_EFFICIENCY_FUN Net progress along the centerline per metre of TCP path.
%
% Each TCP sample is projected onto the closed centerline to get its arclength
% position. Steps are wrapped the short way around the loop; their sum is the
% net progress. Lateral wobble and back-tracking lengthen the TCP path without
% adding progress, so a perfect trace gives about 1.
%
% Math:
%   s(i)         = arclength of the closest centerline point to the TCP sample i
%   ds(i)        = s(i+1) - s(i), wrapped into [-L/2, L/2], L = loop length
%   progress_m   = | sum(ds) |
%   tcp_length_m = sum( hypot(diff(x), diff(y)) )
%   efficiency   = progress_m / tcp_length_m
%
% Inputs:
%   x_m, y_m: [n x 1] TCP position, same frame as the centerline [m]
%   centerline_xy_m: [n_ref x 2] ordered closed centerline [m]
%
% Outputs:
%   efficiency: progress_m / tcp_length_m [dimensionless]
%   progress_m: net distance advanced along the centerline [m]
%   tcp_length_m: TCP path length [m]

[~, ~, position] = nearest_centerline_distance_and_heading(x_m, y_m, centerline_xy_m);

n_ref = size(centerline_xy_m, 1);
closed_xy_m = centerline_xy_m([1:n_ref, 1], :);
segment_length_m = hypot(diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
node_arclength_m = [0; cumsum(segment_length_m)];
loop_length_m = node_arclength_m(end);

k = min(floor(position), n_ref);
arclength_m = node_arclength_m(k) + (position - k) .* segment_length_m(k);

step_m = diff(arclength_m);
step_m = step_m - loop_length_m * round(step_m / loop_length_m);

progress_m = abs(sum(step_m));
tcp_length_m = sum(hypot(diff(x_m), diff(y_m)));
efficiency = progress_m / tcp_length_m;
end
