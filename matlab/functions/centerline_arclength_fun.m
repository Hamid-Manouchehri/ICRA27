function [arclength_m, progress_m, travel_sign] = centerline_arclength_fun( ...
    x_m, y_m, centerline_xy_m)
%CENTERLINE_ARCLENGTH_FUN Position of each TCP sample along the centerline.
%
% Arclength from the first centerline point to the closest centerline point
% of each sample; same lines as in path_efficiency_fun.m. Progress is the
% same position unwrapped over laps, counted from the first sample.
%
% Math:
%   s(i)          = arclength of the closest centerline point to sample i
%   ds(i)         = s(i+1) - s(i), wrapped into [-L/2, L/2], L = loop length
%   progress_m(i) = sum(ds(1:i-1)), sign set so the net travel is positive
%
% Inputs:
%   x_m, y_m: [n x 1] TCP position, same frame as the centerline [m]
%   centerline_xy_m: [n_ref x 2] ordered closed centerline [m]
%
% Outputs:
%   arclength_m: [n x 1] position along the centerline, in [0, loop length] [m]
%   progress_m: [n x 1] distance along the centerline since the first
%     sample, over all laps [m]
%   travel_sign: +1 if the net travel is toward increasing arclength, else -1

[~, ~, position] = nearest_centerline_distance_and_heading(x_m, y_m, ...
    centerline_xy_m);

n_ref = size(centerline_xy_m, 1);
closed_xy_m = centerline_xy_m([1:n_ref, 1], :);
segment_length_m = hypot(diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
node_arclength_m = [0; cumsum(segment_length_m)];
loop_length_m = node_arclength_m(end);

k = min(floor(position), n_ref);
arclength_m = node_arclength_m(k) + (position - k) .* segment_length_m(k);

step_m = diff(arclength_m);
step_m = step_m - loop_length_m * round(step_m / loop_length_m);
progress_m = [0; cumsum(step_m)];
travel_sign = sign(progress_m(end));
progress_m = progress_m * travel_sign;
end
