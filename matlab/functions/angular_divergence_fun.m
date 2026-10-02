function [theta_deg, tcp_chord_length_m, center_chord_length_m, tcp_heading_deg] = angular_divergence_fun( ...
    x_m, y_m, centerline_position, centerline_xy_m, station_spacing_m)
%ANGULAR_DIVERGENCE_FUN Angle between the TCP path and the centerline, per interval.
%
% Stations are placed every station_spacing_m along the centerline, in the
% direction of the TCP's net progress. For each station the TCP point is the
% first sample whose progress reaches it. For two consecutive stations, theta is
% the angle from the centerline chord (between the two stations) to the TCP
% chord (between the two TCP points). Intervals whose two stations were passed
% at the same sample are skipped.
%
% Math (consecutive stations k, k+1; TCP points p_k, p_k+1; centerline points c_k, c_k+1):
%   theta_k = atan2(p_k+1 - p_k) - atan2(c_k+1 - c_k), wrapped into [-pi, pi]
%
% Inputs:
%   x_m, y_m: [n x 1] TCP position, same frame as the centerline [m]
%   centerline_position: [n x 1] fractional centerline index of each sample
%     (3rd output of nearest_centerline_distance_and_heading)
%   centerline_xy_m: [n_ref x 2] ordered closed centerline [m]
%   station_spacing_m: centerline distance between consecutive stations [m]
%
% Outputs:
%   theta_deg: [intervals x 1] signed angle per interval [deg]
%   tcp_chord_length_m: [intervals x 1] distance between the two TCP points
%     of each interval [m]
%   center_chord_length_m: [intervals x 1] distance between the two stations
%     of each interval on the centerline [m]
%   tcp_heading_deg: [intervals x 1] direction of the TCP chord of each
%     interval measured from the horizontal (+x) axis, in [-180, 180] [deg]

n_ref = size(centerline_xy_m, 1);
closed_xy_m = centerline_xy_m([1:n_ref, 1], :);
segment_length_m = hypot(diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
node_arclength_m = [0; cumsum(segment_length_m)];
loop_length_m = node_arclength_m(end);

k = min(floor(centerline_position), n_ref);
arclength_m = node_arclength_m(k) + (centerline_position - k) .* segment_length_m(k);

% Progress along the loop in the direction of travel (laps accumulate).
step_m = diff(arclength_m);
step_m = step_m - loop_length_m * round(step_m / loop_length_m);
progress_m = [0; cumsum(step_m)];
travel_sign = sign(progress_m(end));
furthest_m = cummax(travel_sign * progress_m);

% Stations, and the first TCP sample that reached each one.
station_progress_m = (0:floor(furthest_m(end) / station_spacing_m))' * station_spacing_m;
tcp_index = arrayfun(@(q) find(furthest_m >= q, 1), station_progress_m);

% The same stations on the centerline.
station_arclength_m = mod(arclength_m(1) + travel_sign * station_progress_m, loop_length_m);
station_xy_m = [interp1(node_arclength_m, closed_xy_m(:, 1), station_arclength_m), ...
    interp1(node_arclength_m, closed_xy_m(:, 2), station_arclength_m)];

tcp_chord_m = diff([x_m(tcp_index), y_m(tcp_index)]);
center_chord_m = diff(station_xy_m);
theta_rad = signed_angle_difference( ...
    atan2(tcp_chord_m(:, 2), tcp_chord_m(:, 1)), ...
    atan2(center_chord_m(:, 2), center_chord_m(:, 1)));

moved = diff(tcp_index) > 0;
theta_deg = rad2deg(theta_rad(moved));
tcp_chord_length_m = hypot(tcp_chord_m(moved, 1), tcp_chord_m(moved, 2));
center_chord_length_m = hypot(center_chord_m(moved, 1), center_chord_m(moved, 2));
tcp_heading_deg = rad2deg(atan2(tcp_chord_m(moved, 2), tcp_chord_m(moved, 1)));
end
