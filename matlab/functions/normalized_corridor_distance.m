function normalized_distance = normalized_corridor_distance( ...
    distance_m, centerline_position, corridor_half_width_m)
%NORMALIZED_CORRIDOR_DISTANCE Track error as a fraction of local corridor half-width.
%
% Raw distance-to-centerline isn't comparable across a shape whose
% corridor width varies along its length (e.g. a narrow "fine" section
% next to a wide "gross" section) -- 3 mm off-center means something
% different in a 5 mm-wide corridor than a 20 mm-wide one. Dividing by
% the LOCAL corridor half-width, interpolated between the two centerline
% points on either side of where the TCP's closest point lies, makes the
% error comparable everywhere: 1.0 means exactly at the corridor boundary.
%
% Math (per query point i, with position s = centerline_position(i),
% k = floor(s), t = s - k, and k+1 wrapping to 1 after the last point):
%   half_width(i)          = (1 - t) * corridor_half_width_m(k) + t * corridor_half_width_m(k+1)
%   normalized_distance(i) = distance_m(i) / half_width(i)
%
% Inputs:
%   distance_m: [n x 1] shortest distance to the centerline [m]
%   centerline_position: [n x 1] fractional index along the centerline for
%     each query point (3rd output of nearest_centerline_distance_and_heading)
%   corridor_half_width_m: [n_ref x 1] local corridor half-width at
%     each centerline point [m]
%
% Outputs:
%   normalized_distance: [n x 1] dimensionless, distance_m / local half-width

n_ref = numel(corridor_half_width_m);

% min(): a position of exactly n_ref + 1 is the last segment at t = 1.
segment_index = min(floor(centerline_position), n_ref);
t = centerline_position - segment_index;
next_index = mod(segment_index, n_ref) + 1;

half_width_m = (1 - t) .* corridor_half_width_m(segment_index) + ...
    t .* corridor_half_width_m(next_index);
normalized_distance = distance_m ./ half_width_m;
end
