function normalized_distance = normalized_corridor_distance( ...
    distance_m, nearest_index, corridor_half_width_m)
%NORMALIZED_CORRIDOR_DISTANCE Track error as a fraction of local corridor half-width.
%
% Raw distance-to-centerline isn't comparable across a shape whose
% corridor width varies along its length (e.g. a narrow "fine" section
% next to a wide "gross" section) -- 3 mm off-center means something
% different in a 5 mm-wide corridor than a 20 mm-wide one. Dividing by
% the LOCAL corridor half-width at each point's nearest centerline
% sample makes the error comparable everywhere: 1.0 means exactly at
% the corridor boundary.
%
% Math (per query point i, with j = nearest_index(i)):
%   normalized_distance(i) = distance_m(i) / corridor_half_width_m(j)
%
% Inputs:
%   distance_m: [n x 1] distance to the nearest centerline sample [m]
%   nearest_index: [n x 1] index into corridor_half_width_m for each
%     query point (3rd output of nearest_centerline_distance_and_heading)
%   corridor_half_width_m: [n_ref x 1] local corridor half-width at
%     each centerline sample [m]
%
% Outputs:
%   normalized_distance: [n x 1] dimensionless, distance_m / local half-width

normalized_distance = distance_m ./ corridor_half_width_m(nearest_index);
end
