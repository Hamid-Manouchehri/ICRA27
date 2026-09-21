function distance_m = distance_to_reference_path( ...
    path_x_m, path_y_m, reference_xy_m)
%DISTANCE_TO_REFERENCE_PATH Nearest-reference-sample distance [m].
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

distance_m = inf(numel(path_x_m), 1);

for reference_idx = 1:size(reference_xy_m, 1)
    candidate_m = hypot( ...
        path_x_m - reference_xy_m(reference_idx, 1), ...
        path_y_m - reference_xy_m(reference_idx, 2));
    distance_m = min(distance_m, candidate_m);
end
end
