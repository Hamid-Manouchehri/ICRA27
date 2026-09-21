function resampled_xy_m = resample_closed_loop(loop_xy_m, spacing_m)
%RESAMPLE_CLOSED_LOOP Resample a closed polygon at uniform arclength.
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

closed_xy_m = [loop_xy_m; loop_xy_m(1, :)];
segment_lengths_m = hypot( ...
    diff(closed_xy_m(:, 1)), diff(closed_xy_m(:, 2)));
arclength_m = [0; cumsum(segment_lengths_m)];

keep_mask = [true; segment_lengths_m > 0];
arclength_m = arclength_m(keep_mask);
closed_xy_m = closed_xy_m(keep_mask, :);

total_length_m = arclength_m(end);
sample_count = max(4, round(total_length_m / spacing_m));
query_m = linspace(0, total_length_m, sample_count)';

resampled_xy_m = [ ...
    interp1(arclength_m, closed_xy_m(:, 1), query_m, 'linear'), ...
    interp1(arclength_m, closed_xy_m(:, 2), query_m, 'linear')];
end
