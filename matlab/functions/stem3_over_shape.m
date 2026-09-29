function stem3_over_shape( ...
    x_m, y_m, metric_value, centerline_xy_m, stride, metric_label)
%STEM3_OVER_SHAPE Per-sample metric as a 3D stem above its (x,y) location.
%
% Draws stem3(x_m, y_m, metric_value) into the current axes, so a
% metric's spikes can be seen WHERE on the shape they happen (e.g. a
% corner vs. a straight run), not just against sample index like a
% plain time-series plot. The reference centerline is drawn at z=0 for
% spatial context. Points are subsampled by stride for render
% speed/readability, matching this script's existing quiver stride.
%
% Positive-valued samples are stemmed in blue, negative-valued samples
% in red, split by sign(metric_value) -- for an always-non-negative
% metric (e.g. distance, which is never negative), every stem is blue
% and the red series is simply empty.
%
% Inputs:
%   x_m, y_m: [n x 1] path sample coordinates [m]
%   metric_value: [n x 1] metric to stem, same length as x_m/y_m
%   centerline_xy_m: [n_ref x 2] reference centerline, drawn at z=0
%   stride: plot every Nth sample
%   metric_label: z-axis label, e.g. 'dist to centerline [m]'
%
% Outputs: none (draws into the current axes)

x_sub = x_m(1:stride:end);
y_sub = y_m(1:stride:end);
v_sub = metric_value(1:stride:end);
is_positive = v_sub >= 0;

hold on;
plot3(centerline_xy_m(:,1), centerline_xy_m(:,2), ...
    zeros(size(centerline_xy_m,1), 1), '-.', 'Color', [0.8 0.6 0], ...
    'DisplayName', 'centerline');
% An empty stem3([],[],[],...) call still perturbs axis autoscaling
% (observed: pulls a limit toward 0 even with no data there), so only
% call it for a series that actually has points -- e.g. distance is
% always >= 0, so its "negative" series is empty every time.
if any(is_positive)
    stem3(x_sub(is_positive), y_sub(is_positive), v_sub(is_positive), ...
        'Color', 'b', 'Marker', '.', 'DisplayName', 'positive');
end
if any(~is_positive)
    stem3(x_sub(~is_positive), y_sub(~is_positive), v_sub(~is_positive), ...
        'Color', 'r', 'Marker', '.', 'DisplayName', 'negative');
end
xlabel('x [m]'); ylabel('y [m]'); zlabel(metric_label);
legend;
view(3);
end
