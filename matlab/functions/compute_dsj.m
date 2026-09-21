function dsj = compute_dsj(time_s, squared_jerk_m2_s6, path_length_m)
%COMPUTE_DSJ Dimensionless squared jerk, paper Eq. (16).
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

if path_length_m <= 0 || numel(time_s) < 2
    dsj = NaN;
    return;
end

movement_duration_s = time_s(end) - time_s(1);
squared_jerk_integral = trapz(time_s, squared_jerk_m2_s6);

dsj = squared_jerk_integral * movement_duration_s ^ 5 / ...
    path_length_m ^ 2;
end
