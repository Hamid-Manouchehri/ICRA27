function jerk = compute_jerk_fine(shapeID,raw_time_s, raw_x_m, raw_y_m, raw_xd_m, raw_yd_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

valid_rows = isfinite(raw_time_s) & isfinite(raw_xd_m) & ...
    isfinite(raw_yd_m) & isfinite(raw_x_m) &isfinite(raw_y_m) ;
clean_time_s = raw_time_s(valid_rows);
clean_xd_m = raw_xd_m(valid_rows);
clean_yd_m = raw_yd_m(valid_rows);
clean_x_m = raw_x_m(valid_rows);
clean_y_m = raw_y_m(valid_rows);

[clean_time_s, sort_order] = sort(clean_time_s);
clean_xd_m = clean_xd_m(sort_order);
clean_yd_m = clean_yd_m(sort_order);
clean_x_m = clean_x_m(sort_order);
clean_y_m = clean_y_m(sort_order);

[clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
clean_xd_m = clean_xd_m(unique_indices);
clean_yd_m = clean_yd_m(unique_indices);
clean_x_m = clean_x_m(unique_indices);
clean_y_m = clean_y_m(unique_indices);

sample_intervals_s = diff(clean_time_s);
median_interval_s = median(sample_intervals_s(sample_intervals_s > 0));

uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';
uniform_xd_m = interp1(clean_time_s, clean_xd_m, uniform_time_s, 'linear');
uniform_yd_m = interp1(clean_time_s, clean_yd_m, uniform_time_s, 'linear');
uniform_x_m = interp1(clean_time_s, clean_x_m, uniform_time_s, 'linear');
uniform_y_m = interp1(clean_time_s, clean_y_m, uniform_time_s, 'linear');

outlier_samples = window_to_odd_sample_count( ...
    outlier_window_s, median_interval_s, numel(uniform_time_s));
smoothing_samples = window_to_odd_sample_count( ...
    smoothing_window_s, median_interval_s, numel(uniform_time_s));

if outlier_samples > 1
    disp('helloooooo')
    uniform_xd_m = movmedian(uniform_xd_m, outlier_samples, ...
        'Endpoints', 'shrink');
    uniform_yd_m = movmedian(uniform_yd_m, outlier_samples, ...
        'Endpoints', 'shrink');
    uniform_x_m = movmedian(uniform_x_m, outlier_samples, ...
        'Endpoints', 'shrink');
    uniform_y_m = movmedian(uniform_y_m, outlier_samples, ...
        'Endpoints', 'shrink');
end

velocity_x_m_s = movmean(uniform_xd_m, smoothing_samples, ...
    'Endpoints', 'shrink');
velocity_y_m_s = movmean(uniform_yd_m, smoothing_samples, ...
    'Endpoints', 'shrink');

acceleration_x_m_s2 = movmean( ...
    gradient(velocity_x_m_s, uniform_time_s), ...
    smoothing_samples, 'Endpoints', 'shrink');
acceleration_y_m_s2 = movmean( ...
    gradient(velocity_y_m_s, uniform_time_s), ...
    smoothing_samples, 'Endpoints', 'shrink');

jerk_x_m_s3 = gradient(acceleration_x_m_s2, uniform_time_s);
jerk_y_m_s3 = gradient(acceleration_y_m_s2, uniform_time_s);

if shapeID ==1
    fineID = uniform_x_m>0.05;
elseif shapeID==2
    fineID = uniform_x_m>0.07;
else
    fineID = uniform_x_m>0.08 | uniform_x_m<-0.11;
end

jerk.time_s = uniform_time_s;
jerk.squared_jerk_m2_s6 = jerk_x_m_s3 .^ 2 + jerk_y_m_s3 .^ 2;
jerk.path_length_m = sum(hypot(diff(uniform_x_m), diff(uniform_y_m)));

ts = find(diff(fineID)>0);
te = find(diff(fineID)<0);
nSeg = min(length(ts),length(te));
f_dsj = 0;

for i=1:nSeg
    J2 = jerk.squared_jerk_m2_s6(ts(i):te(i));
    J2_int = trapz(ts(i), J2);
    A =sum(hypot(diff(uniform_x_m(ts(i):te(i))), diff(uniform_y_m(ts(i):te(i)))));
    dT = uniform_time_s(te(i)) - uniform_time_s(ts(i));
    f_dsj = f_dsj + J2_int * (dT)^5 / A^2;
end

jerk.f_dsj = f_dsj;

end

%{
function jerk = compute_jerk_series( ...
    raw_time_s, raw_xd_m, raw_yd_m, outlier_window_s, smoothing_window_s)
%COMPUTE_JERK_SERIES Filtered planar jerk for the smoothness metric.
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

valid_rows = isfinite(raw_time_s) & isfinite(raw_xd_m) & ...
    isfinite(raw_yd_m);
clean_time_s = raw_time_s(valid_rows);
clean_xd_m = raw_xd_m(valid_rows);
clean_yd_m = raw_yd_m(valid_rows);

[clean_time_s, sort_order] = sort(clean_time_s);
clean_xd_m = clean_xd_m(sort_order);
clean_yd_m = clean_yd_m(sort_order);

[clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
clean_xd_m = clean_xd_m(unique_indices);
clean_yd_m = clean_yd_m(unique_indices);

sample_intervals_s = diff(clean_time_s);
median_interval_s = median(sample_intervals_s(sample_intervals_s > 0));

uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';
uniform_xd_m = interp1(clean_time_s, clean_xd_m, uniform_time_s, 'linear');
uniform_yd_m = interp1(clean_time_s, clean_yd_m, uniform_time_s, 'linear');

outlier_samples = window_to_odd_sample_count( ...
    outlier_window_s, median_interval_s, numel(uniform_time_s));
smoothing_samples = window_to_odd_sample_count( ...
    smoothing_window_s, median_interval_s, numel(uniform_time_s));

if outlier_samples > 1
    uniform_xd_m = movmedian(uniform_xd_m, outlier_samples, ...
        'Endpoints', 'shrink');
    uniform_yd_m = movmedian(uniform_yd_m, outlier_samples, ...
        'Endpoints', 'shrink');
end

velocity_x_m_s = movmean(uniform_xd_m, smoothing_samples, ...
    'Endpoints', 'shrink');
velocity_y_m_s = movmean(uniform_yd_m, smoothing_samples, ...
    'Endpoints', 'shrink');

%velocity_x_m_s = movmean(gradient(filtered_x_m, uniform_time_s), smoothing_samples, 'Endpoints', 'shrink');
%velocity_y_m_s = movmean(gradient(filtered_y_m, uniform_time_s), smoothing_samples, 'Endpoints', 'shrink');

acceleration_x_m_s2 = movmean( ...
    gradient(velocity_x_m_s, uniform_time_s), ...
    smoothing_samples, 'Endpoints', 'shrink');
acceleration_y_m_s2 = movmean( ...
    gradient(velocity_y_m_s, uniform_time_s), ...
    smoothing_samples, 'Endpoints', 'shrink');

jerk_x_m_s3 = gradient(acceleration_x_m_s2, uniform_time_s);
jerk_y_m_s3 = gradient(acceleration_y_m_s2, uniform_time_s);

jerk.time_s = uniform_time_s;
jerk.squared_jerk_m2_s6 = jerk_x_m_s3 .^ 2 + jerk_y_m_s3 .^ 2;
jerk.path_length_m = median_interval_s*sum(hypot(velocity_x_m_s, velocity_y_m_s));
end
%}
