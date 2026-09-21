function sample_count = window_to_odd_sample_count( ...
    window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample count.
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

if window_s <= 0
    sample_count = 1;
    return;
end

sample_count = max(3, round(window_s / sample_interval_s));
if mod(sample_count, 2) == 0
    sample_count = sample_count + 1;
end

largest_odd_count = maximum_sample_count;
if mod(largest_odd_count, 2) == 0
    largest_odd_count = largest_odd_count - 1;
end
sample_count = min(sample_count, largest_odd_count);
end
