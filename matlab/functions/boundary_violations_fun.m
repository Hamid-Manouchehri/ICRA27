function [n_violations, violation_duration_s] = boundary_violations_fun( ...
    normalized_distance, time_s, min_duration_s)
%BOUNDARY_VIOLATIONS_FUN Number of times the TCP left the band and came back.
%
% An excursion is a run of consecutive samples outside the band
% (normalized_distance > 1). It counts only if it is a complete out-and-back:
% it starts after the first sample (the trial did not begin outside) and ends
% before the last sample (the TCP returned), and it lasts at least
% min_duration_s (filters single-sample grazes of the boundary).
%
% Inputs:
%   normalized_distance: [n x 1] distance as a fraction of the corridor
%     half-width (see normalized_corridor_distance)
%   time_s: [n x 1] time [s]
%   min_duration_s: shortest excursion counted [s]
%
% Outputs:
%   n_violations: number of counted excursions
%   violation_duration_s: [n_violations x 1] duration of each [s]

outside = normalized_distance(:) > 1;
change = diff([0; outside; 0]);
first = find(change == 1);
last = find(change == -1) - 1;

duration_s = (last - first + 1) * median(diff(time_s));
counted = first > 1 & last < numel(outside) & duration_s >= min_duration_s;

violation_duration_s = duration_s(counted);
n_violations = numel(violation_duration_s);
end
