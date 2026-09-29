function violation = boundary_violation_magnitude(normalized_distance)
%BOUNDARY_VIOLATION_MAGNITUDE How far outside the corridor, when outside.
%
% Companion to in_band_accuracy: that one reports HOW OFTEN the TCP left
% the band, this one reports HOW BADLY. Excess is measured beyond the
% boundary only and in units of the local corridor half-width, so a
% violation scores the same whether it happened in a narrow or a wide
% section of the shape.
%
% Two means are returned on purpose, because they answer different
% questions and behave differently in a group statistic:
%   - mean_excess_when_outside is severity GIVEN a violation, so it is
%     NaN for a trial that never left the band ("how badly did you
%     violate" is undefined, not zero). Right for describing severity,
%     but NaN drops that trial from a paired test -- and the dropped
%     trials are exactly the clean ones, which biases the comparison.
%   - mean_excess_overall charges 0 to every in-band sample, so it is
%     defined for every trial, is 0 for a perfectly compliant one, and
%     grows with both the frequency and the depth of violations. This is
%     the field to feed a paired test across subjects.
%
% Math (excess measured only where normalized_distance > 1):
%   excess(i)                = normalized_distance(i) - 1
%   mean_excess_when_outside = mean(excess), over violating samples only
%   max_excess               = max(excess)
%   mean_excess_overall      = mean(max(normalized_distance - 1, 0)),
%                              over ALL finite samples
%
% Inputs:
%   normalized_distance: [n x 1] distance as a fraction of the local
%     corridor half-width (from normalized_corridor_distance)
%
% Outputs:
%   violation: struct with fields
%     sample_count - number of samples outside the band [count]
%     mean_excess_when_outside - mean excess while outside, NaN if never
%     max_excess - worst single excursion past the boundary, NaN if never
%     mean_excess_overall - mean one-sided excess over all samples, 0 if never

valid_distance = normalized_distance(isfinite(normalized_distance));
excess = valid_distance - 1;
is_outside = excess > 0;

violation.sample_count = sum(is_outside);
violation.mean_excess_when_outside = mean(excess(is_outside));
violation.max_excess = max(excess(is_outside));
violation.mean_excess_overall = mean(max(excess, 0));

if violation.sample_count == 0
    violation.mean_excess_when_outside = NaN;
    violation.max_excess = NaN;
end
end
