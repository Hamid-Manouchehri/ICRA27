function in_band_percent = in_band_accuracy(normalized_distance)
%IN_BAND_ACCURACY Percentage of samples with the TCP inside the corridor.
%
% The task's own success criterion: "stay inside the band". Unlike mean
% distance-to-centerline (Chen Eq. (15)), which charges an error for any
% deviation from the centre even while the TCP sits comfortably inside
% the corridor, this scores only what the task actually asks for. It is
% corridor-relative, so it stays comparable across shapes, across the
% narrow/wide bandwidth conditions, and across the fine/gross regions of
% one shape -- none of which is true of an error in millimetres.
%
% Math:
%   in_band_percent = 100 * count(normalized_distance <= 1) / count(finite samples)
%   normalized_distance == 1 is exactly ON the boundary and counts as in
%   band (see normalized_corridor_distance).
%
% Inputs:
%   normalized_distance: [n x 1] distance as a fraction of the local
%     corridor half-width (from normalized_corridor_distance)
%
% Outputs:
%   in_band_percent: percentage of valid samples inside the band [%]

valid = isfinite(normalized_distance);
in_band_percent = 100 * sum(normalized_distance(valid) <= 1) / sum(valid);
end
