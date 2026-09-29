function reversal_rate_per_m = sign_reversal_rate_per_meter( ...
    signed_signal, path_length_m)
%SIGN_REVERSAL_RATE_PER_METER How often a signed signal changes sign, per meter travelled.
%
% Counts direction reversals (e.g. angular divergence swinging from one
% side of the path heading to the other) and normalizes by distance
% travelled rather than trial duration, so trials run at different
% speeds stay comparable. Mirrors this project's existing
% robot_curvature_sign_reversal_rate feature
% (scripts/classifier_feature_bank.py), applied here to a
% tracking-error signal instead of path curvature.
%
% Math:
%   reversal_rate_per_m = count(sign(signed_signal(i+1)) ~= sign(signed_signal(i))) / path_length_m
%
% Inputs:
%   signed_signal: [n x 1] point-wise signed signal (e.g. angular
%     divergence [rad] -- positive/negative meaningful)
%   path_length_m: total distance travelled over the signal's duration [m]
%
% Outputs:
%   reversal_rate_per_m: reversals per meter travelled [1/m], NaN if
%     path_length_m <= 0 or fewer than 2 valid samples

valid_signal = signed_signal(isfinite(signed_signal));
if numel(valid_signal) < 2 || path_length_m <= 0
    reversal_rate_per_m = NaN;
    return;
end
sign_changes = sum(diff(sign(valid_signal)) ~= 0);
reversal_rate_per_m = sign_changes / path_length_m;
end
