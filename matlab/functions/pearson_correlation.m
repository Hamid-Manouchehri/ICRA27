function r = pearson_correlation(signal_a, signal_b)
%PEARSON_CORRELATION Pearson correlation coefficient between two point-wise signals.
%
% Uses base-MATLAB corrcoef (no Statistics Toolbox dependency), matching
% this project's existing convention (test_gaze_anticipation_hypothesis.m).
%
% Math:
%   r = cov(signal_a, signal_b) / (std(signal_a) * std(signal_b))
%
% Inputs:
%   signal_a, signal_b: [n x 1] point-wise signals, same length, paired
%     sample-by-sample (e.g. angular divergence and TCP speed)
%
% Outputs:
%   r: Pearson correlation coefficient in [-1, 1]; NaN if either signal
%     is constant or fewer than 2 valid paired samples remain

valid = isfinite(signal_a) & isfinite(signal_b);
if nnz(valid) < 2
    r = NaN;
    return;
end
c = corrcoef(signal_a(valid), signal_b(valid));
r = c(1, 2);
end
