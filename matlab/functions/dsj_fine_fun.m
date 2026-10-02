function [dsj_mean, dsj_passes] = dsj_fine_fun( ...
    time_s, x_m, y_m, squared_jerk_m2_s6, fine_mask, min_pass_s)
%DSJ_FINE_FUN Dimensionless squared jerk of each fine-area pass, and their mean.
%
% A pass is one run of consecutive true samples in fine_mask. DSJ is taken per
% pass with that pass's own duration and path length, never over joined
% passes. Passes shorter than min_pass_s are dropped.
%
% Math (per pass, via compute_dsj.m):
%   dsj = integral(squared_jerk dt) * duration^5 / path_length^2
%
% Inputs:
%   time_s, x_m, y_m, squared_jerk_m2_s6: uniform-grid series from jerk_series_fun
%   fine_mask: [n x 1] logical on the same grid (see fine_mask_fun)
%   min_pass_s: shortest pass kept [s]
%
% Outputs:
%   dsj_mean: mean DSJ over the kept passes, NaN if there are none
%   dsj_passes: [passes x 1] DSJ of each kept pass

change = diff([0; fine_mask(:); 0]);
first = find(change == 1);
last = find(change == -1) - 1;

kept = (time_s(last) - time_s(first)) >= min_pass_s;
first = first(kept);
last = last(kept);

dsj_passes = zeros(numel(first), 1);
for k = 1:numel(first)
    pass = first(k):last(k);
    path_length_m = sum(hypot(diff(x_m(pass)), diff(y_m(pass))));
    dsj_passes(k) = compute_dsj(time_s(pass), squared_jerk_m2_s6(pass), path_length_m);
end
dsj_mean = mean(dsj_passes);
end
