function [sparc_mean, sparc_passes, pass_duration_s] = sparc_fine_fun( ...
    speed_m_s, fine_mask, fs_hz, min_pass_s, padlevel, fc_hz, amp_threshold)
%SPARC_FINE_FUN SPARC of each pass through the fine area, and their mean.
%
% A pass is one run of consecutive true samples in fine_mask. SPARC is taken
% per pass, never over joined passes (joining creates artificial jumps).
% Passes shorter than min_pass_s are dropped.
%
% Inputs:
%   speed_m_s: [n x 1] TCP speed [m/s]
%   fine_mask: [n x 1] logical, true inside the fine area (see fine_mask_fun)
%   fs_hz: sampling frequency [Hz]
%   min_pass_s: shortest pass kept [s]
%   padlevel, fc_hz, amp_threshold: SPARC settings (see sparc_fun)
%
% Outputs:
%   sparc_mean: mean SPARC over the kept passes, NaN if there are none
%   sparc_passes: [passes x 1] SPARC of each kept pass
%   pass_duration_s: [passes x 1] duration of each kept pass [s]

change = diff([0; fine_mask(:); 0]);
first = find(change == 1);
last = find(change == -1) - 1;

duration_s = (last - first + 1) / fs_hz;
kept = duration_s >= min_pass_s;
first = first(kept);
last = last(kept);
pass_duration_s = duration_s(kept);

sparc_passes = zeros(numel(first), 1);
for k = 1:numel(first)
    sparc_passes(k) = sparc_fun(speed_m_s(first(k):last(k)), fs_hz, ...
        padlevel, fc_hz, amp_threshold);
end
sparc_mean = mean(sparc_passes);
end
