function [sparc, f_hz, mag, f_sel_hz, mag_sel] = sparc_fun( ...
    speed_m_s, fs_hz, padlevel, fc_hz, amp_threshold)
%SPARC_FUN Spectral arc length (SPARC) smoothness of a speed profile.
%
% Balasubramanian, Melendez-Calderon, Burdet (2012), "A robust and sensitive
% metric for quantifying movement smoothness", IEEE Trans Biomed Eng
% 59(8):2126-2136. Follows the authors' reference implementation
% (github.com/siva82kb/SPARC, scripts/smoothness.py). Tested by the
% authors for discrete movements only. More negative = less smooth.
%
% Math:
%   V(f)   = |FFT(speed, nfft)|,  nfft = 2^(ceil(log2(n)) + padlevel)
%   Vn(f)  = V(f) / max(V)
%   band   = f(first) .. f(last), among f <= fc_hz, where Vn >= amp_threshold
%   sparc  = - sum_k sqrt( (df_k / (f_last - f_first))^2 + dVn_k^2 )
%
% Inputs:
%   speed_m_s: [n x 1] speed profile of the movement [m/s]
%   fs_hz: sampling frequency [Hz]
%   padlevel: zero-padding level (reference default 4)
%   fc_hz: max cutoff frequency [Hz] (reference default 10)
%   amp_threshold: adaptive cutoff amplitude threshold (reference default 0.05)
%
% Outputs:
%   sparc: spectral arc length [dimensionless]
%   f_hz, mag: full frequency axis [Hz] and normalized magnitude spectrum
%   f_sel_hz, mag_sel: the part of the spectrum the arc length is taken over

nfft = 2 ^ (ceil(log2(numel(speed_m_s))) + padlevel);
f_hz = (0:nfft - 1)' * fs_hz / nfft;
mag = abs(fft(speed_m_s(:), nfft));
mag = mag / max(mag);

below_cutoff = find(f_hz <= fc_hz);
above_threshold = find(mag(below_cutoff) >= amp_threshold);
selected = below_cutoff(above_threshold(1):above_threshold(end));
f_sel_hz = f_hz(selected);
mag_sel = mag(selected);

sparc = -sum(sqrt((diff(f_sel_hz) / (f_sel_hz(end) - f_sel_hz(1))) .^ 2 + ...
    diff(mag_sel) .^ 2));
end
