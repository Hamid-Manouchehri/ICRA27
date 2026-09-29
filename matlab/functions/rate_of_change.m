function rate = rate_of_change(signal, time_s)
%RATE_OF_CHANGE Central-difference time derivative of a point-wise signal.
%
% How fast a metric is changing, not just how big it is -- e.g. a
% steadily-large angular divergence (a real, sustained heading offset)
% looks very different in rate from one that oscillates rapidly around
% zero (jittery/uncertain heading), even when their mean magnitude
% matches.
%
% Math:
%   rate(i) = d(signal)/dt at time_s(i), via gradient() (central
%   difference in the interior, one-sided at the two endpoints)
%
% Inputs:
%   signal: [n x 1] point-wise signal, any unit
%   time_s: [n x 1] matching sample times [s]
%
% Outputs:
%   rate: [n x 1] d(signal)/dt, [unit of signal] / s

rate = gradient(signal, time_s);
end
