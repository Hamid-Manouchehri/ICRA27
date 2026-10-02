function mean_force_n = mean_force_fun(fx_n, fy_n, speed_m_s, moving_speed_threshold_m_s)
%MEAN_FORCE_FUN Mean interaction force magnitude while the TCP is moving.
%
% Math:
%   mean_force_n = mean( hypot(fx, fy) ) over samples with speed > threshold
%
% Inputs:
%   fx_n, fy_n: [n x 1] filtered interaction force [N]
%   speed_m_s: [n x 1] TCP speed [m/s]
%   moving_speed_threshold_m_s: samples at or below this speed are skipped [m/s]
%
% Outputs:
%   mean_force_n: mean force magnitude [N]

moving = speed_m_s > moving_speed_threshold_m_s;
mean_force_n = mean(hypot(fx_n(moving), fy_n(moving)));
end
