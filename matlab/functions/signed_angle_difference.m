function diff_rad = signed_angle_difference(angle_a_rad, angle_b_rad)
%SIGNED_ANGLE_DIFFERENCE Shortest signed angle from b to a, in [-pi, pi].
%
% Math:
%   diff_rad = atan2(sin(angle_a_rad - angle_b_rad), cos(angle_a_rad - angle_b_rad))
%
% Wraps the plain subtraction (angle_a_rad - angle_b_rad) so it can't
% jump across the +-pi seam, e.g. a_rad = -179 deg, b_rad = +179 deg
% gives +2 deg, not -358 deg.
%
% Inputs:
%   angle_a_rad, angle_b_rad: angles [rad], any real value
%
% Outputs:
%   diff_rad: wrapped signed difference in [-pi, pi] [rad]

diff_rad = atan2(sin(angle_a_rad - angle_b_rad), cos(angle_a_rad - angle_b_rad));
end
