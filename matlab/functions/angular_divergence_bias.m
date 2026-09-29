function bias_deg = angular_divergence_bias(angular_divergence_rad)
%ANGULAR_DIVERGENCE_BIAS Signed mean angular divergence -- a consistent heading bias.
%
% Unlike mean(abs(angular_divergence)), which only measures how far off
% heading the TCP typically is, the SIGNED mean reveals a one-sided
% tendency -- e.g. consistently cutting to one side of the intended
% heading rather than tracking it symmetrically.
%
% Sign convention: positive/negative here inherits directly from
% signed_angle_difference(tcp_angle, path_angle), which in turn depends
% on which rotational direction the reference centerline
% (read_shape_band_obj) happens to be ordered in -- it is NOT a fixed
% "left of path"/"right of path" convention across shapes. Compare
% signs only within the same shape and traversal direction, not across
% shapes.
%
% Math:
%   bias_deg = rad2deg( mean(angular_divergence_rad, 'omitnan') )
%
% Inputs:
%   angular_divergence_rad: [n x 1] signed angular divergence [rad]
%
% Outputs:
%   bias_deg: signed mean angular divergence [deg]

bias_deg = rad2deg(mean(angular_divergence_rad, 'omitnan'));
end
