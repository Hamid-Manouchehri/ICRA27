function traveled_path_m = traveled_path_fun(x_m, y_m)
%TRAVELED_PATH_FUN Length of the path the TCP travelled.
%
% Math:
%   traveled_path_m = sum( hypot(diff(x), diff(y)) )
%
% Inputs:
%   x_m, y_m: [n x 1] TCP position [m]
%
% Outputs:
%   traveled_path_m: path length [m]

traveled_path_m = sum(hypot(diff(x_m), diff(y_m)));
end
