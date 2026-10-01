function fine_mask = fine_mask_fun(x_m, shape_index)
%FINE_MASK_FUN True where the TCP is in the fine area of the shape.
%
% Geometric rule on the CoppeliaSim x coordinate [m], per shape:
%   fine_gross (1) and pear (2): x > 0.06
%   ellipse (3): x > 0.08 or x < -0.11
%
% Inputs:
%   x_m: [n x 1] TCP x position, CoppeliaSim frame [m]
%   shape_index: shape code (fine_gross:1, pear: 2, ellipse: 3)
%
% Outputs:
%   fine_mask: [n x 1] logical, true inside the fine area

if shape_index == 3
    fine_mask = x_m > 0.08 | x_m < -0.11;
else
    fine_mask = x_m > 0.06;
end
end
