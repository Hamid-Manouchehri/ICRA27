function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display bounds.
% Copied from plot_raw_data_chen_vac_2d.m.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end
