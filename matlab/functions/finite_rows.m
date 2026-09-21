function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.
% Copied from plot_raw_data_chen_vac_2d.m.

    valid = all(isfinite(values), 2);
end
