function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied from plot_raw_data_chen_vac_2d.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end
