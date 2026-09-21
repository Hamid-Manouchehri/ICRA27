function column_idx = find_column(column_names, name)
%FIND_COLUMN Index of name in column_names, or a clear error.

    column_idx = find(strcmp(column_names, name), 1);
    if isempty(column_idx)
        error('Column "%s" not found in x_column_names.', name);
    end
end
