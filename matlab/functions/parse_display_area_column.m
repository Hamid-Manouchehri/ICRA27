function xy = parse_display_area_column(text_column)
%PARSE_DISPLAY_AREA_COLUMN Parse a column of "[x,y]" strings (Tobii's
% *_gaze_point_on_display_area cells) into an N-by-2 array, NaN rows
% where the cell is unparseable (e.g. "[NaN,NaN]" during a dropout).

    row_count = numel(text_column);
    xy = nan(row_count, 2);
    for row_idx = 1:row_count
        text = strtrim(char(text_column(row_idx)));
        text = strrep(strrep(text, '[', ''), ']', '');
        parts = strsplit(text, ',');
        if numel(parts) == 2
            xy(row_idx, 1) = str2double(parts{1});
            xy(row_idx, 2) = str2double(parts{2});
        end
    end
end
