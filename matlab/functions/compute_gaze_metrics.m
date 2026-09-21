function metrics = compute_gaze_metrics(gaze_csv_file, ...
        require_both_eyes_valid, require_gaze_on_display, ...
        max_gaze_gap_for_speed_s)
%COMPUTE_GAZE_METRICS Simple gaze-behavior summary for one trial, from
% the raw Tobii CSV alone -- normalized display-area coordinates
% ([0, 1] x [0, 1]), both eyes combined per require_both_eyes_valid.
%
% Output: scalar struct -- sample_count, valid_fraction,
%   mean_pupil_diameter_mm, gaze_dispersion, mean_gaze_speed_per_s.

    gaze_table = readtable(gaze_csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');

    time_s = gaze_table.time_s;
    left_xy = parse_display_area_column( ...
        gaze_table.left_gaze_point_on_display_area);
    right_xy = parse_display_area_column( ...
        gaze_table.right_gaze_point_on_display_area);
    left_point_valid = gaze_table.left_gaze_point_validity == 1;
    right_point_valid = gaze_table.right_gaze_point_validity == 1;

    if require_gaze_on_display
        left_on_display = all( ...
            isfinite(left_xy) & left_xy >= 0 & left_xy <= 1, 2);
        right_on_display = all( ...
            isfinite(right_xy) & right_xy >= 0 & right_xy <= 1, 2);
        left_point_valid = left_point_valid & left_on_display;
        right_point_valid = right_point_valid & right_on_display;
    end

    row_count = height(gaze_table);
    gaze_xy = nan(row_count, 2);
    if require_both_eyes_valid
        gaze_valid = left_point_valid & right_point_valid;
        gaze_xy(gaze_valid, :) = 0.5 * ( ...
            left_xy(gaze_valid, :) + right_xy(gaze_valid, :));
    else
        gaze_valid = left_point_valid | right_point_valid;
        both = left_point_valid & right_point_valid;
        only_left = left_point_valid & ~right_point_valid;
        only_right = right_point_valid & ~left_point_valid;
        gaze_xy(both, :) = 0.5 * (left_xy(both, :) + right_xy(both, :));
        gaze_xy(only_left, :) = left_xy(only_left, :);
        gaze_xy(only_right, :) = right_xy(only_right, :);
    end

    metrics.sample_count = row_count;
    metrics.valid_fraction = mean(gaze_valid);

    valid_xy = gaze_xy(gaze_valid, :);
    valid_time_s = time_s(gaze_valid);
    metrics.gaze_dispersion = mean(std(valid_xy, 0, 1));

    if numel(valid_time_s) > 1
        gaze_step = hypot(diff(valid_xy(:, 1)), diff(valid_xy(:, 2)));
        time_step_s = diff(valid_time_s);
        step_ok = time_step_s > 0 & time_step_s <= max_gaze_gap_for_speed_s;
        metrics.mean_gaze_speed_per_s = ...
            mean(gaze_step(step_ok) ./ time_step_s(step_ok));
    else
        metrics.mean_gaze_speed_per_s = NaN;
    end

    left_pupil_valid = gaze_table.left_pupil_validity == 1;
    right_pupil_valid = gaze_table.right_pupil_validity == 1;
    both_pupil_valid = left_pupil_valid & right_pupil_valid;
    metrics.mean_pupil_diameter_mm = mean(0.5 * ( ...
        gaze_table.left_pupil_diameter(both_pupil_valid) + ...
        gaze_table.right_pupil_diameter(both_pupil_valid)));
end
