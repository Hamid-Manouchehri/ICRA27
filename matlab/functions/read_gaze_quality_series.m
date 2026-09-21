function [gaze_time_s, pupil_time_s, pupil_diameter_mm, ...
        left_point_valid, right_point_valid] = ...
        read_gaze_quality_series(gaze_table, plot_only_on_display_gaze)
%READ_GAZE_QUALITY_SERIES Per-eye validity and both-eye pupil diameter.
%
% Input:
%   gaze_table - table read from a tobii_*.csv of this project.
%   plot_only_on_display_gaze - true also requires each eye's gaze point
%       to lie inside the normalized display area.
%
% Output:
%   gaze_time_s - trial time of every gaze sample [s].
%   pupil_time_s, pupil_diameter_mm - time and strictly both-eye-averaged
%       pupil diameter [mm]; samples with only one valid pupil are
%       excluded, not substituted, because a single-eye value is not
%       comparable with a two-eye average.
%   left_point_valid, right_point_valid - per-eye gaze-point validity.

    gaze_time_s = get_numeric_column(gaze_table.time_s);

    left_pupil_mm = get_numeric_column(gaze_table.left_pupil_diameter);
    right_pupil_mm = get_numeric_column(gaze_table.right_pupil_diameter);
    left_pupil_valid = get_numeric_column( ...
        gaze_table.left_pupil_validity) == 1 & isfinite(left_pupil_mm);
    right_pupil_valid = get_numeric_column( ...
        gaze_table.right_pupil_validity) == 1 & isfinite(right_pupil_mm);
    both_pupil_valid = left_pupil_valid & right_pupil_valid;

    pupil_time_s = gaze_time_s(both_pupil_valid);
    pupil_diameter_mm = 0.5 * ( ...
        left_pupil_mm(both_pupil_valid) + right_pupil_mm(both_pupil_valid));

    left_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1 & ...
        finite_rows(left_gaze_ucs_mm);
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1 & ...
        finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_display_gaze = parse_vector_column( ...
            gaze_table.left_gaze_point_on_display_area, 2);
        right_display_gaze = parse_vector_column( ...
            gaze_table.right_gaze_point_on_display_area, 2);
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end
end
