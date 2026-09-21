function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes.
% Copied from plot_raw_data_chen_vac_2d.m.

    sample_count = size(left_point, 1);
    valid_eye_count = double(left_valid) + double(right_valid);
    combined_valid = valid_eye_count > 0;

    point_sum = zeros(sample_count, 3);
    point_sum(left_valid, :) = left_point(left_valid, :);
    point_sum(right_valid, :) = point_sum(right_valid, :) + ...
        right_point(right_valid, :);

    combined_point = NaN(sample_count, 3);
    combined_point(combined_valid, :) = ...
        point_sum(combined_valid, :) ./ valid_eye_count(combined_valid);
end
