function [combined_direction, combined_valid] = ...
        combine_valid_eye_directions( ...
        left_direction, right_direction, left_valid, right_valid)
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye directions.
% Copied from plot_raw_data_chen_vac_2d.m.

    sample_count = size(left_direction, 1);
    combined_valid = left_valid | right_valid;

    direction_sum = zeros(sample_count, 3);
    direction_sum(left_valid, :) = left_direction(left_valid, :);
    direction_sum(right_valid, :) = direction_sum(right_valid, :) + ...
        right_direction(right_valid, :);

    direction_norm = sqrt(sum(direction_sum .^ 2, 2));
    combined_valid = combined_valid & isfinite(direction_norm) & ...
        direction_norm > 0;

    combined_direction = NaN(sample_count, 3);
    combined_direction(combined_valid, :) = ...
        direction_sum(combined_valid, :) ./ direction_norm(combined_valid);
end
