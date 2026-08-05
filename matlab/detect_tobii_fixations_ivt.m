%% Detect fixations in a Tobii recording using the I-VT algorithm.
%
% Input:
%   tobii_fixed_adm_2d_ellipse_s1_d50.csv, containing normalized display
%   gaze, 3-D gaze origins/points in Tobii user coordinates [mm], validity,
%   and system timestamps [us].
%
% Processing:
%   1. Combine all valid eyes at each timestamp.
%   2. Calculate angular gaze velocity from consecutive 3-D gaze directions.
%   3. Mark samples below the I-VT threshold as fixation candidates.
%   4. Group consecutive candidates and reject groups shorter than 100 ms.
%
% Outputs:
%   fixation_table - retained fixation events in the MATLAB workspace.
%   sample_results - I-VT classification for every input sample.
%   A processed CSV with start/end time, duration, mean normalized gaze x/y,
%   and number of samples. The raw Tobii CSV is never modified.
%
% Important tunable parameters:
%   velocity_threshold_deg_s and minimum_fixation_duration_s are under User
%   Settings. I-VT thresholds should be validated for the task and tracker.

clear;
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
input_csv_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'tobii_fixed_adm_2d_ellipse_s2_d100.csv');
output_csv_file = fullfile(script_directory, ...
    '..', 'data', 'processed_data', ...
    'tobii_fixed_adm_2d_ellipse_s2_d100_fixations_ivt.csv');

% TODO I-VT angular-velocity threshold [deg/s]. Recommended initial range:
% 20 to 70 deg/s; tune and report the selected value for each experiment.
velocity_threshold_deg_s = 20.0;  % detects fixation from non-fixation

% TODO minimum retained fixation duration [s]. Requested value: 0.100 s.
minimum_fixation_duration_s = 0.050;  % filter out short fixation events.

if velocity_threshold_deg_s <= 0
    error('velocity_threshold_deg_s must be positive.');
end
if minimum_fixation_duration_s < 0
    error('minimum_fixation_duration_s cannot be negative.');
end

%% Load Tobii data
gaze_table = readtable(input_csv_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_variables = {
    'system_time_stamp'
    'left_gaze_origin_in_user_coordinate_system'
    'left_gaze_origin_validity'
    'left_gaze_point_in_user_coordinate_system'
    'left_gaze_point_on_display_area'
    'left_gaze_point_validity'
    'right_gaze_origin_in_user_coordinate_system'
    'right_gaze_origin_validity'
    'right_gaze_point_in_user_coordinate_system'
    'right_gaze_point_on_display_area'
    'right_gaze_point_validity'
    };
require_table_variables(gaze_table, required_variables);

system_time_us = get_numeric_column(gaze_table.system_time_stamp);
left_display_gaze = parse_vector_column( ...
    gaze_table.left_gaze_point_on_display_area, 2);
right_display_gaze = parse_vector_column( ...
    gaze_table.right_gaze_point_on_display_area, 2);
left_gaze_point_user_mm = parse_vector_column( ...
    gaze_table.left_gaze_point_in_user_coordinate_system, 3);
right_gaze_point_user_mm = parse_vector_column( ...
    gaze_table.right_gaze_point_in_user_coordinate_system, 3);
left_gaze_origin_user_mm = parse_vector_column( ...
    gaze_table.left_gaze_origin_in_user_coordinate_system, 3);
right_gaze_origin_user_mm = parse_vector_column( ...
    gaze_table.right_gaze_origin_in_user_coordinate_system, 3);

left_point_valid = get_numeric_column( ...
    gaze_table.left_gaze_point_validity) == 1;
right_point_valid = get_numeric_column( ...
    gaze_table.right_gaze_point_validity) == 1;
left_origin_valid = get_numeric_column( ...
    gaze_table.left_gaze_origin_validity) == 1;
right_origin_valid = get_numeric_column( ...
    gaze_table.right_gaze_origin_validity) == 1;

first_timestamp_idx = find(isfinite(system_time_us), 1, 'first');
if isempty(first_timestamp_idx)
    error('The Tobii CSV contains no finite system timestamps.');
end
time_s = (system_time_us - system_time_us(first_timestamp_idx)) * 1e-6;

%% Build and combine valid eye-gaze directions
[left_direction, left_eye_valid] = calculate_eye_direction( ...
    left_gaze_origin_user_mm, ...
    left_gaze_point_user_mm, ...
    left_display_gaze, ...
    left_origin_valid, ...
    left_point_valid);
[right_direction, right_eye_valid] = calculate_eye_direction( ...
    right_gaze_origin_user_mm, ...
    right_gaze_point_user_mm, ...
    right_display_gaze, ...
    right_origin_valid, ...
    right_point_valid);

[gaze_x, gaze_y, gaze_direction, combined_gaze_valid] = ...
    combine_valid_eyes( ...
    left_display_gaze, right_display_gaze, ...
    left_direction, right_direction, ...
    left_eye_valid, right_eye_valid);

%% I-VT sample classification
angular_velocity_deg_s = calculate_angular_velocity( ...
    gaze_direction, time_s, combined_gaze_valid);

fixation_candidate = combined_gaze_valid & ...
    isfinite(angular_velocity_deg_s) & ...
    angular_velocity_deg_s <= velocity_threshold_deg_s;

candidate_changes = diff(double([false; fixation_candidate; false]));
candidate_start_indices = find(candidate_changes == 1);
candidate_end_indices = find(candidate_changes == -1) - 1;

sample_classification = repmat("invalid_gaze", height(gaze_table), 1);
sample_classification(combined_gaze_valid) = "non_fixation";
valid_without_velocity = combined_gaze_valid & ...
    ~isfinite(angular_velocity_deg_s);
sample_classification(valid_without_velocity) = "unclassified_velocity";

start_time_s = zeros(0, 1);
end_time_s = zeros(0, 1);
duration_s = zeros(0, 1);
mean_gaze_x = zeros(0, 1);
mean_gaze_y = zeros(0, 1);
number_of_samples = zeros(0, 1);

for candidate_idx = 1:numel(candidate_start_indices)
    start_idx = candidate_start_indices(candidate_idx);
    end_idx = candidate_end_indices(candidate_idx);
    candidate_duration_s = time_s(end_idx) - time_s(start_idx);

    if candidate_duration_s < minimum_fixation_duration_s
        continue;
    end

    fixation_rows = start_idx:end_idx;
    sample_classification(fixation_rows) = "fixation";

    start_time_s(end + 1, 1) = time_s(start_idx);
    end_time_s(end + 1, 1) = time_s(end_idx);
    duration_s(end + 1, 1) = candidate_duration_s;
    mean_gaze_x(end + 1, 1) = mean(gaze_x(fixation_rows));
    mean_gaze_y(end + 1, 1) = mean(gaze_y(fixation_rows));
    number_of_samples(end + 1, 1) = numel(fixation_rows);
end

fixation_table = table( ...
    start_time_s, ...
    end_time_s, ...
    duration_s, ...
    mean_gaze_x, ...
    mean_gaze_y, ...
    number_of_samples);

sample_results = table( ...
    time_s, ...
    gaze_x, ...
    gaze_y, ...
    angular_velocity_deg_s, ...
    fixation_candidate, ...
    sample_classification);

%% Save retained fixation events
output_directory = fileparts(output_csv_file);
if ~isfolder(output_directory)
    mkdir(output_directory);
end
writetable(fixation_table, output_csv_file);

fprintf('Input samples: %d\n', height(gaze_table));
fprintf('Samples with at least one valid eye: %d\n', ...
    nnz(combined_gaze_valid));
fprintf('I-VT velocity threshold: %.2f deg/s\n', ...
    velocity_threshold_deg_s);
fprintf('Minimum fixation duration: %.1f ms\n', ...
    minimum_fixation_duration_s * 1e3);
fprintf('Retained fixations: %d\n', height(fixation_table));
fprintf('Fixation CSV saved to: %s\n', output_csv_file);

%% Plot retained fixations on the normalized display
figure('Name', 'I-VT fixation detection', 'Color', 'w');
hold on;
plot(gaze_x(combined_gaze_valid), gaze_y(combined_gaze_valid), ...
    '.', ...
    'Color', [0.85, 0.10, 0.10], ...
    'DisplayName', 'Valid binocular gaze samples');
rectangle('Position', [0, 0, 1, 1], ...
    'EdgeColor', 'k', ...
    'LineStyle', '--');
plot(NaN, NaN, 'k--', 'DisplayName', 'Active display');

if isempty(fixation_table)
    text(0.5, 0.5, 'No fixation met the I-VT duration criterion', ...
        'HorizontalAlignment', 'center');
else
    maximum_duration_s = max(fixation_table.duration_s);
    fixation_marker_area = 40.0 + ...
        160.0 * fixation_table.duration_s / maximum_duration_s;

    plot(fixation_table.mean_gaze_x, fixation_table.mean_gaze_y, ...
        '-', ...
        'Color', [0.20, 0.20, 0.20], ...
        'LineWidth', 1.0, ...
        'DisplayName', 'Fixation sequence');
    scatter( ...
        fixation_table.mean_gaze_x, ...
        fixation_table.mean_gaze_y, ...
        fixation_marker_area, ...
        [0.10, 0.45, 0.85], ...
        'filled', ...
        'MarkerEdgeColor', 'k', ...
        'DisplayName', 'Retained fixations');
end

grid on;
set(gca, 'YDir', 'reverse');
xlim([0, 1]);
ylim([0, 1]);
xlabel('Normalized display x');
ylabel('Normalized display y');
title(sprintf( ...
    'I-VT fixations: %.1f deg/s threshold, %.0f ms minimum', ...
    velocity_threshold_deg_s, minimum_fixation_duration_s * 1e3));
legend('Location', 'best');

%% Local functions
function require_table_variables(data_table, required_variables)
%REQUIRE_TABLE_VARIABLES Verify that the Tobii table has required columns.
    imported_variables = data_table.Properties.VariableNames;
    missing_variables = setdiff(required_variables, imported_variables);
    if ~isempty(missing_variables)
        error('Tobii CSV is missing columns: %s', ...
            strjoin(missing_variables, ', '));
    end
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like numeric vectors stored in CSV cells.
    if ischar(raw_column)
        raw_column = cellstr(raw_column);
    end

    sample_count = numel(raw_column);
    vector_values = NaN(sample_count, vector_size);
    for sample_idx = 1:sample_count
        if iscell(raw_column)
            raw_value = raw_column{sample_idx};
        else
            raw_value = raw_column(sample_idx);
        end

        if isstring(raw_value)
            raw_value = char(raw_value);
        end
        if isempty(raw_value)
            continue;
        end

        raw_value = strrep(raw_value, '[', '');
        raw_value = strrep(raw_value, ']', '');
        raw_value = strrep(raw_value, ',', ' ');
        parsed_values = sscanf(raw_value, '%f');
        value_count = min(numel(parsed_values), vector_size);
        vector_values(sample_idx, 1:value_count) = ...
            parsed_values(1:value_count)';
    end
end


function [unit_direction, eye_valid] = calculate_eye_direction( ...
        gaze_origin_user_mm, gaze_point_user_mm, display_gaze, ...
        origin_validity, point_validity)
%CALCULATE_EYE_DIRECTION Build one eye's unit gaze direction in user space.
%
% Inputs are N-by-3 origin/point coordinates [mm], N-by-2 normalized display
% gaze, and N-by-1 SDK validity flags. Output direction is unitless N-by-3.
    direction = gaze_point_user_mm - gaze_origin_user_mm;
    direction_norm = sqrt(sum(direction.^2, 2));
    eye_valid = origin_validity & point_validity & ...
        all(isfinite(gaze_origin_user_mm), 2) & ...
        all(isfinite(gaze_point_user_mm), 2) & ...
        all(isfinite(display_gaze), 2) & ...
        isfinite(direction_norm) & direction_norm > 0;

    unit_direction = NaN(size(direction));
    unit_direction(eye_valid, :) = direction(eye_valid, :) ./ ...
        direction_norm(eye_valid);
end


function [gaze_x, gaze_y, gaze_direction, combined_valid] = ...
        combine_valid_eyes( ...
        left_display, right_display, ...
        left_direction, right_direction, ...
        left_valid, right_valid)
%COMBINE_VALID_EYES Average valid normalized gaze and 3-D eye directions.
    sample_count = size(left_display, 1);
    eye_count = double(left_valid) + double(right_valid);
    combined_valid = eye_count > 0;

    gaze_x_sum = zeros(sample_count, 1);
    gaze_y_sum = zeros(sample_count, 1);
    gaze_x_sum(left_valid) = left_display(left_valid, 1);
    gaze_y_sum(left_valid) = left_display(left_valid, 2);
    gaze_x_sum(right_valid) = gaze_x_sum(right_valid) + ...
        right_display(right_valid, 1);
    gaze_y_sum(right_valid) = gaze_y_sum(right_valid) + ...
        right_display(right_valid, 2);

    gaze_x = NaN(sample_count, 1);
    gaze_y = NaN(sample_count, 1);
    gaze_x(combined_valid) = gaze_x_sum(combined_valid) ./ ...
        eye_count(combined_valid);
    gaze_y(combined_valid) = gaze_y_sum(combined_valid) ./ ...
        eye_count(combined_valid);

    direction_sum = zeros(sample_count, 3);
    direction_sum(left_valid, :) = left_direction(left_valid, :);
    direction_sum(right_valid, :) = direction_sum(right_valid, :) + ...
        right_direction(right_valid, :);
    direction_norm = sqrt(sum(direction_sum.^2, 2));
    combined_valid = combined_valid & direction_norm > 0;

    gaze_direction = NaN(sample_count, 3);
    gaze_direction(combined_valid, :) = ...
        direction_sum(combined_valid, :) ./ direction_norm(combined_valid);
    gaze_x(~combined_valid) = NaN;
    gaze_y(~combined_valid) = NaN;
end


function angular_velocity_deg_s = calculate_angular_velocity( ...
        gaze_direction, time_s, gaze_valid)
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s].
%
% Velocity at sample i uses the angular displacement from i-1 to i. The
% first sample of each valid run receives the run's first interval velocity
% so every multi-sample valid run can be classified.
    sample_count = numel(time_s);
    angular_velocity_deg_s = NaN(sample_count, 1);
    sample_interval_s = diff(time_s);
    valid_pair = gaze_valid(1:end-1) & gaze_valid(2:end) & ...
        isfinite(sample_interval_s) & sample_interval_s > 0;

    direction_dot_product = sum( ...
        gaze_direction(1:end-1, :) .* gaze_direction(2:end, :), 2);
    direction_dot_product = min(max(direction_dot_product, -1.0), 1.0);
    angular_step_deg = acosd(direction_dot_product);

    interval_velocity_deg_s = NaN(sample_count - 1, 1);
    interval_velocity_deg_s(valid_pair) = ...
        angular_step_deg(valid_pair) ./ sample_interval_s(valid_pair);
    angular_velocity_deg_s(2:end) = interval_velocity_deg_s;

    valid_changes = diff(double([false; gaze_valid; false]));
    run_starts = find(valid_changes == 1);
    run_ends = find(valid_changes == -1) - 1;
    for run_idx = 1:numel(run_starts)
        start_idx = run_starts(run_idx);
        end_idx = run_ends(run_idx);
        if end_idx > start_idx && ...
                isfinite(angular_velocity_deg_s(start_idx + 1))
            angular_velocity_deg_s(start_idx) = ...
                angular_velocity_deg_s(start_idx + 1);
        end
    end
end
