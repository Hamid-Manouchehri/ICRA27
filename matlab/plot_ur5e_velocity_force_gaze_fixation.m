%% Plot synchronized UR5e motion, force, and directional gaze fixations.
%
% Inputs:
%   tobii_fixed_adm_2d_ellipse_s1_d50.csv
%     Tobii normalized gaze, 3-D gaze origins/points [mm], validity, and
%     system timestamps [us].
%   ur5e_fixed_adm_2d_ellipse_s1_d50.csv
%     UR5e base-frame TCP position [m], filtered planar force [N], and time
%     [s].
%
% Processing:
%   Combine valid eyes, calculate 3-D angular gaze velocity, apply I-VT and
%   its minimum-duration criterion, calculate measured planar TCP velocity
%   from position derivatives, and align both recordings in relative time.
%
% Outputs:
%   fixation_table       - retained gaze fixation events.
%   gaze_sample_results  - final classification of each Tobii sample.
%   robot_sample_results - instantaneous UR5e velocity, force, and whether
%                          each robot sample occurs during a fixation.
%   A spatial figure showing velocity and force vectors along the CoppeliaSim
%   shape, plus gaze-direction vectors from each fixation to the next.
%   No source file is modified and no output file is written.
%
% Spatial-frame assumption:
%   Robot plots use recorded CoppeliaSim coordinates [m]. Tobii normalized
%   gaze is converted into these meter bounds for the overlay; this gaze
%   conversion is visual and is not a calibrated camera homography.

clear;
close all;
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
gaze_csv_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'tobii_fixed_adm_2d_ellipse_s2_d100.csv');
robot_csv_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'ur5e_fixed_adm_2d_ellipse_s2_d100.csv');

% TODO I-VT threshold [deg/s]. Recommended initial range: 20 to 70 deg/s.
velocity_threshold_deg_s = 20.0;

% TODO minimum retained fixation duration [s]. Current value: 0.050 s.
minimum_fixation_duration_s = 0.050;

% TODO add to Tobii relative time to align gaze with robot time [s].
% Positive values shift gaze samples and fixation events later.
gaze_to_robot_time_offset_s = 0.0;

% TODO plot one robot vector every N samples. Recommended range: 10 to 100.
robot_vector_stride = 10;

% TODO MATLAB quiver display factor [unitless], recommended 0.2 to 1.0.
% Vector components remain in m/s and N; this only prevents plot overlap.
quiver_auto_scale_factor = 0.8;

if velocity_threshold_deg_s <= 0
    error('velocity_threshold_deg_s must be positive.');
end
if minimum_fixation_duration_s < 0
    error('minimum_fixation_duration_s cannot be negative.');
end
if robot_vector_stride < 1 || robot_vector_stride ~= floor(robot_vector_stride)
    error('robot_vector_stride must be a positive integer.');
end
if quiver_auto_scale_factor <= 0
    error('quiver_auto_scale_factor must be positive.');
end

%% Load Tobii gaze data
gaze_table = readtable(gaze_csv_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_gaze_variables = {
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
require_table_variables( ...
    gaze_table, required_gaze_variables, 'Tobii CSV');

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
gaze_time_s = ...
    (system_time_us - system_time_us(first_timestamp_idx)) * 1e-6;
gaze_time_s = gaze_time_s + gaze_to_robot_time_offset_s;

%% Combine valid eyes and calculate directional I-VT fixations
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

angular_velocity_deg_s = calculate_angular_velocity( ...
    gaze_direction, gaze_time_s, combined_gaze_valid);
fixation_candidate = combined_gaze_valid & ...
    isfinite(angular_velocity_deg_s) & ...
    angular_velocity_deg_s <= velocity_threshold_deg_s;

candidate_changes = diff(double([false; fixation_candidate; false]));
candidate_start_indices = find(candidate_changes == 1);
candidate_end_indices = find(candidate_changes == -1) - 1;

gaze_is_fixation = false(height(gaze_table), 1);
fixation_start_time_s = zeros(0, 1);
fixation_end_time_s = zeros(0, 1);
fixation_duration_s = zeros(0, 1);
fixation_mean_gaze_x = zeros(0, 1);
fixation_mean_gaze_y = zeros(0, 1);
fixation_number_of_samples = zeros(0, 1);

for candidate_idx = 1:numel(candidate_start_indices)
    start_idx = candidate_start_indices(candidate_idx);
    end_idx = candidate_end_indices(candidate_idx);
    duration_s = gaze_time_s(end_idx) - gaze_time_s(start_idx);
    if duration_s < minimum_fixation_duration_s
        continue;
    end

    fixation_rows = start_idx:end_idx;
    gaze_is_fixation(fixation_rows) = true;
    fixation_start_time_s(end + 1, 1) = gaze_time_s(start_idx);
    fixation_end_time_s(end + 1, 1) = gaze_time_s(end_idx);
    fixation_duration_s(end + 1, 1) = duration_s;
    fixation_mean_gaze_x(end + 1, 1) = mean(gaze_x(fixation_rows));
    fixation_mean_gaze_y(end + 1, 1) = mean(gaze_y(fixation_rows));
    fixation_number_of_samples(end + 1, 1) = numel(fixation_rows);
end

fixation_table = table( ...
    fixation_start_time_s, ...
    fixation_end_time_s, ...
    fixation_duration_s, ...
    fixation_mean_gaze_x, ...
    fixation_mean_gaze_y, ...
    fixation_number_of_samples, ...
    'VariableNames', { ...
    'start_time_s', ...
    'end_time_s', ...
    'duration_s', ...
    'mean_gaze_x', ...
    'mean_gaze_y', ...
    'number_of_samples'});

gaze_classification = repmat("invalid_gaze", height(gaze_table), 1);
gaze_classification(combined_gaze_valid) = "non_fixation";
gaze_classification(gaze_is_fixation) = "fixation";
gaze_classification(combined_gaze_valid & ...
    ~isfinite(angular_velocity_deg_s)) = "unclassified_velocity";

gaze_sample_results = table( ...
    gaze_time_s, ...
    gaze_x, ...
    gaze_y, ...
    angular_velocity_deg_s, ...
    fixation_candidate, ...
    gaze_is_fixation, ...
    gaze_classification);

%% Load UR5e data and calculate instantaneous planar velocity and force
robot_table = readtable(robot_csv_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_robot_variables = {
    'time_s'
    'x_m'
    'y_m'
    'coppeliasim_x_m'
    'coppeliasim_y_m'
    'Fx_filt_N'
    'Fy_filt_N'
    };
require_table_variables( ...
    robot_table, required_robot_variables, 'UR5e CSV');

robot_time_s = get_numeric_column(robot_table.time_s);
tcp_x_base_m = get_numeric_column(robot_table.x_m);
tcp_y_base_m = get_numeric_column(robot_table.y_m);
coppeliasim_x_m = get_numeric_column(robot_table.coppeliasim_x_m);
coppeliasim_y_m = get_numeric_column(robot_table.coppeliasim_y_m);
force_x_filtered_n = get_numeric_column(robot_table.Fx_filt_N);
force_y_filtered_n = get_numeric_column(robot_table.Fy_filt_N);

[robot_time_s, tcp_x_base_m, tcp_y_base_m, ...
    coppeliasim_x_m, coppeliasim_y_m, ...
    force_x_filtered_n, force_y_filtered_n] = clean_robot_samples( ...
    robot_time_s, tcp_x_base_m, tcp_y_base_m, ...
    coppeliasim_x_m, coppeliasim_y_m, ...
    force_x_filtered_n, force_y_filtered_n);

tcp_velocity_x_m_s = gradient(tcp_x_base_m, robot_time_s);
tcp_velocity_y_m_s = gradient(tcp_y_base_m, robot_time_s);
tcp_speed_m_s = hypot(tcp_velocity_x_m_s, tcp_velocity_y_m_s);
force_magnitude_n = hypot(force_x_filtered_n, force_y_filtered_n);

% Keep the robot trajectory and vectors in the recorded CoppeliaSim scale.
coppeliasim_x_limits_m = [min(coppeliasim_x_m), max(coppeliasim_x_m)];
coppeliasim_y_limits_m = [min(coppeliasim_y_m), max(coppeliasim_y_m)];
coppeliasim_x_range_m = diff(coppeliasim_x_limits_m);
coppeliasim_y_range_m = diff(coppeliasim_y_limits_m);
if coppeliasim_x_range_m <= 0 || coppeliasim_y_range_m <= 0
    error('CoppeliaSim trajectory must span both planar axes.');
end

coppeliasim_velocity_x_m_s = gradient(coppeliasim_x_m, robot_time_s);
coppeliasim_velocity_y_m_s = gradient(coppeliasim_y_m, robot_time_s);

% Logged mapping uses CoppeliaSim x=-robot x and y=-robot y.
force_x_coppeliasim_n = -force_x_filtered_n;
force_y_coppeliasim_n = -force_y_filtered_n;

% Convert Tobii normalized display coordinates back into CoppeliaSim meters.
gaze_x_coppeliasim_m = coppeliasim_x_limits_m(1) + ...
    gaze_x * coppeliasim_x_range_m;
gaze_y_coppeliasim_m = coppeliasim_y_limits_m(1) + ...
    (1.0 - gaze_y) * coppeliasim_y_range_m;
fixation_x_coppeliasim_m = coppeliasim_x_limits_m(1) + ...
    fixation_table.mean_gaze_x * coppeliasim_x_range_m;
fixation_y_coppeliasim_m = coppeliasim_y_limits_m(1) + ...
    (1.0 - fixation_table.mean_gaze_y) * coppeliasim_y_range_m;

robot_is_fixation = false(size(robot_time_s));
for fixation_idx = 1:height(fixation_table)
    robot_is_fixation = robot_is_fixation | ...
        (robot_time_s >= fixation_table.start_time_s(fixation_idx) & ...
        robot_time_s <= fixation_table.end_time_s(fixation_idx));
end

robot_sample_results = table( ...
    robot_time_s, ...
    tcp_x_base_m, ...
    tcp_y_base_m, ...
    coppeliasim_x_m, ...
    coppeliasim_y_m, ...
    tcp_velocity_x_m_s, ...
    tcp_velocity_y_m_s, ...
    tcp_speed_m_s, ...
    coppeliasim_velocity_x_m_s, ...
    coppeliasim_velocity_y_m_s, ...
    force_x_filtered_n, ...
    force_y_filtered_n, ...
    force_x_coppeliasim_n, ...
    force_y_coppeliasim_n, ...
    force_magnitude_n, ...
    robot_is_fixation);

%% Determine the common gaze-robot interval for reporting
finite_gaze_time_s = gaze_time_s(isfinite(gaze_time_s));
analysis_start_s = max(robot_time_s(1), min(finite_gaze_time_s));
analysis_end_s = min(robot_time_s(end), max(finite_gaze_time_s));
if analysis_end_s <= analysis_start_s
    error('The gaze and robot recordings do not overlap in time.');
end

%% Plot spatial velocity, force, and consecutive-fixation vectors
vector_indices = (1:robot_vector_stride:numel(robot_time_s))';

figure('Name', 'Spatial robot and gaze direction vectors', ...
    'Color', 'w', ...
    'Position', [100, 100, 1500, 500]);

subplot(1, 3, 1);
hold on;
plot(coppeliasim_x_m, coppeliasim_y_m, ...
    'Color', [0.70, 0.70, 0.70], ...
    'DisplayName', 'End-effector shape');
quiver( ...
    coppeliasim_x_m(vector_indices), ...
    coppeliasim_y_m(vector_indices), ...
    coppeliasim_velocity_x_m_s(vector_indices), ...
    coppeliasim_velocity_y_m_s(vector_indices), ...
    quiver_auto_scale_factor, ...
    'Color', [0.10, 0.45, 0.85], ...
    'LineWidth', 1.0, ...
    'MaxHeadSize', 1.5, ...
    'DisplayName', 'Instantaneous velocity');
format_coppeliasim_axes( ...
    'Velocity vectors along shape', ...
    coppeliasim_x_limits_m, coppeliasim_y_limits_m);
legend('Location', 'best');

subplot(1, 3, 2);
hold on;
plot(coppeliasim_x_m, coppeliasim_y_m, ...
    'Color', [0.70, 0.70, 0.70], ...
    'DisplayName', 'End-effector shape');
quiver( ...
    coppeliasim_x_m(vector_indices), ...
    coppeliasim_y_m(vector_indices), ...
    force_x_coppeliasim_n(vector_indices), ...
    force_y_coppeliasim_n(vector_indices), ...
    quiver_auto_scale_factor, ...
    'Color', [0.85, 0.25, 0.10], ...
    'LineWidth', 1.0, ...
    'MaxHeadSize', 1.5, ...
    'DisplayName', 'Instantaneous force');
format_coppeliasim_axes( ...
    'Force vectors along shape', ...
    coppeliasim_x_limits_m, coppeliasim_y_limits_m);
legend('Location', 'best');

subplot(1, 3, 3);
hold on;
plot(coppeliasim_x_m, coppeliasim_y_m, ...
    'Color', [0.75, 0.75, 0.75], ...
    'DisplayName', 'End-effector shape');
plot( ...
    gaze_x_coppeliasim_m(combined_gaze_valid), ...
    gaze_y_coppeliasim_m(combined_gaze_valid), ...
    '.', ...
    'Color', [0.85, 0.10, 0.10], ...
    'DisplayName', 'Valid gaze samples');
scatter( ...
    fixation_x_coppeliasim_m, ...
    fixation_y_coppeliasim_m, ...
    35, ...
    [0.10, 0.45, 0.85], ...
    'filled', ...
    'MarkerEdgeColor', 'k', ...
    'DisplayName', 'Fixation centers');

if height(fixation_table) >= 2
    fixation_direction_x_m = diff(fixation_x_coppeliasim_m);
    fixation_direction_y_m = diff(fixation_y_coppeliasim_m);
    quiver( ...
        fixation_x_coppeliasim_m(1:end-1), ...
        fixation_y_coppeliasim_m(1:end-1), ...
        fixation_direction_x_m, ...
        fixation_direction_y_m, ...
        0, ...
        'Color', [0.15, 0.55, 0.20], ...
        'LineWidth', 1.5, ...
        'MaxHeadSize', 0.6, ...
        'DisplayName', 'Direction to next fixation');
end
format_coppeliasim_axes( ...
    'Direction between consecutive fixations', ...
    coppeliasim_x_limits_m, coppeliasim_y_limits_m);
legend('Location', 'best');

fprintf('Gaze samples: %d\n', height(gaze_table));
fprintf('Robot samples: %d\n', height(robot_table));
fprintf('Retained fixations: %d\n', height(fixation_table));
fprintf('Common analyzed interval: %.3f to %.3f s\n', ...
    analysis_start_s, analysis_end_s);
fprintf('Gaze-to-robot time offset: %.6f s\n', ...
    gaze_to_robot_time_offset_s);

%% Local functions
function require_table_variables( ...
        data_table, required_variables, table_label)
%REQUIRE_TABLE_VARIABLES Verify required input columns are present.
    imported_variables = data_table.Properties.VariableNames;
    missing_variables = setdiff(required_variables, imported_variables);
    if ~isempty(missing_variables)
        error('%s is missing columns: %s', ...
            table_label, strjoin(missing_variables, ', '));
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
%CALCULATE_EYE_DIRECTION Build one eye's unit direction in Tobii user space.
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
%COMBINE_VALID_EYES Average valid display gaze and 3-D eye directions.
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


function [time_s, x_m, y_m, sim_x_m, sim_y_m, ...
        force_x_n, force_y_n] = clean_robot_samples( ...
        time_s, x_m, y_m, sim_x_m, sim_y_m, force_x_n, force_y_n)
%CLEAN_ROBOT_SAMPLES Remove invalid samples and sort unique robot times.
    valid = isfinite(time_s) & isfinite(x_m) & isfinite(y_m) & ...
        isfinite(sim_x_m) & isfinite(sim_y_m) & ...
        isfinite(force_x_n) & isfinite(force_y_n);
    time_s = time_s(valid);
    x_m = x_m(valid);
    y_m = y_m(valid);
    sim_x_m = sim_x_m(valid);
    sim_y_m = sim_y_m(valid);
    force_x_n = force_x_n(valid);
    force_y_n = force_y_n(valid);

    [time_s, sort_order] = sort(time_s);
    x_m = x_m(sort_order);
    y_m = y_m(sort_order);
    sim_x_m = sim_x_m(sort_order);
    sim_y_m = sim_y_m(sort_order);
    force_x_n = force_x_n(sort_order);
    force_y_n = force_y_n(sort_order);

    [time_s, unique_indices] = unique(time_s, 'stable');
    x_m = x_m(unique_indices);
    y_m = y_m(unique_indices);
    sim_x_m = sim_x_m(unique_indices);
    sim_y_m = sim_y_m(unique_indices);
    force_x_n = force_x_n(unique_indices);
    force_y_n = force_y_n(unique_indices);

    if numel(time_s) < 2
        error('UR5e CSV has fewer than two valid time samples.');
    end
end


function format_coppeliasim_axes( ...
        plot_title, x_limits_m, y_limits_m)
%FORMAT_COPPELIASIM_AXES Format an actual-scale CoppeliaSim x-y plot.
    x_margin_m = 0.05 * diff(x_limits_m);
    y_margin_m = 0.05 * diff(y_limits_m);
    grid on;
    axis equal;
    xlim([x_limits_m(1) - x_margin_m, x_limits_m(2) + x_margin_m]);
    ylim([y_limits_m(1) - y_margin_m, y_limits_m(2) + y_margin_m]);
    xlabel('CoppeliaSim x [m]');
    ylabel('CoppeliaSim y [m]');
    title(plot_title);
end
