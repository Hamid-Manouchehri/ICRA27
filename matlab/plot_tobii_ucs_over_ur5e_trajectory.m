%% Overlay Tobii UCS gaze points on the UR5e/CoppeliaSim trajectory.
%
% Inputs:
%   tobii_fixed_adm_2d_random_shape_d100.csv
%     Left/right gaze points in Tobii User Coordinate System (UCS) [mm],
%     display-area coordinates, and validity flags.
%   ur5e_fixed_adm_2d_random_shape_d100.csv
%     End-effector trajectory in the CoppeliaSim x-y frame [m].
%
% Output:
%   One spatial plot containing the robot trajectory and combined valid gaze
%   points in the CoppeliaSim x-y frame. No input data is modified.
%
% Assumptions:
%   The CoppeliaSim view is orthographic, unrotated, and fixed during data
%   collection. The view zoom is adjusted so a 1 cm robot motion appears as
%   1 cm on the physical display. Tobii UCS +x/+y are aligned with and point
%   in the same directions as CoppeliaSim +x/+y. The Tobii UCS z component
%   is not used because the displayed shape lies in the CoppeliaSim x-y
%   plane. Raw gaze is expected to form a band around the robot trajectory
%   rather than coincide sample-for-sample with it.
%
% User settings:
%   alignment_mode selects either an approximate data-center translation or
%   a calibrated manual reference pair. Use "manual_reference" for absolute
%   spatial interpretation after measuring one corresponding point.

clear;
close all;
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
gaze_csv_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'tobii_fixed_adm_2d_random_shape_d100.csv');
robot_csv_file = fullfile(script_directory, ...
    '..', 'data', 'raw_data', ...
    'ur5e_fixed_adm_2d_random_shape_d100.csv');

% TODO physical screen motion divided by robot motion [unitless], must be > 0.
% Keep at 1.0 while the CoppeliaSim view is adjusted to physical 1:1 scale.
screen_motion_per_robot_motion = 1.0;

% TODO Tobii UCS axis signs in CoppeliaSim [unitless], each must be -1 or +1.
% The current aligned setup uses the same positive directions on both axes.
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO translation method: "data_centers" or "manual_reference".
% "data_centers" makes the plot immediately usable but removes the mean
% gaze-to-robot offset. It is not an absolute spatial calibration.
alignment_mode = "data_centers";

% TODO one corresponding Tobii/CoppeliaSim point used only when
% alignment_mode="manual_reference". Enter finite [x, y] values in [mm]
% and [m], respectively.
manual_tobii_reference_xy_mm = [NaN, NaN];  % mm
manual_coppelia_reference_xy_m = [NaN, NaN];  % m

% TODO true excludes gaze estimates outside normalized display [0, 1]^2.
plot_only_on_display_gaze = true;

% TODO raw gaze marker size [points^2], recommended range: 2 to 20.
gaze_marker_size = 8;

if screen_motion_per_robot_motion <= 0
    error('screen_motion_per_robot_motion must be positive.');
end
if any(abs(tobii_to_coppelia_axis_sign) ~= 1)
    error('Each Tobii-to-CoppeliaSim axis sign must be -1 or +1.');
end

%% Load and combine valid Tobii gaze points
gaze_table = readtable(gaze_csv_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_gaze_variables = {
    'left_gaze_point_in_user_coordinate_system'
    'left_gaze_point_on_display_area'
    'left_gaze_point_validity'
    'right_gaze_point_in_user_coordinate_system'
    'right_gaze_point_on_display_area'
    'right_gaze_point_validity'
    };
require_table_variables( ...
    gaze_table, required_gaze_variables, 'Tobii CSV');

left_gaze_ucs_mm = parse_vector_column( ...
    gaze_table.left_gaze_point_in_user_coordinate_system, 3);
right_gaze_ucs_mm = parse_vector_column( ...
    gaze_table.right_gaze_point_in_user_coordinate_system, 3);
left_display_gaze = parse_vector_column( ...
    gaze_table.left_gaze_point_on_display_area, 2);
right_display_gaze = parse_vector_column( ...
    gaze_table.right_gaze_point_on_display_area, 2);

left_point_valid = get_numeric_column( ...
    gaze_table.left_gaze_point_validity) == 1;
right_point_valid = get_numeric_column( ...
    gaze_table.right_gaze_point_validity) == 1;

left_point_valid = left_point_valid & finite_rows(left_gaze_ucs_mm);
right_point_valid = right_point_valid & finite_rows(right_gaze_ucs_mm);

if plot_only_on_display_gaze
    left_point_valid = left_point_valid & ...
        point_is_on_display(left_display_gaze);
    right_point_valid = right_point_valid & ...
        point_is_on_display(right_display_gaze);
end

[combined_gaze_ucs_mm, combined_gaze_valid] = combine_valid_eye_points( ...
    left_gaze_ucs_mm, right_gaze_ucs_mm, ...
    left_point_valid, right_point_valid);

if ~any(combined_gaze_valid)
    error('The Tobii CSV contains no valid gaze points for plotting.');
end

%% Load the robot trajectory in the CoppeliaSim frame
robot_table = readtable(robot_csv_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_robot_variables = {
    'coppeliasim_x_m'
    'coppeliasim_y_m'
    };
require_table_variables( ...
    robot_table, required_robot_variables, 'UR5e CSV');

coppeliasim_x_m = get_numeric_column(robot_table.coppeliasim_x_m);
coppeliasim_y_m = get_numeric_column(robot_table.coppeliasim_y_m);
robot_position_valid = isfinite(coppeliasim_x_m) & ...
    isfinite(coppeliasim_y_m);
coppeliasim_xy_m = [ ...
    coppeliasim_x_m(robot_position_valid), ...
    coppeliasim_y_m(robot_position_valid)];

if isempty(coppeliasim_xy_m)
    error('The UR5e CSV contains no finite CoppeliaSim x-y positions.');
end

%% Transform Tobii UCS millimetres into CoppeliaSim metres
% The current 1:1 physical display/robot scale reduces this conversion to
% millimetres-to-metres. Retaining the ratio in the formula makes a future
% deliberate change in CoppeliaSim zoom visible and adjustable.
coppeliasim_m_per_tobii_mm = ...
    1e-3 / screen_motion_per_robot_motion;

if alignment_mode == "data_centers"
    tobii_reference_xy_mm = mean( ...
        combined_gaze_ucs_mm(combined_gaze_valid, 1:2), 1);
    coppelia_reference_xy_m = mean(coppeliasim_xy_m, 1);
    alignment_description = ...
        'data-center translation (not absolute calibration)';
elseif alignment_mode == "manual_reference"
    if any(~isfinite(manual_tobii_reference_xy_mm)) || ...
            any(~isfinite(manual_coppelia_reference_xy_m))
        error(['Manual alignment requires finite values for ' ...
            'manual_tobii_reference_xy_mm and ' ...
            'manual_coppelia_reference_xy_m.']);
    end
    tobii_reference_xy_mm = manual_tobii_reference_xy_mm;
    coppelia_reference_xy_m = manual_coppelia_reference_xy_m;
    alignment_description = 'manual calibrated reference';
else
    error('alignment_mode must be "data_centers" or "manual_reference".');
end

gaze_displacement_ucs_mm = ...
    combined_gaze_ucs_mm(:, 1:2) - tobii_reference_xy_mm;
gaze_displacement_coppelia_m = ...
    gaze_displacement_ucs_mm .* ...
    tobii_to_coppelia_axis_sign .* ...
    coppeliasim_m_per_tobii_mm;
gaze_coppeliasim_xy_m = ...
    coppelia_reference_xy_m + gaze_displacement_coppelia_m;

%% Plot the gaze data over the robot trajectory
figure('Name', 'Tobii UCS gaze over UR5e trajectory', 'Color', 'w');
hold on;

plot( ...
    coppeliasim_xy_m(:, 1), ...
    coppeliasim_xy_m(:, 2), ...
    'Color', [0.20, 0.20, 0.20], ...
    'LineWidth', 1.5, ...
    'DisplayName', 'UR5e end-effector trajectory');

scatter( ...
    gaze_coppeliasim_xy_m(combined_gaze_valid, 1), ...
    gaze_coppeliasim_xy_m(combined_gaze_valid, 2), ...
    gaze_marker_size, ...
    [0.85, 0.10, 0.10], ...
    'filled', ...
    'DisplayName', 'Combined valid gaze points');

plot( ...
    coppelia_reference_xy_m(1), ...
    coppelia_reference_xy_m(2), ...
    'kx', ...
    'MarkerSize', 10, ...
    'LineWidth', 1.5, ...
    'DisplayName', 'Alignment reference');

grid on;
axis equal;
xlabel('CoppeliaSim x [m]');
ylabel('CoppeliaSim y [m]');
title({ ...
    'Tobii UCS gaze over UR5e end-effector trajectory', ...
    alignment_description});
legend('Location', 'best');

fprintf('Robot samples plotted: %d\n', size(coppeliasim_xy_m, 1));
fprintf('Valid combined gaze samples plotted: %d of %d\n', ...
    sum(combined_gaze_valid), height(gaze_table));
fprintf('Tobii-to-CoppeliaSim scale: %.9f m/mm\n', ...
    coppeliasim_m_per_tobii_mm);
fprintf('Tobii reference [mm]: [%.3f, %.3f]\n', ...
    tobii_reference_xy_mm(1), tobii_reference_xy_mm(2));
fprintf('CoppeliaSim reference [m]: [%.6f, %.6f]\n', ...
    coppelia_reference_xy_m(1), coppelia_reference_xy_m(2));
fprintf('Alignment: %s\n', alignment_description);

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
%GET_NUMERIC_COLUMN Convert a table column into a numeric column vector.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV cells.

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


function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display bounds.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end


function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes.
%
% Inputs:
%   left_point, right_point - N-by-3 Tobii UCS gaze points [mm].
%   left_valid, right_valid - N-by-1 logical validity masks.
%
% Outputs:
%   combined_point - N-by-3 combined Tobii UCS gaze point [mm].
%   combined_valid - N-by-1 true where at least one eye is valid.

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
