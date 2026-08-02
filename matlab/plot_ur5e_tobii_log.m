%% Plot and inspect one UR5e/Tobii gaze recording.
%
% Input:
%   CSV file written by ur5e_fixed_adm_2d.py. Gaze vectors are stored as
%   JSON-like strings and timestamps are in microseconds.
%
% Outputs:
%   Figures for normalized display gaze, gaze validity, pupil diameter,
%   gaze origins/points in Tobii's user coordinate system, and timing.
%   No source data is modified.
%
% User settings:
%   Change data_file below. display_area_margin is a normalized-coordinate
%   margin used only for the display-space plot; recommended range [0, 0.2].

clear;
close all;
clc;

%% User Settings
script_directory = fileparts(mfilename('fullpath'));
data_file = fullfile(script_directory, ...
    '..', ...
    'data', ...
    'raw_data', ...
    'ur5e_fixed_adm_2d_tobii_log_test6.csv');
display_area_margin = 0.05;  % TODO normalized display margin [0, 0.2]
timestamp_variable = 'system_time_stamp';  % TODO system or device timestamp

%% Read and decode the recording
data_table = readtable(data_file, ...
    'Delimiter', ',', ...
    'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', ...
    'TextType', 'string');

required_variables = {
    timestamp_variable
    'left_gaze_point_on_display_area'
    'right_gaze_point_on_display_area'
    'left_gaze_point_in_user_coordinate_system'
    'right_gaze_point_in_user_coordinate_system'
    'left_gaze_origin_in_user_coordinate_system'
    'right_gaze_origin_in_user_coordinate_system'
    'left_gaze_point_validity'
    'right_gaze_point_validity'
    'left_gaze_origin_validity'
    'right_gaze_origin_validity'
    'left_pupil_diameter'
    'right_pupil_diameter'
    'left_pupil_validity'
    'right_pupil_validity'
    };

imported_variables = data_table.Properties.VariableNames;
missing_variables = setdiff(required_variables, imported_variables);
if ~isempty(missing_variables)
    error(['The CSV import is missing these expected columns: %s\n' ...
        'Columns MATLAB imported: %s'], ...
        strjoin(missing_variables, ', '), ...
        strjoin(imported_variables, ', '));
end

system_time_us = get_numeric_column(data_table.(timestamp_variable));
time_s = (system_time_us - system_time_us(1)) * 1e-6;

left_display = parse_vector_column( ...
    data_table.left_gaze_point_on_display_area, 2);
right_display = parse_vector_column(...
    data_table.right_gaze_point_on_display_area, 2);

left_gaze_point_user = parse_vector_column(...
    data_table.left_gaze_point_in_user_coordinate_system, 3);
right_gaze_point_user = parse_vector_column(...
    data_table.right_gaze_point_in_user_coordinate_system, 3);

left_origin_user = parse_vector_column(...
    data_table.left_gaze_origin_in_user_coordinate_system, 3);
right_origin_user = parse_vector_column(...
    data_table.right_gaze_origin_in_user_coordinate_system, 3);

left_gaze_validity = get_numeric_column(...
    data_table.left_gaze_point_validity);
right_gaze_validity = get_numeric_column(...
    data_table.right_gaze_point_validity);
left_origin_validity = get_numeric_column(...
    data_table.left_gaze_origin_validity);
right_origin_validity = get_numeric_column(...
    data_table.right_gaze_origin_validity);
left_pupil_validity = get_numeric_column(data_table.left_pupil_validity);
right_pupil_validity = get_numeric_column(data_table.right_pupil_validity);
left_pupil_diameter_mm = get_numeric_column(...
    data_table.left_pupil_diameter);
right_pupil_diameter_mm = get_numeric_column(...
    data_table.right_pupil_diameter);

left_gaze_valid = left_gaze_validity == 1 & finite_rows(left_display);
right_gaze_valid = right_gaze_validity == 1 & finite_rows(right_display);
left_origin_valid = left_origin_validity == 1 & finite_rows(left_origin_user);
right_origin_valid = right_origin_validity == 1 & ...
    finite_rows(right_origin_user);
left_pupil_valid = left_pupil_validity == 1 & ...
    finite_scalar_rows(left_pupil_diameter_mm);
right_pupil_valid = right_pupil_validity == 1 & ...
    finite_scalar_rows(right_pupil_diameter_mm);

left_display_plot = left_display;
right_display_plot = right_display;
left_display_plot(~left_gaze_valid, :) = NaN;
right_display_plot(~right_gaze_valid, :) = NaN;

left_pupil_plot = left_pupil_diameter_mm;
right_pupil_plot = right_pupil_diameter_mm;
left_pupil_plot(~left_pupil_valid) = NaN;
right_pupil_plot(~right_pupil_valid) = NaN;

%% Recording summary
sample_interval_s = diff(time_s);
positive_intervals_s = sample_interval_s(sample_interval_s > 0);
median_interval_s = median(positive_intervals_s);
estimated_rate_hz = 1 / median_interval_s;

fprintf('File: %s\n', data_file);
fprintf('Samples: %d\n', height(data_table));
fprintf('Duration: %.3f s\n', time_s(end));
fprintf('Median sample interval: %.3f ms\n', median_interval_s * 1e3);
fprintf('Estimated sample rate: %.2f Hz\n', estimated_rate_hz);
fprintf('Left/right gaze validity: %.1f%% / %.1f%%\n', ...
    100 * mean(left_gaze_valid), ...
    100 * mean(right_gaze_valid));
fprintf( 'Left/right pupil validity: %.1f%% / %.1f%%\n', ...
    100 * mean(left_pupil_valid), ...
    100 * mean(right_pupil_valid));

%% Figure 1: normalized gaze on the active display
figure('Name', 'Tobii normalized display gaze', 'Color', 'w');

subplot(2, 2, 1);
hold on;
plot(left_display_plot(:, 1), ...
    left_display_plot(:, 2), ...
    '.', ...
    'DisplayName', 'Left eye');
plot(right_display_plot(:, 1), ...
    right_display_plot(:, 2), ...
    '.', ...
    'DisplayName', 'Right eye');
hRect = rectangle('Position', [0, 0, 1, 1], ...
    'EdgeColor', 'k', ...
    'LineStyle', '--');

% Add a dummy (invisible) line so the rectangle appears in the legend
plot(nan, nan, 'k--', 'DisplayName', 'Active display');

grid on;
axis equal;
set(gca, 'YDir', 'reverse');
xlim([-display_area_margin, 1 + display_area_margin]);
ylim([-display_area_margin, 1 + display_area_margin]);
xlabel('Normalized display x');
ylabel('Normalized display y');
title('Gaze trajectory on active display');
legend('Location', 'best');

subplot(2, 2, 2);
hold on;
plot(time_s, left_display_plot(:, 1), 'DisplayName', 'Left eye');
plot(time_s, right_display_plot(:, 1), 'DisplayName', 'Right eye');
yline(0, 'k--');
yline(1, 'k--');
grid on;
xlabel('Time [s]');
ylabel('Display x [normalized]');
title('Horizontal gaze position');
legend('Location', 'best');

subplot(2, 2, 3);
hold on;
plot(time_s, left_display_plot(:, 2), 'DisplayName', 'Left eye');
plot(time_s, right_display_plot(:, 2), 'DisplayName', 'Right eye');
yline(0, 'k--');
yline(1, 'k--');
grid on;
xlabel('Time [s]');
ylabel('Display y [normalized]');
title('Vertical gaze position');
legend('Location', 'best');

subplot(2, 2, 4);
hold on;
stairs(time_s, left_gaze_validity, 'DisplayName', 'Left gaze');
stairs(time_s, right_gaze_validity, 'DisplayName', 'Right gaze');
stairs(time_s, left_origin_validity, '--', 'DisplayName', 'Left origin');
stairs(time_s, right_origin_validity, '--', 'DisplayName', 'Right origin');
ylim([-0.1, 1.1]);
grid on;
xlabel('Time [s]');
ylabel('Validity [0 invalid, 1 valid]');
title('Gaze and origin validity');
legend('Location', 'best');

%% Figure 2: pupil diameter and eye-origin movement
figure('Name', 'Tobii pupil and gaze origin', 'Color', 'w');

subplot(2, 1, 1);
hold on;
plot(time_s, left_pupil_plot, 'DisplayName', 'Left pupil');
plot(time_s, right_pupil_plot, 'DisplayName', 'Right pupil');
grid on;
xlabel('Time [s]');
ylabel('Pupil diameter [mm]');
title('Estimated pupil diameter; invalid samples removed');
legend('Location', 'best');

subplot(2, 1, 2);
hold on;
plot(time_s, masked_component(left_origin_user(:, 1), left_origin_valid), ...
    'DisplayName', 'Left origin x');
plot(time_s, masked_component(left_origin_user(:, 2), left_origin_valid), ...
    'DisplayName', 'Left origin y');
plot(time_s, masked_component(left_origin_user(:, 3), left_origin_valid), ...
    'DisplayName', 'Left origin z');
plot(time_s, masked_component(right_origin_user(:, 1), right_origin_valid), ...
    '--', 'DisplayName', 'Right origin x');
plot(time_s, masked_component(right_origin_user(:, 2), right_origin_valid), ...
    '--', 'DisplayName', 'Right origin y');
plot(time_s, masked_component(right_origin_user(:, 3), right_origin_valid), ...
    '--', 'DisplayName', 'Right origin z');
grid on;
xlabel('Time [s]');
ylabel('Position [mm]');
title('Gaze-origin movement in Tobii user coordinates');
legend('Location', 'best');

%% Figure 3: 3-D gaze point and gaze origin in Tobii user coordinates
figure('Name', 'Tobii 3-D user coordinates', 'Color', 'w');
hold on;
plot3( ...
    masked_component(left_gaze_point_user(:, 1), left_gaze_valid), ...
    masked_component(left_gaze_point_user(:, 2), left_gaze_valid), ...
    masked_component(left_gaze_point_user(:, 3), left_gaze_valid), ...
    '.', ...
    'DisplayName', 'Left gaze point');
plot3( ...
    masked_component(right_gaze_point_user(:, 1), right_gaze_valid), ...
    masked_component(right_gaze_point_user(:, 2), right_gaze_valid), ...
    masked_component(right_gaze_point_user(:, 3), right_gaze_valid), ...
    '.', ...
    'DisplayName', 'Right gaze point');
plot3( ...
    masked_component(left_origin_user(:, 1), left_origin_valid), ...
    masked_component(left_origin_user(:, 2), left_origin_valid), ...
    masked_component(left_origin_user(:, 3), left_origin_valid), ...
    'LineWidth', 1.0, ...
    'DisplayName', 'Left gaze origin');
plot3( ...
    masked_component(right_origin_user(:, 1), right_origin_valid), ...
    masked_component(right_origin_user(:, 2), right_origin_valid), ...
    masked_component(right_origin_user(:, 3), right_origin_valid), ...
    'LineWidth', 1.0, ...
    'DisplayName', 'Right gaze origin');
grid on;
axis equal;
xlabel('User x [mm]');
ylabel('User y [mm]');
zlabel('User z [mm]');
title('Gaze points and origins in Tobii user coordinates');
legend('Location', 'best');
view(3);

%% Figure 4: sampling interval and timing gaps
figure('Name', 'Tobii timing', 'Color', 'w');

subplot(2, 1, 1);
plot(time_s(2:end), sample_interval_s * 1e3, '.');
grid on;
xlabel('Time [s]');
ylabel('Sample interval [ms]');
title('Inter-sample interval');

subplot(2, 1, 2);
histogram(positive_intervals_s * 1e3, 50);
grid on;
xlabel('Sample interval [ms]');
ylabel('Count');
title('Distribution of positive sample intervals');

%% Local functions
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


function valid_rows = finite_rows(values)
%FINITE_ROWS Return rows with no NaN or Inf values.

    valid_rows = all(~isnan(values) & ~isinf(values), 2);
end


function valid_rows = finite_scalar_rows(values)
%FINITE_SCALAR_ROWS Return scalar values that are finite.

    valid_rows = ~isnan(values) & ~isinf(values);
end


function masked_values = masked_component(values, valid_rows)
%MASKED_COMPONENT Replace invalid rows with NaN for plotting.

    masked_values = values;
    masked_values(~valid_rows) = NaN;
end
