%% Check Chen trials' Tobii calibration for coverage gaps and swap-type
% errors, before trusting gaze features for labeling.
%
% Two checks:
%   1. COVERAGE: every ur5e_chen_var_adm_2d_*.csv found is checked for
%      a matching gaze CSV and calibration JSON. Missing ones are
%      printed, not silently skipped.
%   2. CALIBRATION: for every trial with all three files, gaze is
%      mapped to CoppeliaSim and compared to the TCP path. Outliers are
%      flagged relative to their own (bandwidth, shape) group (not a
%      fixed threshold, since "normal" bias differs per shape), then
%      classified: a bias that is the SAME sign/magnitude in both
%      clockwise and counter-clockwise is a fixed calibration error
%      (this found and fixed a real subject-pair swap); a bias that
%      FLIPS sign between directions is real look-ahead gaze behavior,
%      since "ahead of travel" rotates with heading around a closed
%      loop. Flagged pairs are also tested for a reference-point swap.
%
% Inputs: data/raw_data/ur5e_chen_var_adm_2d_*.csv + matching tobii_*
%   and data/calibrations/tobii_coppelia_reference_calibration_*.json.
% Output: printed report + one scatter figure (dx per trial by group).
% Nothing written to disk.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');
calibration_directory = fullfile(script_directory, '..', 'data', 'calibrations');

%% User Settings
require_both_eyes_valid = true;
require_gaze_on_display = true;
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

moving_speed_threshold_m_s = 0.005;  % for the along/perp decomposition

outlier_abs_threshold_mm = 10;  % flag if |dx-median| or |dy-median| exceeds this
outlier_mad_multiple = 5;       % or exceeds this many group MADs

%% Report I/O before doing anything
fprintf('Input:  %s\n', fullfile(raw_data_directory, 'ur5e_chen_var_adm_2d_*.csv'));
fprintf('Input:  %s\n', fullfile(raw_data_directory, 'tobii_chen_var_adm_2d_*.csv'));
fprintf('Input:  %s\n', ...
    fullfile(calibration_directory, 'tobii_coppelia_reference_calibration_*.json'));
fprintf('Output: none written to disk (printed report + one figure only)\n\n');

%% Coverage check -- every robot CSV found, gaze + calibration required
robot_csv_listing = dir(fullfile(raw_data_directory, 'ur5e_chen_var_adm_2d_*.csv'));
missing_gaze = {};
missing_calibration = {};
trial_names = strings(0, 1);
robot_files = strings(0, 1);
gaze_files = strings(0, 1);
calibration_files = strings(0, 1);

for file_idx = 1:numel(robot_csv_listing)
    robot_file_name = robot_csv_listing(file_idx).name;
    trial_name = extractAfter(robot_file_name, 'ur5e_chen_var_adm_2d_');
    trial_name = extractBefore(trial_name, '.csv');

    robot_csv_file = fullfile(raw_data_directory, robot_file_name);
    gaze_csv_file = fullfile(raw_data_directory, ...
        "tobii_chen_var_adm_2d_" + trial_name + ".csv");
    calibration_json_file = fullfile(calibration_directory, ...
        "tobii_coppelia_reference_calibration_" + trial_name + ".json");

    gaze_ok = isfile(gaze_csv_file);
    calibration_ok = isfile(calibration_json_file);
    if ~gaze_ok
        missing_gaze{end + 1} = robot_file_name; %#ok<SAGROW>
    end
    if ~calibration_ok
        missing_calibration{end + 1} = robot_file_name; %#ok<SAGROW>
    end
    if ~gaze_ok || ~calibration_ok
        continue;
    end

    trial_names(end + 1, 1) = trial_name; %#ok<AGROW>
    robot_files(end + 1, 1) = robot_csv_file; %#ok<AGROW>
    gaze_files(end + 1, 1) = gaze_csv_file; %#ok<AGROW>
    calibration_files(end + 1, 1) = calibration_json_file; %#ok<AGROW>
end

fprintf('Found %d robot CSVs.\n', numel(robot_csv_listing));
fprintf('Missing gaze CSV: %d\n', numel(missing_gaze));
for i = 1:numel(missing_gaze)
    fprintf('  %s\n', missing_gaze{i});
end
fprintf('Missing calibration JSON: %d\n', numel(missing_calibration));
for i = 1:numel(missing_calibration)
    fprintf('  %s\n', missing_calibration{i});
end
fprintf('%d trials have all three files; checking calibration.\n\n', numel(trial_names));

%% Parse (bandwidth, shape, direction, subject) where the name fits the
% pattern -- trials that don't (older single-subject recordings) are
% still checked above for coverage, just not grouped below.
pattern = '^(wide|narrow)_(.+)_shape_(clockwise|counter_clockwise)_s(\d+)$';
bandwidth = strings(numel(trial_names), 1);
shape_token = strings(numel(trial_names), 1);
direction = strings(numel(trial_names), 1);
subject_id = nan(numel(trial_names), 1);
parsed = false(numel(trial_names), 1);

for trial_idx = 1:numel(trial_names)
    tokens = regexp(trial_names(trial_idx), pattern, 'tokens', 'once');
    if isempty(tokens)
        continue;
    end
    bandwidth(trial_idx) = tokens{1};
    shape_token(trial_idx) = tokens{2};
    direction(trial_idx) = tokens{3};
    subject_id(trial_idx) = str2double(tokens{4});
    parsed(trial_idx) = true;
end

%% Compute gaze-TCP bias for every parsed trial
dx_mm = nan(numel(trial_names), 1);
dy_mm = nan(numel(trial_names), 1);
along_mm = nan(numel(trial_names), 1);
perp_mm = nan(numel(trial_names), 1);

for trial_idx = 1:numel(trial_names)
    if ~parsed(trial_idx)
        continue;
    end
    bias = compute_gaze_tcp_bias(robot_files(trial_idx), gaze_files(trial_idx), ...
        calibration_files(trial_idx), [], require_both_eyes_valid, ...
        require_gaze_on_display, screen_motion_per_robot_motion, ...
        tobii_to_coppelia_axis_sign, moving_speed_threshold_m_s);
    if isempty(bias)
        continue;
    end
    dx_mm(trial_idx) = bias.dx_mm;
    dy_mm(trial_idx) = bias.dy_mm;
    along_mm(trial_idx) = bias.along_mm;
    perp_mm(trial_idx) = bias.perp_mm;
end

%% Flag outliers relative to each (bandwidth, shape) group
group_key = bandwidth + "_" + shape_token;
[unique_groups, ~, group_idx_of_trial] = unique(group_key(parsed));
parsed_indices = find(parsed);

is_flagged = false(numel(trial_names), 1);
for group_idx = 1:numel(unique_groups)
    member_indices = parsed_indices(group_idx_of_trial == group_idx);
    group_dx = dx_mm(member_indices);
    group_dy = dy_mm(member_indices);
    median_dx = median(group_dx);
    median_dy = median(group_dy);
    mad_dx = max(median(abs(group_dx - median_dx)), 1);
    mad_dy = max(median(abs(group_dy - median_dy)), 1);

    for k = 1:numel(member_indices)
        trial_idx = member_indices(k);
        z_score = max(abs(dx_mm(trial_idx) - median_dx) / mad_dx, ...
            abs(dy_mm(trial_idx) - median_dy) / mad_dy);
        if abs(dx_mm(trial_idx) - median_dx) > outlier_abs_threshold_mm || ...
                abs(dy_mm(trial_idx) - median_dy) > outlier_abs_threshold_mm || ...
                z_score > outlier_mad_multiple
            is_flagged(trial_idx) = true;
        end
    end
end

flagged_indices = find(is_flagged);
fprintf('Flagged %d outlier trials (>%.0f mm from group median, or >%dx group MAD):\n', ...
    numel(flagged_indices), outlier_abs_threshold_mm, outlier_mad_multiple);

for k = 1:numel(flagged_indices)
    trial_idx = flagged_indices(k);
    fprintf('\n%s\n', trial_names(trial_idx));
    fprintf('  world dx=%+6.1f mm  dy=%+6.1f mm  |  along-motion=%+6.1f  perp=%+6.1f\n', ...
        dx_mm(trial_idx), dy_mm(trial_idx), along_mm(trial_idx), perp_mm(trial_idx));

    % Direction-flip check: same subject/shape/bandwidth, other
    % direction, SAME calibration file (verified below). Same-sign
    % world dx in both directions = fixed calibration error; opposite
    % sign = real gaze lead rotating with heading.
    other_direction = "clockwise";
    if direction(trial_idx) == "clockwise"
        other_direction = "counter_clockwise";
    end
    other_mask = parsed & bandwidth == bandwidth(trial_idx) & ...
        shape_token == shape_token(trial_idx) & ...
        subject_id == subject_id(trial_idx) & direction == other_direction;
    other_idx = find(other_mask, 1);

    if isempty(other_idx)
        fprintf('  no counterpart in the other direction to cross-check.\n');
    else
        same_calibration = isequal( ...
            fileread(calibration_files(trial_idx)), fileread(calibration_files(other_idx)));
        fprintf('  %s counterpart: dx=%+6.1f mm  (same calibration file: %s)\n', ...
            other_direction, dx_mm(other_idx), string(same_calibration));
        if same_calibration
            if sign(dx_mm(trial_idx)) == sign(dx_mm(other_idx)) && ...
                    abs(dx_mm(trial_idx) - dx_mm(other_idx)) < outlier_abs_threshold_mm
                fprintf('  ==> SUSPECT: direction-independent offset, check this calibration.\n');
            else
                fprintf('  ==> likely real gaze-lead behavior (sign flips with direction).\n');
            end
        end
    end
end

%% Pairwise swap test among flagged trials in the same group
fprintf('\nSwap test among flagged trials (same bandwidth+shape group):\n');
swap_found = false;
for a = 1:numel(flagged_indices)
    for b = (a + 1):numel(flagged_indices)
        idx_a = flagged_indices(a);
        idx_b = flagged_indices(b);
        if group_key(idx_a) ~= group_key(idx_b)
            continue;
        end
        bias_a_swapped = compute_gaze_tcp_bias(robot_files(idx_a), gaze_files(idx_a), ...
            calibration_files(idx_a), calibration_files(idx_b), require_both_eyes_valid, ...
            require_gaze_on_display, screen_motion_per_robot_motion, ...
            tobii_to_coppelia_axis_sign, moving_speed_threshold_m_s);
        bias_b_swapped = compute_gaze_tcp_bias(robot_files(idx_b), gaze_files(idx_b), ...
            calibration_files(idx_b), calibration_files(idx_a), require_both_eyes_valid, ...
            require_gaze_on_display, screen_motion_per_robot_motion, ...
            tobii_to_coppelia_axis_sign, moving_speed_threshold_m_s);
        if isempty(bias_a_swapped) || isempty(bias_b_swapped)
            continue;
        end
        if max(abs(bias_a_swapped.dx_mm), abs(bias_a_swapped.dy_mm)) < outlier_abs_threshold_mm && ...
                max(abs(bias_b_swapped.dx_mm), abs(bias_b_swapped.dy_mm)) < outlier_abs_threshold_mm
            fprintf('  LIKELY SWAP: %s <-> %s (both drop under %.0f mm after swapping)\n', ...
                trial_names(idx_a), trial_names(idx_b), outlier_abs_threshold_mm);
            swap_found = true;
        end
    end
end
if ~swap_found
    fprintf('  none found.\n');
end

%% Figure: dx per trial, grouped, outliers marked
figure('Name', 'Gaze-TCP calibration check', 'Color', 'w');
hold on;
colors = lines(numel(unique_groups));
for group_idx = 1:numel(unique_groups)
    member_indices = parsed_indices(group_idx_of_trial == group_idx);
    scatter(group_idx * ones(numel(member_indices), 1) + 0.15 * (rand(numel(member_indices), 1) - 0.5), ...
        dx_mm(member_indices), 24, colors(group_idx, :), 'filled', 'DisplayName', unique_groups(group_idx));
end
if any(is_flagged)
    for k = 1:numel(flagged_indices)
        trial_idx = flagged_indices(k);
        g = find(unique_groups == group_key(trial_idx));
        plot(g, dx_mm(trial_idx), 'ko', 'MarkerSize', 12, 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
end
yline(0, 'k--', 'HandleVisibility', 'off');
xticks(1:numel(unique_groups));
xticklabels(unique_groups);
xtickangle(30);
grid on;
ylabel('Gaze - TCP mean dx [mm]');
title('Gaze-TCP x offset by group (circled = flagged outlier)');
legend('Location', 'bestoutside');


%% ================================================================
% Local functions
% =================================================================

function bias = compute_gaze_tcp_bias(robot_csv_file, gaze_csv_file, ...
        calibration_json_file, override_calibration_json_file, ...
        require_both_eyes_valid, require_gaze_on_display, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        moving_speed_threshold_m_s)
%COMPUTE_GAZE_TCP_BIAS World-frame and motion-relative gaze-TCP offset
% [mm]. override_calibration_json_file, if non-empty, uses THAT file's
% reference point instead of calibration_json_file's own (for swap
% testing). Returns [] if too few valid gaze samples.

    robot_table = readtable(robot_csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');
    gaze_table = readtable(gaze_csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');

    if isempty(override_calibration_json_file)
        calibration = jsondecode(fileread(calibration_json_file));
    else
        calibration = jsondecode(fileread(override_calibration_json_file));
    end
    tobii_reference_xy_mm = double(calibration.manual_tobii_reference_xy_mm(:)');
    coppelia_reference_xy_m = double(calibration.manual_coppelia_reference_xy_m(:)');

    [gaze_xy_m, gaze_valid] = map_gaze_ucs_to_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        require_gaze_on_display, require_both_eyes_valid);

    if nnz(gaze_valid) < 200
        bias = [];
        return;
    end

    gaze_time_s = get_numeric_column(gaze_table.time_s);
    gaze_time_s = gaze_time_s(gaze_valid);
    gaze_xy_m = gaze_xy_m(gaze_valid, :);

    robot_time_s = robot_table.time_s;
    tcp_x_at_gaze_m = interp1(robot_time_s, robot_table.coppeliasim_x_m, ...
        gaze_time_s, 'linear', 'extrap');
    tcp_y_at_gaze_m = interp1(robot_time_s, robot_table.coppeliasim_y_m, ...
        gaze_time_s, 'linear', 'extrap');
    vx_at_gaze_m_s = interp1(robot_time_s, robot_table.tcp_vx_actual_m_s, ...
        gaze_time_s, 'linear', 'extrap');
    vy_at_gaze_m_s = interp1(robot_time_s, robot_table.tcp_vy_actual_m_s, ...
        gaze_time_s, 'linear', 'extrap');

    dx_m = gaze_xy_m(:, 1) - tcp_x_at_gaze_m;
    dy_m = gaze_xy_m(:, 2) - tcp_y_at_gaze_m;

    speed_m_s = hypot(vx_at_gaze_m_s, vy_at_gaze_m_s);
    moving = speed_m_s > moving_speed_threshold_m_s;
    unit_x = vx_at_gaze_m_s(moving) ./ speed_m_s(moving);
    unit_y = vy_at_gaze_m_s(moving) ./ speed_m_s(moving);
    along_m = dx_m(moving) .* unit_x + dy_m(moving) .* unit_y;
    perp_m = -dx_m(moving) .* unit_y + dy_m(moving) .* unit_x;

    bias = struct( ...
        'dx_mm', mean(dx_m) * 1000, 'dy_mm', mean(dy_m) * 1000, ...
        'along_mm', mean(along_m) * 1000, 'perp_mm', mean(perp_m) * 1000, ...
        'n', nnz(gaze_valid));
end


function [gaze_xy_m, gaze_valid] = map_gaze_ucs_to_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        require_gaze_on_display, require_both_eyes_valid)
%MAP_GAZE_UCS_TO_COPPELIA_XY Raw gaze position in the CoppeliaSim frame
% [m]. Copied from label_chen_var_adm_2d_by_live_replay.m (self-
% contained here per this project's convention).

    left_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);
    left_display = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1 & finite_rows(left_ucs_mm);
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1 & finite_rows(right_ucs_mm);

    if require_gaze_on_display
        left_point_valid = left_point_valid & point_is_on_display(left_display);
        right_point_valid = right_point_valid & point_is_on_display(right_display);
    end

    sample_count = height(gaze_table);
    if require_both_eyes_valid
        gaze_valid = left_point_valid & right_point_valid;
        gaze_ucs_xy_mm = 0.5 * (left_ucs_mm(:, 1:2) + right_ucs_mm(:, 1:2));
    else
        eye_count = double(left_point_valid) + double(right_point_valid);
        gaze_valid = eye_count > 0;
        gaze_sum_mm = zeros(sample_count, 2);
        gaze_sum_mm(left_point_valid, :) = ...
            gaze_sum_mm(left_point_valid, :) + left_ucs_mm(left_point_valid, 1:2);
        gaze_sum_mm(right_point_valid, :) = ...
            gaze_sum_mm(right_point_valid, :) + right_ucs_mm(right_point_valid, 1:2);
        gaze_ucs_xy_mm = NaN(sample_count, 2);
        gaze_ucs_xy_mm(gaze_valid, :) = ...
            gaze_sum_mm(gaze_valid, :) ./ eye_count(gaze_valid);
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    gaze_displacement_ucs_mm = gaze_ucs_xy_mm - tobii_reference_xy_mm;
    gaze_displacement_coppelia_m = gaze_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    gaze_xy_m = coppelia_reference_xy_m + gaze_displacement_coppelia_m;
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied from plot_raw_data_chen_var_adm_2d.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV
% cells. Copied from plot_raw_data_chen_var_adm_2d.m.

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
        vector_values(sample_idx, 1:value_count) = parsed_values(1:value_count)';
    end
end


function valid = finite_rows(values)
%FINITE_ROWS Return true for rows containing only finite numeric values.
% Copied from plot_raw_data_chen_var_adm_2d.m.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display
% bounds. Copied from plot_raw_data_chen_var_adm_2d.m.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end
