%% Compare per-trial Otsu damping labeling across subjects: TCP path
% (colored by label, fixations in red) and damping over time (raw
% signal gray, moving average colored by label).
%
% Same structure as compare_chen_var_adm_2d_across_subjects.m: grouped
% into one or more figures per (shape, bandwidth) pair, one row per
% (subject, direction) trial. Threshold is computed PER TRIAL (Otsu on
% that trial's own smoothed damping), matching label_chen_var_adm_2d_
% by_damping_threshold.m -- this is a review tool for that labeling,
% not a copy of it.
%
% SCALABLE: add a new subject_ids entry and rerun. Trials with no gaze/
% calibration still get the TCP+damping panels, just no fixation dots.
%
% Inputs: data/raw_data/ur5e_chen_var_adm_2d_<bandwidth>_<shape>_shape_
%   <direction>_s<id>.csv + matching tobii_* and calibration JSON.
% Output: figures only. NO LABELED DATA IS SAVED.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');
calibration_directory = fullfile(script_directory, '..', 'data', 'calibrations');

%% User Settings
shape_tokens = ["pear", "fine_gross", "ellipse"];
shape_labels = ["Pear", "Curved Fine/Gross", "Teeth Ellipse"];
bandwidths = ["wide", "narrow"];
directions = ["clockwise", "counter_clockwise"];
direction_labels = ["CW", "CCW"];
subject_ids = 10:18;  % add new subjects here

max_rows_per_figure = 6;  % trials per figure
subplot_columns = 2;  % grid layout, e.g. 3x2 instead of 6x1

damping_column_name = "bd_applied_Ns_m";
damping_min_n_s_m = 10.0;  % shared damping axis [N s/m]
damping_max_n_s_m = 200.0;

damping_moving_average_window_s = 1.0;  % drives the Otsu label
otsu_bin_count = 50;

fine_color = [0.00, 0.45, 0.74];
gross_color = [0.85, 0.33, 0.10];
raw_signal_color = [0.75, 0.75, 0.75];
fixation_color = 'g';
fixation_marker_alpha = 0.5;

tcp_marker_size = 2;
fixation_marker_size = 14;
line_width = 1.0;

velocity_threshold_deg_s = 20.0;  % I-VT fixation detection
minimum_fixation_duration_s = 0.060;
plot_only_on_display_gaze = true;

screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

%% Discover + label available trials, grouped by shape/bandwidth
empty_trial = struct('subject_id', {}, 'direction_label', {}, ...
    'time_s', {}, 'coppeliasim_x_m', {}, 'coppeliasim_y_m', {}, ...
    'damping_n_s_m', {}, 'moving_avg_n_s_m', {}, 'is_fine', {}, ...
    'threshold_n_s_m', {}, 'fixation_xy_m', {}, 'gaze_available', {});
trial_groups = struct('shape_label', {}, 'bandwidth', {}, 'trials', {});

for shape_idx = 1:numel(shape_tokens)
    for bandwidth_idx = 1:numel(bandwidths)
        trials = empty_trial;
        for subject_id = subject_ids
            for direction_idx = 1:numel(directions)
                trial_token = sprintf('%s_%s_shape_%s_s%d', ...
                    bandwidths(bandwidth_idx), shape_tokens(shape_idx), ...
                    directions(direction_idx), subject_id);
                robot_csv_file = fullfile(raw_data_directory, ...
                    "ur5e_chen_var_adm_2d_" + trial_token + ".csv");
                if ~isfile(robot_csv_file)
                    continue;
                end

                robot_table = readtable(robot_csv_file, 'Delimiter', ',', ...
                    'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
                    'TextType', 'string');
                robot_time_s = robot_table.time_s;
                damping_n_s_m = robot_table.(damping_column_name);

                sample_interval_s = median(diff(robot_time_s));
                moving_average_sample_count = window_to_odd_sample_count( ...
                    damping_moving_average_window_s, sample_interval_s, ...
                    numel(damping_n_s_m));
                moving_avg_n_s_m = movmean(damping_n_s_m, moving_average_sample_count);
                threshold_n_s_m = otsu_threshold(moving_avg_n_s_m, otsu_bin_count);
                is_fine = moving_avg_n_s_m > threshold_n_s_m;

                gaze_csv_file = fullfile(raw_data_directory, ...
                    "tobii_chen_var_adm_2d_" + trial_token + ".csv");
                calibration_json_file = fullfile(calibration_directory, ...
                    "tobii_coppelia_reference_calibration_" + trial_token + ".json");
                fixation_xy_m = zeros(0, 2);
                gaze_available = false;
                if isfile(gaze_csv_file) && isfile(calibration_json_file)
                    gaze_table = readtable(gaze_csv_file, 'Delimiter', ',', ...
                        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
                        'TextType', 'string');
                    calibration = jsondecode(fileread(calibration_json_file));
                    tobii_reference_xy_mm = double(calibration.manual_tobii_reference_xy_mm(:)');
                    coppelia_reference_xy_m = double(calibration.manual_coppelia_reference_xy_m(:)');
                    fixation_xy_m = detect_fixations_coppelia_xy( ...
                        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
                        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
                        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
                        plot_only_on_display_gaze);
                    gaze_available = true;
                end

                trials(end + 1) = struct( ...
                    'subject_id', subject_id, ...
                    'direction_label', direction_labels(direction_idx), ...
                    'time_s', robot_time_s, ...
                    'coppeliasim_x_m', robot_table.coppeliasim_x_m, ...
                    'coppeliasim_y_m', robot_table.coppeliasim_y_m, ...
                    'damping_n_s_m', damping_n_s_m, ...
                    'moving_avg_n_s_m', moving_avg_n_s_m, ...
                    'is_fine', is_fine, ...
                    'threshold_n_s_m', threshold_n_s_m, ...
                    'fixation_xy_m', fixation_xy_m, ...
                    'gaze_available', gaze_available); %#ok<SAGROW>
            end
        end

        if isempty(trials)
            continue;
        end
        trial_groups(end + 1) = struct( ...
            'shape_label', shape_labels(shape_idx), ...
            'bandwidth', bandwidths(bandwidth_idx), 'trials', trials); %#ok<SAGROW>
    end
end

if isempty(trial_groups)
    error('No trials found under %s -- check subject_ids and file paths.', ...
        raw_data_directory);
end

fprintf('Found %d (shape, bandwidth) groups:\n', numel(trial_groups));
for group_idx = 1:numel(trial_groups)
    fprintf('  %s, %s: %d trials\n', trial_groups(group_idx).shape_label, ...
        trial_groups(group_idx).bandwidth, numel(trial_groups(group_idx).trials));
end

%% Plot -- TCP paths and damping profiles go in separate figures, one or
%% more figure pairs per (shape, bandwidth) group, same trial row order
%% in both so row N of the TCP figure matches row N of the damping figure.
for group_idx = 1:numel(trial_groups)
    group = trial_groups(group_idx);
    trial_count = numel(group.trials);
    figure_count = ceil(trial_count / max_rows_per_figure);

    for figure_idx = 1:figure_count
        row_start = (figure_idx - 1) * max_rows_per_figure + 1;
        row_end = min(figure_idx * max_rows_per_figure, trial_count);
        row_count = row_end - row_start + 1;

        % ---- Figure: TCP paths, colored by label, fixations overlaid ----
        figure('Name', sprintf('Otsu labeling -- TCP paths: %s, %s (%d/%d)', ...
            group.shape_label, group.bandwidth, figure_idx, figure_count), ...
            'Color', 'w');
        grid_columns = min(subplot_columns, row_count);
        grid_rows = ceil(row_count / grid_columns);
        tiledlayout(grid_rows, grid_columns, 'TileSpacing', 'compact', 'Padding', 'compact');

        for row_idx = 1:row_count
            trial = group.trials(row_start + row_idx - 1);
            is_first_row = row_idx == 1;

            nexttile;
            hold on;
            scatter(trial.coppeliasim_x_m(trial.is_fine), ...
                trial.coppeliasim_y_m(trial.is_fine), tcp_marker_size, ...
                fine_color, 'filled', 'DisplayName', 'Fine');
            scatter(trial.coppeliasim_x_m(~trial.is_fine), ...
                trial.coppeliasim_y_m(~trial.is_fine), tcp_marker_size, ...
                gross_color, 'filled', 'DisplayName', 'Gross');
            if trial.gaze_available && ~isempty(trial.fixation_xy_m)
                scatter(trial.fixation_xy_m(:, 1), trial.fixation_xy_m(:, 2), ...
                    fixation_marker_size, fixation_color, 'filled', ...
                    'MarkerFaceAlpha', fixation_marker_alpha, 'DisplayName', 'Fixation');
            end
            axis equal;
            grid on;
            xlabel('x [m]');
            ylabel('y [m]');
            title(sprintf('s%d %s', trial.subject_id, trial.direction_label));
            if is_first_row
                legend('Location', 'best', 'FontSize', 7);
            end
        end

        % ---- Figure: damping vs time, raw gray, moving average colored ----
        figure('Name', sprintf('Otsu labeling -- damping: %s, %s (%d/%d)', ...
            group.shape_label, group.bandwidth, figure_idx, figure_count), ...
            'Color', 'w');
        grid_columns = min(subplot_columns, row_count);
        grid_rows = ceil(row_count / grid_columns);
        tiledlayout(grid_rows, grid_columns, 'TileSpacing', 'compact', 'Padding', 'compact');

        for row_idx = 1:row_count
            trial = group.trials(row_start + row_idx - 1);
            is_first_row = row_idx == 1;

            nexttile;
            hold on;
            plot(trial.time_s, trial.damping_n_s_m, '-', ...
                'Color', raw_signal_color, 'LineWidth', line_width, ...
                'DisplayName', damping_column_name);
            fine_avg = trial.moving_avg_n_s_m;
            fine_avg(~trial.is_fine) = NaN;
            gross_avg = trial.moving_avg_n_s_m;
            gross_avg(trial.is_fine) = NaN;
            plot(trial.time_s, fine_avg, '-', 'Color', fine_color, ...
                'LineWidth', 1.5 * line_width, 'DisplayName', 'Fine (moving avg)');
            plot(trial.time_s, gross_avg, '-', 'Color', gross_color, ...
                'LineWidth', 1.5 * line_width, 'DisplayName', 'Gross (moving avg)');
            yline(trial.threshold_n_s_m, 'k--', 'HandleVisibility', 'off');
            grid on;
            ylim([damping_min_n_s_m, damping_max_n_s_m]);
            xlabel('Time [s]');
            ylabel('Damping [N s/m]');
            title(sprintf('s%d %s, threshold = %.1f N s/m', ...
                trial.subject_id, trial.direction_label, trial.threshold_n_s_m));
            if is_first_row
                legend('Location', 'best', 'FontSize', 7);
            end
        end
    end
end


%% ================================================================
% Local functions
% =================================================================

function threshold = otsu_threshold(data, bin_count)
%OTSU_THRESHOLD Data-driven threshold maximizing between-class variance
% (Otsu, 1979). Copied from label_chen_var_adm_2d_by_damping_
% threshold.m (self-contained here per this project's convention).

    finite_data = data(isfinite(data));
    [counts, edges] = histcounts(finite_data, bin_count);
    bin_centers = 0.5 * (edges(1:end-1) + edges(2:end));

    total_count = sum(counts);
    cumulative_count = cumsum(counts);
    cumulative_weighted_sum = cumsum(bin_centers .* counts);

    weight0 = cumulative_count / total_count;
    weight1 = 1 - weight0;
    mean0 = cumulative_weighted_sum ./ max(cumulative_count, 1);
    mean1 = (cumulative_weighted_sum(end) - cumulative_weighted_sum) ./ ...
        max(total_count - cumulative_count, 1);

    between_class_variance = weight0 .* weight1 .* (mean0 - mean1) .^ 2;
    between_class_variance(weight0 == 0 | weight1 == 0) = -Inf;

    [~, best_bin_idx] = max(between_class_variance);
    threshold = bin_centers(best_bin_idx);
end


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample
% count. Copied from label_chen_var_adm_2d_by_damping_threshold.m.

    if window_s <= 0
        sample_count = 1;
        return;
    end

    sample_count = max(3, round(window_s / sample_interval_s));
    if mod(sample_count, 2) == 0
        sample_count = sample_count + 1;
    end

    largest_odd_count = maximum_sample_count;
    if mod(largest_odd_count, 2) == 0
        largest_odd_count = largest_odd_count - 1;
    end
    sample_count = min(sample_count, largest_odd_count);
end


function fixation_xy_m = detect_fixations_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres in CoppeliaSim [m].
% Copied from label_chen_var_adm_2d_by_live_replay.m (self-contained
% here per this project's convention), trimmed to return only
% fixation_xy_m since that is all this script overlays.

    left_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_point_in_user_coordinate_system, 3);
    right_gaze_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_point_in_user_coordinate_system, 3);
    left_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.left_gaze_origin_in_user_coordinate_system, 3);
    right_gaze_origin_ucs_mm = parse_vector_column( ...
        gaze_table.right_gaze_origin_in_user_coordinate_system, 3);
    left_display_gaze = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display_gaze = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1;
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1;
    left_origin_valid = get_numeric_column( ...
        gaze_table.left_gaze_origin_validity) == 1;
    right_origin_valid = get_numeric_column( ...
        gaze_table.right_gaze_origin_validity) == 1;

    gaze_time_s = get_numeric_column(gaze_table.time_s);

    left_point_valid = left_point_valid & finite_rows(left_gaze_ucs_mm);
    right_point_valid = right_point_valid & finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end

    [left_gaze_direction, left_eye_valid] = calculate_eye_direction( ...
        left_gaze_origin_ucs_mm, left_gaze_ucs_mm, ...
        left_origin_valid, left_point_valid);
    [right_gaze_direction, right_eye_valid] = calculate_eye_direction( ...
        right_gaze_origin_ucs_mm, right_gaze_ucs_mm, ...
        right_origin_valid, right_point_valid);

    [combined_gaze_ucs_mm, combined_point_valid] = combine_valid_eye_points( ...
        left_gaze_ucs_mm, right_gaze_ucs_mm, left_eye_valid, right_eye_valid);
    [combined_gaze_direction, combined_direction_valid] = ...
        combine_valid_eye_directions( ...
        left_gaze_direction, right_gaze_direction, ...
        left_eye_valid, right_eye_valid);
    combined_gaze_valid = combined_point_valid & combined_direction_valid;
    combined_gaze_ucs_mm(~combined_gaze_valid, :) = NaN;

    angular_velocity_deg_s = calculate_angular_velocity( ...
        combined_gaze_direction, gaze_time_s, combined_gaze_valid);
    fixation_candidate = combined_gaze_valid & ...
        isfinite(angular_velocity_deg_s) & ...
        angular_velocity_deg_s <= velocity_threshold_deg_s;

    candidate_changes = diff(double([false; fixation_candidate; false]));
    candidate_start_indices = find(candidate_changes == 1);
    candidate_end_indices = find(candidate_changes == -1) - 1;

    fixation_mean_ucs_x_mm = zeros(0, 1);
    fixation_mean_ucs_y_mm = zeros(0, 1);

    for candidate_idx = 1:numel(candidate_start_indices)
        start_idx = candidate_start_indices(candidate_idx);
        end_idx = candidate_end_indices(candidate_idx);
        candidate_duration_s = gaze_time_s(end_idx) - gaze_time_s(start_idx);

        if candidate_duration_s < minimum_fixation_duration_s
            continue;
        end

        fixation_rows = start_idx:end_idx;
        mean_ucs_mm = mean(combined_gaze_ucs_mm(fixation_rows, :), 1);
        fixation_mean_ucs_x_mm(end + 1, 1) = mean_ucs_mm(1); %#ok<AGROW>
        fixation_mean_ucs_y_mm(end + 1, 1) = mean_ucs_mm(2); %#ok<AGROW>
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    fixation_ucs_xy_mm = [fixation_mean_ucs_x_mm, fixation_mean_ucs_y_mm];
    fixation_displacement_ucs_mm = fixation_ucs_xy_mm - tobii_reference_xy_mm;
    fixation_displacement_coppelia_m = fixation_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    fixation_xy_m = coppelia_reference_xy_m + fixation_displacement_coppelia_m;
end


function [combined_point, combined_valid] = combine_valid_eye_points( ...
        left_point, right_point, left_valid, right_valid)
%COMBINE_VALID_EYE_POINTS Average UCS points from all valid eyes.
% Copied from plot_raw_data_chen_var_adm_2d.m.

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


function [unit_direction, eye_valid] = calculate_eye_direction( ...
        gaze_origin_ucs_mm, gaze_point_ucs_mm, ...
        origin_validity, point_validity)
%CALCULATE_EYE_DIRECTION Build one eye's unit direction in Tobii UCS.
% Copied from plot_raw_data_chen_var_adm_2d.m.

    direction = gaze_point_ucs_mm - gaze_origin_ucs_mm;
    direction_norm = sqrt(sum(direction .^ 2, 2));
    eye_valid = origin_validity & point_validity & ...
        finite_rows(gaze_origin_ucs_mm) & finite_rows(gaze_point_ucs_mm) & ...
        isfinite(direction_norm) & direction_norm > 0;

    unit_direction = NaN(size(direction));
    unit_direction(eye_valid, :) = direction(eye_valid, :) ./ ...
        direction_norm(eye_valid);
end


function [combined_direction, combined_valid] = ...
        combine_valid_eye_directions( ...
        left_direction, right_direction, left_valid, right_valid)
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye
% directions. Copied from plot_raw_data_chen_var_adm_2d.m.

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


function angular_velocity_deg_s = calculate_angular_velocity( ...
        gaze_direction, time_s, gaze_valid)
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s]. Copied
% from plot_raw_data_chen_var_adm_2d.m.

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
        if end_idx <= start_idx
            continue;
        end
        if isfinite(angular_velocity_deg_s(start_idx + 1))
            angular_velocity_deg_s(start_idx) = ...
                angular_velocity_deg_s(start_idx + 1);
        end
    end
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
