function label_chen_var_adm_2d_by_live_replay()
%LABEL_CHEN_VAR_ADM_2D_BY_LIVE_REPLAY Live keyboard-driven Fine/Gross
% labeling, replaying every discovered Chen trial in turn (same
% discovery loop as compare_plot_save_chen_otsu_labeling_across_
% subjects.m). Hold fine_key while a stretch of motion looks Fine,
% gross_key while it looks Gross; release both for "None" (unsure /
% transition, a genuine third class, not forced-binary). Judged from
% watching the TCP path animate -- the damping trace is shown in plain
% gray for context only, NEVER colored by your label, so this label
% does not just re-encode the Otsu threshold's own force-derived
% boundary (see ../README_gaze_fine_gross_pipeline.md on that
% circularity).
%
% RESUMABLE: a trial whose output CSV already exists is skipped, so the
% total labeling effort can be split across multiple sessions -- rerun
% this same script and it picks up wherever it left off.
%
% Per trial: waitforbuttonpress gates the start (click the figure or
% press any key) so playback never starts before you're ready, and
% figure focus is guaranteed. Closing the figure window early ends THAT
% trial's replay (saving whatever was labeled so far, rest stays
% "None") and moves on to the next trial -- use this to bail on one
% confusing/corrupted trial without losing the rest of the session.
% Ctrl+C saves the CURRENT trial the same way but ends the whole
% session (MATLAB aborts immediately on Ctrl+C and skips ordinary
% cleanup code, so onCleanup + a NESTED save function, registered fresh
% per trial, is the only reliable way to keep partial progress either
% way).
%
% Not tuned empirically (no way to run/watch MATLAB from here) --
% animation_stride and playback_speed_multiplier are starting values;
% raise animation_stride if playback stutters on your machine.
%
% Inputs: data/raw_data/{ur5e,tobii}_chen_var_adm_2d_<trial>.csv +
%   data/calibrations/<trial>.json (gaze overlay only, skipped with a
%   warning if unavailable -- never blocks labeling).
% Output: data/processed_data/{ur5e,tobii}_chen_var_adm_2d_live_labeled_
%   <trial>.csv, written one trial at a time as each finishes.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');
calibration_directory = fullfile(script_directory, '..', 'data', 'calibrations');
processed_data_directory = fullfile(script_directory, '..', 'data', 'processed_data');

%% User Settings
shape_tokens = ["pear", "fine_gross", "ellipse"];
bandwidths = ["wide", "narrow"];
directions = ["clockwise", "counter_clockwise"];
subject_ids = 10:18;  % add new subjects here

fine_key = 'f';
gross_key = 'g';
fine_label_text = "Fine";
gross_label_text = "Gross";
none_label_text = "None";  % neither key held: unsure / transition

fine_color = [0.85, 0.10, 0.10];
gross_color = [0.10, 0.65, 0.20];
none_color = [0.55, 0.55, 0.55];
raw_signal_color = [0.75, 0.75, 0.75];
fixation_color = [1.0, 0.55, 0.55];
fixation_marker_alpha = 0.5;  % lighter so it doesn't dominate the TCP path

tcp_marker_size = 8;
fixation_marker_size = 14;
line_width = 1.0;

playback_speed_multiplier = 2.0;  % >1 = faster than real time
animation_stride = 5;  % samples advanced per redraw -- raise if it stutters

damping_moving_average_window_s = 1.0;  % display only, not used for labels
damping_min_n_s_m = 10.0;
damping_max_n_s_m = 200.0;

velocity_threshold_deg_s = 20.0;  % I-VT fixation detection
minimum_fixation_duration_s = 0.060;
plot_only_on_display_gaze = true;
screen_motion_per_robot_motion = 1.0;
tobii_to_coppelia_axis_sign = [1.0, 1.0];

if ~isfolder(processed_data_directory)
    mkdir(processed_data_directory);
end

%% Loop over every discovered trial, skipping ones already labeled
labeled_count = 0;
skipped_existing_count = 0;
for shape_idx = 1:numel(shape_tokens)
    for bandwidth_idx = 1:numel(bandwidths)
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

                write_robot_csv_file = fullfile(processed_data_directory, ...
                    "ur5e_chen_var_adm_2d_live_labeled_" + trial_token + ".csv");
                if isfile(write_robot_csv_file)
                    skipped_existing_count = skipped_existing_count + 1;
                    continue;
                end

                gaze_csv_file = fullfile(raw_data_directory, ...
                    "tobii_chen_var_adm_2d_" + trial_token + ".csv");
                calibration_json_file = fullfile(calibration_directory, ...
                    "tobii_coppelia_reference_calibration_" + trial_token + ".json");

                label_one_trial(trial_token, robot_csv_file, gaze_csv_file, ...
                    calibration_json_file, processed_data_directory, ...
                    fine_key, gross_key, fine_label_text, gross_label_text, ...
                    none_label_text, fine_color, gross_color, none_color, ...
                    raw_signal_color, fixation_color, fixation_marker_alpha, ...
                    tcp_marker_size, fixation_marker_size, line_width, ...
                    playback_speed_multiplier, animation_stride, ...
                    damping_moving_average_window_s, damping_min_n_s_m, ...
                    damping_max_n_s_m, velocity_threshold_deg_s, ...
                    minimum_fixation_duration_s, plot_only_on_display_gaze, ...
                    screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign);
                labeled_count = labeled_count + 1;
            end
        end
    end
end

fprintf('\nSession done: %d trial(s) labeled, %d already-labeled trial(s) skipped.\n', ...
    labeled_count, skipped_existing_count);
end


%% ================================================================
% Local functions
% =================================================================

function label_one_trial(trial_token, robot_csv_file, gaze_csv_file, ...
        calibration_json_file, processed_data_directory, ...
        fine_key, gross_key, fine_label_text, gross_label_text, ...
        none_label_text, fine_color, gross_color, none_color, ...
        raw_signal_color, fixation_color, fixation_marker_alpha, ...
        tcp_marker_size, fixation_marker_size, line_width, ...
        playback_speed_multiplier, animation_stride, ...
        damping_moving_average_window_s, damping_min_n_s_m, ...
        damping_max_n_s_m, velocity_threshold_deg_s, ...
        minimum_fixation_duration_s, plot_only_on_display_gaze, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign)
%LABEL_ONE_TRIAL Replay + label a single trial. Registers its OWN
% onCleanup, so it saves on normal completion, on an early figure
% close, AND on Ctrl+C -- every exit path from this function triggers
% it (see the module docstring).

    fprintf('\n--- %s ---\n', trial_token);

    robot_table = readtable(robot_csv_file, 'Delimiter', ',', ...
        'ReadVariableNames', true, 'VariableNamingRule', 'preserve', ...
        'TextType', 'string');
    robot_time_s = robot_table.time_s;
    damping_n_s_m = robot_table.bd_applied_Ns_m;
    coppeliasim_x_m = robot_table.coppeliasim_x_m;
    coppeliasim_y_m = robot_table.coppeliasim_y_m;
    sample_count = height(robot_table);

    sample_interval_s = median(diff(robot_time_s));
    moving_average_sample_count = window_to_odd_sample_count( ...
        damping_moving_average_window_s, sample_interval_s, sample_count);
    moving_avg_n_s_m = movmean(damping_n_s_m, moving_average_sample_count);

    fixation_xy_m = zeros(0, 2);
    gaze_available = false;
    gaze_table = table();
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
    else
        warning('%s: no gaze/calibration -- labeling without gaze overlay.', ...
            trial_token);
    end

    robot_label = repmat(none_label_text, sample_count, 1);

    %% Figure: TCP path (left) + plain-gray damping context (right)
    replay_figure = figure('Name', ['Label: ' char(trial_token)], 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    path_axes = nexttile;
    hold(path_axes, 'on');
    path_handle_fine = plot(path_axes, nan, nan, '.', 'Color', fine_color, ...
        'MarkerSize', tcp_marker_size, 'DisplayName', char(fine_label_text));
    path_handle_gross = plot(path_axes, nan, nan, '.', 'Color', gross_color, ...
        'MarkerSize', tcp_marker_size, 'DisplayName', char(gross_label_text));
    path_handle_none = plot(path_axes, nan, nan, '.', 'Color', none_color, ...
        'MarkerSize', tcp_marker_size, 'DisplayName', char(none_label_text));
    current_marker_handle = plot(path_axes, nan, nan, 'o', ...
        'MarkerFaceColor', none_color, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', tcp_marker_size + 4, 'HandleVisibility', 'off');
    if gaze_available && ~isempty(fixation_xy_m)
        scatter(path_axes, fixation_xy_m(:, 1), fixation_xy_m(:, 2), ...
            fixation_marker_size, fixation_color, 'filled', ...
            'MarkerFaceAlpha', fixation_marker_alpha, 'DisplayName', 'Fixation');
    end
    axis(path_axes, 'equal');
    grid(path_axes, 'on');
    xlabel(path_axes, 'x [m]');
    ylabel(path_axes, 'y [m]');
    xlim(path_axes, [min(coppeliasim_x_m), max(coppeliasim_x_m)] + [-0.02, 0.02]);
    ylim(path_axes, [min(coppeliasim_y_m), max(coppeliasim_y_m)] + [-0.02, 0.02]);
    legend(path_axes, 'Location', 'best', 'FontSize', 7);
    title(path_axes, sprintf(['Click this window, then press any key to start -- ' ...
        'hold ''%s''=%s, ''%s''=%s'], fine_key, fine_label_text, gross_key, ...
        gross_label_text));

    damping_axes = nexttile;
    hold(damping_axes, 'on');
    plot(damping_axes, robot_time_s, damping_n_s_m, '-', ...
        'Color', raw_signal_color, 'LineWidth', line_width, ...
        'DisplayName', 'bd\_applied\_Ns\_m');
    plot(damping_axes, robot_time_s, moving_avg_n_s_m, '-', ...
        'Color', [0.3, 0.3, 0.3], 'LineWidth', line_width, ...
        'DisplayName', 'moving avg (context only)');
    damping_playhead_handle = xline(damping_axes, robot_time_s(1), 'k-', ...
        'LineWidth', 1.5, 'HandleVisibility', 'off');
    grid(damping_axes, 'on');
    ylim(damping_axes, [damping_min_n_s_m, damping_max_n_s_m]);
    xlim(damping_axes, [robot_time_s(1), robot_time_s(end)]);
    xlabel(damping_axes, 'Time [s]');
    ylabel(damping_axes, 'Damping [N s/m]');
    title(damping_axes, 'Context only -- NOT colored by your label');
    legend(damping_axes, 'Location', 'best', 'FontSize', 7);

    tracked_keys = struct(fine_key, false, gross_key, false);
    setappdata(replay_figure, 'tracked_keys', tracked_keys);
    set(replay_figure, 'KeyPressFcn', ...
        @(source, event) on_key_change(source, event, true));
    set(replay_figure, 'KeyReleaseFcn', ...
        @(source, event) on_key_change(source, event, false));

    cleanup_saver = onCleanup(@save_labeled_csvs_now); %#ok<NASGU>

    figure(replay_figure);
    waitforbuttonpress;

    replay_start_tic = tic;
    trial_start_s = robot_time_s(1);
    for sample_idx = 1:animation_stride:sample_count
        if ~isvalid(replay_figure)
            break;
        end

        keys = getappdata(replay_figure, 'tracked_keys');
        if keys.(fine_key)
            current_label = fine_label_text;
            current_color = fine_color;
        elseif keys.(gross_key)
            current_label = gross_label_text;
            current_color = gross_color;
        else
            current_label = none_label_text;
            current_color = none_color;
        end

        chunk_end_idx = min(sample_idx + animation_stride - 1, sample_count);
        robot_label(sample_idx:chunk_end_idx) = current_label;

        shown_x = coppeliasim_x_m(1:chunk_end_idx);
        shown_y = coppeliasim_y_m(1:chunk_end_idx);
        shown_label = robot_label(1:chunk_end_idx);

        fine_x = shown_x; fine_y = shown_y;
        gross_x = shown_x; gross_y = shown_y;
        none_x = shown_x; none_y = shown_y;
        is_fine_so_far = shown_label == fine_label_text;
        is_gross_so_far = shown_label == gross_label_text;
        is_none_so_far = shown_label == none_label_text;
        fine_x(~is_fine_so_far) = NaN; fine_y(~is_fine_so_far) = NaN;
        gross_x(~is_gross_so_far) = NaN; gross_y(~is_gross_so_far) = NaN;
        none_x(~is_none_so_far) = NaN; none_y(~is_none_so_far) = NaN;

        set(path_handle_fine, 'XData', fine_x, 'YData', fine_y);
        set(path_handle_gross, 'XData', gross_x, 'YData', gross_y);
        set(path_handle_none, 'XData', none_x, 'YData', none_y);
        set(current_marker_handle, 'XData', coppeliasim_x_m(chunk_end_idx), ...
            'YData', coppeliasim_y_m(chunk_end_idx), ...
            'MarkerFaceColor', current_color);
        set(damping_playhead_handle, 'Value', robot_time_s(chunk_end_idx));

        drawnow;

        trial_time_elapsed_s = robot_time_s(chunk_end_idx) - trial_start_s;
        target_wall_time_s = trial_time_elapsed_s / playback_speed_multiplier;
        remaining_pause_s = target_wall_time_s - toc(replay_start_tic);
        if remaining_pause_s > 0
            pause(remaining_pause_s);
        end
    end

    if isvalid(replay_figure)
        title(path_axes, 'Replay finished -- saved. Close this window to continue.');
    end

    function save_labeled_csvs_now()
        % NESTED -- shares label_one_trial's workspace, so it sees the
        % live robot_label array as it stood at whatever point this
        % fires (normal completion, closed figure, or Ctrl+C).
        robot_table.label = robot_label;
        write_robot_csv_file = fullfile(processed_data_directory, ...
            "ur5e_chen_var_adm_2d_live_labeled_" + trial_token + ".csv");
        writetable(robot_table, write_robot_csv_file);

        if gaze_available
            gaze_table.label = resample_labels_nearest( ...
                robot_time_s, robot_label, gaze_table.time_s);
            write_gaze_csv_file = fullfile(processed_data_directory, ...
                "tobii_chen_var_adm_2d_live_labeled_" + trial_token + ".csv");
            writetable(gaze_table, write_gaze_csv_file);
        end

        label_categorical = categorical(robot_label);
        label_names = categories(label_categorical);
        label_counts = countcats(label_categorical);
        fprintf('  Saved %s: %s\n', trial_token, strjoin(compose( ...
            '%s=%d', string(label_names), label_counts), ', '));
    end
end


function on_key_change(source, event, is_press)
%ON_KEY_CHANGE Shared KeyPressFcn/KeyReleaseFcn -- updates tracked_keys
% in the figure's appdata, read back by label_one_trial's animation
% loop on the next iteration (event queue is serviced by drawnow/pause,
% NOT drawnow limitrate, which does not reliably service it).

    tracked_keys = getappdata(source, 'tracked_keys');
    if isfield(tracked_keys, event.Key)
        tracked_keys.(event.Key) = is_press;
        setappdata(source, 'tracked_keys', tracked_keys);
    end
end


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample
% count. Copied from compare_plot_save_chen_otsu_labeling_across_
% subjects.m (self-contained here per this project's convention).

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


function resampled_label = resample_labels_nearest( ...
        raw_time_s, raw_label, query_time_s)
%RESAMPLE_LABELS_NEAREST Nearest-in-time label transfer onto
% query_time_s. Copied from compare_plot_save_chen_otsu_labeling_
% across_subjects.m (self-contained here per this project's convention).

    valid_rows = isfinite(raw_time_s);
    clean_time_s = raw_time_s(valid_rows);
    clean_label = raw_label(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_label = clean_label(sort_order);
    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_label = clean_label(unique_indices);

    nearest_index = round(interp1(clean_time_s, ...
        (1:numel(clean_time_s))', query_time_s, 'nearest', 'extrap'));
    resampled_label = clean_label(nearest_index);
end


function fixation_xy_m = detect_fixations_coppelia_xy( ...
        gaze_table, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres in CoppeliaSim [m].
% Copied from compare_plot_save_chen_otsu_labeling_across_subjects.m
% (self-contained here per this project's convention).

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
% Copied from compare_plot_save_chen_otsu_labeling_across_subjects.m.

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
% Copied from compare_plot_save_chen_otsu_labeling_across_subjects.m.

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
% directions. Copied from compare_plot_save_chen_otsu_labeling_across_
% subjects.m.

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
% from compare_plot_save_chen_otsu_labeling_across_subjects.m.

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
% Copied from compare_plot_save_chen_otsu_labeling_across_subjects.m.

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end


function vector_values = parse_vector_column(raw_column, vector_size)
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV
% cells. Copied from compare_plot_save_chen_otsu_labeling_across_
% subjects.m.

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
% Copied from compare_plot_save_chen_otsu_labeling_across_subjects.m.

    valid = all(isfinite(values), 2);
end


function valid = point_is_on_display(display_point)
%POINT_IS_ON_DISPLAY Identify normalized points inside active display
% bounds. Copied from compare_plot_save_chen_otsu_labeling_across_
% subjects.m.

    valid = finite_rows(display_point) & ...
        display_point(:, 1) >= 0 & display_point(:, 1) <= 1 & ...
        display_point(:, 2) >= 0 & display_point(:, 2) <= 1;
end
