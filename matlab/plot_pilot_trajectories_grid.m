%% Compact per-condition trajectory/fixation and pupil-diameter grids.
%
% Inputs:
%   data/raw_data/ur5e_fixed_adm_2d_<band>_<shape>_shape_d<damping>.csv
%   data/raw_data/tobii_fixed_adm_2d_<band>_<shape>_shape_d<damping>.csv
%   data/calibrations/tobii_coppelia_reference_calibration_<band>_<shape>.json
%     for every band x shape x damping in the pilot grid (User Settings).
%
% Output:
%   Figures 1a-1c: one figure per damping level, each a
%   tiledlayout(numel(bands), numel(shapes)) (rows = band, columns =
%   shape). Each tile shows exactly one trial's raw averaged-eye gaze
%   (faint neutral-color background, both eyes averaged per sample),
%   TCP path, I-VT fixation points, and fixation-to-fixation gaze-
%   direction arrows (saccade vectors, true scale), in the CoppeliaSim
%   x-y frame. Trials are never overlaid in the same axes. This dataset
%   was recorded with ur5e_fixed_adm_2d.py, which does not compute
%   gaze-based intent, so gaze direction here means the recorded eye-
%   movement sequence (fixation N to fixation N+1), not a controller-
%   computed intent direction.
%   Figures 2a-2c: one figure per damping level, same tile layout; each
%   tile is pupil diameter [mm] over time for that one trial, strictly
%   averaged over both eyes (samples with only one valid eye are
%   excluded, not substituted). All three figures share one y-axis
%   range (5th-95th percentile pooled across all 18 trials).
%   Figures 3a-3c: one figure per damping level, same tile layout; each
%   tile is a scatter of that trial's gaze path colored by the same
%   both-eye-averaged pupil diameter [mm], sharing one color axis
%   (same percentile range as Figures 2a-2c) across all three figures.
%   Figures 4a-4c: one figure per damping level; each figure is a
%   grouped bar chart of the number of I-VT fixations detected per
%   trial (x-axis = shape, grouped bars = band). All three figures
%   share one y-axis range so counts are comparable across damping.
%   Figure 5: one figure, a single boxplot of saccade length (Euclidean
%   distance between consecutive I-VT fixations) [m], pooled from every
%   trial and grouped by damping (major grouping) then band-shape
%   (minor grouping), so the full distribution is visible in one view.
%   No file is written; this script only reads raw CSVs and calibration
%   JSON.
%
% Method reuse:
%   I-VT fixation detection, eye combination, and UCS-to-CoppeliaSim
%   mapping are copied from compute_plot_similarity_fixation_direction_
%   force_vel.m (also reused in build_pilot_condition_table.m), self-
%   contained here per this project's convention that MATLAB scripts do
%   not call one another.
%
% Assumptions:
%   Tobii UCS +x/+y align with CoppeliaSim +x/+y (screen_motion_per_
%   robot_motion = 1:1 view scale). One calibration JSON per band x
%   shape, shared across its 3 damping trials. Each ur5e_fixed_adm_2d_
%   *.csv has a matching tobii_fixed_adm_2d_*.csv from the same trial.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO Experiment grid. Must match the raw_data filename stems exactly:
% ur5e_fixed_adm_2d_<band>_<shape>_shape_d<damping>.csv
bands = ["narrow", "wide"];
shapes = ["circle", "curved", "edge"];
dampings = [10, 50, 100];

% TODO raw CSV and calibration JSON input folders [folder path].
read_raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');
read_calibrations_directory = fullfile( ...
    script_directory, '..', 'data', 'calibrations');

% TODO I-VT angular-velocity threshold [deg/s]. Matches
% compute_plot_similarity_fixation_direction_force_vel.m.
velocity_threshold_deg_s = 20.0;

% TODO minimum retained fixation duration [s]. Matches
% compute_plot_similarity_fixation_direction_force_vel.m.
minimum_fixation_duration_s = 0.060;

% TODO true excludes gaze estimates outside normalized display [0, 1]^2.
plot_only_on_display_gaze = true;

% TODO physical screen motion / robot motion [unitless], must be > 0.
screen_motion_per_robot_motion = 1.0;

% TODO Tobii UCS axis signs in CoppeliaSim [unitless], each -1 or +1.
tobii_to_coppelia_axis_sign = [1.0, 1.0];

% TODO Figure 1 fixation marker size [points^2] and RGB color.
fixation_marker_size = 14;
fixation_marker_color = 'g';

% TODO Figure 1 raw (per-sample, pre-fixation) averaged-eye gaze marker
% size [points^2], RGB color, and opacity [0-1]. Kept light/neutral so
% this background layer does not compete with the trajectory, fixation,
% and gaze-direction layers drawn on top of it.
raw_gaze_marker_size = 4;
raw_gaze_marker_color = 'r';
raw_gaze_marker_alpha = 1.0;

% TODO Figure 1 TCP trajectory line color and width. One trial per
% tile now, so this no longer needs to vary by damping.
trajectory_line_color = 'k';
trajectory_line_width = 1.4;

% TODO Figure 1 gaze-direction arrow color and style: consecutive-
% fixation (saccade) vectors, drawn at true scale (no auto-scaling).
gaze_direction_color = [0.10, 0.45, 0.85];
gaze_direction_line_width = 0.8;
gaze_direction_max_head_size = 0.5;

% TODO Figures 2-3 pupil diameter [mm] two-sided percentile range
% [0-1 each]; pupil diameter has no meaningful zero floor (unlike
% curvature), so both ends are clipped against rare artifacts.
pupil_lower_percentile = 0.05;
pupil_upper_percentile = 0.95;

% TODO Figure 2 temporal pupil-diameter line width.
pupil_line_width = 1.0;

% TODO Figure 3 spatial pupil-diameter marker size [points^2].
pupil_spatial_marker_size = 8;

%% ================================================================
% Figure 1a-1c: one figure per damping -- TCP trajectory, raw gaze,
% fixations, and gaze direction, one trial per tile (no overlap)
% =================================================================

for damping = dampings
    figure('Name', sprintf('TCP trajectory and gaze, B = %d N s/m', ...
        damping), 'Color', 'w');
    tiledlayout(numel(bands), numel(shapes), ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    for band = bands
        for shape = shapes
            nexttile;
            hold on;

            calibration_json_file = fullfile( ...
                read_calibrations_directory, ...
                sprintf('tobii_coppelia_reference_calibration_%s_%s.json', ...
                band, shape));

            if ~isfile(calibration_json_file)
                warning('Skipping %s %s: calibration JSON not found.', ...
                    band, shape);
                title(sprintf('%s %s (no calibration)', band, shape));
                continue;
            end

            calibration_reference = jsondecode(fileread(calibration_json_file));
            tobii_reference_xy_mm = double( ...
                calibration_reference.manual_tobii_reference_xy_mm(:)');
            coppelia_reference_xy_m = double( ...
                calibration_reference.manual_coppelia_reference_xy_m(:)');

            trial_id = sprintf('%s_%s_shape_d%d', band, shape, damping);
            robot_csv_file = fullfile(read_raw_data_directory, ...
                sprintf('ur5e_fixed_adm_2d_%s.csv', trial_id));
            gaze_csv_file = fullfile(read_raw_data_directory, ...
                sprintf('tobii_fixed_adm_2d_%s.csv', trial_id));

            if ~isfile(robot_csv_file) || ~isfile(gaze_csv_file)
                warning('Skipping %s: raw CSV pair not found.', trial_id);
                title(sprintf('%s %s (no data)', band, shape));
                continue;
            end

            robot_table = readtable(robot_csv_file, ...
                'Delimiter', ',', 'ReadVariableNames', true, ...
                'VariableNamingRule', 'preserve', 'TextType', 'string');
            coppeliasim_x_m = get_numeric_column(robot_table.coppeliasim_x_m);
            coppeliasim_y_m = get_numeric_column(robot_table.coppeliasim_y_m);

            [fixation_xy_m, raw_gaze_xy_m] = detect_fixations_coppelia_xy( ...
                gaze_csv_file, tobii_reference_xy_mm, ...
                coppelia_reference_xy_m, screen_motion_per_robot_motion, ...
                tobii_to_coppelia_axis_sign, velocity_threshold_deg_s, ...
                minimum_fixation_duration_s, plot_only_on_display_gaze);

            % Raw two-eye-averaged gaze cloud, drawn first (background).
            if ~isempty(raw_gaze_xy_m)
                scatter(raw_gaze_xy_m(:, 1), raw_gaze_xy_m(:, 2), ...
                    raw_gaze_marker_size, raw_gaze_marker_color, ...
                    'filled', 'MarkerFaceAlpha', raw_gaze_marker_alpha, ...
                    'DisplayName', 'Raw gaze (both eyes)');
            end

            plot(coppeliasim_x_m, coppeliasim_y_m, '-', ...
                'Color', trajectory_line_color, ...
                'LineWidth', trajectory_line_width, ...
                'DisplayName', 'TCP trajectory');

            if ~isempty(fixation_xy_m)
                scatter(fixation_xy_m(:, 1), fixation_xy_m(:, 2), ...
                    fixation_marker_size, fixation_marker_color, ...
                    'filled', 'MarkerFaceAlpha', 0.6, ...
                    'DisplayName', 'I-VT fixation');
            end

            % Gaze direction: vector from each fixation to the next one
            % (saccade direction), true scale.
            if size(fixation_xy_m, 1) >= 2
                transition_dx_m = diff(fixation_xy_m(:, 1));
                transition_dy_m = diff(fixation_xy_m(:, 2));
                quiver(fixation_xy_m(1:end-1, 1), fixation_xy_m(1:end-1, 2), ...
                    transition_dx_m, transition_dy_m, 0, ...
                    'Color', gaze_direction_color, ...
                    'LineWidth', gaze_direction_line_width, ...
                    'MaxHeadSize', gaze_direction_max_head_size, ...
                    'DisplayName', 'Gaze direction');
            end

            axis equal;
            grid on;
            title(sprintf('%s %s', band, shape));
            if band == bands(1) && shape == shapes(1)
                legend('Location', 'best', 'FontSize', 7);
            end
        end
    end
    sgtitle(sprintf(['B = %d N s/m -- TCP path, raw gaze (faint dots), ' ...
        'I-VT fixations, and gaze direction (arrows)'], damping));
end

%% ================================================================
% Figures 2a-2c and 3a-3c: pupil diameter, temporal and spatial,
% averaged over both eyes (strict two-eye average, not single-eye
% fallback), one figure per damping, one trial per tile (no overlap)
% =================================================================

pupil_tile_data = cell(numel(bands), numel(shapes), numel(dampings));
global_pupil_diameter_mm = [];

for band_idx = 1:numel(bands)
    for shape_idx = 1:numel(shapes)
        for damping_idx = 1:numel(dampings)
            band = bands(band_idx);
            shape = shapes(shape_idx);

            calibration_json_file = fullfile( ...
                read_calibrations_directory, ...
                sprintf('tobii_coppelia_reference_calibration_%s_%s.json', ...
                band, shape));
            trial_id = sprintf('%s_%s_shape_d%d', band, shape, ...
                dampings(damping_idx));
            gaze_csv_file = fullfile(read_raw_data_directory, ...
                sprintf('tobii_fixed_adm_2d_%s.csv', trial_id));

            if ~isfile(calibration_json_file) || ~isfile(gaze_csv_file)
                continue;
            end

            calibration_reference = jsondecode(fileread(calibration_json_file));
            tobii_reference_xy_mm = double( ...
                calibration_reference.manual_tobii_reference_xy_mm(:)');
            coppelia_reference_xy_m = double( ...
                calibration_reference.manual_coppelia_reference_xy_m(:)');

            [pupil_time_s, pupil_diameter_mm, pupil_gaze_xy_m, ...
                pupil_diameter_at_gaze_mm] = read_gaze_pupil_series( ...
                gaze_csv_file, tobii_reference_xy_mm, ...
                coppelia_reference_xy_m, screen_motion_per_robot_motion, ...
                tobii_to_coppelia_axis_sign, plot_only_on_display_gaze);

            pupil_tile_data{band_idx, shape_idx, damping_idx} = struct( ...
                'time_s', pupil_time_s, ...
                'diameter_mm', pupil_diameter_mm, ...
                'xy_m', pupil_gaze_xy_m, ...
                'diameter_at_xy_mm', pupil_diameter_at_gaze_mm);
            global_pupil_diameter_mm = [global_pupil_diameter_mm; ...
                pupil_diameter_mm]; %#ok<AGROW>
        end
    end
end

pupil_range_mm = [ ...
    compute_percentiles(global_pupil_diameter_mm, pupil_lower_percentile), ...
    compute_percentiles(global_pupil_diameter_mm, pupil_upper_percentile)];

% ---- Figure 2a-2c: pupil diameter over time
for damping_idx = 1:numel(dampings)
    figure('Name', sprintf('Pupil diameter over time, B = %d N s/m', ...
        dampings(damping_idx)), 'Color', 'w');
    tiledlayout(numel(bands), numel(shapes), ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    for band_idx = 1:numel(bands)
        for shape_idx = 1:numel(shapes)
            nexttile;
            this_tile = pupil_tile_data{band_idx, shape_idx, damping_idx};

            if isempty(this_tile)
                title(sprintf('%s %s (no data)', ...
                    bands(band_idx), shapes(shape_idx)));
                continue;
            end

            plot(this_tile.time_s, this_tile.diameter_mm, '-', ...
                'LineWidth', pupil_line_width);

            grid on;
            ylim(pupil_range_mm);
            xlabel('Time [s]');
            ylabel('Pupil diameter [mm]');
            title(sprintf('%s %s', bands(band_idx), shapes(shape_idx)));
        end
    end
    sgtitle(sprintf( ...
        'B = %d N s/m -- pupil diameter over time (both eyes averaged)', ...
        dampings(damping_idx)));
end

% ---- Figure 3a-3c: pupil diameter over the gaze path (spatial heatmap)
for damping_idx = 1:numel(dampings)
    figure('Name', sprintf('Pupil diameter heatmap, B = %d N s/m', ...
        dampings(damping_idx)), 'Color', 'w');
    tile_layout = tiledlayout(numel(bands), numel(shapes), ...
        'TileSpacing', 'compact', 'Padding', 'compact');

    for band_idx = 1:numel(bands)
        for shape_idx = 1:numel(shapes)
            nexttile;
            this_tile = pupil_tile_data{band_idx, shape_idx, damping_idx};

            if isempty(this_tile)
                title(sprintf('%s %s (no data)', ...
                    bands(band_idx), shapes(shape_idx)));
                continue;
            end

            scatter(this_tile.xy_m(:, 1), this_tile.xy_m(:, 2), ...
                pupil_spatial_marker_size, this_tile.diameter_at_xy_mm, ...
                'filled');

            axis equal;
            grid on;
            caxis(pupil_range_mm); %#ok<CAXIS>
            title(sprintf('%s %s', bands(band_idx), shapes(shape_idx)));
        end
    end

    colormap;
    pupil_colorbar = colorbar;
    pupil_colorbar.Layout.Tile = 'east';
    pupil_colorbar.Label.String = 'Pupil diameter [mm]';
    sgtitle(sprintf( ...
        'B = %d N s/m -- pupil diameter over the gaze path (both eyes)', ...
        dampings(damping_idx)));
end

%% ================================================================
% Figures 4a-4c and 5: I-VT fixation count and saccade length --
% needs only the fixation points themselves, no TCP-path preprocessing
% =================================================================

fixation_count = NaN(numel(bands), numel(shapes), numel(dampings));
all_saccade_lengths_m = zeros(0, 1);
all_saccade_damping_label = strings(0, 1);
all_saccade_bandshape_label = strings(0, 1);

for band_idx = 1:numel(bands)
    for shape_idx = 1:numel(shapes)
        for damping_idx = 1:numel(dampings)
            band = bands(band_idx);
            shape = shapes(shape_idx);

            calibration_json_file = fullfile( ...
                read_calibrations_directory, ...
                sprintf('tobii_coppelia_reference_calibration_%s_%s.json', ...
                band, shape));
            trial_id = sprintf('%s_%s_shape_d%d', band, shape, ...
                dampings(damping_idx));
            gaze_csv_file = fullfile(read_raw_data_directory, ...
                sprintf('tobii_fixed_adm_2d_%s.csv', trial_id));

            if ~isfile(calibration_json_file) || ~isfile(gaze_csv_file)
                continue;
            end

            calibration_reference = jsondecode(fileread(calibration_json_file));
            tobii_reference_xy_mm = double( ...
                calibration_reference.manual_tobii_reference_xy_mm(:)');
            coppelia_reference_xy_m = double( ...
                calibration_reference.manual_coppelia_reference_xy_m(:)');

            fixation_xy_m = detect_fixations_coppelia_xy( ...
                gaze_csv_file, tobii_reference_xy_mm, ...
                coppelia_reference_xy_m, screen_motion_per_robot_motion, ...
                tobii_to_coppelia_axis_sign, velocity_threshold_deg_s, ...
                minimum_fixation_duration_s, plot_only_on_display_gaze);

            fixation_count(band_idx, shape_idx, damping_idx) = ...
                size(fixation_xy_m, 1);

            if size(fixation_xy_m, 1) >= 2
                trial_saccade_lengths_m = vecnorm( ...
                    diff(fixation_xy_m, 1, 1), 2, 2);
                trial_saccade_count = numel(trial_saccade_lengths_m);
                all_saccade_lengths_m = [all_saccade_lengths_m; ...
                    trial_saccade_lengths_m]; %#ok<AGROW>
                all_saccade_damping_label = [all_saccade_damping_label; ...
                    repmat(sprintf("B = %d", dampings(damping_idx)), ...
                    trial_saccade_count, 1)]; %#ok<AGROW>
                all_saccade_bandshape_label = [all_saccade_bandshape_label; ...
                    repmat(band + "-" + shape, trial_saccade_count, 1)]; %#ok<AGROW>
            end
        end
    end
end

% ---- Figure 4a-4c: number of I-VT fixations per trial. A true
% histogram needs many samples per bin, but each condition here has
% exactly one trial (one count), so a grouped bar chart -- one bar per
% shape x band, one figure per damping, shared y-axis -- is the
% interpretable equivalent for this categorical, single-sample-per-
% condition count data.
fixation_count_y_limits = [0, max(fixation_count(:)) * 1.05];

for damping_idx = 1:numel(dampings)
    figure('Name', sprintf('Fixation count, B = %d N s/m', ...
        dampings(damping_idx)), 'Color', 'w');

    bar(fixation_count(:, :, damping_idx)');

    set(gca, 'XTickLabel', cellstr(shapes));
    ylim(fixation_count_y_limits);
    grid on;
    xlabel('Shape');
    ylabel('Number of I-VT fixations');
    legend(cellstr(bands), 'Location', 'best');
    title(sprintf('B = %d N s/m -- fixation count per trial', ...
        dampings(damping_idx)));
end

% ---- Figure 5: saccade length (Euclidean distance between consecutive
% I-VT fixations), pooled across all 18 trials and grouped by damping
% (major grouping) then band-shape (minor grouping) in one boxplot, so
% every condition's distribution is visible in a single view.
figure('Name', 'Saccade length by condition', 'Color', 'w');
boxplot(all_saccade_lengths_m, ...
    {all_saccade_damping_label, all_saccade_bandshape_label}, ...
    'FactorGap', [10, 2], 'LabelOrientation', 'inline');
grid on;
ylabel('Saccade length [m]');
title(['Length of consecutive-fixation saccades, grouped by damping ' ...
    'then band-shape']);


%% ================================================================
%% Local functions
%% ================================================================

function [fixation_xy_m, raw_gaze_xy_m] = detect_fixations_coppelia_xy( ...
        gaze_csv_file, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze)
%DETECT_FIXATIONS_COPPELIA_XY I-VT fixation centres mapped to CoppeliaSim [m].
%
% Reuses the I-VT and UCS-to-CoppeliaSim mapping formulas from
% compute_plot_similarity_fixation_direction_force_vel.m /
% build_pilot_condition_table.m. fixation_xy_m is N-by-2 (empty if no
% fixation meets minimum_fixation_duration_s). raw_gaze_xy_m is every
% raw, pre-fixation-grouping sample where BOTH left and right eyes are
% valid, strictly averaged (left + right) / 2 -- samples with only one
% valid eye are excluded, not substituted -- mapped the same way.

    gaze_table = readtable(gaze_csv_file, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');

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
        mean_fixation_ucs_mm = mean(combined_gaze_ucs_mm(fixation_rows, :), 1);
        fixation_mean_ucs_x_mm(end + 1, 1) = mean_fixation_ucs_mm(1); %#ok<AGROW>
        fixation_mean_ucs_y_mm(end + 1, 1) = mean_fixation_ucs_mm(2); %#ok<AGROW>
    end

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    fixation_ucs_xy_mm = [fixation_mean_ucs_x_mm, fixation_mean_ucs_y_mm];
    fixation_displacement_ucs_mm = fixation_ucs_xy_mm - tobii_reference_xy_mm;
    fixation_displacement_coppelia_m = fixation_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    fixation_xy_m = coppelia_reference_xy_m + fixation_displacement_coppelia_m;

    % Every raw (pre-fixation-grouping) sample where BOTH eyes are
    % valid, averaged left+right (strict two-eye average -- unlike
    % combined_gaze_ucs_mm above, this does not fall back to a single
    % eye when only one is valid), mapped the same way.
    both_eyes_point_valid = left_point_valid & right_point_valid;
    raw_gaze_ucs_xy_mm = 0.5 * ( ...
        left_gaze_ucs_mm(both_eyes_point_valid, 1:2) + ...
        right_gaze_ucs_mm(both_eyes_point_valid, 1:2));
    raw_gaze_displacement_ucs_mm = raw_gaze_ucs_xy_mm - tobii_reference_xy_mm;
    raw_gaze_displacement_coppelia_m = raw_gaze_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    raw_gaze_xy_m = coppelia_reference_xy_m + raw_gaze_displacement_coppelia_m;
end


function [pupil_time_s, pupil_diameter_mm, pupil_gaze_xy_m, ...
        pupil_diameter_at_gaze_mm] = read_gaze_pupil_series( ...
        gaze_csv_file, tobii_reference_xy_mm, coppelia_reference_xy_m, ...
        screen_motion_per_robot_motion, tobii_to_coppelia_axis_sign, ...
        plot_only_on_display_gaze)
%READ_GAZE_PUPIL_SERIES Both-eye-averaged pupil diameter, temporal and spatial.
%
% pupil_time_s / pupil_diameter_mm: every sample where BOTH left and
% right pupil measurements are valid, strictly averaged (left+right)/2
% (not a single-eye fallback) -- independent of gaze-point validity,
% since pupil diameter is a physiological measurement, not a position.
%
% pupil_gaze_xy_m / pupil_diameter_at_gaze_mm: the subset of those
% samples where BOTH gaze points are also valid (and on-display, if
% plot_only_on_display_gaze), so each pupil value has a meaningful
% CoppeliaSim [m] location; mapped with the same UCS-to-CoppeliaSim
% formula as detect_fixations_coppelia_xy.

    gaze_table = readtable(gaze_csv_file, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');

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
    left_display_gaze = parse_vector_column( ...
        gaze_table.left_gaze_point_on_display_area, 2);
    right_display_gaze = parse_vector_column( ...
        gaze_table.right_gaze_point_on_display_area, 2);

    left_point_valid = get_numeric_column( ...
        gaze_table.left_gaze_point_validity) == 1 & finite_rows(left_gaze_ucs_mm);
    right_point_valid = get_numeric_column( ...
        gaze_table.right_gaze_point_validity) == 1 & finite_rows(right_gaze_ucs_mm);

    if plot_only_on_display_gaze
        left_point_valid = left_point_valid & ...
            point_is_on_display(left_display_gaze);
        right_point_valid = right_point_valid & ...
            point_is_on_display(right_display_gaze);
    end

    spatial_valid = both_pupil_valid & left_point_valid & right_point_valid;

    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;
    gaze_ucs_xy_mm = 0.5 * ( ...
        left_gaze_ucs_mm(spatial_valid, 1:2) + ...
        right_gaze_ucs_mm(spatial_valid, 1:2));
    gaze_displacement_coppelia_m = (gaze_ucs_xy_mm - tobii_reference_xy_mm) .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    pupil_gaze_xy_m = coppelia_reference_xy_m + gaze_displacement_coppelia_m;

    pupil_diameter_at_gaze_mm = 0.5 * ( ...
        left_pupil_mm(spatial_valid) + right_pupil_mm(spatial_valid));
end


function percentile_value = compute_percentiles(data, fraction)
%COMPUTE_PERCENTILES Linear-interpolation percentile without a toolbox.
%
% Matches the default quantile() linear-interpolation method without
% requiring the Statistics and Machine Learning Toolbox (not used
% elsewhere in this project's MATLAB scripts).

    sorted_data = sort(data(:));
    sample_count = numel(sorted_data);

    rank = fraction * (sample_count - 1) + 1;
    lower_index = floor(rank);
    upper_index = ceil(rank);
    interpolation_weight = rank - lower_index;
    percentile_value = ...
        (1 - interpolation_weight) * sorted_data(lower_index) + ...
        interpolation_weight * sorted_data(upper_index);
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
%PARSE_VECTOR_COLUMN Parse JSON-like vectors stored in individual CSV cells.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    direction = gaze_point_ucs_mm - gaze_origin_ucs_mm;
    direction_norm = sqrt(sum(direction .^ 2, 2));
    eye_valid = origin_validity & point_validity & ...
        finite_rows(gaze_origin_ucs_mm) & finite_rows(gaze_point_ucs_mm) & ...
        isfinite(direction_norm) & direction_norm > 0;

    unit_direction = NaN(size(direction));
    unit_direction(eye_valid, :) = direction(eye_valid, :) ./ ...
        direction_norm(eye_valid);
end


function [combined_direction, combined_valid] = combine_valid_eye_directions( ...
        left_direction, right_direction, left_valid, right_valid)
%COMBINE_VALID_EYE_DIRECTIONS Average and normalize all valid eye directions.
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

    sample_count = size(left_direction, 1);
    combined_valid = left_valid | right_valid;

    direction_sum = zeros(sample_count, 3);
    direction_sum(left_valid, :) = left_direction(left_valid, :);
    direction_sum(right_valid, :) = direction_sum(right_valid, :) + ...
        right_direction(right_valid, :);

    direction_norm = sqrt(sum(direction_sum .^ 2, 2));
    combined_valid = combined_valid & isfinite(direction_norm) & direction_norm > 0;

    combined_direction = NaN(sample_count, 3);
    combined_direction(combined_valid, :) = ...
        direction_sum(combined_valid, :) ./ direction_norm(combined_valid);
end


function angular_velocity_deg_s = calculate_angular_velocity( ...
        gaze_direction, time_s, gaze_valid)
%CALCULATE_ANGULAR_VELOCITY Point-to-point gaze speed [deg/s].
% Copied from compute_plot_similarity_fixation_direction_force_vel.m.

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
        if end_idx > start_idx && isfinite(angular_velocity_deg_s(start_idx + 1))
            angular_velocity_deg_s(start_idx) = angular_velocity_deg_s(start_idx + 1);
        end
    end
end
