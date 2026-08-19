%% Aggregate the 18-condition pilot set into one per-condition summary table.
%
% Inputs:
%   data/raw_data/ur5e_fixed_adm_2d_<band>_<shape>_shape_d<damping>.csv
%   data/raw_data/tobii_fixed_adm_2d_<band>_<shape>_shape_d<damping>.csv
%     for every combination of BANDS x SHAPES x DAMPINGS below.
%   data/calibrations/tobii_coppelia_reference_calibration_<band>_<shape>.json
%     Manual Tobii UCS [mm] / CoppeliaSim [m] reference points.
%
% Output:
%   One row per condition written to WRITE_CONDITION_TABLE_CSV_PATH:
%   robot effort/kinematics, gaze quality, I-VT fixation statistics, a
%   speed-curvature power-law fit, and curvature-tertile fixation
%   density. condition_table remains in the workspace. No raw CSV is
%   modified.
%
% Processing:
%   Two passes per condition. Pass 1 loads both CSVs, detects I-VT
%   fixations (Tobii UCS gaze-direction space), maps fixations into the
%   CoppeliaSim x-y frame using the manual calibration reference, and
%   filters the TCP path into a uniformly sampled, twice-smoothed
%   trajectory (position -> velocity -> acceleration) from which
%   curvature and a speed-curvature power-law fit are computed. Results
%   are cached. Pass 2 computes fixation density per global curvature
%   tertile (edges pooled across all 18 conditions, so density is
%   comparable between conditions) using the Pass-1 cache.
%
% Method reuse:
%   The I-VT classification, eye-combination, and UCS-to-CoppeliaSim
%   mapping formulas are copied from
%   compute_plot_similarity_fixation_direction_force_vel.m. The TCP
%   position/curvature filter chain is copied from
%   analyze_gaze_tcp_lead_lag_curvature_v2.m's prepare_tcp_trajectory.
%   Both are self-contained local functions here, matching this
%   project's convention that MATLAB scripts do not call one another.
%
% Assumptions:
%   Tobii UCS +x/+y align with CoppeliaSim +x/+y (screen_motion_per_
%   robot_motion = 1:1 view scale), matching every sibling script's
%   calibration convention. Each ur5e_fixed_adm_2d_*.csv has a matching
%   tobii_fixed_adm_2d_*.csv from the same trial (shared time_s origin).
%   n = 1 trial per condition: this table supports descriptive/direction-
%   consistency reporting only, not inferential statistics.

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

% TODO condition-table output path [file path]; overwritten each run.
write_condition_table_csv_path = fullfile( ...
    script_directory, '..', 'data', 'processed_data', ...
    'pilot_condition_table.csv');

% TODO I-VT angular-velocity threshold [deg/s]. Matches
% compute_plot_similarity_fixation_direction_force_vel.m. Recommended
% range: 20 to 70.
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

% TODO optional correction added to synchronized gaze time [s] before
% matching a fixation to a TCP sample. Keep at zero unless an
% independently measured sensor/response delay is applied.
gaze_to_robot_time_offset_s = 0.0;

% TODO number of equal-population global curvature bins for the
% fixation-density-vs-curvature analysis [count], recommended 3 to 5.
curvature_bin_count = 3;

%% Fixed preprocessing values
% TCP position/curvature filter chain, copied from
% analyze_gaze_tcp_lead_lag_curvature_v2.m. These keep curvature
% numerically stable and do not normally need to change between trials.
tcp_position_outlier_window_s = 0.02;
tcp_position_smoothing_window_s = 0.08;
minimum_curvature_speed_m_s = 0.002;

% TODO speed-curvature power-law fit window: exclude near-zero-speed
% samples (denominator instability) and implausible curvature outliers
% [1/m], recommended kappa upper bound 200 to 1000 for a tabletop task.
power_law_kappa_lower_1_m = 1.0;
power_law_kappa_upper_1_m = 500.0;

%% Validate settings
if velocity_threshold_deg_s <= 0
    error('velocity_threshold_deg_s must be positive.');
end
if minimum_fixation_duration_s < 0
    error('minimum_fixation_duration_s cannot be negative.');
end
if screen_motion_per_robot_motion <= 0
    error('screen_motion_per_robot_motion must be positive.');
end
if any(abs(tobii_to_coppelia_axis_sign) ~= 1)
    error('Each Tobii-to-CoppeliaSim axis sign must be -1 or +1.');
end
if curvature_bin_count < 2
    error('curvature_bin_count must be at least 2.');
end
if power_law_kappa_upper_1_m <= power_law_kappa_lower_1_m
    error('power_law_kappa_upper_1_m must exceed power_law_kappa_lower_1_m.');
end

%% Pass 1: load, detect fixations, filter TCP path, cache per condition
condition_count = numel(bands) * numel(shapes) * numel(dampings);
cache = cell(condition_count, 1);
cache_index = 0;

fprintf('--- Pilot condition table: pass 1 (load + fixations + curvature) ---\n');

for band = bands
    for shape = shapes
        for damping = dampings
            cache_index = cache_index + 1;
            trial_id = sprintf('%s_%s_shape_d%d', band, shape, damping);

            robot_csv_file = fullfile( ...
                read_raw_data_directory, ...
                sprintf('ur5e_fixed_adm_2d_%s.csv', trial_id));
            gaze_csv_file = fullfile( ...
                read_raw_data_directory, ...
                sprintf('tobii_fixed_adm_2d_%s.csv', trial_id));
            calibration_json_file = fullfile( ...
                read_calibrations_directory, ...
                sprintf('tobii_coppelia_reference_calibration_%s_%s.json', ...
                band, shape));

            if ~isfile(robot_csv_file) || ~isfile(gaze_csv_file)
                warning('Skipping %s: raw CSV pair not found.', trial_id);
                continue;
            end
            if ~isfile(calibration_json_file)
                warning('Skipping %s: calibration JSON not found.', trial_id);
                continue;
            end

            fprintf('  %s\n', trial_id);

            result = process_condition( ...
                trial_id, string(band), string(shape), damping, ...
                robot_csv_file, gaze_csv_file, calibration_json_file, ...
                velocity_threshold_deg_s, minimum_fixation_duration_s, ...
                plot_only_on_display_gaze, screen_motion_per_robot_motion, ...
                tobii_to_coppelia_axis_sign, gaze_to_robot_time_offset_s, ...
                tcp_position_outlier_window_s, ...
                tcp_position_smoothing_window_s, ...
                minimum_curvature_speed_m_s, ...
                power_law_kappa_lower_1_m, power_law_kappa_upper_1_m);

            cache{cache_index} = result;
        end
    end
end

cache = cache(~cellfun(@isempty, cache));
if isempty(cache)
    error('No condition produced a result; nothing to write.');
end

%% Global curvature bin edges (pooled across all conditions)
pooled_curvature_1_m = [];
for idx = 1:numel(cache)
    pooled_curvature_1_m = [pooled_curvature_1_m; ...
        cache{idx}.tcp.curvature_1_m(cache{idx}.tcp.valid_curvature)]; %#ok<AGROW>
end
pooled_curvature_1_m = pooled_curvature_1_m(isfinite(pooled_curvature_1_m));

curvature_bin_edges_1_m = compute_percentiles( ...
    pooled_curvature_1_m, linspace(0, 1, curvature_bin_count + 1));
curvature_bin_edges_1_m(1) = -Inf;
curvature_bin_edges_1_m(end) = Inf;

fprintf('\nGlobal curvature bin edges [1/m]: %s\n', ...
    mat2str(curvature_bin_edges_1_m(2:end-1), 4));

%% Pass 2: fixation density per global curvature bin
for idx = 1:numel(cache)
    cache{idx} = add_fixation_curvature_density( ...
        cache{idx}, curvature_bin_edges_1_m);
end

%% Assemble and write the condition table
row_struct_cells = cellfun(@(c) c.row, cache, 'UniformOutput', false);
row_structs = [row_struct_cells{:}];
condition_table = struct2table(row_structs);

output_directory = fileparts(write_condition_table_csv_path);
if ~isfolder(output_directory)
    mkdir(output_directory);
end
writetable(condition_table, write_condition_table_csv_path);

fprintf('\nCondition table: %d rows x %d columns\n', ...
    height(condition_table), width(condition_table));
fprintf('Written to: %s\n', write_condition_table_csv_path);
disp(condition_table);


%% ================================================================
%% Local functions
%% ================================================================

function result = process_condition( ...
        trial_id, band, shape, damping, ...
        robot_csv_file, gaze_csv_file, calibration_json_file, ...
        velocity_threshold_deg_s, minimum_fixation_duration_s, ...
        plot_only_on_display_gaze, screen_motion_per_robot_motion, ...
        tobii_to_coppelia_axis_sign, gaze_to_robot_time_offset_s, ...
        tcp_position_outlier_window_s, tcp_position_smoothing_window_s, ...
        minimum_curvature_speed_m_s, ...
        power_law_kappa_lower_1_m, power_law_kappa_upper_1_m)
%PROCESS_CONDITION Load one trial pair and compute its summary metrics.

    %% Load and combine valid Tobii gaze points (I-VT, in UCS 3-D space)
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

    fixation_start_time_s = zeros(0, 1);
    fixation_end_time_s = zeros(0, 1);
    fixation_duration_s = zeros(0, 1);
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

        fixation_start_time_s(end + 1, 1) = gaze_time_s(start_idx); %#ok<AGROW>
        fixation_end_time_s(end + 1, 1) = gaze_time_s(end_idx); %#ok<AGROW>
        fixation_duration_s(end + 1, 1) = candidate_duration_s; %#ok<AGROW>
        fixation_mean_ucs_x_mm(end + 1, 1) = mean_fixation_ucs_mm(1); %#ok<AGROW>
        fixation_mean_ucs_y_mm(end + 1, 1) = mean_fixation_ucs_mm(2); %#ok<AGROW>
    end

    %% Load the robot trajectory and force in the CoppeliaSim frame
    robot_table = readtable(robot_csv_file, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');

    robot_time_s = get_numeric_column(robot_table.time_s);
    coppeliasim_x_m = get_numeric_column(robot_table.coppeliasim_x_m);
    coppeliasim_y_m = get_numeric_column(robot_table.coppeliasim_y_m);
    force_x_n = get_numeric_column(robot_table.Fx_filt_N);
    force_y_n = get_numeric_column(robot_table.Fy_filt_N);
    damping_x_n_s_m = get_numeric_column(robot_table.B_x_Ns_m);

    robot_valid = isfinite(robot_time_s) & isfinite(coppeliasim_x_m) & ...
        isfinite(coppeliasim_y_m) & isfinite(force_x_n) & isfinite(force_y_n);
    robot_time_s = robot_time_s(robot_valid);
    coppeliasim_x_m = coppeliasim_x_m(robot_valid);
    coppeliasim_y_m = coppeliasim_y_m(robot_valid);
    force_x_n = force_x_n(robot_valid);
    force_y_n = force_y_n(robot_valid);
    damping_x_n_s_m = damping_x_n_s_m(robot_valid);

    %% Robot effort/kinematic metrics (raw logged samples)
    force_magnitude_n = hypot(force_x_n, force_y_n);
    movement_time_s = robot_time_s(end) - robot_time_s(1);
    path_length_m = sum(hypot( ...
        diff(coppeliasim_x_m), diff(coppeliasim_y_m)));
    mean_force_n = mean(force_magnitude_n);
    peak_force_n = max(force_magnitude_n);
    impulse_n_s = trapz(robot_time_s, force_magnitude_n);

    %% Filtered TCP trajectory and curvature (CoppeliaSim frame)
    tcp = prepare_tcp_trajectory( ...
        robot_time_s, coppeliasim_x_m, coppeliasim_y_m, ...
        tcp_position_outlier_window_s, tcp_position_smoothing_window_s, ...
        minimum_curvature_speed_m_s);

    mean_speed_m_s = mean(tcp.speed_m_s);
    peak_speed_m_s = max(tcp.speed_m_s);

    %% Speed-curvature power-law fit: log(v) = log(k) - beta*log(kappa)
    fit_valid = tcp.valid_curvature & ...
        tcp.curvature_1_m >= power_law_kappa_lower_1_m & ...
        tcp.curvature_1_m <= power_law_kappa_upper_1_m;

    power_law_beta = NaN;
    power_law_r2 = NaN;
    power_law_n_used = nnz(fit_valid);

    if power_law_n_used >= 20
        log_speed = log(tcp.speed_m_s(fit_valid));
        log_kappa = log(tcp.curvature_1_m(fit_valid));
        fit_coefficients = polyfit(log_kappa, log_speed, 1);
        power_law_beta = -fit_coefficients(1);
        predicted_log_speed = polyval(fit_coefficients, log_kappa);
        residual_sum_squares = sum((log_speed - predicted_log_speed) .^ 2);
        total_sum_squares = sum((log_speed - mean(log_speed)) .^ 2);
        if total_sum_squares > 0
            power_law_r2 = 1.0 - residual_sum_squares / total_sum_squares;
        end
    end

    %% Gaze quality and fixation statistics
    gaze_n_samples = height(gaze_table);
    gaze_valid_pct = 100.0 * nnz(combined_gaze_valid) / gaze_n_samples;
    finite_gaze_time_s = gaze_time_s(isfinite(gaze_time_s));
    gaze_duration_s = max(finite_gaze_time_s) - min(finite_gaze_time_s);
    gaze_hz = (gaze_n_samples - 1) / gaze_duration_s;

    fixation_count = numel(fixation_start_time_s);
    fixation_rate_hz = fixation_count / movement_time_s;
    mean_fixation_duration_s = mean(fixation_duration_s);
    if fixation_count == 0
        mean_fixation_duration_s = NaN;
    end

    %% Map fixation centres from Tobii UCS [mm] into CoppeliaSim [m]
    % Formula matches compute_plot_similarity_fixation_direction_force_vel.m.
    calibration_reference = jsondecode(fileread(calibration_json_file));
    tobii_reference_xy_mm = double( ...
        calibration_reference.manual_tobii_reference_xy_mm(:)');
    coppelia_reference_xy_m = double( ...
        calibration_reference.manual_coppelia_reference_xy_m(:)');
    coppeliasim_m_per_tobii_mm = 1e-3 / screen_motion_per_robot_motion;

    fixation_ucs_xy_mm = [fixation_mean_ucs_x_mm, fixation_mean_ucs_y_mm];
    fixation_displacement_ucs_mm = fixation_ucs_xy_mm - tobii_reference_xy_mm;
    fixation_displacement_coppelia_m = fixation_displacement_ucs_mm .* ...
        tobii_to_coppelia_axis_sign .* coppeliasim_m_per_tobii_mm;
    fixation_coppeliasim_xy_m = ...
        coppelia_reference_xy_m + fixation_displacement_coppelia_m;

    fixation_center_time_s = 0.5 * ...
        (fixation_start_time_s + fixation_end_time_s) + ...
        gaze_to_robot_time_offset_s;

    %% Assemble the output row (fixation-density columns added in pass 2)
    row = struct( ...
        'trial_id', string(trial_id), ...
        'band', band, ...
        'shape', shape, ...
        'damping_n_s_m', mean(damping_x_n_s_m), ...
        'movement_time_s', movement_time_s, ...
        'path_length_m', path_length_m, ...
        'mean_speed_m_s', mean_speed_m_s, ...
        'peak_speed_m_s', peak_speed_m_s, ...
        'mean_force_n', mean_force_n, ...
        'peak_force_n', peak_force_n, ...
        'impulse_n_s', impulse_n_s, ...
        'gaze_n_samples', gaze_n_samples, ...
        'gaze_valid_pct', gaze_valid_pct, ...
        'gaze_hz', gaze_hz, ...
        'fixation_count', fixation_count, ...
        'fixation_rate_hz', fixation_rate_hz, ...
        'mean_fixation_duration_s', mean_fixation_duration_s, ...
        'median_curvature_1_m', median(tcp.curvature_1_m(tcp.valid_curvature)), ...
        'power_law_beta', power_law_beta, ...
        'power_law_r2', power_law_r2, ...
        'power_law_n_used', power_law_n_used, ...
        'power_law_frac_used', power_law_n_used / numel(tcp.curvature_1_m), ...
        'calibration_id', string(sprintf( ...
            'tobii_coppelia_reference_calibration_%s_%s.json', band, shape)));

    result.row = row;
    result.tcp = tcp;
    result.fixation_coppeliasim_xy_m = fixation_coppeliasim_xy_m;
    result.fixation_center_time_s = fixation_center_time_s;
    result.movement_time_s = movement_time_s;
end


function result = add_fixation_curvature_density(result, curvature_bin_edges_1_m)
%ADD_FIXATION_CURVATURE_DENSITY Fixations per global curvature tertile [Hz].
%
% Each fixation is matched to the curvature of the TCP sample nearest in
% time (Tobii and robot share the same time_s origin). Density is
% fixation count in that bin divided by total TCP time spent in that
% bin, so it is not confounded by how much time the path spends at each
% curvature level.

    bin_count = numel(curvature_bin_edges_1_m) - 1;
    tcp_bin = discretize(result.tcp.curvature_1_m, curvature_bin_edges_1_m);
    sample_interval_s = result.tcp.sample_interval_s;

    fixation_count = numel(result.fixation_center_time_s);
    fixation_bin = NaN(fixation_count, 1);
    if fixation_count > 0
        for fixation_idx = 1:fixation_count
            [~, nearest_index] = min(abs( ...
                result.tcp.time_s - result.fixation_center_time_s(fixation_idx)));
            fixation_bin(fixation_idx) = tcp_bin(nearest_index);
        end
    end

    for bin_idx = 1:bin_count
        time_in_bin_s = nnz(tcp_bin == bin_idx) * sample_interval_s;
        fixations_in_bin = nnz(fixation_bin == bin_idx);

        density_hz = NaN;
        if time_in_bin_s > 0
            density_hz = fixations_in_bin / time_in_bin_s;
        end

        field_name = sprintf('fixation_density_curvature_bin%d_hz', bin_idx);
        result.row.(field_name) = density_hz;
    end
end


function percentile_values = compute_percentiles(data, fractions)
%COMPUTE_PERCENTILES Linear-interpolation percentiles without a toolbox.
%
% Matches the default quantile() linear-interpolation method without
% requiring the Statistics and Machine Learning Toolbox (not used
% elsewhere in this project's MATLAB scripts).

    sorted_data = sort(data(:));
    sample_count = numel(sorted_data);
    percentile_values = NaN(size(fractions));

    for fraction_idx = 1:numel(fractions)
        rank = fractions(fraction_idx) * (sample_count - 1) + 1;
        lower_index = floor(rank);
        upper_index = ceil(rank);
        interpolation_weight = rank - lower_index;
        percentile_values(fraction_idx) = ...
            (1 - interpolation_weight) * sorted_data(lower_index) + ...
            interpolation_weight * sorted_data(upper_index);
    end
end


function tcp = prepare_tcp_trajectory( ...
        raw_time_s, raw_x_m, raw_y_m, ...
        outlier_window_s, smoothing_window_s, minimum_speed_m_s)
%PREPARE_TCP_TRAJECTORY Filter TCP position and calculate planar curvature.
%
% Copied from analyze_gaze_tcp_lead_lag_curvature_v2.m. Removes invalid/
% duplicate samples, interpolates onto a uniform time grid, applies
% symmetric moving-median (outlier) and moving-mean (smoothing) filters
% to position, then differentiates twice with a second smoothing pass on
% velocity before computing acceleration. Curvature is
% kappa = (vx*ay - vy*ax) / (vx^2 + vy^2)^(3/2).

    valid_rows = isfinite(raw_time_s) & isfinite(raw_x_m) & isfinite(raw_y_m);
    clean_time_s = raw_time_s(valid_rows);
    clean_x_m = raw_x_m(valid_rows);
    clean_y_m = raw_y_m(valid_rows);

    [clean_time_s, sort_order] = sort(clean_time_s);
    clean_x_m = clean_x_m(sort_order);
    clean_y_m = clean_y_m(sort_order);

    [clean_time_s, unique_indices] = unique(clean_time_s, 'stable');
    clean_x_m = clean_x_m(unique_indices);
    clean_y_m = clean_y_m(unique_indices);

    sample_intervals_s = diff(clean_time_s);
    sample_intervals_s = sample_intervals_s(sample_intervals_s > 0);
    median_interval_s = median(sample_intervals_s);

    uniform_time_s = (clean_time_s(1):median_interval_s:clean_time_s(end))';

    uniform_x_m = interp1(clean_time_s, clean_x_m, uniform_time_s, 'linear');
    uniform_y_m = interp1(clean_time_s, clean_y_m, uniform_time_s, 'linear');

    outlier_sample_count = window_to_odd_sample_count( ...
        outlier_window_s, median_interval_s, numel(uniform_time_s));
    smoothing_sample_count = window_to_odd_sample_count( ...
        smoothing_window_s, median_interval_s, numel(uniform_time_s));

    if outlier_sample_count > 1
        filtered_x_m = movmedian( ...
            uniform_x_m, outlier_sample_count, 'Endpoints', 'shrink');
        filtered_y_m = movmedian( ...
            uniform_y_m, outlier_sample_count, 'Endpoints', 'shrink');
    else
        filtered_x_m = uniform_x_m;
        filtered_y_m = uniform_y_m;
    end

    filtered_x_m = movmean( ...
        filtered_x_m, smoothing_sample_count, 'Endpoints', 'shrink');
    filtered_y_m = movmean( ...
        filtered_y_m, smoothing_sample_count, 'Endpoints', 'shrink');

    velocity_x_m_s = gradient(filtered_x_m, uniform_time_s);
    velocity_y_m_s = gradient(filtered_y_m, uniform_time_s);

    velocity_x_m_s = movmean( ...
        velocity_x_m_s, smoothing_sample_count, 'Endpoints', 'shrink');
    velocity_y_m_s = movmean( ...
        velocity_y_m_s, smoothing_sample_count, 'Endpoints', 'shrink');

    acceleration_x_m_s2 = gradient(velocity_x_m_s, uniform_time_s);
    acceleration_y_m_s2 = gradient(velocity_y_m_s, uniform_time_s);
    speed_m_s = hypot(velocity_x_m_s, velocity_y_m_s);

    signed_curvature_1_m = NaN(size(speed_m_s));
    valid_curvature = isfinite(speed_m_s) & speed_m_s >= minimum_speed_m_s;
    curvature_denominator = (velocity_x_m_s .^ 2 + velocity_y_m_s .^ 2) .^ 1.5;
    signed_curvature_1_m(valid_curvature) = ( ...
        velocity_x_m_s(valid_curvature) .* acceleration_y_m_s2(valid_curvature) - ...
        velocity_y_m_s(valid_curvature) .* acceleration_x_m_s2(valid_curvature)) ./ ...
        curvature_denominator(valid_curvature);

    tcp.time_s = uniform_time_s;
    tcp.speed_m_s = speed_m_s;
    tcp.signed_curvature_1_m = signed_curvature_1_m;
    tcp.curvature_1_m = abs(signed_curvature_1_m);
    tcp.valid_curvature = valid_curvature;
    tcp.sample_interval_s = median_interval_s;
end


function sample_count = window_to_odd_sample_count( ...
        window_s, sample_interval_s, maximum_sample_count)
%WINDOW_TO_ODD_SAMPLE_COUNT Convert a time window to an odd sample count.
% Copied from analyze_gaze_tcp_lead_lag_curvature_v2.m.

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
