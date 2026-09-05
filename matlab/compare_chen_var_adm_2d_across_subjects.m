%% Compare Chen trials across subjects: TCP path (colored by damping,
% no gaze), force over time, damping over time.
%
% Grouped into one or more figures per (shape, bandwidth) pair, one row
% per (subject, direction) trial, 3 columns: TCP path | force vs time |
% damping vs time. Color/axis scales are SHARED across every trial so
% rows are directly comparable.
%
% SCALABLE: add a new subject_ids entry and rerun -- new trials are
% found automatically (missing files are just skipped, not an error),
% and groups split into more figures once max_rows_per_figure is
% exceeded.
%
% Inputs: data/raw_data/ur5e_chen_var_adm_2d_<bandwidth>_<shape>_shape_
%   <direction>_s<id>.csv (scripts/ur5e_chen_var_adm_2d.py's schema).
% Output: figures only, nothing written to disk.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));
raw_data_directory = fullfile(script_directory, '..', 'data', 'raw_data');

%% User Settings
% Shapes: file token + display label (parallel arrays)
shape_tokens = ["pear", "fine_gross", "ellipse"];
shape_labels = ["Pear", "Curved Fine/Gross", "Teeth Ellipse"];

bandwidths = ["wide", "narrow"];
directions = ["clockwise", "counter_clockwise"];
direction_labels = ["CW", "CCW"];

subject_ids = 10:18;  % add new subjects here

max_rows_per_figure = 6;  % split into more figures beyond this

damping_min_n_s_m = 10.0;  % shared damping color/axis limits [N s/m]
damping_max_n_s_m = 200.0;

tcp_marker_size = 6;
line_width = 1.0;

%% Discover + load available trials, grouped by shape/bandwidth
empty_trial = struct('subject_id', {}, 'direction_label', {}, ...
    'time_s', {}, 'coppeliasim_x_m', {}, 'coppeliasim_y_m', {}, ...
    'damping_n_s_m', {}, 'force_magnitude_n', {});
trial_groups = struct('shape_label', {}, 'bandwidth', {}, 'trials', {});

for shape_idx = 1:numel(shape_tokens)
    for bandwidth_idx = 1:numel(bandwidths)
        trials = empty_trial;
        for subject_id = subject_ids
            for direction_idx = 1:numel(directions)
                robot_csv_file = fullfile(raw_data_directory, sprintf( ...
                    'ur5e_chen_var_adm_2d_%s_%s_shape_%s_s%d.csv', ...
                    bandwidths(bandwidth_idx), shape_tokens(shape_idx), ...
                    directions(direction_idx), subject_id));
                if ~isfile(robot_csv_file)
                    continue;
                end

                robot_table = readtable(robot_csv_file, ...
                    'Delimiter', ',', 'ReadVariableNames', true, ...
                    'VariableNamingRule', 'preserve', 'TextType', 'string');

                trials(end + 1) = struct( ...
                    'subject_id', subject_id, ...
                    'direction_label', direction_labels(direction_idx), ...
                    'time_s', robot_table.time_s, ...
                    'coppeliasim_x_m', robot_table.coppeliasim_x_m, ...
                    'coppeliasim_y_m', robot_table.coppeliasim_y_m, ...
                    'damping_n_s_m', robot_table.bd_applied_Ns_m, ...
                    'force_magnitude_n', hypot( ...
                        robot_table.Fx_filt_N, robot_table.Fy_filt_N)); %#ok<SAGROW>
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

%% Shared force-axis limit, computed across every loaded trial
all_force_n = [];
for group_idx = 1:numel(trial_groups)
    for trial_idx = 1:numel(trial_groups(group_idx).trials)
        all_force_n = [all_force_n; ...
            trial_groups(group_idx).trials(trial_idx).force_magnitude_n]; %#ok<AGROW>
    end
end
force_axis_max_n = max(all_force_n) * 1.05;

%% Plot -- one or more figures per (shape, bandwidth) group
for group_idx = 1:numel(trial_groups)
    group = trial_groups(group_idx);
    trial_count = numel(group.trials);
    figure_count = ceil(trial_count / max_rows_per_figure);

    for figure_idx = 1:figure_count
        row_start = (figure_idx - 1) * max_rows_per_figure + 1;
        row_end = min(figure_idx * max_rows_per_figure, trial_count);
        row_count = row_end - row_start + 1;

        figure('Name', sprintf('%s, %s (%d/%d)', group.shape_label, ...
            group.bandwidth, figure_idx, figure_count), 'Color', 'w');
        tiledlayout(row_count, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

        for row_idx = 1:row_count
            trial = group.trials(row_start + row_idx - 1);

            nexttile;
            scatter(trial.coppeliasim_x_m, trial.coppeliasim_y_m, ...
                tcp_marker_size, trial.damping_n_s_m, 'filled');
            axis equal;
            grid on;
            clim([damping_min_n_s_m, damping_max_n_s_m]);
            damping_colorbar = colorbar;
            damping_colorbar.Label.String = 'Damping [N s/m]';
            xlabel('x [m]');
            ylabel('y [m]');
            title(sprintf('s%d %s', trial.subject_id, trial.direction_label));

            nexttile;
            plot(trial.time_s, trial.force_magnitude_n, '-', ...
                'LineWidth', line_width);
            grid on;
            ylim([0, force_axis_max_n]);
            xlabel('Time [s]');
            ylabel('|F| [N]');

            nexttile;
            plot(trial.time_s, trial.damping_n_s_m, '-', ...
                'LineWidth', line_width);
            grid on;
            ylim([damping_min_n_s_m, damping_max_n_s_m]);
            xlabel('Time [s]');
            ylabel('Damping [N s/m]');
        end
    end
end
