%% One Gated trial: the logged classifier features over time.
% No z-scoring, averaging or rescaling: each tile is a signal as the
% controller had it. Gray = TCP in the fine area (fine_mask_fun); dashed
% lines = the entries (green) and exits (red) the timing analysis uses.
%
% Inputs:
%   - data/trial_manifest.csv (shape of the trial)
%   - data/raw_data/test_subjects/<robot_csv_name>
%
% Outputs:
%   - one figure, nothing written to disk
%
% Depends on:
%   - functions/fine_mask_fun.m

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
% robot_csv of one Gated trial, see data/trial_manifest.csv
robot_csv_name = "ur5e_gaze_assisted_chen_vac_2d_held_out_fine_gross_and_" + ...
    "s11_on_narrow_fine_gross_shape.csv";
read_robot_csv = fullfile(data_directory, 'raw_data', 'test_subjects', ...
    robot_csv_name);
manifest_file = fullfile(data_directory, 'trial_manifest.csv');

time_window_s = [0, Inf];  % shown part of the trial [s]
min_before_s = 2.0;  % as in d_1019_feature_timing_at_regime_change [s]
min_after_s = 3.0;  % as in d_1019_feature_timing_at_regime_change [s]

signal_names = ["coppeliasim_x_m", "gaze_gaze_path_efficiency", ...
    "gaze_p90_gaze_ee_distance", "robot_direction_concentration", ...
    "robot_std_curvature", "fine_probability"];
signal_labels = ["x [m]", "path efficiency (gaze)", ...
    "p90 gaze-EE distance (gaze)", "direction concentration (robot)", ...
    "std curvature (robot)", "P_t"];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\n', manifest_file, read_robot_csv);
fprintf('writing:\n  nothing (figure only)\n');

M = readtable(manifest_file, 'TextType', 'string');
trial = M(M.robot_csv == robot_csv_name, :);
T = readtable(read_robot_csv);
time_s = T.time_s;

% No features before the first classifier update.
for k = 2:numel(signal_names)
    T.(signal_names(k))(T.classifier_update_count == 0) = NaN;
end

% Fine-area periods, for the gray shading.
fine = fine_mask_fun(T.coppeliasim_x_m, ...
    find(["fine_gross", "pear", "ellipse"] == trial.shape));
edges = diff([0; double(fine); 0]);
fine_first = find(edges == 1);
fine_last = find(edges == -1) - 1;

% Entries and exits kept by the timing analysis (long enough segments).
change = find(diff(double(fine)) ~= 0) + 1;
bounds = [1; change; numel(fine) + 1];
segment_s = time_s(bounds(2:end) - 1) - time_s(bounds(1:end - 1));
used = segment_s(1:end - 1) >= min_before_s & ...
    segment_s(2:end) >= min_after_s;
entry_s = time_s(change(used & fine(change)));
exit_s = time_s(change(used & ~fine(change)));

fprintf('\n%d boundary crossings, used: entries %s s, exits %s s\n', ...
    numel(change), mat2str(entry_s', 4), mat2str(exit_s', 4));

%% Plot
figure('Name', sprintf('one_trial_features_%s', ...
    erase(robot_csv_name, '.csv')), 'Color', 'w');
tiledlayout(numel(signal_names), 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');
for k = 1:numel(signal_names)
    nexttile;
    hold on;
    value = T.(signal_names(k));
    y_limits = [min(value), max(value)] + ...
        [-0.05, 0.05] * (max(value) - min(value));
    for s = 1:numel(fine_first)
        patch(time_s([fine_first(s), fine_last(s), fine_last(s), ...
            fine_first(s)])', y_limits([1, 1, 2, 2]), [0.85, 0.85, 0.85], ...
            'EdgeColor', 'none');
    end
    plot(time_s, value, 'k');
    plot([entry_s, entry_s]', repmat(y_limits', 1, numel(entry_s)), ...
        'g--', 'LineWidth', 1.5);
    plot([exit_s, exit_s]', repmat(y_limits', 1, numel(exit_s)), ...
        'r--', 'LineWidth', 1.5);
    ylim(y_limits);
    xlim(time_window_s);
    ylabel(signal_labels(k));
    grid on;
    if k == 1
        title({sprintf('s%d, Gated, %s %s', trial.subject, ...
            trial.bandwidth, replace(trial.shape, "fine_gross", "paw")), ...
            ['gray: fine area (x rule), green/red dashed: entries/' ...
            'exits used in the timing analysis']});
    end
end
xlabel('time [s]');
linkaxes(findobj(gcf, 'Type', 'axes'), 'x');  % zoom all tiles together
