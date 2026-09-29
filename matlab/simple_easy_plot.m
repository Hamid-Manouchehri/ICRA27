%% Angular divergence over time per subject, Chen vs Gated, one tile per subject.
% Same shape and bandwidth for both controllers, so each tile compares like
% with like.
%
% Inputs:
%   - data/processed_data/accuracy_metrics_exploration_all_subjects.mat
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - d_0921_tracking_metrics_all_subjects.m (writes the input .mat)

clc;
clear;
close all;

%% User Setting
read_metrics_mat = "C:\Users\Hamidjoon\projects\ICRA27\data\processed_data\" + ...
    "accuracy_metrics_exploration_all_subjects.mat";

% TODO condition to compare. Angular divergence is only reliable for
% fine_gross + wide and for ellipse (narrow or wide): the other centerlines
% are ordered against the travel direction.
plot_shape_name = "fine_gross";
plot_bandwidth_label = "wide";

chen_color = [0.20 0.40 0.80];
gated_color = [0.85 0.35 0.10];

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_metrics_mat);
fprintf('writing:\n  nothing (figure only)\n');

data = load(read_metrics_mat);

T = data.trial_metrics;

shape_code = find(string(data.shape_name_L) == plot_shape_name);
bandwidth_code = find(string(data.bandwidth_label_L) == plot_bandwidth_label);
chen_code = find(string(data.controller_label_L) == "Chen");
gated_code = find(string(data.controller_label_L) == "Gated");

subjects = unique(T.subject_id);

figure('Name', 'angular_divergence_per_subject');
layout = tiledlayout('flow', 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:numel(subjects)
    same_condition = T.subject_id == subjects(i) & ...
        T.shape_index == shape_code & T.bandwidth_index == bandwidth_code;
    chen_row = find(same_condition & T.controller_index == chen_code);
    gated_row = find(same_condition & T.controller_index == gated_code);

    % s19 has no Chen trial for the narrow ellipse / narrow fine_gross.
    if isempty(chen_row) || isempty(gated_row)
        continue;
    end

    nexttile;
    hold on;

    plot(T.time_s{chen_row}, rad2deg(T.angular_divergence_rad{chen_row}), ...
        'Color', chen_color, 'LineWidth', 0.5, 'DisplayName', 'Chen');

    plot(T.time_s{gated_row}, rad2deg(T.angular_divergence_rad{gated_row}), ...
        'Color', gated_color, 'LineWidth', 0.5, 'DisplayName', 'Gated');

    title(sprintf('s%d', subjects(i)), 'FontSize', 9);
    grid on;
end

key = legend(gca);  % Chen / Gated key, one for the whole figure
key.Layout.Tile = 'east';
% xlabel(layout, 'time [s]');
% ylabel(layout, 'angular divergence [deg]');
% title(layout, sprintf('Angular divergence, Chen vs Gated: shape = %s, bandwidth = %s', ...
%     plot_shape_name, plot_bandwidth_label), 'Interpreter', 'none');
