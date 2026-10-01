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

shape_code = 1;  % fine_gross:1, pear: 2, ellipse: 3
bandwidth_code = 1;  % wide: 1, narrow: 2
chen_code = 1;
gated_code = 2;

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_metrics_mat);
fprintf('writing:\n  nothing (figure only)\n');

data = load(read_metrics_mat);
T = data.trial_metrics;
subjects = unique(T.subject_id);


% plots per subject:
figure('Name', 'angular_divergence_per_subject');
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
        'Color', 'r');

    plot(T.time_s{gated_row}, rad2deg(T.angular_divergence_rad{gated_row}), ...
        'Color', 'b');

    title(sprintf('s%d', subjects(i)));
    legend(["Chen", "Gated"])
    grid on;
end

figure('Name', 'mean angular_divergence_per_subject');
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

    % plot(T.time_s{chen_row}, rad2deg(T.angular_divergence_rad{chen_row}), ...
    %     'Color', 'r');
    % 
    % plot(T.time_s{gated_row}, rad2deg(T.angular_divergence_rad{gated_row}), ...
    %     'Color', 'b');

    bar(["chen", "gated"], [mean(T.angular_divergence_rad{chen_row}), ...
        mean(T.angular_divergence_rad{gated_row})])

    title(sprintf('s%d', subjects(i)));
    % legend(["Chen", "Gated"])
    grid on;
end



