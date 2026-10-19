%% Draft Fig. B: Gated vs Chen per participant, one figure per metric.
% Left: participant means over the conditions with both trials. Right:
% Gated - Chen per participant in each condition, bar = median.
%
% Inputs:
%   - data/processed_data/performance_features_all_trials.csv
%
% Outputs:
%   - figures only, nothing written to disk
%
% Depends on:
%   - none (the CSV is an export of the d_0929 features)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
read_features_csv = fullfile(data_directory, 'processed_data', ...
    'performance_features_all_trials.csv');

chen_code = 1;
gated_code = 2;

% violations_per_m is derived below.
metric_names = ["sparc", "mean_force", "violations_per_m", ...
    "total_progress_m"];
metric_labels = ["SPARC (smoothness)", "mean force [N] (effort)", ...
    "boundary violations per m (accuracy)", ...
    "centerline progress per trial [m] (pace)"];

tick_font_size = 18;  % axis numbers and tick labels [pt]
label_font_size = 18;  % axis labels [pt]
subject_dot_size = 60;  % subject dot area, both panels [pt^2]

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', read_features_csv);
fprintf('writing:\n  nothing (figures only)\n');

T = readtable(read_features_csv);
T.violations_per_m = T.n_boundary_violation ./ T.travelled_path_m;
subjects = unique(T.subject_id);
n_subjects = numel(subjects);

% Conditions in plot order.
condition_shape = [1, 1, 2, 2, 3, 3];  % fine_gross(paw):1, pear: 2, ellipse: 3
condition_bandwidth = [1, 2, 1, 2, 1, 2];  % wide: 1, narrow: 2
condition_labels = ["wide paw", "narrow paw", "wide pear", ...
    "narrow pear", "wide ellipse", "narrow ellipse"];

for k = 1:numel(metric_names)
    metric = T.(metric_names(k));

    % One value per participant and condition, NaN without a pair.
    chen_value = nan(n_subjects, 6);
    gated_value = nan(n_subjects, 6);
    for i = 1:n_subjects
        for c = 1:6
            same_condition = T.subject_id == subjects(i) & ...
                T.shape_index == condition_shape(c) & ...
                T.bandwidth_index == condition_bandwidth(c);
            chen_row = find(same_condition & T.controller_index == chen_code);
            gated_row = find(same_condition & ...
                T.controller_index == gated_code);

            % s19 has no Chen trial for the narrow ellipse / narrow paw.
            if isempty(chen_row) || isempty(gated_row)
                continue;
            end
            chen_value(i, c) = metric(chen_row);
            gated_value(i, c) = metric(gated_row);
        end
    end
    difference = gated_value - chen_value;
    chen_mean = mean(chen_value, 2, 'omitnan');
    gated_mean = mean(gated_value, 2, 'omitnan');

    figure('Name', sprintf('draft_fig_b_%s', metric_names(k)), ...
        'WindowStyle', 'docked', 'Color', 'w');
    tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    % Left: one line per participant.
    nexttile;
    hold on;
    plot([1, 2], [chen_mean, gated_mean]', 'Color', [0.6, 0.8, 1]);
    % plot MarkerSize is a width, so sqrt of the area matches the dots.
    plot(ones(n_subjects, 1), chen_mean, 'ro', 'MarkerFaceColor', 'r', ...
        'MarkerSize', sqrt(subject_dot_size));
    plot(2 * ones(n_subjects, 1), gated_mean, 'bo', 'MarkerFaceColor', 'b', ...
        'MarkerSize', sqrt(subject_dot_size));
    xlim([0.5, 2.5]);
    xticks([1, 2]);
    xticklabels({'Chen', 'Gated'});
    set(gca, 'FontSize', tick_font_size);
    ylabel(metric_labels(k), 'FontSize', label_font_size);
    grid on;
    % title({sprintf('mean of paired trials, %d participants', n_subjects), ...
    %     sprintf('lower under Gated for %d of %d', ...
    %     sum(gated_mean < chen_mean), n_subjects)});

    % Right: one dot per participant, one column per condition.
    nexttile;
    hold on;
    for c = 1:6
        subject_dots = swarmchart(c * ones(n_subjects, 1), ...
            difference(:, c), subject_dot_size, 'k', 'filled', ...
            'XJitterWidth', 0.5);
        median_line = plot(c + [-0.3, 0.3], ...
            median(difference(:, c), 'omitnan') * [1, 1], 'm', 'LineWidth', 4);
    end
    yline(0, 'k--');
    xlim([0.5, 6.5]);
    xticks(1:6);
    xticklabels(condition_labels);
    set(gca, 'FontSize', tick_font_size);
    ylabel(sprintf('Gated - Chen, %s', metric_labels(k)), ...
        'FontSize', label_font_size);
    grid on;
    legend([subject_dots, median_line], {'subject', 'median'});
    % title({'Gated - Chen per participant and condition', ...
    %     'magenta bar = median, below 0 = lower under Gated'});
end
