%% Performance features per subject, Chen vs Gated, one figure per metric.
% Each figure has one tile per condition (rows: bandwidth, columns: shape),
% one bar pair per subject.
%
% Inputs:
%   - data/processed_data/performance_features_all_trials.mat
%
% Outputs:
%   - figures only, nothing written to disk
%
% Depends on:
%   - d_0929_performance_feature_extraction_all_trials.m (writes the input .mat)

clc;
clear;
close all;

%% User Setting
data_mat_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', ...
    "processed_data", "performance_features_all_trials.mat");
read_feature_data_mat = load(data_mat_dir);

% One figure per metric (column of trial_features).
metric_names = ["sparc", "sparc_fine", "dsj", "dsj_fine", "mean_force_n", ...
    "path_efficiency", "traveled_path_m", "n_boundary_violations", ...
    "angular_divergence_deg", "angular_divergence_sq_deg2", ...
    "positive_progress", "total_progress_m", ...
    "horizon_angle_deg", "horizon_angle_sq_deg2"];
metric_ylabels = ["SPARC (more negative = less smooth)", ...
    "SPARC in the fine area", ...
    "DSJ (higher = less smooth)", ...
    "DSJ in the fine area", ...
    "mean force [N]", ...
    "path efficiency", ...
    "traveled path [m]", ...
    "times out of band and back", ...
    "angular divergence: mean abs(theta) [deg]", ...
    "angular divergence: mean theta^2 [deg^2]", ...
    "positive progress (ratio)", ...
    "total progress [m]", ...
    "horizon angle: mean abs(psi) [deg]", ...
    "horizon angle: mean psi^2 [deg^2]"];
log_axis = [false false true true false false false false false false false false false false];  % log y axis for DSJ
zoom_axis = [false false false false false false false false false false true false true true];  % y axis fitted to the data, not from 0

chen_code = 1;
gated_code = 2;

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', data_mat_dir);
fprintf('writing:\n  nothing (figures only)\n');

T = read_feature_data_mat.trial_features;
subjects = unique(T.subject_id);
shape_names = read_feature_data_mat.shape_name_L;  % fine_gross:1, pear: 2, ellipse: 3
bandwidth_names = read_feature_data_mat.bandwidth_label_L;  % wide: 1, narrow: 2

for m = 1:numel(metric_names)
    figure('Name', metric_names(m));
    tiledlayout(numel(bandwidth_names), numel(shape_names));

    for bandwidth_code = 1:numel(bandwidth_names)
        for shape_code = 1:numel(shape_names)
            chen_values = nan(numel(subjects), 1);
            gated_values = nan(numel(subjects), 1);
            for i = 1:numel(subjects)
                same_condition = T.subject_id == subjects(i) & ...
                    T.shape_index == shape_code & T.bandwidth_index == bandwidth_code;

                % mean of nothing is NaN, so a missing trial (s19 narrow Chen) draws no bar.
                chen_values(i) = mean(T.(metric_names(m))(same_condition & T.controller_index == chen_code));
                gated_values(i) = mean(T.(metric_names(m))(same_condition & T.controller_index == gated_code));
            end

            nexttile;
            bars = bar(1:numel(subjects), [chen_values, gated_values]);
            bars(1).FaceColor = 'r';
            bars(2).FaceColor = 'b';

            xticks(1:numel(subjects));
            xticklabels(compose('s%d', subjects));
            ylabel(metric_ylabels(m));
            title(sprintf('%s, %s', shape_names{shape_code}, bandwidth_names{bandwidth_code}), ...
                'Interpreter', 'none');
            if log_axis(m)
                set(gca, 'YScale', 'log');
            end
            if zoom_axis(m)
                all_values = [chen_values; gated_values];
                margin = 0.2 * (max(all_values) - min(all_values));
                ylim([min(all_values) - margin, max(all_values) + margin]);
            end
            legend(["Chen", "Gated"]);
            grid on;
        end
    end
end
