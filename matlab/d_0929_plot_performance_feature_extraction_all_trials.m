%% SPARC per subject, Chen vs Gated, for one shape and bandwidth.
% One bar pair per subject. More negative SPARC = less smooth.
%
% Inputs:
%   - data/processed_data/performance_features_all_trials.mat
%
% Outputs:
%   - figure only, nothing written to disk
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

shape_code = 3;  % fine_gross:1, pear: 2, ellipse: 3
bandwidth_code = 1;  % wide: 1, narrow: 2
chen_code = 1;
gated_code = 2;

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n', data_mat_dir);
fprintf('writing:\n  nothing (figure only)\n');

T = read_feature_data_mat.trial_features;
subjects = unique(T.subject_id);

chen_sparc = nan(numel(subjects), 1);
gated_sparc = nan(numel(subjects), 1);
for i = 1:numel(subjects)
    same_condition = T.subject_id == subjects(i) & ...
        T.shape_index == shape_code & T.bandwidth_index == bandwidth_code;

    % mean of nothing is NaN, so a missing trial (s19 narrow Chen) just draws no bar.
    chen_sparc(i) = mean(T.sparc(same_condition & T.controller_index == chen_code));
    gated_sparc(i) = mean(T.sparc(same_condition & T.controller_index == gated_code));
end

figure('Name', 'sparc_per_subject');
bars = bar(1:numel(subjects), [chen_sparc, gated_sparc]);
bars(1).FaceColor = 'r';
bars(2).FaceColor = 'b';

xticks(1:numel(subjects));
xticklabels(compose('s%d', subjects));
ylabel('SPARC');
title(sprintf('SPARC per subject: %s, %s', ...
    read_feature_data_mat.shape_name_L{shape_code}, ...
    read_feature_data_mat.bandwidth_label_L{bandwidth_code}), 'Interpreter', 'none');
legend(["Chen", "Gated"]);
grid on;
