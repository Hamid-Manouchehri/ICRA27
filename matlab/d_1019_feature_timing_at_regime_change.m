%% Timing of the classifier features at fine-area entry and exit, Gated.
% Each logged feature (z-scored per trial, the value the controller had at
% each moment) is cut from -3 to +3 s around the moments the TCP enters or
% leaves the fine area (fine_mask_fun), averaged and smoothed per
% participant, and rescaled to 0 = level before and 1 = level after.
% Crossing time = when it first reaches 0.5; negative means it changed
% before the TCP crossed.
%
% Inputs:
%   - data/trial_manifest.csv (Gated rows)
%   - data/raw_data/test_subjects/<robot_csv>
%
% Outputs:
%   - printed crossing times and paired tests, one figure; nothing saved
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
manifest_file = fullfile(data_directory, 'trial_manifest.csv');
raw_data_directory = fullfile(data_directory, 'raw_data', 'test_subjects');

shape_names = ["fine_gross", "pear", "ellipse"];  % shapes included
bandwidth_names = ["wide", "narrow"];  % widths included

% Order matters below: 1-2 gaze, 3-4 robot, 5 classifier output.
feature_names = ["gaze_gaze_path_efficiency", "gaze_p90_gaze_ee_distance", ...
    "robot_direction_concentration", "robot_std_curvature", ...
    "fine_probability"];
feature_labels = ["path efficiency (gaze)", "p90 gaze-EE distance (gaze)", ...
    "direction concentration (robot)", "std curvature (robot)", ...
    "P_t (classifier output)"];
feature_colors = ["b", "c", "r", "m", "k"];

min_before_s = 2.0;  % shortest segment before a change [s]
min_after_s = 3.0;  % shortest segment after a change [s]
pre_window_s = [-2, -1];  % where the level before is measured [s]
post_window_s = [2, 3];  % where the level after is measured [s]
smoothing_s = 0.25;  % moving average on each participant's curve [s]

% End User Setting
%% Main Body

fprintf('reading:\n  %s\n  %s\\<robot_csv of the Gated trials>\n', ...
    manifest_file, raw_data_directory);
fprintf('writing:\n  nothing (printout and figure only)\n');

M = readtable(manifest_file, 'TextType', 'string');
M = M(M.controller == "Gated" & ismember(M.shape, shape_names) & ...
    ismember(M.bandwidth, bandwidth_names), :);
subjects = unique(M.subject);
n_subjects = numel(subjects);
n_features = numel(feature_names);
grid_step_s = 0.05;
grid_s = round(-3:grid_step_s:3, 2);
event_names = ["entry", "exit"];

% Sum and count of the event-locked traces: subject x time x feature x
% event (1 entry, 2 exit), so the mean skips missing values.
trace_sum = zeros(n_subjects, numel(grid_s), n_features, 2);
trace_count = zeros(n_subjects, numel(grid_s), n_features, 2);
n_events = zeros(n_subjects, 2);

for m = 1:height(M)
    fprintf('Reading s%d: %s\n', M.subject(m), M.robot_csv(m));
    T = readtable(fullfile(raw_data_directory, M.robot_csv(m)));
    i = find(subjects == M.subject(m));
    time_s = T.time_s;

    % Fine area by the x rule; a change is the first sample of a segment.
    fine = fine_mask_fun(T.coppeliasim_x_m, ...
        find(["fine_gross", "pear", "ellipse"] == M.shape(m)));
    change = find(diff(double(fine)) ~= 0) + 1;
    bounds = [1; change; numel(fine) + 1];
    segment_s = time_s(bounds(2:end) - 1) - time_s(bounds(1:end - 1));

    % Features z-scored within the trial, NaN before the first update.
    Z = nan(height(T), n_features);
    for k = 1:n_features
        value = T.(feature_names(k));
        value(T.classifier_update_count == 0) = NaN;
        Z(:, k) = (value - mean(value, 'omitnan')) / std(value, 'omitnan');
    end

    for c = 1:numel(change)
        if segment_s(c) < min_before_s || segment_s(c + 1) < min_after_s
            continue;
        end
        e = 2 - fine(change(c));  % into the fine area: 1, out of it: 2
        n_events(i, e) = n_events(i, e) + 1;

        % Last logged value at or before each grid time.
        Z_event = interp1(time_s, Z, time_s(change(c)) + grid_s', ...
            'previous');
        valid = ~isnan(Z_event);
        Z_event(~valid) = 0;
        trace_sum(i, :, :, e) = trace_sum(i, :, :, e) + ...
            reshape(Z_event, [1, size(Z_event)]);
        trace_count(i, :, :, e) = trace_count(i, :, :, e) + ...
            reshape(valid, [1, size(valid)]);
    end
end
mean_trace = trace_sum ./ trace_count;

% Smooth and rescale each participant's mean trace, find its 0.5 crossing.
% The search ends where the after level starts, so later crossings are NaN.
in_pre = grid_s >= pre_window_s(1) & grid_s <= pre_window_s(2);
in_post = grid_s >= post_window_s(1) & grid_s <= post_window_s(2);
in_search = find(grid_s >= pre_window_s(2) & grid_s <= post_window_s(1));
normalized = nan(size(mean_trace));
crossing_s = nan(n_subjects, n_features, 2);

for e = 1:2
    for k = 1:n_features
        for i = 1:n_subjects
            trace = movmean(mean_trace(i, :, k, e), ...
                round(smoothing_s / grid_step_s), 'omitnan');
            level_pre = mean(trace(in_pre), 'omitnan');
            level_post = mean(trace(in_post), 'omitnan');
            normalized(i, :, k, e) = (trace - level_pre) / ...
                (level_post - level_pre);

            % NaN if never reached, or already above at the search start.
            n_search = normalized(i, in_search, k, e);
            j = find(n_search >= 0.5, 1);
            if j > 1
                t1_s = grid_s(in_search(j - 1));
                t2_s = grid_s(in_search(j));
                crossing_s(i, k, e) = t1_s + (0.5 - n_search(j - 1)) * ...
                    (t2_s - t1_s) / (n_search(j) - n_search(j - 1));
            end
        end
    end
end

for e = 1:2
    fprintf('\nTCP %s of the fine area: %d events, %d participants\n', ...
        event_names(e), sum(n_events(:, e)), n_subjects);
    for k = 1:n_features
        x = crossing_s(:, k, e);
        fprintf(['%-30s crossing median %+.2f s, range %+.2f to %+.2f, ' ...
            'valid %d of %d\n'], feature_names(k), median(x, 'omitnan'), ...
            min(x), max(x), sum(~isnan(x)), n_subjects);
    end

    % Gaze minus robot crossing time, paired over participants.
    for g = 1:2
        for r = 3:4
            d = crossing_s(:, g, e) - crossing_s(:, r, e);
            d = d(~isnan(d));
            fprintf(['  %s - %s: median %+.2f s, gaze earlier in %d of ' ...
                '%d, Wilcoxon p = %.3f\n'], feature_names(g), ...
                feature_names(r), median(d), sum(d < 0), numel(d), ...
                signrank(d));
        end
    end
end

%% Plot
figure('Name', 'feature_timing_at_fine_area_change_gated', 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
for e = 1:2
    nexttile;
    hold on;
    for k = 1:n_features
        plot(grid_s, median(normalized(:, :, k, e), 1, 'omitnan'), ...
            'Color', feature_colors(k), 'LineWidth', 1.5);
    end
    xline(0, 'k-');
    yline(0.5, 'k--');
    xlim([grid_s(1), grid_s(end)]);
    xlabel(sprintf('time from TCP %s of the fine area [s]', ...
        event_names(e)));
    ylabel('feature rescaled: 0 = before, 1 = after');
    grid on;
    legend(feature_labels, 'Location', 'southeast');
    title({sprintf('%s, %d events, %d participants', event_names(e), ...
        sum(n_events(:, e)), n_subjects), ...
        sprintf(['median over participants, Gated, after level ' ...
        '%g to %g s, smoothed %g s'], post_window_s, smoothing_s)});
end
