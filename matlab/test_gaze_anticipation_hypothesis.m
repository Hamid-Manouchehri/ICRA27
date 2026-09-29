clc; 
clear; 
close all;

script_directory = fileparts(mfilename('fullpath'));
addpath(fullfile(script_directory, 'functions'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
subject_id = 15;
shape_name = "fine_gross";
bandwidth_label = "wide";
direction_label = "clockwise";

robot_csv = fullfile(data_directory, 'processed_data', sprintf( ...
    'ur5e_chen_var_adm_2d_live_labeled_%s_%s_shape_%s_s%d.csv', ...
    bandwidth_label, shape_name, direction_label, subject_id));
gaze_csv = fullfile(data_directory, 'processed_data', sprintf( ...
    'tobii_chen_var_adm_2d_live_labeled_%s_%s_shape_%s_s%d.csv', ...
    bandwidth_label, shape_name, direction_label, subject_id));
calib_json = fullfile(data_directory, 'calibrations', sprintf( ...
    'tobii_coppelia_reference_calibration_%s_%s_shape_%s_s%d.json', ...
    bandwidth_label, shape_name, direction_label, subject_id));
predictions_csv = fullfile(data_directory, 'processed_data', sprintf( ...
    'logistic_regression_held_out_%s_shape_and_s%d_1win_ahead_W_1s_predictions.csv', ...
    shape_name, subject_id));

match_tolerance_s = 2.0;  % window around a true transition to look for a detection
debounce_s = 0.3;         % a threshold crossing must persist this long to count (prevent noise/jittery sample)
lag_sweep_s = 2.0;        % +/- range of shifts to test for the cross-correlation lag check [s]
lag_step_s = 0.05;        % step size for that sweep [s]

% End User Setting
%% Main Body

T = readtable(robot_csv, 'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
G = readtable(gaze_csv, 'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');
calib = jsondecode(fileread(calib_json));
P = readtable(predictions_csv, 'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

t = T.time_s;
x = T.coppeliasim_x_m;
y = T.coppeliasim_y_m;
vx = T.tcp_vx_actual_m_s;
vy = T.tcp_vy_actual_m_s;
lbl = T.label;

%% Ground truth transitions from the human live label
change_idx = find(lbl(2:end) ~= lbl(1:end-1)) + 1;  % moment when transition happens
true_times = t(change_idx);
true_dirs = repmat({'out'}, length(change_idx), 1);
true_dirs(lbl(change_idx) == "Fine") = {'in'};  % gross -> fine: "in", fine -> gross: "out"

%% Gaze rule: instantaneous look-ahead along TCP velocity direction
[~, raw_gaze_xy_m, ~, raw_gaze_time_s] = detect_fixations_coppelia_xy( ...
    G, double(calib.manual_tobii_reference_xy_mm(:)'), ...
    double(calib.manual_coppelia_reference_xy_m(:)'), ...
    1.0, [1.0, 1.0], 20.0, 0.060, true);

% gaze and robot have different sampling rate: at every robot timestamp t_i
% use the most recent available gaze position
gaze_x = interp1(raw_gaze_time_s, raw_gaze_xy_m(:,1), t, 'previous');
gaze_y = interp1(raw_gaze_time_s, raw_gaze_xy_m(:,2), t, 'previous');

speed = hypot(vx, vy);
dir_x = vx ./ speed;
dir_y = vy ./ speed;
gaze_lead_m = (gaze_x - x) .* dir_x + (gaze_y - y) .* dir_y;  % L(t) = (g - p).v
% L > 0 : positive Lookahead, gaze is spatially ahead in current TCP
% movement direction

%% Classifier signal: this trial's own held-out predictions
Ptrial = P(P.subject == subject_id & P.bandwidth == bandwidth_label & ...
    P.direction == direction_label & P.shape == shape_name, :);
Ptrial = sortrows(Ptrial, 'window_center_s');

%% Detect threshold-crossing events on each signal
[gaze_events_t, gaze_events_d] = detect_crossings(t, gaze_lead_m, median(gaze_lead_m, 'omitnan'), debounce_s);
[clf_events_t, clf_events_d] = detect_crossings(Ptrial.window_center_s, Ptrial.probability_Fine, 0.5, debounce_s);

fprintf('gaze events: %d, clf events: %d (Ptrial rows: %d)\n', ...
    numel(gaze_events_t), numel(clf_events_t), height(Ptrial));

%% Match each method's events to the true transitions -> signed lead time
gaze_lead_s = match_events(true_times, true_dirs, gaze_events_t, gaze_events_d, match_tolerance_s);
clf_lead_s  = match_events(true_times, true_dirs, clf_events_t, clf_events_d, match_tolerance_s);

fprintf('%d true transitions\n', length(true_times));
fprintf('%-12s median lead %.3f s, missed %d/%d\n', 'gaze rule', ...
    median(gaze_lead_s, 'omitnan'), sum(isnan(gaze_lead_s)), length(true_times));
fprintf('%-12s median lead %.3f s, missed %d/%d\n', 'classifier', ...
    median(clf_lead_s, 'omitnan'), sum(isnan(clf_lead_s)), length(true_times));

%% Cross-correlation check: does gaze's PAST value predict today's label?
% Shift gaze_lead_m by candidate lags and see which shift correlates best
% with the true label. circshift(v, n), n > 0, moves values to LATER
% indices, so shifted_gaze(i) holds gaze_lead_m from n samples earlier.
% Best correlation at a POSITIVE lag means "gaze's past matches today's
% label" fits best -> gaze LEADS. Negative -> gaze LAGS.
lags_s = -lag_sweep_s:lag_step_s:lag_sweep_s;
lag_samples = round(lags_s / median(diff(t)));
true_signal = double(lbl == "Fine");
corr_vals = nan(size(lags_s));

for k = 1:length(lag_samples)
    shifted_gaze = circshift(gaze_lead_m, lag_samples(k));
    valid = ~isnan(shifted_gaze);
    c = corrcoef(shifted_gaze(valid), true_signal(valid));
    corr_vals(k) = c(1,2);
end

[best_corr, best_idx] = max(corr_vals);
best_lag_s = lags_s(best_idx);
fprintf('gaze leads label by %.3f s (correlation %.3f)\n', best_lag_s, best_corr);

figure;
plot(lags_s, corr_vals); hold on;
xline(best_lag_s, '--', sprintf('best lag = %.3f s', best_lag_s));
xline(0, '-.');
xlabel('lag [s]  (+ = gaze shifted to match FUTURE label)');
ylabel('correlation(shifted gaze lead, true label)');

%% Plot: signals over time with true label, plus lead-time comparison
figure;
subplot(3,1,1);
area(t, double(lbl == "Fine")); ylabel('true label (Fine=1)');
subplot(3,1,2);
plot(t, gaze_lead_m); hold on;
% yline(median(gaze_lead_m, 'omitnan'), '--', 'median');
yline(mean(gaze_lead_m, 'omitnan'), ':', sprintf('mean = %.4f', mean(gaze_lead_m, 'omitnan')));
ylabel('gaze lead [m]');
yline(0, '-.')
subplot(3,1,3);
plot(Ptrial.window_center_s, Ptrial.probability_Fine); hold on; yline(0.5, '--');
ylabel('P(Fine)'); xlabel('time [s]');

figure;
boxplot([gaze_lead_s, clf_lead_s], {'gaze rule', 'classifier'});
ylabel('lead time [s]  (+ = anticipated true transition)');

fprintf("percentage of gaze spatially preceding tcp: %.3f \n", ...
    sum(gaze_lead_m > 0) / sum(~isnan(gaze_lead_m)) * 100)

function [event_times, event_dirs] = detect_crossings(t, s, thresh, debounce_s)
    above = s > thresh;
    change_idx = find(diff(above) ~= 0) + 1;
    event_times = [];
    event_dirs = {};
    for k = 1:length(change_idx)
        idx = change_idx(k);
        if k < length(change_idx)
            persist_end = change_idx(k+1) - 1;
        else
            persist_end = length(t);
        end
        if t(persist_end) - t(idx) >= debounce_s
            event_times(end+1) = t(idx); %#ok<AGROW>
            if above(idx)
                event_dirs{end+1} = 'in'; %#ok<AGROW>
            else
                event_dirs{end+1} = 'out'; %#ok<AGROW>
            end
        end
    end
end

function lead_s = match_events(true_times, true_dirs, event_times, event_dirs, tol_s)
    lead_s = nan(length(true_times), 1);
    for i = 1:length(true_times)
        same_dir = strcmp(event_dirs, true_dirs{i});
        candidates = event_times(same_dir);
        candidates = candidates(abs(candidates - true_times(i)) <= tol_s);
        if ~isempty(candidates)
            [~, closest] = min(abs(candidates - true_times(i)));
            lead_s(i) = true_times(i) - candidates(closest);
        end
    end
end
