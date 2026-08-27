%% Plot windowed features from extract_pilot_features_fixed_adm_2d.m,
%% grouped by trial_class_label (Fine / Gross / Natural).
%
% Loads one or more feature CSVs (typically one file per recorded
% trial, each trial being one class), concatenates them, and for every
% feature draws a boxplot of its distribution split by class -- the
% "distributions given class" view a classifier's own feature choice
% starts from.
%
% Inputs:
%   data/raw_data/features_fixed_adm_2d_<trial>.csv, one or more, as
%   written by extract_pilot_features_fixed_adm_2d.m. Every listed file
%   must share that script's exact column names.
%
% Output:
%   Figure 1: the 11 robot features (tiledlayout 3x4, one tile unused).
%   Figure 2: the 6 eye-gaze features (tiledlayout 2x3).
%   Figure 3: the 6 robot+gaze fusion features (tiledlayout 2x3).
%   Each tile is one feature's boxplot, one box per class in
%   class_order. A per-class window count is printed to the command
%   window as a loading sanity check.
%   No file is written; this script only reads the feature CSVs.
%
% A numerical note worth knowing before reading the DSJ panel: DSJ
% divides by the window's own path length squared (paper Eq. 16), so a
% near-stationary window (very little distance travelled) can produce a
% spuriously huge DSJ from ordinary force/position noise, not from
% actually jerky motion. That panel uses a log y-axis so a handful of
% such windows cannot compress every other window onto the axis, but
% the values themselves are not filtered or corrected here -- decide
% separately whether near-stationary windows belong in classifier
% training data at all.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO one feature CSV per recorded trial. Add to this list as more
% trials (ideally covering all three classes) get recorded and run
% through extract_pilot_features_fixed_adm_2d.m.
feature_csv_files = { ...
    fullfile(script_directory, '..', 'data', 'processed_data', ...
        'features_fixed_adm_2d_fine_gross_shape_d50_s0.csv')};

% TODO the classes to show and the order to show them in, left to
% right within each box group. Must match the trial_class_label values
% actually written into your feature CSVs.
class_order = {'fine', 'gross', 'natural'};

% TODO one RGB color per entry in class_order, same order.
class_colors = [ ...
    0.80, 0.20, 0.20;   % fine
    0.20, 0.50, 0.80;   % gross
    0.30, 0.60, 0.20];  % natural

%% Load and concatenate every listed feature CSV
feature_tables = cell(1, numel(feature_csv_files));
for file_idx = 1:numel(feature_csv_files)
    feature_tables{file_idx} = readtable(feature_csv_files{file_idx}, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
end
feature_table = vertcat(feature_tables{:});

class_labels = categorical(feature_table.trial_class_label, class_order);

fprintf('\nLoaded %d windows from %d file(s):\n', ...
    height(feature_table), numel(feature_csv_files));
for class_idx = 1:numel(class_order)
    window_count = sum(class_labels == class_order{class_idx});
    fprintf('  %-10s %d windows\n', class_order{class_idx}, window_count);
end
fprintf('\n');

%% ================================================================
% Figure 1: robot features
% =================================================================

robot_feature_specs = { ...
    'mean_force_n', 'Mean |F| [N]', false; ...
    'mean_speed_m_s', 'Mean |v| [m/s]', false; ...
    'mean_curvature_1_m', 'Mean \kappa [1/m]', false; ...
    'mean_curvature_dot_1_m_s', 'Mean d\kappa/dt [1/(m s)]', false; ...
    'mean_acceleration_m_s2', 'Mean |a| [m/s^2]', false; ...
    'mean_power_w', 'Mean F.v [W]', false; ...
    'median_speed_moving_m_s', 'Median |v|, moving [m/s]', false; ...
    'median_abs_curvature_1_m', 'Median |\kappa| [1/m]', false; ...
    'dsj', 'DSJ [-]', true; ...
    'power_law_r2', 'Power-law R^2 [-]', false; ...
    'curvature_sign_reversal_rate_per_m', ...
        'Sign-reversal rate [1/m]', false};

figure('Name', 'Robot features by class', 'Color', 'w');
tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_boxplots(feature_table, class_labels, class_order, ...
    class_colors, robot_feature_specs);
sgtitle('Robot features, distribution by class');

%% ================================================================
% Figure 2: eye-gaze features
% =================================================================

gaze_feature_specs = { ...
    'fixation_duration_s', 'Fixation duration [s]', false; ...
    'fixation_rate_per_s', 'Fixation rate [1/s]', false; ...
    'd_lookahead_m', 'd_{lookahead} [m]', false; ...
    'fixation_transition_m', 'Fixation transition [m]', false; ...
    'fixation_transition_angle_deg', ...
        'Fixation transition angle [deg]', false; ...
    'gaze_dispersion_m', 'Gaze dispersion [m]', false};

figure('Name', 'Eye-gaze features by class', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_boxplots(feature_table, class_labels, class_order, ...
    class_colors, gaze_feature_specs);
sgtitle('Eye-gaze features, distribution by class');

%% ================================================================
% Figure 3: robot + gaze fusion features
% =================================================================

fusion_feature_specs = { ...
    'angle_gaze_v_deg', 'angle(gaze, v) [deg]', false; ...
    'angle_gaze_f_deg', 'angle(gaze, F) [deg]', false; ...
    'd_lookahead_over_v_s', 'd_{lookahead}/|v| [s]', false; ...
    'd_lookahead_times_curvature', 'd_{lookahead} * \kappa [-]', false; ...
    'curvature_at_gaze_point_1_m', '\kappa at gaze point [1/m]', false; ...
    'fraction_gaze_ahead', 'Fraction gaze ahead [-]', false};

figure('Name', 'Robot+gaze fusion features by class', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_boxplots(feature_table, class_labels, class_order, ...
    class_colors, fusion_feature_specs);
sgtitle('Robot+gaze fusion features, distribution by class');


%% ================================================================
%% Local functions
%% ================================================================

function plot_feature_boxplots( ...
        feature_table, class_labels, class_order, class_colors, feature_specs)
%PLOT_FEATURE_BOXPLOTS One boxplot-by-class tile per row of feature_specs.
%
% feature_specs is an N-by-3 cell array: {column_name, y_axis_label,
% use_log_y_axis}. Assumes nexttile()'s tiledlayout is already current
% and has enough tiles for size(feature_specs, 1).

    for feature_idx = 1:size(feature_specs, 1)
        column_name = feature_specs{feature_idx, 1};
        y_label = feature_specs{feature_idx, 2};
        use_log_y_axis = feature_specs{feature_idx, 3};

        nexttile;
        boxplot(feature_table.(column_name), class_labels, ...
            'Colors', class_colors, 'GroupOrder', class_order);
        grid on;
        ylabel(y_label);
        title(y_label);

        if use_log_y_axis
            set(gca, 'YScale', 'log');
        end
    end
end
