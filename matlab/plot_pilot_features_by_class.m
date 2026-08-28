%% Plot windowed features from extract_pilot_features_fixed_adm_2d.m,
%% grouped by trial_class_label (Fine / Gross / Natural).
%
% Loads one or more feature CSVs -- typically one file per recording
% session, where a single ur5e_fixed_adm_2d_labeled.py session can
% itself contain windows of every label, since the operator can sw/itch
% keys mid-recording -- concatenates them, and for every feature draws
% a boxplot of its distribution split by class -- the "distributions
% given class" view a classifier's own feature choice starts from.
%
% "None" (no key held) is not a fourth class, just unlabeled time --
% rest between deliberate segments, or the operator not yet having
% picked a label -- so windows labeled "None" are dropped up front and
% never appear in any figure; see the "Loaded/dropped" line this script
% prints at the command window to confirm how many that was.
%
% Inputs:
%   feature_csv_files - data/processed_data/features_fixed_adm_2d_
%       <trial>.csv, one or more, as written by extract_pilot_features_
%       fixed_adm_2d.m. Every listed file must share that script's
%       exact column names, including trial_class_label and
%       label_purity -- an older feature CSV written before that
%       script read per-sample labels has neither and will fail here
%       with "Unrecognized table variable name".
%   raw_csv_files - data/raw_data/ur5e_fixed_adm_2d_<trial>.csv, the
%       raw per-sample recording each feature_csv_files entry was
%       extracted from (same order, same length). Used only for
%       Figure 7.
%
% Output:
%   Figures 1-3: 11 robot / 6 gaze / 6 fusion WINDOW-level features
%   (tiledlayout 3x4/2x3/2x3), as boxplots. Each tile is one feature's
%   boxplot, one box per class in class_order -- the summary-statistics
%   view (median, IQR, outliers).
%   Figures 4-6: the identical 11/6/6 window-level features, as
%   histograms (same tiledlayouts). Each tile overlays one semi-
%   transparent, class-colored histogram per class in class_order,
%   normalized so each class's bars sum to 1 regardless of how many
%   windows that class has -- the distribution-shape view
%   (multimodality, skew, how much two classes actually overlap) a
%   boxplot's five numbers cannot show. With only tens of windows per
%   class, expect these to look coarse -- see Figure 7.
%   Figure 7: raw |F| and |v|, one histogram tile each, at the full
%   per-sample rate (thousands of samples per class) rather than
%   window-averaged -- see the file header note above Figure 7 for why
%   only these two features get this treatment.
%   Besides dropping "None" windows/samples outright, windows below
%   minimum_label_purity (a window whose time span crossed a label
%   change while recording) are also dropped before any of this -- see
%   User Settings. A per-class window/sample count (post-filtering) is
%   printed to the command window as a loading sanity check, alongside
%   how many windows/samples each filter dropped.
%   No file is written; this script only reads the CSVs.
%
% A numerical note worth knowing before reading the DSJ panels: DSJ
% divides by the window's own path length squared (paper Eq. 16), so a
% near-stationary window (very little distance travelled) can produce a
% spuriously huge DSJ from ordinary force/position noise, not from
% actually jerky motion. Both DSJ panels (boxplot and histogram) use a
% log axis so a handful of such windows cannot compress every other
% window onto the axis, but the values themselves are not filtered or
% corrected here -- decide separately whether near-stationary windows
% belong in classifier training data at all.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO one feature CSV per recording session. A single labeled session
% can already contain all three classes (plus unlabeled "None" time),
% so this list is for combining multiple separate sessions, not one
% file per class. Add to it as more sessions get recorded and run
% through extract_pilot_features_fixed_adm_2d.m.
feature_csv_files = { ...
    fullfile(script_directory, '..', 'data', 'processed_data', ...
        'features_fixed_adm_2d_labeled_d50_s1.csv')};

% TODO the raw per-sample robot CSV each entry of feature_csv_files was
% extracted from, same order, same length -- used only for Figure 7
% (raw |F| and |v|, at full sample rate rather than window-averaged;
% see the file header for why only these two features get this
% treatment).
raw_csv_files = { ...
    fullfile(script_directory, '..', 'data', 'raw_data', ...
        'ur5e_fixed_adm_2d_labeled_d50_s1.csv')};

% TODO the classes to show and the order to show them in, left to
% right within each box group. Must match the trial_class_label values
% actually written into your feature CSVs -- ur5e_fixed_adm_2d_
% labeled.py writes exactly 'Fine'/'Gross'/'Natural' (plus 'None',
% dropped below since it is not a class -- see file header).
class_order = {'Fine', 'Gross', 'Natural'};

% TODO one RGB color per entry in class_order, same order.
class_colors = [ ...
    0.80, 0.20, 0.20;   % Fine
    0.20, 0.50, 0.80;   % Gross
    0.30, 0.60, 0.20];  % Natural

% TODO minimum label_purity (fraction of a window's samples that
% carried the majority label, see extract_pilot_features_fixed_adm_2d.m)
% to keep a window [0, 1]. 1.0 keeps only windows the operator's key
% presses never crossed mid-window -- the cleanest exemplars of each
% class, which is what a "distribution given class" plot should show.
% Lower it to trade that purity for more windows per class.
minimum_label_purity = 1.0;

% Bin width per histogram tile (Figures 4-7) is set per feature, not
% here -- see the 4th column of robot_feature_specs /
% gaze_feature_specs / fusion_feature_specs / raw_feature_specs below,
% next to each figure's block.

%% Load and concatenate every listed feature CSV
feature_tables = cell(1, numel(feature_csv_files));
for file_idx = 1:numel(feature_csv_files)
    feature_tables{file_idx} = readtable(feature_csv_files{file_idx}, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
end
feature_table = vertcat(feature_tables{:});
loaded_window_count = height(feature_table);

% "None" is unlabeled time, not a fourth class -- drop it before doing
% anything else so it can never leak into a figure. Everything past
% this point assumes trial_class_label only ever holds a real class.
is_a_class = feature_table.trial_class_label ~= "None";
feature_table = feature_table(is_a_class, :);
excluded_none_count = loaded_window_count - height(feature_table);

is_pure_enough = feature_table.label_purity >= minimum_label_purity;
excluded_impure_count = sum(~is_pure_enough);
feature_table = feature_table(is_pure_enough, :);

class_labels = categorical(feature_table.trial_class_label, class_order);

fprintf('\nLoaded %d windows from %d file(s): %d dropped ("None"), ', ...
    loaded_window_count, numel(feature_csv_files), excluded_none_count);
fprintf('%d dropped (purity < %.2f).\n%d windows remain:\n', ...
    excluded_impure_count, minimum_label_purity, height(feature_table));
for class_idx = 1:numel(class_order)
    window_count = sum(class_labels == class_order{class_idx});
    fprintf('  %-10s %d windows\n', class_order{class_idx}, window_count);
end
fprintf('\n');

%% Load and concatenate every listed raw CSV, for Figure 7
raw_tables = cell(1, numel(raw_csv_files));
for file_idx = 1:numel(raw_csv_files)
    raw_tables{file_idx} = readtable(raw_csv_files{file_idx}, ...
        'Delimiter', ',', 'ReadVariableNames', true, ...
        'VariableNamingRule', 'preserve', 'TextType', 'string');
end
raw_table = vertcat(raw_tables{:});
raw_loaded_count = height(raw_table);

% Same "None is not a class" exclusion as feature_table above, applied
% per raw sample instead of per window -- no purity concept needed
% here since a single sample cannot itself straddle a label change.
raw_is_a_class = raw_table.label ~= "None";

raw_sample.force_mag_n = hypot( ...
    get_numeric_column(raw_table.Fx_filt_N(raw_is_a_class)), ...
    get_numeric_column(raw_table.Fy_filt_N(raw_is_a_class)));
raw_sample.speed_m_s = hypot( ...
    get_numeric_column(raw_table.tcp_vx_actual_m_s(raw_is_a_class)), ...
    get_numeric_column(raw_table.tcp_vy_actual_m_s(raw_is_a_class)));
raw_sample.label = raw_table.label(raw_is_a_class);

raw_class_labels = categorical(raw_sample.label, class_order);

fprintf('Loaded %d raw samples from %d file(s): %d dropped ("None").\n', ...
    raw_loaded_count, numel(raw_csv_files), ...
    raw_loaded_count - numel(raw_sample.label));
fprintf('%d samples remain:\n', numel(raw_sample.label));
for class_idx = 1:numel(class_order)
    sample_count = sum(raw_class_labels == class_order{class_idx});
    fprintf('  %-10s %d samples\n', class_order{class_idx}, sample_count);
end
fprintf('\n');

%% ================================================================
% Figure 1: robot features
% =================================================================

% TODO 4th column: histogram bin WIDTH for this feature, in its own
% units (Figures 4-7 only -- plot_feature_boxplots ignores it), 'auto'
% or a positive scalar, set per feature since one width rarely suits
% all of them.
robot_feature_specs = { ...
    'mean_force_n', 'Mean |F| [N]', false, 'auto'; ...
    'mean_speed_m_s', 'Mean |v| [m/s]', false, 'auto'; ...
    'mean_curvature_1_m', 'Mean \kappa [1/m]', false, 'auto'; ...
    'mean_curvature_dot_1_m_s', 'Mean d\kappa/dt [1/(m s)]', false, 'auto'; ...
    'mean_acceleration_m_s2', 'Mean |a| [m/s^2]', false, 'auto'; ...
    'mean_power_w', 'Mean F.v [W]', false, 'auto'; ...
    'median_speed_moving_m_s', 'Median |v|, moving [m/s]', false, 'auto'; ...
    'median_abs_curvature_1_m', 'Median |\kappa| [1/m]', false, 'auto'; ...
    'dsj', 'DSJ [-]', true, 'auto'; ...
    'power_law_r2', 'Power-law R^2 [-]', false, 'auto'; ...
    'curvature_sign_reversal_rate_per_m', ...
        'Sign-reversal rate [1/m]', false, 'auto'};

% figure('Name', 'Robot features by class', 'Color', 'w');
% tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
% plot_feature_boxplots(feature_table, class_labels, class_order, ...
%     class_colors, robot_feature_specs);
% sgtitle('Robot features, distribution by class');

%% ================================================================
% Figure 2: eye-gaze features
% =================================================================

gaze_feature_specs = { ...
    'fixation_duration_s', 'Fixation duration [s]', false, 'auto'; ...
    'fixation_rate_per_s', 'Fixation rate [1/s]', false, 'auto'; ...
    'd_lookahead_m', 'd_{lookahead} [m]', false, 'auto'; ...
    'fixation_transition_m', 'Fixation transition [m]', false, 'auto'; ...
    'fixation_transition_angle_deg', ...
        'Fixation transition angle [deg]', false, 'auto'; ...
    'gaze_dispersion_m', 'Gaze dispersion [m]', false, 'auto'};

% figure('Name', 'Eye-gaze features by class', 'Color', 'w');
% tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
% plot_feature_boxplots(feature_table, class_labels, class_order, ...
%     class_colors, gaze_feature_specs);
% sgtitle('Eye-gaze features, distribution by class');

%% ================================================================
% Figure 3: robot + gaze fusion features
% =================================================================

fusion_feature_specs = { ...
    'angle_gaze_v_deg', 'angle(gaze, v) [deg]', false, 'auto'; ...
    'angle_gaze_f_deg', 'angle(gaze, F) [deg]', false, 'auto'; ...
    'd_lookahead_over_v_s', 'd_{lookahead}/|v| [s]', false, 'auto'; ...
    'd_lookahead_times_curvature', ...
        'd_{lookahead} * \kappa [-]', false, 'auto'; ...
    'curvature_at_gaze_point_1_m', ...
        '\kappa at gaze point [1/m]', false, 'auto'; ...
    'fraction_gaze_ahead', 'Fraction gaze ahead [-]', false, 'auto'};

% figure('Name', 'Robot+gaze fusion features by class', 'Color', 'w');
% tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
% plot_feature_boxplots(feature_table, class_labels, class_order, ...
%     class_colors, fusion_feature_specs);
% sgtitle('Robot+gaze fusion features, distribution by class');

%% ================================================================
% Figure 4: robot features, histograms
% =================================================================

figure('Name', 'Robot features by class (histograms)', 'Color', 'w');
tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_histograms(feature_table, class_labels, class_order, ...
    class_colors, robot_feature_specs);
sgtitle('Robot features, histogram by class');

%% ================================================================
% Figure 5: eye-gaze features, histograms
% =================================================================

figure('Name', 'Eye-gaze features by class (histograms)', 'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_histograms(feature_table, class_labels, class_order, ...
    class_colors, gaze_feature_specs);
sgtitle('Eye-gaze features, histogram by class');

%% ================================================================
% Figure 6: robot + gaze fusion features, histograms
% =================================================================

figure('Name', 'Robot+gaze fusion features by class (histograms)', ...
    'Color', 'w');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_histograms(feature_table, class_labels, class_order, ...
    class_colors, fusion_feature_specs);
sgtitle('Robot+gaze fusion features, histogram by class');

%% ================================================================
% Figure 7: raw per-sample |F| and |v|, histograms
% =================================================================
% Not window-averaged -- compare against Figure 4's mean_force_n /
% mean_speed_m_s tiles, which are the SAME two underlying quantities
% but averaged within each 2 s window first. Thousands of raw samples
% per class (vs. tens of windows) give histcounts()'s auto binning
% enough data to produce a genuinely resolved shape instead of a
% handful of wide, overlapping bars.

raw_feature_specs = { ...
    'force_mag_n', 'Raw |F| [N]', false, 'auto'; ...
    'speed_m_s', 'Raw |v| [m/s]', false, 'auto'};

figure('Name', 'Raw per-sample force/speed by class (histograms)', ...
    'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
plot_feature_histograms(raw_sample, raw_class_labels, class_order, ...
    class_colors, raw_feature_specs);
sgtitle('Raw per-sample |F| and |v| by class (not window-averaged)');


%% ================================================================
%% Local functions
%% ================================================================

function plot_feature_boxplots( ...
        feature_table, class_labels, class_order, class_colors, feature_specs)
%PLOT_FEATURE_BOXPLOTS One boxplot-by-class tile per row of feature_specs.
%
% feature_specs is an N-by-3-or-4 cell array: {column_name,
% y_axis_label, use_log_y_axis[, bin_width]}. Only the first 3 columns
% are used here; a 4th (bin_width, for plot_feature_histograms) is
% ignored if present -- the same feature_specs variables are passed to
% both functions. Assumes nexttile()'s tiledlayout is already current
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


function plot_feature_histograms( ...
        feature_table, class_labels, class_order, class_colors, feature_specs)
%PLOT_FEATURE_HISTOGRAMS One overlaid-by-class histogram tile per row of
% feature_specs -- the same features and layout as
% plot_feature_boxplots(), showing distribution SHAPE instead of just
% summary statistics.
%
% feature_specs is an N-by-4 cell array: {column_name, x_axis_label,
% use_log_x_axis, bin_width} -- one row per tile, so bin_width is set
% per feature, not once for the whole figure (a width that suits one
% feature's units/spread rarely suits all of them). bin_width is
% 'auto' (defer to histcounts()' own heuristic) or a positive scalar
% giving the width of every bin directly, in that feature's own units
% (or in log10 units for a log_x_axis feature -- see below). Unlike a
% bin COUNT, a fixed WIDTH keeps a bin's meaning the same (e.g. "1 N
% wide") regardless of how many samples happen to be in the pooled
% data, which is what makes it comparable across figures that draw
% the same feature at very different sample sizes (Figure 4's window-
% level mean_force_n vs. Figure 7's raw per-sample force_mag_n, say).
% Every class within one tile shares that tile's bin edges (computed
% once from the feature's pooled data across all classes), so the
% bars stay directly comparable; each class's own bars are separately
% normalized to 'probability' (sum to 1) rather than raw counts, since
% the classes have different window/sample counts and raw counts
% would mostly just show which class has more, not how its values are
% shaped. Bars are drawn semi-transparent and overlaid rather than
% side by side, so where two classes' distributions overlap is as
% visible as where they diverge.
%
% use_log_x_axis (DSJ only, matching plot_feature_boxplots' log Y-axis
% for the same feature) bins in log10 space -- DSJ spans several orders
% of magnitude (see the file header's near-stationary-window note), so
% linear bins would put nearly every window in one bar -- then displays
% the result on a log-scaled x-axis so tick labels stay in real DSJ
% units instead of log10(DSJ). bin_width still applies here, but in
% log10 units: e.g. bin_width = 0.2 means each bin spans a factor of
% 10^0.2 (~1.6x) in real DSJ, not an additive width in DSJ itself.

    for feature_idx = 1:size(feature_specs, 1)
        column_name = feature_specs{feature_idx, 1};
        x_label = feature_specs{feature_idx, 2};
        use_log_x_axis = feature_specs{feature_idx, 3};
        bin_width = feature_specs{feature_idx, 4};

        nexttile;

        all_values = feature_table.(column_name);
        all_values = all_values(isfinite(all_values));
        if use_log_x_axis
            all_values = all_values(all_values > 0);
        end
        if isempty(all_values)
            title(x_label);
            continue;
        end

        if use_log_x_axis
            binning_values = log10(all_values);
        else
            binning_values = all_values;
        end
        if isnumeric(bin_width)
            [~, edges] = histcounts(binning_values, 'BinWidth', bin_width);
        else
            [~, edges] = histcounts(binning_values);
        end
        if use_log_x_axis
            bin_edges = 10 .^ linspace(edges(1), edges(end), numel(edges));
        else
            bin_edges = edges;
        end

        hold on;
        for class_idx = 1:numel(class_order)
            class_values = feature_table.(column_name)( ...
                class_labels == class_order{class_idx});
            class_values = class_values(isfinite(class_values));
            if use_log_x_axis
                class_values = class_values(class_values > 0);
            end
            histogram(class_values, 'BinEdges', bin_edges, ...
                'Normalization', 'probability', ...
                'FaceColor', class_colors(class_idx, :), ...
                'FaceAlpha', 0.45, 'EdgeColor', 'none');
        end
        hold off;

        grid on;
        xlabel(x_label);
        ylabel('Probability');
        title(x_label);
        if use_log_x_axis
            set(gca, 'XScale', 'log');
        end
        if feature_idx == 1
            legend(class_order, 'Location', 'best');
        end
    end
end


function numeric_values = get_numeric_column(raw_column)
%GET_NUMERIC_COLUMN Convert a table column to a numeric column vector.
% Copied verbatim from extract_pilot_features_fixed_adm_2d.m (itself
% copied from plot_raw_data_fixed_adm_2d.m).

    if iscell(raw_column) || isstring(raw_column)
        numeric_values = str2double(string(raw_column));
    else
        numeric_values = double(raw_column);
    end
    numeric_values = numeric_values(:);
end
