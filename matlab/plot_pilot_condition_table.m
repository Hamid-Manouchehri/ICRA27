%% Plot the 18-condition pilot summary built by build_pilot_condition_table.m.
%
% Input:
%   data/processed_data/pilot_condition_table.csv
%     One row per band x shape x damping condition (see
%     build_pilot_condition_table.m for column definitions).
%
% Output:
%   Figure 1: gaze validity QC bar chart.
%   Figure 2: damping manipulation check (mean force, impulse, movement
%             time vs damping), one line per shape x band block.
%   Figure 3: kinematics-vs-effort compensation plot (mean speed and
%             mean force vs damping, averaged across blocks).
%   Figure 4: speed-curvature power-law exponent and fit quality vs
%             damping, one line per block.
%   Figure 5: fixation density per curvature tertile, one line per
%             damping level.
%   Figure 6: pairwise correlation matrix of condition-level metrics.
%   Console: monotonic-in-damping block counts for the manipulation-
%             check metrics (sign-test-style direction consistency).
%   No file is written; this script only reads the condition table.
%
% Not included (not present in the condition table):
%   A log-log speed-vs-curvature scatter per condition needs the raw
%   per-sample (speed, kappa) arrays, which the condition table does not
%   store (one row per condition, by design). Gaze lead/lag vs damping
%   needs the fixation-to-TCP-path timing analysis, which
%   build_pilot_condition_table.m deliberately deferred (see its
%   header). Both need a separate script operating on the raw CSVs.
%
% Assumptions:
%   n = 1 trial per condition. Figures show all 18 points and per-block
%   trends; no inferential statistics (t-tests, ANOVA) are computed or
%   implied. The 6 shape x band blocks are treated as quasi-replicates
%   only for the printed monotonic-count summary.

clear;
close all;
clc;

script_directory = fileparts(mfilename('fullpath'));

%% User Settings
% TODO condition-table CSV to plot [file path].
read_condition_table_csv_path = fullfile( ...
    script_directory, '..', 'data', 'processed_data', ...
    'pilot_condition_table.csv');

% TODO must match build_pilot_condition_table.m's experiment grid.
bands = ["narrow", "wide"];
shapes = ["circle", "curved", "edge"];
dampings = [10, 50, 100];

% TODO must match build_pilot_condition_table.m's curvature_bin_count.
curvature_bin_labels = ["Low \kappa", "Mid \kappa", "High \kappa"];

% TODO gaze validity rejection guideline for Figure 1 [%], 0 to 100.
gaze_valid_pct_rejection_threshold = 70.0;

%% Load the condition table
data = readtable(read_condition_table_csv_path, ...
    'Delimiter', ',', 'ReadVariableNames', true, ...
    'VariableNamingRule', 'preserve', 'TextType', 'string');

if height(data) == 0
    error('Condition table is empty: %s', read_condition_table_csv_path);
end

block_colors = lines(numel(bands) * numel(shapes));
damping_colors = lines(numel(dampings));

%% ================================================================
% Figure 1: Gaze validity QC
% =================================================================

sorted_data = sortrows(data, {'band', 'shape', 'damping_n_s_m'});

figure('Name', 'Gaze validity QC', 'Color', 'w');

bar(categorical(sorted_data.trial_id, sorted_data.trial_id), ...
    sorted_data.gaze_valid_pct, 'FaceColor', [0.30, 0.55, 0.80]);

hold on;
yline(gaze_valid_pct_rejection_threshold, 'r--', ...
    sprintf('%.0f%% guide', gaze_valid_pct_rejection_threshold));

grid on;
ylabel('Valid gaze samples [%]');
title('Gaze validity per condition');
xtickangle(60);

%% ================================================================
% Figure 2: Damping manipulation check
% =================================================================

figure('Name', 'Manipulation check: effect of damping', 'Color', 'w');
tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

manipulation_metrics = { ...
    'mean_force_n', 'F_{mean} [N]'; ...
    'impulse_n_s', 'Force impulse [N s]'; ...
    'movement_time_s', 'Movement time [s]'};

fprintf('--- Direction consistency: metric vs damping, per block ---\n');

for panel_idx = 1:size(manipulation_metrics, 1)
    metric_name = manipulation_metrics{panel_idx, 1};
    metric_label = manipulation_metrics{panel_idx, 2};

    [block_labels, value_matrix] = pivot_metric_by_block( ...
        data, bands, shapes, dampings, metric_name);

    nexttile;
    hold on;
    for block_idx = 1:numel(block_labels)
        plot(dampings, value_matrix(:, block_idx)', '-o', ...
            'Color', block_colors(block_idx, :), ...
            'LineWidth', 1.3, ...
            'DisplayName', strrep(block_labels(block_idx), '_', ' '));
    end
    grid on;
    xlabel('Damping B [N s/m]');
    ylabel(metric_label);
    if panel_idx == 1
        legend('Location', 'best', 'FontSize', 7);
    end

    print_direction_consistency(value_matrix, metric_name);
end
sgtitle('Damping manipulation check (one line per shape x band block)');

%% ================================================================
% Figure 3: Kinematics-vs-effort compensation
% =================================================================

[~, speed_matrix] = pivot_metric_by_block( ...
    data, bands, shapes, dampings, 'mean_speed_m_s');
[~, force_matrix] = pivot_metric_by_block( ...
    data, bands, shapes, dampings, 'mean_force_n');

speed_mean = mean(speed_matrix, 2, 'omitnan')';
speed_std = std(speed_matrix, 0, 2, 'omitnan')';
force_mean = mean(force_matrix, 2, 'omitnan')';
force_std = std(force_matrix, 0, 2, 'omitnan')';

figure('Name', 'Kinematics vs effort compensation', 'Color', 'w');

yyaxis left;
errorbar(dampings, speed_mean, speed_std, '-o', 'LineWidth', 1.5);
ylabel('Mean TCP speed [m/s]');

yyaxis right;
errorbar(dampings, force_mean, force_std, '-s', 'LineWidth', 1.5);
ylabel('Mean force [N]');

grid on;
xlim([0, 110]);
xlabel('Damping B [N s/m]');
title({'Speed vs. effort as damping increases', ...
    'Mean \pm SD across the 6 shape x band blocks'});

%% ================================================================
% Figure 4: Speed-curvature power law vs damping
% =================================================================

[block_labels, beta_matrix] = pivot_metric_by_block( ...
    data, bands, shapes, dampings, 'power_law_beta');
[~, r2_matrix] = pivot_metric_by_block( ...
    data, bands, shapes, dampings, 'power_law_r2');

figure('Name', 'Speed-curvature power law vs damping', 'Color', 'w');
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
for block_idx = 1:numel(block_labels)
    plot(dampings, beta_matrix(:, block_idx)', '-o', ...
        'Color', block_colors(block_idx, :), 'LineWidth', 1.3, ...
        'DisplayName', strrep(block_labels(block_idx), '_', ' '));
end
yline(1.0 / 3.0, 'k--', 'Two-thirds power law (\beta = 1/3)');
grid on;
xlabel('Damping B [N s/m]');
ylabel('\beta   (v = k \cdot \kappa^{-\beta})');
title('Power-law exponent');
legend('Location', 'best', 'FontSize', 7);

nexttile;
hold on;
for block_idx = 1:numel(block_labels)
    plot(dampings, r2_matrix(:, block_idx)', '-o', ...
        'Color', block_colors(block_idx, :), 'LineWidth', 1.3);
end
grid on;
xlabel('Damping B [N s/m]');
ylabel('R^2 of log-log fit');
title('Power-law fit quality');

print_direction_consistency(beta_matrix, 'power_law_beta');
print_direction_consistency(r2_matrix, 'power_law_r2');

%% ================================================================
% Figure 5: Fixation density per curvature tertile, by damping
% =================================================================

bin_columns = {'fixation_density_curvature_bin1_hz', ...
    'fixation_density_curvature_bin2_hz', ...
    'fixation_density_curvature_bin3_hz'};

if all(ismember(bin_columns, data.Properties.VariableNames))
    figure('Name', 'Fixation density vs curvature', 'Color', 'w');
    hold on;

    for damping_idx = 1:numel(dampings)
        row_mask = abs(data.damping_n_s_m - dampings(damping_idx)) < 0.5;

        density_mean = NaN(1, numel(bin_columns));
        density_std = NaN(1, numel(bin_columns));
        for bin_idx = 1:numel(bin_columns)
            bin_values = data.(bin_columns{bin_idx})(row_mask);
            density_mean(bin_idx) = mean(bin_values, 'omitnan');
            density_std(bin_idx) = std(bin_values, 0, 'omitnan');
        end

        errorbar(1:numel(bin_columns), density_mean, density_std, '-o', ...
            'Color', damping_colors(damping_idx, :), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('B = %d N s/m', dampings(damping_idx)));
    end

    grid on;
    xlim([0.5, numel(bin_columns) + 0.5]);
    xticks(1:numel(bin_columns));
    xticklabels(curvature_bin_labels);
    ylabel('Fixation density [Hz]  (count / time spent in bin)');
    title('Fixation concentration by path curvature, per damping level');
    legend('Location', 'best');
else
    warning(['Skipping Figure 5: fixation_density_curvature_bin*_hz ' ...
        'columns not found in the condition table.']);
end

%% ================================================================
% Figure 6: Condition-level correlation matrix
% =================================================================

band_is_narrow = double(data.band == "narrow");

correlation_variable_names = {'damping_n_s_m', 'band_is_narrow', ...
    'movement_time_s', 'path_length_m', 'mean_speed_m_s', ...
    'mean_force_n', 'impulse_n_s', 'gaze_valid_pct', 'fixation_count', ...
    'fixation_rate_hz', 'mean_fixation_duration_s', ...
    'median_curvature_1_m', 'power_law_beta', 'power_law_r2'};

correlation_matrix_data = NaN(height(data), numel(correlation_variable_names));
for variable_idx = 1:numel(correlation_variable_names)
    variable_name = correlation_variable_names{variable_idx};
    if strcmp(variable_name, 'band_is_narrow')
        correlation_matrix_data(:, variable_idx) = band_is_narrow;
    else
        correlation_matrix_data(:, variable_idx) = data.(variable_name);
    end
end

correlation_matrix = corrcoef(correlation_matrix_data, 'Rows', 'pairwise');
variable_count = numel(correlation_variable_names);

figure('Name', 'Condition-level correlation matrix', 'Color', 'w');

imagesc(correlation_matrix, [-1, 1]);
colormap(build_diverging_colormap(256));
colorbar;
axis square;
set(gca, ...
    'XTick', 1:variable_count, ...
    'XTickLabel', strrep(correlation_variable_names, '_', ' '), ...
    'XTickLabelRotation', 60, ...
    'YTick', 1:variable_count, ...
    'YTickLabel', strrep(correlation_variable_names, '_', ' '));
title('Pairwise Pearson correlation (n = 18 conditions; descriptive only)');

for row_idx = 1:variable_count
    for col_idx = 1:variable_count
        text_color = [0, 0, 0];
        if abs(correlation_matrix(row_idx, col_idx)) > 0.6
            text_color = [1, 1, 1];
        end
        text(col_idx, row_idx, ...
            sprintf('%.2f', correlation_matrix(row_idx, col_idx)), ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 7, ...
            'Color', text_color);
    end
end


%% ================================================================
%% Local functions
%% ================================================================

function [block_labels, value_matrix] = pivot_metric_by_block( ...
        data, bands, shapes, dampings, metric_name)
%PIVOT_METRIC_BY_BLOCK Arrange one metric as [damping x block].
%
% block_labels is band + "_" + shape for each of the numel(bands) x
% numel(shapes) blocks, in nested band-then-shape order. value_matrix
% is numel(dampings)-by-numel(block_labels); NaN where a condition is
% missing from the table.

    block_labels = strings(1, numel(bands) * numel(shapes));
    value_matrix = NaN(numel(dampings), numel(block_labels));

    block_idx = 0;
    for band = bands
        for shape = shapes
            block_idx = block_idx + 1;
            block_labels(block_idx) = band + "_" + shape;

            for damping_idx = 1:numel(dampings)
                row_mask = data.band == band & data.shape == shape & ...
                    abs(data.damping_n_s_m - dampings(damping_idx)) < 0.5;
                matched_rows = find(row_mask, 1);
                if ~isempty(matched_rows)
                    value_matrix(damping_idx, block_idx) = ...
                        data.(metric_name)(matched_rows);
                end
            end
        end
    end
end


function print_direction_consistency(value_matrix, metric_label)
%PRINT_DIRECTION_CONSISTENCY Count blocks monotonic in damping.
%
% Treats each column (shape x band block) as a quasi-replicate. Sign-
% test framing for n = 6: 6/6 or 0/6 ~ p = 0.03 (two-sided), 5/6 or 1/6
% ~ p = 0.11. Not a substitute for a properly powered study.

    block_count = size(value_matrix, 2);
    monotonic_up_count = 0;
    monotonic_down_count = 0;

    for block_idx = 1:block_count
        values = value_matrix(:, block_idx);
        if any(isnan(values))
            continue;
        end
        if all(diff(values) > 0)
            monotonic_up_count = monotonic_up_count + 1;
        elseif all(diff(values) < 0)
            monotonic_down_count = monotonic_down_count + 1;
        end
    end

    fprintf('%-22s monotonic-up %d/%d, monotonic-down %d/%d\n', ...
        metric_label, monotonic_up_count, block_count, ...
        monotonic_down_count, block_count);
end


function color_map = build_diverging_colormap(level_count)
%BUILD_DIVERGING_COLORMAP Blue-white-red colormap without a toolbox.

    half_count = floor(level_count / 2);
    negative_half = [ ...
        linspace(0.10, 1.0, half_count)', ...
        linspace(0.30, 1.0, half_count)', ...
        ones(half_count, 1)];
    positive_half = [ ...
        ones(half_count, 1), ...
        linspace(1.0, 0.30, half_count)', ...
        linspace(1.0, 0.10, half_count)'];
    color_map = [negative_half; positive_half];
end
