%% Fig. 4(b): class-conditional distributions of the four deployed features.
% One probability-normalized histogram per feature, Fine vs Gross, pooled over
% all labeled windows (train and test CSVs together).
%
% Inputs:
%   - data/processed_data/features_all_train_W_1s.csv
%   - data/processed_data/features_all_test_W_1s.csv
%
% Outputs:
%   - figure only, nothing written to disk
%
% Depends on:
%   - scripts/build_modality_feature_datasets.py (writes the input CSVs)

clc;
clear;
close all;

script_directory = fileparts(mfilename('fullpath'));
data_directory = fullfile(script_directory, '..', 'data');

%% User Setting
% Class-balanced random window split of the same labeled data (same 9
% subjects in both), so only the union is the full labeled set.
feature_csv_files = { ...
    fullfile(data_directory, 'processed_data', 'features_all_train_W_1s.csv'), ...
    fullfile(data_directory, 'processed_data', 'features_all_test_W_1s.csv')};

window_title_label = "1 s windows";  % display only, must match the CSV names

class_order = {'Fine', 'Gross'};
class_colors = [ ...
    0.80, 0.20, 0.20;   % Fine
    0.20, 0.50, 0.80];  % Gross

% {column, x label, log x axis, bin width (log10 units if log x axis)}
feature_specs = { ...
    'gaze_gaze_path_efficiency', 'Gaze path efficiency [-]', false, 0.025; ...
    'gaze_p90_gaze_ee_distance', 'P90 gaze-TCP distance [m]', true, 0.05; ...
    'robot_direction_concentration', ...
        'Direction concentration R [-]', false, 0.02; ...
    'robot_std_curvature', 'Std(\kappa) [1/m]', true, 0.1};

font_size_pt = 22;

% End User Setting
%% Main Body

fprintf('reading:\n');
fprintf('  %s\n', feature_csv_files{:});
fprintf('writing:\n  nothing (figure only)\n');

selected_columns = [{'trial_class_label'}, feature_specs(:, 1)'];
feature_tables = cell(1, numel(feature_csv_files));
for file_idx = 1:numel(feature_csv_files)
    import_options = detectImportOptions(feature_csv_files{file_idx}, ...
        'TextType', 'string');
    import_options.SelectedVariableNames = selected_columns;
    feature_tables{file_idx} = readtable(feature_csv_files{file_idx}, ...
        import_options);
end
feature_table = vertcat(feature_tables{:});

class_labels = categorical(feature_table.trial_class_label, class_order);
class_window_count = zeros(1, numel(class_order));
fprintf('%d windows:\n', height(feature_table));
for class_idx = 1:numel(class_order)
    class_window_count(class_idx) = sum(class_labels == class_order{class_idx});
    fprintf('  %-6s %d\n', class_order{class_idx}, class_window_count(class_idx));
end

figure('Name', 'deployed_features_by_class', 'Color', 'w');
tiledlayout(1, size(feature_specs, 1), 'TileSpacing', 'compact', ...
    'Padding', 'compact');
plot_feature_histograms(feature_table, class_labels, class_order, ...
    class_colors, feature_specs);
set(findall(gcf, 'Type', 'axes'), 'FontSize', font_size_pt);
% sgtitle(sprintf('Deployed features by class, %s, N = %d (%s %d, %s %d)', ...
%     window_title_label, sum(class_window_count), class_order{1}, ...
%     class_window_count(1), class_order{2}, class_window_count(2)));



%% Local functions

function plot_feature_histograms( ...
        feature_table, class_labels, class_order, class_colors, feature_specs)
%PLOT_FEATURE_HISTOGRAMS One tile per row of feature_specs, one histogram per class.
%
% feature_specs rows: {column_name, x_label, use_log_x_axis, bin_width}.
% bin_width is 'auto' or a positive scalar (log10 units if use_log_x_axis).
% Bin edges are shared by all classes of a tile; each class is normalized
% to probability, so classes of different size stay comparable.

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
        if use_log_x_axis
            set(gca, 'XScale', 'log');
        end
        if feature_idx == 1
            legend(class_order, 'Location', 'best');
        end
        ylim([0 0.14])
    end
end
