%% LOSO accuracy by feature set, and LOSHO accuracy by held-out shape.
%
% Panel A: one column per feature set, individual per-subject dots plus
% mean +/- SD, and a dashed line at the mean majority baseline. Panel B:
% grouped bars per held-out shape (robot vs all features), with a tick
% at that shape's own majority baseline (shapes are imbalanced, so the
% baseline differs by shape and cannot be one flat line).
%
% Inputs:
%   data/processed_data/paper_fig3_accuracy_W_2s.csv -- one row per
%     (scheme, feature_set, fold_id): scheme is LOSO or LOSHO, fold_id
%     is a subject id under LOSO or a shape name under LOSHO.
% Output:
%   data/images/paper_fig3_accuracy_W_2s.{png,pdf}
% Depends on:
%   matlab/paper_style.m
%
% Requires MATLAB R2020b or newer (exportgraphics, Bar XEndPoints, the
% '_' scatter marker). Not run in this session -- no MATLAB available
% here; reasoned through manually per this project's convention for
% .m files. Report the first error back if one occurs.

% -------------------- User Settings --------------------
dataset_name = "W_2s";  % TODO must match build_paper_results.py
loso_feature_sets_to_show = ["robot", "gaze", "gaze_pure", "joint", ...
    "robot_plus_gaze", "robot_plus_gaze_top20"];  % TODO subset/reorder
losho_feature_sets_to_show = ["robot", "all"];    % TODO only these exist
losho_shape_order = ["ellipse", "fine_gross", "pear"];  % TODO x-axis order
y_axis_limits_pct = [0, 100];  % TODO full range, not truncated
output_formats = ["png", "pdf"];
% ---------------------------------------------------------

project_root = fileparts(fileparts(mfilename('fullpath')));
read_csv_path = fullfile(project_root, "data", "processed_data", ...
    "paper_fig3_accuracy_" + dataset_name + ".csv");
write_image_path = fullfile(project_root, "data", "images", ...
    "paper_fig3_accuracy_" + dataset_name);

fprintf("paper_fig3_accuracy.m\n");
fprintf("  reading:  %s\n", read_csv_path);
for fmt = output_formats
    fprintf("  writing:  %s.%s\n", write_image_path, fmt);
end

% TextType 'string' is required: without it, text columns come back as
% cell arrays of char, and "== name" below would error on a cell array.
data = readtable(read_csv_path, 'Delimiter', ',', 'TextType', 'string');
colors = paper_style();

figure('Name', 'LOSO and LOSHO accuracy', 'Position', [100 100 760 340]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% ---- Panel A: LOSO, per-subject dots + mean +/- SD ----
nexttile; hold on;
loso = data(data.scheme == "LOSO", :);
% Majority baseline depends only on each subject's test-fold label
% distribution, not on the feature set, so one mean over every LOSO row
% is the correct reference line regardless of which sets are plotted.
baseline_pct = mean(loso.majority_baseline);
n_sets = numel(loso_feature_sets_to_show);
for i = 1:n_sets
    name = loso_feature_sets_to_show(i);
    rows = loso(loso.feature_set == name, :);
    if isempty(rows)
        warning("feature set '%s' not found in LOSO rows", name);
        continue
    end
    color = colors.(char(name));
    jitter = (rand(height(rows), 1) - 0.5) * 0.25;
    scatter(i + jitter, rows.accuracy, 18, color, 'filled', ...
        'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
    errorbar(i, mean(rows.accuracy), std(rows.accuracy), 'o', ...
        'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.5, ...
        'CapSize', 6, 'HandleVisibility', 'off');
end
yline(baseline_pct, '--', 'Color', colors.baseline, ...
    'Label', 'majority baseline', 'LabelHorizontalAlignment', 'left', ...
    'FontSize', 7);
xlim([0.5, n_sets + 0.5]);
ylim(y_axis_limits_pct);
xticks(1:n_sets);
xticklabels(strrep(cellstr(loso_feature_sets_to_show), '_', ' '));
xtickangle(30);
ylabel('LOSO accuracy (%)');
title('A: leave-one-subject-out');
box on;

% ---- Panel B: LOSHO, grouped bars per shape ----
nexttile; hold on;
losho = data(data.scheme == "LOSHO", :);
n_shapes = numel(losho_shape_order);
n_losho_sets = numel(losho_feature_sets_to_show);
bar_values = nan(n_shapes, n_losho_sets);
baseline_values = nan(n_shapes, 1);
for s = 1:n_shapes
    shape_rows = losho(losho.fold_id == losho_shape_order(s), :);
    baseline_values(s) = mean(shape_rows.majority_baseline);
    for f = 1:n_losho_sets
        row = shape_rows(shape_rows.feature_set == ...
            losho_feature_sets_to_show(f), :);
        if ~isempty(row)
            bar_values(s, f) = row.accuracy(1);
        end
    end
end
bar_handle = bar(bar_values, 'grouped');
for f = 1:n_losho_sets
    bar_handle(f).FaceColor = colors.(char(losho_feature_sets_to_show(f)));
end
for f = 1:n_losho_sets
    x_positions = bar_handle(f).XEndPoints;
    scatter(x_positions, baseline_values, 40, colors.baseline, ...
        '_', 'LineWidth', 1.5, 'HandleVisibility', 'off');
end
xticks(1:n_shapes);
xticklabels(strrep(cellstr(losho_shape_order), '_', ' '));
ylim(y_axis_limits_pct);
ylabel('LOSHO accuracy (%)');
legend(strrep(cellstr(losho_feature_sets_to_show), '_', ' '), ...
    'Location', 'southoutside', 'Orientation', 'horizontal', ...
    'FontSize', 7, 'Box', 'off');
title('B: leave-one-shape-out (dash = per-shape baseline)');
box on;

for fmt = output_formats
    exportgraphics(gcf, write_image_path + "." + fmt, ...
        'ContentType', 'vector', 'Resolution', 300);
end
fprintf("done.\n");
