%% Chen's commanded damping as an implicit fine/gross detector.
%
% Panel A: commanded damping distribution during hand-labeled Fine vs
% Gross windows (each normalized to its own total, since Fine and Gross
% have different sample counts and raw counts would not be comparable).
% Panel B: per-trial AUC of that damping against the hand labels, split
% by trace bandwidth, with a chance-level reference line at 0.5.
%
% Inputs:
%   data/processed_data/paper_fig5_chen_damping_hist_W_2s.csv -- one
%     row per (label, damping bin): histogram counts.
%   data/processed_data/paper_fig5_chen_auc_W_2s.csv -- one row per
%     trial: per-trial AUC and mean damping by label.
% Output:
%   data/images/paper_fig5_chen_auc_W_2s.{png,pdf}
% Depends on:
%   matlab/paper_style.m
%
% Requires MATLAB R2020a or newer (exportgraphics). Not run in this
% session -- no MATLAB available here; reasoned through manually per
% this project's convention for .m files. Report the first error back
% if one occurs.

% -------------------- User Settings --------------------
dataset_name = "W_2s";           % TODO must match build_paper_results.py
bandwidths_to_show = ["narrow", "wide"];  % TODO subset of column values
y_axis_limits_auc = [0.4, 1.0];  % TODO AUC axis range for panel B
output_formats = ["png", "pdf"];
% ---------------------------------------------------------

project_root = fileparts(fileparts(mfilename('fullpath')));
read_hist_csv_path = fullfile(project_root, "data", "processed_data", ...
    "paper_fig5_chen_damping_hist_" + dataset_name + ".csv");
read_auc_csv_path = fullfile(project_root, "data", "processed_data", ...
    "paper_fig5_chen_auc_" + dataset_name + ".csv");
write_image_path = fullfile(project_root, "data", "images", ...
    "paper_fig5_chen_auc_" + dataset_name);

fprintf("paper_fig5_chen_auc.m\n");
fprintf("  reading:  %s\n", read_hist_csv_path);
fprintf("  reading:  %s\n", read_auc_csv_path);
for fmt = output_formats
    fprintf("  writing:  %s.%s\n", write_image_path, fmt);
end

% TextType 'string' is required: without it, text columns come back as
% cell arrays of char, and "== name" below would error on a cell array.
histogram_table = readtable(read_hist_csv_path, 'Delimiter', ',', ...
    'TextType', 'string');
auc_table = readtable(read_auc_csv_path, 'Delimiter', ',', ...
    'TextType', 'string');
colors = paper_style();

figure('Name', 'Chen damping as an implicit detector', ...
    'Position', [100 100 760 340]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

% ---- Panel A: damping distribution by label, each normalized to 1 ----
nexttile; hold on;
for label = ["Fine", "Gross"]
    rows = histogram_table(histogram_table.label == label, :);
    rows = sortrows(rows, 'bd_bin_center_n_s_m');
    proportion = rows.count / sum(rows.count);
    bar(rows.bd_bin_center_n_s_m, proportion, 1, ...
        'FaceColor', colors.(lower(char(label))), 'FaceAlpha', 0.55, ...
        'EdgeColor', 'none', 'DisplayName', char(label));
end
xlabel('commanded damping, b_d (N{\cdot}s/m)');
ylabel('proportion of samples (within label)');
legend('Location', 'northeast', 'FontSize', 7, 'Box', 'off');
title('A: damping distribution by hand label');
box on;

% ---- Panel B: per-trial AUC, split by bandwidth ----
nexttile; hold on;
yline(0.5, ':', 'Color', [0.5 0.5 0.5], 'Label', 'chance', ...
    'FontSize', 7, 'HandleVisibility', 'off');
n_groups = numel(bandwidths_to_show);
for i = 1:n_groups
    name = bandwidths_to_show(i);
    rows = auc_table(auc_table.bandwidth == name, :);
    color = colors.(char(name));
    jitter = (rand(height(rows), 1) - 0.5) * 0.3;
    scatter(i + jitter, rows.auc, 18, color, 'filled', ...
        'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
    errorbar(i, mean(rows.auc), std(rows.auc), 'o', 'Color', color, ...
        'MarkerFaceColor', color, 'LineWidth', 1.5, 'CapSize', 6, ...
        'HandleVisibility', 'off');
end
xlim([0.5, n_groups + 0.5]);
ylim(y_axis_limits_auc);
xticks(1:n_groups);
xticklabels(cellstr(bandwidths_to_show));
ylabel('per-trial AUC (damping vs hand label)');
title(sprintf('B: %d trials, mean AUC = %.3f', height(auc_table), ...
    mean(auc_table.auc)));
box on;

for fmt = output_formats
    exportgraphics(gcf, write_image_path + "." + fmt, ...
        'ContentType', 'vector', 'Resolution', 300);
end
fprintf("done.\n");
