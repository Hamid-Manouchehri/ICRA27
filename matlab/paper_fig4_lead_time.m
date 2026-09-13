%% Accuracy vs anticipation horizon (lead time), robot vs robot+gaze.
%
% The paper's central figure. Individual subject lines sit behind the
% mean +/- SD per feature set. The persistence baseline ("assume the
% label didn't change") is only drawn where it is not degenerate: at
% lead 0s it scores the label against itself, and at lead 2s the
% purity filter removes every window pair that could disagree, so both
% read exactly 100% by construction (see decision_log.md, 2026-09-12).
%
% Inputs:
%   data/processed_data/paper_fig4_lead_time_W_2s.csv -- one row per
%     (feature_set, lead_time_s, subject): per-subject accuracy.
%   data/processed_data/paper_tab3_lead_time_stats_W_2s.csv -- one row
%     per lead_time_s: pre-computed mean/SD and paired-test p-values,
%     read here for display only and never recomputed.
% Output:
%   data/images/paper_fig4_lead_time_W_2s.{png,pdf}
% Depends on:
%   matlab/paper_style.m
%
% Requires MATLAB R2020a or newer (exportgraphics). Not run in this
% session -- no MATLAB available here; reasoned through manually per
% this project's convention for .m files. Report the first error back
% if one occurs.

% -------------------- User Settings --------------------
dataset_name = "W_2s";                    % TODO must match build_paper_results.py
feature_sets_to_show = ["robot", "all"];  % TODO subset of column values
show_subject_lines = true;                % TODO individual subject trajectories
y_axis_limits_pct = [0, 100];             % TODO full range, not truncated
significance_alpha = 0.05;                % TODO threshold for the ** annotation
output_formats = ["png", "pdf"];
% ---------------------------------------------------------

project_root = fileparts(fileparts(mfilename('fullpath')));
read_fold_csv_path = fullfile(project_root, "data", "processed_data", ...
    "paper_fig4_lead_time_" + dataset_name + ".csv");
read_summary_csv_path = fullfile(project_root, "data", "processed_data", ...
    "paper_tab3_lead_time_stats_" + dataset_name + ".csv");
write_image_path = fullfile(project_root, "data", "images", ...
    "paper_fig4_lead_time_" + dataset_name);

fprintf("paper_fig4_lead_time.m\n");
fprintf("  reading:  %s\n", read_fold_csv_path);
fprintf("  reading:  %s\n", read_summary_csv_path);
for fmt = output_formats
    fprintf("  writing:  %s.%s\n", write_image_path, fmt);
end

% TextType 'string' is required: without it, text columns come back as
% cell arrays of char, and "== name" below would error on a cell array.
folds = readtable(read_fold_csv_path, 'Delimiter', ',', ...
    'TextType', 'string');
summary = readtable(read_summary_csv_path, 'Delimiter', ',', ...
    'TextType', 'string');
colors = paper_style();

lead_times = sort(unique(summary.lead_time_s));
is_degenerate = strcmpi(string(summary.persistence_degenerate), "true");

figure('Name', 'Accuracy vs anticipation horizon', ...
    'Position', [100 100 480 380]);
hold on;

% ---- individual subject lines, drawn first so they sit behind ----
if show_subject_lines
    for name = feature_sets_to_show
        color = colors.(char(name));
        for s = unique(folds.subject)'
            rows = folds(folds.feature_set == name & ...
                folds.subject == s, :);
            rows = sortrows(rows, 'lead_time_s');
            plot(rows.lead_time_s, rows.accuracy, '-', ...
                'Color', [color, 0.15], 'LineWidth', 0.75, ...
                'HandleVisibility', 'off');
        end
    end
end

% ---- persistence baseline, only where it is not degenerate ----
valid = ~is_degenerate;
plot(summary.lead_time_s(valid), summary.persistence_mean(valid), ...
    'd--', 'Color', colors.persistence, 'MarkerFaceColor', ...
    colors.persistence, 'DisplayName', 'persistence (valid horizons)');

% shade the horizon range where persistence carries no information
degenerate_lead_times = summary.lead_time_s(is_degenerate);
if ~isempty(degenerate_lead_times)
    x_left = min(lead_times) - 0.5;
    x_right = max(degenerate_lead_times) + 1.0;
    patch([x_left, x_right, x_right, x_left], ...
        [y_axis_limits_pct(1), y_axis_limits_pct(1), ...
         y_axis_limits_pct(2), y_axis_limits_pct(2)], ...
        [0.9 0.9 0.9], 'FaceAlpha', 0.4, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');
    text(mean([x_left, x_right]), y_axis_limits_pct(2) - 4, ...
        'persistence undefined here', 'HorizontalAlignment', 'center', ...
        'FontSize', 6.5, 'Color', [0.4 0.4 0.4]);
end

% ---- mean +/- SD per feature set, read from the summary, not recomputed ----
for name = feature_sets_to_show
    color = colors.(char(name));
    mean_col = [char(name) '_accuracy_mean'];
    sd_col = [char(name) '_accuracy_sd'];
    errorbar(summary.lead_time_s, summary.(mean_col), summary.(sd_col), ...
        '-o', 'Color', color, 'MarkerFaceColor', color, 'LineWidth', 1.6, ...
        'CapSize', 5, 'DisplayName', strrep(char(name), '_', ' '));
end

% ---- ** where accuracy significantly beats persistence (paired Wilcoxon) ----
for i = 1:height(summary)
    if is_degenerate(i) || isnan(summary.robot_vs_persistence_p(i))
        continue
    end
    if summary.robot_vs_persistence_p(i) < significance_alpha
        y_position = max(summary.robot_accuracy_mean(i), ...
            summary.all_accuracy_mean(i)) + 6;
        text(summary.lead_time_s(i), y_position, '**', ...
            'HorizontalAlignment', 'center', 'FontSize', 11);
    end
end

xlim([min(lead_times) - 0.5, max(lead_times) + 0.5]);
ylim(y_axis_limits_pct);
xlabel('lead time (s)');
ylabel('accuracy (%)');
legend('Location', 'southwest', 'FontSize', 7, 'Box', 'off');
title({'Accuracy vs anticipation horizon', ...
    '** p < 0.05 vs persistence (paired Wilcoxon, n = 8)'}, ...
    'FontSize', 8);
box on;

for fmt = output_formats
    exportgraphics(gcf, write_image_path + "." + fmt, ...
        'ContentType', 'vector', 'Resolution', 300);
end
fprintf("done.\n");
