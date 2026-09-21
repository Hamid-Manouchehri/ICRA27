function plot_metric_bars(ax, chen_values, gated_values, chen_label, ...
    gated_label, chen_color, gated_color, y_label_text, title_text, ...
    value_format)
%PLOT_METRIC_BARS One metric, one tile: two colored bars, a std error
% bar (zero-length with only one trial per arm), and the mean value
% printed above each bar.

means = [mean(chen_values), mean(gated_values)];
stds = [0, 0];
if numel(chen_values) > 1
    stds(1) = std(chen_values);
end
if numel(gated_values) > 1
    stds(2) = std(gated_values);
end

hold(ax, 'on');
bar_handle = bar(ax, 1:2, means, 'FaceColor', 'flat');
bar_handle.CData(1, :) = chen_color;
bar_handle.CData(2, :) = gated_color;
errorbar(ax, 1:2, means, stds, 'LineStyle', 'none', 'Color', 'k', ...
    'LineWidth', 1.2, 'CapSize', 8, 'HandleVisibility', 'off');
for bar_idx = 1:2
    text(ax, bar_idx, means(bar_idx) + stds(bar_idx), ...
        sprintf(value_format, means(bar_idx)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 8);
end
set(ax, 'XTick', 1:2, 'XTickLabel', {char(chen_label), char(gated_label)});
grid(ax, 'on');
ylabel(ax, y_label_text);
title(ax, title_text, 'FontSize', 9);
end
