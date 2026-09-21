function plot_region_bars(ax, chen_fine, gated_fine, chen_gross, ...
    gated_gross, chen_label, gated_label, chen_color, gated_color, ...
    y_label_text, title_text, value_format)
%PLOT_REGION_BARS One metric as grouped bars: two regions x two arms,
% each bar the mean across that arm's trials.

values = [mean(chen_fine), mean(gated_fine); ...
    mean(chen_gross), mean(gated_gross)];

hold(ax, 'on');
bar_handle = bar(ax, values);
bar_handle(1).FaceColor = chen_color;
bar_handle(2).FaceColor = gated_color;
bar_handle(1).DisplayName = char(chen_label);
bar_handle(2).DisplayName = char(gated_label);

for group_idx = 1:2
    for series_idx = 1:2
        text(ax, bar_handle(series_idx).XEndPoints(group_idx), ...
            bar_handle(series_idx).YEndPoints(group_idx), ...
            sprintf(value_format, values(group_idx, series_idx)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', 'FontSize', 7);
    end
end

set(ax, 'XTick', 1:2, 'XTickLabel', {'Fine region', 'Gross region'});
grid(ax, 'on');
ylabel(ax, y_label_text);
title(ax, title_text, 'FontSize', 9);
legend(ax, 'Location', 'best', 'FontSize', 7);
end
