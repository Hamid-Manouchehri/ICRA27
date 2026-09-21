function plot_grouped_metric_bars(ax, chen_values, gated_values, ...
        group_labels, chen_label, gated_label, chen_color, gated_color, ...
        y_label_text, title_text, value_format)
%PLOT_GROUPED_METRIC_BARS One metric, one tile: N (shape, bandwidth)
% groups, two colored bars per group (Chen vs. gated), each bar's mean
% value printed above it. NaN entries (a missing cell) are simply not
% drawn -- bar() skips them.

    group_count = numel(chen_values);
    values = [chen_values(:), gated_values(:)];

    hold(ax, 'on');
    bar_handle = bar(ax, values);
    bar_handle(1).FaceColor = chen_color;
    bar_handle(2).FaceColor = gated_color;
    bar_handle(1).DisplayName = char(chen_label);
    bar_handle(2).DisplayName = char(gated_label);

    for group_idx = 1:group_count
        for series_idx = 1:2
            value = values(group_idx, series_idx);
            if isnan(value)
                continue;
            end
            text(ax, bar_handle(series_idx).XEndPoints(group_idx), ...
                bar_handle(series_idx).YEndPoints(group_idx), ...
                sprintf(value_format, value), ...
                'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'bottom', 'FontSize', 6);
        end
    end

    set(ax, 'XTick', 1:group_count, 'XTickLabel', cellstr(group_labels), ...
        'XTickLabelRotation', 20);
    grid(ax, 'on');
    ylabel(ax, y_label_text);
    title(ax, title_text, 'FontSize', 9);
    legend(ax, 'Location', 'best', 'FontSize', 7);
end
