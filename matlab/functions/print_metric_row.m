function print_metric_row(name, chen_values, gated_values, number_format)
%PRINT_METRIC_ROW One summary-table row: arm means and their difference.

chen_mean = mean(chen_values);
gated_mean = mean(gated_values);
fprintf(['%-28s ' number_format ' ' number_format ' %+10.3g\n'], ...
    name, chen_mean, gated_mean, gated_mean - chen_mean);
end
