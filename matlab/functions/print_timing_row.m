function print_timing_row(arm_name, metrics)
%PRINT_TIMING_ROW Control-loop period stats, pooled across an arm's trials.

all_period_ms = vertcat(metrics.period_ms{:});
fprintf(['  %-6s median %.3f ms, max %.3f ms, mean overrun fraction ' ...
    '%.3f %%\n'], arm_name, median(all_period_ms), ...
    max(all_period_ms), 100 * mean(metrics.overrun_fraction));
end
