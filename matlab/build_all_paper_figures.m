%% Convenience wrapper: run every paper_fig*.m in one call.
%
% Each script remains independently runnable on its own; this just
% saves calling them one by one after regenerating the underlying CSVs.
%
% Inputs:
%   None directly -- each paper_fig*.m reports its own inputs.
% Output:
%   None directly -- each paper_fig*.m writes its own image(s).
% Depends on:
%   matlab/paper_fig3_accuracy.m
%   matlab/paper_fig4_lead_time.m
%   matlab/paper_fig5_chen_auc.m

clc; clear; close all

fprintf("build_all_paper_figures.m: running all paper_fig*.m scripts\n");
scripts_to_run = ["paper_fig3_accuracy", "paper_fig4_lead_time", ...
    "paper_fig5_chen_auc"];
for script_name = scripts_to_run
    fprintf("\n--- %s ---\n", script_name);
    run(script_name);
end
fprintf("\nall figures regenerated.\n");
