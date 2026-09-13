function colors = paper_style()
%% Shared colorblind-safe palette and default axes styling for paper figs.
%
% Call once at the top of each paper_fig*.m, before creating a figure.
% Sets MATLAB's default font/line width for every figure created
% afterward and returns named colors so every script uses the same one
% for e.g. "robot" or "Fine" instead of picking its own RGB triplet.
%
% Inputs:
%   None.
% Outputs:
%   colors: struct of [R G B] triplets in [0, 1], one field per series
%     name used across paper_fig3/4/5.m, plus colors.order (the raw
%     8-color Okabe-Ito palette this is built from).
% Depends on:
%   None.

colors.order = [
    0.000 0.000 0.000    % 1 black
    0.902 0.624 0.000    % 2 orange
    0.337 0.706 0.914    % 3 sky blue
    0.000 0.620 0.451    % 4 bluish green
    0.941 0.894 0.259    % 5 yellow
    0.000 0.447 0.698    % 6 blue
    0.835 0.369 0.000    % 7 vermillion
    0.800 0.475 0.655];  % 8 reddish purple

colors.robot                 = colors.order(6, :);
colors.gaze                  = colors.order(4, :);
colors.gaze_pure              = colors.order(3, :);
colors.joint                 = colors.order(8, :);
colors.robot_plus_gaze        = colors.order(7, :);
colors.robot_plus_gaze_top20 = colors.order(2, :);
colors.all                   = colors.order(7, :);  % alias, same as above
colors.fine                  = colors.order(2, :);
colors.gross                 = colors.order(6, :);
colors.narrow                = colors.order(6, :);
colors.wide                  = colors.order(7, :);
colors.baseline               = [0.55 0.55 0.55];    % majority baseline
colors.persistence             = [0.30 0.30 0.30];    % persistence baseline

set(groot, 'defaultAxesFontSize', 8, 'defaultAxesFontName', 'Helvetica', ...
    'defaultLineLineWidth', 1.2, 'defaultAxesLineWidth', 0.75, ...
    'defaultFigureColor', 'w');
end
