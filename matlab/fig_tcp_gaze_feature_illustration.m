
clc; clear; close all;

%% 1. Bézier control points extracted from PowerPoint SVG
P0 = [189, 362.983];

% Each row: [P1x P1y P2x P2y P3x P3y]
C = [
    251.417 216.771 313.833  70.5594 365  99.7517;
    416.167 128.944 423.833 507.109  496 538.137;
    568.167 569.164 700.833 338.962  798 285.915;
    895.167 232.869 1031.17 173.483 1079 219.857
];

%% 2. Sample the spline and calculate analytical derivatives
P = [];
D1 = [];
D2 = [];

for i = 1:size(C,1)
    P1 = C(i,1:2);
    P2 = C(i,3:4);
    P3 = C(i,5:6);

    t = linspace(0,1,200)';
    if i > 1
        t = t(2:end);  % Avoid duplicate endpoints
    end

    % Cubic Bezier curve
    B = (1-t).^3*P0 + ...
        3*(1-t).^2.*t*P1 + ...
        3*(1-t).*t.^2*P2 + ...
        t.^3*P3;

    % First derivative
    dB = 3*(1-t).^2*(P1-P0) + ...
         6*(1-t).*t*(P2-P1) + ...
         3*t.^2*(P3-P2);

    % Second derivative
    ddB = 6*(1-t)*(P2-2*P1+P0) + ...
          6*t*(P3-2*P2+P1);

    P  = [P; B];
    D1 = [D1; dB];
    D2 = [D2; ddB];

    P0 = P3;
end

%% 3. Convert SVG coordinates to robot workspace (meters)
desired_width_m = 0.30;

scale = desired_width_m / (max(P(:,1))-min(P(:,1)));

x = (P(:,1)-min(P(:,1))) * scale;
y = -(P(:,2)-min(P(:,2))) * scale;

% Scale derivatives consistently
vx =  D1(:,1)*scale;
vy = -D1(:,2)*scale;

ax =  D2(:,1)*scale;
ay = -D2(:,2)*scale;

%% 4. Calculate curvature (1/m)
kappa = abs(vx.*ay - vy.*ax) ./ ...
        (vx.^2 + vy.^2).^(3/2);

%% 5. Plot colored trajectory
figure('Color','w');

surface([x';x'], [y';y'], zeros(2,length(x)), ...
        [kappa';kappa'], ...
        'FaceColor','none', ...
        'EdgeColor','interp', ...
        'LineWidth',50);

axis equal;
axis off;
view(2);





% Sky-blue colormap: light -> dark
colors = [
    0.85 0.95 1.00;   % Light sky blue (low curvature)
    0.35 0.70 0.95;   % Sky blue
    0.02 0.25 0.65    % Deep blue (high curvature)
];

cmap = interp1(linspace(0,1,3), colors, ...
               linspace(0,1,256));

colormap(cmap);

cb = colorbar;
% cb.Label.String = 'Curvature (m^{-1})';

cb.Ticks = [];          % Remove numerical ticks
cb.Label.String = '';   % Remove colorbar label

% Get colorbar position
cb.Units = 'normalized';
drawnow;
p = cb.Position;

% Text above the colorbar
annotation('textbox', ...
    [p(1)-0.04, p(2)+p(4)+0.01, p(3)+0.08, 0.04], ...
    'String','High curvature', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontSize',10);

% Text below the colorbar
annotation('textbox', ...
    [p(1)-0.04, p(2)-0.06, p(3)+0.08, 0.04], ...
    'String','Low curvature', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontSize',10);

%% 6. Export as vector PDF
exportgraphics(gcf,'powerpoint_curvature.pdf', ...
               'ContentType','vector');
