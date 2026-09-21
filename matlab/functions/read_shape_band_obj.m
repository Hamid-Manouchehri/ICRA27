function [outer_xy_m, inner_xy_m, centre_xy_m] = read_shape_band_obj( ...
    obj_file, resample_spacing_m)
%READ_SHAPE_BAND_OBJ Reference centre line of a flat band mesh.
%
% Output centre_xy_m is empty when fewer than two boundary loops are
% found. More than two loops are handled too -- a band that pinches to
% a thin neck (e.g. the pear shape's tip) can detach a small extra loop
% there, so the two loops with the LARGEST area are taken as outer/
% inner and any smaller ones are discarded with a warning, rather than
% failing the whole shape.
%
% Copied verbatim from compare_chen_vs_gaze_assisted_performance.m.

outer_xy_m = [];
inner_xy_m = [];
centre_xy_m = [];

file_id = fopen(obj_file, 'r');
if file_id < 0
    return;
end
file_text = textscan(file_id, '%s', 'Delimiter', '\n', ...
    'WhiteSpace', '');
fclose(file_id);
file_lines = file_text{1};

vertices = zeros(0, 3);
faces = {};

for line_idx = 1:numel(file_lines)
    this_line = strtrim(file_lines{line_idx});
    if startsWith(this_line, 'v ')
        values = sscanf(this_line(2:end), '%f');
        vertices(end + 1, 1:3) = values(1:3)'; %#ok<AGROW>
    elseif startsWith(this_line, 'f ')
        tokens = strsplit(strtrim(this_line(2:end)));
        face_indices = zeros(1, numel(tokens));
        for token_idx = 1:numel(tokens)
            slash_parts = strsplit(tokens{token_idx}, '/');
            face_indices(token_idx) = str2double(slash_parts{1});
        end
        faces{end + 1} = face_indices; %#ok<AGROW>
    end
end

if isempty(vertices) || isempty(faces)
    return;
end

z_bottom = min(vertices(:, 3));
z_tolerance = 1e-9 + 1e-6 * (max(vertices(:, 3)) - z_bottom);

edge_list = zeros(0, 2);
for face_idx = 1:numel(faces)
    face_indices = faces{face_idx};
    if ~all(abs(vertices(face_indices, 3) - z_bottom) <= z_tolerance)
        continue;
    end
    for corner_idx = 1:numel(face_indices)
        next_corner = mod(corner_idx, numel(face_indices)) + 1;
        edge_list(end + 1, :) = sort([ ...
            face_indices(corner_idx), ...
            face_indices(next_corner)]); %#ok<AGROW>
    end
end

if isempty(edge_list)
    return;
end

[unique_edges, ~, edge_group] = unique(edge_list, 'rows');
edge_use_count = accumarray(edge_group, 1);
boundary_edges = unique_edges(edge_use_count == 1, :);

if isempty(boundary_edges)
    return;
end

remaining_edges = boundary_edges;
loops = {};
while ~isempty(remaining_edges)
    loop_indices = remaining_edges(1, 1);
    current_vertex = remaining_edges(1, 2);
    remaining_edges(1, :) = [];

    while true
        loop_indices(end + 1) = current_vertex; %#ok<AGROW>
        next_row = find(any(remaining_edges == current_vertex, 2), 1);
        if isempty(next_row)
            break;
        end
        edge_pair = remaining_edges(next_row, :);
        remaining_edges(next_row, :) = [];
        if edge_pair(1) == current_vertex
            current_vertex = edge_pair(2);
        else
            current_vertex = edge_pair(1);
        end
        if current_vertex == loop_indices(1)
            break;
        end
    end

    if numel(loop_indices) > 3
        loops{end + 1} = vertices(loop_indices, 1:2); %#ok<AGROW>
    end
end

if numel(loops) < 2
    return;
end

loop_areas_m2 = zeros(1, numel(loops));
for loop_idx = 1:numel(loops)
    loop_xy = loops{loop_idx};
    loop_areas_m2(loop_idx) = abs(0.5 * sum( ...
        loop_xy(:, 1) .* circshift(loop_xy(:, 2), -1) - ...
        circshift(loop_xy(:, 1), -1) .* loop_xy(:, 2)));
end

[sorted_areas_m2, sorted_order] = sort(loop_areas_m2, 'descend');
outer_index = sorted_order(1);
inner_index = sorted_order(2);

if numel(loops) > 2
    warning(['%s: found %d boundary loop(s) beyond the outer/inner ' ...
        'pair (discarded area(s) %s mm^2, likely a meshing artifact ' ...
        'at a thin pinch point in the band) -- using only the two ' ...
        'largest loops by area.'], obj_file, numel(loops) - 2, ...
        mat2str(round(1e6 * sorted_areas_m2(3:end), 1)));
end

outer_xy_m = resample_closed_loop( ...
    loops{outer_index}, resample_spacing_m);
inner_xy_m = resample_closed_loop( ...
    loops{inner_index}, resample_spacing_m);

centre_xy_m = zeros(size(outer_xy_m));
for sample_idx = 1:size(outer_xy_m, 1)
    distances_m = hypot( ...
        inner_xy_m(:, 1) - outer_xy_m(sample_idx, 1), ...
        inner_xy_m(:, 2) - outer_xy_m(sample_idx, 2));
    [~, nearest_index] = min(distances_m);
    centre_xy_m(sample_idx, :) = 0.5 * ( ...
        outer_xy_m(sample_idx, :) + inner_xy_m(nearest_index, :));
end
end
