function indices = resolve_subject_indices(sID, subject_ids)
%RESOLVE_SUBJECT_INDICES sID indices for a cell array of subject id
% strings, or a clear error if one isn't found. string(sID), not
% strcmp -- sID's elements are MATLAB "string" (double-quoted in ET.m),
% and strcmp's cell-array contract requires the other side to be a cell
% array or a CHARACTER array, not a string scalar.

    indices = zeros(1, numel(subject_ids));
    for idx = 1:numel(subject_ids)
        this_index = find(string(sID) == subject_ids{idx});
        if isempty(this_index)
            error('Subject id "%s" not found in sID.', subject_ids{idx});
        end
        indices(idx) = this_index;
    end
end
