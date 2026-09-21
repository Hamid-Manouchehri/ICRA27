function print_region_row(name, chen_fine, gated_fine, chen_gross, ...
    gated_gross, number_format)
%PRINT_REGION_ROW One per-region row: both arms and the gated-minus-Chen
% difference, for each region.

fine_chen = mean(chen_fine);
fine_gated = mean(gated_fine);
gross_chen = mean(chen_gross);
gross_gated = mean(gated_gross);

fprintf(['%-22s ' number_format ' ' number_format ' %+10.3g   ' ...
    number_format ' ' number_format ' %+10.3g\n'], ...
    name, fine_chen, fine_gated, fine_gated - fine_chen, ...
    gross_chen, gross_gated, gross_gated - gross_chen);
end
