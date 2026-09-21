function text_value = format_or_dash(value)
%FORMAT_OR_DASH "%.3f" for a real number, "  -  " for NaN (empty bin).

    if isnan(value)
        text_value = '  -  ';
    else
        text_value = sprintf('%.3f', value);
    end
end
