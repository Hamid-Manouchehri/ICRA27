function result = iif(condition, true_value, false_value)
%IIF Inline conditional, since MATLAB has no ternary operator.

if condition
    result = true_value;
else
    result = false_value;
end
end
