function var = getVarFromStruct(object, fieldName)
    if isfield(object, fieldName)
        fullName = sprintf('object.%s', fieldName);
        var = eval(fullName);
        if iscell(var)
            for i=1:length(var)
                var{i} = double(var{i});
            end
        else
             var = double(var);
        end
    else
        var = [];
    end
end
