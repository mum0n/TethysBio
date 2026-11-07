"""
    tethysbio_project_path(path) -> String

Resolve a data or output path inside the TethysBio project directory.
"""
function tethysbio_project_path(path::AbstractString)::String
    value = strip(String(path))
    isempty(value) && throw(ArgumentError("TethysBio path cannot be empty"))
    root = abspath(_PROJECT_ROOT)
    resolved = normpath(isabspath(value) ? value : joinpath(root, value))
    relative = relpath(resolved, root)
    if relative == ".." || startswith(relative, "..$(Base.Filesystem.path_separator)")
        throw(ArgumentError("TethysBio data and output paths must remain inside $(root): $(path)"))
    end
    return resolved
end

function tethysbio_data_path(subpaths...)::String
    return tethysbio_project_path(joinpath("data", subpaths...))
end
