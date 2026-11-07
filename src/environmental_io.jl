function load_environmental_dataset(path::AbstractString; backend = nothing)
    return GeoData.geoload(String(path); backend = backend)
end

function environmental_axis(dataset, name::Symbol)
    return GeoData.axis(dataset, name)
end

function environmental_bathymetry_interpolator(dataset; varname::AbstractString = "elevation")
    variables = dataset.variables
    variable = if haskey(variables, varname)
        varname
    else
        candidates = ("elevation", "altitude", "z", "topo", "bedrock_altitude")
        found = findfirst(candidate -> haskey(variables, candidate), candidates)
        found === nothing && error(
            "Bathymetry dataset has no $(repr(varname)) or supported elevation variable."
        )
        candidates[found]
    end
    geoarray = variables[variable]
    dims = geoarray.dims
    lon_index = findfirst(dim -> dim.name in (:lon, :longitude, :x), dims)
    lat_index = findfirst(dim -> dim.name in (:lat, :latitude, :y), dims)
    (lon_index === nothing || lat_index === nothing) && error(
        "Bathymetry variable $(repr(variable)) must have longitude and latitude dimensions."
    )
    ndims(geoarray.data) == 2 || error(
        "Bathymetry variable $(repr(variable)) must be two-dimensional."
    )

    lons, _ = environmental_axis(dataset, :lon)
    lats, _ = environmental_axis(dataset, :lat)
    order = (lon_index, lat_index)
    elevation = permutedims(Float64.(Array(geoarray.data)), order)
    size(elevation) == (length(lons), length(lats)) || error(
        "Bathymetry dimensions $(size(elevation)) do not match coordinate axes " *
        "($(length(lons)), $(length(lats)))."
    )

    lon_order = sortperm(lons)
    lat_order = sortperm(lats)
    sorted_lons = Float64.(lons[lon_order])
    sorted_lats = Float64.(lats[lat_order])
    sorted_elevation = elevation[lon_order, lat_order]
    interpolator = Interpolations.interpolate(
        (sorted_lons, sorted_lats),
        sorted_elevation,
        Interpolations.Gridded(Interpolations.Linear())
    )
    flat = Interpolations.extrapolate(interpolator, Interpolations.Flat())
    return (lon::Real, lat::Real) -> Float64(flat(Float64(lon), Float64(lat)))
end

function local_bathymetry_path(output_dir::AbstractString)
    candidates = (
        joinpath(output_dir, "bathymetry.zarr"),
        joinpath(output_dir, "bathymetry.nc"),
        joinpath(output_dir, "nwa_bathymetry.zarr"),
        tethysbio_data_path("inputs", "bathymetry.zarr"),
        tethysbio_data_path("inputs", "bathymetry.nc")
    )
    index = findfirst(path -> isdir(path) || isfile(path), candidates)
    return index === nothing ? nothing : candidates[index]
end

function local_hydrography_path(env_params::EnvironmentParams, output_dir::AbstractString)
    source = Symbol(env_params.temperature_source)
    month = lpad(string(env_params.temperature_month), 2, '0')
    candidates = source === :woa23 ? (
        joinpath(output_dir, "woa23_temperature_$(month)_0.25deg.zarr"),
    ) : (
        joinpath(output_dir, "boundary_$(source).zarr"),
        joinpath(output_dir, "boundary_$(source).nc")
    )
    index = findfirst(path -> isdir(path) || isfile(path), candidates)
    return index === nothing ? nothing : candidates[index]
end

function fetch_local_bathymetry!(
    env_params::EnvironmentParams,
    output_dir::AbstractString;
    force::Bool = false,
    verbose::Bool = true
)
    source = lowercase(env_params.bathymetry_source)
    source in ("etopo2022", "nwa_bathymetry", "erddap") || throw(
        ArgumentError("Unsupported bathymetry source: $(env_params.bathymetry_source)")
    )
    target = joinpath(output_dir, "bathymetry.zarr")
    local_path = local_bathymetry_path(output_dir)
    !force && local_path !== nothing && return local_path
    if force || !isdir(target)
        GeoDataSources.fetch_erddap_bathymetry(
            dataset_id = "ETOPO_2022_v1_15s",
            lon_range = Tuple(Float64.(env_params.domain_lon)),
            lat_range = Tuple(Float64.(env_params.domain_lat)),
            output_path = target,
            force = force,
            verbose = verbose
        )
    end
    return target
end

function fetch_local_hydrography!(
    env_params::EnvironmentParams,
    output_dir::AbstractString;
    force::Bool = false,
    verbose::Bool = true
)
    source = Symbol(env_params.temperature_source)
    month = env_params.temperature_month
    target = local_hydrography_path(env_params, output_dir)
    target !== nothing && !force && return target
    source === :woa23 || throw(ArgumentError(
        "Hydrography source $(source) is not in TethysBio's local data directory. " *
        "Add a local boundary_$(source).zarr or select temperature_source = \"woa23\"."
    ))
    month_string = lpad(string(month), 2, '0')
    target = joinpath(output_dir, "woa23_temperature_$(month_string)_0.25deg.zarr")
    if force || !isdir(target)
        GeoDataSources.fetch_woa23(
            lon_range = Tuple(Float64.(env_params.domain_lon)),
            lat_range = Tuple(Float64.(env_params.domain_lat)),
            month = month,
            output_dir = output_dir,
            include_o2 = false,
            backend = :zarr,
            force = force,
            verbose = verbose
        )
    end
    return target
end
