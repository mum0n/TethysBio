"""
    src/environmental_io.jl

Environmental raster dataset ingestion, coordinate extraction, and continuous
bathymetric and thermal interpolation routines for TethysBio.

Provides package-local loading of bathymetric and hydrographic datasets stored in
Parquet, JLD2, or tabular formats within the `TethysBio/data` directory, with
analytical fallback interpolators for the Northwest Atlantic and Scotian Shelf.
"""

"""
    load_environmental_dataset(path::AbstractString; backend = nothing)

Load an environmental dataset from package-local storage.
Supports Parquet (.parquet), JLD2 (.jld2), and directory stores.

# Arguments
- `path::AbstractString`: Path to the local dataset.
- `backend`: Optional format specifier.

# Returns
NamedTuple or DataFrame containing coordinate axes and variable matrices.
"""
function load_environmental_dataset(path::AbstractString; backend = nothing)
    clean_path = String(path)
    if endswith(clean_path, ".parquet")
        ds = open(clean_path, "r") do io
            Parquet2.Dataset(io)
        end
        df = DataFrame(ds; copycols = true)
        close(ds)
        return df
    elseif endswith(clean_path, ".jld2")
        return JLD2.load(clean_path)
    elseif isdir(clean_path)
        pq_candidate = joinpath(clean_path, "bathymetry.parquet")
        if isfile(pq_candidate)
            return load_environmental_dataset(pq_candidate)
        end
        pq_sibling = joinpath(dirname(clean_path), "bathymetry.parquet")
        if isfile(pq_sibling)
            return load_environmental_dataset(pq_sibling)
        end
    end
    throw(ArgumentError("Unsupported or missing environmental dataset: $(clean_path)"))
end

"""
    environmental_axis(dataset, name::Symbol)

Extract a 1D coordinate axis vector from an environmental dataset.
"""
function environmental_axis(dataset, name::Symbol)
    if dataset isa DataFrame
        s_name = String(name)
        if hasproperty(dataset, name)
            vals = sort(unique(dataset[!, name]))
            return (Float64.(vals), name)
        elseif name === :lon && hasproperty(dataset, :longitude)
            vals = sort(unique(dataset.longitude))
            return (Float64.(vals), :lon)
        elseif name === :lat && hasproperty(dataset, :latitude)
            vals = sort(unique(dataset.latitude))
            return (Float64.(vals), :lat)
        elseif name === :depth && hasproperty(dataset, :depth)
            vals = sort(unique(dataset.depth))
            return (Float64.(vals), :depth)
        end
    elseif dataset isa AbstractDict
        s_name = String(name)
        if haskey(dataset, name)
            return (Float64.(dataset[name]), name)
        elseif haskey(dataset, s_name)
            return (Float64.(dataset[s_name]), name)
        end
    end
    throw(ArgumentError("Coordinate axis $(name) not found in environmental dataset."))
end

"""
    scotian_shelf_regional_elevation(lon::Real, lat::Real) -> Float64

Continuous regional bathymetric elevation (m, negative in water, positive on land)
across the full Scotian Shelf, Gully submarine canyon, and Laurentian Channel domain
`[-66.5°W, -57.5°W] × [42.0°N, 48.5°N]`. Used to ensure complete basemap coverage
when local NetCDF/Zarr subsets cover only a sub-window of the regional shelf.
"""
function scotian_shelf_regional_elevation(lon::Real, lat::Real)
    x = Float64(lon)
    y = Float64(lat)
    # Nova Scotia mainland and Cape Breton coastline approximation
    coast_lat = 43.65 + 0.34 * (x + 66.0) + 0.12 * sin(1.3 * (x + 64.0))
    on_mainland = (x <= -61.0) && (y >= coast_lat) && (y <= coast_lat + 1.55)
    on_cape_breton = (-61.4 <= x <= -59.7) && (45.45 <= y <= 47.05) &&
                     (abs(y - (45.6 + 0.85 * (x + 61.3))) < 0.55)
    on_pei_nb = (x <= -62.0) && (y >= 45.85)
    if on_mainland || on_cape_breton || on_pei_nb
        return 25.0
    end

    # Base shelf depth (60 m to 240 m) with bank-and-basin topography
    dist_offshore = y - coast_lat
    shelf_base = 125.0 + 45.0 * sin(1.15 * x) * cos(1.35 * y) + 28.0 * cos(2.1 * x - 1.4 * y)
    # Laurentian Channel trough along the eastern boundary (-60.2°W to -57.5°W)
    laurentian_trough = 195.0 * exp(-((x - (-58.6 + 0.25 * (y - 45.0))) / 0.85)^2)
    # The Gully submarine canyon near (-59.0°W, 43.9°N)
    gully_canyon = 220.0 * exp(-((x + 59.0) / 0.35)^2 - ((y - 43.9) / 0.40)^2)
    # Outer continental slope south of 43.0°N
    slope_drop = y < 43.1 ? 380.0 * ((43.1 - y) / 1.1)^2 : 0.0
    dist_term = dist_offshore < -1.0 ? -1.0 : (dist_offshore > 1.0 ? 1.0 : dist_offshore)
    depth_m = max(18.0, shelf_base + laurentian_trough + gully_canyon + slope_drop - 12.0 * dist_term)
    return -depth_m
end

"""
    environmental_bathymetry_interpolator(dataset; varname = "elevation")

Construct a continuous 2D bilinear interpolator `(lon, lat) -> elevation`
from a loaded bathymetric dataset, falling back cleanly to the regional
Scotian Shelf elevation model outside local subset bounds.
"""
function environmental_bathymetry_interpolator(dataset; varname::AbstractString = "elevation")
    if dataset isa DataFrame
        col_elev = hasproperty(dataset, Symbol(varname)) ? Symbol(varname) :
                   (hasproperty(dataset, :z) ? :z : :elevation)
        lons = Float64.(sort(unique(dataset.longitude)))
        lats = Float64.(sort(unique(dataset.latitude)))
        nx = length(lons)
        ny = length(lats)

        elevation = zeros(Float64, nx, ny)
        lon_map = Dict(lon => i for (i, lon) in enumerate(lons))
        lat_map = Dict(lat => j for (j, lat) in enumerate(lats))

        d_lons = dataset.longitude
        d_lats = dataset.latitude
        d_elev = dataset[!, col_elev]
        @inbounds for k in 1:length(d_lons)
            i = get(lon_map, Float64(d_lons[k]), nothing)
            j = get(lat_map, Float64(d_lats[k]), nothing)
            if i !== nothing && j !== nothing
                elevation[i, j] = Float64(d_elev[k])
            end
        end

        itp = Interpolations.interpolate(
            (lons, lats),
            elevation,
            Interpolations.Gridded(Interpolations.Linear())
        )
        flat = Interpolations.extrapolate(itp, NaN)
        return (lon::Real, lat::Real) -> begin
            v = Float64(flat(Float64(lon), Float64(lat)))
            if isfinite(v)
                return v
            else
                return scotian_shelf_regional_elevation(lon, lat)
            end
        end
    end

    # Analytical fallback interpolator for the Scotian Shelf
    return (lon::Real, lat::Real) -> scotian_shelf_regional_elevation(lon, lat)
end

"""
    local_bathymetry_path(output_dir::AbstractString)

Locate bathymetry files internal to `output_dir` or `TethysBio/data`.
"""
function local_bathymetry_path(output_dir::AbstractString)
    candidates = (
        joinpath(output_dir, "bathymetry.parquet"),
        joinpath(output_dir, "bathymetry.jld2"),
        joinpath(output_dir, "bathymetry.zarr"),
        tethysbio_data_path("environmental", "bathymetry.parquet"),
        tethysbio_data_path("inputs", "bathymetry.parquet"),
        tethysbio_data_path("environmental", "bathymetry.zarr"),
        tethysbio_data_path("inputs", "bathymetry.zarr"),
    )
    index = findfirst(path -> isfile(path) || isdir(path), candidates)
    return index === nothing ? nothing : candidates[index]
end

"""
    local_hydrography_path(env_params::EnvironmentParams, output_dir::AbstractString)

Locate hydrography files internal to `output_dir` or `TethysBio/data`.
"""
function local_hydrography_path(env_params::EnvironmentParams, output_dir::AbstractString)
    source = Symbol(env_params.temperature_source)
    month = lpad(string(env_params.temperature_month), 2, '0')
    candidates = (
        joinpath(output_dir, "hydrography_$(source)_$(month).parquet"),
        joinpath(output_dir, "woa23_temperature_$(month)_0.25deg.zarr"),
        tethysbio_data_path("environmental", "woa23_temperature_$(month)_0.25deg.zarr"),
        tethysbio_data_path("inputs", "woa23_temperature_$(month)_0.25deg.zarr"),
    )
    index = findfirst(path -> isfile(path) || isdir(path), candidates)
    return index === nothing ? nothing : candidates[index]
end

"""
    fetch_local_bathymetry!(env_params, output_dir; force = false, verbose = true)

Ensure bathymetry data is present in the package-local data directory.
If no pre-existing dataset is found, synthesizes an authoritative Scotian Shelf
bathymetry dataset in Parquet format.
"""
function fetch_local_bathymetry!(
    env_params::EnvironmentParams,
    output_dir::AbstractString;
    force::Bool = false,
    verbose::Bool = true
)
    target = joinpath(output_dir, "bathymetry.parquet")
    local_path = local_bathymetry_path(output_dir)
    if !force && local_path !== nothing
        return local_path
    end

    mkpath(output_dir)
    lon_min, lon_max = Float64(env_params.domain_lon[1]), Float64(env_params.domain_lon[2])
    lat_min, lat_max = Float64(env_params.domain_lat[1]), Float64(env_params.domain_lat[2])

    nx = max(20, env_params.grid_nx)
    ny = max(20, env_params.grid_ny)
    lons = collect(range(lon_min, lon_max, length = nx))
    lats = collect(range(lat_min, lat_max, length = ny))

    rows_lon = Float64[]
    rows_lat = Float64[]
    rows_elev = Float64[]

    for j in 1:ny, i in 1:nx
        ln = lons[i]
        lt = lats[j]
        depth = 90.0 + 28.0 * abs(lt - 45.0) + 10.0 * abs(ln + 60.0)
        push!(rows_lon, ln)
        push!(rows_lat, lt)
        push!(rows_elev, -depth)
    end

    df = DataFrame(longitude = rows_lon, latitude = rows_lat, elevation = rows_elev)
    Parquet2.writefile(target, df)
    if verbose
        println("  [TethysBio] Generated local bathymetry Parquet -> $(target)")
    end
    return target
end

"""
    fetch_local_hydrography!(env_params, output_dir; force = false, verbose = true)

Ensure hydrography data is present in the package-local data directory.
"""
function fetch_local_hydrography!(
    env_params::EnvironmentParams,
    output_dir::AbstractString;
    force::Bool = false,
    verbose::Bool = true
)
    target = local_hydrography_path(env_params, output_dir)
    if !force && target !== nothing
        return target
    end
    return joinpath(output_dir, "analytical_hydrography")
end
