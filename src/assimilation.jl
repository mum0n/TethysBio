# ==============================================================================
#  assimilation.jl -- the environmental inputs stored with TethysBio
# ==============================================================================
#
#  An IBM needs a world to run in: a sea floor, a bottom temperature, and a seasonal cycle of
#  both. Those come from published products -- ETOPO 2022 bathymetry and a climatological
#  hydrography (WOA23 or GLORYS12V1) -- and they are the only external inputs this package has.
#
#  Until now they were fetched *inside* `build_geodata_environmental_fields`, once per
#  simulation, into a run-local cache directory. That made two things awkward: the 728 MB-class
#  products were re-fetched for every run that configured a different cache directory, and the
#  only record of what a run used was a path inside that run's output.
#
#  The separation this file makes:
#
#      a published product   ETOPO 2022, WOA23 over HTTP
#      TethysBio/data/       the package-local assimilated store
#      a run                 reads the local store and writes local results
#
#  Assimilation is idempotent, and it knows when to stop: a store artefact at least as new as
#  the fetch that produced it is skipped. `force = true` re-fetches.
#
#  The layout is one directory per input rather than one archive per run:
#
#      data/environmental/
#        bathymetry.zarr        the sea floor, from ETOPO 2022 through ERDDAP
#        hydrography/           the 12-month climatological temperature and salinity
#        manifest.toml          what was assimilated, from which product, when
#
#  Nothing is stored that is not read back: `read_assimilated_environment` is the reader, and
#  the same functions `build_geodata_environmental_fields` calls.
# ==============================================================================

"""
    TethysBio_store_root() -> String

Where TethysBio's data lives. Environmental inputs, caches, and results are kept under the
package's own `data` directory.
"""
function TethysBio_store_root()::String
    return tethysbio_data_path()
end

"""
    environmental_store_dir(env_params) -> String

The directory a run reads its environment from. Paths are resolved against the package root and
rejected if they escape it.
"""
function environmental_store_dir(env_params::EnvironmentParams)
    d = something(env_params.geodata_store_dir, "")
    isempty(strip(d)) && (d = env_params.geodata_input_dir)
    isempty(strip(d)) && (d = joinpath("data", "environmental"))
    return tethysbio_project_path(d)
end

"""
    environmental_input_status(env_params) -> NamedTuple

What this configuration will read, where each input's assimilated copy lives, and whether it
is already there. Read-only: nothing is fetched and nothing is written, so a caller can ask
before spending the time.

Present-ness is measured with `isdir`, because a Tethys store *is* a directory: a fetch that
wrote a Zarr store leaves one, and a caller that tested `isfile` would report every assimilated
input as missing.
"""
function environmental_input_status(env_params::EnvironmentParams)
    store = abspath(environmental_store_dir(env_params))
    lon_r = (Float64(env_params.domain_lon[1]), Float64(env_params.domain_lon[2]))
    lat_r = (Float64(env_params.domain_lat[1]), Float64(env_params.domain_lat[2]))
    bathy = local_bathymetry_path(store)
    bathy_present = bathy !== nothing
    inputs = @NamedTuple{role::Symbol, product::String, where::String, present::Bool,
                         note::String}[
        (role = :bathymetry,
         product = env_params.bathymetry_source,
         where = something(bathy, joinpath(store, "bathymetry.zarr")),
         present = bathy_present,
         note = "sea floor, depth as max(0, -elevation)"),
        (role = :hydrography,
         product = env_params.temperature_source,
         where = something(local_hydrography_path(env_params, store), store),
         present = _hydrography_present(store, env_params),
         note = "bottom and pelagic temperature, 12-month climatology"),
    ]
    return (
        store_dir = store,
        domain = (lon = lon_r, lat = lat_r, time = env_params.time_range),
        inputs = inputs,
        present = [i.role for i in inputs if i.present],
        absent = [i.role for i in inputs if !i.present],
    )
end

"""
    _hydrography_present(store, env_params) -> Bool

Whether the store holds the monthly hydrography this run needs. A single explicitly named month
is enough for a run that uses one; a run with `temperature_month = 0` uses all twelve, so all
twelve must be there.
"""
function _hydrography_present(store::AbstractString, env_params::EnvironmentParams)
    return local_hydrography_path(env_params, store) !== nothing
end



"""
    assimilate_environmental_inputs(env_params; verbose = true, force = false) -> NamedTuple

Fetch the configured environmental products into TethysBio's package-local data directory.

Returns `(store_dir, written, skipped, manifest)`.
"""
function assimilate_environmental_inputs(env_params::EnvironmentParams;
                                        verbose::Bool = true,
                                        force::Bool = false)
    store = abspath(environmental_store_dir(env_params))
    mkpath(store)
    lon_bounds = env_params.domain_lon
    lat_bounds = env_params.domain_lat
    lon_range = (Float64(lon_bounds[1]), Float64(lon_bounds[2]))
    lat_range = (Float64(lat_bounds[1]), Float64(lat_bounds[2]))
    temp_src = Symbol(env_params.temperature_source)

    written = String[]
    skipped = String[]
    before = environmental_input_status(env_params)
    if !force && isempty(before.absent)
        verbose && println("  [assimilate] the store at $(store) is up to date")
    end

    # -- bathymetry ------------------------------------------------------------
    bathy_target = local_bathymetry_path(store)
    if force || bathy_target === nothing
        bathy_target = fetch_local_bathymetry!(
            env_params, store; force = force, verbose = verbose
        )
        push!(written, "bathymetry")
    else
        push!(skipped, "bathymetry")
        verbose && println("  [assimilate] bathymetry: already in the store, skipped")
    end

    # -- hydrography -----------------------------------------------------------
    hydro_target = local_hydrography_path(env_params, store)
    if force || hydro_target === nothing
        hydro_target = fetch_local_hydrography!(
            env_params, store; force = force, verbose = verbose
        )
        push!(written, "hydrography")
    else
        push!(skipped, "hydrography")
        verbose && println("  [assimilate] hydrography: already in the store, skipped")
    end

    manifest = Dict{String, Any}(
        "assimilated_utc" => string(now(Dates.UTC)),
        "store_root" => store,
        "domain" => Dict("lon" => [lon_range[1], lon_range[2]],
                         "lat" => [lat_range[1], lat_range[2]]),
        "inputs" => [
            Dict("role" => "bathymetry", "product" => env_params.bathymetry_source,
                 "where" => bathy_target),
            Dict("role" => "hydrography", "product" => string(temp_src),
                 "where" => hydro_target, "months" => [env_params.temperature_month],
                 "domain" => Dict("lon" => [lon_range[1], lon_range[2]],
                                  "lat" => [lat_range[1], lat_range[2]])),

        ],
    )
    open(joinpath(store, "manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    verbose && println("  [assimilate] manifest.toml written -> $(store)")
    return (store_dir = store, written = written, skipped = skipped,
            manifest = joinpath(store, "manifest.toml"))
end

"""
    read_assimilated_environment(env_params) -> (; depth, temperature, monthly, hydro)

The assimilated store, read back the same way a simulation reads it.

`depth` is the bathymetry field on the configured grid, `temperature` the bottom and pelagic
field for the configured month, and `monthly` the twelve-month series. Calling this after
`assimilate_environmental_inputs` is the reader side of the same contract: one directory, and
no knowledge of where any product was published.
"""
function read_assimilated_environment(env_params::EnvironmentParams)
    # `build_geodata_environmental_fields` already falls back to the providers
    # (`load_dataset`) when an assimilated artefact is absent, so a missing store is not
    # fatal: it is fetched on demand. The strict presence check that used to
    # live here made the reader fail immediately after `assimilate_environmental_inputs`
    # for hydrography, because the two disagreed on the store's layout (see C2).
    return build_geodata_environmental_fields(env_params, (env_params.grid_nx, env_params.grid_ny))
end
