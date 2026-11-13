# =============================================================================
# scripts/fetch_environmental.jl
#
# One-time, re-runnable acquisition of the regional environmental datasets used
# by every TethysBio frontend (GLMakie GUI, HTML dashboard, Pluto notebook):
#
#   1. ETOPO 2022 15-arc-second (~450 m) bathymetry for the full Scotian Shelf
#      domain (lon -66.5 .. -57.5, lat 42.0 .. 48.5), stored as
#      `data/environmental/bathymetry.zarr` (replacing the earlier 2 deg x 2 deg
#      subset which covered only Emerald Basin).
#   2. WOA23 monthly (12 months) 0.25 deg temperature climatology subset,
#      stored as `data/environmental/woa23_temperature_monthly_0.25deg.zarr`.
#      The annual store `woa23_temperature_00_0.25deg.zarr` is kept untouched;
#      the monthly store adds the seasonal cycle.
#   3. High-resolution coastline polylines (Northwest Atlantic, clipped to the
#      map domain), stored as `data/environmental/coastline.jld2`.
#
# Usage:
#   julia --project=. scripts/fetch_environmental.jl [--force] [--check]
#       [--bathymetry] [--temperature] [--coastline]
#
# Parts are idempotent: each is skipped when its target store already exists
# unless --force is given. Network sources:
#   - ETOPO: coastwatch.pfeg.noaa.gov ERDDAP griddap (subset request)
#   - WOA23: www.ncei.noaa.gov direct file server (global files, subset locally)
#   - Coastline: WKB polylines parsed from the lakehouse GeoOceanModel inputs
# =============================================================================
using Dates
using Downloads
using NCDatasets
using Zarr
using Parquet2
using DataFrames
using JLD2

const DATA_DIR = normpath(joinpath(@__DIR__, "..", "data", "environmental"))
const BATHY_PATH = joinpath(DATA_DIR, "bathymetry.zarr")
const MONTHLY_PATH = joinpath(DATA_DIR, "woa23_temperature_monthly_0.25deg.zarr")
const COASTLINE_PATH = joinpath(DATA_DIR, "coastline.jld2")

const DOMAIN_LON = (-66.5, -57.5)
const DOMAIN_LAT = (42.0, 48.5)

const BATHY_LEGACY_BACKUP = joinpath(DATA_DIR, "bathymetry_legacy_2deg_subset.bak")

# -----------------------------------------------------------------------------
# Small JSON writers (zarr v2 metadata; kept dependency-free)
# -----------------------------------------------------------------------------
write_group_metadata(out::AbstractString, attrs::Dict{String, Any}) = begin
    mkpath(out)
    write(joinpath(out, ".zgroup"), "{\"zarr_format\":2}")
    open(joinpath(out, ".zattrs"), "w") do io
        print(io, "{\"")
        is_first = true
        for (k, v) in sort(collect(pairs(attrs)); by = first)
            is_first || print(io, "\",\"")
            is_first = false
            v isa AbstractString || (v = string(v))
            print(io, k, "\":\"", replace(String(v), "\"" => "\\\""))
        end
        is_first || print(io, "\"")
        print(io, "}")
    end
end

# -----------------------------------------------------------------------------
# 1. Regional bathymetry
# -----------------------------------------------------------------------------
# The pre-2026-11 store was a 2 deg x 2 deg legacy subset (lon axis of 481
# cells). Treat any store whose lon axis is smaller than the regional grid as
# stale so it gets replaced even without --force.
bathy_store_is_regional() = begin
    meta = joinpath(BATHY_PATH, "lon", ".zarray")
    isfile(meta) || return false
    m = match(r"\"shape\":\[(\d+)", read(meta, String))
    m === nothing && return false
    parse(Int, m.captures[1]) >= 1000
end

function fetch_regional_bathymetry(; force::Bool = false, verbose::Bool = true)
    if isdir(BATHY_PATH) && (force || bathy_store_is_regional())
        verbose && println("[bathymetry] regional store exists, skipping")
        return
    end
    mkpath(DATA_DIR)

    if isdir(BATHY_PATH)
        if isfile(joinpath(BATHY_PATH, "lon", ".zarray"))
            verbose && println("[bathymetry] backing up previous store -> $(basename(BATHY_LEGACY_BACKUP))")
            rm(BATHY_LEGACY_BACKUP; recursive = true, force = true)
            mv(BATHY_PATH, BATHY_LEGACY_BACKUP)
        else
            verbose && println("[bathymetry] removing partial store from an earlier failed run")
            rm(BATHY_PATH; recursive = true, force = true)
        end
    end

    url = "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc?" *
          "z[($(DOMAIN_LAT[1])):($(DOMAIN_LAT[2]))][($(DOMAIN_LON[1])):($(DOMAIN_LON[2]))]"
    tmp_nc = joinpath(tempdir(), "tethysbio_etopo_regional.nc")
    if isfile(tmp_nc)
        verbose && println("[bathymetry] cached subset found ($(round((stat(tmp_nc).size / 2^20), digits = 1)) MiB), reusing")
    else
        verbose && println("[bathymetry] downloading ETOPO 2022 15-arcsec regional subset ...")
        downloaded = false
        for attempt in 1:4
            try
                Downloads.download(url, tmp_nc; timeout = 600)
                downloaded = true
                break
            catch err
                verbose && println("[bathymetry] attempt $attempt failed: $(sprint(show, err)[1:min(end, 120)])")
                attempt < 4 && sleep(30 * attempt)
            end
        end
        downloaded || error("[bathymetry] ETOPO download failed after retries")
    end

    verbose && println("[bathymetry] reading netCDF subset ...")
    lat_axis = Float64[]
    lon_axis = Float64[]
    z_lonlat = Matrix{Float64}(undef, 0, 0)
    NCDataset(tmp_nc, "r") do dsv
        lat_axis = Float64.(dsv["latitude"][:])
        lon_axis = Float64.(dsv["longitude"][:])
        zvar = dsv["z"]
        zdims = NCDatasets.dimnames(zvar)
        raw = Float64.(replace(Array(zvar[:, :]), missing => NaN))
        if zdims == ("latitude", "longitude")
            z_lonlat = permutedims(raw)          # -> (lon, lat)
        elseif zdims == ("longitude", "latitude")
            z_lonlat = raw                        # already (lon, lat)
        else
            error("[bathymetry] unexpected z dims $zdims")
        end
    end
    # Keep tmp_nc as a rerun cache; it lives in the OS temp dir.
    nx, ny = length(lon_axis), length(lat_axis)
    @assert size(z_lonlat) == (nx, ny) "unexpected z shape $(size(z_lonlat))"

    verbose && println("[bathymetry] grid: $(nx) x $(ny) cells ($(nx*ny) points) over lon [$(lon_axis[1]), $(lon_axis[end])] lat [$(lat_axis[1]), $(lat_axis[end])]")
    @assert lon_axis[1] <= DOMAIN_LON[1] + 0.01 && lon_axis[end] >= DOMAIN_LON[2] - 0.01 "longitude coverage too small"
    @assert lat_axis[1] <= DOMAIN_LAT[1] + 0.01 && lat_axis[end] >= DOMAIN_LAT[2] - 0.01 "latitude coverage too small"

    write_group_metadata(BATHY_PATH, Dict{String, Any}(
        "title" => "ETOPO_2022_v1_15s regional subset (Scotian Shelf)",
        "institution" => "NOAA NCEI",
        "sourceUrl" => "https://coastwatch.pfeg.noaa.gov/erddap/griddap/ETOPO_2022_v1_15s.nc",
        "geospatial_lon_resolution" => 0.004166666666666667,
        "geospatial_lat_resolution" => 0.004166666666666667,
        "history" => "converted to Zarr $(Dates.format(Dates.now(), "yyyy-mm-dd")) by TethysBio scripts/fetch_environmental.jl",
    ))

    za_lon = Zarr.zcreate(Float64, nx; name = "lon", path = BATHY_PATH, chunks = (nx,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["lon"]))
    za_lon[:] = lon_axis
    za_lat = Zarr.zcreate(Float64, ny; name = "lat", path = BATHY_PATH, chunks = (ny,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["lat"]))
    za_lat[:] = lat_axis
    za_z = Zarr.zcreate(Float64, nx, ny; name = "z", path = BATHY_PATH, chunks = (240, 240),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["lon", "lat"], "long_name" => "Z", "ioos_category" => "Location"))
    za_z[:, :] = z_lonlat

    verbose && println("[bathymetry] verifying store ...")
    g = Zarr.zopen(BATHY_PATH, "r")
    @assert length(g["lon"]) == nx && length(g["lat"]) == ny
    @assert size(g["z"]) == (nx, ny)
    sample = g["z"][div(nx, 2), div(ny, 2)]
    verbose && println("[bathymetry] OK (centre-cell elevation = $(round(sample, digits = 1)) m)")
    return BATHY_PATH
end

# -----------------------------------------------------------------------------
# 2. WOA23 monthly temperature (bottom-temperature seasonal cycle)
# -----------------------------------------------------------------------------
const WOA_SUBSET_LON = (-67.0, -57.0)
const WOA_SUBSET_LAT = (41.5, 49.0)

# A store left behind by an earlier failed run must not be treated as current:
# require the t_an array to declare a 12-long time axis and hold every chunk.
function monthly_store_is_complete()
    meta = joinpath(MONTHLY_PATH, "t_an", ".zarray")
    isfile(meta) || return false
    txt = read(meta, String)
    sh = match(r"\"shape\":\[([^\]]*)\]", txt)
    ch = match(r"\"chunks\":\[([^\]]*)\]", txt)
    (sh === nothing || ch === nothing) && return false
    dims = parse.(Int, split(sh.captures[1], ","))
    chunks = parse.(Int, split(ch.captures[1], ","))
    (length(dims) == 4 && dims[1] == 12) || return false
    n_expected = prod(ceil.(Int, dims ./ chunks))
    actual = count(f -> !startswith(f, "."), readdir(joinpath(MONTHLY_PATH, "t_an")))
    return actual >= n_expected
end

function fetch_monthly_temperature(; force::Bool = false, verbose::Bool = true)
    if isdir(MONTHLY_PATH) && (force || monthly_store_is_complete())
        verbose && println("[temperature] store exists, skipping")
        return
    end
    mkpath(DATA_DIR)
    if isdir(MONTHLY_PATH)
        verbose && println("[temperature] removing partial/incomplete store")
        rm(MONTHLY_PATH; recursive = true, force = true)
    end

    months = 1:12
    stack = Vector{Array{Float32, 4}}(undef, 12)
    depth_axis = Float64[]
    lat_axis = Float64[]
    lon_axis = Float64[]

    tmpdir = joinpath(tempdir(), "tethysbio_woa23")
    mkpath(tmpdir)
    try
        for m in months
            name = "woa23_decav_t" * lpad(m, 2, "0") * "_04.nc"
            local_path = joinpath(tmpdir, name)
            if !isfile(local_path)
                url = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/0.25/" * name
                verbose && println("[temperature] month $m: downloading $name (global 0.25 deg) ...")
                ok = false
                for attempt in 1:3
                    try
                        Downloads.download(url, local_path; timeout = 1800)
                        ok = true
                        break
                    catch err
                        verbose && println("[temperature]   attempt $attempt failed: $(sprint(show, err)[1:min(end, 100)])")
                        attempt < 3 && sleep(20 * attempt)
                    end
                end
                ok || error("[temperature] download of $name failed")
            else
                verbose && println("[temperature] month $m: cached file found, reusing")
            end
            verbose && println("[temperature] month $m: reading + subsetting ...")
            NCDataset(local_path, "r") do dsv
                lat_var = haskey(dsv, "lat") ? dsv["lat"] : dsv["latitude"]
                lon_var = haskey(dsv, "lon") ? dsv["lon"] : dsv["longitude"]
                m === 1 && (depth_axis = Float64.(dsv["depth"][:]); lat_axis = Float64.(lat_var[:]); lon_axis = Float64.(lon_var[:]))

                i_lon = findall(x -> WOA_SUBSET_LON[1] <= x <= WOA_SUBSET_LON[2], lon_axis)
                i_lat = findall(x -> WOA_SUBSET_LAT[1] <= x <= WOA_SUBSET_LAT[2], lat_axis)

                t_an = dsv["t_an"]                       # WOA23: (lon, lat, depth, time)
                dims = NCDatasets.dimnames(t_an)
                @assert all(d in dims for d in ("time", "depth", "lat", "lon")) "unexpected t_an dims $dims"
                pos = Dict(d => i for (i, d) in enumerate(dims))
                sel = Any[Colon() for _ in dims]
                sel[pos["lon"]] = i_lon
                sel[pos["lat"]] = i_lat
                sel[pos["time"]] = 1:1
                block = t_an[sel...]                      # retained dims in file order, time = 1
                vals = Float32.(replace(block, missing => NaN32))
                fill_missing = isnan.(vals) .| (abs.(vals) .> 1e30)
                vals[fill_missing] .= Float32(NaN)
                # canonical (time, depth, lat, lon) regardless of file dim order
                stack[m] = permutedims(vals, (pos["time"], pos["depth"], pos["lat"], pos["lon"]))
            end
        end
    finally
        # Keep the temp directory (reuse as cache on re-run); not part of the repo.
    end

    ntime, ndep, nlat, nlon = size(stack[1])
    for m in 2:12
        @assert size(stack[m]) == (ntime, ndep, nlat, nlon) "month $m subset shape mismatch"
    end
    t_final = cat(stack...; dims = 1)                    # months along time -> (time, depth, lat, lon)
    @assert size(t_final) == (12, ndep, nlat, nlon) "unexpected t_final shape $(size(t_final))"

    write_group_metadata(MONTHLY_PATH, Dict{String, Any}(
        "title" => "WOA23 decadal-averaged monthly temperature 0.25 deg (regional subset)",
        "institution" => "NOAA NCEI",
        "sourceUrl" => "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/0.25/",
        "history" => "converted to Zarr $(Dates.format(Dates.now(), "yyyy-mm-dd")) by TethysBio scripts/fetch_environmental.jl",
    ))
    za_time = Zarr.zcreate(Float64, 12; name = "time", path = MONTHLY_PATH, chunks = (12,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["time"], "units" => "month index (1-12)"))
    za_time[:] = collect(1.0:12.0)
    za_depth = Zarr.zcreate(Float64, ndep; name = "depth", path = MONTHLY_PATH, chunks = (ndep,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["depth"], "units" => "m"))
    za_depth[:] = depth_axis
    za_lat = Zarr.zcreate(Float64, nlat; name = "lat", path = MONTHLY_PATH, chunks = (nlat,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["lat"], "units" => "degrees_north"))
    za_lat[:] = lat_axis[findall(x -> WOA_SUBSET_LAT[1] <= x <= WOA_SUBSET_LAT[2], lat_axis)]
    za_lon = Zarr.zcreate(Float64, nlon; name = "lon", path = MONTHLY_PATH, chunks = (nlon,),
        attrs = Dict("_ARRAY_DIMENSIONS" => ["lon"], "units" => "degrees_east"))
    za_lon[:] = lon_axis[findall(x -> WOA_SUBSET_LON[1] <= x <= WOA_SUBSET_LON[2], lon_axis)]
    za_t = Zarr.zcreate(Float32, 12, ndep, nlat, nlon; name = "t_an", path = MONTHLY_PATH,
        chunks = (1, 51, 15, 20), fill_value = NaN32,
        attrs = Dict("_ARRAY_DIMENSIONS" => ["time", "depth", "lat", "lon"], "units" => "degC"))
    za_t[:, :, :, :] = t_final

    verbose && println("[temperature] verifying store ...")
    g = Zarr.zopen(MONTHLY_PATH, "r")
    @assert size(g["t_an"]) == (12, ndep, nlat, nlon)
    s = g["t_an"][1, 1, div(nlat, 2), div(nlon, 2)]
    verbose && println("[temperature] OK (Jan surface sample = $(round(s, digits = 2)) degC)")
    return MONTHLY_PATH
end

# -----------------------------------------------------------------------------
# 3. Coastline polylines
# -----------------------------------------------------------------------------
const COASTLINE_DOMAIN_MARGIN_LON = 0.7
const COASTLINE_DOMAIN_MARGIN_LAT = 0.7

function coastline_sources()
    candidates = [
        "C:/Users/Nathalie/projects/lakehouse/GeoOceanModel/scotian_shelf/inputs/coastline.parquet",
        "C:/Users/Nathalie/projects/lakehouse/GeoOceanModel/default/inputs/coastline.parquet",
        joinpath(DATA_DIR, "coastline.parquet"),
    ]
    return filter(isfile, candidates)
end

wkb_le32(buf, off) = reinterpret(Int32, view(buf, off:(off + 3)))[1]

function wkb_walk!(lines, buf::AbstractVector{UInt8}, off::Int)
    buf[off] == 0x01 || error("unexpected WKB byte order")
    gtype = wkb_le32(buf, off + 1)
    if gtype == 2                                  # LineString
        n = wkb_le32(buf, off + 5)
        push!(lines, reinterpret(Float64, view(buf, (off + 9):(off + 8 + 16 * n))))
        return off + 9 + 16 * n
    elseif gtype == 3                              # Polygon (rings as closed lines)
        nrings = wkb_le32(buf, off + 5)
        off2 = off + 9
        for _ in 1:nrings
            n = wkb_le32(buf, off2)
            push!(lines, reinterpret(Float64, view(buf, (off2 + 5):(off2 + 4 + 16 * n))))
            off2 += 4 + 16 * n
        end
        return off2
    elseif gtype in (5, 6)                         # MultiLineString / MultiPolygon
        n = wkb_le32(buf, off + 5)
        off2 = off + 9
        for _ in 1:n
            off2 = wkb_walk!(lines, buf, off2)
        end
        return off2
    end
    error("unsupported WKB geometry type $gtype")
end

function fetch_coastline(; force::Bool = false, verbose::Bool = true)
    isfile(COASTLINE_PATH) && !force && (verbose && println("[coastline] file exists, skipping"); return)
    srcs = coastline_sources()
    isempty(srcs) && error("[coastline] no coastline.parquet source found")
    verbose && println("[coastline] parsing $(srcs[1]) ...")

    df = open(srcs[1], "r") do io
        DataFrame(Parquet2.Dataset(io))
    end

    raw = Vector{Vector{Float64}}()
    for geom in df.geometry
        flat = Vector{Vector{Float64}}()
        wkb_walk!(flat, geom, 1)
        for flatline in flat
            n = length(flatline) ÷ 2
            for k in 1:n
                push!(raw, [flatline[2k - 1], flatline[2k]])
            end
        end
    end
    verbose && println("[coastline] total vertices parsed: $(length(raw))")

    # Split into polylines clipped to the map domain (+ margin). Consecutive
    # vertices outside the box terminate the current polyline; segments are
    # re-started when the path re-enters, so no long artificial jumps occur.
    lon_lo = DOMAIN_LON[1] - COASTLINE_DOMAIN_MARGIN_LON
    lon_hi = DOMAIN_LON[2] + COASTLINE_DOMAIN_MARGIN_LON
    lat_lo = DOMAIN_LAT[1] - COASTLINE_DOMAIN_MARGIN_LAT
    lat_hi = DOMAIN_LAT[2] + COASTLINE_DOMAIN_MARGIN_LAT

    segs = Vector{Vector{NTuple{2, Float64}}}()
    current = NTuple{2, Float64}[]
    function flush!()
        length(current) >= 2 && push!(segs, current)
        current = NTuple{2, Float64}[]
    end
    function inside(p)
        lon_lo <= p[1] <= lon_hi && lat_lo <= p[2] <= lat_hi
    end
    for k in 1:(length(raw) - 1)
        p, q = raw[k], raw[k + 1]
        pin, qin = inside(p), inside(q)
        if pin
            push!(current, (p[1], p[2]))
        elseif qin
            # segment from outside into the box: clip p to the box edge
            push!(current, (p[1], p[2]))
        elseif !isempty(current)
            push!(current, (p[1], p[2]))
            flush!()
        end
        if pin && !qin
            push!(current, (q[1], q[2]))
            flush!()
        elseif !pin && qin
            flush_if_needed = nothing
        end
    end
    flush!()

    kept = sum(length(s) for s in segs)
    verbose && println("[coastline] polylines: $(length(segs))  vertices kept: $kept")

    JLD2.jldopen(COASTLINE_PATH, "w") do file
        write(file, "segs", segs)
        write(file, "domain_lon", collect(DOMAIN_LON))
        write(file, "domain_lat", collect(DOMAIN_LAT))
        write(file, "source", srcs[1])
    end
    verbose && println("[coastline] wrote $COASTLINE_PATH")
    return COASTLINE_PATH
end

# -----------------------------------------------------------------------------
# CLI driver
# -----------------------------------------------------------------------------
function main(args)
    force = "--force" in args
    check_only = "--check" in args
    parts = Dict(
        :bathymetry => "--bathymetry" in args || (count(s -> s in args, ("--bathymetry", "--temperature", "--coastline")) == 0),
        :temperature => "--temperature" in args || (count(s -> s in args, ("--bathymetry", "--temperature", "--coastline")) == 0),
        :coastline => "--coastline" in args || (count(s -> s in args, ("--bathymetry", "--temperature", "--coastline")) == 0),
    )
    println("== TethysBio environmental data fetcher $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")) ==")
    println("   data dir: $DATA_DIR")
    if check_only
        println("   bathymetry.zarr            exists: ", isdir(BATHY_PATH))
        println("   woa23_temperature_monthly  exists: ", isdir(MONTHLY_PATH))
        println("   coastline.jld2             exists: ", isfile(COASTLINE_PATH))
        return
    end
    parts[:bathymetry] && fetch_regional_bathymetry(; force = force)
    parts[:temperature] && fetch_monthly_temperature(; force = force)
    parts[:coastline] && fetch_coastline(; force = force)
    println("== done ==")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
