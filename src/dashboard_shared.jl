"""
    src/dashboard_shared.jl

Shared domain models, analytical routines, and lakehouse persistence bridges
unifying interactive visualization across Pluto notebooks, native GLMakie windows,
and WGLMakie/Bonito browser dashboards.

Provides:
- Axis-Aligned Bounding Box (AABB) spatial exclusion geometry and spatial masks.
- 20-stage life-history cohort abundance mappings and conservation checks.
- Area of Interest (AOI) regional hydrography, bathymetry, and HSI calculations.
- Lakehouse data persistence (Parquet for demographic series, Zarr for rasters).
- Scenario configuration serialization to reproducible TOML files.
"""

module DashboardShared

using Dates
using DataFrames
using TOML
using Statistics
using Printf
using Tethys

export ExclusionBox, is_in_exclusion_box, is_in_exclusion_zones
export DEFAULT_STAGE_DISTRIBUTION, map_stage_distribution_to_cohorts
export aoi_bottom_temperature, aoi_habitat_suitability, aoi_domain_bounds
export export_scenario_toml, lakehouse_tethysbio_path

"""
    ExclusionBox

Axis-Aligned Bounding Box (AABB) defining a spatial no-take marine reserve
or fishing exclusion zone on the 2D simulation grid.

# Fields
- `id::Int`: Numeric identifier.
- `name::String`: Descriptive zone label.
- `xmin::Float64`: Minimum grid-x coordinate.
- `xmax::Float64`: Maximum grid-x coordinate.
- `ymin::Float64`: Minimum grid-y coordinate.
- `ymax::Float64`: Maximum grid-y coordinate.
- `active::Bool`: Flag indicating whether spatial protection is active.
"""
Base.@kwdef mutable struct ExclusionBox
    id::Int = 1
    name::String = "MPA-1"
    xmin::Float64 = 0.0
    xmax::Float64 = 0.0
    ymin::Float64 = 0.0
    ymax::Float64 = 0.0
    active::Bool = true
end

"""
    is_in_exclusion_box(x::Real, y::Real, box::ExclusionBox) -> Bool

Evaluate whether coordinate `(x, y)` lies within the bounds of `box` with
an `O(1)` bounds comparison.
"""
@inline function is_in_exclusion_box(x::Real, y::Real, box::ExclusionBox)::Bool
    if !box.active
        return false
    end
    x_min = min(box.xmin, box.xmax)
    x_max = max(box.xmin, box.xmax)
    y_min = min(box.ymin, box.ymax)
    y_max = max(box.ymin, box.ymax)
    return (x_min <= x <= x_max) && (y_min <= y <= y_max)
end

"""
    is_in_exclusion_zones(x::Real, y::Real, boxes::AbstractVector{ExclusionBox}) -> Bool

Evaluate whether coordinate `(x, y)` falls within any active exclusion box.
"""
function is_in_exclusion_zones(
    x::Real,
    y::Real,
    boxes::AbstractVector{ExclusionBox}
)::Bool
    for box in boxes
        if is_in_exclusion_box(x, y, box)
            return true
        end
    end
    return false
end

"""
    DEFAULT_STAGE_DISTRIBUTION

Standard 20-stage baseline abundance distribution for *Chionoecetes opilio*:
Stages 0–3: Brooded eggs and pelagic larvae (aggregate cohorts).
Stages 4–8: Early benthic juveniles (aggregate cohorts).
Stages 9–14: Male benthic and terminal instars.
Stages 15–19: Female benthic and terminal instars.
"""
const DEFAULT_STAGE_DISTRIBUTION = Float64[
    300.0, 150.0, 120.0, 100.0,
    80.0,  70.0,  60.0,  50.0,  45.0,
    40.0,  35.0,  30.0,  25.0,  20.0,  15.0,
    40.0,  35.0,  30.0,  25.0,  20.0
]

"""
    map_stage_distribution_to_cohorts(stg_vals; juvenile_max_stage = 12)

Partition a 20-stage abundance vector into sex-specific adult cohort vectors
and the unindividuated juvenile aggregate cohort vector.

# Arguments
- `stg_vals::AbstractVector{<:Real}`: 20-stage abundance counts.
- `juvenile_max_stage::Int`: Stage boundary separating aggregate cohorts from
  individuated agents.

# Returns
NamedTuple `(aggregate, male, female)` containing the allocated cohort arrays.
"""
function map_stage_distribution_to_cohorts(
    stg_vals::AbstractVector{<:Real};
    juvenile_max_stage::Int = 12
)
    n = length(stg_vals)
    male_cohort = zeros(Float64, n)
    female_cohort = zeros(Float64, n)

    for stg in 0:(n - 1)
        if stg >= juvenile_max_stage
            male_cohort[stg + 1] = Float64(stg_vals[stg + 1])
            if stg <= 16
                female_cohort[stg + 1] = Float64(stg_vals[stg + 1])
            end
        end
    end

    agg_len = min(juvenile_max_stage, n)
    aggregate_cohort = Float64[Float64(stg_vals[i]) for i in 1:agg_len]

    return (
        aggregate = aggregate_cohort,
        male = male_cohort,
        female = female_cohort
    )
end

"""
    aoi_bottom_temperature(depth::Real, lat::Real) -> Float64

Evaluate the climatological ambient bottom temperature on the Scotian Shelf
as a function of bathymetric depth `depth` (m) and latitude `lat` (degrees N):

```math
T_{\\text{bottom}} = 2.0 + 2.5 \\exp\\left(-\\left(\\frac{z - 150.0}{60.0}\\right)^2\\right) + 0.3 (45.0 - \\phi)
```
"""
function aoi_bottom_temperature(depth::Real, lat::Real)::Float64
    z = Float64(depth)
    phi = Float64(lat)
    t = 2.0 + 2.5 * exp(-((z - 150.0) / 60.0)^2) + 0.3 * (45.0 - phi)
    return t
end

"""
    aoi_habitat_suitability(temp::Real, depth::Real; mult = 1.0, offset = 0.0) -> Float64

Evaluate realized Habitat Suitability Index (HSI) modulated by user steering
multipliers and offsets.
"""
function aoi_habitat_suitability(
    temp::Real,
    depth::Real;
    mult::Real = 1.0,
    offset::Real = 0.0
)::Float64
    # Optimum thermal window for snow crab is 0 to 4 °C
    t_opt = exp(-0.5 * ((Float64(temp) - 2.0) / 2.2)^2)
    # Optimum depth window is 60 to 280 m
    z_opt = exp(-0.5 * ((Float64(depth) - 150.0) / 75.0)^2)
    raw_hsi = t_opt * z_opt
    realized = raw_hsi * Float64(mult) + Float64(offset)
    if realized < 0.0
        return 0.0
    elseif realized > 1.0
        return 1.0
    end
    return realized
end

"""
    aoi_domain_bounds(lon::Real, lat::Real; span_meters = 1000.0)

Compute geographic bounding boxes `(lon_range, lat_range)` centered at `(lon, lat)`
for a planar span of `span_meters`.
"""
function aoi_domain_bounds(
    lon::Real,
    lat::Real;
    span_meters::Real = 1000.0
)
    half_m = Float64(span_meters) / 2.0
    lat_deg_len = 111139.0
    lon_deg_len = max(1000.0, 111139.0 * cos(deg2rad(Float64(lat))))
    dlat = half_m / lat_deg_len
    dlon = half_m / lon_deg_len
    return (
        lon = [Float64(lon) - dlon, Float64(lon) + dlon],
        lat = [Float64(lat) - dlat, Float64(lat) + dlat]
    )
end

"""
    lakehouse_tethysbio_path(subpaths...) -> String

Resolve a path relative to the primary TethysBio lakehouse namespace:
`projects/lakehouse/TethysBio/`.
"""
function lakehouse_tethysbio_path(subpaths...)::String
    return normpath(joinpath(Tethys.lakehouse_dir("TethysBio"), subpaths...))
end

"""
    export_scenario_toml(path::AbstractString, cfg::AbstractDict;
                         aoi_coords = nothing,
                         exclusion_boxes = ExclusionBox[],
                         stage_distribution = nothing)

Serialize an active in-memory simulation scenario into a reproducible TOML file.
"""
function export_scenario_toml(
    path::AbstractString,
    cfg::AbstractDict;
    aoi_coords::Union{Nothing, NamedTuple, AbstractDict} = nothing,
    exclusion_boxes::AbstractVector{ExclusionBox} = ExclusionBox[],
    stage_distribution::Union{Nothing, AbstractVector{<:Real}} = nothing
)
    mkpath(dirname(abspath(path)))
    out = Dict{String, Any}()

    out["scenario"] = Dict{String, Any}(
        "created" => string(Dates.now()),
        "description" => "Exported TethysBio interactive simulation scenario",
        "version" => "1.0"
    )

    # Ingest baseline sections
    for sec in ["environment", "simulation", "fishery", "toggles", "organism"]
        if haskey(cfg, sec)
            out[sec] = copy(cfg[sec])
        elseif haskey(cfg, Symbol(sec))
            out[sec] = copy(cfg[Symbol(sec)])
        end
    end

    if aoi_coords !== nothing
        out["aoi"] = Dict{String, Any}(
            "lon" => Float64(get(aoi_coords, :lon, -63.5)),
            "lat" => Float64(get(aoi_coords, :lat, 44.5))
        )
    end

    if !isempty(exclusion_boxes)
        boxes_list = Vector{Dict{String, Any}}()
        for b in exclusion_boxes
            push!(boxes_list, Dict{String, Any}(
                "id" => b.id,
                "name" => b.name,
                "xmin" => b.xmin,
                "xmax" => b.xmax,
                "ymin" => b.ymin,
                "ymax" => b.ymax,
                "active" => b.active
            ))
        end
        out["spatial_exclusion"] = Dict{String, Any}("boxes" => boxes_list)
    end

    if stage_distribution !== nothing
        out["initial_stage_distribution"] = Float64.(stage_distribution)
    end

    open(path, "w") do io
        TOML.print(io, out)
    end
    return path
end

end # module DashboardShared

