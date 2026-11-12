"""
    src/gui.jl

Interactive Graphical User Interface (GUI) and real-time visualization
dashboard for the organism Individual-Based Model (IBM).

Provides three interlinked dashboard interfaces:
1. Parameter inputs/controls, real-time spatial environmental maps,
   and dynamic population time-series trajectories.
2. Demographic size structures (length frequencies by sex, instar stages)
   and real-time fitting of macro-ecological dynamics models
   (Schaefer logistic surplus production, Deriso-Schnute delay-difference,
   and Generalized Leslie/Lefkovitch transition matrix).
3. Realized process functional relationships (empirical maturity ogives,
   fishery selectivity, bioenergetic thermal response, and stock-recruitment)
   plotted against theoretical and empirical drivers.

Built upon `GLMakie` and `Agents.jl` reactive Observables.
"""

import GLMakie
import GLMakie: Figure, Axis, SliderGrid, Toggle, Button, Menu, Label, GridLayout
import GLMakie: Point2f, Observable, on, lift, @lift
using DataFrames
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics
using Agents

"""
    create_empty_annual_data()

Initialize an empty DataFrame schema for annual demographic and fishery metrics
used in macro-model fitting and biomass dynamics tracking.

Built from `ANNUAL_PRODUCTION_SCHEMA` so the GUI's annual frame carries exactly
the columns `extract_annual_production` produces, plus the two GUI-only tracking
columns. The two were separate hand-written lists, so the GUI frame was missing
`day`, `total_biomass_kg` and `annual_egg_production` and could not be fed to the
macro-model fitters without a missing-column failure.
"""
function create_empty_annual_data()
    df = empty_annual_production()
    df.recruits = Float64[]
    df.recruits_male = Float64[]
    df.cumulative_catch = Float64[]
    # Realised mean wet weight per component, in kg. Recorded so the
    # number-to-biomass conversion applied to the macro-model trajectories is
    # carried with the data rather than being implicit in the plotting code.
    df.mean_weight_legal_kg = Float64[]
    df.mean_weight_male_kg = Float64[]
    df.mean_weight_female_kg = Float64[]
    df.mean_weight_total_kg = Float64[]
    return df
end

# ------------------------------------------------------------------------------
# Plot-scale conventions
# ------------------------------------------------------------------------------

"""
    COUNT_PLOT_FLOOR

Smallest count any log-scaled dashboard axis will draw.

`log10(0)` is `-Inf`, which no axis can place, and zero is a real value here
rather than a missing one: a stage with no survivors, a clutch that has not been
laid yet, a cohort that has already individuated. Flooring the count before the
log keeps the transform finite and keeps the axis monotonic in the count.

One individual is the floor, so a plotted value of `0` reads as "at most one",
which is the honest statement about a sub-unit count. It is deliberately not
`0.1`: that would imply the model resolves fractions of an animal, and it puts a
spurious decade between "extinct" and "present".
"""
const COUNT_PLOT_FLOOR = 1.0

"""
    log10_count(n::Real)

`log10` of a count, floored at [`COUNT_PLOT_FLOOR`](@ref) so that an empty stage
plots at a finite position instead of `-Inf`.

Applied at *draw* time rather than at record time. The history buffers hold raw
counts, so a run can still be exported, refitted or re-plotted on a linear axis
without having lost the numbers.
"""
log10_count(n::Real) = log10(max(n, COUNT_PLOT_FLOOR))

"""
    GUI_EARLY_STAGES

The aggregate cohort stages tracked as a time series on the stage figure.

These are `AggregateGroup` cohorts, not `Organism` agents, and `AggregateGroup`
carries no `sex` field, so they are drawn as a single undifferentiated series.

Starts at `FIRST_COHORT_STAGE`, not 0: stage 0 is the brooded egg, which is
carried by a female rather than being an `AggregateGroup`, so including it left a
series that was permanently zero and made the panel look as though the pelagic
stages were never populated.
"""
const GUI_EARLY_STAGES = FIRST_COHORT_STAGE:3

# Early stage labels, line palettes, and dash patterns are retrieved dynamically
# from `OrganismParams.stage_labels` via `cohort_stage_label` to ensure exact
# correspondence with active TOML parameter files. Undifferentiated aggregate cohorts
# are visualized with balanced sex allocation.

"""
    RESCAN_IMMUTABLE

Parameters that describe the *shape* of a run rather than its dynamics, and so
cannot be changed on a live model: the grid extent, agent count and time step are
baked into the `StandardABM` when it is constructed, and `seed` only has meaning
at construction. Applying a new `grid_nx` to a running model would leave every
agent's position outside the new space.

[`rescan_config!`](@ref) reports these as requiring a restart instead of applying
them.
"""
const RESCAN_IMMUTABLE = (:grid_nx, :grid_ny, :grid_size, :dt, :seed)

"""
    CONTROL_PANEL_WIDTH

Width of the control column, in pixels. Named because the figure's three columns
are sized from it — see the `fig.layout.colsizes` assignment in
`launch_organism_gui`.

Sized to the panel's own content. The widest row is a steering label beside its
text box ("Restore Year:" against a 90 px menu), which needs a little over 340 px;
at 340 the column overflowed and clipped its left-hand labels, since a grid wider
than its slot spills symmetrically about its centre.
"""
const CONTROL_PANEL_WIDTH = 400

"""
    DASHBOARD_SIZE

Window size in pixels.

920 tall, which is 80% of the 1150 the balanced layout needed. The width is not a
free choice: every plotting panel is the same width, so the window is exactly
`CONTROL_PANEL_WIDTH + 2 x (height / 2) + 2 x COLUMN_GAP` =
400 + 2 x 460 + 40 = 1360. Leaving the previous 1680 would have put 320 px of empty
margin down the right-hand side.

The height is what sets the panel width: the plot columns are half the height wide
so the square map fills its column, and the four panels stacked down the right take
three rows each.
"""
const DASHBOARD_SIZE = (1360, 920)

"""
    SEX_DENSITY_ALPHA

Fill opacity of every kernel density curve in the size-frequency panel.

One value for all four series. The mature subsets are the same kind of thing as the
whole-sex curves and overlay them, so they share their transparency; when the mature
curves were opaque they read as a different kind of quantity rather than as a subset.
They stay distinguishable by hue — slate grey and dark orange against the whole-sex
blue and red — and by the heavier stroke.
"""
const SEX_DENSITY_ALPHA = 0.45

"""
    SIZE_DENSITY_BANDWIDTH_MM

Kernel bandwidth for the size-frequency panel, in millimetres.

One value for all four series. A per-series bandwidth is chosen from each sample's
own spread and size, so the mature series — a few tens of individuals in a narrow
size range — would be smoothed far more finely than the whole-sex series and drawn
as a spike beside it. A common bandwidth puts all four on the same footing, which is
what makes them comparable.
"""
const SIZE_DENSITY_BANDWIDTH_MM = 3.0

"""
    kde_density(sizes, grid; bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale = :probability)

Gaussian kernel density of `sizes` over `grid`.

`scale = :probability` returns a density integrating to one, so curves from groups
of very different sizes are directly comparable on a linear axis — each is read as
a shape. `scale = :count` returns individuals per millimetre, whose area is the
group size, so a subset reads as lying inside its parent.

Non-finite sizes are dropped rather than contributing a spike, and an empty sample
returns all zeros — no placeholder value is substituted, since a single fake
observation at 0 mm would draw a phantom cohort at the origin.
"""
function kde_density(sizes, grid;
    bandwidth = SIZE_DENSITY_BANDWIDTH_MM, scale::Symbol = :probability)
    scale in (:probability, :count) ||
        error("scale must be :probability or :count, got $scale")
    vals = [Float64(s) for s in sizes if isfinite(s)]
    n = length(vals)
    out = zeros(Float64, length(grid))
    n == 0 && return out
    h = Float64(bandwidth)
    h > 0 || error("bandwidth must be positive, got $h")
    k = 1 / (h * sqrt(2 * pi))
    for s in vals
        @inbounds for (i, x) in enumerate(grid)
            z = (x - s) / h
            out[i] += k * exp(-0.5 * z * z)
        end
    end
    # No extra scaling by `n` here: the loop already accumulates one kernel per
    # observation, so the integral is already the group size and each peak is `n`
    # times a single observation's peak. Multiplying again made it `n²`.
return scale === :probability ? out ./ n : out
end

"""
    PELAGIC_STAGE_LABELS

Names of the pelagic cohort stages, in order: zoea1, zoea2 and the semibenthic
megalopa. Supplied by `OrganismParams.stage_labels` in the configuration file for
the figures; kept here for the tests and for anything that needs the names without
a parameter object.
"""
const PELAGIC_STAGE_LABELS = ("z1", "z2", "m")

# `STAGE_UNSEXED_COLOUR` used to sit here, a grey for the stages that carry no
# sex. Every stage from the egg to the first individuated one is now drawn as a
# 50/50 male/female pair, so nothing on the stage figure is unsexed and the
# constant has no remaining use.

"""
    cohort_stage_label(stage, params)

Plot name for a cohort stage, from `OrganismParams.stage_labels` in the
configuration file.

`stage_labels` is in stage order — element `i` is model stage `i - 1` — so the
names live in the TOML rather than being derived here, which is what keeps the
figure from disagreeing with the file. A stage past the end of the declared list,
or a species that declares none, falls back to the derived name: a benthic instar
number counting from one at `larval_stage_count`, which is the convention the
default list encodes.
"""
function cohort_stage_label(stage::Integer, params, larval_count::Integer)
    labels = params.organism.stage_labels
    idx = stage + 1
    if 1 <= idx <= length(labels)
        return String(labels[idx])
    end
    return stage < larval_count ? string(stage) : string(stage - larval_count + 1)
end

"""
    stem!(ax, xs, ys; kwargs...)

A vertical stem from the axis up to a marker at each point — a lollipop glyph.

Used for the stage figure, where the quantity is a sum of discrete individuals
rather than an estimate. An earlier version drew a Poisson interval around each
point; that implied a precision a head count does not have, so the interval is gone
and the stem now carries the meaning: height, not precision.

The segments are drawn with `linesegments!` from explicit endpoints rather than
with `vlines!`. `vlines!` given an observable height vector drew every stem at full
axis height regardless of the value, which put a line through all eighteen stages
whatever their counts were. Explicit from/to points cannot drift that way: each
segment ends at its own point.

`linesegments!` takes one point vector and reads consecutive pairs as segments, so
the from/to points are interleaved here rather than passed as two vectors — that
two-vector form is not a conversion it accepts.
"""
function stem!(ax, xs::AbstractVector{<:Real}, ys; color = :black, markersize = 7,
    linewidth = 2.0, label = nothing)
    xs32 = Float32.(xs)
    if ys isa Observable
        segs = lift(ys) do v
            pts = Point2f[]
            for i in eachindex(xs32)
                push!(pts, Point2f(xs32[i], 0.0f0))
                push!(pts, Point2f(xs32[i], Float32(v[i])))
            end
            pts
        end
        tops = lift(ys) do v
            [Point2f(xs32[i], Float32(v[i])) for i in eachindex(xs32)]
        end
        linesegments!(ax, segs; color = color, linewidth = linewidth, label = label)
        scatter!(ax, tops; color = color, markersize = markersize)
    else
        pts = Point2f[]
        tops = Point2f[]
        for i in eachindex(xs32)
            push!(pts, Point2f(xs32[i], 0.0f0))
            push!(pts, Point2f(xs32[i], Float32(ys[i])))
            push!(tops, Point2f(xs32[i], Float32(ys[i])))
        end
        linesegments!(ax, pts; color = color, linewidth = linewidth, label = label)
        scatter!(ax, tops; color = color, markersize = markersize)
    end
end

# ------------------------------------------------------------------------------
# Shared Dashboard Panel Setup Kernels (Common across GUI and Notebooks)
# ------------------------------------------------------------------------------

"""
    set_grid_visibility!(gl::GridLayout, vis::Bool)

Recursively set visibility for all blocks and sub-grids within a `GridLayout`.
"""
function set_grid_visibility!(gl::GridLayout, vis::Bool)
    for c in gl.content
        item = c.content
        if item isa GridLayout
            set_grid_visibility!(item, vis)
        elseif hasfield(typeof(item), :blockscene)
            item.blockscene.visible[] = vis
            if hasfield(typeof(item), :scene)
                item.scene.visible[] = vis
            end
        end
    end
    return nothing
end

"""
    setup_spatial_panel!(ax, env_field, xs, ys, sizes, colors)

Populate spatial thermal environment heatmap with `:thermal` colormap and
superimposed demographic agent particles. Accepts either Observables or Arrays.
"""
function setup_spatial_panel!(
    ax,
    env_field,
    xs,
    ys,
    sizes,
    colors;
    colormap = :thermal,
    colorrange = nothing,
    exclusion_lines = nothing,
    exclusion_rects = nothing
)
    cr = if colorrange !== nothing
        colorrange
    elseif env_field isa AbstractArray
        valid_vals = filter(!isnan, env_field)
        if isempty(valid_vals)
            (0.0, 1.0)
        else
            ext = extrema(valid_vals)
            ext[1] == ext[2] ? (ext[1] - 1.0, ext[2] + 1.0) : ext
        end
    else
        nothing
    end

    hm = if cr !== nothing
        heatmap!(ax, env_field; colormap = colormap, colorrange = cr, rasterize = 2)
    else
        heatmap!(ax, env_field; colormap = colormap, rasterize = 2)
    end

    if exclusion_rects !== nothing
        poly!(
            ax, exclusion_rects;
            color = RGBAf(1.0, 0.75, 0.0, 0.22),
            strokecolor = :darkorange,
            strokewidth = 2.0,
            linestyle = :dash
        )
    end

    pts = if xs isa Observable && ys isa Observable
        lift((x, y) -> [Point2f(x[i], y[i]) for i in 1:min(length(x), length(y))], xs, ys)
    elseif xs isa Observable
        xs
    else
        Point2f.(xs, ys)
    end

    scatter!(
        ax, pts;
        markersize = sizes,
        color = colors,
        strokewidth = 0.5,
        strokecolor = RGBAf(0.05f0, 0.05f0, 0.05f0, 0.85f0)
    )
    if exclusion_lines !== nothing
        linesegments!(ax, exclusion_lines; color = :crimson, linewidth = 2.5, linestyle = :dash)
    end
    return hm
end


"""
    setup_abundance_panel!(ax, ts_total, ts_females, ts_males, ts_legal)

Populate log10 abundance trajectories for Total, Mature Females, Mature Males,
and Legal Fishable Males. Accepts either Observables or Arrays.
"""
function setup_abundance_panel!(ax, ts_total, ts_females, ts_males, ts_legal)
    lines!(ax, ts_total, label = "Total Organisms", color = :royalblue, linewidth = 2.0)
    lines!(ax, ts_females, label = "Mature Females", color = :crimson, linewidth = 1.8)
    lines!(ax, ts_males, label = "Mature Males", color = :seagreen, linewidth = 1.8)
    lines!(ax, ts_legal, label = "Legal Fishable Males", color = :purple, linewidth = 2.0)
    return axislegend(
        ax, position = :rt, labelsize = 9, framevisible = true,
        framecolor = (:gray75, 0.6), backgroundcolor = (:white, 0.88),
        padding = (5, 5, 3, 3), margin = (6, 6, 6, 6), patchsize = (14, 8)
    )
end

"""
    setup_stages_panel!(ax, half_w, base_agg, base_agent, n_male_st, n_fem_st,
                        egg_y, agg_y, male_y, female_y)

Populate grouped stage abundance forest marks (`stem!`), splitting brooded egg
and early aggregate cohort stages 50/50 and individuated stages by sex.
"""
function setup_stages_panel!(ax, half_w, base_agg, base_agent, n_male_st, n_fem_st,
                             egg_y, agg_y, male_y, female_y)
    stem!(ax, [1.0 - half_w], egg_y; color = :royalblue, label = "Males")
    stem!(ax, [1.0 + half_w], egg_y; color = :crimson, label = "Females")
    stem!(ax, base_agg .- half_w, agg_y; color = :royalblue, label = nothing)
    stem!(ax, base_agg .+ half_w, agg_y; color = :crimson, label = nothing)
    stem!(ax, base_agent[1:n_male_st] .- half_w, male_y; color = :royalblue, label = nothing)
    stem!(ax, base_agent[1:n_fem_st] .+ half_w, female_y; color = :crimson, label = nothing)
    return axislegend(
        ax, position = :rt, labelsize = 9, framevisible = true,
        framecolor = (:gray75, 0.6), backgroundcolor = (:white, 0.88),
        padding = (5, 5, 3, 3), margin = (6, 6, 6, 6), patchsize = (14, 8)
    )
end

"""
    setup_size_density_panel!(ax, imm_m, mat_m, imm_f, mat_f, legal_size; peak = nothing)

Populate normalized size probability densities for immature/mature males and females
alongside the legal size reference line.
"""
function setup_size_density_panel!(ax, imm_m, mat_m, imm_f, mat_f, legal_size; peak = nothing)
    lines!(ax, imm_m; color = :royalblue, linewidth = 1.5, linestyle = :dash, label = "Immature Males")
    lines!(ax, mat_m; color = :slategrey, linewidth = 2.2, label = "Mature Males")
    lines!(ax, imm_f; color = :crimson, linewidth = 1.5, linestyle = :dash, label = "Immature Females")
    lines!(ax, mat_f; color = :darkorange, linewidth = 2.2, label = "Mature Females")
    
    if legal_size isa Observable
        peak_obs = peak === nothing ? Observable(1.0f0) : peak
        legal_line = lift(legal_size, peak_obs) do size, max_density
            Point2f[Point2f(size, 0.0), Point2f(size, max_density)]
        end
        lines!(ax, legal_line; color = :purple, linewidth = 1.5,
            linestyle = :dot, label = "Legal size (male)")
    elseif peak !== nothing
        vlines!(ax, legal_size, 0.0f0, peak;
                color = :purple, linewidth = 1.5, linestyle = :dot, label = "Legal size (male)")
    else
        vlines!(ax, [legal_size];
                color = :purple, linewidth = 1.5, linestyle = :dot, label = "Legal size (male)")
    end
    return axislegend(
        ax, position = :rt, labelsize = 9, framevisible = true,
        framecolor = (:gray75, 0.6), backgroundcolor = (:white, 0.88),
        padding = (5, 5, 3, 3), margin = (6, 6, 6, 6), patchsize = (14, 8)
    )
end

"""
    setup_ogive_panel!(ax, legal_size, realized_males, realized_females)

Populate theoretical maturity curves, selectivity retention, and realized binned observations.
"""
function setup_ogive_panel!(ax, legal_size, realized_males, realized_females)
    cw_ref = 20.0:1.0:140.0
    lines!(ax, cw_ref, [maturity(0, log(c); deterministic = true) for c in cw_ref],
           color = :black, linewidth = 2.0, label = "Theoretical Male")
    lines!(ax, cw_ref, [maturity(1, log(c); deterministic = true) for c in cw_ref],
           color = :grey40, linewidth = 2.0, linestyle = :dash, label = "Theoretical Female")
    selectivity_curve = if legal_size isa Observable
        lift(s50 -> [selectivity(c; s50 = s50) for c in cw_ref], legal_size)
    else
        [selectivity(c; s50 = legal_size) for c in cw_ref]
    end
    lines!(ax, cw_ref, selectivity_curve,
           color = :purple, linewidth = 2.0, label = "Theoretical S(L)")
    
    scatter!(ax, realized_males, color = :royalblue, markersize = 8, label = "Realized Males")
    scatter!(ax, realized_females, color = :crimson, markersize = 8, label = "Realized Females")
    return axislegend(
        ax, position = :lt, labelsize = 9, framevisible = true,
        framecolor = (:gray75, 0.6), backgroundcolor = (:white, 0.88),
        padding = (5, 5, 3, 3), margin = (6, 6, 6, 6), patchsize = (14, 8)
    )
end

"""
    setup_macro_panel!(ax, ts_biomass, ts_catch, obs_pts, sch_pts, dd_pts, les_pts;
                       label_obs = "Annual Surveyed Biomass", label_sch = "Schaefer Fit",
                       label_dd = "Delay-Diff Fit", label_les = "Leslie Fit")

Populate biomass and catch trajectories alongside macro-ecological surplus production,
delay-difference, and Leslie model fits.
"""
function setup_macro_panel!(ax, ts_biomass, ts_catch, obs_pts, sch_pts, dd_pts, les_pts;
                            label_obs = "Annual Surveyed Biomass",
                            label_sch = "Schaefer Fit",
                            label_dd = "Delay-Diff Fit",
                            label_les = "Leslie Fit",
                            label_catch = "Annual Catch",
                            baseline_biomass = nothing)
    if baseline_biomass !== nothing
        lines!(ax, baseline_biomass, label = "Unzoned Baseline", color = (:gray, 0.6), linewidth = 1.6, linestyle = :dot)
    end
    lines!(ax, ts_biomass, label = "Legal Fishable Biomass", color = :purple, linewidth = 2.2)
    lines!(ax, ts_catch, label = label_catch, color = :darkorange, linewidth = 1.8, linestyle = :dash)
    scatter!(ax, obs_pts, color = :black, markersize = 8, label = label_obs)
    lines!(ax, sch_pts, color = :royalblue, linewidth = 2.0, linestyle = :dash, label = label_sch)
    lines!(ax, dd_pts, color = :seagreen, linewidth = 2.0, linestyle = :dashdot, label = label_dd)
    lines!(ax, les_pts, color = :darkorange, linewidth = 2.2, label = label_les)
    return axislegend(
        ax, position = :lt, labelsize = 9, framevisible = true,
        framecolor = (:gray75, 0.6), backgroundcolor = (:white, 0.88),
        padding = (5, 5, 3, 3), margin = (6, 6, 6, 6), patchsize = (14, 8)
    )
end

"""
    setup_stock_recruitment_panel!(ax, sr_pts, ricker_pts, bh_pts)

Populate realized log10 spawner-recruit points and fitted Ricker and Beverton-Holt curves.
"""
function setup_stock_recruitment_panel!(
    ax,
    sr_pts,
    ricker_pts,
    bh_pts;
    markersize = 12,
    linewidth = 3.0,
    labelsize = 10
)
    scatter!(
        ax, sr_pts;
        color = :seagreen,
        markersize = markersize,
        strokewidth = 1.0,
        strokecolor = :black,
        label = "Realized Male Recruits"
    )
    lines!(ax, ricker_pts; color = :crimson, linewidth = linewidth, label = "Ricker Fit")
    lines!(
        ax, bh_pts;
        color = :royalblue,
        linewidth = linewidth,
        linestyle = :dash,
        label = "Beverton–Holt Fit"
    )
    return axislegend(
        ax;
        position = :rb,
        labelsize = labelsize,
        framevisible = true,
        framecolor = (:gray75, 0.6),
        backgroundcolor = (:white, 0.88),
        padding = (6, 6, 4, 4),
        margin = (8, 8, 8, 8),
        patchsize = (18, 10)
    )
end

"""
    COLUMN_GAP

Space reserved between the figure's columns, in pixels. Makie inserts its own gap
as well; this is subtracted so the two plotting columns still fit inside the window
once the figure's default gap is accounted for.
"""
const COLUMN_GAP = 20

"""
    flatten_toml(cfg, prefix = "")

Flatten a parsed TOML table to `"section.key" => value` pairs.

The configuration is nested (`[environment]` then `temp_base`) while the parameter
structs are flat fields on a named struct, so every comparison between the file
and the model works on the flattened form.
"""
function flatten_toml(cfg::AbstractDict, prefix = "")
    out = Dict{String, Any}()
    for (k, v) in cfg
        key = isempty(prefix) ? String(k) : string(prefix, ".", k)
        if v isa AbstractDict
            merge!(out, flatten_toml(v, key))
        else
            out[key] = v
        end
    end
    return out
end

"""
    config_diff(previous, current)

Values that differ between two configuration tables, as `key => (old, new)` pairs
sorted by key.

Both arguments may be nested (as parsed from TOML) or already flattened;
`flatten_toml` is applied to each, so passing a nested table by mistake compares
like with like instead of reporting every nested table as a single changed key.

Keys present in `previous` but absent from `current` are reported with a `nothing`
new value, so a parameter deleted from the file shows up in the report rather than
being silently ignored.
"""
function config_diff(previous::AbstractDict, current::AbstractDict)
    prev = flatten_toml(previous)
    cur = flatten_toml(current)
    out = Pair{String, Any}[]
    for k in sort!(collect(keys(cur)))
        new = cur[k]
        if !haskey(prev, k)
            push!(out, k => (nothing, new))
        elseif !isequal(prev[k], new)
            push!(out, k => (prev[k], new))
        end
    end
    for k in sort!(collect(keys(prev)))
        haskey(cur, k) || push!(out, k => (prev[k], nothing))
    end
    return out
end

"""
    rescan_config!(state::Any) -> String

Re-read `state.config_path`, apply only what changed since the last scan, and
update the snapshot. Returns a one-line report for the status bar, or `""` when
there was nothing to do.

The diff is the point, and it is not an optimisation. `configs/snowcrab.toml` is
complete — it declares every parameter — so re-parsing it and rebuilding the
parameter tree from the result would re-apply all 260 values annually and silently
discard anything the user had changed in the parameter-steering panel since the
last scan. Comparing against the previous scan and touching only the fields that
genuinely moved keeps those live edits. The panel's edits are not written back to
the file, so the two are independent by design: the file is the scenario, the
panel is the perturbation.

Each changed key is applied as a leaf keyword via `reconstruct_params`, starting
from the *live* parameters rather than from a fresh struct.

Keys in [`RESCAN_IMMUTABLE`](@ref) are reported and skipped, as are keys deleted
from the file: for a deletion there is no value to restore, and the panel may have
changed it in the meantime. The snapshot records what the file now says rather than
what was applied, so those keys are not re-reported on every subsequent scan.
"""
# `state` is unannotated: `IBMGuiState` is declared further down this file, so the
# type is not yet in scope to annotate a signature here.
function rescan_config!(state)
    path = state.config_path
    (isempty(path) || !isfile(path)) && return ""

    current = flatten_toml(TOML.parsefile(path))
    diff = config_diff(state.config_snapshot, current)
    isempty(diff) && return ""

    applied = String[]
    skipped = String[]
    # `reconstruct_params` takes *leaf* field names, not whole sections: it passes
    # every keyword to every parameter struct and keeps the ones that are fields of
    # that struct. Handing it `environment => new_struct` would be silently ignored,
    # because `environment` is not a field of `EnvironmentParams`. That is safe to
    # rely on here only because all 260 leaf names are unique across sections, so a
    # name always identifies exactly one parameter; `test/runtests.jl` asserts it.
    kwargs_list = Pair{Symbol, Any}[]
    for (key, (old, new)) in diff
        parts = split(key, '.')
        if length(parts) < 2 || new === nothing || Symbol(parts[2]) in RESCAN_IMMUTABLE
            push!(skipped, key)
            continue
        end
        push!(kwargs_list, Symbol(parts[2]) => new)
        push!(applied, "$key: $old -> $new")
    end

    if !isempty(kwargs_list)
        p = abmproperties(state.model)[:params]
        abmproperties(state.model)[:params] = reconstruct_params(p; kwargs_list...)
    end

    state.config_snapshot = current

    msg = "Config: $(length(applied)) applied"
    isempty(skipped) || (msg *= ", restart-only: " * join(skipped, ", "))
    return msg
end

# ------------------------------------------------------------------------------
# Data Structure for GUI State Management
# ------------------------------------------------------------------------------

"""
    CALENDAR_MONTH_END

Last julian day of each calendar month in a 365-day year, so day 59 is the end of
February. Used to turn the julian day into a month the reader recognises.
"""
const CALENDAR_MONTH_END = (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)

"""
    calendar_month(day_of_year::Integer, days_per_year::Real)

Calendar month (1-12) containing the 1-based julian `day_of_year`.

Exact for a 365-day year. In a 366-day year, February is given its extra day by
`days_per_year >= 366`, so every later month boundary stays on its calendar date;
the only cost is that day 366 reports as December rather than as a 13th month.

This replaces `floor((day % 365) / 30.416) + 1`, which put the last day of every
month into the next one.
"""
function calendar_month(day_of_year::Integer, days_per_year::Real = 365.0)
    leap = days_per_year >= 366.0
    acc = 0
    for (m, len) in enumerate((31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
        acc += (m == 2 && leap) ? 1 : 0
        acc += len
        day_of_year <= acc && return m
    end
    return 12
end

"""
    IBMGuiState

Mutable container maintaining active simulation objects, reactive Makie
observables, time-series history buffers, and concurrent background tasks.

# Mathematical Formulation
At each simulation step ``t \\to t + \\Delta t``:
- Spatial coordinates ``(x_i, y_i) \\in \\Omega`` and physical traits
  (carapace width ``L_i``, wet weight ``W_i``, developmental stage ``S_i``)
  are streamed to OpenGL particle buffers.
- Population abundance ``N(t)`` and fishable biomass ``B_{\\text{legal}}(t)``
  are appended to temporal histories.
- Realized empirical maturity rates ``\\hat{P}(L) = \\frac{N_{\\text{adult}, L}}{N_{\\text{total}, L}}``
  are dynamically derived across discrete carapace width bins.
- At annual intervals (day 274 = September 1 snapshot), macro-ecological
  models are recalibrated against empirical simulated production.
"""
# `@kwdef` generates a keyword constructor alongside the positional one. This
# matters here: the state carries 66 fields, and a purely positional call site
# silently desynchronises whenever a field is added, removed or reordered on one
# side only. The failure surfaces as a `MethodError` whose signature dump runs to
# several thousand characters, with no indication of *which* field is wrong. With
# keywords, Julia checks names rather than positions, so the compiler and the test
# suite both catch it immediately.
Base.@kwdef mutable struct IBMGuiState
    model::StandardABM
    is_running::Observable{Bool}
    sim_speed_days::Observable{Float64}
    current_time::Observable{Float64}
    status_text::Observable{String}

    # Spatial map observables
    obs_env_field::Observable{Matrix{Float64}}
    obs_bg_layer::Observable{Symbol}
    obs_agent_x::Observable{Vector{Float32}}
    obs_agent_y::Observable{Vector{Float32}}
    obs_agent_pts::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_agent_sizes::Observable{Vector{Float32}}
    obs_agent_colors::Observable{Vector{String}}

    # Time series history buffers & observables
    t_history::Vector{Float64}
    n_total_history::Vector{Float64}
    n_females_history::Vector{Float64}
    n_males_history::Vector{Float64}
    n_legal_history::Vector{Float64}
    biomass_legal_history::Vector{Float64}
    catch_history::Vector{Float64}
    # Realised mean wet weight (kg) of each live component, used to convert the
    # number-based macro-model fits to biomass for the dashboard figure.
    mean_weight_legal_history::Vector{Float64}
    mean_weight_male_history::Vector{Float64}
    mean_weight_female_history::Vector{Float64}
    mean_weight_total_history::Vector{Float64}
    catch_at_year_start::Float64
    current_year_idx::Int
    config_path::String  # Path to the TOML config file for annual rescan
    # Flattened form of the config as last read, so a rescan can tell what the
    # user edited since. See `rescan_config!`.
    config_snapshot::Dict{String, Any}

    obs_ts_total::Observable{Vector{Point2f}}
    obs_ts_females::Observable{Vector{Point2f}}
    obs_ts_males::Observable{Vector{Point2f}}
    obs_ts_legal::Observable{Vector{Point2f}}
    obs_ts_biomass::Observable{Vector{Point2f}}
    obs_ts_catch::Observable{Vector{Point2f}}

    # Pre-individuation abundances for the stage-abundance figure: eggs currently
    # brooded, then one series per `GUI_EARLY_STAGES` aggregate cohort. Raw
    # counts, as above; the log transform happens when the figure is drawn.
    n_egg_history::Vector{Float64}
    n_early_stage_history::Vector{Vector{Float64}}
    # Individuated (sex-discriminated) abundances over the same stages.
    n_early_agents_history::Vector{NTuple{3, Float64}}  # (male, female, undifferentiated)

    # Current (not time-series) abundances for the stage figure, written every
    # step. These are separate observables rather than a `lift` over the history
    # buffers because `lift` over a plain `Vector` *iterates* it, so the closure
    # would receive one buffer element at a time rather than the buffer.
    obs_current_egg::Observable{Float64}
        obs_current_agents::Observable{NTuple{3, Float64}}

    # Per-stage abundance for the stage figure, as three parallel series rather than
    # one series per stage. Aggregate cohorts are `AggregateGroup`, which carries no
    # `sex`, so stages below the individuation boundary are a single undifferentiated
    # series; agent stages are split by sex.
    #
    # Indexed by stage through the offsets below: `obs_stage_agg[k]` is stage `k - 1`;
    # `obs_stage_male[k]` is stage `juvenile_max_stage - 1 + k`.
    obs_stage_agg::Observable{Vector{Float64}}
    obs_stage_male::Observable{Vector{Float64}}
    obs_stage_female::Observable{Vector{Float64}}

    obs_ts_egg::Observable{Vector{Point2f}}
    # One observable per `GUI_EARLY_STAGES` value, held as a vector of observables
    # rather than one observable wrapping a vector of vectors: `lines!` takes a
    # single point vector, so a nested `Observable{Vector{Vector{Point2f}}}` has
    # to be unwrapped with `[]` at every call site and silently fails if skipped.
    obs_ts_early_stage::Vector{Observable{Vector{Point2f}}}
    obs_ts_early_agents::Observable{Vector{Point2f}}
    obs_ts_early_agents_f::Observable{Vector{Point2f}}
    obs_ts_early_agents_u::Observable{Vector{Point2f}}


    # Size structure & demographic observables. `*_mature_*` are the subset of
    # each sex that the model itself flags as mature, so the panel shows maturity
    # as a proportion of each sex's own size structure rather than against an
    # independently chosen threshold.
    obs_cw_male::Observable{Vector{Float32}}
    obs_cw_female::Observable{Vector{Float32}}
    obs_cw_mature_male::Observable{Vector{Float32}}
    obs_cw_mature_female::Observable{Vector{Float32}}
    # The complement of each mature series, split by the agent's own `is_mature`
    # flag. The size panel plots immature and mature side by side rather than the
    # whole sex, because a whole-sex curve mixes the two populations and its shape
    # then depends on how many of each are alive rather than on size alone.
    obs_cw_immature_male::Observable{Vector{Float32}}
    obs_cw_immature_female::Observable{Vector{Float32}}

    # Realized functional relationships observables
    obs_ogive_male_pts::Observable{Vector{Point2f}}
    obs_ogive_female_pts::Observable{Vector{Point2f}}
    obs_selectivity_pts::Observable{Vector{Point2f}}
    obs_sr_pts::Observable{Vector{Point2f}}
    # Raw (untransformed) spawner/recruit pairs, in the order plotted. The panel
    # draws log10 of both axes, but `fit_stock_recruitment` is fitted in linear
    # space, so the untransformed pairs have to survive the transform.
    sr_raw::Vector{Tuple{Float64, Float64}}
    # Ricker fit `R = a·S·exp(-b·S)`, drawn in log10 and annotated with its
    # parameters. Empty when there are too few points to fit.
    obs_sr_fit_curve::Observable{Vector{Point2f}}
    obs_sr_fit_curve_bh::Observable{Vector{Point2f}}
    obs_sr_fit_text::Observable{String}

    # Macro-model calibration state
    annual_data::DataFrame
    macro_obs_biomass::Observable{Vector{Point2f}}
    macro_sch_biomass::Observable{Vector{Point2f}}
    macro_dd_biomass::Observable{Vector{Point2f}}
    macro_les_biomass::Observable{Vector{Point2f}}
    macro_metrics_text::Observable{String}
    # Legend entries carrying the key estimated parameters of each macro model,
    # rebuilt on every re-fit so the legend tracks the current calibration.
    macro_label_observed::Observable{String}
    macro_label_sch::Observable{String}
    macro_label_dd::Observable{String}
    macro_label_les::Observable{String}
    macro_conversion_text::Observable{String}

    # Snapshot state cache
    snapshots::Dict{Int, Any}
    snapshot_labels::Observable{Vector{String}}

    # Interactive AOI Hydrography & HSI Observables
    aoi_lon::Observable{Float64} = Observable(-59.0)
    aoi_lat::Observable{Float64} = Observable(45.0)
    aoi_depth_str::Observable{String} = Observable("Depth: 135.0 m")
    aoi_temp_str::Observable{String} = Observable("Temp: 3.20 °C")
    aoi_hsi_str::Observable{String} = Observable("HSI: 0.8800")

    # Spatial Exclusion (AABB) Marine Protected Areas
    exclusion_boxes::Observable{Vector{ExclusionBox}} = Observable(ExclusionBox[])
    is_drawing_exclusion::Observable{Bool} = Observable(false)
    exclusion_draw_start::Observable{Union{Nothing, Point2f}} = Observable{Union{Nothing, Point2f}}(nothing)
    obs_exclusion_lines::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_exclusion_rects::Observable{Vector{Rect2f}} = Observable(Rect2f[])
    obs_exclusion_status::Observable{String} = Observable("No active exclusion zone")

    # Interactive Initial Stage Distribution Editor
    obs_stage_dist_init::Observable{Vector{Float64}} = Observable(copy(DEFAULT_STAGE_DISTRIBUTION))
    obs_total_seed_str::Observable{String} = Observable("Total: " * string(round(Int, sum(DEFAULT_STAGE_DISTRIBUTION))))

    # Marine Protected Area Contrast Overlay (Ghost baseline trajectory)
    obs_baseline_biomass::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_baseline_catch::Observable{Vector{Point2f}} = Observable(Point2f[])
    has_baseline::Observable{Bool} = Observable(false)

    # Multi-attribute calibration criteria text
    calib_criteria_text::Observable{String} = Observable("Calibration Criteria: Initialized")

    # Environmental distributions observables (depth, temp, hsi)
    obs_env_depths::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_env_temps::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_env_hsi::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_env_depth_vals::Observable{Vector{Float64}} = Observable(Float64[])
    obs_env_temp_vals::Observable{Vector{Float64}} = Observable(Float64[])
    obs_env_hsi_vals::Observable{Vector{Float64}} = Observable(Float64[])
    ax_env_depth::Any = nothing
    ax_env_temp::Any = nothing
    ax_env_hsi::Any = nothing

    # Annual summary table text
    annual_table_text::Observable{String} = Observable("Annual Summary Table: Awaiting Simulation Steps")

    # Seasonal Bottom Temperature Climatology Observables (Tab 1)
    obs_temp_cycle_pts::Observable{Vector{Point2f}} = Observable(Point2f[])
    obs_temp_cycle_mean::Observable{Float64} = Observable(3.5)
    obs_temp_cycle_title::Observable{String} = Observable("Seasonal Bottom Temperature Climatology")

    # Lifecycle control
    is_active::Observable{Bool} = Observable(true)
    obs_unit_meters::Observable{Float64} = Observable(1.0)
    active_tab::Observable{Int} = Observable(1)
    ax_aoi_map::Any = nothing
    ax_temp_cycle::Any = nothing
    ax_map::Any = nothing
    ax_ts_n::Any = nothing
    ax_macro::Any = nothing

    # Asynchronous task handle
    sim_task::Union{Nothing, Task} = nothing
end

"""
    close_gui!(state::IBMGuiState)

Halt asynchronous background simulation loops and deactivate interactive observables.
"""
function close_gui!(state::IBMGuiState)
    state.is_running[] = false
    state.is_active[] = false
    if state.sim_task !== nothing && !istaskdone(state.sim_task)
        yield()
    end
    return nothing
end

# ------------------------------------------------------------------------------
# Core Stepping & Analytical Update Functions
# ------------------------------------------------------------------------------

function gui_environment_layer_field(model, current_layer::Symbol)
    mprops = abmproperties(model)
    if current_layer === :habitat_quality && haskey(mprops, :habitat_quality_field)
        return copy(model.habitat_quality_field)
    elseif current_layer === :depth && haskey(mprops, :depth_field)
        return copy(mprops[:depth_field])
    else
        return copy(model.temperature_field)
    end
end

"""
    update_environmental_distributions!(state::IBMGuiState)

Update empirical distributions for seabed depth, ambient bottom temperature,
and Habitat Suitability Index (HSI) across the Area of Interest (AOI) expanded to the
dimension specified by the box size (where each grid unit = 1 meter).
"""
function update_environmental_distributions!(state::IBMGuiState)
    model = state.model
    mprops = abmproperties(model)
    d_field = haskey(mprops, :depth_field) ? mprops[:depth_field] :
              (haskey(mprops, :depth_map) ? mprops[:depth_map] : nothing)
    t_field = haskey(mprops, :temperature_field) ? mprops[:temperature_field] : nothing
    h_field = haskey(mprops, :habitat_quality_field) ? mprops[:habitat_quality_field] : nothing

    if d_field !== nothing && !isempty(d_field)
        d_vals = filter(isfinite, Float64.(vec(d_field)))
        t_vals = t_field !== nothing ? filter(isfinite, Float64.(vec(t_field))) : Float64[]
        h_vals = h_field !== nothing ? filter(isfinite, Float64.(vec(h_field))) : Float64[]

        if !isempty(d_vals)
            state.obs_env_depth_vals[] = d_vals
            max_d = maximum(d_vals)
            d_norm = max_d > 0.0 ? d_vals ./ max_d : fill(0.0, length(d_vals))
            grid_x = collect(0.0:0.02:1.0)
            kd_z = kde_density(d_norm, grid_x; bandwidth = 0.05, scale = :probability)
            state.obs_env_depths[] = [Point2f(grid_x[i], kd_z[i]) for i in eachindex(grid_x)]

            if !isempty(t_vals)
                state.obs_env_temp_vals[] = t_vals
                t_scaled = [isfinite(v) ? (v / 10.0) : 0.0 for v in t_vals]
                kd_t = kde_density(t_scaled, grid_x; bandwidth = 0.05, scale = :probability)
                state.obs_env_temps[] = [Point2f(grid_x[i], kd_t[i]) for i in eachindex(grid_x)]
            end

            if !isempty(h_vals)
                state.obs_env_hsi_vals[] = h_vals
                h_scaled = [isfinite(v) ? v : 0.0 for v in h_vals]
                kd_h = kde_density(h_scaled, grid_x; bandwidth = 0.05, scale = :probability)
                state.obs_env_hsi[] = [Point2f(grid_x[i], kd_h[i]) for i in eachindex(grid_x)]
            end
        end
    end
    return nothing
end

"""
    step_gui_simulation!(state::IBMGuiState, days::Float64)

Advance the underlying agent-based model by `days` days and refresh all
reactive GUI observables across the three dashboard interfaces.

# Arguments
- `state::IBMGuiState`: Active GUI state container.
- `days::Float64`: Elapsed simulation time to step forward.

# Returns
`Nothing`.
"""
function step_gui_simulation!(state::IBMGuiState, days::Float64)
    model = state.model
    dt = abmproperties(model)[:dt]
    n_steps = max(1, round(Int, days / dt))
    params = abmproperties(model)[:params]
    # Year length comes from the configuration, not a literal 365. The GUI used to
    # assume a 365-day year here while the rest of the model honoured
    # `SimulationParams.days_per_year`, so a leap-year configuration shifted every
    # GUI year boundary, snapshot and season test relative to the simulation's own.
    dpy = params.simulation.days_per_year

    # Advance model
    prev_year = floor(Int, get(abmproperties(model), :current_time, 0.0) / dpy)

    if days > 0.0
        Agents.step!(model, n_steps)
    end

    curr_t = Float64(abmproperties(model)[:current_time])
    curr_t == state.current_time[] || (state.current_time[] = curr_t)

    curr_year = floor(Int, curr_t / dpy)
    if curr_year > prev_year
        state.snapshots[curr_year] = (
            model = deepcopy(model),
            t_history = copy(state.t_history),
            n_total_history = copy(state.n_total_history),
            n_females_history = copy(state.n_females_history),
            n_males_history = copy(state.n_males_history),
            n_legal_history = copy(state.n_legal_history),
            biomass_legal_history = copy(state.biomass_legal_history),
            catch_history = copy(state.catch_history),
            annual_data = copy(state.annual_data)
        )
        labels = String[]
        for y in sort(collect(keys(state.snapshots)))
            push!(labels, "Year $y")
        end
        state.snapshot_labels[] = labels

        # Re-read the configuration once a simulated year, so a scenario can be
        # edited while the run is in progress. Only fields that actually moved in
        # the file are applied, so parameter-steering edits made in the GUI survive
        # the scan. A failure here must not stop the run: a half-typed TOML file is
        # a normal thing to have on disk while someone is editing it.
        if !isempty(state.config_path)
            try
                msg = rescan_config!(state)
                isempty(msg) || (state.status_text[] *= " | $msg")
            catch err
                state.status_text[] *= " | Config rescan failed: " *
                    first(split(sprint(showerror, err), '\n'))
            end
        end
    end

    # 1. Update Spatial Map Observables
    state.obs_env_field[] = gui_environment_layer_field(model, state.obs_bg_layer[])

    living = collect(allagents(model))
    n_living = length(living)

    xs = Float32[Float32(a.pos[1]) for a in living]
    ys = Float32[Float32(a.pos[2]) for a in living]
    szs = Float32[Float32(min(5.5, max(2.0, 1.8 + a.size * 0.03))) for a in living]
    cols = String[organism_color(a, state.model.params) for a in living]

    state.obs_agent_sizes[] = szs
    state.obs_agent_colors[] = cols
    state.obs_agent_pts[] = [Point2f(xs[i], ys[i]) for i in eachindex(xs)]
    state.obs_agent_x[] = xs
    state.obs_agent_y[] = ys

    # 2. Update Demographic Counts & Timeseries Buffers
    legal_sz = params.fishery.legal_size_male
    adult_sz = params.organism.adult_size_threshold
    n_fem = count(is_mature_female, living)
    n_mal = count(is_mature_male, living)
    n_leg = count(o -> is_legal_male(o, legal_sz, adult_sz), living)

    # Compute legal fishable biomass in kg
    b_leg_kg = sum(
        mass_if_legal_male(o, legal_sz, adult_sz) for o in living;
        init = 0.0
    ) / 1000.0
    cum_catch = Float64(abmproperties(model)[:total_catch_biomass_kg])

    # Mean wet weight of each live component, in kg, measured from the individuals
    # actually present rather than from the stage-mean allometry. The macro-model
    # fits are in number, so a count trajectory is converted to biomass by
    # multiplying by the mean weight of the component it describes; using the
    # realised mean rather than a fitted constant keeps the plotted conversion
    # consistent with the measured biomass on the same axes.
    mean_weight_kg = function (orgs)
        isempty(orgs) ? 0.0 : sum(o -> o.mass, orgs; init = 0.0) / (1000.0 * length(orgs))
    end
    legal_orgs = [o for o in living if is_legal_male(o, legal_sz, adult_sz)]
    male_orgs = [o for o in living if is_mature_male(o)]
    female_orgs = [o for o in living if is_mature_female(o)]
    mw_leg_kg = mean_weight_kg(legal_orgs)
    mw_mal_kg = mean_weight_kg(male_orgs)
    mw_fem_kg = mean_weight_kg(female_orgs)
    mw_tot_kg = mean_weight_kg(living)

    # Record one sample per *advance*, not per call. `launch_organism_gui` ends
    # with an initial `step_gui_simulation!(state, 0.0)` to paint the panel before
    # any stepping, and the snapshot/restore handler also repaints with `0.0`;
    # pushing unconditionally there appended a second point at t = 0, so every
    # trajectory in the dashboard carried a duplicated origin.
    #
    # Decided once, before any buffer is appended. Re-evaluating `t_history[end]`
    # at the later stage/egg block would read the value this block just wrote and
    # see "no advance", silently leaving those buffers one short of the time axis.
    # An empty buffer always records: Reset clears these before repainting, so the
    # post-reset step must not compare against a previous time that no longer exists.
    advanced = isempty(state.t_history) || curr_t != state.t_history[end]
    if advanced
        push!(state.t_history, curr_t)
        push!(state.n_total_history, Float64(n_living))
        push!(state.n_females_history, Float64(n_fem))
        push!(state.n_males_history, Float64(n_mal))
        push!(state.n_legal_history, Float64(n_leg))
        push!(state.biomass_legal_history, b_leg_kg)
        push!(state.mean_weight_legal_history, mw_leg_kg)
        push!(state.mean_weight_male_history, mw_mal_kg)
        push!(state.mean_weight_female_history, mw_fem_kg)
        push!(state.mean_weight_total_history, mw_tot_kg)
    end

    year_idx = floor(Int, curr_t / dpy) + 1
    if year_idx > state.current_year_idx
        state.current_year_idx = year_idx
        state.catch_at_year_start = cum_catch
    end
    annual_catch = cum_catch - state.catch_at_year_start

    # Annual catch is rescaled against `catch_at_year_start` above, which must
    # happen on every call including a repaint; only the *record* is gated on an
    # advance, so this buffer stays the same length as the rest.
    if advanced
        push!(state.catch_history, annual_catch)
    end

    # Update timeseries line observables. The four abundance series are counts and
    # go through `log10_count`; biomass and annual catch are kilograms, not counts,
    # and a log axis would be the wrong scale for them (and would divide by zero
    # on any day the catch is nil).
    pts_tot = [Point2f(state.t_history[i], Float32(log10_count(state.n_total_history[i])))
               for i in eachindex(state.t_history)]
    pts_fem = [Point2f(state.t_history[i], Float32(log10_count(state.n_females_history[i])))
               for i in eachindex(state.t_history)]
    pts_mal = [Point2f(state.t_history[i], Float32(log10_count(state.n_males_history[i])))
               for i in eachindex(state.t_history)]
    pts_leg = [Point2f(state.t_history[i], Float32(log10_count(state.n_legal_history[i])))
               for i in eachindex(state.t_history)]
    pts_bio = [Point2f(state.t_history[i], state.biomass_legal_history[i])
               for i in eachindex(state.t_history)]
    pts_cat = [Point2f(state.t_history[i], state.catch_history[i])
               for i in eachindex(state.t_history)]

    state.obs_ts_total[] = pts_tot
    state.obs_ts_females[] = pts_fem
    state.obs_ts_males[] = pts_mal
    state.obs_ts_legal[] = pts_leg
    state.obs_ts_biomass[] = pts_bio
    state.obs_ts_catch[] = pts_cat

    if state.ax_ts_n !== nothing
        max_t = max(60.0, curr_t)
        xlims!(state.ax_ts_n, 0.0, max_t * 1.05)
        all_counts = [state.n_total_history; state.n_females_history; state.n_males_history; state.n_legal_history]
        max_c = maximum(all_counts; init = 10.0)
        ylims!(state.ax_ts_n, 0.0, max(2.5, log10_count(max_c) + 0.3))
    end
    if state.ax_macro !== nothing
        xlims!(state.ax_macro, 0.0, max(60.0, curr_t) * 1.05)
        max_b = max(
            maximum(state.biomass_legal_history; init = 10.0),
            maximum(state.catch_history; init = 10.0)
        )
        ylims!(state.ax_macro, 0.0, max(50.0, max_b * 1.15))
    end

    # 3. Update Size Structure & Stage Demographics
    cw_m = Float32[Float32(a.size) for a in living if a.sex == male]
    cw_f = Float32[Float32(a.size) for a in living if a.sex == female]
    # An empty series stays empty. It used to be replaced by the single value 0.0 so
    # that the plot had something to draw, but that 0.0 is a size like any other and
    # put a phantom cohort of one animal at the origin of the size axis.
    state.obs_cw_male[] = cw_m
    state.obs_cw_female[] = cw_f
    # Maturity uses the agent's own flag rather than re-deriving a size threshold,
    # so the mature curve tracks exactly the individuals the rest of the model
    # treats as mature (and which the ogive and macro panels also use).
    cw_mm = Float32[Float32(a.size) for a in living if a.sex == male && a.is_mature]
    cw_mf = Float32[Float32(a.size) for a in living if a.sex == female && a.is_mature]
    state.obs_cw_mature_male[] = cw_mm
    state.obs_cw_mature_female[] = cw_mf
    cw_im = Float32[Float32(a.size) for a in living if a.sex == male && !a.is_mature]
    cw_if = Float32[Float32(a.size) for a in living if a.sex == female && !a.is_mature]
    state.obs_cw_immature_male[] = cw_im
    state.obs_cw_immature_female[] = cw_if

    # Pre-individuation abundances for the stage-abundance figure, in one pass.
    #
    # Eggs are counted from the brooding females' clutches, which is the pool
    # *before* hatching: a clutch becomes a `stage = 0` aggregate cohort only at
    # `org.hatch_day`, so eggs and stage 0 are disjoint pools and both are needed.
    # Summing `egg_count` unconditionally matches `summarize_population_state`,
    # which reports the same quantity as `total_eggs_brooded`; the undifferentiated
    # branch is defensive, since `build_stages` currently emits only male and female
    # agents, but a species that tracked sex later would otherwise be silently
    # dropped from the figure.
    n_egg = 0
    n_agent_male = 0
    n_agent_female = 0
    n_agent_undiff = 0
    # Per-stage tallies for the stage figure. The agent stages run from
    # `juvenile_max_stage` (the individuation boundary) up to each sex's own
    # terminal stage — 6..16 for males and 6..14 for females under the snow crab
    # configuration. The species stage tables stop at 12 and 11; those bound the
    # Leslie projection, not the agents, which molt on past them, so the figure is
    # sized from `max_stage_*` instead or the largest cohorts would fall off the end.
    stage_base = params.organism.juvenile_max_stage
    stage_agg = zeros(Float64, length(state.obs_stage_agg[]))
    stage_male = zeros(Float64, length(state.obs_stage_male[]))
    stage_female = zeros(Float64, length(state.obs_stage_female[]))
    for o in living
        k = o.stage - stage_base + 1
        if o.sex === male
            n_agent_male += 1
            if 1 <= k <= length(stage_male)
                stage_male[k] += 1.0
            end
        elseif o.sex === female
            n_agent_female += 1
            if 1 <= k <= length(stage_female)
                stage_female[k] += 1.0
            end
        else
            n_agent_undiff += 1
        end
        n_egg += o.egg_count
    end

    # Aggregate cohort abundances for `GUI_EARLY_STAGES`. `AggregateGroup` has no
    # `sex` field, so these stages are undifferentiated by construction.
    early_counts = zeros(Float64, length(GUI_EARLY_STAGES))
    agg = get(abmproperties(model), :aggregate_populations, nothing)
    if agg !== nothing
        for g in agg.groups
            i = findfirst(==(g.stage), GUI_EARLY_STAGES)
            i === nothing || (early_counts[i] += g.abundance)
            # Full per-stage tally for the stage figure. Indexed by stage, not
            # stage + 1: stage 0 is the brooded egg, which is carried by the female
            # rather than being a cohort, so the cohorts start at stage 1.
            k = g.stage
            if 1 <= k <= length(stage_agg)
                stage_agg[k] += g.abundance
            end
        end
    end

    # Same `advanced` flag as the demographic buffers above: these are read back
    # with `eachindex(t_history)`, so a repaint that appended without advancing
    # would leave them one longer than the time axis they are plotted against.
    if advanced
        push!(state.n_egg_history, Float64(n_egg))
        for i in eachindex(state.n_early_stage_history)
            push!(state.n_early_stage_history[i], early_counts[i])
        end
        push!(state.n_early_agents_history,
            (Float64(n_agent_male), Float64(n_agent_female), Float64(n_agent_undiff)))
    end

    # Current values are recomputed on every call, including a repaint, so the
    # stage figure reflects the model after a snapshot restore as well as after a
    # step.
    state.obs_current_egg[] = Float64(n_egg)
    state.obs_current_agents[] =
        (Float64(n_agent_male), Float64(n_agent_female), Float64(n_agent_undiff))
    state.obs_stage_agg[] = stage_agg
    state.obs_stage_male[] = stage_male
    state.obs_stage_female[] = stage_female

    # The figure is drawn on a log10 count axis; the buffers above keep raw counts.
    t_hist = state.t_history
    state.obs_ts_egg[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_egg_history[i])))
         for i in eachindex(t_hist)]
    for k in eachindex(state.obs_ts_early_stage)
        buf = state.n_early_stage_history[k]
        state.obs_ts_early_stage[k][] =
            [Point2f(t_hist[i], Float32(log10_count(buf[i]))) for i in eachindex(t_hist)]
    end
    state.obs_ts_early_agents[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][1])))
         for i in eachindex(t_hist)]
    state.obs_ts_early_agents_f[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][2])))
         for i in eachindex(t_hist)]
    state.obs_ts_early_agents_u[] =
        [Point2f(t_hist[i], Float32(log10_count(state.n_early_agents_history[i][3])))
         for i in eachindex(t_hist)]

    # 4. Update Realized Functional Relationships (Ogives, Selectivity, SR)
    update_functional_observables!(state, living)

    # 5. Check and Record Annual Snapshots for Macro-Model Fitting
    check_and_update_macro_models!(state, curr_t, n_living, n_leg, n_fem,
                                   n_mal, b_leg_kg, cum_catch)

    # Status summary. Four distinct quantities that used to be conflated under
    # "Day": the year, the month, the julian day *within* the year, and the total
    # elapsed model time. Year is an integer year count, day is the 1-based day of
    # the year (so it wraps at the year boundary instead of growing without bound),
    # and the elapsed total is reported under its own name rather than as "Day" —
    # "Day: 730" in the second year of a run is not a day of anything.
    year_n = floor(Int, curr_t / dpy)
    day_of_year = round(Int, min(mod(curr_t, dpy) + 1.0, dpy))
    month_n = calendar_month(day_of_year, dpy)
    state.status_text[] = string("Year: ", year_n,
        "   Month: ", month_n,
        "   Day: ", day_of_year,
        "   Duration: ", round(Int, curr_t), " d")

    # Update environmental driver distributions
    try
        z_grid = get(abmproperties(model), :depth_map, nothing)
        if z_grid !== nothing && !isempty(z_grid)
            z_vals = Float64.(vec(z_grid))
            z_norm = z_vals ./ max(1.0, maximum(z_vals))
            t_vals = Float64.(vec(model.temperature_field)) ./ 10.0
            hsi_grid = get(abmproperties(model), :habitat_quality_field, nothing)
            h_vals = hsi_grid !== nothing ? Float64.(vec(hsi_grid)) : fill(0.5, length(z_vals))

            grid_x = collect(0.0:0.02:1.0)
            kd_z = kde_density(z_norm, grid_x; scale = :probability)
            kd_t = kde_density(clamp.(t_vals, 0.0, 1.0), grid_x; scale = :probability)
            kd_h = kde_density(clamp.(h_vals, 0.0, 1.0), grid_x; scale = :probability)

            state.obs_env_depths[] = [Point2f(grid_x[i], kd_z[i]) for i in eachindex(grid_x)]
            state.obs_env_temps[] = [Point2f(grid_x[i], kd_t[i]) for i in eachindex(grid_x)]
            state.obs_env_hsi[] = [Point2f(grid_x[i], kd_h[i]) for i in eachindex(grid_x)]
        end
    catch
    end

    # Update Multi-Attribute Calibration Criteria Summary
    try
        n_cur = length(living)
        n_min = 20
        n_max = round(Int, benthic_carrying_capacity(state.model) * 1.5)
        p_pass = n_cur >= n_min ? "PASS" : "FAIL"
        b_pass = n_cur <= n_max ? "PASS" : "FAIL"
        cw_m = isempty(living) ? 0.0 : mean(o.size for o in living)
        crit_str = string(
            "Criterion 1 (Persistence): ", n_cur, " living (min: ", n_min, ") -> [", p_pass, "]\n",
            "Criterion 2 (Bounds):      ", n_cur, " living (max: ", n_max, ") -> [", b_pass, "]\n",
            "Criterion 3 (Size Drift):  cw_mean = ", round(cw_m, digits = 1), " mm (adult: ", round(adult_sz, digits = 1), " mm)\n",
            "Criterion 4 (Maturity):    n_mature_m = ", n_mal, ", n_mature_f = ", n_fem, "\n",
            "Criterion 5 (Fishery):     legal_m = ", n_leg, ", landings = ", round(cum_catch, digits = 1), " kg"
        )
        state.calib_criteria_text[] = crit_str
    catch
    end

    # Update Annual Table Text
    try
        ann = state.annual_data
        if !isempty(ann) && nrow(ann) > 0
            n_rows = nrow(ann)
            start_r = max(1, n_rows - 5)
            lines_tbl = ["Year | Total Biomass (kg) | Legal Biomass (kg) | Catch (kg) | Recruits | Spawners"]
            push!(lines_tbl, "-----|--------------------|--------------------|------------|----------|---------")
            for r in start_r:n_rows
                row = ann[r, :]
                tot_b = hasproperty(row, :total_biomass_kg) ? row.total_biomass_kg : 0.0
                leg_b = hasproperty(row, :fishable_male_biomass_kg) ? row.fishable_male_biomass_kg : (hasproperty(row, :biomass_legal) ? row.biomass_legal : 0.0)
                c_val = hasproperty(row, :catch_total) ? row.catch_total : (hasproperty(row, :cumulative_catch) ? row.cumulative_catch : 0.0)
                rec = hasproperty(row, :recruits) ? row.recruits : 0.0
                spaw = hasproperty(row, :mature_females) ? row.mature_females : 0.0
                push!(lines_tbl, @sprintf("%4d | %18.1f | %18.1f | %10.1f | %8.0f | %8.0f",
                                         row.year, tot_b, leg_b, c_val, rec, spaw))
            end
            state.annual_table_text[] = join(lines_tbl, "\n")
        else
            state.annual_table_text[] = "Annual Summary Table: Awaiting Year 1 snapshot (day 274)"
        end
    catch
    end

    return nothing
end

"""
    update_functional_observables!(state::IBMGuiState, living::Vector{Organism})

Compute empirical maturity fractions and selectivity distributions from living
agents and stream points to interface 3 observables.
"""
function update_functional_observables!(state::IBMGuiState,
                                       living::Vector{Organism})
    if isempty(living)
        state.obs_cw_male[] = Float64[]
        state.obs_cw_female[] = Float64[]
        state.obs_cw_mature_male[] = Float64[]
        state.obs_cw_mature_female[] = Float64[]
        state.obs_cw_immature_male[] = Float64[]
        state.obs_cw_immature_female[] = Float64[]
        return nothing
    end

    bins = 20.0:10.0:150.0
    n_bins = length(bins) - 1

    # Empirical maturity ogive for males
    m_pts = Point2f[]
    f_pts = Point2f[]
    sel_pts = Point2f[]

    legal_sz = abmproperties(state.model)[:params].fishery.legal_size_male

    for k in 1:n_bins
        b_low = bins[k]
        b_high = bins[k + 1]
        mid = Float32((b_low + b_high) / 2.0)

        # Males
        m_in_bin = filter(a -> a.sex == male && b_low <= a.size < b_high, living)
        if !isempty(m_in_bin)
            p_mat = count(a -> a.is_mature, m_in_bin) / length(m_in_bin)
            push!(m_pts, Point2f(mid, Float32(p_mat)))

            # Realized selectivity fraction: legal males in bin
            p_sel = count(a -> a.size >= legal_sz, m_in_bin) / length(m_in_bin)
            push!(sel_pts, Point2f(mid, Float32(p_sel)))
        end

        # Females
        f_in_bin = filter(a -> a.sex == female && b_low <= a.size < b_high, living)
        if !isempty(f_in_bin)
            p_mat_f = count(a -> a.is_mature, f_in_bin) / length(f_in_bin)
            push!(f_pts, Point2f(mid, Float32(p_mat_f)))
        end
    end

    state.obs_ogive_male_pts[] = m_pts
    state.obs_ogive_female_pts[] = f_pts
    state.obs_selectivity_pts[] = sel_pts

    # Realized Stock-Recruitment Points, for the **male** component.
    #
    # Spawners are annual egg production *halved*. The panel follows one sex —
    # recruitment at the stage just before maturity onset, for males — and only half
    # of a clutch becomes male, so the denominator has to be the eggs that could
    # produce those males. Using the whole clutch would halve every ratio and bias
    # both the Ricker and the Beverton-Holt parameters by the same factor.
    #
    # `annual_egg_production`, not `total_egg_production`. The latter is named
    # "total" and described as cumulative, and a stock-recruit fit interprets the
    # spawner value as the production *in that year* — pairing a running total
    # against an annual recruit count. The GUI writes an annual value into both
    # columns, so the two only agree by accident; the correctly named one is used.
    #
    # Both axes are counts spanning several decades, so both are plotted as log10
    # and floored: a year with no recruitment is a real point at the origin of the
    # count scale, not a gap, and `log10(0)` is `-Inf`.
    sr_pts = Point2f[]
    sr_raw = Tuple{Float64, Float64}[]
    ann_df = state.annual_data
    eggs_col = (:annual_egg_production in propertynames(ann_df)) ?
               :annual_egg_production : :total_egg_production
    recruits_col = (:recruits_male in propertynames(ann_df)) ?
                   :recruits_male : :recruits
    if nrow(ann_df) >= 2 && (recruits_col in propertynames(ann_df)) &&
       (eggs_col in propertynames(ann_df))
        for yr in 1:(nrow(ann_df) - 1)
            s_raw = Float64(ann_df[!, eggs_col][yr]) / 2
            r_raw = Float64(ann_df[!, recruits_col][yr + 1])
            push!(sr_raw, (s_raw, r_raw))
            push!(sr_pts, Point2f(Float32(log10_count(s_raw)),
                Float32(log10_count(r_raw))))
        end
    end
    state.obs_sr_pts[] = sr_pts
    state.sr_raw = sr_raw

    update_sr_fit!(state)

    return nothing
end

"""
    update_sr_fit!(state::IBMGuiState)

Fit a Ricker stock-recruitment curve to `state.sr_raw` and stream it to the
panel's curve and annotation observables.

`R = a*S*exp(-b*S)`, fitted by `fit_stock_recruitment`, which linearises it as
`ln(R/S) = u - b*S`. That fit is in linear space even though the panel draws
log10 axes, which is why it is done on `state.sr_raw` rather than on the plotted
points.

The curve is evaluated on a spawner grid spanning the observed range and then
logged for display. Parameters are rounded to three significant digits: `a` is a
recruits-per-spawner ratio of order one while `b` is of order `1/S` and often
many decades smaller, so fixed decimals would print one of them as `0.000`.

Nothing is drawn until `fit_stock_recruitment` reports it actually fitted — it
returns a through-origin fallback below three usable points, and plotting that
fallback would put a confident-looking curve through two numbers.
"""
function update_sr_fit!(state::IBMGuiState)
    spawners = [s for (s, _) in state.sr_raw]
    recruits = [r for (_, r) in state.sr_raw]
    fit = fit_stock_recruitment(spawners, recruits)
    rk = fit.ricker
    bh = fit.beverton_holt

    if !fit.is_fitted
        state.obs_sr_fit_curve[] = Point2f[]
        state.obs_sr_fit_curve_bh[] = Point2f[]
        state.obs_sr_fit_text[] = ""
        return nothing
    end

    s_lo, s_hi = extrema(spawners)
    # Extend a little past the observed range so each curve's turnover is visible
    # rather than clipped at the last point.
    span = max(s_hi - s_lo, s_hi * 0.05, eps(Float64))
    grid_s = range(max(s_lo - 0.1 * span, 0.0), s_hi + 0.5 * span; length = 128)
    ricker = [Point2f(Float32(log10_count(s)), Float32(log10_count(rk.predict(s))))
              for s in grid_s if s > 0.0 && rk.predict(s) > 0.0]
    beverton = [Point2f(Float32(log10_count(s)), Float32(log10_count(bh.predict(s))))
                for s in grid_s if s > 0.0 && bh.predict(s) > 0.0]
    state.obs_sr_fit_curve[] = ricker
    state.obs_sr_fit_curve_bh[] = beverton

    # Both forms, with the better-supported one named. They answer the same
    # question — recruits per spawner — and routinely disagree about it: Ricker
    # has a stock-dependent, overcompensating form and Beverton-Holt saturates
    # smoothly, so on a short unexploited series the ranking is a genuine model
    # choice rather than a formality, and AIC is what settles it.
    a_r = round(rk.alpha; sigdigits = 3)
    b_r = round(rk.beta; sigdigits = 3)
    a_b = round(bh.alpha; sigdigits = 3)
    b_b = round(bh.beta; sigdigits = 8)
    lines = String[
        string("Ricker       R = ", a_r, "·S·exp(-", b_r, "·S)"),
        string("             R² = ", round(rk.R2; digits = 3),
            "   AIC = ", round(rk.aic; digits = 1)),
        string("Beverton–Holt R = ", a_b, "·S / (1 + ", b_b, "·S)"),
        string("             R² = ", round(bh.R2; digits = 3),
            "   AIC = ", round(bh.aic; digits = 1)),
    ]
    push!(lines, string("n = ", fit.n_obs,
        "   better: ", fit.best_model === :ricker ? "Ricker" : "Beverton–Holt"))
    state.obs_sr_fit_text[] = join(lines, '\n')
    return nothing
end

"""
    check_and_update_macro_models!(state, curr_t, n_tot, n_leg, n_fem,
                                   n_mal, b_leg_kg, cum_catch)

Detect annual snapshot intervals (Day 274 ~ September 1), register annual
demographic observations, and recalibrate the three macro-ecological models.
"""
function check_and_update_macro_models!(
    state::IBMGuiState,
    curr_t::Float64,
    n_tot::Int,
    n_leg::Int,
    n_fem::Int,
    n_mal::Int,
    b_leg_kg::Float64,
    cum_catch::Float64
)
    # Annual snapshot day comes from the configuration, not a literal: a run
    # configured for a different survey day must record on that day.
    snapshot_day = Float64(state.model.params.output.snapshot_day)
    dpy = state.model.params.simulation.days_per_year
    year_idx = floor(Int, curr_t / dpy) + 1
    target_snapshot_day = (year_idx - 1) * dpy + snapshot_day

    # Record annual production row if within step window of snapshot
    # and not already recorded for this year
    already_recorded = !isempty(state.annual_data) &&
                       any(state.annual_data.sample_index .== year_idx)

    if curr_t >= target_snapshot_day && !already_recorded
        # Compute interval catch for biomass (for records if needed)
        prev_catch_kg = isempty(state.annual_data) ? 0.0 :
                     last(state.annual_data.cumulative_catch)
        int_catch_kg = max(0.0, cum_catch - prev_catch_kg)

        # Compute interval catch in numbers (CRITICAL for Macro-models)
        cum_catch_num = Float64(abmproperties(state.model)[:total_catch_numbers])
        prev_catch_num = isempty(state.annual_data) ? 0.0 :
                         sum(state.annual_data.catch_total; init = 0.0)
        int_catch_num = max(0.0, cum_catch_num - prev_catch_num)

        # Compute interval recruits settled (CRITICAL for Stock-Recruitment)
        cum_rec = Float64(abmproperties(state.model)[:total_recruits_settled])
        prev_rec = isempty(state.annual_data) ? 0.0 :
                   sum(state.annual_data.recruits; init = 0.0)
        int_rec = max(0.0, cum_rec - prev_rec)

        # Live biomass comes from the same summarizer the CLI report uses, so the
        # GUI and the saved output cannot disagree.
        summary = summarize_population(state.model)

        # Julian day within the year: 1-based, so the last day of a 365-day year
        # reads as 365 rather than 0. `floor(curr_t) - (year_idx - 1) * dpy` is
        # exact at the year boundary, where `mod` would return 0.
        julian_day = Float64(clamp(
            floor(Int, curr_t) - (year_idx - 1) * round(Int, dpy), 1, round(Int, dpy)
        ))

        # Recruitment: the cohort immediately before maturity onset, counted by sex.
    #
    # Sex-specific, so it is counted among individuated `Organism` agents — an
    # `AggregateGroup` carries no `sex` field, and the old counter ran off the
    # aggregate transition into `benthic_anchor_stage`, a *size-model* threshold
    # ("first stage with sex-specific bounds"). That made "recruits settled" mean
    # "cohort members that advanced into the stage where the size formula changes",
    # which is neither settlement nor anything a reader would call recruitment.
    #
    # `recruits` is the female count, which is the classical stock-recruit pairing:
    # recruits per spawner. `recruits_male` is the same stage for males, used by the
    # fishable-component delay-difference fit, since the fishable stock is male.
    rec_stage = abmproperties(state.model)[:params].species.recruitment_stage
    rec_female = 0
    rec_male = 0
    for a in allagents(state.model)
        a.stage == rec_stage || continue
        a.sex === female && (rec_female += 1)
        a.sex === male && (rec_male += 1)
    end

    new_row = DataFrame(
            sample_index = [year_idx],
            day = [julian_day],
            time = [curr_t],
            total_numbers = [Float64(n_tot)],
            total_biomass_kg = [Float64(summary.total_biomass_kg)],
            legal_numbers = [Float64(n_leg)],
            mature_females = [Float64(n_fem)],
            mature_males = [Float64(n_mal)],
            fishable_male_biomass_kg = [b_leg_kg],
            mature_male_biomass_kg = [summary.total_mature_male_mass_kg],
            mature_female_biomass_kg = [summary.total_mature_female_mass_kg],
            total_egg_production = [Float64(abmproperties(state.model)[:annual_eggs_spawned])],
            annual_egg_production = [Float64(abmproperties(state.model)[:annual_eggs_spawned])],
            recruits = [Float64(rec_female)],
            recruits_male = [Float64(rec_male)],
            catch_total = [int_catch_num],
            cumulative_catch = [cum_catch],
            catch_legal = [int_catch_num],
            surplus_production_total = [0.0],
            surplus_production_legal = [0.0],
            # Realised mean wet weight per component, so the number-to-biomass
            # conversion applied to the macro-model trajectories is recorded with
            # the data rather than being implicit in the plotting code. Taken from
            # the history the update step has just appended, so there is one
            # computation of these rather than two that could disagree.
            mean_weight_legal_kg = [last(state.mean_weight_legal_history)],
            mean_weight_male_kg = [last(state.mean_weight_male_history)],
            mean_weight_female_kg = [last(state.mean_weight_female_history)],
            mean_weight_total_kg = [last(state.mean_weight_total_history)]
        )
        append!(state.annual_data, new_row)

        # Trigger macro-model recalibration if we have >= 4 annual periods
        if nrow(state.annual_data) >= 4
            recalibrate_macro_models!(state)
        end
    end

    return nothing
end

"""
    recalibrate_macro_models!(state::IBMGuiState)

Fit Schaefer logistic surplus production, Deriso-Schnute delay-difference,
and Generalized Leslie matrix models against the accumulated annual time series.
"""
function recalibrate_macro_models!(state::IBMGuiState)
    ann_df = state.annual_data
    T = nrow(ann_df)
    if T < 4
        return nothing
    end

    try
        # Extract stage transition matrices and fit Leslie model
        stage_mat, catch_mat, yrs = extract_annual_stage_matrix(ann_df)
        leslie_fit = fit_generalized_leslie_model(stage_mat, catch_mat; years = yrs)
        macro_comp = compare_macro_models(ann_df, leslie_fit; lag = 1)

        # Update macro-model trajectory observables.
        #
        # The observed series is the *measured* annual legal-biomass sum, not
        # `macro_comp.biomass_obs_kg`. The latter is what the macro models are
        # calibrated in: legal abundance times the stage-*mean* allometric mass,
        # which runs about 12% below the true sum of individual masses. Plotting
        # it as "Annual Survey Biomass" put two different quantities on the same
        # axes, so the survey markers could never sit on the fishable-biomass line
        # even though both are legal biomass. x alignment was never the problem.
        measured = if :fishable_male_biomass_kg in propertynames(ann_df)
            Float64.(ann_df.fishable_male_biomass_kg)
        else
            Float64.(macro_comp.biomass_obs_kg)
        end
        obs_pts = [Point2f(ann_df.time[i], measured[i]) for i in 1:T]
        # The macro-model fits are in NUMBER, not biomass: `fit_population_dynamics`
        # passes `legal_numbers` and `catch_legal`, and the Leslie fit works on a
        # per-stage count matrix. `compare_macro_models` also returns a `biomass_*`
        # series, but that one is rescaled by a single pooled *stage-mean* legal
        # mass, which sits below the real mean weight of the legal males present
        # (and below the measured sum, because a stage mean ignores the within-stage
        # size spread). For this figure the conversion is done here instead:
        # numbers x the realised mean weight of the component, per snapshot.
        mw_col = if :mean_weight_legal_kg in propertynames(ann_df)
            Float64.(ann_df.mean_weight_legal_kg)
        else
            fill(macro_comp.mean_component_weight_kg, T)
        end
        to_biomass(nums) = [Float64(nums[i]) * mw_col[i] for i in 1:T]

        sch_kg = to_biomass(macro_comp.numbers_schaefer)
        dd_kg = to_biomass(macro_comp.numbers_delay_diff)
        les_kg = to_biomass(macro_comp.numbers_leslie)

        sch_pts = [Point2f(ann_df.time[i], sch_kg[i]) for i in 1:T]
        dd_pts  = [Point2f(ann_df.time[i], dd_kg[i]) for i in 1:T]
        les_pts = [Point2f(ann_df.time[i], les_kg[i]) for i in 1:T]

        state.macro_obs_biomass[] = obs_pts
        state.macro_sch_biomass[] = sch_pts
        state.macro_dd_biomass[]  = dd_pts
        state.macro_les_biomass[] = les_pts

        # Legend entries carry the key estimated parameters, rebuilt on each
        # re-fit so they track the current calibration.
        state.macro_label_observed[] = "Annual Surveyed Biomass"
        state.macro_label_sch[] = macro_comp.label_schaefer
        state.macro_label_dd[] = macro_comp.label_delay_diff
        state.macro_label_les[] = macro_comp.label_leslie
        # The conversion actually applied, so a reader can see what "biomass" means
        # on this panel.
        mw_now = mw_col[end]
        state.macro_conversion_text[] = "kg = numbers × $(round(mw_now, digits=4)) kg mean legal-male weight"

        pref = uppercase(string(macro_comp.preferred_model))
        r2_sch = round(macro_comp.r2_schaefer, digits = 2)
        r2_dd  = round(macro_comp.r2_delay_diff, digits = 2)
        r2_les = round(macro_comp.r2_leslie, digits = 2)

        state.macro_metrics_text[] = "Preferred: $(pref) | " *
            "Schaefer R²: $(r2_sch) | Delay-Diff R²: $(r2_dd) | " *
            "Leslie Matrix R²: $(r2_les)"
    catch err
        # Gracefully handle non-invertible matrix edge cases during early burn-in
        @warn "Macro-model calibration deferred: $(err)"
    end

    return nothing
end

"""
    with_overrides(::Type{T}, base::T; kwargs...) where {T}

Return a copy of the immutable parameter struct `base` with the supplied keyword
arguments applied. Fields absent from `kwargs` are carried over unchanged, so
callers never need to restate the full field list.
"""
function with_overrides(::Type{T}, base::T; kwargs...) where {T}
    values = Any[get(kwargs, field, getfield(base, field)) for field in fieldnames(T)]
    return T(values...)
end

"""
    reconstruct_params(p::IBMParameters; kwargs...)

Reconstruct an immutable `IBMParameters` hierarchy with modified parameters.
Any keyword matching a field of any nested parameter struct is applied. Derived
from `parameter_structs()`, so a newly added parameter struct is carried through
without editing this function.
"""
function reconstruct_params(p::IBMParameters; kwargs...)
    # Derived from `parameter_structs()` rather than hand-listing the fields: the
    # hand-written version silently dropped any struct added later, which is how
    # the GUI's Reset button lost a parameter without any test noticing.
    nested = Dict{Symbol, Any}()
    for (field, struct_type, _) in parameter_structs()
        nested[field] = with_overrides(struct_type, getfield(p, field); kwargs...)
    end
    return IBMParameters(; nested...)
end

# ------------------------------------------------------------------------------
# GUI Construction and View Layouts
# ------------------------------------------------------------------------------

"""
    launch_organism_gui(; display_window = true, kwargs...)

Build the interactive GUI from `default_ibm_configuration()` overridden by
`kwargs`. Keyword names must match the declared parameter names exactly, such as
`grid`, `dt`, and `seed`.
"""

"""
    setup_stage_editor_panel!(ax, stage_dist_obs, stage_labels)

Populate the interactive initial stage abundance stem plot editor.
"""
function setup_stage_editor_panel!(ax, stage_dist_obs, stage_labels)
    n_stages = length(stage_labels)
    stem!(ax, 1:n_stages, stage_dist_obs; color = :steelblue, linewidth = 2)
    scatter!(ax, 1:n_stages, stage_dist_obs; color = :crimson, markersize = 6)
    ylims!(ax, 0, nothing)
end

"""
    setup_env_dist_panel!(ax, depth_pts, temp_pts, hsi_pts)

Populate empirical environmental driver distribution curves across the grid.
"""
function setup_env_dist_panel!(ax, depth_pts, temp_pts, hsi_pts)
    lines!(ax, depth_pts, color = :royalblue, linewidth = 2.0, label = "Depth (norm)")
    lines!(ax, temp_pts, color = :firebrick, linewidth = 2.0, label = "Temp (°C/10)")
    lines!(ax, hsi_pts, color = :seagreen, linewidth = 2.0, label = "HSI")
    axislegend(ax, position = :rt, labelsize = 9, framevisible = false)
end

"""
    setup_criteria_panel!(ax, criteria_text_obs)

Display multi-attribute calibration evaluation criteria in a monospaced block
bounded strictly within `ax`.
"""
function setup_criteria_panel!(ax, criteria_text_obs)
    hidedecorations!(ax)
    xlims!(ax, 0.0, 1.0)
    ylims!(ax, 0.0, 1.0)
    poly!(
        ax,
        Point2f[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        color = RGBf(0.97, 0.98, 0.99),
        strokecolor = RGBf(0.80, 0.84, 0.88),
        strokewidth = 1.0
    )
    text!(
        ax, 0.03, 0.92;
        text = criteria_text_obs,
        space = :relative,
        align = (:left, :top),
        fontsize = 11,
        font = "monospace",
        color = :black
    )
    return nothing
end

"""
    setup_annual_table_panel!(ax, table_text_obs)

Display formatted tabular annual production metrics in a monospaced block
bounded strictly within `ax`.
"""
function setup_annual_table_panel!(ax, table_text_obs)
    hidedecorations!(ax)
    xlims!(ax, 0.0, 1.0)
    ylims!(ax, 0.0, 1.0)
    poly!(
        ax,
        Point2f[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        color = RGBf(0.97, 0.98, 0.99),
        strokecolor = RGBf(0.80, 0.84, 0.88),
        strokewidth = 1.0
    )
    text!(
        ax, 0.03, 0.92;
        text = table_text_obs,
        space = :relative,
        align = (:left, :top),
        fontsize = 10,
        font = "monospace",
        color = :black
    )
    return nothing
end

const GUI_AOI_LON_BOUNDS = (-66.5, -57.5)
const GUI_AOI_LAT_BOUNDS = (42.0, 48.5)
const GUI_AOI_BATHYMETRY_CACHE = Ref{Union{Nothing, NamedTuple}}(nothing)


function load_gui_aoi_bathymetry()
    GUI_AOI_BATHYMETRY_CACHE[] !== nothing && return GUI_AOI_BATHYMETRY_CACHE[]

    local_interp = nothing
    local_source = "scotian_shelf_regional"
    candidates = [
        tethysbio_data_path("environmental", "bathymetry.parquet"),
        tethysbio_data_path("environmental", "bathymetry.zarr"),
        tethysbio_data_path("environmental", "bathymetry.nc"),
        tethysbio_data_path("inputs", "bathymetry.parquet"),
        tethysbio_data_path("inputs", "bathymetry.zarr"),
        tethysbio_data_path("inputs", "bathymetry.nc")
    ]
    for path in candidates
        ispath(path) || continue
        try
            dataset = occursin(".nc", path) ?
                load_environmental_dataset(path; backend = :ncdatasets) :
                load_environmental_dataset(path)
            local_interp = environmental_bathymetry_interpolator(dataset)
            local_source = path
            break
        catch err
            @warn "Could not load local bathymetry for the AOI map" path exception = err
        end
    end

    lons = collect(range(GUI_AOI_LON_BOUNDS[1], GUI_AOI_LON_BOUNDS[2]; length = 220))
    lats = collect(range(GUI_AOI_LAT_BOUNDS[1], GUI_AOI_LAT_BOUNDS[2]; length = 160))

    combined_interpolator = function (lon::Real, lat::Real)
        if local_interp !== nothing
            val = try
                Float64(local_interp(lon, lat))
            catch
                NaN
            end
            isfinite(val) && return val
        end
        return scotian_shelf_regional_elevation(lon, lat)
    end

    depths = Matrix{Float64}(undef, length(lons), length(lats))
    for j in eachindex(lats), i in eachindex(lons)
        elevation = combined_interpolator(lons[i], lats[j])
        depths[i, j] = isfinite(elevation) ? max(0.0, -Float64(elevation)) : 0.0
    end

    bathymetry = (
        interpolator = combined_interpolator,
        lons = lons,
        lats = lats,
        depths = depths,
        source = local_source
    )
    GUI_AOI_BATHYMETRY_CACHE[] = bathymetry
    return bathymetry
end

function gui_aoi_domain(lon::Real, lat::Real; span_meters::Real = 50_000.0)
    half_m = Float64(span_meters) / 2
    dlat = half_m / 111139.0
    dlon = half_m / max(1000.0, 111139.0 * cosd(Float64(lat)))
    return ([Float64(lon) - dlon, Float64(lon) + dlon],
            [Float64(lat) - dlat, Float64(lat) + dlat])
end

function gui_aoi_depth_map(bathymetry, lon::Real, lat::Real, grid::Tuple{Int, Int};
                           span_meters::Real = 50_000.0)
    bathymetry === nothing && return nothing
    lon_bounds, lat_bounds = gui_aoi_domain(lon, lat; span_meters = span_meters)
    lons = range(lon_bounds...; length = grid[1])
    lats = range(lat_bounds...; length = grid[2])
    depths = Matrix{Float64}(undef, grid)
    for j in eachindex(lats), i in eachindex(lons)
        elevation = bathymetry.interpolator(lons[i], lats[j])
        depths[i, j] = isfinite(elevation) ? max(0.0, -Float64(elevation)) : 0.0
    end
    return depths
end

function launch_organism_gui(; display_window::Bool = true,
    config_path::AbstractString = "", kwargs...)
    # A `config_path` that names a readable file is the base configuration, not
    # merely the file the annual rescan re-reads. It used to start from
    # `default_ibm_configuration()` unconditionally, so pointing the dashboard at a
    # TOML loaded nothing from it — the initial abundance distribution, the stage
    # schedules and every other declared value were silently ignored, and the panel
    # ran on struct defaults instead.
    config = if isempty(config_path) || !isfile(config_path)
        default_ibm_configuration()
    else
        load_ibm_configuration(String(config_path))
    end
    # `n_agents` is not a configuration key — it is the total the seeding vectors
    # sum to — so it is kept out of `config` and passed straight through to
    # `initialize_model`. It cannot be `pop!`ed: `kwargs` is an immutable
    # `Base.Pairs`, so it is skipped in the merge instead.
    n_agents = get(kwargs, :n_agents, nothing)
    for (key, value) in kwargs
        key === :n_agents && continue
        config[Symbol(key)] = value
    end
    sync_grid_dimensions!(config; explicit = Set(Symbol.(keys(kwargs))))
    return launch_organism_gui(config; display_window = display_window,
        config_path = String(config_path), n_agents = n_agents)
end

"""
    launch_organism_gui(config::Dict{Symbol, Any}; display_window = true,
                         config_path = "")

Build and display the interactive, three-dashboard GUI for `organism_ibm.jl`
using the provided configuration dictionary.

# Arguments
- `config::Dict{Symbol, Any}`: Configuration dictionary containing all simulation parameters.
- `display_window::Bool`: Set to `true` to render native OpenGL window.
- `config_path::String`: Path to the TOML config file. If provided, the GUI will
  rescan this file every simulation year to pick up user edits.

# Returns
`Tuple{Figure, IBMGuiState}`: Makie figure handle and GUI state container.
"""
function launch_organism_gui(config::Dict{Symbol, Any}; display_window::Bool = true,
    config_path::String = "", n_agents::Union{Int, Nothing} = nothing)
    # Activate GLMakie for high-performance interactive OpenGL rendering
    GLMakie.activate!()

    # 1. Initialize the regional bathymetry and simulation model
    aoi_bathymetry = load_gui_aoi_bathymetry()
    initial_aoi_lon = -59.0
    initial_aoi_lat = 45.0
    initial_aoi_domain = gui_aoi_domain(initial_aoi_lon, initial_aoi_lat)
    params = IBMParameters(config)
    params = reconstruct_params(
        params;
        domain_lon = initial_aoi_domain[1],
        domain_lat = initial_aoi_domain[2]
    )
    initial_depth_map = gui_aoi_depth_map(
        aoi_bathymetry, initial_aoi_lon, initial_aoi_lat, config[:grid]
    )

    model = initialize_model(
        grid = config[:grid],
        dt = config[:dt],
        params = params,
        seed = config[:seed],
        depth_map = initial_depth_map,
        # The initial abundance is three vectors in the configuration. `n_agents` is
        # the sum of the two sex vectors when it is not given, and an override when it
        # is; if both vectors are empty, defaults to 120 for interactive exploration.
        n_agents = (n_agents !== nothing) ? n_agents :
            (sum(params.simulation.initial_female_cohort) + sum(params.simulation.initial_male_cohort) > 0 ? nothing : 120)
    )

    living_init = collect(allagents(model))
    xs_init = Float32[Float32(a.pos[1]) for a in living_init]
    ys_init = Float32[Float32(a.pos[2]) for a in living_init]
    sz_init = Float32[Float32(min(5.5, max(2.0, 1.8 + a.size * 0.03))) for a in living_init]
    col_init = String[organism_color(a, params) for a in living_init]

    agg_init = get(abmproperties(model), :aggregate_populations, nothing)
    early_init = zeros(Float64, length(GUI_EARLY_STAGES))
    if agg_init !== nothing
        for g in agg_init.groups
            i = findfirst(==(g.stage), GUI_EARLY_STAGES)
            i === nothing || (early_init[i] += g.abundance)
        end
    end

    # 2. Construct IBMGuiState Container
    #
    # Keyword form throughout: see the `@kwdef` note on the struct. Every field is
    # named, so the compiler rejects a misspelled or missing one at the call site
    # instead of the failure surfacing as an unreadable signature mismatch.
    state = IBMGuiState(
        model = model,
        is_running = Observable(false),
        sim_speed_days = Observable(5.0),
        current_time = Observable(0.0),
        status_text = Observable("Ready — Day 0"),
        obs_env_field = Observable(copy(model.depth_field)),
        obs_bg_layer = Observable(:depth),
        obs_agent_x = Observable(xs_init),
        obs_agent_y = Observable(ys_init),
        obs_agent_pts = Observable([Point2f(xs_init[i], ys_init[i]) for i in eachindex(xs_init)]),
        obs_agent_sizes = Observable(sz_init),
        obs_agent_colors = Observable(col_init),
        t_history = Float64[0.0],
        n_total_history = Float64[length(living_init)],
        n_females_history = Float64[0.0],
        n_males_history = Float64[0.0],
        n_legal_history = Float64[0.0],
        biomass_legal_history = Float64[0.0],
        catch_history = Float64[0.0],
        mean_weight_legal_history = Float64[0.0],
        mean_weight_male_history = Float64[0.0],
        mean_weight_female_history = Float64[0.0],
        mean_weight_total_history = Float64[0.0],
        catch_at_year_start = 0.0,
        current_year_idx = 1,
        config_path = config_path,
        # Seeded from the file so the first scan reports only what changes after
        # launch, rather than treating every declared parameter as an edit.
        config_snapshot = isempty(config_path) || !isfile(config_path) ?
            Dict{String, Any}() : flatten_toml(TOML.parsefile(config_path)),
        obs_ts_total = Observable(Point2f[(0.0, log10_count(length(living_init)))]),
        obs_ts_females = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_males = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_legal = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_biomass = Observable(Point2f[(0.0, 0.0)]),
        obs_ts_catch = Observable(Point2f[(0.0, 0.0)]),
        n_egg_history = Float64[0.0],
        n_early_stage_history = [Float64[early_init[i]] for i in eachindex(GUI_EARLY_STAGES)],
        n_early_agents_history = NTuple{3, Float64}[(0.0, 0.0, 0.0)],
        obs_current_egg = Observable(0.0),
                obs_current_agents = Observable((0.0, 0.0, 0.0)),
        # Stage 0 is the brooded egg, which is not a cohort, so the aggregate series
        # covers stages 1 to `juvenile_max_stage - 1` and is indexed by stage.
        obs_stage_agg = Observable(zeros(Float64, params.organism.juvenile_max_stage - 1)),
        obs_stage_male = Observable(zeros(Float64,
            params.organism.max_stage_male - params.organism.juvenile_max_stage + 1)),
        obs_stage_female = Observable(zeros(Float64,
            params.organism.max_stage_female - params.organism.juvenile_max_stage + 1)),
        obs_ts_egg = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_stage = [Observable(Point2f[(0.0, log10_count(early_init[i]))])
                               for i in eachindex(GUI_EARLY_STAGES)],
        obs_ts_early_agents = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_agents_f = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_ts_early_agents_u = Observable(Point2f[(0.0, log10_count(0.0))]),
        obs_cw_male = Observable(Float32[Float32(a.size) for a in living_init if a.sex == male]),
        obs_cw_female = Observable(Float32[Float32(a.size) for a in living_init if a.sex == female]),
        obs_cw_mature_male = Observable(Float32[Float32(a.size)
                                        for a in living_init if a.sex == male && a.is_mature]),
        obs_cw_mature_female = Observable(Float32[Float32(a.size)
                                          for a in living_init if a.sex == female && a.is_mature]),
        obs_cw_immature_male = Observable(Float32[Float32(a.size)
                                          for a in living_init if a.sex == male && !a.is_mature]),
        obs_cw_immature_female = Observable(Float32[Float32(a.size)
                                            for a in living_init if a.sex == female && !a.is_mature]),
        obs_ogive_male_pts = Observable(Point2f[]),
        obs_ogive_female_pts = Observable(Point2f[]),
        obs_selectivity_pts = Observable(Point2f[]),
        obs_sr_pts = Observable(Point2f[]),
        sr_raw = Tuple{Float64, Float64}[],
        obs_sr_fit_curve = Observable(Point2f[]),
        obs_sr_fit_curve_bh = Observable(Point2f[]),
        obs_sr_fit_text = Observable(""),
        annual_data = create_empty_annual_data(),
        macro_obs_biomass = Observable(Point2f[]),
        macro_sch_biomass = Observable(Point2f[]),
        macro_dd_biomass = Observable(Point2f[]),
        macro_les_biomass = Observable(Point2f[]),
        macro_metrics_text = Observable("Macro models awaiting annual data (>= 4 years)"),
        macro_label_observed = Observable("Annual Surveyed Biomass"),
        macro_label_sch = Observable("Schaefer"),
        macro_label_dd = Observable("Delay-Difference"),
        macro_label_les = Observable("Leslie Matrix"),
        macro_conversion_text = Observable(""),
        snapshots = Dict{Int, Any}(),
        snapshot_labels = Observable(String["Year 0"]),
    )
    
    # Save year 0 initial state
    state.snapshots[0] = (
        model = deepcopy(model),
        t_history = copy(state.t_history),
        n_total_history = copy(state.n_total_history),
        n_females_history = copy(state.n_females_history),
        n_males_history = copy(state.n_males_history),
        n_legal_history = copy(state.n_legal_history),
        biomass_legal_history = copy(state.biomass_legal_history),
        catch_history = copy(state.catch_history),
        annual_data = copy(state.annual_data)
    )

    # 3. Create Main Makie Figure Window with 2-Row Top-Tab Architecture
    fig = Figure(size = DASHBOARD_SIZE, figure_padding = (14, 14, 10, 10))

    # Top Header & Transport Controls (Row 1 — anchored strictly at top of window)
    top_bar = fig[1, 1] = GridLayout(tellwidth = false, halign = :left, valign = :center)
    top_bar.default_colgap = Fixed(4)

    btn_tab1 = Button(top_bar[1, 1], label = "1. Controls & AOI",
                      fontsize = 10, width = 120, height = 28)
    btn_tab2 = Button(top_bar[1, 2], label = "2. Simulation & Habitat",
                      fontsize = 10, width = 140, height = 28)
    btn_tab3 = Button(top_bar[1, 3], label = "3. Demographics & Growth",
                      fontsize = 10, width = 150, height = 28)
    btn_tab4 = Button(top_bar[1, 4], label = "4. Assessment & Calibration",
                      fontsize = 10, width = 160, height = 28)
    tab_buttons = [btn_tab1, btn_tab2, btn_tab3, btn_tab4]

    btn_quick_run = Button(top_bar[1, 5],
                           label = @lift($(state.is_running) ? "Pause" : "Run"),
                           fontsize = 10, width = 50, height = 28)
    btn_quick_step = Button(top_bar[1, 6], label = "Step",
                            fontsize = 10, width = 45, height = 28)
    btn_quick_reset = Button(top_bar[1, 7], label = "Reset",
                             fontsize = 10, width = 50, height = 28)

    Label(top_bar[1, 8], "Step (d):", fontsize = 10, halign = :right)
    step_tb = Textbox(top_bar[1, 9], stored_string = "10",
                      validator = Float64, width = 36, fontsize = 10)

    Label(top_bar[1, 10], "Duration (y):", fontsize = 10, halign = :right)
    years_tb = Textbox(top_bar[1, 11], stored_string = "2",
                       validator = Int, width = 30, fontsize = 10)

    Label(top_bar[1, 12], "Rewind (d):", fontsize = 10, halign = :right)
    rewind_tb = Textbox(top_bar[1, 13], stored_string = "10",
                        validator = Int, width = 36, fontsize = 10)
    btn_rewind = Button(top_bar[1, 14], label = "Rewind",
                        width = 52, height = 28, fontsize = 10)

    Label(top_bar[1, 15], state.status_text, fontsize = 10, font = :bold,
          tellwidth = false, halign = :left)

    # Main Content Area (Row 2)
    main_area = fig[2, 1] = GridLayout(tellwidth = false, tellheight = false,
                                       halign = :left, valign = :top)
    rowsize!(fig.layout, 1, Fixed(42))
    rowsize!(fig.layout, 2, Auto())
    rowgap!(fig.layout, 1, Fixed(8))

    # Dedicated Controls & AOI Setup Panel on Tab 1
    controls_tab_grid = main_area[1, 1] = GridLayout(
        1, 1, tellwidth = false, tellheight = false, halign = :left, valign = :top
    )
    colsize!(controls_tab_grid, 1, Fixed(280))
    controls_tab_grid.default_colgap = Fixed(12)
    controls_tab_grid.default_rowgap = Fixed(8)

    # Dedicated Simulation Controls Panel on Tab 2 (Below spatial representation)
    tab2_controls_grid = main_area[1, 1] = GridLayout(
        1, 2, tellwidth = false, tellheight = false, halign = :left, valign = :bottom
    )
    colsize!(tab2_controls_grid, 1, Fixed(230))
    colsize!(tab2_controls_grid, 2, Fixed(230))
    tab2_controls_grid.default_colgap = Fixed(14)
    tab2_controls_grid.default_rowgap = Fixed(6)

    # Dedicated Assessment & Calibration Toolbar on Tab 4
    tab4_toolbar = main_area[1, 1] = GridLayout(
        tellwidth = false, tellheight = false, halign = :left, valign = :top
    )
    tab4_toolbar.default_colgap = Fixed(12)
    btn_refit = Button(
        tab4_toolbar[1, 1], label = "Refit Macro",
        fontsize = 11, width = 120, height = 28
    )
    Label(
        tab4_toolbar[1, 2],
        "Fit Schaefer, Delay-Difference & Leslie Models to Annual Surveyed Biomass",
        fontsize = 10, halign = :left
    )

    ctrl_col1 = controls_tab_grid[1, 1] = GridLayout(
        tellwidth = false, tellheight = false, valign = :top
    )
    ctrl_col1.default_rowgap = Fixed(6)

    tab2_col1 = tab2_controls_grid[1, 1] = GridLayout(
        tellwidth = false, tellheight = false, valign = :top
    )
    tab2_col1.default_rowgap = Fixed(4)
    tab2_col2 = tab2_controls_grid[1, 2] = GridLayout(
        tellwidth = false, tellheight = false, valign = :top
    )
    tab2_col2.default_rowgap = Fixed(4)

    # --------------------------------------------------------------------------
    # Tab 1: Area of Interest (AOI) Configuration
    # --------------------------------------------------------------------------
    Label(ctrl_col1[1, 1], "Area of Interest (AOI) Configuration",
          font = :bold, fontsize = 12, halign = :left)
    aoi_grid = ctrl_col1[2, 1] = GridLayout(tellwidth = false)
    aoi_grid.default_rowgap = Fixed(4)
    aoi_grid.default_colgap = Fixed(8)
    Label(aoi_grid[1, 1], "Longitude (°W):", halign = :right, fontsize = 10)
    lon_tb = Textbox(aoi_grid[1, 2], stored_string = "-59.0",
                     validator = Float64, width = 85, fontsize = 10)
    Label(aoi_grid[2, 1], "Latitude (°N):", halign = :right, fontsize = 10)
    lat_tb = Textbox(aoi_grid[2, 2], stored_string = "45.0",
                     validator = Float64, width = 85, fontsize = 10)
    Label(aoi_grid[3, 1], "Grid Cell Unit (m):", halign = :right, fontsize = 10)
    unit_tb = Textbox(aoi_grid[3, 2],
                      stored_string = string(round(state.obs_unit_meters[], digits = 1)),
                      validator = Float64, width = 85, fontsize = 10)
    Label(aoi_grid[4, 1], state.aoi_depth_str, fontsize = 10, halign = :left)
    Label(aoi_grid[4, 2], state.aoi_temp_str, fontsize = 10, halign = :left)
    Label(aoi_grid[5, 1:2], state.aoi_hsi_str, fontsize = 10,
          halign = :left, font = :bold)

    # Presets Section on Tab 1
    Label(ctrl_col1[3, 1], "Presets", font = :bold, fontsize = 11, halign = :left)
    presets_grid = ctrl_col1[4, 1] = GridLayout(2, 2, tellwidth = false)
    presets_grid.default_colgap = Fixed(6)
    presets_grid.default_rowgap = Fixed(4)
    btn_pre_laurentian = Button(presets_grid[1, 1], label = "Laurentian",
                                fontsize = 10, width = 125, height = 26)
    btn_pre_sable = Button(presets_grid[1, 2], label = "Sable Bank",
                           fontsize = 10, width = 125, height = 26)
    btn_pre_banquereau = Button(presets_grid[2, 1], label = "Banquereau",
                                fontsize = 10, width = 125, height = 26)
    btn_pre_emerald = Button(presets_grid[2, 2], label = "Emerald",
                             fontsize = 10, width = 125, height = 26)

    # Persistence Buttons on Tab 1
    export_grid = ctrl_col1[5, 1] = GridLayout(1, 2, tellwidth = false)
    export_grid.default_colgap = Fixed(6)
    btn_export_toml = Button(export_grid[1, 1], label = "Export TOML",
                             width = 125, height = 28, fontsize = 10)
    btn_save_parquet = Button(export_grid[1, 2], label = "Save Lakehouse",
                              width = 125, height = 28, fontsize = 10)

    # --------------------------------------------------------------------------
    # Tab 2 Controls: Simulation & Environmental Controls (Col 1)
    # --------------------------------------------------------------------------
    Label(tab2_col1[1, 1], "Simulation & Environmental Controls",
          font = :bold, fontsize = 11, halign = :left)
    sim_env_grid = tab2_col1[2, 1] = GridLayout(9, 2, tellwidth = false)
    sim_env_grid.default_colgap = Fixed(6)
    sim_env_grid.default_rowgap = Fixed(3)

    Label(sim_env_grid[1, 1], "Grid Resolution (cells):", halign = :right, fontsize = 9)
    grid_tb = Textbox(sim_env_grid[1, 2], stored_string = string(config[:grid][1]),
                      validator = Int, width = 60, fontsize = 9)

    Label(sim_env_grid[2, 1], "Grid Unit / Cell Size (m):", halign = :right, fontsize = 9)
    unit_tb_tab2 = Textbox(sim_env_grid[2, 2],
                           stored_string = string(round(state.obs_unit_meters[], digits = 1)),
                           validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[3, 1], "Base Bottom Temp Offset (°C):", halign = :right, fontsize = 9)
    temp_offset_tb = Textbox(sim_env_grid[3, 2],
                            stored_string = string(params.environment.temp_offset),
                            validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[4, 1], "Seasonal Temp Multiplier:", halign = :right, fontsize = 9)
    temp_mult_tb = Textbox(sim_env_grid[4, 2],
                          stored_string = string(params.environment.temp_multiplier),
                          validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[5, 1], "Habitat Suitability Offset:", halign = :right, fontsize = 9)
    hsi_offset_tb = Textbox(sim_env_grid[5, 2],
                           stored_string = string(params.environment.habitat_offset),
                           validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[6, 1], "Habitat Multiplier:", halign = :right, fontsize = 9)
    hsi_mult_tb = Textbox(sim_env_grid[6, 2],
                         stored_string = string(params.environment.habitat_multiplier),
                         validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[7, 1], "Food Availability Multiplier:", halign = :right, fontsize = 9)
    food_tb = Textbox(sim_env_grid[7, 2],
                      stored_string = string(params.environment.food_availability),
                      validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[8, 1], "Carrying Capacity Density (/km²):", halign = :right, fontsize = 9)
    cap_density_tb = Textbox(sim_env_grid[8, 2],
                            stored_string = string(params.environment.benthic_carrying_capacity_density),
                            validator = Float64, width = 60, fontsize = 9)

    Label(sim_env_grid[9, 1], "Spatial Map Background Layer:", halign = :right, fontsize = 9)
    layer_menu = Menu(
        sim_env_grid[9, 2],
        options = ["Seabed Depth (m)", "Bottom Temperature (°C)", "Habitat Quality (HSI)"],
        default = "Seabed Depth (m)",
        width = 135,
        fontsize = 9
    )

    # --------------------------------------------------------------------------
    # Tab 2 Controls: Commercial Fishing, Biological Processes & Spatial Exclusion (Col 2)
    # --------------------------------------------------------------------------
    Label(tab2_col2[1, 1], "Commercial Fishing Activity",
          font = :bold, fontsize = 11, halign = :left)
    fish_grid = tab2_col2[2, 1] = GridLayout(4, 2, tellwidth = false)
    fish_grid.default_colgap = Fixed(6)
    fish_grid.default_rowgap = Fixed(3)

    Label(fish_grid[1, 1], "Fishing Season Start (d):", halign = :right, fontsize = 9)
    fstart_tb = Textbox(fish_grid[1, 2],
                        stored_string = string(params.fishery.fishing_season_start),
                        validator = Float64, width = 60, fontsize = 9)

    Label(fish_grid[2, 1], "Fishing Season End (d):", halign = :right, fontsize = 9)
    fend_tb = Textbox(fish_grid[2, 2],
                      stored_string = string(params.fishery.fishing_season_end),
                      validator = Float64, width = 60, fontsize = 9)

    Label(fish_grid[3, 1], "Exploitation Rate (daily):", halign = :right, fontsize = 9)
    frate_tb = Textbox(fish_grid[3, 2],
                       stored_string = string(params.fishery.fishing_probability_daily),
                       validator = Float64, width = 60, fontsize = 9)

    Label(fish_grid[4, 1], "Legal Minimum Size (mm):", halign = :right, fontsize = 9)
    legal_size_tb = Textbox(fish_grid[4, 2],
                            stored_string = string(params.fishery.legal_size_male),
                            validator = Float64, width = 60, fontsize = 9)

    Label(tab2_col2[3, 1], "Biological Processes",
          font = :bold, fontsize = 11, halign = :left)
    tog_grid = tab2_col2[4, 1] = GridLayout(3, 4, tellwidth = false)
    tog_grid.default_colgap = Fixed(6)
    tog_grid.default_rowgap = Fixed(3)

    tog_energetics = Toggle(tog_grid[1, 1], active = params.toggles.use_energetics)
    Label(tog_grid[1, 2], "Energetics", fontsize = 9, halign = :left)
    tog_thermal = Toggle(tog_grid[1, 3], active = params.toggles.use_thermal_stress)
    Label(tog_grid[1, 4], "Thermal Stress", fontsize = 9, halign = :left)
    tog_cannibalism = Toggle(tog_grid[2, 1], active = params.toggles.use_cannibalism)
    Label(tog_grid[2, 2], "Cannibalism", fontsize = 9, halign = :left)
    tog_senescence = Toggle(tog_grid[2, 3], active = params.toggles.use_senescence)
    Label(tog_grid[2, 4], "Senescence", fontsize = 9, halign = :left)
    tog_movement = Toggle(tog_grid[3, 1], active = params.toggles.use_movement)
    Label(tog_grid[3, 2], "Movement", fontsize = 9, halign = :left)
    tog_reproduction = Toggle(tog_grid[3, 3], active = params.toggles.use_reproduction)
    Label(tog_grid[3, 4], "Reproduction", fontsize = 9, halign = :left)

    Label(tab2_col2[5, 1], "Spatial Exclusion Controls",
          font = :bold, fontsize = 11, halign = :left)
    mpa_grid = tab2_col2[6, 1] = GridLayout(tellwidth = false)
    mpa_grid.default_rowgap = Fixed(3)
    mpa_grid.default_colgap = Fixed(6)
    btn_draw_mpa = Button(mpa_grid[1, 1],
                          label = @lift($(state.is_drawing_exclusion) ? "Click Map..." : "Draw Area"),
                          fontsize = 10, width = 95, height = 26)
    btn_clear_mpa = Button(mpa_grid[1, 2], label = "Clear",
                           fontsize = 10, width = 75, height = 26)

    menu_mpa_preset = Menu(
        mpa_grid[2, 1:2],
        options = ["Presets...", "Center 25%", "East Shelf", "North Shelf", "West Refuge", "Full Grid"],
        fontsize = 9,
        width = 175
    )

    mpa_coords_sub = mpa_grid[3, 1:2] = GridLayout(tellwidth = false)
    mpa_coords_sub.default_colgap = Fixed(4)
    Label(mpa_coords_sub[1, 1], "X:", fontsize = 9, halign = :right)
    tb_mpa_x1 = Textbox(mpa_coords_sub[1, 2], stored_string = "15", width = 34, fontsize = 9)
    tb_mpa_x2 = Textbox(mpa_coords_sub[1, 3], stored_string = "35", width = 34, fontsize = 9)
    Label(mpa_coords_sub[1, 4], "Y:", fontsize = 9, halign = :right)
    tb_mpa_y1 = Textbox(mpa_coords_sub[1, 5], stored_string = "15", width = 34, fontsize = 9)
    tb_mpa_y2 = Textbox(mpa_coords_sub[1, 6], stored_string = "35", width = 34, fontsize = 9)

    mpa_ctrl_sub = mpa_grid[4, 1:2] = GridLayout(tellwidth = false)
    mpa_ctrl_sub.default_colgap = Fixed(6)
    btn_apply_mpa = Button(mpa_ctrl_sub[1, 1], label = "Apply",
                           fontsize = 10, width = 60, height = 24)
    tog_mpa = Toggle(mpa_ctrl_sub[1, 2], active = true)
    Label(mpa_ctrl_sub[1, 3], "Active", fontsize = 9)

    lbl_mpa_status = Label(mpa_grid[5, 1:2], state.obs_exclusion_status,
                           fontsize = 9, halign = :left, color = :darkgreen)

    # --------------------------------------------------------------------------
    # Parameter Steering and Process Bindings
    # --------------------------------------------------------------------------
    max_duration_days = Observable(2.0 * params.simulation.days_per_year)

    function steering_overrides()
        return Pair{Symbol, Any}[
            :temp_offset => parse(Float64, temp_offset_tb.stored_string[]),
            :temp_multiplier => parse(Float64, temp_mult_tb.stored_string[]),
            :habitat_offset => parse(Float64, hsi_offset_tb.stored_string[]),
            :habitat_multiplier => parse(Float64, hsi_mult_tb.stored_string[]),
            :food_availability => parse(Float64, food_tb.stored_string[]),
            :benthic_carrying_capacity_density => parse(Float64, cap_density_tb.stored_string[]),
            :fishing_season_start => parse(Float64, fstart_tb.stored_string[]),
            :fishing_season_end => parse(Float64, fend_tb.stored_string[]),
            :fishing_probability_daily => parse(Float64, frate_tb.stored_string[]),
            :legal_size_male => parse(Float64, legal_size_tb.stored_string[])
        ]
    end

    legal_size_obs = Observable(params.fishery.legal_size_male)
    on(legal_size_tb.stored_string) do s
        v = tryparse(Float64, s)
        if v !== nothing && v > 0.0
            legal_size_obs[] = v
            p = abmproperties(state.model)[:params]
            abmproperties(state.model)[:params] = reconstruct_params(p; legal_size_male = v)
        end
    end

    on(temp_offset_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; temp_offset = v))
    end
    on(temp_mult_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; temp_multiplier = v))
    end
    on(hsi_offset_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; habitat_offset = v))
    end
    on(hsi_mult_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; habitat_multiplier = v))
    end
    on(food_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; food_availability = v))
    end
    on(cap_density_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; benthic_carrying_capacity_density = v))
    end
    on(fstart_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; fishing_season_start = v))
    end
    on(fend_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; fishing_season_end = v))
    end
    on(frate_tb.stored_string) do s
        v = tryparse(Float64, s)
        v !== nothing && (abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; fishing_probability_daily = v))
    end

    on(unit_tb.stored_string) do s
        u = tryparse(Float64, s)
        if u !== nothing && u > 0.0
            state.obs_unit_meters[] = u
            unit_tb_tab2.stored_string[] = s
        end
    end
    on(unit_tb_tab2.stored_string) do s
        u = tryparse(Float64, s)
        if u !== nothing && u > 0.0
            state.obs_unit_meters[] = u
            unit_tb.stored_string[] = s
        end
    end

    on(layer_menu.selection) do sel
        if sel !== nothing
            state.obs_bg_layer[] = sel == "Habitat Quality (HSI)" ? :habitat_quality :
                (sel == "Seabed Depth (m)" ? :depth : :temperature)
            state.obs_env_field[] = gui_environment_layer_field(
                state.model, state.obs_bg_layer[]
            )
        end
    end

    on(grid_tb.stored_string) do value
        resolution = tryparse(Int, value)
        if resolution !== nothing && 10 <= resolution <= 2000
            if config[:grid] != (resolution, resolution)
                config[:grid] = (resolution, resolution)
                state.is_running[] = false
                reinitialize_gui_model!(abmproperties(state.model)[:params])
            end
        else
            state.status_text[] = "Grid resolution must be 10–2000"
        end
    end

    on(years_tb.stored_string) do s
        years = tryparse(Int, s)
        if years !== nothing && 1 <= years <= 100
            max_duration_days[] = Float64(years) * params.simulation.days_per_year
        else
            state.status_text[] = "Duration must be between 1 and 100 years"
        end
    end

    on(btn_pre_laurentian.clicks) do _
        select_aoi!(-58.50, 45.20; sync_fields = true)
    end
    on(btn_pre_sable.clicks) do _
        select_aoi!(-60.00, 43.80; sync_fields = true)
    end
    on(btn_pre_banquereau.clicks) do _
        select_aoi!(-58.50, 44.50; sync_fields = true)
    end
    on(btn_pre_emerald.clicks) do _
        select_aoi!(-63.60, 43.50; sync_fields = true)
    end

    lift(tog_energetics.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_energetics = a
        )
    end
    lift(tog_thermal.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_thermal_stress = a
        )
    end
    lift(tog_cannibalism.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_cannibalism = a
        )
    end
    lift(tog_senescence.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_senescence = a
        )
    end
    lift(tog_movement.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_movement = a
        )
    end
    lift(tog_reproduction.active) do a
        abmproperties(state.model)[:params] = reconstruct_params(
            abmproperties(state.model)[:params]; use_reproduction = a
        )
    end

    on(btn_quick_run.clicks) do _
        state.is_running[] = !state.is_running[]
    end
    on(btn_quick_step.clicks) do _
        val_str = step_tb.stored_string[]
        remaining = max(0.0, max_duration_days[] - state.current_time[])
        dt = Float64(abmproperties(state.model)[:dt])
        step_days = min(parse(Float64, val_str), floor(remaining / dt) * dt)
        step_gui_simulation!(state, step_days)
    end
    on(btn_quick_reset.clicks) do _
        p = reconstruct_params(
            abmproperties(state.model)[:params]; steering_overrides()...
        )
        state.is_running[] = false
        reinitialize_gui_model!(p)
        state.status_text[] = "Model Reset | Day 0"
    end

    on(btn_rewind.clicks) do _
        target_day = tryparse(Int, rewind_tb.stored_string[])
        if target_day === nothing || target_day < 0
            state.status_text[] = "Rewind day must be a non-negative integer"
        elseif target_day > state.current_time[]
            state.status_text[] = "Rewind day must not exceed the current simulation day"
        else
            state.is_running[] = false
            reinitialize_gui_model!(
                abmproperties(state.model)[:params];
                elapsed_days = Float64(target_day)
            )
        end
    end

    on(btn_apply_mpa.clicks) do _
        x1 = tryparse(Float64, tb_mpa_x1.stored_string[])
        x2 = tryparse(Float64, tb_mpa_x2.stored_string[])
        y1 = tryparse(Float64, tb_mpa_y1.stored_string[])
        y2 = tryparse(Float64, tb_mpa_y2.stored_string[])
        if x1 !== nothing && x2 !== nothing && y1 !== nothing && y2 !== nothing
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            ymin = min(y1, y2)
            ymax = max(y1, y2)
            new_box = ExclusionBox(
                id = 1,
                name = "Area-Custom",
                xmin = xmin, xmax = xmax,
                ymin = ymin, ymax = ymax,
                active = tog_mpa.active[]
            )
            state.exclusion_boxes[] = [new_box]
            update_exclusion_visuals!(state)
        end
    end

    on(menu_mpa_preset.selection) do sel
        sel === nothing && return
        sel == "Presets..." && return
        nx, ny = config[:grid]
        xmin, xmax, ymin, ymax = if sel == "Center 25%"
            round(0.35 * nx; digits = 1), round(0.65 * nx; digits = 1),
            round(0.35 * ny; digits = 1), round(0.65 * ny; digits = 1)
        elseif sel == "East Shelf"
            round(0.65 * nx; digits = 1), round(0.95 * nx; digits = 1),
            round(0.20 * ny; digits = 1), round(0.80 * ny; digits = 1)
        elseif sel == "North Shelf"
            round(0.20 * nx; digits = 1), round(0.80 * nx; digits = 1),
            round(0.65 * ny; digits = 1), round(0.95 * ny; digits = 1)
        elseif sel == "West Refuge"
            round(0.05 * nx; digits = 1), round(0.35 * nx; digits = 1),
            round(0.20 * ny; digits = 1), round(0.80 * ny; digits = 1)
        elseif sel == "Full Grid"
            1.0, Float64(nx), 1.0, Float64(ny)
        else
            0.0, 0.0, 0.0, 0.0
        end
        tb_mpa_x1.stored_string[] = string(xmin)
        tb_mpa_x2.stored_string[] = string(xmax)
        tb_mpa_y1.stored_string[] = string(ymin)
        tb_mpa_y2.stored_string[] = string(ymax)
        new_box = ExclusionBox(
            id = 1,
            name = sel,
            xmin = xmin, xmax = xmax,
            ymin = ymin, ymax = ymax,
            active = tog_mpa.active[]
        )
        state.exclusion_boxes[] = [new_box]
        update_exclusion_visuals!(state)
    end

    on(tog_mpa.active) do is_act
        for b in state.exclusion_boxes[]
            b.active = is_act
        end
        update_exclusion_visuals!(state)
    end

    function update_tab_styling!(tab_idx)
        for (i, b) in enumerate(tab_buttons)
            if i == tab_idx
                b.buttoncolor = RGBf(0.12, 0.42, 0.72)
                b.labelcolor = :white
            else
                b.buttoncolor = RGBf(0.92, 0.93, 0.95)
                b.labelcolor = RGBf(0.15, 0.15, 0.15)
            end
        end
    end

    for (i, b) in enumerate(tab_buttons)
        on(b.clicks) do _
            state.active_tab[] = i
        end
    end

    scene_size = Observable(GLMakie.size(fig.scene))
    on(fig.scene.viewport) do _
        scene_size[] = GLMakie.size(fig.scene)
    end

    function compute_tab_axes_bboxes(tab, current_size)
        fig_w, fig_h = current_size
        viewport_y = 55.0
        viewport_top = Float64(fig_h) - 75.0
        avail_h = max(240.0, viewport_top - viewport_y)

        full_x = 75.0
        full_w = max(300.0, Float64(fig_w) - full_x - 35.0)
        gap_x = 75.0
        gap_y = 65.0

        offscreen = Rect2i(-4000, -4000, 10, 10)

        # Tab 1: Controls & AOI Setup
        # Left: controls ~280px; Right: AOI Map (top) & Climatology (bottom)
        ctrl_w_tab1 = 280.0
        gap_ctrl_map = 24.0
        right_x = full_x + ctrl_w_tab1 + gap_ctrl_map
        right_w = max(380.0, Float64(fig_w) - right_x - 35.0)

        gap_map_cycle = 24.0
        map_h = floor((avail_h - gap_map_cycle) * 0.58)
        cycle_h = avail_h - map_h - gap_map_cycle
        map_y = viewport_top - map_h
        cycle_y = viewport_y

        b_aoi_map    = tab == 1 ? Rect2i(round(Int, right_x), round(Int, map_y), round(Int, right_w), round(Int, map_h)) : offscreen
        b_temp_cycle = tab == 1 ? Rect2i(round(Int, right_x), round(Int, cycle_y), round(Int, right_w), round(Int, cycle_h)) : offscreen

        # Tab 2: Simulation & Habitat
        # Upper row: Map on left (square), Abundance Trajectories on right, aligned in vertical height!
        map_side = floor(min(avail_h * 0.48, (full_w - gap_x) * 0.44))
        upper_y = viewport_top - map_side
        b_map = tab == 2 ? Rect2i(round(Int, full_x), round(Int, upper_y), round(Int, map_side), round(Int, map_side)) : offscreen

        col2_x_tab2 = full_x + map_side + gap_x
        col2_w_tab2 = max(300.0, Float64(fig_w) - col2_x_tab2 - 35.0)
        b_ts_n = tab == 2 ? Rect2i(round(Int, col2_x_tab2), round(Int, upper_y), round(Int, col2_w_tab2), round(Int, map_side)) : offscreen

        # Lower row on right (below ts_n): 3 sub-axes in a row for Depth, Temp, HSI
        avail_below_tab2 = max(60.0, upper_y - viewport_y - gap_y)
        env_sub_w = floor((col2_w_tab2 - 2 * 14.0) / 3.0)
        env_sub_h = avail_below_tab2
        env_y = viewport_y

        b_env_depth = tab == 2 ? Rect2i(round(Int, col2_x_tab2), round(Int, env_y), round(Int, env_sub_w), round(Int, env_sub_h)) : offscreen
        b_env_temp  = tab == 2 ? Rect2i(round(Int, col2_x_tab2 + env_sub_w + 14.0), round(Int, env_y), round(Int, env_sub_w), round(Int, env_sub_h)) : offscreen
        b_env_hsi   = tab == 2 ? Rect2i(round(Int, col2_x_tab2 + 2 * (env_sub_w + 14.0)), round(Int, env_y), round(Int, env_sub_w), round(Int, env_sub_h)) : offscreen

        # Tab 3: Demographics & Growth (2 columns, reduced to ~75% size)
        col_w_half = max(150.0, floor((full_w - gap_x) / 2.0))
        row_h_half = max(120.0, floor((avail_h - gap_y) / 2.0))
        col_w_tab3 = floor(col_w_half * 0.75)
        row_h_tab3 = floor(row_h_half * 0.75)
        col2_x_tab3 = full_x + col_w_tab3 + gap_x
        row1_y_tab3 = viewport_y + row_h_tab3 + gap_y
        row2_y_tab3 = viewport_y

        b_stage = tab == 3 ? Rect2i(round(Int, full_x), round(Int, row1_y_tab3), round(Int, col_w_tab3), round(Int, row_h_tab3)) : offscreen
        b_ogive = tab == 3 ? Rect2i(round(Int, full_x), round(Int, row2_y_tab3), round(Int, col_w_tab3), round(Int, row_h_tab3)) : offscreen
        b_size  = tab == 3 ? Rect2i(round(Int, col2_x_tab3), round(Int, row1_y_tab3), round(Int, col_w_tab3), round(Int, row_h_tab3)) : offscreen
        b_sr    = tab == 3 ? Rect2i(round(Int, col2_x_tab3), round(Int, row2_y_tab3), round(Int, col_w_tab3), round(Int, row_h_tab3)) : offscreen

        # Tab 4: Assessment & Calibration
        top_offset_tab4 = 36.0
        avail_h_tab4 = max(100.0, avail_h - top_offset_tab4)
        macro_h = floor(avail_h_tab4 * 0.52 * 0.75)
        macro_y = viewport_top - top_offset_tab4 - macro_h
        macro_w = floor(full_w * 0.85)

        lower_h_tab4 = floor((avail_h_tab4 - macro_h - gap_y) * 0.75)
        lower_w_tab4 = floor((macro_w - gap_x) / 2.0)
        lower_y_tab4 = viewport_y

        b_macro     = tab == 4 ? Rect2i(round(Int, full_x), round(Int, macro_y), round(Int, macro_w), round(Int, macro_h)) : offscreen
        b_crit      = tab == 4 ? Rect2i(round(Int, full_x), round(Int, lower_y_tab4), round(Int, lower_w_tab4), round(Int, lower_h_tab4)) : offscreen
        b_ann_table = tab == 4 ? Rect2i(round(Int, full_x + lower_w_tab4 + gap_x), round(Int, lower_y_tab4), round(Int, lower_w_tab4), round(Int, lower_h_tab4)) : offscreen

        return (
            map = b_map,
            ts_n = b_ts_n,
            macro_panel = b_macro,
            env_depth = b_env_depth,
            env_temp = b_env_temp,
            env_hsi = b_env_hsi,
            stage = b_stage,
            size = b_size,
            ogive = b_ogive,
            sr = b_sr,
            crit = b_crit,
            ann_table = b_ann_table,
            aoi_map = b_aoi_map,
            temp_cycle = b_temp_cycle
        )
    end

    b_map = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).map, state.active_tab, scene_size)
    ax_map = Axis(
        fig.scene,
        bbox = b_map,
        aspect = AxisAspect(1.0),
        title = "Spatial Habitat & Organism Distribution",
        xlabel = "Grid X (cells)",
        ylabel = "Grid Y (cells)"
    )
    spatial_colormap = lift(state.obs_bg_layer) do layer
        layer === :depth ? :deep :
            (layer === :habitat_quality ? :viridis : :thermal)
    end
    setup_spatial_panel!(
        ax_map, state.obs_env_field, state.obs_agent_x, state.obs_agent_y,
        state.obs_agent_sizes, state.obs_agent_colors; colormap = spatial_colormap,
        exclusion_lines = state.obs_exclusion_lines,
        exclusion_rects = state.obs_exclusion_rects
    )
    state.ax_map = ax_map

    b_ts_n = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).ts_n, state.active_tab, scene_size)
    ax_ts_n = Axis(
        fig.scene,
        bbox = b_ts_n,
        title = "Population Abundance Trajectories",
        xlabel = "Time (days)",
        ylabel = "log10(count)"
    )
    leg_ts_n = setup_abundance_panel!(
        ax_ts_n, state.obs_ts_total, state.obs_ts_females, state.obs_ts_males, state.obs_ts_legal
    )
    state.ax_ts_n = ax_ts_n

    b_macro = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).macro_panel, state.active_tab, scene_size)
    ax_macro = Axis(
        fig.scene,
        bbox = b_macro,
        title = "Biomass, Catch & Stock Assessment Model Fits",
        xlabel = "Simulation Time (days)",
        ylabel = "Biomass (kg)",
        titlesize = 14,
        xlabelsize = 12,
        ylabelsize = 12
    )
    leg_macro = setup_macro_panel!(
        ax_macro, state.obs_ts_biomass, state.obs_ts_catch, state.macro_obs_biomass,
        state.macro_sch_biomass, state.macro_dd_biomass, state.macro_les_biomass;
        label_obs = state.macro_label_observed[],
        label_sch = state.macro_label_sch[],
        label_dd = state.macro_label_dd[],
        label_les = state.macro_label_les[],
        baseline_biomass = state.obs_baseline_biomass
    )
    state.ax_macro = ax_macro

    b_env_depth = lift(
        (t, sz) -> compute_tab_axes_bboxes(t, sz).env_depth, state.active_tab, scene_size
    )
    ax_env_depth = Axis(
        fig.scene,
        bbox = b_env_depth,
        title = "Seabed Depth Distribution (m)",
        xlabel = "Depth (m)",
        ylabel = "Frequency",
        titlesize = 11,
        xlabelsize = 9,
        ylabelsize = 9,
        xticklabelsize = 8,
        yticklabelsize = 8
    )
    hist!(
        ax_env_depth, state.obs_env_depth_vals;
        bins = 20, color = (:royalblue, 0.65), strokewidth = 1.0,
        strokecolor = :royalblue
    )
    state.ax_env_depth = ax_env_depth

    b_env_temp = lift(
        (t, sz) -> compute_tab_axes_bboxes(t, sz).env_temp, state.active_tab, scene_size
    )
    ax_env_temp = Axis(
        fig.scene,
        bbox = b_env_temp,
        title = "Bottom Temperature Distribution (°C)",
        xlabel = "Temperature (°C)",
        ylabel = "Frequency",
        titlesize = 11,
        xlabelsize = 9,
        ylabelsize = 9,
        xticklabelsize = 8,
        yticklabelsize = 8
    )
    hist!(
        ax_env_temp, state.obs_env_temp_vals;
        bins = 20, color = (:firebrick, 0.65), strokewidth = 1.0,
        strokecolor = :firebrick
    )
    state.ax_env_temp = ax_env_temp

    b_env_hsi = lift(
        (t, sz) -> compute_tab_axes_bboxes(t, sz).env_hsi, state.active_tab, scene_size
    )
    ax_env_hsi = Axis(
        fig.scene,
        bbox = b_env_hsi,
        title = "Habitat Suitability Index (HSI)",
        xlabel = "HSI [0, 1]",
        ylabel = "Frequency",
        titlesize = 11,
        xlabelsize = 9,
        ylabelsize = 9,
        xticklabelsize = 8,
        yticklabelsize = 8
    )
    hist!(
        ax_env_hsi, state.obs_env_hsi_vals;
        bins = 20, color = (:seagreen, 0.65), strokewidth = 1.0,
        strokecolor = :seagreen
    )
    state.ax_env_hsi = ax_env_hsi

    # Tab 3: Demographics & Growth
    n_agg = params.organism.juvenile_max_stage - 1
    n_male_st = params.organism.max_stage_male - params.organism.juvenile_max_stage + 1
    n_fem_st = params.organism.max_stage_female - params.organism.juvenile_max_stage + 1
    max_stage = params.organism.max_stage_male

    larval_count = larval_stage_count(SpeciesBehaviorImpl())
    stage_labels = [cohort_stage_label(s, params, larval_count) for s in 0:max_stage]
    first_agent_cat = 2 + n_agg
    n_cats = length(stage_labels)

    b_stage = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).stage, state.active_tab, scene_size)
    ax_stage = Axis(
        fig.scene,
        bbox = b_stage,
        title = "Stage Abundance",
        xlabel = "Stage",
        ylabel = "log10(number)",
        xticks = (collect(1:n_cats), stage_labels),
        xticklabelrotation = pi / 3,
        xticklabelsize = 9
    )

    half_w = 0.21
    obs_male_y = lift(state.obs_stage_male) do v
        Float64[log10_count(x) for x in v]
    end
    obs_female_y = lift(state.obs_stage_female) do v
        Float64[log10_count(x) for x in v]
    end
    obs_agg_y = lift(state.obs_stage_agg) do v
        Float64[log10_count(x / 2) for x in v]
    end
    obs_egg_y = lift(state.obs_current_egg) do n
        Float64[log10_count(n / 2)]
    end

    base_agg = Float64.(2:(1 + n_agg))
    base_agent = Float64.(first_agent_cat:(first_agent_cat + max(n_male_st, n_fem_st) - 1))

    leg_stage = setup_stages_panel!(
        ax_stage, half_w, base_agg, base_agent, n_male_st, n_fem_st,
        obs_egg_y, obs_agg_y, obs_male_y, obs_female_y
    )

    # Click and drag handler for drawing exclusion zones on ax_map (Tab 2)
    map_drag_start = Ref{Union{Nothing, Point2f}}(nothing)

    on(events(ax_map).mousebutton) do event
        state.active_tab[] == 2 || return
        if event.button === Mouse.left
            pos = mouseposition(ax_map)
            if event.action === Mouse.press
                map_drag_start[] = Point2f(pos)
                if state.is_drawing_exclusion[]
                    if state.exclusion_draw_start[] === nothing
                        state.exclusion_draw_start[] = Point2f(pos)
                        state.obs_exclusion_status[] = @sprintf(
                            "Drawing: First corner (%.1f, %.1f) set. Click or drag opposite corner.",
                            pos[1], pos[2]
                        )
                    else
                        p1 = state.exclusion_draw_start[]
                        p2 = Point2f(pos)
                        x1 = min(p1[1], p2[1])
                        x2 = max(p1[1], p2[1])
                        y1 = min(p1[2], p2[2])
                        y2 = max(p1[2], p2[2])
                        new_box = ExclusionBox(
                            id = length(state.exclusion_boxes[]) + 1,
                            name = "Area-$(length(state.exclusion_boxes[]) + 1)",
                            xmin = Float64(x1), xmax = Float64(x2),
                            ymin = Float64(y1), ymax = Float64(y2),
                            active = tog_mpa.active[]
                        )
                        state.exclusion_boxes[] = [new_box]
                        state.exclusion_draw_start[] = nothing
                        state.is_drawing_exclusion[] = false
                        tb_mpa_x1.stored_string[] = string(round(x1; digits = 1))
                        tb_mpa_x2.stored_string[] = string(round(x2; digits = 1))
                        tb_mpa_y1.stored_string[] = string(round(y1; digits = 1))
                        tb_mpa_y2.stored_string[] = string(round(y2; digits = 1))
                        update_exclusion_visuals!(state)
                        state.obs_exclusion_status[] = @sprintf(
                            "Exclusion set: X [%.1f–%.1f], Y [%.1f–%.1f]",
                            x1, x2, y1, y2
                        )
                    end
                end
            elseif event.action === Mouse.release
                if map_drag_start[] !== nothing
                    p1 = map_drag_start[]
                    p2 = Point2f(pos)
                    map_drag_start[] = nothing
                    if abs(p2[1] - p1[1]) >= 2.0 && abs(p2[2] - p1[2]) >= 2.0
                        x1 = min(p1[1], p2[1])
                        x2 = max(p1[1], p2[1])
                        y1 = min(p1[2], p2[2])
                        y2 = max(p1[2], p2[2])
                        new_box = ExclusionBox(
                            id = length(state.exclusion_boxes[]) + 1,
                            name = "Area-$(length(state.exclusion_boxes[]) + 1)",
                            xmin = Float64(x1), xmax = Float64(x2),
                            ymin = Float64(y1), ymax = Float64(y2),
                            active = tog_mpa.active[]
                        )
                        state.exclusion_boxes[] = [new_box]
                        state.exclusion_draw_start[] = nothing
                        state.is_drawing_exclusion[] = false
                        tb_mpa_x1.stored_string[] = string(round(x1; digits = 1))
                        tb_mpa_x2.stored_string[] = string(round(x2; digits = 1))
                        tb_mpa_y1.stored_string[] = string(round(y1; digits = 1))
                        tb_mpa_y2.stored_string[] = string(round(y2; digits = 1))
                        update_exclusion_visuals!(state)
                        state.obs_exclusion_status[] = @sprintf(
                            "Exclusion set: X [%.1f–%.1f], Y [%.1f–%.1f]",
                            x1, x2, y1, y2
                        )
                    end
                end
            end
        end
    end

    b_size = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).size, state.active_tab, scene_size)
    ax_size = Axis(
        fig.scene,
        bbox = b_size,
        title = "Size Frequency by Sex and Maturity",
        xlabel = "Size (mm)",
        ylabel = "Density"
    )
    xlims!(ax_size, 0, 165)
    size_grid = collect(0.0:0.5:165.0)
    kde(obs) = lift(v -> [Point2f(Float32(size_grid[i]),
            Float32(kde_density(v, size_grid; scale = :probability)[i]))
            for i in eachindex(size_grid)], obs)

    obs_immature_m = kde(state.obs_cw_immature_male)
    obs_mature_m = kde(state.obs_cw_mature_male)
    obs_immature_f = kde(state.obs_cw_immature_female)
    obs_mature_f = kde(state.obs_cw_mature_female)

    peak = lift((a, b, c, d) -> maximum(p[2] for p in vcat(a, b, c, d)),
        obs_immature_m, obs_mature_m, obs_immature_f, obs_mature_f)

    leg_size = setup_size_density_panel!(
        ax_size, obs_immature_m, obs_mature_m, obs_immature_f, obs_mature_f,
        legal_size_obs; peak = peak
    )

    b_ogive = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).ogive, state.active_tab, scene_size)
    ax_ogive = Axis(
        fig.scene,
        bbox = b_ogive,
        title = "Maturity Ogive & Fishery Selectivity",
        xlabel = "Size (mm)",
        ylabel = "P(Mature) / Selectivity"
    )
    xlims!(ax_ogive, 0, 165)
    leg_ogive = setup_ogive_panel!(
        ax_ogive, legal_size_obs,
        state.obs_ogive_male_pts, state.obs_ogive_female_pts
    )

    b_sr = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).sr, state.active_tab, scene_size)
    ax_sr = Axis(
        fig.scene,
        bbox = b_sr,
        title = "Stock-Recruitment (Male Component)",
        xlabel = "log10(Annual Eggs / 2)",
        ylabel = "log10(Males at Recruitment Stage)",
        titlesize = 12,
        xlabelsize = 10,
        ylabelsize = 10,
        xticklabelsize = 9,
        yticklabelsize = 9
    )
    leg_sr = setup_stock_recruitment_panel!(
        ax_sr, state.obs_sr_pts, state.obs_sr_fit_curve, state.obs_sr_fit_curve_bh
    )

    # Tab 4: Assessment & Calibration
    b_crit = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).crit, state.active_tab, scene_size)
    ax_crit = Axis(
        fig.scene,
        bbox = b_crit,
        title = "Multi-Attribute Calibration & Stopping Criteria",
        titlesize = 13
    )
    setup_criteria_panel!(ax_crit, state.calib_criteria_text)

    b_ann_table = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).ann_table, state.active_tab, scene_size)
    ax_ann_table = Axis(
        fig.scene,
        bbox = b_ann_table,
        title = "Annual Production & Assessment Summary Table",
        titlesize = 13
    )
    setup_annual_table_panel!(ax_ann_table, state.annual_table_text)

    # Tab 1 Right Half: Scotian Shelf regional bathymetry map for interactive AOI selection
    b_aoi_map = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).aoi_map, state.active_tab, scene_size)
    ax_aoi_map = Axis(
        fig.scene,
        bbox = b_aoi_map,
        title = "Scotian Shelf Regional Bathymetry (Click Map to Select AOI)",
        titlesize = 13,
        xlabel = "Longitude (°W)",
        ylabel = "Latitude (°N)",
        xlabelsize = 11,
        ylabelsize = 11,
        xticklabelsize = 10,
        yticklabelsize = 10,
        xreversed = false
    )
    xlims!(ax_aoi_map, GUI_AOI_LON_BOUNDS...)
    ax_aoi_map.xreversed = false
    ylims!(ax_aoi_map, GUI_AOI_LAT_BOUNDS...)
    if aoi_bathymetry !== nothing
        heatmap!(
            ax_aoi_map, aoi_bathymetry.lons, aoi_bathymetry.lats,
            aoi_bathymetry.depths; colormap = :deep,
            colorrange = (0.0, 450.0)
        )
        contour!(
            ax_aoi_map, aoi_bathymetry.lons, aoi_bathymetry.lats,
            aoi_bathymetry.depths; levels = [1.0, 50.0, 100.0, 200.0, 300.0],
            color = (:gray25, 0.75), linewidth = 1.0
        )
    else
        text!(
            ax_aoi_map, mean(GUI_AOI_LON_BOUNDS), mean(GUI_AOI_LAT_BOUNDS);
            text = "Local bathymetry data unavailable",
            align = (:center, :center), color = :black
        )
    end
    aoi_marker = lift(state.aoi_lon, state.aoi_lat) do lon, lat
        Point2f[Point2f(lon, lat)]
    end
    scatter!(ax_aoi_map, aoi_marker; color = :red, markersize = 14,
        strokecolor = :white, strokewidth = 1.5)

    state.ax_aoi_map = ax_aoi_map

    # Tab 1 Lower Right: Seasonal Bottom Temperature Climatology for selected AOI
    b_temp_cycle = lift((t, sz) -> compute_tab_axes_bboxes(t, sz).temp_cycle, state.active_tab, scene_size)
    ax_temp_cycle = Axis(
        fig.scene,
        bbox = b_temp_cycle,
        title = state.obs_temp_cycle_title,
        xlabel = "Day of Year",
        ylabel = "Bottom Temp (°C)",
        titlesize = 12,
        xlabelsize = 10,
        ylabelsize = 10,
        xticklabelsize = 9,
        yticklabelsize = 9,
        xticks = (
            [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335],
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        )
    )
    hspan!(ax_temp_cycle, 0.0, 4.0; color = (:teal, 0.12), label = "Preferred Envelope (0–4°C)")
    hlines!(ax_temp_cycle, state.obs_temp_cycle_mean; color = :gray50, linestyle = :dash, label = "Annual Mean")
    lines!(ax_temp_cycle, state.obs_temp_cycle_pts; color = :crimson, linewidth = 2.0, label = "Seasonal Cycle")
    axislegend(ax_temp_cycle, position = :rt, labelsize = 9, framevisible = true,
               backgroundcolor = (:white, 0.85), padding = (4, 4, 2, 2))
    state.ax_temp_cycle = ax_temp_cycle

    # List of all axes and legends for visibility management across tabs
    all_tab_axes = [
        (ax_aoi_map, 1),
        (ax_temp_cycle, 1),
        (ax_map, 2),
        (ax_ts_n, 2),
        (ax_env_depth, 2),
        (ax_env_temp, 2),
        (ax_env_hsi, 2),
        (ax_stage, 3),
        (ax_size, 3),
        (ax_ogive, 3),
        (ax_sr, 3),
        (ax_macro, 4),
        (ax_crit, 4),
        (ax_ann_table, 4)
    ]

    all_tab_legends = [
        (leg_ts_n, 2),
        (leg_stage, 3),
        (leg_size, 3),
        (leg_ogive, 3),
        (leg_sr, 3),
        (leg_macro, 4)
    ]

    function update_tab_visibility!(t)
        for (ax, tab_id) in all_tab_axes
            vis = (t == tab_id)
            if ax.blockscene.visible[] != vis
                ax.blockscene.visible[] = vis
                ax.scene.visible[] = vis
            end
        end
        for (leg, tab_id) in all_tab_legends
            leg === nothing && continue
            vis = (t == tab_id)
            if leg.blockscene.visible[] != vis
                leg.blockscene.visible[] = vis
            end
        end
        set_grid_visibility!(controls_tab_grid, t == 1)
        set_grid_visibility!(tab2_controls_grid, t == 2)
        set_grid_visibility!(tab4_toolbar, t == 4)
        return nothing
    end

    on(state.active_tab) do t
        update_tab_styling!(t)
        update_tab_visibility!(t)
    end

    # Initial visibility and styling setup
    update_tab_styling!(state.active_tab[])
    update_tab_visibility!(state.active_tab[])

    on(state.current_time) do _
        try
            autolimits!(ax_ts_n)
            autolimits!(ax_stage)
            autolimits!(ax_macro)
            autolimits!(ax_size)
            xlims!(ax_size, 0, 165)
            autolimits!(ax_ogive)
            xlims!(ax_ogive, 0, 165)
            autolimits!(ax_sr)
        catch e
            @warn "axis autolimits failed" exception = e
        end
    end

    # --------------------------------------------------------------------------
    # Transport Button Callbacks
    # --------------------------------------------------------------------------
    function reinitialize_gui_model!(p::IBMParameters; elapsed_days::Float64 = 0.0)
        lon = state.aoi_lon[]
        lat = state.aoi_lat[]
        domain_lon, domain_lat = gui_aoi_domain(lon, lat)
        p = reconstruct_params(p; domain_lon = domain_lon, domain_lat = domain_lat)
        grid = config[:grid]
        state.model = initialize_model(
            grid = grid,
            dt = config[:dt],
            params = p,
            seed = config[:seed],
            depth_map = gui_aoi_depth_map(aoi_bathymetry, lon, lat, grid),
            n_agents = n_agents
        )
        empty!(state.t_history)
        empty!(state.n_total_history)
        empty!(state.n_females_history)
        empty!(state.n_males_history)
        empty!(state.n_legal_history)
        empty!(state.biomass_legal_history)
        empty!(state.catch_history)
        empty!(state.mean_weight_legal_history)
        empty!(state.mean_weight_male_history)
        empty!(state.mean_weight_female_history)
        empty!(state.mean_weight_total_history)
        empty!(state.n_egg_history)
        foreach(empty!, state.n_early_stage_history)
        empty!(state.n_early_agents_history)
        empty!(state.sr_raw)
        state.catch_at_year_start = 0.0
        state.current_year_idx = 1
        state.annual_data = create_empty_annual_data()
        empty!(state.snapshots)
        state.snapshot_labels[] = String["Year 0"]
        state.macro_obs_biomass[] = Point2f[]
        state.macro_sch_biomass[] = Point2f[]
        state.macro_dd_biomass[] = Point2f[]
        state.macro_les_biomass[] = Point2f[]
        state.macro_metrics_text[] = "Macro models awaiting annual data (>= 4 years)"
        state.macro_label_observed[] = "Annual Surveyed Biomass"
        state.macro_label_sch[] = "Schaefer"
        state.macro_label_dd[] = "Delay-Difference"
        state.macro_label_les[] = "Leslie Matrix"
        state.macro_conversion_text[] = ""
        state.obs_sr_fit_curve[] = Point2f[]
        state.obs_sr_fit_curve_bh[] = Point2f[]
        state.obs_sr_fit_text[] = ""
        state.obs_baseline_biomass[] = Point2f[]
        state.obs_baseline_catch[] = Point2f[]
        state.has_baseline[] = false
        state.current_time[] = 0.0
        step_gui_simulation!(state, 0.0)
        state.snapshots[0] = (
            model = deepcopy(state.model),
            t_history = copy(state.t_history),
            n_total_history = copy(state.n_total_history),
            n_females_history = copy(state.n_females_history),
            n_males_history = copy(state.n_males_history),
            n_legal_history = copy(state.n_legal_history),
            biomass_legal_history = copy(state.biomass_legal_history),
            catch_history = copy(state.catch_history),
            annual_data = copy(state.annual_data)
        )
        if elapsed_days > 0.0
            step_gui_simulation!(state, elapsed_days)
        end
        state.status_text[] = elapsed_days > 0.0 ?
            "Replayed from day 0 to day $(round(Int, state.current_time[]))" :
            "Model reinitialized | Day 0"
        return nothing
    end

    on(btn_rewind.clicks) do _
        target_day = tryparse(Int, rewind_tb.stored_string[])
        if target_day === nothing || target_day < 0
            state.status_text[] = "Rewind day must be a non-negative integer"
        elseif target_day > state.current_time[]
            state.status_text[] = "Rewind day must not exceed the current simulation day"
        else
            state.is_running[] = false
            reinitialize_gui_model!(
                abmproperties(state.model)[:params];
                elapsed_days = Float64(target_day)
            )
        end
    end

    on(grid_tb.stored_string) do value
        resolution = tryparse(Int, value)
        if resolution !== nothing && 30 <= resolution <= 150 && resolution % 10 == 0
            if config[:grid] != (resolution, resolution)
                config[:grid] = (resolution, resolution)
                state.is_running[] = false
                reinitialize_gui_model!(abmproperties(state.model)[:params])
            end
        else
            state.status_text[] = "Grid resolution must be 30–150 in steps of 10"
        end
    end

    on(btn_refit.clicks) do _
        if nrow(state.annual_data) >= 4
            recalibrate_macro_models!(state)
        else
            state.status_text[] = "Refit failed: Need >= 4 annual periods (currently $(nrow(state.annual_data)))"
        end
    end

    on(btn_draw_mpa.clicks) do _
        state.is_drawing_exclusion[] = !state.is_drawing_exclusion[]
        state.exclusion_draw_start[] = nothing
        if state.is_drawing_exclusion[]
            state.obs_exclusion_status[] = "Click map on Tab 2: 1st corner..."
            state.active_tab[] = 2
        else
            state.obs_exclusion_status[] = "Drawing cancelled"
        end
    end

    on(btn_clear_mpa.clicks) do _
        empty!(state.exclusion_boxes[])
        empty!(state.obs_exclusion_lines[])
        if haskey(abmproperties(state.model), :exclusion_boxes)
            empty!(abmproperties(state.model)[:exclusion_boxes])
        end
        state.obs_exclusion_status[] = "No active exclusion zone"
    end

    on(btn_export_toml.clicks) do _
        try
            ts_str = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
            scen_dir = normpath(joinpath(_PROJECT_ROOT, "configs", "scenarios"))
            mkpath(scen_dir)
            scen_file = joinpath(scen_dir, "scenario_$(ts_str).toml")
            cfg_out = Dict{Symbol, Any}(
                :environment => copy(params.environment),
                :simulation => copy(params.simulation),
                :fishery => copy(params.fishery)
            )
            export_scenario_toml(
                scen_file, cfg_out;
                aoi_coords = (lon = state.aoi_lon[], lat = state.aoi_lat[]),
                exclusion_boxes = state.exclusion_boxes[],
                stage_distribution = state.obs_stage_dist_init[]
            )
            state.status_text[] = "Exported: $(basename(scen_file))"
        catch e
            state.status_text[] = "Export failed: $e"
        end
    end

    on(btn_save_parquet.clicks) do _
        if !isempty(state.annual_data)
            try
                results_path = resolve_simulation_results_path("simulation_results")
                _write_annual_production_collection(
                    results_path, state.annual_data;
                    dt_year = 365.0,
                    frequency_days = 365.0,
                    snapshot_day = 274.0,
                    meta = Dict(:days => state.current_time[])
                )
                state.status_text[] = "Saved: annual_production.parquet"
            catch e
                state.status_text[] = "Local results save failed: $e"
            end
        else
            state.status_text[] = "Local results save requires annual data"
        end
    end

    function update_aoi_calculations!()
        lon = state.aoi_lon[]
        lat = state.aoi_lat[]
        mprops = abmproperties(state.model)
        grid = config[:grid]
        span_m = (Float64(grid[1]), Float64(grid[2]))

        mean_depth, mean_temp, mean_hsi = if haskey(mprops, :depth_field) &&
                                             haskey(mprops, :temperature_field) &&
                                             haskey(mprops, :habitat_quality_field)
            d_vals = filter(isfinite, Float64.(vec(mprops[:depth_field])))
            t_vals = filter(isfinite, Float64.(vec(mprops[:temperature_field])))
            h_vals = filter(isfinite, Float64.(vec(mprops[:habitat_quality_field])))
            (
                isempty(d_vals) ? 150.0 : mean(d_vals),
                isempty(t_vals) ? 3.5 : mean(t_vals),
                isempty(h_vals) ? 0.85 : mean(h_vals)
            )
        else
            bathy = estimate_aoi_bathymetry(lon, lat; span_meters = span_m, n_points = grid)
            t_bot = aoi_bottom_temperature(bathy.mean_depth, lat)
            hsi_val = aoi_habitat_suitability(t_bot, bathy.mean_depth)
            (bathy.mean_depth, t_bot, hsi_val)
        end
        state.aoi_depth_str[] = @sprintf("Depth: %.1f m", mean_depth)
        state.aoi_temp_str[] = @sprintf("Temp: %.2f °C", mean_temp)
        state.aoi_hsi_str[] = @sprintf("HSI: %.4f", mean_hsi)
        update_environmental_distributions!(state)
    end

    aoi_update_in_progress = Ref(false)
    function select_aoi!(lon::Float64, lat::Float64; sync_fields::Bool = false)
        if !isfinite(lon) || !isfinite(lat) ||
           !(GUI_AOI_LON_BOUNDS[1] <= lon <= GUI_AOI_LON_BOUNDS[2]) ||
           !(GUI_AOI_LAT_BOUNDS[1] <= lat <= GUI_AOI_LAT_BOUNDS[2])
            state.status_text[] = "AOI must be within the mapped Scotian Shelf bounds"
            return nothing
        end
        if aoi_bathymetry !== nothing
            elevation = aoi_bathymetry.interpolator(lon, lat)
            if !isfinite(elevation) || elevation >= 0.0
                state.status_text[] = "Select a valid water location on the bathymetry map"
                return nothing
            end
        end
        state.is_running[] = false
        aoi_update_in_progress[] = true
        state.aoi_lon[] = lon
        state.aoi_lat[] = lat
        if sync_fields
            lon_tb.stored_string[] = string(round(lon; digits = 4))
            lat_tb.stored_string[] = string(round(lat; digits = 4))
        end
        aoi_update_in_progress[] = false
        update_aoi_calculations!()
        reinitialize_gui_model!(abmproperties(state.model)[:params])
        state.status_text[] = @sprintf("AOI set to %.4f°W, %.4f°N | Model reset", -lon, lat)
        return nothing
    end

    on(lon_tb.stored_string) do s
        aoi_update_in_progress[] && return
        lon = tryparse(Float64, s)
        if lon === nothing
            state.status_text[] = "Longitude must be a number"
        else
            select_aoi!(lon, state.aoi_lat[])
        end
    end

    on(lat_tb.stored_string) do s
        aoi_update_in_progress[] && return
        lat = tryparse(Float64, s)
        if lat === nothing
            state.status_text[] = "Latitude must be a number"
        else
            select_aoi!(state.aoi_lon[], lat)
        end
    end

    on(events(ax_aoi_map).mousebutton) do event
        if state.active_tab[] == 1 && event.button === Mouse.left && event.action === Mouse.press
            pos = mouseposition(ax_aoi_map)
            select_aoi!(Float64(pos[1]), Float64(pos[2]); sync_fields = true)
        end
    end

    # --------------------------------------------------------------------------
    # Asynchronous Stepping Loop
    # --------------------------------------------------------------------------
    if display_window
        state.sim_task = @async while state.is_active[]
            if state.is_running[]
                remaining = max_duration_days[] - state.current_time[]
                if remaining <= 0.0
                    state.is_running[] = false
                    state.status_text[] = "Configured run duration reached"
                else
                    dt = Float64(abmproperties(state.model)[:dt])
                    available = floor(remaining / dt) * dt
                    if available <= 0.0
                        state.is_running[] = false
                        state.status_text[] = "Configured run duration reached"
                    else
                        step_gui_simulation!(state, min(state.sim_speed_days[], available))
                        if state.current_time[] >= max_duration_days[]
                            state.is_running[] = false
                            state.status_text[] = "Configured run duration reached"
                        end
                    end
                end
                sleep(0.02) # Yield to OpenGL UI event loop
            else
                sleep(0.08)
            end
        end

        on(events(fig.scene).window_open) do is_open
            if !is_open
                close_gui!(state)
            end
        end
    end

    # Initial paint
    update_aoi_calculations!()
    step_gui_simulation!(state, 0.0)

    if display_window
        display(fig)
    end

    return (fig, state)
end
