"""
    src/web_dashboard.jl

Standalone browser dashboard for the organism Individual-Based Model (IBM),
powered by Bonito and CairoMakie. It does not start Pluto or load a notebook.
"""

module WebDashboard

using Agents
using Base64
using Bonito
using CairoMakie
using DataFrames
using Statistics
using TOML

export launch_web_dashboard

const TETHYSBIO = parentmodule(@__MODULE__)

mutable struct DashboardState
    config::Dict{Symbol, Any}
    model::Any
    agent_data::DataFrame
    model_data::DataFrame
    annual_data::DataFrame
    current_day::Int
    horizon_days::Int
    update_interval::Int
    background_layer::Symbol
    needs_reset::Bool
    exclusion_boxes::Vector{Any}
    lock::ReentrantLock
end

function web_dashboard_config_toml(config::AbstractDict)
    sections = Dict{String, Any}()
    for (_, struct_type, section) in TETHYSBIO.parameter_structs()
        values = Dict{String, Any}()
        for field in fieldnames(struct_type)
            haskey(config, field) && (values[string(field)] = config[field])
        end
        sections[section] = values
    end
    if haskey(config, :grid)
        grid = config[:grid]
        sections["simulation"]["grid_nx"] = Int(grid[1])
        sections["simulation"]["grid_ny"] = Int(grid[2])
    end

    runtime = Dict{String, Any}()
    for key in (:n_agents, :days)
        haskey(config, key) && (runtime[string(key)] = config[key])
    end
    isempty(runtime) || (sections["runtime"] = runtime)

    io = IOBuffer()
    TOML.print(io, sections)
    return String(take!(io))
end

function flattened_dashboard_config(config::AbstractDict)
    parsed = TOML.parse(web_dashboard_config_toml(config))
    return Dict{Symbol, Any}(
        Symbol(key) => value
        for section in values(parsed)
        for (key, value) in section
    )
end

function dashboard_controls_document(config::AbstractDict)
    lon = get(config, :aoi_lon, -59.0)
    lat = get(config, :aoi_lat, 45.0)
    span_km = get(config, :aoi_span_km, 50.0)
    map_widget = TETHYSBIO.aoi_coords_widget(
        default_lon = Float64(lon), default_lat = Float64(lat),
        default_span_km = Float64(span_km), standalone = true
    )
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; padding: 0; background: transparent; font-family: system-ui, -apple-system, sans-serif; color: #0f172a; }
    * { box-sizing: border-box; }
    .controls-top-grid { width: 100%; margin: 0; }
  </style>
  <script>
    function reportHeight() {
      if (window.parent && window.parent !== window) {
        var h = document.body.scrollHeight || document.documentElement.scrollHeight;
        window.parent.postMessage({ type: 'iframe_height', id: 'aoi_controls_frame', height: h }, '*');
      }
    }
    window.addEventListener('load', reportHeight);
    window.addEventListener('resize', reportHeight);
    window.addEventListener('message', function(ev) {
      if (!ev.data) return;
      if (ev.data.type === 'switch_tab') {
        if (window.aoi_map) {
          setTimeout(function() { window.aoi_map.invalidateSize(); reportHeight(); }, 60);
        }
      }
    });
  </script>
</head>
<body>
  <div class="controls-top-grid">
    $(map_widget)
  </div>
</body>
</html>
"""
end

function dashboard_steering_document(config::AbstractDict)
    defaults = flattened_dashboard_config(config)
    years = max(1, ceil(Int, get(defaults, :days, 730) / 365))
    steering_widget = TETHYSBIO.parameter_steering_sliders_widget(
        default_interval = 10, default_years = years,
        defaults = defaults, standalone = true
    )
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { margin: 0; padding: 2px; background: transparent; font-family: system-ui, -apple-system, sans-serif; color: #0f172a; }
    * { box-sizing: border-box; }
  </style>
</head>
<body>
  $(steering_widget)
</body>
</html>
"""
end

function initial_dashboard_state(config::AbstractDict)
    cfg = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in config)
    lon = Float64(get(cfg, :aoi_lon, -59.0))
    lat = Float64(get(cfg, :aoi_lat, 45.0))
    span_km = Float64(get(cfg, :aoi_span_km, 50.0))
    bounds = TETHYSBIO.aoi_domain_bounds(lon, lat; span_meters = span_km * 1000.0)
    cfg[:domain_lon] = bounds.lon
    cfg[:domain_lat] = bounds.lat
    cfg[:aoi_lon] = lon
    cfg[:aoi_lat] = lat
    cfg[:aoi_span_km] = span_km
    cfg[:use_geodata] = get(cfg, :use_geodata, true)
    cfg[:grid] = get(cfg, :grid, (1000, 1000))
    cfg[:grid_nx] = cfg[:grid][1]
    cfg[:grid_ny] = cfg[:grid][2]
    n_ag = get(cfg, :n_agents, 120)
    if n_ag == 0
        n_ag = 120
    end
    cfg[:n_agents] = n_ag
    model = TETHYSBIO.initialize_model(cfg; n_agents = n_ag)
    agent_data, model_data = TETHYSBIO.run_simulation(model, 0)
    annual_data = TETHYSBIO.extract_annual_production(agent_data, model_data)
    return DashboardState(
        cfg, model, agent_data, model_data, annual_data, 0,
        max(1, Int(get(cfg, :days, 730))), 10, :depth, false,
        Any[], ReentrantLock()
    )
end

function append_observations!(current::DataFrame, next::DataFrame)
    nrow(next) <= 1 && return current
    append!(current, next[2:end, :]; cols = :union)
    return current
end

function rebuild_model!(state::DashboardState)
    n_ag = get(state.config, :n_agents, 120)
    if n_ag == 0
        n_ag = 120
    end
    state.config[:n_agents] = n_ag
    state.config[:use_geodata] = get(state.config, :use_geodata, true)
    state.model = TETHYSBIO.initialize_model(state.config; n_agents = n_ag)
    state.agent_data, state.model_data = TETHYSBIO.run_simulation(state.model, 0)
    state.annual_data = TETHYSBIO.extract_annual_production(state.agent_data, state.model_data)
    state.current_day = 0
    state.needs_reset = false
    return state
end

function update_live_parameters!(state::DashboardState, overrides::Dict{Symbol, Any})
    isempty(overrides) && return
    params = Agents.abmproperties(state.model)[:params]
    Agents.abmproperties(state.model)[:params] =
        TETHYSBIO.reconstruct_params(params; overrides...)
end

function update_control_values!(state::DashboardState, values)
    values isa AbstractDict || return
    known_keys = TETHYSBIO.known_config_keys()
    overrides = Dict{Symbol, Any}()
    old_grid = get(state.config, :grid, (1000, 1000))

    for (raw_key, raw_value) in values
        key = Symbol(raw_key)
        if key === :background_layer
            layer = lowercase(string(raw_value))
            state.background_layer = occursin("habitat", layer) ? :habitat_quality :
                (occursin("depth", layer) ? :depth : :temperature)
        elseif key === :grid_res
            resolution = round(Int, Float64(raw_value))
            state.config[:grid] = (resolution, resolution)
            state.config[:grid_nx] = resolution
            state.config[:grid_ny] = resolution
        elseif key in (:sim_status, :current_day, :tick, :update_interval, :sim_years)
            continue
        else
            canonical_key = key === :temp_base ? :temp_offset :
                (key === :temp_amplitude ? :temp_multiplier :
                (key === :food || key === :food_multiplier ? :food_availability :
                (key === :carrying_capacity || key === :k_density || key === :carrying_density ? :benthic_carrying_capacity_density :
                (key === :fishing_rate ? :fishing_probability_daily :
                (key === :legal_size ? :legal_size_male : key)))))
            canonical_key in known_keys || continue
            declared_type = TETHYSBIO.config_field_type(canonical_key)
            declared_type === nothing && continue
            value = TETHYSBIO.coerce_to_declared(declared_type, raw_value, canonical_key)
            state.config[canonical_key] = value
            overrides[canonical_key] = value
        end
    end

    TETHYSBIO.sync_grid_dimensions!(state.config; explicit = Set([:grid]))
    if state.config[:grid] != old_grid
        state.needs_reset = true
    elseif !state.needs_reset
        update_live_parameters!(state, overrides)
    end
    return
end

function update_area_of_interest!(state::DashboardState, value)
    value isa AbstractDict || return
    haskey(value, "lon") && haskey(value, "lat") || return
    lon = Float64(value["lon"])
    lat = Float64(value["lat"])
    span_m = if haskey(value, "span_km")
        Float64(value["span_km"]) * 1000.0
    elseif haskey(value, "span_meters")
        Float64(value["span_meters"])
    else
        50000.0
    end
    bounds = if haskey(value, "bounds") && value["bounds"] isa AbstractDict
        b = value["bounds"]
        (lon = [Float64(b["min_lon"]), Float64(b["max_lon"])],
         lat = [Float64(b["min_lat"]), Float64(b["max_lat"])])
    else
        TETHYSBIO.aoi_domain_bounds(lon, lat; span_meters = span_m)
    end
    state.config[:domain_lon] = bounds.lon
    state.config[:domain_lat] = bounds.lat
    state.config[:aoi_lon] = lon
    state.config[:aoi_lat] = lat
    state.config[:aoi_span_km] = span_m / 1000.0
    state.needs_reset = true
    return
end

function advance_to_day!(state::DashboardState, requested_day::Int)
    target_day = clamp(requested_day, 0, state.horizon_days)
    if state.needs_reset || (target_day == 0 && state.current_day != 0) || target_day < state.current_day
        rebuild_model!(state)
    end

    if target_day > state.current_day
        dt = Float64(get(state.config, :dt, 1.0))
        steps = round(Int, (target_day - state.current_day) / dt)
        if steps > 0
            next_agents, next_model = TETHYSBIO.run_simulation(state.model, steps)
            append_observations!(state.agent_data, next_agents)
            append_observations!(state.model_data, next_model)
            state.current_day = round(Int, maximum(state.agent_data.time))
            state.annual_data = TETHYSBIO.extract_annual_production(
                state.agent_data, state.model_data
            )
        end
    end
    return target_day
end

function svg_data_uri(figure)
    io = IOBuffer()
    show(io, MIME("image/svg+xml"), figure)
    return "data:image/svg+xml;base64," * base64encode(take!(io))
end

function render_dashboard_images(state::DashboardState)
    CairoMakie.activate!()
    params = Agents.abmproperties(state.model)[:params]
    figures = (
        :spatial => TETHYSBIO.plot_dashboard_spatial(
            state.model, params; sim_days = state.current_day,
            horizon_days = state.horizon_days, background_layer = state.background_layer,
            exclusion_boxes = state.exclusion_boxes
        ),
        :environment => TETHYSBIO.plot_dashboard_environmental_distributions(
            state.model; sim_days = state.current_day
        ),
        :abundance => TETHYSBIO.plot_dashboard_abundance(
            state.agent_data; sim_days = state.current_day
        ),
        :stages => TETHYSBIO.plot_dashboard_stages(
            state.model, params; sim_days = state.current_day
        ),
        :size => TETHYSBIO.plot_dashboard_size_density(
            state.model, params; sim_days = state.current_day
        ),
        :ogive => TETHYSBIO.plot_dashboard_ogive(
            state.model, params; sim_days = state.current_day
        ),
        :macro => TETHYSBIO.plot_dashboard_macro(
            state.agent_data, state.model_data, state.annual_data;
            sim_days = state.current_day
        ),
        :recruitment => TETHYSBIO.plot_dashboard_stock_recruitment(
            state.annual_data; sim_days = state.current_day
        ),
        :temp_cycle => TETHYSBIO.plot_bottom_temperature_cycle(
            Float64(get(state.config, :aoi_lon, -59.0)),
            Float64(get(state.config, :aoi_lat, 45.0));
            temp_offset = Float64(get(state.config, :temp_offset, 0.0)),
            temp_multiplier = Float64(get(state.config, :temp_multiplier, 1.0))
        )
    )
    return Dict(name => svg_data_uri(figure) for (name, figure) in figures)
end

const DASHBOARD_PANELS = (
    (:spatial, "Spatial Habitat & Organisms"),
    (:environment, "Environmental Distributions"),
    (:abundance, "Population Abundance"),
    (:stages, "Stage Structure"),
    (:size, "Size Frequency by Sex & Maturity"),
    (:ogive, "Maturity Ogives & Selectivity"),
    (:macro, "Biomass, Catch & Macro-Models"),
    (:recruitment, "Stock-Recruitment"),
    (:temp_cycle, "Seasonal Bottom Temperature Climatology")
)

function dashboard_panel(name::Symbol, title::String, image)
    img_style = if name === :temp_cycle
        "display: block; width: 100%; height: auto; aspect-ratio: 5 / 4; object-fit: contain; background: white;"
    else
        "display: block; width: 100%; height: auto; background: white;"
    end
    return Bonito.DOM.section(
        Bonito.DOM.h2(title, style = "font-size: 15px; margin: 0 0 8px; color: #0f172a;"),
        Bonito.DOM.img(
            src = image,
            alt = title,
            style = img_style
        ),
        style = "min-width: 0; overflow: hidden; border: 1px solid #cbd5e1; " *
                "border-radius: 8px; padding: 10px; background: white;"
    )
end

function apply_browser_message!(state::DashboardState, message, status, images)
    message isa AbstractDict || return
    kind = get(message, "type", "")
    kind == "dashboard_tab" && return
    value = get(message, "value", Dict{String, Any}())
    lock(state.lock) do
        if kind == "aoi"
            update_area_of_interest!(state, value)
            lon_f = round(Float64(get(value, "lon", 0.0)), digits = 2)
            lat_f = round(Float64(get(value, "lat", 0.0)), digits = 2)
            status[] = "AOI updated: $(lat_f)°N, $(lon_f)°W; model will reset on next transport update."
            updated = render_dashboard_images(state)
            for (name, _) in DASHBOARD_PANELS
                images[name][] = updated[name]
            end
        elseif kind == "exclusion_zone"
            xmin = Float64(get(value, "xmin", 0.0))
            xmax = Float64(get(value, "xmax", 0.0))
            ymin = Float64(get(value, "ymin", 0.0))
            ymax = Float64(get(value, "ymax", 0.0))
            active = Bool(get(value, "active", true))
            clear_all = Bool(get(value, "clear", false)) || !active || (xmin == 0.0 && xmax == 0.0)
            if clear_all
                empty!(state.exclusion_boxes)
                status[] = "Exclusion area cleared."
            else
                box = TETHYSBIO.ExclusionBox(
                    id = 1,
                    name = "MPA-Custom",
                    xmin = min(xmin, xmax),
                    xmax = max(xmin, xmax),
                    ymin = min(ymin, ymax),
                    ymax = max(ymin, ymax),
                    active = true
                )
                state.exclusion_boxes = [box]
                status[] = "Exclusion area set: X [$(round(Int, box.xmin))–$(round(Int, box.xmax))], Y [$(round(Int, box.ymin))–$(round(Int, box.ymax))]"
            end
            updated = render_dashboard_images(state)
            for (name, _) in DASHBOARD_PANELS
                images[name][] = updated[name]
            end
        elseif kind == "parameters"
            old_layer = state.background_layer
            update_control_values!(state, value)
            if old_layer != state.background_layer
                updated = render_dashboard_images(state)
                for (name, _) in DASHBOARD_PANELS
                    images[name][] = updated[name]
                end
            end
            status[] = state.needs_reset ?
                "Grid or AOI changed; model will reset on the next transport update." :
                "Parameters updated for the next simulation step."
        elseif kind == "transport"
            payload = isempty(value) ? message : value
            update_control_values!(state, payload)
            status_text = string(get(payload, "sim_status", "stopped"))
            target_day = round(Int, Float64(get(payload, "current_day", state.current_day)))
            state.horizon_days = max(1, round(Int, Float64(get(payload, "sim_years", 2)) * 365))
            state.update_interval = max(1, round(Int, Float64(get(payload, "update_interval", 10))))
            day = advance_to_day!(state, target_day)
            updated = render_dashboard_images(state)
            for (name, _) in DASHBOARD_PANELS
                images[name][] = updated[name]
            end
            status[] = "Simulation $(status_text) — day $(day) of $(state.horizon_days)"
        end
    end
end

function dashboard_spatial_interactive_panel(image_obs)
    return Bonito.DOM.div(
        Bonito.DOM.div(
            Bonito.DOM.div(
                Bonito.DOM.span("Spatial Habitat & Organism Distribution",
                    style = "font-weight: 700; font-size: 13px; color: #1e293b;"),
                Bonito.DOM.span("Drag mouse to set Exclusion Area",
                    id = "spatial-drag-hint",
                    style = "font-size: 11px; color: #64748b; font-style: italic;"),
                style = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;"
            ),
            Bonito.DOM.div(
                Bonito.DOM.img(
                    src = image_obs,
                    id = "spatial-plot-img",
                    style = "width: 100%; height: auto; display: block; user-select: none; pointer-events: none;"
                ),
                Bonito.DOM.div(
                    id = "spatial-drag-box",
                    style = "position: absolute; border: 2px dashed #f59e0b; background: rgba(245, 158, 11, 0.28); display: none; pointer-events: none; border-radius: 2px;"
                ),
                id = "spatial-interactive-container",
                style = "position: relative; cursor: crosshair; overflow: hidden; border-radius: 6px; border: 1px solid #cbd5e1; background: white; width: 100%;"
            ),
            Bonito.DOM.div(
                Bonito.DOM.div(
                    Bonito.DOM.span("Exclusion: ", style = "font-weight: 600; color: #334155;"),
                    Bonito.DOM.span("None", id = "exclusion-summary-text", style = "color: #64748b; font-family: monospace;"),
                    style = "font-size: 11px;"
                ),
                Bonito.DOM.div(
                    Bonito.DOM.button("Clear MPA", type = "button", id = "btn-clear-mpa",
                        onclick = "clearExclusionArea()",
                        style = "background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: 600; cursor: pointer; color: #475569;"),
                    style = "display: flex; gap: 6px;"
                ),
                style = "display: flex; justify-content: space-between; align-items: center; margin-top: 6px; padding: 4px 8px; background: #f8fafc; border-radius: 4px; border: 1px solid #e2e8f0;"
            ),
            style = "background: white; border-radius: 8px; padding: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: 1px solid #e2e8f0;"
        ),
        style = "min-width: 0; width: 100%;"
    )
end

function dashboard_app(config::AbstractDict)
    CairoMakie.activate!()
    state = initial_dashboard_state(config)
    initial_images = render_dashboard_images(state)

    return Bonito.App(title = "TethysBio Interactive Dashboard") do session
        status = Bonito.Observable("Ready — day 0 of $(state.horizon_days)")
        images = Dict(
            name => Bonito.Observable(initial_images[name])
            for (name, _) in DASHBOARD_PANELS
        )
        messages = Bonito.Observable(Dict{String, Any}())
        Bonito.on(messages) do message
            try
                apply_browser_message!(state, message, status, images)
            catch err
                status[] = "Dashboard update failed: $(sprint(showerror, err))"
                @error "Standalone dashboard update failed" exception = (err, catch_backtrace())
            end
        end

        controls = Bonito.DOM.iframe(
            srcdoc = dashboard_controls_document(config),
            id = "aoi-controls-iframe",
            title = "Dashboard controls",
            style = "width: 100%; min-height: 540px; height: 580px; border: 0; background: transparent;"
        )
        controls_tab2 = Bonito.DOM.iframe(
            srcdoc = dashboard_steering_document(config),
            title = "Simulation and Environmental Controls",
            style = "width: 100%; height: 620px; border: 0; background: transparent;"
        )

        Bonito.onload(
            session,
            controls,
            Bonito.@js_str"""function(frame) {
                window.addEventListener("message", function(event) {
                    if (event.data && (event.data.type || event.data.kind)) {
                        $(messages).notify(event.data);
                    }
                });
            }"""
        )

        tab1_panels = [
            Bonito.DOM.div(
                controls,
                style = "min-width: 0; width: 100%;"
            ),
            Bonito.DOM.div(
                dashboard_panel(
                    :temp_cycle,
                    "Seasonal Bottom Temperature Climatology",
                    images[:temp_cycle]
                ),
                style = "min-width: 0; width: 100%;"
            )
        ]

        tab2_panels = [
            Bonito.DOM.div(
                dashboard_spatial_interactive_panel(images[:spatial]),
                controls_tab2,
                style = "display: flex; flex-direction: column; gap: 14px; min-width: 0;"
            ),
            Bonito.DOM.div(
                dashboard_panel(:abundance, "Population Abundance Trajectories", images[:abundance]),
                dashboard_panel(:environment, "Environmental Distributions", images[:environment]),
                style = "display: flex; flex-direction: column; gap: 16px; min-width: 0;"
            )
        ]

        tab3_panels = [
            dashboard_panel(:stages, "Stage Structure & Cohort Progression", images[:stages]),
            dashboard_panel(:size, "Size Frequency by Sex & Maturity", images[:size]),
            dashboard_panel(:ogive, "Maturity Ogives & Fishery Selectivity", images[:ogive])
        ]

        tab4_panels = [
            dashboard_panel(:recruitment, "Stock-Recruitment Relationship (Male Component)", images[:recruitment]),
            dashboard_panel(:macro, "Biomass, Catch & Stock Assessment Model Fits", images[:macro])
        ]

        transport_controls = Bonito.DOM.div(
            Bonito.DOM.div(
                Bonito.DOM.button("▶ Start", type = "button", onclick = "sendTransportCmd('start')",
                    style = "background: #16a34a; color: white; border: none; border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 12px; cursor: pointer;"),
                Bonito.DOM.button("▶❚ Step", type = "button", onclick = "sendTransportCmd('step')",
                    style = "background: #0284c7; color: white; border: none; border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 12px; cursor: pointer;"),
                Bonito.DOM.button("⏸ Pause", type = "button", onclick = "sendTransportCmd('pause')",
                    style = "background: #d97706; color: white; border: none; border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 12px; cursor: pointer;"),
                Bonito.DOM.button("⏹ Stop / Reset", type = "button", onclick = "sendTransportCmd('stop')",
                    style = "background: #dc2626; color: white; border: none; border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 12px; cursor: pointer;"),
                style = "display: flex; gap: 6px; align-items: center;"
            ),
            Bonito.DOM.div(
                Bonito.DOM.span("|", style = "color: #cbd5e1; margin: 0 4px;"),
                Bonito.DOM.label(
                    Bonito.DOM.span("Step (d): ", style = "font-size: 12px; font-weight: 600; color: #475569;"),
                    Bonito.DOM.input(type = "number", id = "transport-step", value = "$(state.update_interval)", min = "1", max = "365",
                        style = "width: 55px; padding: 4px 6px; font-size: 12px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace; text-align: right;"),
                    style = "display: flex; align-items: center; gap: 4px;"
                ),
                Bonito.DOM.label(
                    Bonito.DOM.span("Duration (y): ", style = "font-size: 12px; font-weight: 600; color: #475569;"),
                    Bonito.DOM.input(type = "number", id = "transport-years", value = "$(round(Int, state.horizon_days / 365))", min = "1", max = "100",
                        style = "width: 48px; padding: 4px 6px; font-size: 12px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace; text-align: right;"),
                    style = "display: flex; align-items: center; gap: 4px;"
                ),
                Bonito.DOM.label(
                    Bonito.DOM.span("Rewind (d): ", style = "font-size: 12px; font-weight: 600; color: #475569;"),
                    Bonito.DOM.input(type = "number", id = "transport-rewind", value = "10", min = "1", max = "3650",
                        style = "width: 55px; padding: 4px 6px; font-size: 12px; border: 1px solid #cbd5e1; border-radius: 4px; font-family: monospace; text-align: right;"),
                    style = "display: flex; align-items: center; gap: 4px;"
                ),
                Bonito.DOM.button("⏪ Rewind", type = "button", onclick = "sendTransportRewind()",
                    style = "background: #64748b; color: white; border: none; border-radius: 4px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer;"),
                style = "display: flex; gap: 8px; align-items: center;"
            ),
            style = "display: flex; gap: 10px; align-items: center; flex-wrap: wrap;"
        )

        tab_nav = Bonito.DOM.div(
            Bonito.DOM.nav(
                Bonito.DOM.button("1. Controls & AOI Setup", type = "button", class = "tethysbio-tab-btn active", data_tab = "1", onclick = "switchDashboardTab(1)"),
                Bonito.DOM.button("2. Simulation & Habitat", type = "button", class = "tethysbio-tab-btn", data_tab = "2", onclick = "switchDashboardTab(2)"),
                Bonito.DOM.button("3. Demographics & Growth", type = "button", class = "tethysbio-tab-btn", data_tab = "3", onclick = "switchDashboardTab(3)"),
                Bonito.DOM.button("4. Assessment & Calibration", type = "button", class = "tethysbio-tab-btn", data_tab = "4", onclick = "switchDashboardTab(4)"),
                class = "tethysbio-tab-bar"
            ),
            transport_controls,
            style = "display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 12px;"
        )

        tab_content = Bonito.DOM.div(
            Bonito.DOM.div(tab1_panels..., id = "tab-page-1", class = "tethysbio-tab-page active"),
            Bonito.DOM.div(tab2_panels..., id = "tab-page-2", class = "tethysbio-tab-page"),
            Bonito.DOM.div(tab3_panels..., id = "tab-page-3", class = "tethysbio-tab-page"),
            Bonito.DOM.div(tab4_panels..., id = "tab-page-4", class = "tethysbio-tab-page"),
            class = "tethysbio-tab-container"
        )

        return Bonito.DOM.div(
            Bonito.DOM.style("""
                body { margin: 0; background: #f1f5f9; color: #0f172a;
                       font-family: system-ui, -apple-system, sans-serif; }
                .tethysbio-header {
                    padding: 14px 24px; background: white; border-bottom: 1px solid #cbd5e1;
                }
                .tethysbio-tab-bar {
                    display: flex; gap: 8px;
                    background: #e2e8f0; padding: 4px; border-radius: 8px; width: fit-content;
                }
                .tethysbio-tab-btn {
                    padding: 8px 18px; border: none; border-radius: 6px;
                    background: transparent; color: #475569; font-size: 13px;
                    font-weight: 600; cursor: pointer; transition: all 0.15s ease;
                }
                .tethysbio-tab-btn:hover { background: #cbd5e1; color: #0f172a; }
                .tethysbio-tab-btn.active {
                    background: #1e3a8a; color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.15);
                }
                .tethysbio-layout {
                    padding: 18px 24px; max-width: 1750px; margin: 0 auto;
                }
                .tethysbio-tab-page {
                    display: none;
                }
                .tethysbio-tab-page.active {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
                    gap: 18px; align-items: start;
                }
                #tab-page-1.active {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 18px;
                    align-items: start;
                }
                #tab-page-2.active {
                    display: grid;
                    grid-template-columns: minmax(320px, 0.35fr) minmax(540px, 0.65fr);
                    gap: 18px;
                    align-items: start;
                }
                #tab-page-3.active {
                    display: grid;
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                    gap: 18px;
                    align-items: start;
                }
                @media (max-width: 1100px) {
                    .tethysbio-tab-page.active,
                    #tab-page-1.active,
                    #tab-page-2.active,
                    #tab-page-3.active {
                        grid-template-columns: minmax(0, 1fr);
                    }
                }
            """),
            Bonito.DOM.header(
                Bonito.DOM.h1("TethysBio Interactive Dashboard",
                    style = "font-size: 22px; margin: 0; color: #0f172a;"),
                Bonito.DOM.div(status,
                    style = "color: #475569; font-size: 13px; margin-top: 4px;"),
                tab_nav,
                class = "tethysbio-header"
            ),
            Bonito.DOM.main(
                tab_content,
                class = "tethysbio-layout"
            ),
            Bonito.DOM.script("""
                var simTimer = null;
                var simCurrentDay = 0;
                var currentGridDim = { nx: $(state.config[:grid][1]), ny: $(state.config[:grid][2]) };

                window.addEventListener('message', function(ev) {
                    if (ev.data && ev.data.type === 'iframe_height' && ev.data.id === 'aoi_controls_frame') {
                        var fr = document.getElementById('aoi-controls-iframe');
                        if (fr && ev.data.height) {
                            fr.style.height = (ev.data.height + 6) + 'px';
                        }
                    }
                });

                function switchDashboardTab(tabIdx) {
                    document.querySelectorAll('.tethysbio-tab-btn').forEach(function(b) {
                        var active = parseInt(b.getAttribute('data_tab') || b.getAttribute('data-tab')) === tabIdx;
                        b.classList.toggle('active', active);
                    });
                    for (var i = 1; i <= 4; i++) {
                        var page = document.getElementById('tab-page-' + i);
                        if (page) page.classList.toggle('active', i === tabIdx);
                    }
                    var frames = document.querySelectorAll('iframe');
                    frames.forEach(function(frame) {
                        if (frame && frame.contentWindow) {
                            frame.contentWindow.postMessage({ type: 'switch_tab', tab: tabIdx }, '*');
                        }
                    });
                }

                function getTransportParams() {
                    var stepEl = document.getElementById('transport-step');
                    var yearsEl = document.getElementById('transport-years');
                    var rewindEl = document.getElementById('transport-rewind');
                    var step = stepEl ? (parseInt(stepEl.value) || 10) : 10;
                    var years = yearsEl ? (parseFloat(yearsEl.value) || 2.0) : 2.0;
                    var rewind = rewindEl ? (parseInt(rewindEl.value) || 10) : 10;
                    return { step: step, years: years, rewind: rewind };
                }

                function notifyTransport(statusText, targetDay) {
                    var p = getTransportParams();
                    var payload = {
                        sim_status: statusText,
                        current_day: targetDay,
                        update_interval: p.step,
                        sim_years: p.years
                    };
                    window.postMessage({
                        type: 'transport',
                        value: payload,
                        sim_status: statusText,
                        current_day: targetDay,
                        update_interval: p.step,
                        sim_years: p.years
                    }, '*');
                }

                var simRunning = false;
                function advanceSimStep() {
                    if (simRunning) return;
                    simRunning = true;
                    var p = getTransportParams();
                    var maxDays = Math.round(p.years * 365);
                    if (simCurrentDay >= maxDays) {
                        if (simTimer) { clearInterval(simTimer); simTimer = null; }
                        notifyTransport('completed', simCurrentDay);
                        simRunning = false;
                        return;
                    }
                    simCurrentDay = Math.min(maxDays, simCurrentDay + p.step);
                    notifyTransport('running', simCurrentDay);
                    setTimeout(function() { simRunning = false; }, 300);
                }

                function sendTransportCmd(cmd) {
                    var p = getTransportParams();
                    var maxDays = Math.round(p.years * 365);
                    if (cmd === 'start') {
                        if (simCurrentDay >= maxDays) simCurrentDay = 0;
                        if (simTimer) clearInterval(simTimer);
                        advanceSimStep();
                        simTimer = setInterval(advanceSimStep, 1000);
                    } else if (cmd === 'step') {
                        if (simTimer) { clearInterval(simTimer); simTimer = null; }
                        advanceSimStep();
                    } else if (cmd === 'pause') {
                        if (simTimer) { clearInterval(simTimer); simTimer = null; }
                        notifyTransport('paused', simCurrentDay);
                    } else if (cmd === 'stop') {
                        if (simTimer) { clearInterval(simTimer); simTimer = null; }
                        simCurrentDay = 0;
                        notifyTransport('stopped', 0);
                    }
                }

                function sendTransportRewind() {
                    if (simTimer) { clearInterval(simTimer); simTimer = null; }
                    var p = getTransportParams();
                    simCurrentDay = Math.max(0, simCurrentDay - p.rewind);
                    notifyTransport('rewound', simCurrentDay);
                }

                function initSpatialMouseSelection() {
                    var container = document.getElementById('spatial-interactive-container');
                    var dragBox = document.getElementById('spatial-drag-box');
                    if (!container) return;
                    var isDragging = false;
                    var startX = 0, startY = 0;

                    container.addEventListener('mousedown', function(e) {
                        if (e.button !== 0) return;
                        var rect = container.getBoundingClientRect();
                        startX = e.clientX - rect.left;
                        startY = e.clientY - rect.top;
                        isDragging = true;
                        if (dragBox) {
                            dragBox.style.left = startX + 'px';
                            dragBox.style.top = startY + 'px';
                            dragBox.style.width = '0px';
                            dragBox.style.height = '0px';
                            dragBox.style.display = 'block';
                        }
                        e.preventDefault();
                    });

                    window.addEventListener('mousemove', function(e) {
                        if (!isDragging || !dragBox) return;
                        var rect = container.getBoundingClientRect();
                        var curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                        var curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
                        var x = Math.min(startX, curX);
                        var y = Math.min(startY, curY);
                        var w = Math.abs(curX - startX);
                        var h = Math.abs(curY - startY);
                        dragBox.style.left = x + 'px';
                        dragBox.style.top = y + 'px';
                        dragBox.style.width = w + 'px';
                        dragBox.style.height = h + 'px';
                    });

                    window.addEventListener('mouseup', function(e) {
                        if (!isDragging) return;
                        isDragging = false;
                        if (dragBox) dragBox.style.display = 'none';
                        var rect = container.getBoundingClientRect();
                        var curX = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
                        var curY = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
                        var w = Math.abs(curX - startX);
                        var h = Math.abs(curY - startY);
                        if (w < 8 || h < 8) return;

                        var x1_frac = Math.min(startX, curX) / rect.width;
                        var x2_frac = Math.max(startX, curX) / rect.width;
                        var y1_frac = Math.min(startY, curY) / rect.height;
                        var y2_frac = Math.max(startY, curY) / rect.height;

                        // Transform from figure coords to grid cells [1..nx, 1..ny]
                        var xmin_rel = Math.max(0, Math.min(1, (x1_frac - 0.11) / (0.82 - 0.11)));
                        var xmax_rel = Math.max(0, Math.min(1, (x2_frac - 0.11) / (0.82 - 0.11)));
                        var ymin_rel = Math.max(0, Math.min(1, (0.88 - y2_frac) / (0.88 - 0.12)));
                        var ymax_rel = Math.max(0, Math.min(1, (0.88 - y1_frac) / (0.88 - 0.12)));

                        var nx = currentGridDim.nx || 50;
                        var ny = currentGridDim.ny || 50;

                        var gx_min = Math.max(1, Math.round(1 + xmin_rel * (nx - 1)));
                        var gx_max = Math.max(gx_min, Math.round(1 + xmax_rel * (nx - 1)));
                        var gy_min = Math.max(1, Math.round(1 + ymin_rel * (ny - 1)));
                        var gy_max = Math.max(gy_min, Math.round(1 + ymax_rel * (ny - 1)));

                        var label = document.getElementById('exclusion-summary-text');
                        if (label) label.textContent = 'X [' + gx_min + '–' + gx_max + '], Y [' + gy_min + '–' + gy_max + ']';

                        window.postMessage({
                            type: 'exclusion_zone',
                            xmin: gx_min,
                            xmax: gx_max,
                            ymin: gy_min,
                            ymax: gy_max,
                            active: true
                        }, '*');
                    });
                }

                function clearExclusionArea() {
                    var label = document.getElementById('exclusion-summary-text');
                    if (label) label.textContent = 'None';
                    window.postMessage({
                        type: 'exclusion_zone',
                        clear: true,
                        active: false
                    }, '*');
                }

                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', initSpatialMouseSelection);
                } else {
                    initSpatialMouseSelection();
                }

                window.addEventListener('message', function(ev) {
                    if (!ev.data) return;
                    if (ev.data.type === 'dashboard_tab') {
                        var t = parseInt(ev.data.tab);
                        if (t >= 1 && t <= 4) switchDashboardTab(t);
                    } else if (ev.data.type === 'parameters' && (ev.data.values || ev.data.value)) {
                        var v = ev.data.value || ev.data.values;
                        if (v.grid_res) {
                            var res = parseInt(v.grid_res) || 1000;
                            currentGridDim.nx = res;
                            currentGridDim.ny = res;
                        }
                    }
                });
            """)
        )
    end
end

"""
    launch_web_dashboard(config; port = 9384, launch_browser = true)

Start and serve the standalone Bonito HTML dashboard. The dashboard renders
interactive browser controls and CairoMakie charts without launching Pluto.
"""
function launch_web_dashboard(
    config::AbstractDict;
    port::Integer = 9384,
    launch_browser::Bool = true
)
    1 <= port <= 65535 || throw(ArgumentError("port must be between 1 and 65535"))

    app = dashboard_app(config)
    server = Bonito.Server(app, "127.0.0.1", Int(port))
    url = "http://127.0.0.1:$(port)"
    println("="^80)
    println(" TethysBio standalone HTML dashboard is running at $(url)")
    println(" Press Ctrl+C in this terminal to stop the server.")
    println("="^80)
    if launch_browser && !Bonito.HTTPServer.openurl(url)
        @warn "Could not open a browser automatically; open the dashboard URL manually" url
    end
    try
        wait(server)
    finally
        close(server)
    end
    return 0
end

end # module WebDashboard
