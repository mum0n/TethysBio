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
    defaults = flattened_dashboard_config(config)
    lon = haskey(defaults, :domain_lon) ? mean(defaults[:domain_lon]) : -63.5
    lat = haskey(defaults, :domain_lat) ? mean(defaults[:domain_lat]) : 44.5
    years = max(1, ceil(Int, get(defaults, :days, 730) / 365))
    map_widget = TETHYSBIO.aoi_coords_widget(
        default_lon = Float64(lon), default_lat = Float64(lat), standalone = true
    )
    transport_widget = TETHYSBIO.simulation_transport_widget(
        default_interval = 10, default_years = years, standalone = true
    )
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
    body { margin: 0; padding: 8px; background: #f1f5f9; }
    * { box-sizing: border-box; }
  </style>
</head>
<body>
  $(map_widget)
  $(transport_widget)
  $(steering_widget)
</body>
</html>
"""
end

function initial_dashboard_state(config::AbstractDict)
    cfg = Dict{Symbol, Any}(Symbol(key) => value for (key, value) in config)
    model = TETHYSBIO.initialize_model(cfg)
    agent_data, model_data = TETHYSBIO.run_simulation(model, 0)
    annual_data = TETHYSBIO.extract_annual_production(agent_data, model_data)
    return DashboardState(
        cfg, model, agent_data, model_data, annual_data, 0,
        max(1, Int(get(cfg, :days, 730))), 10, :temperature, false, ReentrantLock()
    )
end

function append_observations!(current::DataFrame, next::DataFrame)
    nrow(next) <= 1 && return current
    append!(current, next[2:end, :]; cols = :union)
    return current
end

function rebuild_model!(state::DashboardState)
    state.model = TETHYSBIO.initialize_model(state.config)
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
    old_grid = get(state.config, :grid, (50, 50))

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
                (key === :fishing_rate ? :fishing_probability_daily :
                (key === :legal_size ? :legal_size_male : key)))
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
    bounds = TETHYSBIO.aoi_domain_bounds(Float64(value["lon"]), Float64(value["lat"]))
    state.config[:domain_lon] = bounds.lon
    state.config[:domain_lat] = bounds.lat
    state.needs_reset = true
    return
end

function advance_to_day!(state::DashboardState, requested_day::Int)
    target_day = clamp(requested_day, 0, state.horizon_days)
    if state.needs_reset || target_day < state.current_day || target_day == 0
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
    params = Agents.abmproperties(state.model)[:params]
    figures = (
        :spatial => TETHYSBIO.plot_dashboard_spatial(
            state.model, params; sim_days = state.current_day,
            horizon_days = state.horizon_days, background_layer = state.background_layer
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
    (:recruitment, "Stock-Recruitment")
)

function dashboard_panel(name::Symbol, title::String, image)
    return Bonito.DOM.section(
        Bonito.DOM.h2(title, style = "font-size: 15px; margin: 0 0 8px; color: #0f172a;"),
        Bonito.DOM.img(
            src = image,
            alt = title,
            style = "display: block; width: 100%; height: auto; background: white;"
        ),
        style = "min-width: 0; overflow: hidden; border: 1px solid #cbd5e1; " *
                "border-radius: 8px; padding: 10px; background: white;"
    )
end

function apply_browser_message!(state::DashboardState, message, status, images)
    message isa AbstractDict || return
    kind = get(message, "type", "")
    value = get(message, "value", Dict{String, Any}())
    lock(state.lock) do
        if kind == "aoi"
            update_area_of_interest!(state, value)
            status[] = "AOI changed; model will reset on the next transport update."
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
            update_control_values!(state, value)
            status_text = string(get(value, "sim_status", "stopped"))
            target_day = round(Int, Float64(get(value, "current_day", state.current_day)))
            state.horizon_days = max(1, round(Int, Float64(get(value, "sim_years", 2)) * 365))
            state.update_interval = max(1, round(Int, Float64(get(value, "update_interval", 10))))
            day = advance_to_day!(state, target_day)
            updated = render_dashboard_images(state)
            for (name, _) in DASHBOARD_PANELS
                images[name][] = updated[name]
            end
            status[] = "Simulation $(status_text) — day $(day) of $(state.horizon_days)"
        end
    end
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
            title = "Dashboard controls",
            style = "width: 100%; height: 1480px; border: 0; background: #f1f5f9;"
        )
        Bonito.onload(
            session,
            controls,
            Bonito.@js_str"""function(frame) {
                window.addEventListener("message", function(event) {
                    if (event.source === frame.contentWindow && event.data) {
                        $(messages).notify(event.data);
                    }
                });
            }"""
        )

        panels = [
            dashboard_panel(name, title, images[name])
            for (name, title) in DASHBOARD_PANELS
        ]
        return Bonito.DOM.div(
            Bonito.DOM.style("""
                body { margin: 0; background: #f1f5f9; color: #0f172a;
                       font-family: system-ui, sans-serif; }
                .tethys-layout { display: grid; grid-template-columns:
                    minmax(340px, 0.75fr) minmax(0, 1.25fr); gap: 16px;
                    align-items: start; padding: 16px; }
                .tethys-panels { display: grid; grid-template-columns:
                    repeat(auto-fit, minmax(340px, 1fr)); gap: 12px; }
                @media (max-width: 900px) {
                    .tethys-layout { grid-template-columns: minmax(0, 1fr); }
                }
            """),
            Bonito.DOM.header(
                Bonito.DOM.h1("TethysBio Interactive Dashboard",
                    style = "font-size: 22px; margin: 0;"),
                Bonito.DOM.div(status,
                    style = "color: #475569; font-size: 13px; margin-top: 4px;"),
                style = "padding: 16px 20px; background: white; border-bottom: 1px solid #cbd5e1;"
            ),
            Bonito.DOM.main(
                Bonito.DOM.div(controls, class = "tethys-controls"),
                Bonito.DOM.div(panels, class = "tethys-panels"),
                class = "tethys-layout"
            )
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
