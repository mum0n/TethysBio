using Test
using TOML
using TethysBio

@testset "HTML Dashboard Configuration and Controls" begin
    config = default_ibm_configuration()
    config[:grid] = (70, 80)
    config[:temp_offset] = 1.5
    config[:food_availability] = 1.1
    config[:benthic_carrying_capacity_density] = 7500.0
    config[:use_movement] = false
    config[:n_agents] = 240
    config[:days] = 1095

    encoded = TethysBio.WebDashboard.web_dashboard_config_toml(config)
    decoded = TOML.parse(encoded)
    @test decoded["simulation"]["grid_nx"] == 70
    @test decoded["simulation"]["grid_ny"] == 80
    @test decoded["environment"]["temp_offset"] == 1.5
    @test decoded["environment"]["food_availability"] == 1.1
    @test decoded["environment"]["benthic_carrying_capacity_density"] == 7500.0
    @test decoded["toggles"]["use_movement"] == false
    @test decoded["runtime"]["n_agents"] == 240
    @test decoded["runtime"]["days"] == 1095

    slider_defaults = Dict{Symbol, Any}(
        Symbol(key) => value
        for section in values(decoded)
        for (key, value) in section
    )
    controls = parameter_steering_sliders_widget(
        defaults = slider_defaults, standalone = true
    )
    @test occursin("id=\"ctrl_food_avail\"", controls)
    @test occursin("food_availability: favail", controls)
    @test occursin("id=\"ctrl_k_density\"", controls)
    @test occursin("id=\"ctrl_grid\"", controls) && occursin("value=\"70\"", controls)
    @test occursin("id=\"tog_energetics\" checked", controls)
    @test !occursin("id=\"tog_movement\" checked", controls)
    @test occursin("window.parent.postMessage({ type: 'parameters'", controls)

    dashboard_controls = TethysBio.WebDashboard.dashboard_controls_document(config)
    @test occursin("GEBCO Bathymetry", dashboard_controls)
    @test occursin("45.00°N, -59.00°W", dashboard_controls)
    @test occursin("esriOcean", dashboard_controls)
    @test !occursin("slider_span", dashboard_controls)
    @test !occursin("aoi_box", dashboard_controls)
    @test occursin("aoi-preset-btn", dashboard_controls)
    @test occursin("controls-top-grid", dashboard_controls)
    @test !occursin("controls-tab-bar", dashboard_controls)
    @test !occursin("Simulation Execution & Sequence Controls", dashboard_controls)
    @test occursin("window.parent.postMessage({ type: 'aoi'", dashboard_controls)
    @test !occursin("Pluto.run", dashboard_controls)
    @test occursin("aspect-ratio: 5 / 4", dashboard_controls)

    # Verification: map appears first, then coordinates and presets placed below it
    idx_map = findfirst("aoi_map", dashboard_controls)
    idx_lon = findfirst("slider_lon", dashboard_controls)
    idx_presets = findfirst("aoi-preset-btn", dashboard_controls)
    @test idx_map !== nothing && idx_lon !== nothing && idx_presets !== nothing
    @test idx_map.start < idx_lon.start < idx_presets.start

    steering_doc = TethysBio.WebDashboard.dashboard_steering_document(config)
    @test occursin("Simulation & Environmental Controls", steering_doc)
    @test occursin("Commercial Fishing Activity", steering_doc)
    @test occursin("Biological Processes", steering_doc)
    @test !occursin("Biological Process Toggles", steering_doc)
    @test occursin("ctrl_food_avail", steering_doc)
    @test occursin("window.parent.postMessage({ type: 'parameters'", steering_doc)

    state = TethysBio.WebDashboard.initial_dashboard_state(config)
    @test state.background_layer === :depth
    imgs = TethysBio.WebDashboard.render_dashboard_images(state)
    @test haskey(imgs, :spatial) && startswith(imgs[:spatial], "data:image/svg+xml;base64,")
    @test haskey(imgs, :recruitment) && startswith(imgs[:recruitment], "data:image/svg+xml;base64,")
    @test haskey(imgs, :macro) && startswith(imgs[:macro], "data:image/svg+xml;base64,")
    @test haskey(imgs, :temp_cycle) && startswith(imgs[:temp_cycle], "data:image/svg+xml;base64,")

    day = TethysBio.WebDashboard.advance_to_day!(state, 10)
    @test day == 10
    @test state.current_day == 10
end
