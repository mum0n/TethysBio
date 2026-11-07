using Test
using TOML
using TethysBio

@testset "HTML Dashboard Configuration and Controls" begin
    config = default_ibm_configuration()
    config[:grid] = (70, 80)
    config[:temp_offset] = 1.5
    config[:temp_noise_sd] = 0.75
    config[:food_replenishment_rate] = 0.37
    config[:use_movement] = false
    config[:n_agents] = 240
    config[:days] = 1095

    encoded = TethysBio.WebDashboard.web_dashboard_config_toml(config)
    decoded = TOML.parse(encoded)
    @test decoded["simulation"]["grid_nx"] == 70
    @test decoded["simulation"]["grid_ny"] == 80
    @test decoded["environment"]["temp_offset"] == 1.5
    @test decoded["environment"]["temp_noise_sd"] == 0.75
    @test decoded["environment"]["food_replenishment_rate"] == 0.37
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
    @test occursin("id=\"ctrl_food_rate\"", controls)
    @test occursin("food_replenishment_rate: parseFloat(c_food_rate.value)", controls)
    @test occursin("id=\"ctrl_grid\" min=\"30\" max=\"150\" step=\"10\" value=\"70\"", controls)
    @test occursin("id=\"tog_energetics\" checked", controls)
    @test !occursin("id=\"tog_movement\" checked", controls)
    @test occursin("window.parent.postMessage({ type: 'parameters'", controls)

    dashboard_controls = TethysBio.WebDashboard.dashboard_controls_document(config)
    @test occursin("GEBCO Bathymetry", dashboard_controls)
    @test occursin("window.parent.postMessage({ type: 'aoi'", dashboard_controls)
    @test occursin("window.parent.postMessage({ type: 'transport'", dashboard_controls)
    @test !occursin("Pluto.run", dashboard_controls)
end
