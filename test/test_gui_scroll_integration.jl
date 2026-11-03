using Test
using TethysBio
using GLMakie
using Dates
using TOML

@testset "Scrollable GUI Integration & Shared Interactivity" begin
    # Launch GUI in headless mode (no display window)
    fig, state = launch_organism_gui(
        n_agents = 30,
        grid = (15, 15),
        dt = 1.0,
        seed = 42,
        display_window = false
    )

    @test isa(fig, Figure)
    @test isa(state, IBMGuiState)
    @test length(state.obs_stage_dist_init[]) == 20
    @test state.aoi_lon[] == -63.5
    @test state.aoi_lat[] == 44.5
    @test isempty(state.exclusion_boxes[])

    # Advance model by 15 days
    step_gui_simulation!(state, 15.0)
    @test state.current_time[] >= 15.0
    @test !isempty(state.calib_criteria_text[])

    # Test adding an exclusion box interactively
    box = ExclusionBox(id = 1, name = "TestMPA", xmin = 2.0, xmax = 8.0, ymin = 2.0, ymax = 8.0, active = true)
    push!(state.exclusion_boxes[], box)
    @test is_in_exclusion_zones(5.0, 5.0, state.exclusion_boxes[]) == true
    @test is_in_exclusion_zones(12.0, 12.0, state.exclusion_boxes[]) == false

    # Test exporting scenario TOML
    mktempdir() do tmpdir
        toml_path = joinpath(tmpdir, "test_scen.toml")
        cfg_test = default_ibm_configuration()
        export_scenario_toml(
            toml_path, cfg_test;
            aoi_coords = (lon = state.aoi_lon[], lat = state.aoi_lat[]),
            exclusion_boxes = state.exclusion_boxes[],
            stage_distribution = state.obs_stage_dist_init[]
        )
        @test isfile(toml_path)
        parsed = TOML.parsefile(toml_path)
        @test haskey(parsed, "aoi")
        @test parsed["aoi"]["lon"] == -63.5
        @test haskey(parsed, "spatial_exclusion")
        @test length(parsed["spatial_exclusion"]["boxes"]) == 1
        @test parsed["spatial_exclusion"]["boxes"][1]["name"] == "TestMPA"
        @test haskey(parsed, "initial_stage_distribution")
        @test length(parsed["initial_stage_distribution"]) == 20
    end
end
