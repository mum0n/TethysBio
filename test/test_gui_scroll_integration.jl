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
    window_width = GLMakie.size(fig.scene)[1]
    window_height = GLMakie.size(fig.scene)[2]
    control_bbox = fig.layout.content[1].content.layoutobservables.computedbbox[]
    control_content_bbox =
        fig.layout.content[1].content.content[2].content.layoutobservables.computedbbox[]
    scroll_bbox = fig.layout.content[2].content.layoutobservables.computedbbox[]
    @test control_bbox.origin[1] >= 0
    @test control_bbox.origin[1] + control_bbox.widths[1] <= window_width
    @test control_content_bbox.origin[2] >= 0
    @test control_content_bbox.origin[2] + control_content_bbox.widths[2] <= window_height
    @test scroll_bbox.origin[1] >= 0
    @test scroll_bbox.origin[1] + scroll_bbox.widths[1] <= window_width
    @test length(state.obs_stage_dist_init[]) == 20
    @test state.aoi_lon[] == -63.5
    @test state.aoi_lat[] == 44.5
    @test isempty(state.exclusion_boxes[])

    depth_field = state.model.depth_field
    @test size(depth_field) == (15, 15)
    state.obs_bg_layer[] = :depth
    step_gui_simulation!(state, 0.0)
    @test state.obs_env_field[] == depth_field

    # Advance model by 15 days
    step_gui_simulation!(state, 15.0)
    @test state.current_time[] >= 15.0
    @test state.obs_env_field[] == depth_field
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

    left_grid = fig.layout.content[1].content.content[2].content
    parameter_grid = left_grid.content[5].content
    legal_size_input = first(
        c.content for c in parameter_grid.content
        if c.content isa Textbox && c.span.rows == 6:6 && c.span.cols == 4:4
    )
    legal_size_input.stored_string[] = "100.0"
    @test state.model.params.fishery.legal_size_male == 100.0

    aoi_grid = fig.layout.content[1].content.content[2].content.content[3].content.content[1].content
    lon_input = first(c.content for c in aoi_grid.content if c.content isa Textbox)
    lon_input.stored_string[] = "-63.4"
    @test state.aoi_lon[] == -63.4
    @test state.current_time[] == 0.0
    @test sum(state.model.params.domain_lon) / 2 ≈ -63.4
    @test size(state.model.depth_field) == (15, 15)
    step_gui_simulation!(state, 15.0)
    @test state.current_time[] == 15.0
    transport_grid = left_grid.content[2].content
    rewind_input = first(
        c.content for c in transport_grid.content
        if c.content isa Textbox && c.span.rows == 5:5 && c.span.cols == 8:8
    )
    rewind_button = first(
        c.content for c in transport_grid.content
        if c.content isa Button && c.span.rows == 5:5 && c.span.cols == 9:10
    )
    rewind_input.stored_string[] = "5"
    rewind_button.clicks[] += 1
    @test state.current_time[] == 5.0

    grid_input = first(
        c.content for c in transport_grid.content
        if c.content isa Textbox && c.span.rows == 7:7 && c.span.cols == 9:10
    )
    grid_input.stored_string[] = "30"
    @test size(state.model.depth_field) == (30, 30)
    @test state.current_time[] == 0.0
end
