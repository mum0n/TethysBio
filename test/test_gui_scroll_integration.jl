using Test
using TethysBio
using GLMakie
using Dates
using TOML
using InteractiveUtils

@testset "Tabbed GUI Integration & Shared Interactivity" begin
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
    top_bar = fig.layout.content[1].content
    top_bar_bbox = top_bar.layoutobservables.computedbbox[]
    main_area_bbox = fig.layout.content[2].content.layoutobservables.computedbbox[]

    @test top_bar_bbox.origin[1] >= 0
    @test top_bar_bbox.origin[1] + top_bar_bbox.widths[1] <= window_width
    @test main_area_bbox.origin[2] >= 0
    @test main_area_bbox.origin[2] + main_area_bbox.widths[2] <= window_height

    # Initial tab state
    @test state.active_tab[] == 1
    @test length(state.obs_stage_dist_init[]) == 20
    @test state.aoi_lon[] == -59.0
    @test state.aoi_lat[] == 45.0
    @test isempty(state.exclusion_boxes[])

    # Scotian Shelf regional map
    @test state.ax_aoi_map !== nothing
    @test state.ax_aoi_map.xreversed[] == false

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
        @test parsed["aoi"]["lon"] == -59.0
        @test haskey(parsed, "spatial_exclusion")
        @test length(parsed["spatial_exclusion"]["boxes"]) == 1
        @test parsed["spatial_exclusion"]["boxes"][1]["name"] == "TestMPA"
        @test haskey(parsed, "initial_stage_distribution")
        @test length(parsed["initial_stage_distribution"]) == 20
    end

    # Test parameter steering on tab2_controls_grid (Tab 2)
    main_area = fig.layout.content[2].content
    tab2_controls_grid = main_area.content[2].content
    tab2_col2 = tab2_controls_grid.content[2].content
    fish_grid = tab2_col2.content[2].content
    legal_size_input = first(
        c.content for c in fish_grid.content
        if c.content isa Textbox && c.span.rows == 4:4 && c.span.cols == 2:2
    )
    legal_size_input.stored_string[] = "100.0"
    @test state.model.params.fishery.legal_size_male == 100.0

    # Test adjusting coordinates on aoi_grid (Tab 1)
    controls_tab_grid = main_area.content[1].content
    ctrl_col1 = controls_tab_grid.content[1].content
    aoi_grid = ctrl_col1.content[2].content
    lon_input = first(c.content for c in aoi_grid.content if c.content isa Textbox && c.span.rows == 1:1)
    lat_input = first(c.content for c in aoi_grid.content if c.content isa Textbox && c.span.rows == 2:2)
    lat_input.stored_string[] = "43.5"
    lon_input.stored_string[] = "-63.6"
    @test state.aoi_lon[] == -63.6
    @test state.aoi_lat[] == 43.5
    @test state.current_time[] == 0.0
    @test sum(state.model.params.domain_lon) / 2 ≈ -63.6
    @test size(state.model.depth_field) == (15, 15)

    # Test tab navigation through all pages
    for t in 1:4
        state.active_tab[] = t
        @test state.active_tab[] == t
    end

    # Step simulation and test rewind on top_bar
    state.active_tab[] = 1
    step_gui_simulation!(state, 15.0)
    @test state.current_time[] == 15.0
    top_bar = fig.layout.content[1].content
    rewind_input = first(
        c.content for c in top_bar.content
        if c.content isa Textbox && c.span.cols == 13:13
    )
    rewind_button = first(
        c.content for c in top_bar.content
        if c.content isa Button && c.span.cols == 14:14
    )
    rewind_input.stored_string[] = "5"
    rewind_button.clicks[] += 1
    @test state.current_time[] == 5.0

    # Test grid resolution entry on tab2_controls_grid
    tab2_col1 = tab2_controls_grid.content[1].content
    sim_env_grid = tab2_col1.content[2].content
    grid_input = first(
        c.content for c in sim_env_grid.content
        if c.content isa Textbox && c.span.rows == 1:1 && c.span.cols == 2:2
    )
    grid_input.stored_string[] = "30"
    @test size(state.model.depth_field) == (30, 30)
    @test state.current_time[] == 0.0
end
