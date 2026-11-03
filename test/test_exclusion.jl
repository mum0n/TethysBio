# Test spatial exclusion zones
using Test
using Agents
using TethysBio

@testset "Spatial Exclusion Zones (AABB)" begin
    box1 = ExclusionBox(id = 1, name = "Reserve-A", xmin = 10.0, xmax = 20.0, ymin = 10.0, ymax = 20.0, active = true)
    box2 = ExclusionBox(id = 2, name = "Reserve-B", xmin = 30.0, xmax = 40.0, ymin = 30.0, ymax = 40.0, active = false)
    boxes = [box1, box2]

    @test is_in_exclusion_box(15.0, 15.0, box1) == true
    @test is_in_exclusion_box(5.0, 15.0, box1) == false
    @test is_in_exclusion_box(35.0, 35.0, box2) == false # inactive
    @test is_in_exclusion_zones(15.0, 15.0, boxes) == true
    @test is_in_exclusion_zones(35.0, 35.0, boxes) == false
    @test is_in_exclusion_zones(50.0, 50.0, boxes) == false

    # Test integration with model properties
    cfg = default_ibm_configuration()
    cfg[:days] = 10
    cfg[:grid] = (50, 50)
    params = IBMParameters(cfg)
    model = initialize_model(n_agents = 50, grid = (50, 50), params = params, seed = 42)
    @test haskey(abmproperties(model), :exclusion_boxes)
    push!(abmproperties(model)[:exclusion_boxes], box1)
    @test length(abmproperties(model)[:exclusion_boxes]) == 1
end
