using Dates
using TOML
using TethysBio

function register_tethysbio_catalog!()
    data_root = TethysBio_store_root()
    mkpath(data_root)
    catalog = Dict(
        "updated_utc" => string(now(Dates.UTC)),
        "datasets" => [
            Dict(
                "key" => "tethysbio_bathymetry",
                "name" => "Northwest Atlantic bathymetry",
                "path" => "environmental/bathymetry.zarr",
                "format" => "zarr",
                "variables" => ["elevation", "depth"]
            ),
            Dict(
                "key" => "tethysbio_temperature",
                "name" => "WOA23 temperature climatology",
                "path" => "environmental/woa23_temperature_00_0.25deg.zarr",
                "format" => "zarr",
                "variables" => ["t_an", "thetao"]
            ),
            Dict(
                "key" => "tethysbio_annual_production",
                "name" => "Annual production indicators",
                "path" => "results/simulation_results/annual_production.parquet",
                "format" => "parquet",
                "variables" => ["biomass", "catch", "recruits", "fecundity"]
            )
        ]
    )
    catalog_path = joinpath(data_root, "catalog.toml")
    open(catalog_path, "w") do io
        TOML.print(io, catalog)
    end
    println("Saved TethysBio local data catalog to: $(catalog_path)")
    return catalog_path
end

register_tethysbio_catalog!()
