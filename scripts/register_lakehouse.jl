# Script to register TethysBio datasets in projects/lakehouse/catalog.toml
using Tethys
using Dates
using UUIDs

function register_tethysbio_catalog!()
    lakehouse_root = normpath(Tethys.lakehouse_dir())
    cat_path = joinpath(lakehouse_root, "catalog.toml")
    
    cat = Tethys.GeoDataCatalog()
    if isfile(cat_path)
        Tethys.load_catalog!(cat, cat_path)
    end

    datasets = [
        Tethys.DatasetEntry(
            name = "NW Atlantic Bathymetry (TethysBio)",
            key = :tethysbio_bathymetry,
            variables = ["elevation", "depth"],
            spatial_bounds = (-75.0, -45.0, 38.0, 60.0),
            location = "TethysBio/bathymetry.zarr",
            format = :zarr,
            tier = :processed,
            producer = "TethysBio",
            notes = "Gridded bathymetric depth raster for Scotian Shelf and Gulf of St. Lawrence"
        ),
        Tethys.DatasetEntry(
            name = "World Ocean Atlas 2023 Ambient Bottom Temperature",
            key = :tethysbio_temperature,
            variables = ["temperature"],
            spatial_bounds = (-75.0, -45.0, 38.0, 60.0),
            location = "TethysBio/woa23_temperature_00_0.25deg.zarr",
            format = :zarr,
            tier = :processed,
            producer = "TethysBio",
            notes = "Ambient bottom thermal climatology at 0.25 degree resolution"
        ),
        Tethys.DatasetEntry(
            name = "World Ocean Atlas 2023 Ambient Bottom Salinity",
            key = :tethysbio_salinity,
            variables = ["salinity"],
            spatial_bounds = (-75.0, -45.0, 38.0, 60.0),
            location = "TethysBio/woa23_salinity_00_0.25deg.zarr",
            format = :zarr,
            tier = :processed,
            producer = "TethysBio",
            notes = "Ambient bottom salinity climatology at 0.25 degree resolution"
        ),
        Tethys.DatasetEntry(
            name = "Organism IBM Annual Production Demographics",
            key = :tethysbio_annual_production,
            variables = ["biomass", "catch", "recruits", "fecundity", "mature_numbers"],
            spatial_bounds = (-75.0, -45.0, 38.0, 60.0),
            location = "TethysBio/simulation_results/annual_production.parquet",
            format = :parquet,
            tier = :derived,
            producer = "TethysBio",
            notes = "Annual demographic indicators and fisheries yield series"
        ),
        Tethys.DatasetEntry(
            name = "Organism IBM Run Metadata and Provenance",
            key = :tethysbio_run_metadata,
            variables = ["metadata", "parameters", "seed", "runtime"],
            spatial_bounds = (-75.0, -45.0, 38.0, 60.0),
            location = "TethysBio/simulation_results/run_metadata.parquet",
            format = :parquet,
            tier = :derived,
            producer = "TethysBio",
            notes = "Execution provenance, configuration snapshot, and environment metadata"
        )
    ]

    for d in datasets
        if haskey(cat.key_index, d.key)
            existing_uuid = cat.key_index[d.key]
            d.uuid = existing_uuid
            cat.entries[existing_uuid] = d
            println("Updated catalog dataset: $(d.key)")
        else
            Tethys.register_dataset!(cat, d)
            println("Registered catalog dataset: $(d.key)")
        end
    end

    Tethys.save_catalog(cat, cat_path)
    println("Saved updated catalog to: $(cat_path)")
end

register_tethysbio_catalog!()

