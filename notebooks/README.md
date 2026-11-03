# Pluto Research Notebooks

The `TethysBioNotes.jl` Pluto notebook provides an interactive environment for configuring, running, and analyzing the individual-based model across spatial domains in the Northwest Atlantic (such as the Scotian Shelf and Laurentian Channel). The notebook documents biological assumptions, environmental inputs, allometric relationships, and assessment outputs.

| Notebook | Spatial Domain | Calibrated Species | Data Sources |
|---|---|---|---|
| [`TethysBioNotes.jl`](TethysBioNotes.jl) | Configurable Scotian Shelf / Laurentian Channel window | Snow crab (*Chionoecetes opilio*) | TOML configuration files, interactive controls, and lakehouse store via `Tethys.lakehouse_dir` |

## Functional Capabilities

The notebook organizes the end-to-end simulation workflow into five integrated stages:

Configuration and steering routines load `configs/snowcrab.toml` through `load_ibm_configuration`, layering reactive widgets for spatial domain selection, environmental temperature offsets and multipliers, fishery open seasons, spatial closure proportions, biological process toggles, and initial stage-abundance cohort distributions.

Model execution routines construct an `IBMParameters` instance from the merged configuration, initialize the lattice with bathymetry queried from the environmental store, and advance `run_simulation` over multi-annual horizons. By default, the notebook sets `run_model = false` and loads pre-computed results from local storage to prevent unintentional re-integration, while setting `run_model = true` triggers active simulation and updates results.

Diagnostic dashboards render spatial organism and habitat suitability distributions, bottom temperature fields, stage structures, sex- and maturity-stratified size frequencies, logistic maturity ogives, multi-annual biomass trajectories, commercial yield time series, and stock-recruitment relationships.

Macro-ecological population analysis fits surplus production models (Schaefer), delay-difference models (Deriso-Schnute), and stage-transition projection matrices (generalized Leslie and Lefkovitch formulations) to annual simulation series, summarizing comparative performance through coefficient of determination ($R^2$), root mean square error (RMSE), and Akaike Information Criterion (AIC) statistics.

Calibration assessment evaluates simulation trajectories against five biological criteria: population persistence, demographic bounds, size-structure stability, maturity-ogive fidelity, and growth-increment accuracy.

## Execution

Pluto is included as a project dependency. The notebook server can be launched directly from the package root:

```powershell
julia --project=. -e "using Pluto; Pluto.run()"
```

Upon launching, selecting `TethysBioNotes.jl` activates the parent project environment (`Pkg.activate`). Relative paths resolve against the notebook directory, and environmental spatial datasets resolve through `Tethys.lakehouse_dir("TethysBio")`.

Headless execution can be evaluated through Pluto's session API:

```powershell
julia --project=. scratch/run_notebook.jl
```

The script evaluates each cell sequentially, reporting completion status and verifying execution without requiring a graphical browser interface. Because Pluto notebooks maintain reactive dependency graphs, updating upstream configuration cells automatically propagates state changes throughout downstream simulation and visualization blocks.
