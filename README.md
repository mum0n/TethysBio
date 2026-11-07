# Individual and Agent Based Models (TethysBio)

The `TethysBio` framework implements a species-parameterized individual-based simulation engine in Julia utilizing [`Agents.jl`](https://juliadynamics.github.io/Agents.jl/stable/). Biological processes, including growth allometry, maturation schedules, ecdysial phenology, and reproductive cycles, are decoupled from the generic simulation engine through the `SpeciesBehavior` dispatch interface. The reference implementation is calibrated for the snow crab (*Chionoecetes opilio*) via `configs/snowcrab.toml`.

## Quick Start

The simulation engine and test suite can be initialized and executed from the command line:

```powershell
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. test/runtests.jl
julia --project=. scripts/organism_ibm.jl --config=configs/default.toml --days=365
julia --project=. scripts/organism_ibm.jl --config=configs/snowcrab.toml --save
```

## Behavioral Dispatch and Reporting Invariants

The `SpeciesBehavior` abstract type establishes the primary extension interface for biological implementations. Generic simulation routines query species-specific rates, allometric coefficients, and developmental stage limits strictly through dynamic dispatch on this type. Methods such as `stage_size_lower_bound`, `growth_increment`, and `build_stages` require an explicit behavioral instance, ensuring that newly implemented species are evaluated against their own parameterizations rather than silently inheriting reference coefficients.

Two reporting invariants characterize demographic accounting. In accordance with standard fisheries stock assessment conventions, total catch (`catch_total`) and legal catch (`catch_legal`) are recorded as distinct telemetry columns. Under the baseline commercial configuration, the harvest gate evaluates individual eligibility through `is_legal_male`, restricting removals strictly to morphometrically mature males exceeding legal size limits ($CW \ge 95$ mm). Both series report identical values unless experimental configurations explicitly admit sublegal retention. Additionally, commercial fishery activation is controlled through a single parameter, `fishery.fishing_probability_daily`. A value of zero completely deactivates harvest removals without requiring secondary configuration switches, ensuring unambiguous correspondence between input parameters and model outputs.

## Architecture

The framework consists of two complementary components:

The generic simulation engine manages the discrete two-dimensional spatial lattice (`GridSpace`), executes the agent scheduling loop, advances pre-recruit pelagic instars within aggregate cohort containers, and maintains demographic caches for state reporting.

The biological parameterization layer encapsulates all developmental thresholds, allometric scaling constants, metabolic coefficients, and phenological schedules within twelve typed parameter structures. Simulation code contains no hardcoded species constants. Baseline defaults are declared in `configs/default.toml`, while species-specific calibrations are supplied via overlay files such as `configs/snowcrab.toml`.

```
TethysBio/
├── Project.toml
├── configs/
│   ├── default.toml          # baseline parameter struct defaults
│   └── snowcrab.toml         # snow crab empirical calibration and operating scale
├── docs/
│   ├── TethysBio.md          # comprehensive system reference and API documentation
│   └── snowcrab_ibm_research_paper.md # mathematical specification and research paper
├── src/
│   ├── TethysBio.jl          # module definition and public API exports
│   ├── types.jl              # Organism struct, parameter containers, configuration parsing
│   ├── species.jl            # SpeciesBehavior implementation, stage definitions, allometry
│   ├── organism_dynamics.jl  # bioenergetics, mortality hazards, cohorts, movement
│   ├── simulation.jl         # grid setup, agent/model step execution, batch simulations
│   ├── analysis.jl           # demographic summaries, macro-ecological models, criteria
│   ├── cli.jl                # argument parsing, telemetry reporting, batch orchestration
│   ├── visualization.jl      # diagnostic figures and video recording via CairoMakie
│   ├── gui.jl                # real-time interactive dashboard via GLMakie
│   ├── assimilation.jl       # environmental store ingestion and data reading
│   └── notebook_support.jl   # interactive widgets and plotting helpers for Pluto
├── notebooks/
│   ├── README.md             # instructions for running Pluto notebooks
│   └── TethysBioNotes.jl     # research notebook covering configuration, calibration, and models
├── scripts/
│   ├── organism_ibm.jl       # simulation execution CLI
│   ├── organism_orchestrator.jl # calibration and sensitivity search driver
│   └── fit_macro_models.jl   # post-hoc macro-ecological model fitting tool
└── test/
    ├── runtests.jl           # test suite dispatcher across modular segments
    ├── test_allometry_biology.jl # growth, maturity, fecundity, and mortality validation
    ├── test_config_parameters.jl # configuration parsing, serialization, and parameter checks
    ├── test_fixtures.jl      # shared synthetic test fixtures
    ├── test_habitat_environment.jl # spatial grids and environmental store integration
    ├── test_macro_dynamics.jl # macro surplus production and Leslie matrix tests
    ├── test_simulation_agents.jl # agent lifecycle, bioenergetics, and stepping invariants
    └── test_visualization_gui.jl # figure generation and dashboard tests
```

## Environmental Data Assimilation

TethysBio is independent of Tethys and shared lakehouse stores. Environmental inputs, downloaded caches, simulation results, plots, and notebook caches are stored under this project in `data/`, `outputs/`, and `scratch/`. Paths that escape the project directory are rejected.

Bathymetry and hydrography are read from `data/environmental/` and `data/inputs/`. Missing ETOPO 2022 bathymetry and WOA23 climatology can be downloaded directly into the local data directory. `environmental_input_status` reports local availability, `assimilate_environmental_inputs` fetches missing data, and `read_assimilated_environment` returns the fields used by the simulation. The default configuration uses synthetic environmental fields and does not download data.

## Life-History Formulation and Mathematical Mechanics

The lifecycle models pelagic embryogenesis and larval progression within aggregate cohort structures, followed by post-larval settlement, discontinuous benthic growth, sexual differentiation, terminal morphometric maturity, and reproductive extrusion.

Biological process toggles defined in `IBMParameters.toggles` allow specific physiological mechanisms to be activated or deactivated independently, including bioenergetics, thermal stress, cannibalism, adult senescence, spermathecal storage, directed habitat taxis, degree-day ecdysial gating, molting, and reproduction. Commercial harvesting is governed independently through `fishery.fishing_probability_daily`.

Post-molt carapace width is modeled via the empirical logarithmic Hiatt formulation:

$$\ln(CW_{t+1}) = \beta_0 + \beta_1 \ln(CW_t) + \delta_{\text{mat}} \cdot \mathbb{I}_{\text{mat}} + c \cdot CC + \gamma_T \cdot (T - T_{\text{ref}}) + \epsilon$$

where $\beta_0$ and $\beta_1$ represent log-scale growth coefficients, $\delta_{\text{mat}}$ denotes the growth increment reduction associated with morphometric maturity, $CC$ accounts for carapace condition, and $\epsilon \sim \mathcal{N}(0, \sigma^2)$ reflects individual phenotypic variation. To prevent exponential compounding, post-molt dimensions are bounded by the species asymptotic size limit (`species.max_size_male`, `species.max_size_female`).

The lower size boundary for each developmental stage follows a saturating exponential formulation:

$$\ln S(\iota) = \ln L_\infty + (\ln S_0 - \ln L_\infty) \exp(-\kappa (\iota - \iota_0))$$

anchored at early instars and approaching asymptotic dimension $L_\infty$, ensuring that staged size intervals remain strictly within biological limits.

Morphometric maturity probability at ecdysis is governed by a logistic ogive parameterized by median size $CW_{50}$ and slope $\beta$:

$$P(\text{mature} \mid CW, T) = \frac{1}{1 + \exp(-(\alpha + \beta \ln(CW) + \gamma_T (T - T_{\text{ref}})))}$$

Terminal morphometric maturity is irreversible; once an individual matures, ecdysial growth ceases permanently.

Daily natural mortality integrates multiple independent hazards:

$$M_{\text{daily}} = M_{\text{base}} + M_{\text{pred}}(CW) + M_{\text{soft}}(\Delta t) + M_{\text{starve}}(E) + M_{\text{thermal}}(T) + M_{\text{senescence}}(\tau) + M_{\text{cannibal}}$$

Total daily mortality is modulated by a population density feedback term $1 + (N / K)^\gamma$, where $N$ represents total benthic abundance (individuated agents plus unindividuated aggregate cohorts) and $K$ denotes benthic carrying capacity (`environment.benthic_carrying_capacity`). Carrying capacity must exceed seeded abundance to avoid excessive baseline mortality upon simulation initialization.

Brood fecundity is parameterized through allometric power functions differentiated by maternal reproductive history, distinguishing primiparous females producing their first clutch from multiparous females producing subsequent broods. Fecundity in older multiparous females undergoes cuticular senescence discounting based upon post-terminal shell age.

## Performance and Computational Benchmarks

Simulations evaluated on a $1000 \times 1000$ lattice (representing a $1\text{ km} \times 1\text{ km}$ spatial stratum at 1 m cell resolution) with carrying capacity $K = 10000$ exhibit execution speeds suitable for multi-annual projection and Monte Carlo calibration. Benchmark evaluations record median execution times of 40.3 ms per simulated day for dynamics alone on a $1000 \times 1000$ grid, and 49.9 ms per day with complete per-step observation recording enabled. A 730-day (two-year) simulation completes in approximately 24 seconds for core dynamics and 36 seconds with complete demographic recording.

Computational efficiency is achieved through in-place field updates (`randn!`), affine vector operations for food renewal, cohort grouping to limit agent container overheads, and pre-computed seasonal variables cached on the model container. Random order scheduling via `Agents.Schedulers.Randomly()` ensures reproducible state trajectories when random seeds are held constant.

## Configuration and Command-Line Interface

Parameter configurations are assembled hierarchically across three tiers: struct defaults defined on the twelve parameter types, baseline defaults defined in `configs/default.toml`, and species overlays defined in `configs/snowcrab.toml`, with command-line flags providing final runtime overrides.

The simulation script (`scripts/organism_ibm.jl`) accepts exact parameter names as command-line arguments:

```powershell
julia --project=. scripts/organism_ibm.jl --config=configs/snowcrab.toml --days=730 --save
```

Common options include `--days` for simulated duration, `--n_agents` for initial population size, `--grid` for spatial lattice dimensions, `--temp_offset` for baseline thermal shifts, `--fishing_probability_daily` for commercial harvest rates, and `--save` for writing results under `data/results`.

Simulation results are serialized as Zstd-compressed Parquet files under `data/results/` (`simulation_results/annual_production.parquet` and `run_metadata.parquet`). Stored outputs can be loaded directly for post-hoc macro-ecological model fitting:

```powershell
julia --project=. scripts/fit_macro_models.jl --file=simulation_results --retro
```

## Programmatic API

The package exports high-level functions for initializing, stepping, and analyzing simulations:

```julia
using TethysBio

params = IBMParameters(
    fishery = FisheryParams(fishing_probability_daily = 0.005),
    environment = EnvironmentParams(temp_offset = 1.0)
)

model = initialize_model(
    n_agents = 2000,
    grid = (1000, 1000),
    seed = 1234,
    params = params
)

agent_df, model_df = run_simulation(model, 730; when = 7)
summary = summarize_population(model)
```

## Test Suite

The verification suite in `test/runtests.jl` partitions tests across modular segments:

```powershell
julia --project=. test/runtests.jl biology      # allometric growth, maturity, fecundity
julia --project=. test/runtests.jl habitat      # habitat suitability, environmental store
julia --project=. test/runtests.jl macro        # macro-ecological models and Leslie matrices
julia --project=. test/runtests.jl config       # TOML configuration and parameter parsing
julia --project=. test/runtests.jl simulation   # agent stepping, bioenergetics, dispersal
julia --project=. test/runtests.jl vis          # diagnostic plots and visual interfaces
```

All segments execute automated checks validating empirical growth rules, mortality hazards, configuration round-tripping, cohort individuation, demographic accounting, and macro-model parameter estimation.
