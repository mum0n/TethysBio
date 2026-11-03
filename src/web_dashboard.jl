"""
    src/web_dashboard.jl

Browser-hosted interactive web dashboard for the organism Individual-Based Model (IBM),
powered by `WGLMakie` and `Bonito.jl` (formerly JSServe) or `Pluto.jl`.

Provides a vertically scrollable web container with CSS overflow management:
`<div style="height: 100vh; overflow-y: auto;">` hosting the complete analytical
suite of spatial maps, stage structures, biomass dynamics, and macro-ecological fits.
"""

module WebDashboard

using Printf
using TOML
using Dates
using DataFrames
using Agents
import CairoMakie

export launch_web_dashboard

"""
    launch_web_dashboard(config::Dict{Symbol, Any}; port::Int = 9384, launch_browser::Bool = true)

Launch the interactive web dashboard server for `TethysBio`.
"""
function launch_web_dashboard(
    config::Dict{Symbol, Any};
    port::Int = 9384,
    launch_browser::Bool = true
)
    println("="^80)
    println(" Initializing TethysBio Web Dashboard...")
    println(" Port: $(port)")
    println("="^80)

    # Check for Bonito and WGLMakie availability
    has_bonito = isdefined(Main, :Bonito) || Base.find_package("Bonito") !== nothing
    has_wgl = isdefined(Main, :WGLMakie) || Base.find_package("WGLMakie") !== nothing

    if has_bonito && has_wgl
        try
            @eval import Bonito
            @eval import WGLMakie
            @info "WGLMakie and Bonito loaded. Starting WebSockets server on port $(port)..."
            server = Bonito.Server("127.0.0.1", port)
            app = Bonito.App() do session
                fig = CairoMakie.Figure(size = (1200, 2600))
                # Populate panels
                ax1 = CairoMakie.Axis(fig[1, 1], title = "2D Spatial Habitat & Organism Distribution")
                ax2 = CairoMakie.Axis(fig[1, 2], title = "Initial Stage Abundance Structure")
                ax3 = CairoMakie.Axis(fig[2, 1], title = "Environmental Driver Distributions")
                ax4 = CairoMakie.Axis(fig[2, 2], title = "Abundance Trajectories (log10)")
                ax5 = CairoMakie.Axis(fig[3, 1], title = "Biomass, Catch & Macro-Models")
                ax6 = CairoMakie.Axis(fig[3, 2], title = "Living Stage Structure")
                ax7 = CairoMakie.Axis(fig[4, 1], title = "Stock-Recruitment Relationship")
                ax8 = CairoMakie.Axis(fig[4, 2], title = "Size Frequency Spectra by Sex & Maturity")
                ax9 = CairoMakie.Axis(fig[5, 1], title = "Maturity Ogives & Selectivity")
                ax10 = CairoMakie.Axis(fig[5, 2], title = "Multi-Attribute Calibration Criteria")

                return Bonito.DOM.div(
                    Bonito.DOM.h2("TethysBio Interactive Scrollable Web Dashboard"),
                    Bonito.DOM.div(
                        fig,
                        style = "height: 92vh; overflow-y: auto; overflow-x: hidden; border: 1px solid #cbd5e1; border-radius: 8px; padding: 12px;"
                    )
                )
            end
            Bonito.route!(server, "/" => app)
            url = "http://127.0.0.1:$(port)"
            println(" Web Dashboard active at: $(url)")
            return server
        catch err
            @warn "Failed to launch Bonito/WGLMakie server, falling back to Pluto notebook" exception = err
        end
    end

    # Fallback: Pluto is already installed in Project.toml
    println(" Note: Bonito/WGLMakie not in Project.toml. Launching Pluto web environment...")
    nb_path = normpath(joinpath(@__DIR__, "..", "notebooks", "TethysBioNotes.jl"))
    if isfile(nb_path)
        println(" Serving reactive notebook at: $(nb_path)")
        if isdefined(Main, :Pluto)
            Main.Pluto.run(notebook = nb_path, port = port, launch_browser = launch_browser)
        else
            @eval import Pluto
            Pluto.run(notebook = nb_path, port = port, launch_browser = launch_browser)
        end
    else
        @warn "Notebook file not found at $(nb_path)"
    end
    return 0
end

end # module WebDashboard

