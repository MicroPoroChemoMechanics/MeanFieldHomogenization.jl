# =============================================================================
#  make_excentered_figures.jl — the surrogate of `FEExcenteredSphere` against
#  the finite elements it replaces.
#
#  Maintenance script, run **by hand** after `train_excentered.jl`: it performs a
#  few dozen finite-element solves, which must never happen at documentation-build
#  time. The tutorial `scripts/85_neural_excentered_sphere.jl` prints the
#  committed markdown and embeds the committed PNG.
#
#      julia scripts/nn/make_excentered_figures.jl
#
#  Outputs (committed):
#      docs/src/assets/nn/excentered_accuracy.png
#      src/NeuralInclusions/models/excentered_comparison.md
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

import Ferrite, FerriteGmsh, Gmsh

gr()
default(;
    fontfamily = "sans-serif", framestyle = :box, grid = true, legendfontsize = 8,
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

const NI = MeanFieldHomogenization.NeuralInclusions
const OUT = NI.MODEL_DIR
const ASSET = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nn"))
mkpath(ASSET)

# The same fixed parameters the surrogate was trained at.
const NU = 0.2
const E0 = 20.0
const MU1_RATIO = 3.5
const FRAC = 0.4
const MESH = (; nradial = 18, radius_ratio = 4.0)

ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C0 = ce(E0, NU)
const C1 = ce(MU1_RATIO * E0, NU)

const STRAIN = load_surrogate(model_path("excentered_sphere_strain"))
const STRESS = load_surrogate(model_path("excentered_sphere_stress"))

fe_incl(α, w, e2) = FEExcenteredSphere(
    1.0, (C1, ce(e2 * E0, NU)); core_fraction = w, eccentricity = α, MESH...
)

nn_incl(α, w, e2) = NeuralLocalizationInclusion(
    (1.0, 1.0, 1.0);
    strain = STRAIN, stress = STRESS,
    shape_params = (; eccentricity = α, core_fraction = w),
    fractions = (w, 1 - w), properties = (C1, ce(e2 * E0, NU)), guard = :error,
)

iso(C) = MeanFieldHomogenization.Core.isotropify(C)

function young(incl)
    r = RVE()
    add_phase!(r, :paste, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    add_phase!(r, :rca, incl, Dict(:C => C0); fraction = FRAC)
    k, μ = k_mu(iso(homogenize(r, MoriTanaka(), :C)))
    return 9k * μ / (3k + μ) / E0
end

# ─── Accuracy on the tensors and on the effective modulus ────────────────────

const ALPHAS = (0.0, 0.4, 0.8)
const E2S = 10 .^ range(-1.0, log10(2.0), length = 9)

println("comparing the surrogate against the finite elements …")
E_fe = Dict{Float64, Vector{Float64}}()
E_nn = Dict{Float64, Vector{Float64}}()
tens_err = Float64[]
for α in ALPHAS
    fe, nn = Float64[], Float64[]
    for e2 in E2S
        gf, gn = fe_incl(α, 0.5, e2), nn_incl(α, 0.5, e2)
        A_fe = fe_axi_localization(gf, C0)[1]
        A_nn = strain_strain_loc(gn, C0, C0)
        push!(
            tens_err,
            maximum(abs, get_array(A_nn) .- get_array(A_fe)) / maximum(abs, get_array(A_fe))
        )
        push!(fe, young(gf))
        push!(nn, young(gn))
    end
    E_fe[α], E_nn[α] = fe, nn
    @printf("  α = %.1f done\n", α)
end

# ─── Cost ────────────────────────────────────────────────────────────────────

gf = fe_incl(0.4, 0.5, 0.4)
fe_axi_mesh_report(gf)                                  # mesh out of the timing
t_fe = @elapsed fe_axi_localization(gf, ce(1.01 * E0, NU))

gn = nn_incl(0.4, 0.5, 0.4)
strain_strain_loc(gn, C0, C0)
const N_EVAL = 2000
t_nn = @elapsed for _ in 1:N_EVAL
    strain_strain_loc(gn, C0, C0)
end
t_nn /= N_EVAL

# ─── Figure ──────────────────────────────────────────────────────────────────

# Left, the modulus itself — the three eccentricities all but superpose, so the
# panel shows the agreement and little else. Right, the same data read against
# the concentric case, which is the only way to see the physics *and* the only
# demanding test of the surrogate: the effect it has to reproduce is a percent.
const COLS = (:black, :steelblue, :firebrick)

p_abs = plot(;
    xlabel = "E₂ / E₀   (adhered mortar / fresh paste)", ylabel = "E_eff / E₀",
    xscale = :log10, legend = :topleft,
    title = "Mori-Tanaka, f = $FRAC, w = 0.5, E₁/E₀ = $MU1_RATIO, ν = $NU",
    titlefontsize = 10,
)
for (α, col) in zip(ALPHAS, COLS)
    plot!(p_abs, E2S, E_fe[α]; lc = col, lw = 2, label = "finite elements, α = $α")
    scatter!(
        p_abs, E2S, E_nn[α];
        mc = col, ms = 5, msw = 0, markershape = :circle, label = "surrogate, α = $α"
    )
end

p_rel = plot(;
    xlabel = "E₂ / E₀   (adhered mortar / fresh paste)",
    ylabel = "E_eff(α) / E_eff(0) − 1   (%)", xscale = :log10, legend = :topright,
    title = "effect of the eccentricity, same parameters", titlefontsize = 10,
)
for (α, col) in zip(ALPHAS[2:end], COLS[2:end])
    plot!(
        p_rel, E2S, 100 .* (E_fe[α] ./ E_fe[0.0] .- 1);
        lc = col, lw = 2, label = "finite elements, α = $α"
    )
    scatter!(
        p_rel, E2S, 100 .* (E_nn[α] ./ E_nn[0.0] .- 1);
        mc = col, ms = 5, msw = 0, markershape = :circle, label = "surrogate, α = $α"
    )
end
hline!(p_rel, [0.0]; lc = :black, lw = 0.6, label = "")

savefig(
    plot(
        p_abs, p_rel; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1150, 470),
        plot_title = "Surrogate against the finite elements it replaces " *
            "(nradial = $(MESH.nradial), R/a = $(MESH.radius_ratio))",
        plot_titlefontsize = 11,
    ),
    joinpath(ASSET, "excentered_accuracy.png")
)
println("wrote ", joinpath(ASSET, "excentered_accuracy.png"))

# ─── Report ──────────────────────────────────────────────────────────────────

rel(a, b) = maximum(abs.(a .- b) ./ abs.(b))

open(joinpath(OUT, "excentered_comparison.md"), "w") do io
    println(io, "<!-- Generated by scripts/nn/make_excentered_figures.jl — do not edit by hand. -->\n")
    println(io, "Mori-Tanaka, `f = $FRAC`, `w = 0.5`, `E₁/E₀ = $MU1_RATIO`, `ν = $NU`,")
    println(io, "`nradial = $(MESH.nradial)`, `R/a = $(MESH.radius_ratio)`; `E₂/E₀` swept over a decade and a half.\n")
    println(io, "| Quantity | Worst relative deviation from the finite elements |")
    println(io, "| --- | ---: |")
    @printf(io, "| `𝔸_εε` (the tensor itself) | %.2e |\n", maximum(tens_err))
    for α in ALPHAS
        @printf(io, "| `E_eff/E₀` at α = %.1f | %.2e |\n", α, rel(E_nn[α], E_fe[α]))
    end
    println(io)
    println(io, "| Cost of one evaluation | |")
    println(io, "| --- | ---: |")
    @printf(io, "| finite elements, cold | %.3f s |\n", t_fe)
    @printf(io, "| surrogate | %.1f µs |\n", 1.0e6t_nn)
    @printf(io, "| **speed-up** | **%.0f×** |\n", t_fe / t_nn)
end
println("wrote ", joinpath(OUT, "excentered_comparison.md"))
@printf("\nworst tensor error %.2e, speed-up %.0f×\n", maximum(tens_err), t_fe / t_nn)
