# =============================================================================
#  make_layered_spheroid_figures.jl — the three routes to a layered spheroid,
#  and the honest price of differentiating a fit.
#
#  On the **confocal** slice `𝔸_εε` has a closed form, so a derivative has an
#  exact reference instead of being judged against a discretized cell. That is
#  what this figure uses: `LayeredSpheroid` differenced at h = 1e-4, converged
#  to 1e-9 because the function is smooth and exact.
#
#  Maintenance script, run **by hand** after `train_layered_spheroid.jl`: about
#  seventy finite-element solves, which must never happen at documentation-build
#  time. The tutorial page is static and embeds the committed PNG.
#
#      julia scripts/nn/make_layered_spheroid_figures.jl
#
#  Outputs (committed):
#      docs/src/assets/nn/layered_spheroid_accuracy.png
#      src/NeuralInclusions/models/layered_spheroid_comparison.md
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

import ForwardDiff

import Ferrite, FerriteGmsh, Gmsh

gr()
default(;
    fontfamily = "sans-serif", framestyle = :box, grid = true, legendfontsize = 7,
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
)

const NI = MeanFieldHomogenization.NeuralInclusions
const CAN = TensND.CanonicalBasis{3, Float64}()
const OUT = NI.MODEL_DIR
const ASSET = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "nn"))
mkpath(ASSET)

# Exactly the parameters the surrogate was trained at, the mesh included: it is
# the teacher, not an approximation of something else.
const NU = 0.2
const E0 = 1.0
const R1 = 3.0                    # stiff core,  E₁/E₀, inside the trained box
const R2 = 0.6                    # soft shell,  E₂/E₀
const FRAC = 0.3                  # the inclusion's volume fraction in the RVE
const MESH = (; nradial = 14, radius_ratio = 5.0)

ce(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C0 = ce(E0, NU)
const CS = (ce(R1 * E0, NU), ce(R2 * E0, NU))

const STRAIN = load_surrogate(model_path("layered_spheroid_strain"))
const STRESS = load_surrogate(model_path("layered_spheroid_stress"))

a1111(t) = Array(TensND.get_array(TensND.change_tens(t, CAN)))[1, 1, 1, 1]
radii(ω, w) = confocal_layer_radii(ω, 1.0, (w, 1 - w))

fe_incl(ω, w) = (r = radii(ω, w);
    FEAxiLayeredSpheroid(r[1], r[2], CS; opts = FEAxiMeshOptions(; MESH...)))
an_incl(ω, w) = (r = radii(ω, w); LayeredSpheroid(r[1], r[2], CS))
nn_incl(ω, w) = NeuralLocalizationInclusion(
    (1.0, 1.0, ω);
    strain = STRAIN, stress = STRESS,
    shape_params = (; core_fraction = w),
    fractions = (w, 1 - w), properties = CS, guard = :error,
)

# `(𝔸_εε)₁₁₁₁` by each of the three routes.
A_an(ω, w) = a1111(strain_strain_loc(an_incl(ω, w), C0, C0))
A_fe(ω, w) = a1111(fe_axi_localization(fe_incl(ω, w), C0)[1])
A_nn(ω, w) = a1111(strain_strain_loc(nn_incl(ω, w), C0, C0))

# The effective stiffness, which has **no** closed form: it needs both sides of
# gate B, and the analytic type supplies no stress side.
idx = C -> Array(get_array(C))[1, 1, 1, 1]
function rve_of(incl)
    r = RVE()
    add_phase!(r, :matrix, Ellipsoid(1.0), Dict(:C => C0); fraction = :rest)
    add_phase!(r, :incl, incl, Dict(:C => C0); fraction = FRAC)
    return r
end
c1111(incl) = idx(homogenize(rve_of(incl), MoriTanaka(), :C))

# Step sizes chosen per route, and each for a reason. The closed form is smooth
# and exact, so 1e-4 is converged to 1e-9 — measured, not assumed. The cell
# remeshes at every `w`, but its `ncells` grows smoothly and a step study showed
# the quotient flat from 5e-3 to 1e-1, so 2e-2 is safely inside that plateau.
cdiff(f, ω, w, h) = (f(ω, w + h) - f(ω, w - h)) / 2h
const H_AN = 1.0e-4
const H_FE = 2.0e-2

const OMEGAS = (1.25, 2.0, 3.0)
const WS = collect(range(0.24, 0.66, length = 7))       # inside the box [0.20, 0.70]

println("the three routes, values and derivatives …")
const _D() = Dict{Float64, Vector{Float64}}()
V_an, V_fe, V_nn = _D(), _D(), _D()
D_an, D_fe, D_nn = _D(), _D(), _D()
E_fe, E_nn = _D(), _D()
for ω in OMEGAS
    V_an[ω] = [A_an(ω, w) for w in WS]
    V_nn[ω] = [A_nn(ω, w) for w in WS]
    D_an[ω] = [cdiff(A_an, ω, w, H_AN) for w in WS]
    D_nn[ω] = [ForwardDiff.derivative(w -> A_nn(ω, w), w) for w in WS]
    V_fe[ω] = [A_fe(ω, w) for w in WS]
    D_fe[ω] = [cdiff(A_fe, ω, w, H_FE) for w in WS]
    E_fe[ω] = [c1111(fe_incl(ω, w)) for w in WS]
    E_nn[ω] = [c1111(nn_incl(ω, w)) for w in WS]
    @printf("  c/a = %.2f done\n", ω)
    flush(stdout)
end

# ─── Cost ────────────────────────────────────────────────────────────────────

gf = fe_incl(2.0, 0.4)
fe_axi_mesh_report(gf)                                  # mesh out of the timing
t_fe = @elapsed fe_axi_localization(gf, ce(1.01 * E0, NU))
gn = nn_incl(2.0, 0.4)
strain_strain_loc(gn, C0, C0)
const N_EVAL = 2000
t_nn = @elapsed for _ in 1:N_EVAL
    strain_strain_loc(gn, C0, C0)
end
t_nn /= N_EVAL

# ─── Figure ──────────────────────────────────────────────────────────────────

const COLS = (:black, :steelblue, :firebrick)
pct(a, b) = 100 .* abs.((a .- b) ./ b)

p1 = plot(;
    xlabel = "w   (core's share of the inclusion)", ylabel = "(𝔸_εε)₁₁₁₁",
    legend = :bottomleft, titlefontsize = 9,
    title = "the tensor itself: closed form, cell, surrogate",
)
for (ω, col) in zip(OMEGAS, COLS)
    plot!(p1, WS, V_an[ω]; lc = col, lw = 2, label = "closed form, c/a = $ω")
    scatter!(
        p1, WS, V_fe[ω]; mc = col, ms = 5, msw = 0, marker = :circle,
        label = "finite elements, c/a = $ω"
    )
    scatter!(
        p1, WS, V_nn[ω]; mc = col, msc = col, ms = 5, msw = 2, marker = :xcross,
        label = "surrogate, c/a = $ω"
    )
end

p2 = plot(;
    xlabel = "w   (core's share of the inclusion)",
    ylabel = "|deviation| from the closed form  (%)",
    yscale = :log10, legend = :right, titlefontsize = 9,
    title = "and the hierarchy that comparison establishes",
)
for (ω, col) in zip(OMEGAS, COLS)
    plot!(
        p2, WS, pct(V_fe[ω], V_an[ω]);
        lc = col, lw = 2, marker = :circle, ms = 4, msw = 0,
        label = "finite elements, c/a = $ω"
    )
    plot!(
        p2, WS, pct(V_nn[ω], V_an[ω]);
        lc = col, lw = 2, ls = :dash, marker = :xcross, ms = 5, msw = 2, msc = col,
        label = "surrogate, c/a = $ω"
    )
end

p3 = plot(;
    xlabel = "w   (core's share of the inclusion)", ylabel = "∂(𝔸_εε)₁₁₁₁ / ∂w",
    legend = :bottomright, titlefontsize = 9,
    title = "the derivative, against an exact reference",
)
for (ω, col) in zip(OMEGAS, COLS)
    plot!(p3, WS, D_an[ω]; lc = col, lw = 2, label = "closed form, c/a = $ω")
    scatter!(
        p3, WS, D_fe[ω]; mc = col, ms = 5, msw = 0, marker = :circle,
        label = "differenced cell, c/a = $ω"
    )
    scatter!(
        p3, WS, D_nn[ω]; mc = col, msc = col, ms = 5, msw = 2, marker = :xcross,
        label = "surrogate, c/a = $ω"
    )
end

p4 = plot(;
    xlabel = "w   (core's share of the inclusion)", ylabel = "C₁₁₁₁ of the estimate",
    legend = :topleft, titlefontsize = 9,
    title = "what has no closed form at all: both sides of gate B",
)
for (ω, col) in zip(OMEGAS, COLS)
    plot!(p4, WS, E_fe[ω]; lc = col, lw = 2, label = "finite elements, c/a = $ω")
    scatter!(
        p4, WS, E_nn[ω]; mc = col, msc = col, ms = 5, msw = 2, marker = :xcross,
        label = "surrogate, c/a = $ω"
    )
end

savefig(
    plot(
        p1, p2, p3, p4; layout = (2, 2),
        left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1250, 900),
        plot_title = "Layered spheroid, confocal prolate: closed form, Fourier cell, " *
            "surrogate  (E₁/E₀ = $R1, E₂/E₀ = $R2, ν = $NU, nradial = $(MESH.nradial))",
        plot_titlefontsize = 10,
    ),
    joinpath(ASSET, "layered_spheroid_accuracy.png")
)
println("\nwrote ", joinpath(ASSET, "layered_spheroid_accuracy.png"))

# ─── Report ──────────────────────────────────────────────────────────────────

mx(f, g) = maximum(maximum(pct(f[ω], g[ω])) for ω in OMEGAS)

open(joinpath(OUT, "layered_spheroid_comparison.md"), "w") do io
    println(io, "<!-- Generated by scripts/nn/make_layered_spheroid_figures.jl — do not edit by hand. -->\n")
    println(io, "Confocal prolate layers, `E₁/E₀ = $R1`, `E₂/E₀ = $R2`, `ν = $NU`,")
    println(io, "`nradial = $(MESH.nradial)`, `R/a = $(MESH.radius_ratio)`; the core fraction `w`")
    println(io, "swept over `[$(round(first(WS); digits = 2)), $(round(last(WS); digits = 2))]`")
    println(io, "at `c/a ∈ {1.25, 2, 3}`. The closed form is the reference where it exists.\n")
    println(io, "| Quantity | vs the closed form, worst over the sweep |")
    println(io, "| --- | ---: |")
    @printf(io, "| `(𝔸_εε)₁₁₁₁`, finite elements | %.2f %% |\n", mx(V_fe, V_an))
    @printf(io, "| `(𝔸_εε)₁₁₁₁`, surrogate | %.2f %% |\n", mx(V_nn, V_an))
    @printf(io, "| `∂(𝔸_εε)₁₁₁₁/∂w`, differenced cell | %.3f %% |\n", mx(D_fe, D_an))
    @printf(io, "| `∂(𝔸_εε)₁₁₁₁/∂w`, surrogate | %.1f %% |\n", mx(D_nn, D_an))
    println(io)
    println(io, "| Quantity with no closed form | surrogate vs the cell, worst |")
    println(io, "| --- | ---: |")
    @printf(
        io, "| `C₁₁₁₁` of a Mori-Tanaka estimate, `f = %.2f` | %.2f %% |\n",
        FRAC, mx(E_nn, E_fe)
    )
    println(io)
    println(io, "| Cost of one evaluation | |")
    println(io, "| --- | ---: |")
    @printf(io, "| finite elements, cold | %.3f s |\n", t_fe)
    @printf(io, "| surrogate | %.1f µs |\n", 1.0e6t_nn)
    @printf(io, "| **speed-up** | **%.0f×** |\n", t_fe / t_nn)
end
println("wrote ", joinpath(OUT, "layered_spheroid_comparison.md"))
@printf(
    "\nvalues: cell %.3f %%, surrogate %.3f %%\nderivative: cell %.4f %%, surrogate %.2f %%\n",
    mx(V_fe, V_an), mx(V_nn, V_an), mx(D_fe, D_an), mx(D_nn, D_an)
)
