# =============================================================================
#  56_ageing_creep_order2.jl
#
#  Julia reproduction of the order-2 Maxwell ageing-creep benchmark.
#
#  Order-2 (vector-tensor) ageing linear viscoelasticity:
#    * iso ALV matrix with Dirichlet 2-element chain + ageing prefactor
#    * iso inclusion with similar Dirichlet chain + ageing prefactor
#    * spherical (ω = 1) and prolate spheroidal (ω = 0.1) inclusions
#    * fraction φ = 0.2
#    * schemes : Mori-Tanaka, Dilute, Maxwell
#
#  In Julia, `homogenize_alv(rve, scheme, prop; times)` automatically
#  routes to the order-2 pipeline when the matrix law samples to a
#  3×3 matrix (or `TensND.AbstractTens{2,3}`), and inverts the
#  trapezoidal compliance matrix to the relaxation form when the law
#  mode is `:creep` — same convention as ECHOES `homogenize_visco`.
#
#  Usage  : julia --project scripts/56_ageing_creep_order2.jl
#  Output : scripts/figures/56_ageing_creep_order2.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using LinearAlgebra
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── Dirichlet creep kernels (same parameters as the Python script) ────────

function build_R(r0, r1, r2, τ1, τ2, fag, finst)
    return (t, tp) ->
    finst(tp) * r0 +
        fag(tp) * (
        r1 * (1 - exp(-(t - tp) / τ1)) +
            r2 * (1 - exp(-(t - tp) / τ2))
    )
end

const Rs = build_R(
    1.0, 2.0, 3.0, 2.0, 10.0,
    t -> exp(-(t / 30.0)^2), _ -> 1.0
)
const Ri = build_R(
    0.2, 0.3, 1.2, 1.0, 15.0,
    t -> exp(-(t / 15.0)^2),
    t -> exp(-(t / 30.0)^2)
)

# Wrap as 3×3-matrix-valued ALV laws (creep mode, like ECHOES).
make_law(scalar) = ViscoLaw(
    (t, tp) -> scalar(t, tp) * Matrix{Float64}(I, 3, 3),
    :creep
)
const law_M = make_law(Rs)
const law_I = make_law(Ri)

# ─── Helpers ────────────────────────────────────────────────────────────────

# Build the RVE for a given inclusion shape.
function build_rve(omega, frac)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:Y => law_M))
    sh = omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega)
    add_phase!(rve, :I, sh, Dict(:Y => law_I); fraction = frac)
    return rve
end

# Iso scalar from a (3n)×(3n) order-2 ALV block matrix.
iso_scalar(M) = iso_order2_params_from_blocks(M)

# Resistance curve under unit step intensity:
#   from a relaxation matrix R̃ (3n × 3n iso), extract the scalar α n×n,
#   invert to get the creep scalar 1/α, and sum over rows = response.
function resistance_curve(R̃)
    α = iso_scalar(R̃)
    α_J = volterra_inverse(α; block_size = 1)
    return sum(α_J; dims = 2)[:]
end

# Per-phase response (no homogenization).
function phase_response(law_creep, T)
    R̃ = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(law_creep, T, 3)
    return resistance_curve(R̃)
end

# ─── Plot ──────────────────────────────────────────────────────────────────

const N_TIMES = 200
const omega_v = (1.0, 0.1)
const t0_v = (0.0, 20.0, 40.0)
const f = 0.2

function build_grid(t0, n)
    return t0 .+ vcat(0.0, 10 .^ range(-8.0, log10(100.0 - t0); length = n))
end

plt = plot(
    layout = (1, 1), size = (1100, 700),
    xlabel = "t", ylabel = "R(t)",
    title = "Order-2 ALV — ageing creep (φ=0.2)",
    legend = :topleft
)

for t0 in t0_v
    T = build_grid(t0, N_TIMES)
    Tplot = vcat(t0, T)

    Vmat = phase_response(law_M, T)
    Vinc = phase_response(law_I, T)
    plot!(
        plt, Tplot, vcat(0.0, Vmat); color = :green,
        label = (t0 == 0.0 ? "matrix" : "")
    )
    plot!(
        plt, Tplot, vcat(0.0, Vinc); color = :magenta,
        label = (t0 == 0.0 ? "inhomogeneity" : "")
    )

    for omega in omega_v
        col = omega == 0.1 ? :blue : :black
        for (sch, lstyle, sch_lbl) in (
                (Maxwell(), :solid, "MAX"),
                (Dilute(), :dash, "DIL"),
                (MoriTanaka(), :dot, "MT"),
            )
            rve = build_rve(omega, f)
            R̃ = homogenize_alv(rve, sch, :Y; times = T)
            yvals = vcat(0.0, resistance_curve(R̃))
            plot!(
                plt, Tplot, yvals; color = col, linestyle = lstyle,
                label = (t0 == 0.0 ? "$sch_lbl ω=$omega" : "")
            )
        end
    end
end

xlims!(plt, (0.0, 130.0))

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(
    @__DIR__, "figures",
    "56_ageing_creep_order2.png"
)
savefig(plt, out)
display(plt)
println("Saved : $out")
