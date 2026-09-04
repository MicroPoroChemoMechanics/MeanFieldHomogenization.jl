# =============================================================================
#  57_ageing_creep_cracks.jl
#
#  Julia reproduction of the ageing-creep penny-crack benchmark —
#  pure penny crack (no interface stiffness) in an iso ALV matrix,
#  using all the crack-aware homogenization schemes available in
#  `MeanFieldHomogenization`:  Dilute, Mori-Tanaka, Maxwell, Self-Consistent,
#  Asymmetric Self-Consistent, Ponte-Castañeda Willis, Differential.
#
#  The Python benchmark uses cracks **with interface stiffness**
#  `(Rn(t,t'), Rt(t,t'))` and schemes MT / SC / PCW.  This first
#  Julia version covers the **pure traction-free penny limit** —
#  the interface-stiffness extension is scheduled for v0.6.2.
#
#  Setup
#    * iso ALV matrix R(t,t') = C∞ + (C₀(1+0.2 √t') − C∞) exp(-(t-t')/τ)
#      with `(k₀, μ₀) = (5, 2)`, `(k_∞, μ_∞) = (3, 1)`, `τ = 1`.
#    * penny cracks (η = 1, normal `e_3`), density `d = 0.7`.
#    * loading times `t₀ ∈ {0, 10, 20}`, time grid `t = t₀ +
#      logspace(-2, log₁₀(50−t₀), 50)`.
#
#  Output : effective uniaxial creep response `Eₓₓ(t)` from the
#  homogenized relaxation matrix R̃, as the strain field of a unit
#  longitudinal stress step (cf. the Python `linalg.inv(V).dot(S)`).
#
#  Usage  : julia --project scripts/57_ageing_creep_cracks.jl
#  Output : scripts/figures/57_ageing_creep_cracks.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── Matrix law (Maxwell-like ageing relaxation) ───────────────────────────

const k₀ = 5.0;     const μ₀ = 2.0
const k_inf = 3.0;  const μ_inf = 1.0
const τ = 1.0

const C_inf_t = TensISO{3}(3 * k_inf, 2 * μ_inf)

const _, 𝕁₄, 𝕂₄ = TensND.iso_projectors(Val(3), Val(Float64))
const _J_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕁₄)
const _K_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕂₄)

# Iso relaxation kernel of the matrix.
function R_M(t, tp)
    factor = exp(-(t - tp) / τ)
    α0 = 3 * k₀ * (1 + 0.2 * sqrt(max(tp, 0.0)))
    β0 = 2 * μ₀ * (1 + 0.2 * sqrt(max(tp, 0.0)))
    α_inf = 3 * k_inf;  β_inf = 2 * μ_inf
    α = α_inf + (α0 - α_inf) * factor
    β = β_inf + (β0 - β_inf) * factor
    return α .* (_J_M ./ 3) .+ β .* (_K_M ./ 2)
end
const law_M = ViscoLaw(R_M, :relaxation)

# ─── RVE construction ──────────────────────────────────────────────────────

function build_rve(d)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M); fraction = :rest)
    add_phase!(
        rve, :CRACK, PennyCrack(1.0), Dict(:C => law_M);
        density = d, symmetrize = :iso
    )
    return rve
end

# ─── Effective uniaxial response ───────────────────────────────────────────

function uniaxial_response(R̃, n)
    J̃ = volterra_inverse(R̃; block_size = 6)
    S = zeros(eltype(J̃), 6 * n)
    @inbounds for i in 1:n
        S[6 * (i - 1) + 1] = 1.0
    end
    E = J̃ * S
    return [E[6 * (i - 1) + 1] for i in 1:n]
end

# ─── Plot ──────────────────────────────────────────────────────────────────

const N_TIMES = 50
const dens_v = (0.0, 0.05, 0.1)   # physically meaningful pure-crack densities
# (Bristow-O'Connell percolation ≈ 0.18)
const t0_v = (0.0, 10.0, 20.0)

function build_grid(t0, n)
    return t0 .+ vcat(0.0, 10 .^ range(-2.0, log10(50.0 - t0); length = n))
end

const SCHEMES = (
    (Dilute(), "Dilute", :gray, :dot),
    (MoriTanaka(), "MT", :blue, :solid),
    (Maxwell(), "MAX", :purple, :dash),
    (PonteCastanedaWillis(), "PCW", :green, :dashdot),
    (SelfConsistent(), "SC", :red, :solid),
    (AsymmetricSelfConsistent(), "ASC", :orange, :dot),
    (DifferentialScheme(; nsteps = 50), "DIFF", :teal, :dashdotdot),
)

plt = plot(
    layout = (1, 1), size = (1200, 800),
    title = "ALV penny cracks — all crack-aware schemes (d ∈ {0, 0.05, 0.10})",
    xlabel = "t", ylabel = "Eₓₓ(t)",
    legend = :topleft
)

for t0 in t0_v
    T = build_grid(t0, N_TIMES)
    n = length(T)
    for d in dens_v
        rve = build_rve(d)
        for (sch, lbl, col, ls) in SCHEMES
            try
                R̃ = homogenize_alv(rve, sch, :C; times = T)
                E = uniaxial_response(R̃, n)
                full_lbl = (t0 == 0.0) ? "$lbl d=$d" : ""
                plot!(
                    plt, T, E; color = col, linestyle = ls,
                    linewidth = 1.5, label = full_lbl
                )
            catch e
                @warn "Skipping" sch d t0 exception = e
            end
        end
    end
end

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "57_ageing_creep_cracks.png")
savefig(plt, out)
display(plt)

println("Saved : $out")
println()
println("Crack-aware ALV schemes covered :")
println("  Dilute     → C̃_eff = C̃_M + ΔC̃_cracks (additive)")
println("  MT, Maxwell, PCW → cracks added to numerator (zero volume in den)")
println("  SC, ASC    → cracks iterated against running estimate (Bristow-O'Connell)")
println("  Differential → cracks contribute via dilute correction at each step")
println()
println("Note : interface-stiffness ALV cracks (`Rn`, `Rt`) are scheduled for v0.6.2.")
