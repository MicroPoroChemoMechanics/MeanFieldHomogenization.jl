# =============================================================================
#  55_ageing_creep_dirichlet_chains.jl
#
#  Ageing linear-viscoelastic (ALV) creep with a 2-term Dirichlet
#  (Kelvin-chain) relaxation law, of the type used for concrete in EDF's
#  Code_Aster (the Granger ageing-creep model, @granger1995), applied here
#  to the matrix-inclusion worked example of @barthelemyIJES2019.
#
#  Setup :
#    * iso ALV matrix : `(E, ν) = (1, 0.25)`, 2 Dirichlet chains
#      `(j₁,τ₁) = (2, 2)` and `(j₂,τ₂) = (3, 10)`, ageing prefactor
#      `exp(-(tp/30)²)` on the transient part.
#    * iso inhomogeneity : `(E, ν) = (10, 0.15)`, `(j₁,τ₁) = (0.5, 0.1)`,
#      `(j₂,τ₂) = (0.7, 7)`, ageing `exp(-(tp/15)²)`.
#    * spheroidal inclusions, ω ∈ {1, 0.1}, fractions φ ∈ {0.05, 0.1, 0.2}.
#    * schemes : Maxwell, Dilute, Mori-Tanaka.
#
#  Plot : two figures.
#    (1) `ω = 0.1`, several fractions.
#    (2) `φ = 0.2`, several aspect ratios.
#
#  Usage  : julia --project scripts/55_ageing_creep_dirichlet_chains.jl
#  Output : scripts/figures/55_ageing_creep_dirichlet_chains_{frac,omega}.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── Granger compliance kernels ───────────────────────────────────────────

kmu_Enu(E, ν) = (E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))

# Scalar Granger compliance: 1/k₀ + ageing(tp) · Σ jₗ (1 - exp(-(t-tp)/τₗ)).
function granger_scalar(k0, j_list, τ_list, ageing_f)
    return (t, tp) -> begin
        compl = 0.0
        for (j, τ) in zip(j_list, τ_list)
            compl += j * (1 - exp(-(t - tp) / τ))
        end
        return (1 / k0 + ageing_f(tp) * compl)
    end
end

# (Jk, Jg) creep compliance pair from (E, ν, j_list, τ_list, ageing).
function granger_Jk_Jg(E, ν, j_list, τ_list, ageing_f)
    k, g = kmu_Enu(E, ν)
    j_k = [1 / kmu_Enu(1 / j, ν)[1] for j in j_list]
    j_g = [1 / kmu_Enu(1 / j, ν)[2] for j in j_list]
    return granger_scalar(k, j_k, τ_list, ageing_f),
        granger_scalar(g, j_g, τ_list, ageing_f)
end

# ── Matrix law ──
const Jk_mat, Jg_mat = granger_Jk_Jg(
    1.0, 0.25, [2.0, 3.0], [2.0, 10.0],
    tp -> tp < 0 ? 1.0 : exp(-(tp / 30.0)^2)
)

# ── Inclusion law ──
const Jk_inc, Jg_inc = granger_Jk_Jg(
    10.0, 0.15, [0.5, 0.7], [0.1, 7.0],
    tp -> tp < 0 ? 1.0 : exp(-(tp / 15.0)^2)
)

# Iso projectors in Mandel form (constant 6×6 matrices).
const _, 𝕁₄, 𝕂₄ = TensND.iso_projectors(Val(3), Val(Float64))
const _J_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕁₄)
const _K_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕂₄)

# ALV laws (creep mode, 6×6 matrix-valued).
make_iso_law(Jk, Jg) = ViscoLaw(
    (t, tp) -> (Jk(t, tp) / 3) .* _J_M .+ (Jg(t, tp) / 2) .* _K_M,
    :creep,
)
const law_M = make_iso_law(Jk_mat, Jg_mat)
const law_I = make_iso_law(Jk_inc, Jg_inc)

# ─── homogenization helper ─────────────────────────────────────────────────

function build_rve(omega, frac)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M); fraction = :rest)
    sh = omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega)
    add_phase!(rve, :I, sh, Dict(:C => law_I); fraction = frac)
    return rve
end

# Shear (β = 2μ) component of an iso ALV (6n × 6n) relaxation matrix.
shear_param(R̃) = iso_params_from_blocks(R̃)[2]

# Shear creep curve: Lμ(t) = 2 μ⁻ᵛ(t) · 1 = sum of inverse shear over rows × 2.
function shear_creep_curve(R̃)
    β = shear_param(R̃)
    β_J = volterra_inverse(β; block_size = 1)
    return sum(β_J; dims = 2)[:] .* 2
end

# Per-phase shear creep (matrix or inclusion alone).
function phase_shear_curve(law_creep, T)
    R̃ = MeanFieldHomogenization.Viscoelasticity._trapezoidal_relaxation(law_creep, T, 6)
    return shear_creep_curve(R̃)
end

# ─── Plot 1 : effect of fraction (ω = 0.1) ─────────────────────────────────

const N_TIMES = 200
const t0_v = (0.0, 20.0, 40.0)

function build_grid(t0, n)
    return t0 .+ vcat(0.0, 10 .^ range(-8.0, log10(100.0 - t0); length = n))
end

const FRACTIONS = (0.05, 0.1, 0.2)
const FRAC_COLORS = (:blue, :black, :red)
const SCHEMES_LSTYLE = (
    (Maxwell(), :solid, "MAX"),
    (Dilute(), :dash, "DIL"),
    (MoriTanaka(), :dot, "MTB"),
)

# Figure 1: φ effect at ω = 0.1
plt1 = plot(
    layout = (1, 1), size = (1100, 700),
    title = "Granger ALV — shear creep, ω=0.1, varying φ",
    xlabel = "t", ylabel = "Lμ(t)",
    legend = :topleft
)

for t0 in t0_v
    T = build_grid(t0, N_TIMES)
    Tp = vcat(t0, T)

    Vmat = phase_shear_curve(law_M, T)
    Vinc = phase_shear_curve(law_I, T)
    plot!(
        plt1, Tp, vcat(0.0, Vmat); color = :green,
        label = (t0 == 0.0 ? "matrix" : "")
    )
    plot!(
        plt1, Tp, vcat(0.0, Vinc); color = :magenta,
        label = (t0 == 0.0 ? "inhomogeneity" : "")
    )

    for (sch, ls, sch_lbl) in SCHEMES_LSTYLE
        for (f, col) in zip(FRACTIONS, FRAC_COLORS)
            rve = build_rve(0.1, f)
            R̃ = homogenize_alv(rve, sch, :C; times = T)
            yvals = vcat(0.0, shear_creep_curve(R̃))
            plot!(
                plt1, Tp, yvals; color = col, linestyle = ls,
                label = (t0 == 0.0 ? "$sch_lbl φ=$f" : "")
            )
        end
    end
end

mkpath(joinpath(@__DIR__, "figures"))
savefig(
    plt1, joinpath(
        @__DIR__, "figures",
        "55_ageing_creep_dirichlet_chains_frac.png"
    )
)
display(plt1)

# Figure 2: ω effect at φ = 0.2
const OMEGAS_2 = (0.1, 1.0)
const OMEGA_COLORS = (:blue, :black)

plt2 = plot(
    layout = (1, 1), size = (1100, 700),
    title = "Granger ALV — shear creep, φ=0.2, varying ω",
    xlabel = "t", ylabel = "Lμ(t)",
    legend = :topleft
)

for t0 in t0_v
    T = build_grid(t0, N_TIMES)
    Tp = vcat(t0, T)

    Vmat = phase_shear_curve(law_M, T)
    Vinc = phase_shear_curve(law_I, T)
    plot!(
        plt2, Tp, vcat(0.0, Vmat); color = :green,
        label = (t0 == 0.0 ? "matrix" : "")
    )
    plot!(
        plt2, Tp, vcat(0.0, Vinc); color = :magenta,
        label = (t0 == 0.0 ? "inhomogeneity" : "")
    )

    for (sch, ls, sch_lbl) in SCHEMES_LSTYLE
        for (ω, col) in zip(OMEGAS_2, OMEGA_COLORS)
            rve = build_rve(ω, 0.2)
            R̃ = homogenize_alv(rve, sch, :C; times = T)
            yvals = vcat(0.0, shear_creep_curve(R̃))
            plot!(
                plt2, Tp, yvals; color = col, linestyle = ls,
                label = (t0 == 0.0 ? "$sch_lbl ω=$ω" : "")
            )
        end
    end
end

savefig(
    plt2, joinpath(
        @__DIR__, "figures",
        "55_ageing_creep_dirichlet_chains_omega.png"
    )
)
display(plt2)

println("Saved : 55_ageing_creep_dirichlet_chains_{frac,omega}.png")
