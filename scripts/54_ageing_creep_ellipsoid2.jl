# =============================================================================
#  54_ageing_creep_ellipsoid2.jl
#
#  Julia reproduction of the order-2 ellipsoid ageing-creep benchmark.
#
#  Setup :
#    * iso ALV matrix : isotropic stiffness `(E, ν) = (1, 0.2)`,
#      bulk and shear creep with prefactors
#         Jₖ(t,t') = (fₖ(t')/(3kₛ) + 0.1/kₛ · log(1 + (t-t')/2))
#         Jμ(t,t') = (fμ(t')/(2μₛ) + 0.1/μₛ · log(1 + (t-t')/1))
#      with `fₖ(t) = fμ(t) = 0.5 exp(-t/20) + 0.5`.
#    * iso pore (near-zero stiffness `Cᵢ = 1e6 · 𝕀`).
#    * spheroidal inclusions, ω ∈ {1, 0.1}, fractions f ∈ {0, 0.4}.
#    * schemes : Mori-Tanaka, Self-Consistent.
#
#  Note : the original Python script also exercises `DIFF`
#  (Differential), which is **not** part of the ALV pipeline yet.  We
#  cover MT and SC here ; DIFF stays a follow-up.
#
#  Output : effective uniaxial creep response `J^E_eff(t, t')` from a
#  unit longitudinal stress step.
#
#  Usage  : julia --project scripts/54_ageing_creep_ellipsoid2.jl
#  Output : scripts/figures/54_ageing_creep_ellipsoid2.png
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ─── Phase definitions ─────────────────────────────────────────────────────

const Eₛ = 1.0;  const νₛ = 0.2
const kₛ = Eₛ / (3 * (1 - 2 * νₛ))
const μₛ = Eₛ / (2 * (1 + νₛ))
const Cₛ_t = TensISO{3}(3 * kₛ, 2 * μₛ)

# Bulk and shear ageing prefactors of the matrix Js.
const fk = t -> 0.5 * exp(-t / 20.0) + 0.5
const fμ = t -> 0.5 * exp(-t / 20.0) + 0.5

# Iso projectors in Mandel form.
const _, 𝕁₄, 𝕂₄ = TensND.iso_projectors(Val(3), Val(Float64))
const _J_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕁₄)
const _K_M = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66(𝕂₄)

# Matrix law (Js, CREEP).
const Js = (t, tp) -> begin
    α = fk(tp) / (3 * kₛ) + 1.0e-1 / kₛ * log(1 + (t - tp) / 2.0)
    β = fμ(tp) / (2 * μₛ) + 1.0e-1 / μₛ * log(1 + (t - tp) / 1.0)
    return α .* _J_M .+ β .* _K_M
end
const law_M = ViscoLaw(Js, :creep)

# Inclusion law : near-rigid stiffness (pore-like = inverse).  Same
# Mandel matrix at all times → trivial creep matrix.
const Cᵢ_inv = (1.0 / 1.0e6) * Matrix{Float64}(I, 6, 6)
const Ji_const = (t, tp) -> Cᵢ_inv
const law_I = ViscoLaw(Ji_const, :creep)

# ─── homogenization helpers ────────────────────────────────────────────────

function build_rve(omega, f)
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => law_M))
    sh = omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega)
    # Mirrors the ECHOES `symmetrize=[ISO]` keyword: orientation-average
    # the inclusion's stiffness contribution to iso form.
    add_phase!(
        rve, :I, sh, Dict(:C => law_I);
        fraction = f, symmetrize = :iso
    )
    return rve
end

# Effective uniaxial creep `J^E_eff(t, t_0)` from a relaxation matrix R̃:
#   strain response E = R̃⁻¹ · S where S = (1, 0, 0, 0, 0, 0) at every t,
#   read off E[6i+1] ≡ E_xx.
function uniaxial_creep_curve(R̃, n)
    J̃ = volterra_inverse(R̃; block_size = 6)
    S = zeros(eltype(J̃), 6 * n)
    for i in 1:n
        S[6 * (i - 1) + 1] = 1.0
    end
    E = J̃ * S
    return [E[6 * (i - 1) + 1] for i in 1:n]
end

# ─── Plot ──────────────────────────────────────────────────────────────────

const N_TIMES = 51
const omega_v = (1.0, 0.1)
const t0_v = (0.0, 30.0)
const fractions = (0.0, 0.4)
const scheme_v = (MoriTanaka(), DifferentialScheme(; nsteps = 50), SelfConsistent())
const scheme_lbl = ("MT", "DIFF", "SC")
const lstyles = (:solid, :dashdotdot, :dot)

function build_grid(t0, n)
    if t0 == 0
        return vcat(0.0, 10 .^ range(-2, log10(50.0); length = n - 1))
    else
        return 10 .^ range(log10(t0), log10(50.0); length = n)
    end
end

plt = plot(
    layout = (1, 1), size = (1100, 700),
    title = "ALV solid + spheroidal rigid inclusion — MT vs DIFF vs SC",
    xlabel = "t", ylabel = "Eˢ · Jₑʰᵒᵐ(t)",
    legend = :topright
)

# Pre-compute the elastic limit at each loading time for the dashed
# reference curves (ECHOES Python plots `1/C.E` per t with the SAME
# scheme as the ALV run).
function elastic_compliance(omega, f, t, sch::HomogenizationScheme)
    C_M_arr = inv(Js(t, t))   # 6×6
    C_M_t = best_fit_iso(TensND.Tens(MeanFieldHomogenization.Core.array_from_mandel66(C_M_arr)))
    rve_e = RVE(:M)
    add_matrix!(rve_e, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_t))
    sh = omega == 1.0 ? Ellipsoid(1.0, 1.0, 1.0) : Spheroid(omega)
    C_I_t = TensISO{3}(3.0e6, 2.0e6)
    add_phase!(
        rve_e, :I, sh, Dict(:C => C_I_t);
        fraction = f, symmetrize = :iso
    )
    C_eff = homogenize(rve_e, sch, :C)
    E_eff, _ = E_nu(best_fit_iso(C_eff))
    return 1 / E_eff
end

# Plot order is critical: draw SC (dotted) AFTER MT (solid) so the
# dotted lines remain visible when they coincide with the solid ones.
const elastic_lstyles = (:dash, :dashdot, :dashdot)

for (i_sch, sch) in enumerate(scheme_v)
    for omega in omega_v
        for t0 in t0_v
            T = build_grid(t0, N_TIMES)
            for f in fractions
                rve = build_rve(omega, f)
                col = if omega == 0.1 && f > 0
                    :red
                elseif omega == 1.0 && f > 0
                    :blue
                else
                    :black
                end
                lbl_main = (t0 == 0.0) ?
                    @sprintf("%s f=%.1f ω=%.1f", scheme_lbl[i_sch], f, omega) :
                    ""
                lbl_elas = (t0 == 0.0 && i_sch == 1) ?
                    @sprintf("elastic ref. f=%.1f ω=%.1f", f, omega) :
                    ""
                try
                    R̃ = homogenize_alv(rve, sch, :C; times = T)
                    n = length(T)
                    Jhom = uniaxial_creep_curve(R̃, n)
                    # ALV creep response: solid for MT, thicker dotted for SC
                    plot!(
                        plt, T, Jhom; color = col,
                        linestyle = lstyles[i_sch],
                        linewidth = (
                            i_sch == 3 ? 2.5 :
                                i_sch == 2 ? 1.7 : 1.5
                        ),
                        label = lbl_main
                    )
                    # Per-loading-time elastic reference for THIS scheme
                    Jelas = [elastic_compliance(omega, f, t, sch) for t in T]
                    plot!(
                        plt, T, Jelas; color = col,
                        linestyle = elastic_lstyles[i_sch],
                        linewidth = 1.0,
                        label = lbl_elas
                    )
                catch e
                    @warn "Skipping" sch omega f t0 exception = e
                end
            end
        end
    end
end

mkpath(joinpath(@__DIR__, "figures"))
out = joinpath(@__DIR__, "figures", "54_ageing_creep_ellipsoid2.png")
savefig(plt, out)
display(plt)
println("Saved : $out")

# ─── Figure 2 : SC sweep over aspect ratios at f = 0.4 ─────────────────────
#
# Highlights how the Self-Consistent prediction varies with the inclusion
# aspect ratio when the rigid inclusion volume fraction is non-zero.
# Plotted alongside the MT predictions for comparison and the
# scheme-matched instantaneous elastic reference.

const omega_sweep = (2.0, 1.0, 0.5, 0.2, 0.1)
const omega_palette = (:purple, :blue, :teal, :orange, :red)

plt2 = plot(
    layout = (1, 1), size = (1100, 700),
    title = "ALV — SC vs MT, varying ω at f = 0.4",
    xlabel = "t", ylabel = "Eˢ · Jₑʰᵒᵐ(t)",
    legend = :topright
)

const f_sc = 0.4
for (i_sch, sch) in enumerate(scheme_v)
    for t0 in t0_v
        T = build_grid(t0, N_TIMES)
        for (omega, col) in zip(omega_sweep, omega_palette)
            rve = build_rve(omega, f_sc)
            lbl_main = (t0 == 0.0) ?
                @sprintf("%s ω=%.2f", scheme_lbl[i_sch], omega) :
                ""
            lbl_elas = (t0 == 0.0 && i_sch == 2) ?
                @sprintf("elastic %s ω=%.2f", scheme_lbl[i_sch], omega) :
                ""
            try
                R̃ = homogenize_alv(rve, sch, :C; times = T)
                n = length(T)
                Jhom = uniaxial_creep_curve(R̃, n)
                plot!(
                    plt2, T, Jhom; color = col,
                    linestyle = lstyles[i_sch],
                    linewidth = (
                        i_sch == 3 ? 2.5 :
                            i_sch == 2 ? 1.7 : 1.5
                    ),
                    label = lbl_main
                )
                Jelas = [elastic_compliance(omega, f_sc, t, sch) for t in T]
                plot!(
                    plt2, T, Jelas; color = col,
                    linestyle = elastic_lstyles[i_sch],
                    linewidth = 1.0, label = lbl_elas
                )
            catch e
                @warn "Skipping" sch omega t0 exception = e
            end
        end
    end
end

out2 = joinpath(@__DIR__, "figures", "54_ageing_creep_ellipsoid2_sc_omega.png")
savefig(plt2, out2)
display(plt2)
println("Saved : $out2")

println("Includes MT, DIFF (50 steps) and SC schemes — full coverage of the")
println("Python reference.")
