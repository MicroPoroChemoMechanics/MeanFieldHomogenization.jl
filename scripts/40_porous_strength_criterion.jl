# =============================================================================
#  40_porous_strength_criterion.jl
#
#  Strength criterion ellipses for an isotropic porous medium under macroscopic
#  hydrostatic + deviatoric loading. Reproduces a simple porous-criterion
#  verification benchmark.
#
#  Setup: oblate spheroid solid (ω = 0.1, K_s = 1e6, μ_s = 1) at fraction 1-φ
#         + oblate spheroid pore (ω = 0.1, C_pore ≈ 0) at fraction φ. Both
#         phases iso-symmetrized, so the homogenized stiffness is isotropic
#         (in the limit of infinitely many random orientations).
#
#  For each scheme we plot the strength ellipse
#       (Σ_m / σ_o)² / (2A)  +  (Σ_d / σ_o)² / B = 1 / (1 - φ)
#  where
#       A = (μ_s / k_hom)² · (∂k_hom / ∂μ_s)
#       B = (μ_s / μ_hom)² · (∂μ_hom / ∂μ_s)
#  computed via ForwardDiff on `homogenize(... :C)` and the iso projection of
#  the resulting effective stiffness.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using ForwardDiff
using TensND
using Printf
using Plots
using LinearAlgebra

const ks_value, μs_value = 1.0e6, 1.0
const TINY = 1.0e-12
const ω_aspect = 0.1
const φ_value = 0.15

# Build an iso 2-vector (α=3K, β=2μ) of the homogenized stiffness, parametrized
# by the SOLID shear modulus `μs` (so that ForwardDiff can differentiate).
function _C_hom_iso_2vec(μs::Real, ks, φ, scheme)
    T = typeof(μs)
    rve = RVE(; T = T, distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :SOLID, Spheroid(ω_aspect), Dict(:C => TensISO{3}(convert(T, 3 * ks), 2 * μs)); fraction = :rest, symmetrize = :iso)
    add_phase!(
        rve, :PORE, Spheroid(ω_aspect),
        Dict(:C => TensISO{3}(convert(T, 3 * TINY), convert(T, 2 * TINY)));
        fraction = convert(T, φ), symmetrize = :iso
    )
    C = homogenize(rve, scheme, :C)
    C_iso = MeanFieldHomogenization.Schemes._apply_symmetrize(
        C, MeanFieldHomogenization.Schemes.IsoSymmetrize()
    )
    α, β = TensND.get_data(C_iso)
    return [α, β]
end

# Strength-ellipse semi-axes from the ECHOES reference formula.
function ellipse_radii(ks, μs, φ, scheme; sch_name = "")
    # Modulus chain at the value point.
    Cα0 = _C_hom_iso_2vec(μs, ks, φ, scheme)
    α_hom, β_hom = Cα0[1], Cα0[2]
    K_hom = α_hom / 3
    μ_hom = β_hom / 2

    # Derivative of (3K_hom, 2μ_hom) w.r.t. μ_s, then convert to derivative
    # w.r.t. 2μ_s (matches the C++ reference's `index=1` parameter convention,
    # `stiff_kmu(K,μ) = (3K, 2μ)`).
    dCp_dμs = ForwardDiff.derivative(μ -> _C_hom_iso_2vec(μ, ks, φ, scheme), μs)
    dCp_d2μs = dCp_dμs ./ 2  # divide by 2 because d(2μ_s)/dμ_s = 2

    dK_dμs = dCp_d2μs[1] * 2.0 / 3.0   # d(3K)/d(2μ) = dCp[1] ⇒ dK/dμ_s = dCp[1]·(2/3)
    dμ_dμs = dCp_d2μs[2]                # d(2μ_hom)/d(2μ_s) = dCp[2] = dμ_hom/dμ_s

    A = (μs / K_hom)^2 * dK_dμs
    B = (μs / μ_hom)^2 * dμ_dμs

    @printf "  %-5s K=%.5g μ=%.5g  dK/dμs=%.5g dμ/dμs=%.5g  A=%.5g B=%.5g\n" sch_name K_hom μ_hom dK_dμs dμ_dμs A B

    a = sqrt((1 - φ) / (2 * A))
    b = sqrt((1 - φ) / B)
    return a, b
end

# ── Schemes covered (mirrors the Python script) ──────────────────────────────
const SCHEMES = [
    (MoriTanaka(), "MT"),
    (DiluteDual(), "DILD"),
    (SelfConsistent(; abstol = 1.0e-8, maxiters = 300, select_best = true), "SC"),
    (
        AsymmetricSelfConsistent(; abstol = 1.0e-8, maxiters = 300, select_best = true),
        "ASC",
    ),
    (PonteCastanedaWillis(), "PCW"),
    (Maxwell(), "MAX"),
]

# ── Compute and plot ─────────────────────────────────────────────────────────
println("Strength-ellipse criterion")
println("ks = $ks_value, μs = $μs_value, ω = $ω_aspect, φ = $φ_value")
println("─"^78)

p = plot(;
    xlabel = "Σ_m / σ_o", ylabel = "Σ_d / σ_o",
    aspect_ratio = :equal, legend = :outerright, grid = true
)

ltheta = range(0, π; length = 100)
for (sch, name) in SCHEMES
    a, b = ellipse_radii(ks_value, μs_value, φ_value, sch; sch_name = name)
    @printf "  %-5s a=%.5g b=%.5g\n" name a b
    xs = a .* cos.(ltheta)
    ys = b .* sin.(ltheta)
    plot!(p, xs, ys, lw = 2, label = "$name; φ=$φ_value")
end

hline!(p, [0.0]; color = :black, lw = 0.5, label = "")
vline!(p, [0.0]; color = :black, lw = 0.5, label = "")

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "40_porous_strength_criterion.png")
savefig(p, figpath)
display(p)
@printf "\nSaved : %s\n" figpath
