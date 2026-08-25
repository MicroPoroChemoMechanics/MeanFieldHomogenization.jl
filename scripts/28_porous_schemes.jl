# =============================================================================
#  28_porous_schemes.jl
#
#  Cross-validation of MeanFieldHomogenization against the canonical isotropic porous
#  benchmark : a single solid phase (k_s, μ_s) with spherical pores (k≈0,
#  μ≈0) varying in volume fraction φ ∈ [0, 1]. Every implemented scheme
#  is plotted side-by-side, mirroring the reference benchmark figure for
#  the same canonical porous problem.
#
#  Schemes covered : Voigt, Reuss, Dilute, DiluteDual, Mori-Tanaka,
#  Maxwell, PCW, Self-Consistent, Asymmetric Self-Consistent, Differential.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ── Material moduli ───────────────────────────────────────────────────────
const k_s, μ_s = 72.0, 32.0       # solid moduli
const k_p, μ_p = 1.0e-6, 1.0e-6   # pore moduli (numerical regularization)

const C_s = TensISO{3}(3 * k_s, 2 * μ_s)
const C_p = TensISO{3}(3 * k_p, 2 * μ_p)

# ── 2-phase porous RVE — solid matrix + pore inclusions.
#
# All schemes use the SOLID phase as the matrix. For the iterative
# SC/ASC schemes, the
# Picard iteration started from the solid bulk traces the matrix-stiff
# branch and crosses to the lower (percolating) branch through Picard
# noise around φ ≈ 0.5 for spheres ; the `select_best = true` mode
# (passed via `homogenize`) keeps the best iterate seen during the
# loop, matching the reference's behavior at the percolation
# threshold.
function build_rve(::Any, φ; ω_s = 1.0, ω_p = 1.0, sym_s = nothing, sym_p = nothing)
    rve = RVE(:SOLID)
    geom_s = Spheroid(ω_s)
    geom_p = Spheroid(ω_p)
    add_matrix!(rve, geom_s, Dict(:C => C_s); symmetrize = sym_s)
    add_phase!(
        rve, :PORE, geom_p, Dict(:C => C_p);
        fraction = φ, symmetrize = sym_p
    )
    return rve
end

# ── Extract iso (k, μ) from the homogenized stiffness ──────────────────────
#
# When the homogenized result is iso (the expected case for the porous
# benchmark), the stiffness is stored as a TensISO{4, 3} with two
# data scalars : `(α, β) = (3K, 2μ)`. The simpler accessor below maps
# directly to (k, μ).  A fallback for non-TensISO outputs keeps the
# script robust if a scheme returns a slightly anisotropic tensor due
# to numerical drift.
function extract_kμ(C::TensND.TensISO{4, 3})
    K, μ = k_mu(C)
    return max(K, 0.0), max(μ, 0.0)
end
function extract_kμ(C::TensND.AbstractTens)
    K, μ = k_mu(best_fit_iso(C))
    return max(K, 0.0), max(μ, 0.0)
end

# ── Schemes & display style ────────────────────────────────────────────────
const SCHEMES = [
    (Voigt(), "Voigt", :red, :dash),
    (Reuss(), "Reuss", :blue, :dash),
    (MoriTanaka(), "Mori-Tanaka", :black, :solid),
    (
        SelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true),
        "Self-Consistent", :red, :solid,
    ),
    (
        AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 300, select_best = true),
        "Asym. SC", :purple, :solid,
    ),
    (
        DifferentialScheme(; nsteps = 300),
        "Differential", :gold, :solid,
    ),
    (Dilute(), "Dilute", :green, :dash),
    (DiluteDual(), "DiluteDual", :green, :dot),
    (Maxwell(), "Maxwell", :blue, :solid),
    (PonteCastanedaWillis(), "PCW", :green, :solid),
]

# ── Sweep φ ∈ [0, 1] for every scheme ─────────────────────────────────────
const φs = collect(range(0.0, 1.0; length = 101))

p_k = plot(;
    xlabel = "φ (porosity)", ylabel = "k_hom (GPa)",
    xlims = (0, 1), ylims = (0, k_s + 5), legend = :topright,
    title = "Effective bulk modulus vs porosity"
)
p_μ = plot(;
    xlabel = "φ (porosity)", ylabel = "μ_hom (GPa)",
    xlims = (0, 1), ylims = (0, μ_s + 5), legend = false,
    title = "Effective shear modulus vs porosity"
)

function sweep!(
        p_k, p_μ, scheme, label, color, ls, φs;
        build_kw = (;)
    )
    ks_arr = Float64[]
    μs_arr = Float64[]
    for φ in φs
        try
            rve = build_rve(scheme, φ; build_kw...)
            C = homogenize(rve, scheme, :C)
            K, μ = extract_kμ(C)
            push!(ks_arr, K)
            push!(μs_arr, μ)
        catch
            push!(ks_arr, NaN)
            push!(μs_arr, NaN)
        end
    end
    plot!(p_k, φs, ks_arr, lw = 2, color = color, linestyle = ls, label = label)
    return plot!(p_μ, φs, μs_arr, lw = 2, color = color, linestyle = ls)
end

# Sphere case (default)
for (scheme, label, color, ls) in SCHEMES
    sweep!(p_k, p_μ, scheme, label, color, ls, φs)
end

p_full = plot(
    p_k, p_μ; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600),
    plot_title = "Porous benchmark (spheres) — MeanFieldHomogenization v0.4"
)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "28_porous_schemes.png")
savefig(p_full, figpath)
display(p_full)
@printf "\nSaved : %s\n" figpath

# ── Non-spherical case : oblate spheroids with iso symmetrize on every phase ─
#
# Both solid and pore phases are oblate spheroids with aspect ratio
# `ω_oblate` (= c/a, < 1 for oblate). The `symmetrize = :iso` declaration
# on each phase tells the homogenization kernel to project the
# localization tensor onto its isotropic part, which mirrors a uniform
# spatial distribution of orientations. The macroscopic effective tensor
# is therefore isotropic.

const ω_oblate = 0.2          # c/a aspect ratio (matches `spheroidal(omega)`)

p_k2 = plot(;
    xlabel = "φ (porosity)", ylabel = "k_hom (GPa)",
    xlims = (0, 1), ylims = (0, k_s + 5), legend = :topright,
    title = "Oblate spheroidal phases (ω=$(ω_oblate)), iso-symmetrize"
)
p_μ2 = plot(;
    xlabel = "φ (porosity)", ylabel = "μ_hom (GPa)",
    xlims = (0, 1), ylims = (0, μ_s + 5), legend = false,
    title = "Effective shear modulus"
)

for (scheme, label, color, ls) in SCHEMES
    sweep!(
        p_k2, p_μ2, scheme, label, color, ls, φs;
        build_kw = (;
            ω_s = ω_oblate, ω_p = ω_oblate,
            sym_s = :iso, sym_p = :iso,
        )
    )
end

p_full2 = plot(
    p_k2, p_μ2; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1400, 600),
    plot_title = "Porous benchmark (oblate + iso-symmetrize) — MeanFieldHomogenization v0.4"
)
figpath2 = joinpath(figdir, "28_porous_schemes_oblate.png")
savefig(p_full2, figpath2)
display(p_full2)
@printf "Saved : %s\n" figpath2

# ── Tabular output at selected φ for visual validation ────────────────────
println("\n[Tabular] k_hom and μ_hom at φ = 0.0, 0.1, 0.3, 0.5, 0.7, 0.9")
@printf "  %8s" "scheme"
for φ in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
    @printf "   φ=%-4.2f      " φ
end
println()
println("  " * "─"^95)
for (scheme, label, _, _) in SCHEMES
    @printf "  %-12s" label
    for φ in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9)
        try
            C = homogenize(build_rve(scheme, φ), scheme, :C)
            K, μ = extract_kμ(C)
            @printf "  %5.2f/%-5.2f" K μ
        catch
            @printf "  %5s/%-5s" "NaN" "NaN"
        end
    end
    println()
end

println("\nDone.")
