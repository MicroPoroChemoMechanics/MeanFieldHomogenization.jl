# =============================================================================
#  42_cementpaste_iso.jl
#
#  Simplified ELASTICITY-ONLY variant of the Pichler & Hellmich (2011)
#  cement-paste / mortar model.
#
#  Differences from the full strength model (`41_multiscale_strength.jl`) :
#    * the hydrate needle is symmetrized FULLY ISOTROPIC (`:iso`), i.e. a
#      single random-orientation SC pass — no θ-discretization, no TI families,
#      no strength criterion ;
#    * the needle aspect ratio is ω = 100 (NOT 1e4 — this is the value the
#      companion Python script uses for the `E(w/c)` comparison plot) ;
#    * `αmax` is pulled back by a factor (1 − 1e-3) from the singular boundary,
#      as in the Python original.
#
#  Water and air keep the exact-zero echoes stiffness here (this variant is
#  robust without the percolating-branch TINY regularization because the ISO
#  hydrate foam does not sit exactly at the SC percolation threshold).
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ── Constants (shared with the full model) ─────────────────────────────────
const ρ_w = 1.0
const ρ_clin = 3.15;  const d_clin = ρ_clin / ρ_w
const ρ_hyd = 2.073; const d_hyd = ρ_hyd / ρ_w
const ρ_san = 2.648; const d_san = ρ_san / ρ_w

const K_clin, μ_clin = 116.7, 53.8
const K_hyd, μ_hyd = 18.7, 11.8
const K_san, μ_san = 37.8, 44.3
const TINY = 1.0e-3            # water / air ≈ zero stiffness
const ω_iso = 100.0            # iso-variant aspect ratio (Python uses 100)

f_clin(wc, α) = (1 - α) / (1 + d_clin * wc)
f_w(wc, α) = d_clin * (wc - 0.42 * α) / (1 + d_clin * wc)
f_hyd(wc, α) = 1.42 * d_clin / d_hyd * α / (1 + d_clin * wc)
fh_san(wc, sc) = sc / d_san / (1 / d_clin + wc + sc / d_san)
# (1 − 1e-3) pull-back from the singular boundary — matches the Python iso script
αmax_iso(wc) = (1 - 1.0e-3) * min(1.0, wc / 0.42)

function C_paste_iso(wc, α; sc = 0.0, ω = ω_iso)
    fclin = f_clin(wc, α)
    fw = f_w(wc, α); fw < 0 && return nothing
    fhyd = f_hyd(wc, α)
    fair = max(0.0, 1 - fclin - fw - fhyd)
    fthyd = fhyd / (1 - fclin)
    ftw = fw / (1 - fclin)
    ftair = fair / (1 - fclin)
    fhsan = fh_san(wc, sc)

    C_tiny() = TensISO{3}(3 * TINY, 2 * TINY)

    # Hydrate foam : ISO-symmetrized needle + water + air, self-consistent.
    rve_hf = RVE(:HYD)
    add_matrix!(rve_hf, Spheroid(ω), Dict(:C => TensISO{3}(3 * K_hyd, 2 * μ_hyd)); symmetrize = :iso)
    add_phase!(rve_hf, :HYDi, Spheroid(ω), Dict(:C => TensISO{3}(3 * K_hyd, 2 * μ_hyd)); fraction = fthyd, symmetrize = :iso)
    add_phase!(rve_hf, :W, Ellipsoid(1.0), Dict(:C => C_tiny()); fraction = ftw, symmetrize = :iso)
    add_phase!(rve_hf, :AIR, Ellipsoid(1.0), Dict(:C => C_tiny()); fraction = ftair, symmetrize = :iso)
    C_hf = homogenize(
        rve_hf, SelfConsistent(; abstol = 1.0e-8, maxiters = 1000, damping = 0.5),
        :C; select_best = true
    )

    # Cement paste : MT(HF, clinker).
    rve_cp = RVE(:HF)
    add_matrix!(rve_cp, Ellipsoid(1.0), Dict(:C => C_hf))
    add_phase!(rve_cp, :CLIN, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * K_clin, 2 * μ_clin)); fraction = fclin)
    C_cp = homogenize(rve_cp, MoriTanaka(), :C)

    # Mortar : MT(CP, sand).
    rve_mo = RVE(:CP)
    add_matrix!(rve_mo, Ellipsoid(1.0), Dict(:C => C_cp))
    add_phase!(rve_mo, :SAN, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * K_san, 2 * μ_san)); fraction = fhsan)
    return homogenize(rve_mo, MoriTanaka(), :C)
end

kμ(C) = k_mu(best_fit_iso(C))

println("="^70)
println("Cement-paste / mortar elasticity — ISO Pichler variant (ω = $ω_iso)")
println("="^70)

const wcs = [0.157, 0.25, 0.35, 0.5, 0.65, 0.8]
p1 = plot(; xlabel = "α", ylabel = "k (GPa)", legend = :topleft)
p2 = plot(; xlabel = "α", ylabel = "μ (GPa)", legend = false)

for wc in wcs
    αs = collect(range(αmax_iso(wc), 0.005; length = 20))
    x = Float64[]; yk = Float64[]; yμ = Float64[]
    for α in αs
        C = try
            C_paste_iso(wc, α)
        catch
            nothing
        end
        C === nothing && continue
        K, μ = kμ(C)
        push!(x, α); push!(yk, K); push!(yμ, μ)
    end
    isempty(x) && continue
    plot!(p1, x, yk, marker = :+, lw = 1.5, label = "wc = $wc")
    plot!(p2, x, yμ, marker = :+, lw = 1.5)
end

p = plot(p1, p2; layout = (2, 1), size = (800, 700), plot_title = "ISO Pichler cement paste (ω = $ω_iso)")
figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "42_cementpaste_iso.png")
savefig(p, figpath)
display(p)
@printf "\nSaved : %s\n" figpath
println("Done.")
