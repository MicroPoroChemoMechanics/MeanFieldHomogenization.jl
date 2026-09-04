# =============================================================================
#  25_echoes_crosscheck.jl
#
#  Cross-validation of MeanFieldHomogenization.Schemes against:
#   * the Christensen 1990 closed form for the iso 2-phase Mori-Tanaka bulk
#     modulus (analytical reference);
#   * the iso-porous self-consistent benchmark from the ECHOES reference
#     implementation this Julia port is validated against.
#
#  Closes the loop on the same kind of cross-validation we did for the
#  Hill TI coaxial analytical kernel (script 07).
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf

# ── Christensen 1990 closed-form Mori-Tanaka bulk for iso 2-phase ─────────
#
#   k_MT = k_m + f (k_i - k_m) / (1 + (1-f)(k_i - k_m)/(k_m + 4μ_m/3))
#   μ_MT = μ_m + f (μ_i - μ_m) / (1 + (1-f)(μ_i - μ_m)/(μ_m + ζ_m))
#   ζ_m  = μ_m (9k_m + 8μ_m) / (6(k_m + 2μ_m))
#
function mt_closed_form(km, μm, ki, μi, f)
    k_eff = km + f * (ki - km) / (1 + (1 - f) * (ki - km) / (km + 4μm / 3))
    ζm = μm * (9km + 8μm) / (6 * (km + 2μm))
    μ_eff = μm + f * (μi - μm) / (1 + (1 - f) * (μi - μm) / (μm + ζm))
    return (k_eff, μ_eff)
end

println("=== Mori-Tanaka — Christensen 1990 cross-check ===")
println()
println("Inputs : matrix (k_m, μ_m) = (10, 5),  inclusion (k_i, μ_i) = (40, 20)")
println()
@printf("%6s   %12s   %12s   %12s   %12s\n", "f", "k_MFH", "k_closed", "μ_MFH", "μ_closed")
println("─"^70)

km, μm = 10.0, 5.0
ki, μi = 40.0, 20.0

for f in (0.05, 0.1, 0.2, 0.3, 0.4)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3km, 2μm)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3ki, 2μi));
        fraction = f
    )
    C_mt = homogenize(rve, MoriTanaka())
    k_mfh, μ_mfh = k_mu(C_mt)
    k_cf, μ_cf = mt_closed_form(km, μm, ki, μi, f)
    @printf(
        "%6.3f   %12.6f   %12.6f   %12.6f   %12.6f\n",
        f, k_mfh, k_cf, μ_mfh, μ_cf
    )
end
println()

# ── Self-consistent on iso porous ─────────────────────────────────────────
println("=== Self-consistent — iso porous ===")
println()
println("Inputs : matrix (E, ν) = (1, 0) ⇒ (k_m, μ_m) = (1/3, 0.5);")
println("         spherical pores with vanishingly small moduli (≈0).")
println()

# Compatible with the Hashin-Shtrikman lower bound on a porous iso material.
# We take k_pore, μ_pore = 1e-6 (numerically tiny but non-zero) to keep the
# stiffness invertible at every iteration.
function porous_iso(f, scheme)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)   # 3k=1, 2μ=1
    f > 0 && add_phase!(
        rve, :PORE, Ellipsoid(1.0),
        Dict(:C => TensISO{3}(1.0e-6, 1.0e-6)); fraction = f
    )
    return get_array(homogenize(rve, scheme))[1, 1, 1, 1]
end

@printf(
    "%6s   %12s   %12s   %12s   %12s\n",
    "f", "Voigt", "MT", "ASC", "Differential"
)
println("─"^65)
for f in (0.05, 0.1, 0.2, 0.3, 0.4)
    Cv = porous_iso(f, Voigt())
    Cmt = porous_iso(f, MoriTanaka())
    Casc = porous_iso(f, AsymmetricSelfConsistent(; abstol = 1.0e-10, maxiters = 200))
    Cd = porous_iso(f, DifferentialScheme(; nsteps = 200))
    @printf("%6.3f   %12.6f   %12.6f   %12.6f   %12.6f\n", f, Cv, Cmt, Casc, Cd)
end
println()
println("Done.")
