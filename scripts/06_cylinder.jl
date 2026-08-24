# =============================================================================
#  06_cylinder.jl
#
#  Hill polarization tensor for infinite cylindrical inclusions.
#
#  Run from the MeanFieldHomogenization.jl root:
#    julia --project=. scripts/06_cylinder.jl
#
#  Sections:
#   § 1  Circular cylinder (b = c) in isotropic matrix — TensTI{4} output
#   § 2  Elliptic cylinder (b > c) in isotropic matrix — TensOrtho output
#   § 3  Anisotropic matrix — 1D quadrature path
#   § 4  Redirection Ellipsoid(Inf, …) → Cylinder
#   § 5  Conductivity : 2nd-order Hill tensor
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

const E_ref = 210.0e3
const ν_ref = 0.3
const λ_ref = E_ref * ν_ref / ((1 + ν_ref) * (1 - 2ν_ref))
const μ_ref = E_ref / (2 * (1 + ν_ref))
const k_ref = λ_ref + 2μ_ref / 3
const C_iso = TensISO{3}(3k_ref, 2μ_ref)

const voigt_idx = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
const voigt_lab = ["11", "22", "33", "23", "31", "12"]

function print_voigt(C; label = "P", scale = 1.0e6, unit = "×10⁻⁶ MPa⁻¹")
    println("  Voigt[$label] ($unit):")
    print("      "); for l in voigt_lab
        @printf "%10s" l
    end; println()
    for (I, (i, j)) in enumerate(voigt_idx)
        @printf "  %2s | " voigt_lab[I]
        for (k, l) in voigt_idx
            @printf "%9.4f " scale * C[i, j, k, l]
        end
        println()
    end
    return
end

@printf "Reference matrix: isotropic steel, E = %.1f MPa, ν = %.2f\n\n" E_ref ν_ref

# ─── § 1 — Circular cylinder ────────────────────────────────────────────────
println("§ 1 — Circular cylinder (b = c = 1.0) in isotropic matrix")
cyl_circ = Cylinder(1.0)
P_circ = hill_tensor(cyl_circ, C_iso)
println("  typeof(P) = ", typeof(P_circ), "  (transversely isotropic, axis e₁)")
print_voigt(P_circ; label = "P_circ")

# ─── § 2 — Elliptic cylinder ────────────────────────────────────────────────
println("\n§ 2 — Elliptic cylinder (b = 2, c = 1) in isotropic matrix")
cyl_ell = Cylinder(2.0, 1.0)
P_ell = hill_tensor(cyl_ell, C_iso)
println("  typeof(P) = ", typeof(P_ell), "  (orthotropic)")
print_voigt(P_ell; label = "P_ell")

# ─── § 3 — Anisotropic matrix ───────────────────────────────────────────────
println("\n§ 3 — Elliptic cylinder in orthotropic matrix (quadrature 1D)")
C_ortho = TensND.TensOrtho(
    210.0, 200.0, 150.0, 80.0, 70.0, 60.0, 90.0, 85.0, 75.0,
    TensND.CanonicalBasis{3, Float64}()
)
P_aniso = hill_tensor(cyl_ell, C_ortho)
println("  typeof(P) = ", typeof(P_aniso))
@printf "  P[1,1,1,1] = %.3e  (axial component, should ≈ 0)\n" P_aniso[1, 1, 1, 1]
@printf "  P[2,2,2,2] = %.6e\n" P_aniso[2, 2, 2, 2]
@printf "  P[3,3,3,3] = %.6e\n" P_aniso[3, 3, 3, 3]
@printf "  P[2,3,2,3] = %.6e\n" P_aniso[2, 3, 2, 3]

# Cross-check with DECUHR on a very elongated triaxial ellipsoid
ell_big = Ellipsoid(1.0e6, 2.0, 1.0)
P_ref = hill_tensor(ell_big, C_ortho; method = :decuhr)
rel_err = abs(P_aniso[2, 2, 2, 2] - P_ref[2, 2, 2, 2]) / abs(P_ref[2, 2, 2, 2])
@printf "  Δrel vs DECUHR limit (a=1e6) : %.2e\n" rel_err

# ─── § 4 — Redirection from Ellipsoid ───────────────────────────────────────
println("\n§ 4 — Redirection  Ellipsoid(…) → Cylinder / Crack")
for args in ((Inf, 2.0, 1.0), (1.0, 1.0, Inf), (2.0, 1.0, 0.0), (Inf, 1.0, 0.0))
    obj = Ellipsoid(args...)
    @printf "  Ellipsoid%-22s → %s\n" string(args) typeof(obj).name.name
end

# ─── § 5 — Conductivity ─────────────────────────────────────────────────────
println("\n§ 5 — 2nd-order Hill tensor (conductivity)")
k = 2.5
K_iso = TensISO{3}(k)
H_iso = hill_tensor(cyl_ell, K_iso)
@printf "  Iso matrix (k = %.2f):  H = diag(%g, %.4f, %.4f)\n" k H_iso[1, 1] H_iso[2, 2] H_iso[3, 3]
@printf "  Expected:              H = diag(0, %.4f, %.4f)\n" 1 / (k * 3) 2 / (k * 3)

K_aniso = TensND.Tens([2.0 0.0 0.0; 0.0 3.5 0.5; 0.0 0.5 1.8])
H_aniso = hill_tensor(cyl_ell, K_aniso)
@printf "  Aniso matrix:          H[2,2] = %.4f, H[3,3] = %.4f, H[2,3] = %.4f\n" H_aniso[2, 2] H_aniso[3, 3] H_aniso[2, 3]

println("\nDone.")
