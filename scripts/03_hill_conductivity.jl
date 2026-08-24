# =============================================================================
#  03_hill_conductivity.jl
#
#  2nd-order Hill polarization tensors for conductive inclusions.
#
#  Run from the MeanFieldHomogenization.jl root:
#    julia --project=. scripts/03_hill_conductivity.jl
#
#  Sections:
#   § 1  Isotropic conductivity — sphere, prolate, oblate (P = I^A / k)
#   § 2  Anisotropic conductivity — K^{-1/2} transformation
#   § 3  2D — circle, ellipse, anisotropic K
#   § 4  Dilute effective conductivity — needle / platelet composites
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf

# ─── Reference conductivity: isotropic k₀ = 50 W/(m·K) ─────────────────────
const k₀ = 50.0          # W/(m·K)
const K_iso = TensISO{3}(k₀)

@printf "Reference medium: isotropic k₀ = %.1f W/(m·K)\n\n" k₀

# ─── Helpers ─────────────────────────────────────────────────────────────────

function print_P2(P, dim; label = "P")
    println("  $label:")
    for i in 1:dim
        print("    [")
        for j in 1:dim
            @printf " %12.8f" P[i, j]
        end
        println(" ]")
    end
    return
end

# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("  § 1  ISOTROPIC CONDUCTIVITY   P = I^A / k")
println("="^70)
println("  For an isotropic conductor K₀ = k·1, the Hill tensor is simply")
println("  P(A, K₀) = I^A / k, where I^A is the 2nd-order auxiliary tensor.")

# ─── 1a. Sphere ──────────────────────────────────────────────────────────────
println("\n── Sphere  (a = b = c = 1) ──────────────────────────────────────────")
let ell = Ellipsoid(1.0)
    P = hill_tensor(ell, K_iso)
    IA = tens_IA(ell)

    @printf "\n  P[1,1] = %12.9f (m·K)/W  (expected 1/(3k₀) = %.9f)\n" P[1, 1] 1 / (3k₀)
    @printf "  Isotropy: P[1,1]=P[2,2]=P[3,3]?  err=%.2e\n" maximum(abs(P[i, i] - P[1, 1]) for i in 1:3)
    @printf "  Off-diagonal P[1,2] = %.2e\n" P[1, 2]

    # Verification: P = IA/k
    @printf "  IA[1,1]/k₀ = %12.9f  (should equal P[1,1])\n" IA[1, 1] / k₀
end

# ─── 1b. Prolate spheroid series ─────────────────────────────────────────────
println("\n── Prolate spheroid  (a/b = 1…100)  — axial vs transverse P ─────────")
println("   a/b     P[1,1]        P[2,2]        I^A_1+I^A_2+I^A_3")
for ab in (1.0, 2.0, 5.0, 10.0, 50.0, 100.0)
    ell = Ellipsoid(ab, 1.0, 1.0)
    P = hill_tensor(ell, K_iso)
    IA = tens_IA(ell)
    sum_IA = IA[1, 1] + IA[2, 2] + IA[3, 3]
    @printf "  %5.0f  %12.9f  %12.9f  %12.9f\n" ab P[1, 1] P[2, 2] sum_IA
end
println("  Expected P[i,i] = I^A_i / k₀  and  ΣI^A_i = 1")

# ─── 1c. Oblate spheroid ─────────────────────────────────────────────────────
println("\n── Oblate spheroid  (a=b, a/c = 1…100)  — normal vs in-plane P ──────")
println("   a/c     P_normal      P_inplane     P_normal×k₀ (→1 as penny)")
for ac in (1.0, 2.0, 5.0, 10.0, 50.0, 100.0)
    ell = Ellipsoid(ac, ac, 1.0)
    P = hill_tensor(ell, K_iso)
    @printf "  %5.0f  %12.9f  %12.9f  %12.7f\n" ac P[3, 3] P[1, 1] P[3, 3] * k₀
end
println("  For a penny (a/c→∞): P[3,3] → 1/k₀ (inclusion controls normal direction)")

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 2  ANISOTROPIC CONDUCTIVITY  (K^{-1/2} transformation)")
println("="^70)
println("  For K₀ = V·diag(k₁,k₂,k₃)·Vᵀ: maps to isotropic on a fictitious ellipsoid.")

# Diagonal anisotropic: K = diag(k1, k2, k3)
let
    k1, k2, k3 = 100.0, 50.0, 20.0   # W/(m·K)
    K_arr = zeros(3, 3)
    K_arr[1, 1] = k1;  K_arr[2, 2] = k2;  K_arr[3, 3] = k3
    K_aniso = Tens(K_arr)

    @printf "\nAnisotropic K₀ = diag(%.0f, %.0f, %.0f) W/(m·K)\n" k1 k2 k3

    println("\n── Sphere  (a = b = c = 1) in anisotropic K₀:")
    ell = Ellipsoid(1.0)
    P = hill_tensor(ell, K_aniso)
    print_P2(P, 3)
    @printf "  Note: P is no longer isotropic (diagonal but k₁≠k₂≠k₃)\n"

    println("\n── Prolate spheroid  (a=3, b=c=1)  in anisotropic K₀:")
    ell = Ellipsoid(3.0, 1.0, 1.0)
    P = hill_tensor(ell, K_aniso)
    print_P2(P, 3)
end

# Off-diagonal anisotropic (rotation of a diagonal tensor by π/4 in 1-2 plane)
let
    R = [cos(π / 4) -sin(π / 4) 0; sin(π / 4) cos(π / 4) 0; 0 0 1.0]
    K_diag = [100.0 0 0; 0 20.0 0; 0 0 50.0]
    K_mat = R * K_diag * R'
    K_arr = zeros(3, 3)
    for i in 1:3, j in 1:3
        K_arr[i, j] = K_mat[i, j]
    end
    K_offdiag = Tens(K_arr)

    println("\n── Sphere in off-diagonal K₀ (π/4 rotation of diag(100,20,50)):")
    P = hill_tensor(Ellipsoid(1.0), K_offdiag)
    print_P2(P, 3)
    println("  Note: P acquires off-diagonal components following K₀'s anisotropy.")
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 3  2D CONDUCTIVITY")
println("="^70)

let
    k₀_2d = 30.0
    K_iso2 = TensISO{2}(k₀_2d)

    println("\n── Circle  (r=1)  isotropic K₀:")
    P = hill_tensor(Ellipsoid(1.0; dim = 2), K_iso2)
    @printf "  P[1,1] = %12.9f  (expected 1/(2k₀) = %.9f)\n" P[1, 1] 1 / (2k₀_2d)
    @printf "  P[2,2] = %12.9f  (expected 1/(2k₀) = %.9f)\n" P[2, 2] 1 / (2k₀_2d)
    @printf "  P[1,2] = %12.2e\n" P[1, 2]

    println("\n── Ellipse  (a=4, b=1)  isotropic K₀:")
    ell = Ellipsoid(4.0, 1.0)
    P = hill_tensor(ell, K_iso2)
    IA = tens_IA(ell)
    @printf "  P[1,1] = %12.9f  (I^A_1/k₀ = %.9f)\n" P[1, 1] IA[1, 1] / k₀_2d
    @printf "  P[2,2] = %12.9f  (I^A_2/k₀ = %.9f)\n" P[2, 2] IA[2, 2] / k₀_2d

    println("\n── Ellipse  (a=4, b=1)  anisotropic K₀ = diag(60, 15):")
    K2_arr = zeros(2, 2)
    K2_arr[1, 1] = 60.0;  K2_arr[2, 2] = 15.0
    K2_aniso = Tens(K2_arr)
    P = hill_tensor(ell, K2_aniso)
    print_P2(P, 2)
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 4  DILUTE EFFECTIVE CONDUCTIVITY")
println("="^70)
println("  Dilute estimate:  K_eff = K₀ + f (K_i − K₀) : (I + P:(K_i−K₀))⁻¹")

let
    f = 0.1   # 10% volume fraction
    K₀_val = k₀

    println("\n── Perfectly conducting inclusions (K_i → ∞):")
    println("   Equivalent to: δK = K_i − K₀ → ∞, so A_dil → (P:δK)⁻¹ → 0")
    println("   C_eff ≈ K₀ + f * (I/P + K₀)  (approximation for K_i≫K₀)")
    println("   In practice: use large but finite K_i.")

    K_i_val = 1.0e6 * K₀_val  # very stiff inclusion
    K_iso_i = TensISO{3}(K_i_val)

    for (shape_label, ell) in (
            ("Sphere    (a=b=c=1)", Ellipsoid(1.0)),
            ("Needle    (a=50,b=c=1)", Ellipsoid(50.0, 1.0, 1.0)),
            ("Platelet  (a=b=50,c=1)", Ellipsoid(50.0, 50.0, 1.0)),
        )
        P = hill_tensor(ell, K_iso)
        δk = K_i_val - K₀_val
        # Scalar isotropic case: P_eff = P + δk → K_eff_ii = K₀ + f*δk/(1+P_ii*δk)
        k_eff_1 = K₀_val + f * δk / (1 + P[1, 1] * δk)
        k_eff_3 = K₀_val + f * δk / (1 + P[3, 3] * δk)
        @printf "  %-28s  k_eff_1/k₀=%.3f   k_eff_3/k₀=%.3f\n" shape_label k_eff_1 / K₀_val k_eff_3 / K₀_val
    end

    println()
    println("  Interpretation:")
    println("   Sphere:   isotropic enhancement in all directions")
    println("   Needle:   strong enhancement only in axial direction (k_eff_1)")
    println("   Platelet: strong enhancement in in-plane directions (k_eff_1,2)")
end

println()
println("="^70)
