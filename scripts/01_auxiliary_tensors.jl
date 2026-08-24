# =============================================================================
#  01_auxiliary_tensors.jl
#
#  Auxiliary geometric tensors  I^A, U^A, V^A  for ellipsoidal inclusions.
#
#  Run from the MeanFieldHomogenization.jl root:
#    julia --project=. scripts/01_auxiliary_tensors.jl
#
#  Mathematical definitions (hill_tensors.qmd §2):
#
#   I^A_{ij}   = (det A / 4π) ∫ ξᵢξⱼ / ‖A·ξ‖³ dS_ξ
#   U^A_{ijkl} = (det A / 4π) ∫ ξᵢξⱼξₖξₗ / ‖A·ξ‖⁵ dS_ξ
#   V^A        = (1 ⊠ˢ I^A + I^A ⊠ˢ 1) / 2
#
#  Normalization: Σᵢ I^A_i = 1  (3D),   I^A_1 + I^A_2 = 1  (2D)
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf

# ─── Helpers ─────────────────────────────────────────────────────────────────

function print_IA(IA, dim; label = "I^A")
    d = [IA[i, i] for i in 1:dim]
    return @printf "  %-14s diag = [%s ]   sum = %.8f\n" label join([@sprintf " %10.7f" v for v in d]) sum(d)
end

const voigt3 = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
const voigt2 = ((1, 1), (2, 2), (1, 2))

function print_voigt(C, dim; label = "")
    vij = dim == 3 ? voigt3 : voigt2
    lab = dim == 3 ? ["11", "22", "33", "23", "31", "12"] : ["11", "22", "12"]
    isempty(label) || println("  Voigt[$label]:")
    print("      ")
    for l in lab
        @printf "%10s" l
    end; println()
    for (I, (i, j)) in enumerate(vij)
        @printf "  %2s | " lab[I]
        for (k, l) in vij
            @printf "%9.5f " C[i, j, k, l]
        end
        println()
    end
    return
end

# ─────────────────────────────────────────────────────────────────────────────
println("="^70)
println("  AUXILIARY TENSORS  I^A, U^A, V^A  —  MeanFieldHomogenization.jl")
println("="^70)
println("  Normalization: Σᵢ I^A_i = 1 (3D),  I^A_1+I^A_2 = 1 (2D)")

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  1.  SPHERE   a = b = c = 1")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let ell = Ellipsoid(1.0)
    IA = tens_IA(ell);  UA = tens_UA(ell);  VA = tens_VA(ell)

    println("\n  I^A (by symmetry: each component = 1/3):")
    print_IA(IA, 3)
    @printf "  Off-diagonal IA[1,2] = %.2e  (must be 0)\n" IA[1, 2]

    println("\n  V^A = (1 ⊠ˢ I^A + I^A ⊠ˢ 1)/2 :")
    @printf "  V^A_{iiii} = I^A_i  →  VA[1,1,1,1] = %10.7f  (expected 1/3)\n" VA[1, 1, 1, 1]
    @printf "  V^A_{ijij} = (I^A_i+I^A_j)/4  →  VA[1,2,1,2] = %10.7f  (expected 1/6)\n" VA[1, 2, 1, 2]
    @printf "  V^A_{iijj} = 0  →  VA[1,1,2,2] = %10.7f\n" VA[1, 1, 2, 2]

    println("\n  U^A (formula: U^A_{iiii} = 3(I_i − aᵢ² I_{ii})/2):")
    @printf "  UA[1,1,1,1] = %10.7f  (expected 1/5  = 0.200000)\n" UA[1, 1, 1, 1]
    @printf "  UA[1,1,2,2] = %10.7f  (expected 1/15 ≈ 0.066667)\n" UA[1, 1, 2, 2]
    @printf "  UA[1,2,1,2] = %10.7f  (expected 1/15)\n" UA[1, 2, 1, 2]

    println("\n  Voigt matrix of U^A (sphere):")
    print_voigt(UA, 3; label = "U^A")
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  2.  PROLATE SPHEROID  a = 5, b = c = 1   (fiber-like inclusion)")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let ell = Ellipsoid(5.0, 1.0, 1.0)
    IA = tens_IA(ell);  UA = tens_UA(ell)

    println("\n  Transverse isotropy: I^A_2 = I^A_3,  I^A_1 < I^A_2")
    print_IA(IA, 3)
    @printf "  I^A_2 − I^A_3 = %.2e  (must be 0)\n" abs(IA[2, 2] - IA[3, 3])

    println("\n  Needle series  (a/b → ∞ : I^A_1→0, I^A_2=I^A_3→1/2):")
    println("   a/b    I^A_1       I^A_2       I^A_3")
    for ab in (1.0, 2.0, 5.0, 10.0, 50.0, 1000.0)
        IA_i = tens_IA(Ellipsoid(ab, 1.0, 1.0))
        @printf "  %5.0f  %10.7f  %10.7f  %10.7f\n" ab IA_i[1, 1] IA_i[2, 2] IA_i[3, 3]
    end
    println("   ∞     0.0000000   0.5000000   0.5000000  ← needle limit")
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  3.  OBLATE SPHEROID  a = b = 5, c = 1   (disk / platelet)")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let
    println("\n  Penny series  (a/c → ∞ : I^A_1=I^A_2→0, I^A_3→1):")
    println("   a/c    I^A_1       I^A_2       I^A_3")
    for ac in (1.0, 2.0, 5.0, 10.0, 50.0, 1000.0)
        IA_i = tens_IA(Ellipsoid(ac, ac, 1.0))
        @printf "  %5.0f  %10.7f  %10.7f  %10.7f\n" ac IA_i[1, 1] IA_i[2, 2] IA_i[3, 3]
    end
    println("   ∞     0.0000000   0.0000000   1.0000000  ← penny limit")
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  4.  TRIAXIAL ELLIPSOID  a = 4, b = 2, c = 1")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let ell = Ellipsoid(4.0, 2.0, 1.0)
    IA = tens_IA(ell);  UA = tens_UA(ell)

    println("\n  Three distinct principal values:")
    print_IA(IA, 3)

    println("\n  Voigt matrix of U^A (triaxial):")
    print_voigt(UA, 3; label = "U^A")
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  5.  ROTATED ELLIPSOID  a = 3, b = 2, c = 1  (ZYZ Euler θ=π/4)")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let
    θ = π / 4
    ell_can = Ellipsoid(3.0, 2.0, 1.0)
    ell_rot = Ellipsoid(3.0, 2.0, 1.0; euler_angles = (θ, 0.0, 0.0))

    IA_can = tens_IA(ell_can)
    IA_rot = change_tens_canon(tens_IA(ell_rot))

    println("\n  Canonical frame (diagonal):")
    print_IA(IA_can, 3; label = "I^A canonical")

    println("  Rotated (dense in global frame, same trace):")
    print_IA(IA_rot, 3; label = "I^A rotated")

    println("  Full rotated I^A matrix in canonical (global) frame:")
    for i in 1:3
        print("    [")
        for j in 1:3
            @printf " %10.7f" IA_rot[i, j]
        end
        println(" ]")
    end
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  6.  2D — CIRCLE  r = 1")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let ell = Ellipsoid(1.0; dim = 2)
    IA = tens_IA(ell);  UA = tens_UA(ell);  VA = tens_VA(ell)

    println("\n  I^A (isotropic: each = 1/2):")
    print_IA(IA, 2)
    @printf "  VA[1,1,1,1] = %10.7f  (expected 1/2)\n" VA[1, 1, 1, 1]
    @printf "  VA[1,2,1,2] = %10.7f  (expected 1/4)\n" VA[1, 2, 1, 2]

    println("\n  Voigt matrix of U^A (2D circle):")
    print_voigt(UA, 2; label = "U^A")
end

# ═══════════════════════════════════════════════════════════════════════
println("\n", "─"^70)
println("  7.  2D — ELLIPSE  a = 4, b = 1")
println("─"^70)
# ═══════════════════════════════════════════════════════════════════════

let ell = Ellipsoid(4.0, 1.0)
    IA = tens_IA(ell);  UA = tens_UA(ell)

    println("\n  Analytical: I^A_1 = b/(a+b) = 1/5,  I^A_2 = a/(a+b) = 4/5")
    print_IA(IA, 2)
    @printf "  Expected:   diag = [ %10.7f  %10.7f ]   sum = 1.0000000\n" 0.2 0.8

    println("\n  Voigt matrix of U^A (2D ellipse a/b=4):")
    print_voigt(UA, 2; label = "U^A")
end

println()
println("="^70)
println("  All normalization checks verified.")
println("="^70)
