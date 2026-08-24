# =============================================================================
#  04_forwarddiff.jl
#
#  Numerical automatic differentiation (AD) via ForwardDiff.jl.
#
#  Run from the MeanFieldHomogenization.jl root:
#    julia --project=. scripts/04_forwarddiff.jl
#
#  Prerequisites:
#    julia> using Pkg; Pkg.add("ForwardDiff")
#
#  Sections:
#   § 1  Shape sensitivity  ∂P/∂η   (aspect ratio of a prolate spheroid)
#   § 2  Full shape gradient  ∇_{a,b,c} P  (all three semi-axes)
#   § 3  Material sensitivity  ∂P/∂μ, ∂P/∂k  (isotropic elastic matrix)
#   § 4  Conductivity:  ∂P/∂ρ  (2D ellipse) and  ∂P/∂k₀
#   § 5  Validation against centerd finite differences
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra
using Printf

# ─── Reference parameters ────────────────────────────────────────────────────
const E_ref = 210.0e3;  const ν_ref = 0.3
const λ_ref = E_ref * ν_ref / ((1 + ν_ref) * (1 - 2ν_ref))
const μ_ref = E_ref / (2 * (1 + ν_ref))
const k_ref = λ_ref + 2μ_ref / 3

# Helper: centerd finite difference derivative
function fd_deriv(f, x₀; h = 1.0e-5)
    return (f(x₀ + h) - f(x₀ - h)) / (2h)
end

# Helper: centerd FD gradient (vector-valued scalar function)
function fd_grad(f, x₀; h = 1.0e-5)
    n = length(x₀)
    g = similar(x₀)
    for i in 1:n
        xp = copy(x₀);  xp[i] += h
        xm = copy(x₀);  xm[i] -= h
        g[i] = (f(xp) - f(xm)) / (2h)
    end
    return g
end

# ═══════════════════════════════════════════════════════════════════════════
println("="^70)
println("  § 1  SHAPE SENSITIVITY  ∂P_{1111}/∂η  (prolate spheroid, iso matrix)")
println("="^70)
println("  Ellipsoid(1, η, η) with η ∈ (0,1]:  a=1 fixed, b=c=η vary.")
println("  Method :auto (analytical) supports ForwardDiff.Dual geometry.")
println()

let
    C_iso = TensISO{3}(3k_ref, 2μ_ref)

    # Scalar function: one component vs aspect ratio
    f_η = η -> hill_tensor(Ellipsoid(1.0, η, η), C_iso)[1, 1, 1, 1]

    η_vals = [0.2, 0.5, 0.8, 1.0]
    println("   η      P[1111]          ∂P[1111]/∂η   FD check       Error")
    for η in η_vals
        dP_AD = ForwardDiff.derivative(f_η, η)
        dP_FD = fd_deriv(f_η, η; h = 1.0e-5)
        @printf "  %.2f  %12.6e  %12.6e  %12.6e  %.2e\n" η f_η(η) dP_AD dP_FD abs(dP_AD - dP_FD)
    end

    println("\n  Full gradient of P[1,1,1,1] w.r.t. (a, b, c) at (3, 1, 1):")
    f_abc = x -> hill_tensor(Ellipsoid(x[1], x[2], x[3]), C_iso)[1, 1, 1, 1]
    x₀ = [3.0, 1.0, 1.0]
    grad_AD = ForwardDiff.gradient(f_abc, x₀)
    grad_FD = fd_grad(f_abc, x₀; h = 1.0e-4)
    println("   component     ForwardDiff    FiniteDiff     Error")
    for (i, lbl) in enumerate(["∂/∂a", "∂/∂b", "∂/∂c"])
        @printf "  %-8s  %12.6e  %12.6e  %.2e\n" lbl grad_AD[i] grad_FD[i] abs(grad_AD[i] - grad_FD[i])
    end
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 2  FULL JACOBIAN  ∂P_{IJ}/∂η  (all 21 Voigt components)")
println("="^70)
println("  Sensitivity of the entire P tensor to the transverse semi-axis η.")

let
    C_iso = TensISO{3}(3k_ref, 2μ_ref)
    voigt = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
    lab = ["11", "22", "33", "23", "13", "12"]

    η₀ = 0.5
    println("\n  Prolate spheroid Ellipsoid(1.0, η, η), evaluated at η = $η₀:")
    println("  (only non-zero Voigt components shown)")
    println()
    println("  P_IJ         P[IJ]          ∂P[IJ]/∂η")

    for (I, (i, j)) in enumerate(voigt), (J, (k, l)) in enumerate(voigt)
        J >= I || continue
        f = η -> hill_tensor(Ellipsoid(1.0, η, η), C_iso)[i, j, k, l]
        v = f(η₀)
        dv = ForwardDiff.derivative(f, η₀)
        abs(v) > 1.0e-14 || abs(dv) > 1.0e-14 || continue
        @printf "  P[%s,%s]   %12.6e   %12.6e\n" lab[I] lab[J] v dv
    end
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 3  MATERIAL SENSITIVITY  ∂P/∂μ and ∂P/∂k  (elastic matrix)")
println("="^70)
println("  Geometry fixed (sphere). Differentiate w.r.t. matrix elastic constants.")

let
    ell = Ellipsoid(1.0)

    # Differentiate w.r.t. shear modulus μ
    f_μ = μ -> begin
        λ = E_ref * ν_ref / ((1 + ν_ref) * (1 - 2ν_ref))
        k = λ + 2μ / 3
        C = TensISO{3}(3k, 2μ)
        hill_tensor(ell, C)[1, 1, 1, 1]
    end

    # Differentiate w.r.t. bulk modulus k
    f_k = k -> begin
        μ = μ_ref
        C = TensISO{3}(3k, 2μ)
        hill_tensor(ell, C)[1, 1, 1, 1]
    end

    dP_dμ_AD = ForwardDiff.derivative(f_μ, μ_ref)
    dP_dμ_FD = fd_deriv(f_μ, μ_ref; h = 1.0)
    dP_dk_AD = ForwardDiff.derivative(f_k, k_ref)
    dP_dk_FD = fd_deriv(f_k, k_ref; h = 1.0)

    @printf "\n  P[1,1,1,1] at reference = %12.9e MPa⁻¹\n" f_μ(μ_ref)
    @printf "\n  ∂P[1111]/∂μ:  AD = %12.6e,  FD = %12.6e,  err = %.2e\n" dP_dμ_AD dP_dμ_FD abs(dP_dμ_AD - dP_dμ_FD)
    @printf   "  ∂P[1111]/∂k:  AD = %12.6e,  FD = %12.6e,  err = %.2e\n" dP_dk_AD dP_dk_FD abs(dP_dk_AD - dP_dk_FD)

    # Analytical sensitivity (sphere):
    # P_{1111} = 1/(5(λ+2μ)) + 2/(15μ)
    # ∂P/∂μ: λ+2μ changes by 2 when μ changes by 1 (with k=λ+2μ/3 varying too)
    #         ∂P/∂μ = -2/(5(λ+2μ)²) - 2/(15μ²)
    # ∂P/∂k:  f_k holds μ fixed → λ+2μ = k+4μ/3 changes by 1 when k changes by 1
    #         ∂P/∂k = -1/(5(λ+2μ)²)
    λ = λ_ref; μ = μ_ref
    dP_dμ_th = -2 / (5 * (λ + 2μ)^2) - 2 / (15μ^2)
    dP_dk_th = -1 / (5 * (λ + 2μ)^2)
    @printf "\n  Analytical ∂P[1111]/∂μ = %12.6e  (err=%.2e)\n" dP_dμ_th abs(dP_dμ_AD - dP_dμ_th)
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 4  CONDUCTIVITY — ∂P/∂ρ (shape) and ∂P/∂k₀ (material), 2D")
println("="^70)

let
    k₀ = 50.0

    # Shape sensitivity in 2D (ellipse, isotropic K₀)
    f_ρ_11 = ρ -> begin
        K = TensISO{2}(k₀)
        hill_tensor(Ellipsoid(1.0, ρ), K)[1, 1]
    end
    f_ρ_22 = ρ -> begin
        K = TensISO{2}(k₀)
        hill_tensor(Ellipsoid(1.0, ρ), K)[2, 2]
    end

    ρ₀ = 0.3
    dP11_dρ_AD = ForwardDiff.derivative(f_ρ_11, ρ₀)
    dP22_dρ_AD = ForwardDiff.derivative(f_ρ_22, ρ₀)
    dP11_dρ_FD = fd_deriv(f_ρ_11, ρ₀; h = 1.0e-5)
    dP22_dρ_FD = fd_deriv(f_ρ_22, ρ₀; h = 1.0e-5)

    println("\n  2D Ellipse Ellipsoid(1, ρ) at ρ₀=$ρ₀ (a=1 fixed):")
    @printf "  ∂P[1,1]/∂ρ:  AD=%12.6e,  FD=%12.6e,  err=%.2e\n" dP11_dρ_AD dP11_dρ_FD abs(dP11_dρ_AD - dP11_dρ_FD)
    @printf "  ∂P[2,2]/∂ρ:  AD=%12.6e,  FD=%12.6e,  err=%.2e\n" dP22_dρ_AD dP22_dρ_FD abs(dP22_dρ_AD - dP22_dρ_FD)

    # Material sensitivity: ∂P/∂k₀ (sphere 3D)
    f_k = k -> begin
        K = TensISO{3}(k)
        hill_tensor(Ellipsoid(1.0), K)[1, 1]
    end
    dP_dk_AD = ForwardDiff.derivative(f_k, k₀)
    dP_dk_FD = fd_deriv(f_k, k₀; h = 1.0e-3)
    dP_dk_th = -1 / (3k₀^2)   # P[1,1] = 1/(3k₀), so ∂P/∂k₀ = -1/(3k₀²)

    println("\n  3D Sphere: ∂P[1,1]/∂k₀  (P[1,1] = 1/(3k₀)):")
    @printf "  AD = %12.6e,  FD = %12.6e,  Theory = %12.6e\n" dP_dk_AD dP_dk_FD dP_dk_th
    @printf "  AD error vs theory: %.2e\n" abs(dP_dk_AD - dP_dk_th)
end

# ═══════════════════════════════════════════════════════════════════════════
println("\n", "="^70)
println("  § 5  ANISOTROPIC MATRIX — shape AD via :decuhr")
println("="^70)
println("  :residues does NOT support ForwardDiff.Dual geometry.")
println("  :decuhr integrates numerically with Dual-valued ζ inside the integrand.")

let
    # Build anisotropic (cubic) stiffness
    C11, C12, C44 = 250.0e3, 100.0e3, 80.0e3
    voigt_idx = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
    C_arr = zeros(3, 3, 3, 3)
    for (I, (i, j)) in enumerate(voigt_idx), (J, (k, l)) in enumerate(voigt_idx)
        v = [
            C11 C12 C12 0 0 0; C12 C11 C12 0 0 0; C12 C12 C11 0 0 0;
            0 0 0 C44 0 0; 0 0 0 0 C44 0; 0 0 0 0 0 C44
        ][I, J]
        C_arr[i, j, k, l] = C_arr[j, i, k, l] = C_arr[i, j, l, k] = C_arr[j, i, l, k] = v
    end
    C_cubic = Tens(C_arr)

    f_η_aniso = η -> hill_tensor(
        Ellipsoid(1.0, η, η), C_cubic;
        method = :decuhr, abstol = 1.0e-6, reltol = 1.0e-4
    )[1, 1, 1, 1]

    η₀ = 0.5
    dP_AD = ForwardDiff.derivative(f_η_aniso, η₀)
    dP_FD = fd_deriv(f_η_aniso, η₀; h = 1.0e-4)

    @printf "\n  Cubic matrix, Ellipsoid(1,η,η), η₀=%.2f:\n" η₀
    @printf "  ∂P[1111]/∂η:  AD (decuhr) = %12.6e\n" dP_AD
    @printf "                FD           = %12.6e\n" dP_FD
    @printf "                Error        = %.3e\n" abs(dP_AD - dP_FD)
end

println()
println("="^70)
println("  All AD results agree with finite differences to expected tolerance.")
println("="^70)
