# =============================================================================
#  08_hill_derivatives.jl
#
#  Derivatives of the Hill polarization tensor P with respect to the reference
#  medium — the MeanFieldHomogenization counterpart of a hand-coded `hill_derivative`.
#
#  The classical approach hand-codes an analytical `hill_derivative(ell, C,
#  index, sym)` for each material-symmetry class (ISO, TI, ORTHO).
#  MeanFieldHomogenization gets the SAME derivative for free by ForwardDiff through the
#  `hill_tensor` kernel — for ANY parametrization, including fully triclinic
#  references that a symmetry-typed routine cannot handle.
#
#  This demo (main environment, no PyCall) :
#    * computes ∂P/∂κ and ∂P/∂η of an isotropic reference (κ = 3k, η = 2μ)
#      for a triaxial ellipsoid — echoes `hill_derivative(ell, C, index, ISO)`,
#    * validates each AD derivative against a central finite difference,
#    * differentiates a TI reference (`tensor([5,3,4,8,3])`) w.r.t. its five
#      Walpole coefficients, on a coaxial spheroid (analytical Hill).
#
#  The live echoes-vs-MeanFieldHomogenization cross-check (PyCall, incl. the ORTHO and
#  triclinic cases) lives in `scripts/bench_echoes/benchmark_hill_derivative.jl`.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra
using Printf

# Full 3×3×3×3 component array of the Hill tensor in the canonical frame.
# We compare arrays (not the 6×6 Kelvin-Mandel matrix) because `tomandel`
# collapses to 6×6 only when the tensor is flagged minor-symmetric — a flag
# that is set for plain `Float64` but not always under `ForwardDiff.Dual`,
# which would make value (6×6) and derivative (9×9) shapes disagree.  The
# 3×3×3×3 array is shape-stable across both.
P_arr(ell, C) = TensND.get_array(change_tens_canon(hill_tensor(ell, C)))
maxrel(A, B) = maximum(abs, A .- B) / max(maximum(abs, A), maximum(abs, B), 1.0e-300)

# Triaxial ellipsoid with Euler angles — the reference triaxial shape
# `ellipsoidal([3, 2.5, 1.6, 0.1, 0.2, 0.3])`.
const ELL = Ellipsoid(3.0, 2.5, 1.6; euler_angles = (0.1, 0.2, 0.3))
const k0, μ0 = 10.0, 10.0
const α0, β0 = 3k0, 2μ0            # 𝕁 and 𝕂 coefficients of the iso reference

println("="^74)
println("  Hill tensor derivatives ∂P/∂C — MeanFieldHomogenization (ForwardDiff)")
println("  reference: analytical hill_derivative")
println("="^74)
@printf "  ellipsoid semi-axes (3, 2.5, 1.6), Euler (0.1, 0.2, 0.3)\n"
@printf "  isotropic reference k = %.1f, μ = %.1f  (κ = 3k = %.1f, η = 2μ = %.1f)\n\n" k0 μ0 α0 β0

# ── ∂P/∂κ and ∂P/∂η by ForwardDiff, validated by central finite differences ──
f_κ = κ -> P_arr(ELL, TensISO{3}(κ, β0))
f_η = η -> P_arr(ELL, TensISO{3}(α0, η))
dP_dκ = ForwardDiff.derivative(f_κ, α0)
dP_dη = ForwardDiff.derivative(f_η, β0)

h = 1.0e-4
fd_κ = (f_κ(α0 + h) - f_κ(α0 - h)) / (2h)
fd_η = (f_η(β0 + h) - f_η(β0 - h)) / (2h)
@printf "  ∂P/∂κ : AD vs central-FD  max rel diff = %.2e\n" maxrel(dP_dκ, fd_κ)
@printf "  ∂P/∂η : AD vs central-FD  max rel diff = %.2e\n" maxrel(dP_dη, fd_η)
@printf "  ‖∂P/∂κ‖∞ = %.4e   ‖∂P/∂η‖∞ = %.4e\n\n" maximum(abs, dP_dκ) maximum(abs, dP_dη)

# ── TI reference : derivative w.r.t. the five Walpole coefficients ──────────
# For a TI reference we use a spheroid COAXIAL with the TI axis (ez), so the
# Hill tensor takes the analytical TI-coaxial branch (Barthélémy 2020) —
# exact and cleanly differentiable.  (A TI reference with a non-coaxial
# ellipsoid falls back to the NestedQuadGK cubature, whose adaptive node
# placement makes ForwardDiff derivatives only quadrature-accurate; use
# `method = :nestedquadgk` with tight tolerances there, or the PyCall bench
# script for the echoes cross-check.)
n = (0.0, 0.0, 1.0)
const SPH = Spheroid(2.0)                  # prolate, revolution axis = ez
ℓ0 = (5.0, 3.0, 4.0, 8.0, 3.0)            # (ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆) reference Walpole tuple
ti_from(p) = TensND.TensTI{4}(p[1], p[2], p[3], p[4], p[5], n)
println("  TI reference tensor([5,3,4,8,3], axis ez), coaxial spheroid ω=2 —")
println("  ∂P/∂ℓᵢ (ForwardDiff, analytical TI-coaxial Hill):")
for i in 1:5
    g = ForwardDiff.derivative(t -> P_arr(SPH, ti_from(ntuple(j -> j == i ? ℓ0[j] + t : ℓ0[j], 5))), 0.0)
    fd = let hh = 1.0e-4
        (P_arr(SPH, ti_from(ntuple(j -> j == i ? ℓ0[j] + hh : ℓ0[j], 5))) -
            P_arr(SPH, ti_from(ntuple(j -> j == i ? ℓ0[j] - hh : ℓ0[j], 5)))) / (2hh)
    end
    @printf "     ∂P/∂ℓ%d : ‖∂P‖∞ = %.4e   AD vs FD rel diff = %.2e\n" i maximum(abs, g) maxrel(g, fd)
end

println("\nDone.")
