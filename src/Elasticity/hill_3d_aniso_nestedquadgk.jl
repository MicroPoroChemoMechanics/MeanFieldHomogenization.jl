# =============================================================================
#  hill_3d_aniso_nestedquadgk.jl
#
#  ForwardDiff-compatible 2D cubature for the 3D anisotropic Hill tensor
#  via nested QuadGK: outer integral over u ∈ [0, 1], inner over φ ∈ [0, 2π].
#
#  The specific 2D parametrization
#
#     ζ(z, φ) = (√(1 - z²) cos φ, √(1 - z²) sin φ / η, z / ω)
#
#  is tied to the ellipsoid semi-axes (η, ω) — kept in this file because
#  the surrounding change-of-variable is inclusion-specific.
#
#  Robustness for flat / elongated ellipsoids
#  ==========================================
#  **Power change-of-variable**: z(u) = 1 − (1 − u)^α with
#  α = max(1, log₁₀(1/ω)).  For α = 1 (ω ≥ 0.1) the substitution is the
#  identity and carries no overhead.  For smaller ω the Gauss-Kronrod grid
#  is compressed near z = 1, capturing the steep integrand gradient without
#  increasing maxiters.
#
#  **Axis sort**: the Ellipsoid constructor (`_sort_axes_and_basis`) already
#  sorts `semi_axes` in descending order for real-valued types, so η ≤ 1
#  and ω ≤ 1 are guaranteed on entry.  The explicit re-sort below is kept
#  for safety with non-Real element types (ForwardDiff.Dual, symbolic) where
#  the constructor may not sort.  For the common Float64 case it is a no-op.
#
#  Historical note: this kernel was previously named `_hill_3d_aniso_decuhr`
#  because it targets the same mathematical problem as DECUHR cubature,
#  but the implementation uses nested 1-D QuadGK, not DECUHR.  The actual
#  DECUHR-based implementation lives in `hill_3d_aniso_decuhr.jl`.
# =============================================================================

"""
    _sym3_inv_acoustic(C₀_arr, ζ) -> NTuple{9}

Inverse of the acoustic tensor `K[i,j] = ζₖ C₀[k,i,j,l] ζₗ` (symmetric, since
`C₀` has minor symmetry), in closed form: only the 6 upper-triangle scalars of
`K` are computed, and its inverse is the scalar adjugate/determinant of a
symmetric 3×3 matrix — no `Matrix` allocation, no LU factorization, Dual-safe.
Returned flattened in column-major order, so `iK[i + (j-1)*3]` mirrors `iK[i,j]`.

A standalone top-level function (not a nested closure): closures capturing many
local scalars showed measurably *worse* allocation behavior than hoisting this
computation out, so it is kept separate deliberately.
"""
@inline function _sym3_inv_acoustic(C₀_arr::AbstractArray{T, 4}, ζ) where {T}
    K11 = zero(T); K22 = zero(T); K33 = zero(T)
    K12 = zero(T); K13 = zero(T); K23 = zero(T)
    for k in 1:3, l in 1:3
        ζζ = ζ[k] * ζ[l]
        K11 += ζζ * C₀_arr[k, 1, 1, l]
        K22 += ζζ * C₀_arr[k, 2, 2, l]
        K33 += ζζ * C₀_arr[k, 3, 3, l]
        K12 += ζζ * C₀_arr[k, 1, 2, l]
        K13 += ζζ * C₀_arr[k, 1, 3, l]
        K23 += ζζ * C₀_arr[k, 2, 3, l]
    end
    det = K11 * K22 * K33 + 2 * K12 * K13 * K23 -
        K11 * K23 * K23 - K22 * K13 * K13 - K33 * K12 * K12
    inv_det = one(T) / det
    iK11 = (K22 * K33 - K23 * K23) * inv_det
    iK22 = (K11 * K33 - K13 * K13) * inv_det
    iK33 = (K11 * K22 - K12 * K12) * inv_det
    iK12 = (K13 * K23 - K33 * K12) * inv_det
    iK13 = (K12 * K23 - K22 * K13) * inv_det
    iK23 = (K12 * K13 - K11 * K23) * inv_det
    return (iK11, iK12, iK13, iK12, iK22, iK23, iK13, iK23, iK33)
end

"""
    _hill_3d_aniso_nestedquadgk(ell, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by **nested 1-D QuadGK cubature** over the
unit sphere of the general [willis1977](@cite) integrand.
ForwardDiff-compatible. Includes axis-sort and power change-of-variable for
improved convergence on flat or elongated ellipsoids.
"""
function _hill_3d_aniso_nestedquadgk(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))

    C₀_princ = TensND.change_tens(C₀, ell.basis)
    C₀_arr = Array{T, 4}(undef, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C₀_arr[i, j, k, l] = C₀_princ[i, j, k, l]
    end

    # --- Axis sort (descending) -----------------------------------------------
    # Sort semi-axes so a₁ ≥ a₂ ≥ a₃, guaranteeing η ≤ 1 and ω ≤ 1.
    # The stiffness tensor is permuted consistently so that the parametrization
    # ζ = (√(1-z²)cosφ, √(1-z²)sinφ/η, z/ω) matches the sorted frame.
    # After integration, the result is un-permuted via the inverse permutation.
    p = sortperm(collect(ell.semi_axes), rev = true)
    if p != [1, 2, 3]
        C₀_arr_p = Array{T, 4}(undef, 3, 3, 3, 3)
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            C₀_arr_p[i, j, k, l] = C₀_arr[p[i], p[j], p[k], p[l]]
        end
        C₀_arr = C₀_arr_p
    end

    semi_sorted = ell.semi_axes[p]
    η = semi_sorted[2] / semi_sorted[1]
    ω = semi_sorted[3] / semi_sorted[1]

    # --- Change-of-variable exponent ------------------------------------------
    # z(u) = 1 − (1 − u)^α concentrates Gauss-Kronrod nodes near z = 1
    # where the integrand varies most rapidly for flat ellipsoids (ω ≪ 1).
    # α = 1 (identity) when ω ≥ 0.1; beyond that α ≈ 2, 3, … for each decade.
    α = max(1.0, -log10(Float64(ω)))

    voigt_ij = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

    function integrand_at(z, φ)
        uz = sqrt(one(T) - T(z) * T(z))
        ζ = (uz * cos(T(φ)), uz * sin(T(φ)) / η, T(z) / ω)
        iK = _sym3_inv_acoustic(C₀_arr, ζ)

        vals = Vector{T}(undef, 21)
        idx = 0
        for I in 1:6
            i, j = voigt_ij[I]
            for J in I:6
                k, l = voigt_ij[J]
                γ = T(0.25) * (
                    ζ[i] * (iK[j + (k - 1) * 3] * ζ[l] + iK[j + (l - 1) * 3] * ζ[k]) +
                        ζ[j] * (iK[i + (k - 1) * 3] * ζ[l] + iK[i + (l - 1) * 3] * ζ[k])
                )
                idx += 1
                vals[idx] = γ
            end
        end
        return vals
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    P_vals, _ = QuadGK.quadgk(
        0.0, 1.0;
        atol = abstol,
        rtol = reltol,
        maxevals = maxiters
    ) do u
        one_minus_u = 1.0 - u
        z = 1.0 - one_minus_u^α
        jac = α * one_minus_u^(α - 1.0)
        inner, _ = MFH_Core._counted_quadgk(
            φ -> integrand_at(z, φ),
            0.0, 2π;
            atol = abstol / 10,
            rtol = reltol
        )
        inner .* jac
    end
    P_vals ./= T(2π)

    # --- Build result in sorted frame -----------------------------------------
    P_perm = zeros(T, 3, 3, 3, 3)
    idx = 0
    for I in 1:6, J in I:6
        i, j = voigt_ij[I]
        k, l = voigt_ij[J]
        idx += 1
        v = P_vals[idx]
        P_perm[i, j, k, l] = v;  P_perm[j, i, k, l] = v
        P_perm[i, j, l, k] = v;  P_perm[j, i, l, k] = v
        P_perm[k, l, i, j] = v;  P_perm[l, k, i, j] = v
        P_perm[k, l, j, i] = v;  P_perm[l, k, j, i] = v
    end

    # --- Un-permute back to original axis ordering ----------------------------
    # P_orig[q[i],q[j],q[k],q[l]] = P_perm[i,j,k,l]  where  q = invperm(p).
    q = invperm(p)
    P_arr = zeros(T, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        P_arr[q[i], q[j], q[k], q[l]] = P_perm[i, j, k, l]
    end

    return TensND.Tens(P_arr, ell.basis)
end
