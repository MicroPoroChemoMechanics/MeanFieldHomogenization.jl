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

# The components of a 3-D stiffness as a static array, for the quadrature
# kernels: their size known, the loops below read them from the stack, measured
# 1.7 times faster than from a heap array (9 ns against 15 per acoustic tensor).
_static_stiffness(A::AbstractArray{T, 4}) where {T} = SArray{Tuple{3, 3, 3, 3}, T}(A)

"""
    _sym3_inv_acoustic(C₀_arr, ζ) -> NTuple{9}

Inverse of the acoustic tensor ``K_{ij} = \\zeta_k\\,(C_0)_{kijl}\\,\\zeta_l`` (symmetric, since
`C₀` has minor symmetry), in closed form: TensND's `acoustic_components` gives the
6 upper-triangle scalars of `K` and its `sym3_inv` the adjugate over the
determinant — no `Matrix` allocation, no LU factorization, Dual-safe.
Returned flattened in column-major order, so `iK[i + (j-1)*3]` mirrors `iK[i,j]`.

A standalone top-level function (not a nested closure): closures capturing many
local scalars showed measurably *worse* allocation behavior than hoisting this
computation out, so it is kept separate deliberately.
"""
@inline _sym3_inv_acoustic(C₀_arr::AbstractArray{T, 4}, ζ) where {T} = TensND.sym3_inv(TensND.acoustic_components(C₀_arr, ζ)...)


"""
    _sym3_inv_acoustic_incompressible(C₀_arr, ζ) -> NTuple{9}

The Green kernel of an **incompressible** medium at the direction
``\\underline{\\zeta}``: the displacement is orthogonal to ``\\underline{\\zeta}`` in Fourier
space, so the acoustic tensor is inverted on the plane orthogonal to
``\\underline{\\zeta}`` only,
``\\boldsymbol{N} = (\\boldsymbol{Q}\\boldsymbol{K}\\boldsymbol{Q} + \\lambda\\,\\underline{n}\\otimes\\underline{n})^{-1} - \\underline{n}\\otimes\\underline{n}/\\lambda``
(any ``\\lambda > 0``; ``\\lambda = \\operatorname{tr}\\boldsymbol{K}/3`` keeps it well conditioned),
``\\boldsymbol{Q} = \\boldsymbol{1} - \\underline{n}\\otimes\\underline{n}``, ``\\underline{n}`` the unit vector along
``\\underline{\\zeta}``. It is the limit of the inverse
acoustic tensor as the bulk modulus grows without bound, taken exactly; the
stiffness `C₀` need only be defined on deviators. Returned as
[`_sym3_inv_acoustic`](@ref) returns ``\\boldsymbol{K}^{-1}``.
"""
@inline function _sym3_inv_acoustic_incompressible(C₀_arr::AbstractArray{T, 4}, ζ) where {T}
    K11, K22, K33, K12, K13, K23 = TensND.acoustic_components(C₀_arr, ζ)
    n = SVector{3}(ζ) / sqrt(ζ[1]^2 + ζ[2]^2 + ζ[3]^2)
    K = SMatrix{3, 3}(K11, K12, K13, K12, K22, K23, K13, K23, K33)
    # Q K Q + n⊗n, Q = 𝟏 − n⊗n, is invertible and splits into the plane and the
    # line of n: its inverse is N + n⊗n, one closed-form symmetric inverse.
    # The line of n is given the scale λ = tr K / 3 of the plane: with a unit
    # weight, the inverse mixes entries of order 1 and 1/K, and removing n⊗n
    # cancels all but ~eps·K of the plane part (relative 4e-8 measured, where
    # the quadrature stalled); with λ the matrix is conditioned as K itself.
    λ = (K11 + K22 + K33) / 3
    Kn = K * n
    M = K - n * transpose(Kn) - Kn * transpose(n) + (transpose(n) * Kn + λ) * (n * transpose(n))
    iM = TensND.sym3_inv(M[1, 1], M[2, 2], M[3, 3], M[1, 2], M[1, 3], M[2, 3])
    nn = n * transpose(n)
    return ntuple(q -> iM[q] - nn[q] / λ, Val(9))
end

# The six Voigt pairs of a symmetric 3×3 tensor, and the 21 pairs (I ≤ J) of
# those in row order: the independent components of a Hill tensor.
const _VOIGT_IJ = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))
const _VOIGT_UPPER21 = Tuple((I, J) for I in 1:6 for J in I:6)

"""
    _hill_integrand21(iK, ζ, T) -> SVector{21, T}

The integrand of the Hill tensor at the direction ``\\underline{\\zeta}``, from the
inverse acoustic tensor `iK` (as returned by [`_sym3_inv_acoustic`](@ref)):

```math
\\Gamma_{ijkl} = \\tfrac14\\,\\big(\\zeta_i\\,(K^{-1}_{jk}\\,\\zeta_l + K^{-1}_{jl}\\,\\zeta_k)
+ \\zeta_j\\,(K^{-1}_{ik}\\,\\zeta_l + K^{-1}_{il}\\,\\zeta_k)\\big) ,
```

its 21 independent components in the order of `_VOIGT_UPPER21`, on the stack:
one call per quadrature node, so a heap vector here was an allocation per node.
Shared by the cubatures of the ellipsoid and the cylinder.
"""
@inline function _hill_integrand21(iK, ζ, ::Type{T}) where {T}
    return SVector{21, T}(
        ntuple(Val(21)) do n
            I, J = _VOIGT_UPPER21[n]
            i, j = _VOIGT_IJ[I]
            k, l = _VOIGT_IJ[J]
            T(0.25) * (
                ζ[i] * (iK[j + (k - 1) * 3] * ζ[l] + iK[j + (l - 1) * 3] * ζ[k]) +
                    ζ[j] * (iK[i + (k - 1) * 3] * ζ[l] + iK[i + (l - 1) * 3] * ζ[k])
            )
        end
    )
end

# The 81 entries of a Hill tensor from its 21 independent components.
function _fill_hill21!(A::AbstractArray{<:Any, 4}, vals)
    for (n, (I, J)) in enumerate(_VOIGT_UPPER21)
        i, j = _VOIGT_IJ[I]
        k, l = _VOIGT_IJ[J]
        MFH_Core._fill_sym4!(A, i, j, k, l, vals[n])
    end
    return A
end

# `quadgk` over the segments of `points`, for the rare calls that need
# breakpoints (see the boundary layer below).
@noinline _quadgk_breakpoints(f, points, opts) = QuadGK.quadgk(f, points...; opts...)

"""
    _hill_3d_aniso_nestedquadgk(ell, C₀; abstol, reltol, maxiters, kernel) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by **nested 1-D QuadGK cubature** over the
unit sphere of the general integrand of [willis1977](@citet). `kernel` gives
the Green kernel at a direction: the inverse acoustic tensor, or
[`_sym3_inv_acoustic_incompressible`](@ref) for an incompressible matrix.
ForwardDiff-compatible. Includes axis-sort and power change-of-variable for
improved convergence on flat or elongated ellipsoids.
"""
function _hill_3d_aniso_nestedquadgk(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000,
        kernel::F = _sym3_inv_acoustic
    ) where {F}
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))

    # --- Axis sort (descending) -----------------------------------------------
    # Sort semi-axes so a₁ ≥ a₂ ≥ a₃, guaranteeing η ≤ 1 and ω ≤ 1.
    # The stiffness tensor is permuted consistently so that the parametrization
    # ζ = (√(1-z²)cosφ, √(1-z²)sinφ/η, z/ω) matches the sorted frame.
    # After integration, the result is un-permuted via the inverse permutation.
    # `C₀_arr` is assigned once, already permuted: the integrand captures it,
    # and a captured variable assigned twice is boxed, every access dynamic.
    p = collect(MFH_Core._descending_order(ell.semi_axes))
    C₀_arr = _static_stiffness(T.(Array(TensND.components(C₀, ell.basis))[p, p, p, p]))

    semi_sorted = ell.semi_axes[p]
    η = semi_sorted[2] / semi_sorted[1]
    ω = semi_sorted[3] / semi_sorted[1]

    # --- Change-of-variable exponent ------------------------------------------
    # z(u) = 1 − (1 − u)^α concentrates Gauss-Kronrod nodes near z = 1
    # where the integrand varies most rapidly for flat ellipsoids (ω ≪ 1).
    # α = 1 (identity) when ω ≥ 0.1; beyond that α ≈ 2, 3, … for each decade.
    # Any α gives the same integral, so it is read on the value of ω alone: a
    # dual semi-axis keeps its derivatives in ζ, where they belong.
    ω_val = MFH_Core.primal(ω)
    α = max(1.0, -log10(ω_val))

    # --- Boundary layer at z = 0 ----------------------------------------------
    # A flat ellipsoid's integrand also varies over a layer z ~ ω near z = 0,
    # which the first Gauss-Kronrod nodes miss once ω < 1e-3: QuadGK then stops
    # with an error far above the one asked for (2e-5 relative at ω = 1e-5 for
    # `reltol = 1e-6`). Breakpoints at z = ω, 10ω and 100ω give the layer
    # intervals of its own; read on the value of ω, as α is.
    u_breaks = ω_val < 1.0e-3 ?
        map(z -> 1.0 - (1.0 - z)^(1 / α), (ω_val, 10ω_val, 100ω_val)) : ()

    # The quadrature variables stay plain numbers, and ζ takes the derivatives of
    # the semi-axes from η and ω alone. Promoted to a dual first, √(1 − z²) had a
    # derivative 0·∞ = NaN at the node where 1 − z² rounds to zero, which a flat
    # ellipsoid (ω ≲ 5e-6) reaches; and 1 − z² is formed as (1 − z)(1 + z) from
    # the exact 1 − z = (1 − u)^α, where 1 − z² lost every digit near z = 1.
    function integrand_at(z, one_minus_z, φ)
        uz = sqrt(one_minus_z * (1 + z))
        ζ = (uz * cos(φ), uz * sin(φ) / η, z / ω)
        return _hill_integrand21(kernel(C₀_arr, ζ), ζ, T)
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    function outer(u)
        one_minus_u = 1.0 - u
        one_minus_z = one_minus_u^α
        z = 1.0 - one_minus_z
        jac = α * one_minus_u^(α - 1.0)
        inner, _ = MFH_Core._counted_quadgk(
            φ -> integrand_at(z, one_minus_z, φ),
            0.0, 2π;
            atol = abstol / 10,
            rtol = reltol
        )
        return inner .* jac
    end
    # The boundary layer of a flat ellipsoid adds breakpoints, hence a second
    # instance of QuadGK: behind an inference barrier, it is compiled only when
    # such an ellipsoid comes, not with every first call of this function. The
    # result type is asserted, so that the common branch stays inferred.
    R = Tuple{SVector{21, promote_type(T, Float64)}, Float64}
    opts = (; atol = abstol, rtol = reltol, maxevals = maxiters, norm = MFH_Core._quadgk_norm)
    P_vals, _ = isempty(u_breaks) ? QuadGK.quadgk(outer, 0.0, 1.0; opts...) :
        _quadgk_breakpoints(Base.inferencebarrier(outer), (0.0, u_breaks..., 1.0), opts)::R
    P_vals = P_vals ./ T(2π)

    # --- Build result in sorted frame -----------------------------------------
    P_perm = _fill_hill21!(zeros(T, 3, 3, 3, 3), P_vals)

    # --- Un-permute back to original axis ordering ----------------------------
    # P[q[i], q[j], q[k], q[l]] = P_perm[i, j, k, l] with q = invperm(p), that is
    # P = P_perm[p, p, p, p].
    return MFH_Core._tens_in(P_perm[p, p, p, p], ell.basis)
end
