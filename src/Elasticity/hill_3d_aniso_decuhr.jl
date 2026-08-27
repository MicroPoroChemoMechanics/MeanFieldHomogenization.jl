# =============================================================================
#  hill_3d_aniso_decuhr.jl
#
#  2D adaptive cubature for the 3D anisotropic Hill tensor using the
#  DECUHR algorithm of Espelid & Genz 1994, called through the
#  `Integrals.jl` / `DECUHR.jl` stack.  Same 2D parametrization as the
#  nested-QuadGK variant in `hill_3d_aniso_nestedquadgk.jl`:
#
#     ζ(z, φ) = (√(1 - z²) cos φ, √(1 - z²) sin φ / η, z / ω)
#
#  Integration domain: (z, φ) ∈ [0, 1] × [0, 2π].  The integrand is
#  smooth (no vertex singularity), so we pass `singul = 1, alpha = 0.0`
#  to `DecuhrAlgorithm` — it degrades to ordinary adaptive Gauss.  Using
#  `alpha = 0.0` (rather than the default `-2.0` which triggers
#  auto-estimation) keeps the kernel ForwardDiff-compatible (the
#  auto-estimation path is Float64-only).
# =============================================================================

"""
    _hill_3d_aniso_decuhr(ell, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of a 3-D ellipsoid in an arbitrarily
anisotropic matrix, evaluated by the adaptive 2-D DECUHR cubature of
[espelid1994](@cite) called through `Integrals.jl`
(`DecuhrAlgorithm`).  ForwardDiff-compatible — `alpha = 0.0` is
supplied explicitly so the integrand is treated as smooth (no vertex
singularity) and `DecuhrAlgorithm` bypasses the Float64-only
auto-estimation path.
"""
function _hill_3d_aniso_decuhr(
        ell::Ellipsoid{3}, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))

    η = ell.semi_axes[2] / ell.semi_axes[1]
    ω = ell.semi_axes[3] / ell.semi_axes[1]

    C₀_princ = TensND.change_tens(C₀, ell.basis)
    C₀_arr = Array{T, 4}(undef, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C₀_arr[i, j, k, l] = C₀_princ[i, j, k, l]
    end

    voigt_ij = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

    # Integrand on the unit square (z, φ) ∈ [0, 1] × [0, 2π].
    # Returns a 21-element vector of the independent Hill-tensor
    # components (Voigt upper-triangle ordering).
    function integrand(u, _)
        z, φ = u[1], u[2]
        uz = sqrt(one(T) - T(z) * T(z))
        ζ = (uz * cos(T(φ)), uz * sin(T(φ)) / η, T(z) / ω)

        # The acoustic tensor is symmetric, so only its 6 independent entries
        # are needed, and its inverse has a closed form.  `_sym3_inv_acoustic`
        # (Elasticity/hill_3d_aniso_nestedquadgk.jl) returns the 9 entries of
        # K⁻¹ as a tuple: no `Matrix` allocation and no LU per cubature node.
        # It is a top-level function on purpose — see the comment at its
        # definition about closure boxing.
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

    # `wrksub` must be large enough to hold all adaptive subregions;
    # approximate upper bound `maxiters / 32` matches the 2D deg-13 rule
    # (65 evaluation points per subregion). The actual DECUHR call lives in
    # the `MeanFieldHomogenizationDECUHRExt` extension (see `MFH_Core._decuhr_cubature`).
    u = MFH_Core._decuhr_cubature(
        integrand, [0.0, 0.0], [1.0, 2π];
        wrksub = max(5000, maxiters ÷ 32),
        abstol = abstol, reltol = reltol, maxiters = maxiters
    )
    P_vals = u ./ T(2π)

    P_arr = zeros(T, 3, 3, 3, 3)
    idx = 0
    for I in 1:6, J in I:6
        i, j = voigt_ij[I]
        k, l = voigt_ij[J]
        idx += 1
        v = P_vals[idx]
        P_arr[i, j, k, l] = v;  P_arr[j, i, k, l] = v
        P_arr[i, j, l, k] = v;  P_arr[j, i, l, k] = v
        P_arr[k, l, i, j] = v;  P_arr[l, k, i, j] = v
        P_arr[k, l, j, i] = v;  P_arr[l, k, j, i] = v
    end

    return TensND.Tens(P_arr, ell.basis)
end
