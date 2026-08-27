# =============================================================================
#  hill_3d_cylinder_aniso.jl
#
#  ForwardDiff-compatible 1D quadrature for the 3D Hill tensor of an
#  infinite cylinder (axis = e₁) embedded in a general anisotropic matrix.
#  Single QuadGK integral over φ ∈ [0, 2π] — the transverse-plane
#  parametrization `ζ(φ) = (0, cos φ / b, sin φ / c)`.
#
#  Mirrors the structure of `_hill_3d_aniso_decuhr` but collapses the
#  (z, φ) double cubature to a single 1D integral since the cylinder
#  limit a → ∞ zeroes the axial ζ-component.
# =============================================================================

"""
    _hill_3d_cylinder_aniso(cyl, C₀; abstol, reltol, maxiters) -> AbstractTens{4,3}

Hill polarization tensor of an infinite cylinder in an arbitrarily
anisotropic matrix, evaluated by a single adaptive 1-D quadrature over
the transverse unit circle. At the cylinder limit ``a\\to\\infty`` the
Masson polynomial ([masson2008](@cite)) degenerates (one
root runs to infinity) so the residue path is replaced by a direct
QuadGK quadrature of the [willis1977](@cite) integrand
([mura1987](@cite), §11.22). ForwardDiff-compatible.
"""
function _hill_3d_cylinder_aniso(
        cyl::Cylinder, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    T = promote_type(eltype(cyl.semi_axes), eltype(C₀))
    b, c = cyl.semi_axes

    C₀_princ = TensND.change_tens(C₀, cyl.basis)
    C₀_arr = Array{T, 4}(undef, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C₀_arr[i, j, k, l] = C₀_princ[i, j, k, l]
    end

    voigt_ij = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

    function integrand_at(φ)
        ζ = (zero(T), cos(T(φ)) / b, sin(T(φ)) / c)

        # Symmetric acoustic tensor + closed-form inverse (see
        # `_sym3_inv_acoustic` in hill_3d_aniso_nestedquadgk.jl): 6 independent
        # entries instead of 9, no `Matrix` allocation and no LU per node.
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
    P_vals, _ = MFH_Core._counted_quadgk(
        integrand_at, 0.0, 2π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )
    P_vals ./= T(2π)

    P_arr = zeros(T, 3, 3, 3, 3)
    idx = 0
    for I in 1:6, J in I:6
        i, j = voigt_ij[I]
        k, l = voigt_ij[J]
        idx += 1
        v = P_vals[idx]
        P_arr[i, j, k, l] = v; P_arr[j, i, k, l] = v
        P_arr[i, j, l, k] = v; P_arr[j, i, l, k] = v
        P_arr[k, l, i, j] = v; P_arr[l, k, i, j] = v
        P_arr[k, l, j, i] = v; P_arr[l, k, j, i] = v
    end

    return TensND.Tens(P_arr, cyl.basis)
end
