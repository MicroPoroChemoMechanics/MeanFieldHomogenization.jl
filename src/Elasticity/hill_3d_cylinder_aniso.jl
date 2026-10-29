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
polynomial of [masson2008](@citet) degenerates (one
root runs to infinity) so the residue path is replaced by a direct
QuadGK quadrature of the integrand of [willis1977](@citet)
[mura1987; §11.22](@cite). ForwardDiff-compatible.
"""
function _hill_3d_cylinder_aniso(
        cyl::Cylinder, C₀;
        abstol::Float64 = 1.0e-8,
        reltol::Float64 = 1.0e-6,
        maxiters::Int = 1_000_000
    )
    T = promote_type(eltype(cyl.semi_axes), eltype(C₀))
    b, c = cyl.semi_axes

    C₀_arr = _static_stiffness(T.(Array(TensND.components(C₀, cyl.basis))))   # in the cylinder's frame

    function integrand_at(φ)
        ζ = (zero(T), cos(T(φ)) / b, sin(T(φ)) / c)

        # Symmetric acoustic tensor + closed-form inverse (see
        # `_sym3_inv_acoustic` in hill_3d_aniso_nestedquadgk.jl): 6 independent
        # entries instead of 9, no `Matrix` allocation and no LU per node.
        return _hill_integrand21(_sym3_inv_acoustic(C₀_arr, ζ), ζ, T)
    end

    MFH_Core._bump!(MFH_Core.QUADGK_OUTER)
    P_vals, _ = MFH_Core._counted_quadgk(
        integrand_at, 0.0, 2π;
        atol = abstol, rtol = reltol, maxevals = maxiters
    )
    P_vals = P_vals ./ T(2π)

    P_arr = _fill_hill21!(zeros(T, 3, 3, 3, 3), P_vals)

    return MFH_Core._tens_in(P_arr, cyl.basis)
end
