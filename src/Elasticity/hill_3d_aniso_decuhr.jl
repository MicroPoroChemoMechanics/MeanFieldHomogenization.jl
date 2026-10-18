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
[espelid1994](@citet) called through `Integrals.jl`
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

    # Integrand on the unit square (z, φ) ∈ [0, 1] × [0, 2π]: the 21
    # independent components of the Hill tensor (`_hill_integrand21`). The
    # cubature variables stay plain numbers, as in the nested cubature: ζ takes
    # the derivatives of the semi-axes from η and ω alone.
    function integrand(u, _)
        z, φ = u[1], u[2]
        uz = sqrt((1 - z) * (1 + z))
        ζ = (uz * cos(φ), uz * sin(φ) / η, z / ω)
        return _hill_integrand21(_sym3_inv_acoustic(C₀_arr, ζ), ζ, T)
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

    P_arr = _fill_hill21!(zeros(T, 3, 3, 3, 3), P_vals)

    return TensND.Tens(P_arr, ell.basis)
end
