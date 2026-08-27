# =============================================================================
#  hill_order2_2d.jl — 2nd-order Hill tensor (2D, conductivity / diffusion).
# =============================================================================

"""
    _hill_order2_2d_iso(ell::Ellipsoid{2}, K₀) -> AbstractTens{2,2}

2nd-order Hill polarization tensor of a 2-D ellipse in an isotropic
conductor ``\\mathbf K_0 = K\\,\\mathbf 1``, closed form
``\\mathbf P = \\mathbf I^{\\mathbf A}/K`` specialized to the plane-strain
unit circle (prefactor ``1/(2\\pi)``).
"""
function _hill_order2_2d_iso(ell::Ellipsoid{2, Circular}, K₀)
    T = promote_type(eltype(ell.semi_axes), eltype(K₀))
    k = K₀.data[1]
    return TensND.TensISO{2}(T(1) / (2 * k))
end

function _hill_order2_2d_iso(ell::Ellipsoid{2, Elliptic}, K₀)
    T = promote_type(eltype(ell.semi_axes), eltype(K₀))
    k = K₀.data[1]
    ρ = T(ell.semi_axes[2] / ell.semi_axes[1])
    P_arr = zeros(T, 2, 2)
    P_arr[1, 1] = ρ / (k * (1 + ρ))
    P_arr[2, 2] = one(T) / (k * (1 + ρ))
    return TensND.Tens(P_arr, ell.basis)
end

"""
    _hill_order2_2d(ell::Ellipsoid{2}, K₀) -> AbstractTens{2,2}

2nd-order Hill polarization tensor of a 2-D ellipse in an arbitrarily
anisotropic conductor. Obtained in closed form from the
``\\mathbf K^{-1/2}`` change-of-variable of
[giraudMOM2019](@cite) (2-D specialization);
the code falls back to the nearly-isotropic limit when the acoustic
denominator ``\\det(\\mathbf K) - k_{12}^{2}`` approaches zero.
"""
function _hill_order2_2d(ell::Ellipsoid{2}, K₀)
    T = promote_type(eltype(ell.semi_axes), eltype(K₀))
    ρ = T(ell.semi_axes[2] / ell.semi_axes[1])
    ρ2 = ρ * ρ

    K_princ = TensND.change_tens(K₀, ell.basis)
    k11 = K_princ[1, 1]; k12 = K_princ[1, 2]; k22 = K_princ[2, 2]

    t1 = k11 * k22
    t5 = k12 * k12
    t3 = ρ2 * t1
    t4 = k22 * k22
    t6 = ρ2 * t5
    t19 = k11 * k11
    t20 = ρ2 * ρ2
    den = 4 * t6 + t4 - 2 * t3 + t20 * t19

    P_arr = zeros(T, 2, 2)
    tol = T(1.0e-6)

    if abs(den) < tol * abs(t19 + t4)
        P_arr[1, 1] = 1 / (2 * k11)
        P_arr[2, 2] = 1 / (2 * k22)
    else
        t10 = sqrt(t1 - t5)
        t12 = ρ * ρ2
        t23 = 1 / den
        t24 = 1 / t10
        t28 = k11 * ρ2
        P_arr[1, 1] = t24 * t23 * (-t3 + t4 + 2 * t6 - t10 * k22 * ρ + t10 * k11 * t12) * ρ
        P_arr[1, 2] = -t24 * t23 * (k22 + t28 - 2 * t10 * ρ) * ρ * k12
        P_arr[2, 1] = P_arr[1, 2]
        P_arr[2, 2] = t24 * t23 * (t12 * t19 - ρ * k11 * k22 + 2 * t5 * ρ - t10 * t28 + t10 * k22)
    end

    return TensND.Tens(P_arr, ell.basis)
end
