# =============================================================================
#  hill_3d_cylinder_iso.jl — Hill tensor for an infinite cylinder in an
#  isotropic matrix.  Closed-form formulas obtained by passing to the limit
#  `a → ∞` of [`_hill_3d_iso(::Ellipsoid{3, Prolate/Triaxial}, …)`](@ref),
#  using the cylinder Newton potentials
#  [`Core.newton_potential_3d_cylinder`](@ref).
#
#  Key point: the axial `u_{11}`, `u_{12}`, `u_{13}` Walpole / ortho
#  auxiliaries are identically zero in the cylinder limit (since
#  `a² · I_{1j} → I_j` cancels `I_j`), so we *do not* compute them from
#  the generic formula `(I_i − a² · I_{ij})` — we force them to zero
#  directly.
# =============================================================================

"""
    _hill_3d_iso(cyl::Cylinder, C₀::TensISO{4,3}) -> AbstractTens{4,3}

Analytical Hill tensor of an infinite elliptic (or circular) cylinder
of transverse semi-axes ``b\\ge c>0`` embedded in an isotropic matrix
``\\mathbb C_0 = 3k\\,\\mathbb J + 2\\mu\\,\\mathbb K``, obtained by
passing to the limit ``a\\to\\infty`` of the prolate spheroid
expressions ([mura1987](@cite), §11.22). In the cylinder
frame ``(\\hat{\\mathbf e}_1,\\hat{\\mathbf e}_2,\\hat{\\mathbf e}_3)``
the first row and column of ``\\mathbb P^{\\text{cyl}}`` vanish: no
polarization is transmitted along the cylinder axis.
"""
function _hill_3d_iso(cyl::Cylinder{CircularCylindrical}, C₀)
    T = promote_type(eltype(cyl.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    b = cyl.semi_axes[1]
    b2 = b * b
    Iv, IIv = MFH_Core.newton_potential_3d_cylinder(b, cyl.semi_axes[2])
    fac = 4 * T(π)
    I1 = Iv[1] / fac
    I2 = Iv[2] / fac
    I22 = IIv[2] / fac
    I23 = IIv[4] / fac
    u1 = zero(T)
    u2 = 3 * (I2 - b2 * I22) / 2 + (I2 - b2 * I23) / 2
    u3 = zero(T)
    u5 = 3 * (I2 - b2 * I22) / 2 - (I2 - b2 * I23) / 2
    u6 = zero(T)
    p1 = d * T(u1) + inv_m * T(I1)
    p2 = d * T(u2) + inv_m * T(I2)
    p3 = d * T(u3)
    p5 = d * T(u5) + inv_m * T(I2)
    p6 = d * T(u6) + inv_m * (T(I1) + T(I2)) / 2
    return TensND.TensTI{4}(p1, p2, p3, p5, p6, MFH_Core._basis_col(cyl.basis, 1))
end

function _hill_3d_iso(cyl::Cylinder{EllipticCylindrical}, C₀)
    T = promote_type(eltype(cyl.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    b, c = cyl.semi_axes
    b2, c2 = b * b, c * c
    Iv, IIv = MFH_Core.newton_potential_3d_cylinder(b, c)
    fac = 4 * T(π)
    I1, I2, I3 = Iv[1] / fac, Iv[2] / fac, Iv[3] / fac
    I22, I33 = IIv[2] / fac, IIv[3] / fac
    I23 = IIv[4] / fac
    ua11 = zero(T)
    ua22 = 3 * (I2 - b2 * I22) / 2
    ua33 = 3 * (I3 - c2 * I33) / 2
    ua12 = zero(T)
    ua13 = zero(T)
    ua23 = (I3 - b2 * I23) / 2
    C11 = d * T(ua11) + inv_m * T(I1)
    C22 = d * T(ua22) + inv_m * T(I2)
    C33 = d * T(ua33) + inv_m * T(I3)
    C12 = d * T(ua12)
    C13 = d * T(ua13)
    C23 = d * T(ua23)
    C44 = d * T(ua23) + inv_m * (T(I2) + T(I3)) / 4
    C55 = d * T(ua13) + inv_m * (T(I1) + T(I3)) / 4
    C66 = d * T(ua12) + inv_m * (T(I1) + T(I2)) / 4
    return MFH_Core._make_ortho(
        T, C11, C22, C33, C12, C13, C23, C44, C55, C66,
        nothing, cyl.basis
    )
end
