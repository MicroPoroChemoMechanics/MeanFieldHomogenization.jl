# =============================================================================
#  hill_3d_iso.jl — Hill tensor for a 3-D ellipsoid in an isotropic matrix.
#  Low-level helpers live in Core.
# =============================================================================

"""
    _hill_3d_iso(ell::Ellipsoid{3}, C₀::TensISO{4,3}) -> AbstractTens{4,3}

Analytical Hill polarization tensor ``\\mathbb P`` for a 3-D ellipsoid
in an isotropic matrix ``\\mathbb C_0 = 3k\\,\\mathbb J + 2\\mu\\,\\mathbb K
= 3\\lambda\\,\\mathbb I + 2\\mu\\,\\mathbb K``:

```
P(A, 3λI + 2μK) = U^A/(λ+2μ) + (V^A − U^A)/μ .
```

Uses the Kelvin–Mandel forms of ``\\mathbb U^{\\mathbf A}`` and
``\\mathbb V^{\\mathbf A}`` (see [`tens_UA`](@ref), [`tens_VA`](@ref))
and produces the most specific TensND type compatible with the
ellipsoid symmetry ([willis1977](@cite),
[mura1987](@cite)).
"""
function _hill_3d_iso(ell::Ellipsoid{3, Spherical}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    return TensND.TensISO{3}(
        inv_lm / 3,
        d * T(2) / 15 + inv_m * T(1) / 3
    )
end

function _hill_3d_iso(ell::Ellipsoid{3, Prolate}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    a, b, c = ell.semi_axes
    a2, b2 = a * a, b * b
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac
    I11, I22 = IIv[1] / fac, IIv[2] / fac
    I23, I12 = IIv[4] / fac, IIv[6] / fac
    u1 = 3 * (I1 - a2 * I11) / 2
    u2 = 3 * (I2 - b2 * I22) / 2 + (I2 - b2 * I23) / 2
    u3 = sqrt(T(2)) * (I2 - a2 * I12) / 2
    u5 = 3 * (I2 - b2 * I22) / 2 - (I2 - b2 * I23) / 2
    u6 = I2 - a2 * I12
    p1 = d * T(u1) + inv_m * T(I1)
    p2 = d * T(u2) + inv_m * T(I2)
    p3 = d * T(u3)
    p5 = d * T(u5) + inv_m * T(I2)
    p6 = d * T(u6) + inv_m * (T(I1) + T(I2)) / 2
    return TensND.TensTI{4}(p1, p2, p3, p5, p6, MFH_Core._basis_col(ell.basis, 1))
end

function _hill_3d_iso(ell::Ellipsoid{3, Oblate}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    a, b, c = ell.semi_axes
    a2, c2 = a * a, c * c
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I3 = Iv[1] / fac, Iv[3] / fac
    I11, I33 = IIv[1] / fac, IIv[3] / fac
    I13, I12 = IIv[5] / fac, IIv[6] / fac
    u1 = 3 * (I3 - c2 * I33) / 2
    u2 = 3 * (I1 - a2 * I11) / 2 + (I1 - a2 * I12) / 2
    u3 = sqrt(T(2)) * (I3 - a2 * I13) / 2
    u5 = 3 * (I1 - a2 * I11) / 2 - (I1 - a2 * I12) / 2
    u6 = I3 - a2 * I13
    p1 = d * T(u1) + inv_m * T(I3)
    p2 = d * T(u2) + inv_m * T(I1)
    p3 = d * T(u3)
    p5 = d * T(u5) + inv_m * T(I1)
    p6 = d * T(u6) + inv_m * (T(I1) + T(I3)) / 2
    return TensND.TensTI{4}(p1, p2, p3, p5, p6, MFH_Core._basis_col(ell.basis, 3))
end

function _hill_3d_iso(ell::Ellipsoid{3, Triaxial}, C₀)
    T = promote_type(eltype(ell.semi_axes), eltype(C₀))
    α, β = C₀.data
    inv_lm = T(3) / (α + 2β)
    inv_m = T(2) / β
    d = inv_lm - inv_m
    a, b, c = ell.semi_axes
    a2, b2, c2 = a * a, b * b, c * c
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2, I3 = Iv[1] / fac, Iv[2] / fac, Iv[3] / fac
    I11, I22, I33 = IIv[1] / fac, IIv[2] / fac, IIv[3] / fac
    I23, I13, I12 = IIv[4] / fac, IIv[5] / fac, IIv[6] / fac
    ua11 = 3 * (I1 - a2 * I11) / 2;  ua22 = 3 * (I2 - b2 * I22) / 2;  ua33 = 3 * (I3 - c2 * I33) / 2
    ua12 = (I2 - a2 * I12) / 2;    ua13 = (I3 - a2 * I13) / 2;    ua23 = (I3 - b2 * I23) / 2
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
        nothing, ell.basis
    )
end
