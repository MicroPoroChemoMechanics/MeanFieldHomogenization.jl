# =============================================================================
#  auxiliary_tensors.jl
#
#  2nd- and 4th-order auxiliary tensors `I^A`, `U^A`, `V^A` for an
#  ellipsoid `A`. Low-level helpers (`_basis_col`, `_make_ortho`,
#  `_fill_sym4!`) are routed through the `Core` sub-module.
# =============================================================================

# ── tens_IA ───────────────────────────────────────────────────────────────────

"""
    tens_IA(ell::Ellipsoid{3}) -> AbstractTens{2,3}
    tens_IA(ell::Ellipsoid{2}) -> AbstractTens{2,2}

Newton-potential geometric tensor ``\\mathbf I^{\\mathbf A}`` of an
ellipsoid, defined by

```
I^A = (det A)/(4π) ∫_{|ξ|=1} ξ⊗ξ / ‖A·ξ‖³ dS_ξ .
```

``\\mathbf I^{\\mathbf A}`` is symmetric with the same eigenvectors as
``\\mathbf A`` and with diagonal components ``I_i^{\\mathbf A}``
satisfying ``\\sum_i I_i^{\\mathbf A} = 1`` (see [kellogg1929](@cite), [eshelby1957](@cite),
[parnell2016](@cite)).
"""
function tens_IA(ell::Ellipsoid{3, Spherical})
    T = eltype(ell.semi_axes)
    return TensND.TensISO{3}(T(1) / 3)
end

function tens_IA(ell::Ellipsoid{3})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    Iv, _ = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4 * T(π)
    IA_arr = zeros(T, 3, 3)
    IA_arr[1, 1] = Iv[1] / fac
    IA_arr[2, 2] = Iv[2] / fac
    IA_arr[3, 3] = Iv[3] / fac
    return TensND.Tens(IA_arr, ell.basis)
end

function tens_IA(cyl::Cylinder)
    T = eltype(cyl.semi_axes)
    b, c = cyl.semi_axes
    Iv, _ = MFH_Core.newton_potential_3d_cylinder(b, c)
    fac = 4 * T(π)
    IA_arr = zeros(T, 3, 3)
    IA_arr[1, 1] = Iv[1] / fac
    IA_arr[2, 2] = Iv[2] / fac
    IA_arr[3, 3] = Iv[3] / fac
    return TensND.Tens(IA_arr, cyl.basis)
end

function tens_IA(ell::Ellipsoid{2, Circular})
    T = eltype(ell.semi_axes)
    return TensND.TensISO{2}(T(1) / 2)
end

function tens_IA(ell::Ellipsoid{2})
    T = eltype(ell.semi_axes)
    a, b = ell.semi_axes
    Iv = MFH_Core.newton_potential_2d(a, b)
    fac = 2 * T(π)
    IA_arr = zeros(T, 2, 2)
    IA_arr[1, 1] = Iv[1] / fac
    IA_arr[2, 2] = Iv[2] / fac
    return TensND.Tens(IA_arr, ell.basis)
end

# ── tens_UA ───────────────────────────────────────────────────────────────────

"""
    tens_UA(ell::Ellipsoid{3}) -> AbstractTens{4,3}
    tens_UA(ell::Ellipsoid{2}) -> AbstractTens{4,2}

4th-order Newton-potential geometric tensor ``\\mathbb U^{\\mathbf A}``:

```
U^A = (det A)/(4π) ∫_{|ξ|=1} ξ⊗ξ⊗ξ⊗ξ / ‖A·ξ‖³ dS_ξ .
```

In the principal frame, the non-zero Kelvin–Mandel components are

```
U^A_{iiii} = 3(Iᵢ − ρᵢ² Iᵢᵢ)/2
U^A_{iijj} = U^A_{ijij} = U^A_{ijji} = (Iⱼ − ρᵢ² Iᵢⱼ)/2 ,   i≠j
```

with the ``I_i^{\\mathbf A}`` and ``I_{ij}^{\\mathbf A}`` coefficients
given in the Echoes appendix (tables ``I_i`` / ``I_{ij}``).
"""
function tens_UA(ell::Ellipsoid{3, Spherical})
    T = eltype(ell.semi_axes)
    return TensND.TensISO{3}(T(1) / 3, T(2) / 15)
end

function tens_UA(ell::Ellipsoid{3, Prolate})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    a2, b2 = a * a, b * b
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac
    I11, I22, I23, I12 = IIv[1] / fac, IIv[2] / fac, IIv[4] / fac, IIv[6] / fac
    u1 = 3 * (I1 - a2 * I11) / 2
    u2 = 3 * (I2 - b2 * I22) / 2 + (I2 - b2 * I23) / 2
    u3 = sqrt(T(2)) * (I2 - a2 * I12) / 2
    u5 = 3 * (I2 - b2 * I22) / 2 - (I2 - b2 * I23) / 2
    u6 = I2 - a2 * I12
    return TensND.TensTI{4}(u1, u2, u3, u5, u6, MFH_Core._basis_col(ell.basis, 1))
end

function tens_UA(ell::Ellipsoid{3, Oblate})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    a2, c2 = a * a, c * c
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I3 = Iv[1] / fac, Iv[3] / fac
    I11, I33, I13, I12 = IIv[1] / fac, IIv[3] / fac, IIv[5] / fac, IIv[6] / fac
    u1 = 3 * (I3 - c2 * I33) / 2
    u2 = 3 * (I1 - a2 * I11) / 2 + (I1 - a2 * I12) / 2
    u3 = sqrt(T(2)) * (I3 - a2 * I13) / 2
    u5 = 3 * (I1 - a2 * I11) / 2 - (I1 - a2 * I12) / 2
    u6 = I3 - a2 * I13
    return TensND.TensTI{4}(u1, u2, u3, u5, u6, MFH_Core._basis_col(ell.basis, 3))
end

function tens_UA(ell::Ellipsoid{3, Triaxial})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    a2, b2 = a * a, b * b
    Iv, IIv = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2, I3 = Iv[1] / fac, Iv[2] / fac, Iv[3] / fac
    I11, I22, I33 = IIv[1] / fac, IIv[2] / fac, IIv[3] / fac
    I23, I13, I12 = IIv[4] / fac, IIv[5] / fac, IIv[6] / fac
    C11 = 3 * (I1 - a2 * I11) / 2;     C22 = 3 * (I2 - b2 * I22) / 2;  C33 = 3 * (I3 - c * c * I33) / 2
    C12 = (I2 - a2 * I12) / 2;        C13 = (I3 - a2 * I13) / 2;    C23 = (I3 - b2 * I23) / 2
    C44 = C23;                     C55 = C13;                  C66 = C12
    return MFH_Core._make_ortho(
        T, C11, C22, C33, C12, C13, C23, C44, C55, C66,
        nothing, ell.basis
    )
end

function tens_UA(cyl::Cylinder{CircularCylindrical})
    T = eltype(cyl.semi_axes)
    b = cyl.semi_axes[1]
    b2 = b * b
    Iv, IIv = MFH_Core.newton_potential_3d_cylinder(b, cyl.semi_axes[2])
    fac = 4 * T(π)
    I2 = Iv[2] / fac
    I22 = IIv[2] / fac
    I23 = IIv[4] / fac
    u1 = zero(T)
    u2 = 3 * (I2 - b2 * I22) / 2 + (I2 - b2 * I23) / 2
    u3 = zero(T)
    u5 = 3 * (I2 - b2 * I22) / 2 - (I2 - b2 * I23) / 2
    u6 = zero(T)
    return TensND.TensTI{4}(u1, u2, u3, u5, u6, MFH_Core._basis_col(cyl.basis, 1))
end

function tens_UA(cyl::Cylinder{EllipticCylindrical})
    T = eltype(cyl.semi_axes)
    b, c = cyl.semi_axes
    b2, c2 = b * b, c * c
    Iv, IIv = MFH_Core.newton_potential_3d_cylinder(b, c)
    fac = 4 * T(π)
    I2, I3 = Iv[2] / fac, Iv[3] / fac
    I22, I33 = IIv[2] / fac, IIv[3] / fac
    I23 = IIv[4] / fac
    C11 = zero(T)
    C22 = 3 * (I2 - b2 * I22) / 2
    C33 = 3 * (I3 - c2 * I33) / 2
    C12 = zero(T)
    C13 = zero(T)
    C23 = (I3 - b2 * I23) / 2
    C44 = C23
    C55 = C13
    C66 = C12
    return MFH_Core._make_ortho(
        T, C11, C22, C33, C12, C13, C23, C44, C55, C66,
        nothing, cyl.basis
    )
end

function tens_UA(ell::Ellipsoid{2})
    T = eltype(ell.semi_axes)
    a, b = ell.semi_axes
    a2, b2 = a * a, b * b
    Iv = MFH_Core.newton_potential_2d(a, b)
    fac = 2 * T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac

    is_circle = T <: Real ? (a - b) ≤ a * (1.0e-6 * one(T)) : isequal(a, b)
    I12 = is_circle ? I1 / (4 * a2) : (I2 - I1) / (a2 - b2)
    I11 = (I1 - b2 * I12) / (2 * a2)
    I22 = (I2 - a2 * I12) / (2 * b2)

    U_arr = zeros(T, 2, 2, 2, 2)
    U_arr[1, 1, 1, 1] = 3 * (I1 - a2 * I11) / 2
    U_arr[2, 2, 2, 2] = 3 * (I2 - b2 * I22) / 2
    v12 = (I2 - a2 * I12) / 2
    for idx in ((1, 1, 2, 2), (2, 2, 1, 1), (1, 2, 1, 2), (2, 1, 2, 1), (1, 2, 2, 1), (2, 1, 1, 2))
        U_arr[idx...] = v12
    end
    return TensND.Tens(U_arr, ell.basis)
end

# ── tens_VA ───────────────────────────────────────────────────────────────────

"""
    tens_VA(ell::Ellipsoid{3}) -> AbstractTens{4,3}
    tens_VA(ell::Ellipsoid{2}) -> AbstractTens{4,2}

4th-order Newton-potential geometric tensor ``\\mathbb V^{\\mathbf A}``:

```
V^A = (det A)/(4π) ∫_{|ξ|=1} ξ ⊗ˢ 1 ⊗ˢ ξ / ‖A·ξ‖³ dS_ξ
    = (1 ⊠ˢ I^A + I^A ⊠ˢ 1)/2 .
```

In the principal frame, ``V^{\\mathbf A}_{iiii} = I_i^{\\mathbf A}``
and ``V^{\\mathbf A}_{ijij} = V^{\\mathbf A}_{ijji}
= (I_i^{\\mathbf A}+I_j^{\\mathbf A})/4`` for ``i\\ne j``.
"""
function tens_VA(ell::Ellipsoid{3, Spherical})
    T = eltype(ell.semi_axes)
    return TensND.TensISO{3}(T(1) / 3, T(1) / 3)
end

function tens_VA(ell::Ellipsoid{3, Prolate})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    Iv, _ = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac
    return TensND.TensTI{4}(I1, I2, zero(T), I2, (I1 + I2) / 2, MFH_Core._basis_col(ell.basis, 1))
end

function tens_VA(ell::Ellipsoid{3, Oblate})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    Iv, _ = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I3 = Iv[1] / fac, Iv[3] / fac
    return TensND.TensTI{4}(I3, I1, zero(T), I1, (I1 + I3) / 2, MFH_Core._basis_col(ell.basis, 3))
end

function tens_VA(ell::Ellipsoid{3, Triaxial})
    T = eltype(ell.semi_axes)
    a, b, c = ell.semi_axes
    Iv, _ = MFH_Core.newton_potential_3d(a, b, c)
    fac = 4T(π)
    I1, I2, I3 = Iv[1] / fac, Iv[2] / fac, Iv[3] / fac
    C11 = I1;              C22 = I2;              C33 = I3
    C12 = zero(T);         C13 = zero(T);         C23 = zero(T)
    C44 = (I2 + I3) / 4;  C55 = (I1 + I3) / 4;  C66 = (I1 + I2) / 4
    return MFH_Core._make_ortho(
        T, C11, C22, C33, C12, C13, C23, C44, C55, C66,
        nothing, ell.basis
    )
end

function tens_VA(cyl::Cylinder{CircularCylindrical})
    T = eltype(cyl.semi_axes)
    Iv, _ = MFH_Core.newton_potential_3d_cylinder(cyl.semi_axes[1], cyl.semi_axes[2])
    fac = 4 * T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac
    return TensND.TensTI{4}(I1, I2, zero(T), I2, (I1 + I2) / 2, MFH_Core._basis_col(cyl.basis, 1))
end

function tens_VA(cyl::Cylinder{EllipticCylindrical})
    T = eltype(cyl.semi_axes)
    Iv, _ = MFH_Core.newton_potential_3d_cylinder(cyl.semi_axes[1], cyl.semi_axes[2])
    fac = 4 * T(π)
    I1, I2, I3 = Iv[1] / fac, Iv[2] / fac, Iv[3] / fac
    C11 = I1;              C22 = I2;              C33 = I3
    C12 = zero(T);         C13 = zero(T);         C23 = zero(T)
    C44 = (I2 + I3) / 4;   C55 = (I1 + I3) / 4;   C66 = (I1 + I2) / 4
    return MFH_Core._make_ortho(
        T, C11, C22, C33, C12, C13, C23, C44, C55, C66,
        nothing, cyl.basis
    )
end

function tens_VA(ell::Ellipsoid{2})
    T = eltype(ell.semi_axes)
    a, b = ell.semi_axes
    Iv = MFH_Core.newton_potential_2d(a, b)
    fac = 2 * T(π)
    I1, I2 = Iv[1] / fac, Iv[2] / fac

    V_arr = zeros(T, 2, 2, 2, 2)
    V_arr[1, 1, 1, 1] = I1
    V_arr[2, 2, 2, 2] = I2
    v = (I1 + I2) / 4
    V_arr[1, 2, 1, 2] = V_arr[2, 1, 1, 2] = v
    V_arr[1, 2, 2, 1] = V_arr[2, 1, 2, 1] = v

    return TensND.Tens(V_arr, ell.basis)
end
