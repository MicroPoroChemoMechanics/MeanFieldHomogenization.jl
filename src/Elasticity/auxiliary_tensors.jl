# =============================================================================
#  auxiliary_tensors.jl
#
#  2nd- and 4th-order auxiliary tensors `I^A`, `U^A`, `V^A` for an
#  ellipsoid `A`. Orthotropic results are TensND's `TensOrtho` in the
#  inclusion's basis; `_basis_col` comes from the `Core` sub-module.
# =============================================================================

# ── tens_IA ───────────────────────────────────────────────────────────────────

"""
    tens_IA(ell::Ellipsoid{3}) -> AbstractTens{2,3}
    tens_IA(ell::Ellipsoid{2}) -> AbstractTens{2,2}

Newton-potential geometric tensor ``\\boldsymbol{I}^{\\boldsymbol{A}}`` of an
ellipsoid, defined by

```math
\\boldsymbol{I}^{\\boldsymbol{A}}
= \\frac{\\det\\boldsymbol{A}}{4\\pi}
\\int_{\\|\\underline{\\xi}\\|=1}
\\frac{\\underline{\\xi}\\otimes\\underline{\\xi}}
     {\\|\\boldsymbol{A}\\cdot\\underline{\\xi}\\|^{3}}\\,\\mathrm{d}S_{\\xi} .
```

``\\boldsymbol{I}^{\\boldsymbol{A}}`` is symmetric with the same eigenvectors as
``\\boldsymbol{A}`` and with diagonal components ``I_i^{\\boldsymbol{A}}``
satisfying ``\\sum_i I_i^{\\boldsymbol{A}} = 1`` [kellogg1929, eshelby1957, parnell2016](@cite).
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

4th-order Newton-potential geometric tensor ``\\mathbb{U}^{\\boldsymbol{A}}``:

```math
\\mathbb{U}^{\\boldsymbol{A}}
= \\frac{\\det\\boldsymbol{A}}{4\\pi}
\\int_{\\|\\underline{\\xi}\\|=1}
\\frac{\\underline{\\xi}\\otimes\\underline{\\xi}\\otimes
      \\underline{\\xi}\\otimes\\underline{\\xi}}
     {\\|\\boldsymbol{A}\\cdot\\underline{\\xi}\\|^{3}}\\,\\mathrm{d}S_{\\xi} .
```

In the principal frame, the non-zero Kelvin–Mandel components are

```math
U^{\\boldsymbol{A}}_{iiii} = \\tfrac{3}{2}\\bigl(I_i^{\\boldsymbol{A}}-\\rho_i^{2}I_{ii}^{\\boldsymbol{A}}\\bigr),
\\qquad
U^{\\boldsymbol{A}}_{iijj} = U^{\\boldsymbol{A}}_{ijij} = U^{\\boldsymbol{A}}_{ijji}
= \\tfrac{1}{2}\\bigl(I_j^{\\boldsymbol{A}}-\\rho_i^{2}I_{ij}^{\\boldsymbol{A}}\\bigr)
\\quad (i\\ne j),
```

with the ``I_i^{\\boldsymbol{A}}`` and ``I_{ij}^{\\boldsymbol{A}}`` coefficients
given in the Echoes appendix (tables ``I_i`` / ``I_{ij}``).

In two dimensions the weight is ``\\det\\boldsymbol{A}/(2\\pi\\,\\|\\boldsymbol{A}\\cdot\\underline{\\xi}\\|^{2})``
on the unit circle, the one of [`tens_IA`](@ref) and [`tens_VA`](@ref), and an
ellipse of semi-axes ``(a, b)`` has, with ``s = a + b``,

```math
U^{\\boldsymbol{A}}_{1111} = \\frac{b\\,(a + 2b)}{2s^{2}},\\qquad
U^{\\boldsymbol{A}}_{2222} = \\frac{a\\,(2a + b)}{2s^{2}},\\qquad
U^{\\boldsymbol{A}}_{1122} = U^{\\boldsymbol{A}}_{1212} = \\frac{ab}{2s^{2}},
```

so that the Hill tensor of an isotropic medium is
``\\mathbb{U}^{\\boldsymbol{A}}/(\\lambda + 2\\mu) + (\\mathbb{V}^{\\boldsymbol{A}} - \\mathbb{U}^{\\boldsymbol{A}})/\\mu``
in both dimensions. These closed forms hold at ``a = b``.
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
    return TensND.TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, ell.basis)
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
    return TensND.TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, cyl.basis)
end

# The 2-D coefficients are those of the docstring: with the 2-D weight, the
# ``I_i``, ``I_{ij}`` of the ellipse reduce to ``b/s``, ``a/s`` and ``1/s^2``,
# so no division by ``a^2 - b^2`` (and no switch near the circle) is needed.
function tens_UA(ell::Ellipsoid{2})
    a, b = ell.semi_axes
    s2 = 2 * (a + b)^2
    u = (b * (a + 2b) / s2, a * b / s2, a * (2a + b) / s2)   # U₁₁₁₁, U₁₁₂₂, U₂₂₂₂
    # U is fully symmetric: a component depends only on how many of its
    # indices are 2 — none, two or four; an odd count gives zero.
    U_arr = [_ua_2d(count(==(2), (i, j, k, l)), u) for i in 1:2, j in 1:2, k in 1:2, l in 1:2]
    return TensND.Tens(U_arr, ell.basis)
end

_ua_2d(n2, u) = iseven(n2) ? u[n2 ÷ 2 + 1] : zero(u[1])

# ── tens_VA ───────────────────────────────────────────────────────────────────

"""
    tens_VA(ell::Ellipsoid{3}) -> AbstractTens{4,3}
    tens_VA(ell::Ellipsoid{2}) -> AbstractTens{4,2}

4th-order Newton-potential geometric tensor ``\\mathbb{V}^{\\boldsymbol{A}}``:

```math
\\mathbb{V}^{\\boldsymbol{A}}
= \\frac{\\det\\boldsymbol{A}}{4\\pi}
\\int_{\\|\\underline{\\xi}\\|=1}
\\frac{\\underline{\\xi}\\stackrel{s}{\\otimes}\\boldsymbol{1}
      \\stackrel{s}{\\otimes}\\underline{\\xi}}
     {\\|\\boldsymbol{A}\\cdot\\underline{\\xi}\\|^{3}}\\,\\mathrm{d}S_{\\xi}
= \\frac{\\boldsymbol{1}\\stackrel{s}{\\boxtimes}\\boldsymbol{I}^{\\boldsymbol{A}}
      + \\boldsymbol{I}^{\\boldsymbol{A}}\\stackrel{s}{\\boxtimes}\\boldsymbol{1}}{2} .
```

In the principal frame, ``V^{\\boldsymbol{A}}_{iiii} = I_i^{\\boldsymbol{A}}``
and ``V^{\\boldsymbol{A}}_{ijij} = V^{\\boldsymbol{A}}_{ijji}
= (I_i^{\\boldsymbol{A}}+I_j^{\\boldsymbol{A}})/4`` for ``i\\ne j``.
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
    return TensND.TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, ell.basis)
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
    return TensND.TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, cyl.basis)
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
