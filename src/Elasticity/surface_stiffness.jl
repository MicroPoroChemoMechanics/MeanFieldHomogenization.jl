# =============================================================================
#  surface_stiffness.jl — the *equivalent particle* of Dormieux, Lemarchand &
#  Brisard (2016): a nanoinclusion together with its Gurtin-Murdoch interface
#  stresses, condensed into a single stiffness tensor.
#
#  When the interface energy is of the same order as the bulk energy — the
#  nanocomposite regime — the stress vector is discontinuous across the
#  particle boundary, and the polarization field of the Lippmann-Schwinger
#  equation acquires a singular part carried by that boundary.  Averaging it
#  over the particle turns the surface term into an ordinary stiffness,
#
#       ℂ^int = (1/|P|) ∫_{∂P} ℂ_s(n) dS ,        ℂ^eq = ℂ_I + ℂ^int ,
#
#  after which the strain concentration rule is *formally identical* to the
#  classical one.  That is the whole point of the paper, and the reason this
#  file adds no scheme: `ℂ^eq` drops into Mori-Tanaka — or any other estimate
#  of the package — unchanged.
#
#  Two consequences worth keeping in mind:
#    * ℂ^int scales as 1/size, so the effect is a genuine size effect and
#      disappears for large particles;
#    * for a spheroid it is transversely isotropic about the symmetry axis,
#      so it is carried exactly by a five-parameter `TensTI{4}`.
#
#  NOTE on the paper's Eq. (73).  The 6×6 array printed there for the spherical
#  limit carries 2(6μs+κs)/(5R) in its shear slots, i.e. twice the tensor
#  component C₁₃₁₃.  The intrinsic form of its Eq. (74),
#  ℂ^int = 2(6μs+κs)/(5R) 𝕂 + (4κs/R) 𝕁, is consistent with the general
#  Eq. (72) — both give C₁₃₁₃ = (κs+6μs)/(5R) — so this implementation follows
#  Eqs. (72) and (74), and the tests check that agreement.
# =============================================================================

"""
    surface_stiffness(spheroid, κs, μs) -> TensTI{4}

Average surface stiffness ``\\mathbb{C}^{int}`` of a spheroidal nanoinclusion
carrying a Gurtin-Murdoch interface of surface bulk modulus `κs` and surface
shear modulus `μs`
([dormieux2016](@cite)).

`spheroid` is an `Ellipsoid` of revolution with in-plane semi-axis `a` and
polar semi-axis `c`; the aspect ratio is ``X = c/a``, oblate for ``X < 1`` and
prolate for ``X > 1``. The result is transversely isotropic about the symmetry
axis, with

```math
\\mathbb{C}^{int}_{1111} = \\frac{3X}{16a}(\\kappa_s+\\mu_s)
  \\left[\\frac{3X^4-4X^2-8}{(X^2-1)^{5/2}}\\arctan\\sqrt{X^2-1}
       + \\frac{8-2X^2+3X^4}{X^2(X^2-1)^2}\\right],
```

and four companion components given in the same reference. `κs` and `μs` have
the dimension of a stiffness times a length, so ``\\mathbb{C}^{int}`` scales as
`1/a`: the stiffening it produces is a **size** effect, controlled by the
smallest dimension of the particle.

Adding it to the bulk stiffness gives the *equivalent particle* of the paper,
[`equivalent_particle`](@ref), which any scheme of the package accepts as an
ordinary inclusion property.

Limiting cases are handled exactly: the spherical case ``X = 1`` — where the
closed form has a removable singularity — is evaluated from its Taylor series,
and reproduces the isotropic tensor
``2(6\\mu_s+\\kappa_s)/(5R)\\,\\mathbb K + (4\\kappa_s/R)\\,\\mathbb{J}`` of the
same reference.
"""
function surface_stiffness(ell::Ellipsoid{3}, κs, μs)
    a, c, axis = _spheroid_axes(ell)
    X = c / a
    C1111, C3333, C1133, C1122, C1313 = _surface_components(X, a, κs, μs)
    # Walpole parameters of a transversely isotropic tensor, same mapping as
    # everywhere else in the package: p₁ = C₃₃₃₃, p₂ = C₁₁₁₁+C₁₁₂₂,
    # p₃ = √2 C₁₁₃₃, p₅ = C₁₁₁₁-C₁₁₂₂, p₆ = 2 C₁₃₁₃.
    T = promote_type(typeof(a), typeof(κs), typeof(μs))
    return TensND.TensTI{4}(
        C3333, C1111 + C1122, sqrt(T(2)) * C1133, C1111 - C1122, 2 * C1313, axis
    )
end

"""
    equivalent_particle(C_I, spheroid, κs, μs) -> AbstractTens

Stiffness of the *equivalent particle* — the nanoinclusion `C_I` together with
its interface — of [dormieux2016](@cite):

```math
\\mathbb{C}^{eq}_I = \\mathbb{C}_I + \\mathbb{C}^{int} .
```

The paper's central result is that the strain concentration rule and the
homogenized stiffness keep their classical form once this substitution is made,
so a nanocomposite estimate needs no new scheme:

```julia
C_eq = equivalent_particle(C_i, Spheroid(0.1), κs, μs)
add_phase!(rve, :nano, Spheroid(0.1), Dict(:C => C_eq); fraction = f)
homogenize(rve, MoriTanaka(), :C)
```

Note that the strain average rule is unaffected by the interface, but the
*stress* average rule is not: the interface contributes ``\\mathbb{C}^{int}:
\\varepsilon_I`` to the macroscopic stress, which is precisely what using
``\\mathbb{C}^{eq}`` in the scheme accounts for.
"""
equivalent_particle(C_I::TensND.AbstractTens{4, 3}, ell::Ellipsoid{3}, κs, μs) =
    C_I + surface_stiffness(ell, κs, μs)

# ─── Components ──────────────────────────────────────────────────────────────

# Away from the sphere the closed form is evaluated directly.  The combination
# `atan(√u)/u^{5/2}` continues analytically to u < 0 as `atanh(√-u)/(-u)^{5/2}`
# — the oblate branch — so one real function covers both sides.
function _surface_components(X, a, κs, μs)
    u = X^2 - 1
    abs(u) < 1.0e-3 && return _surface_components_near_sphere(X, a, κs, μs)
    A = _atan_branch(u)
    u2 = u^2
    X2, X4 = X^2, X^4
    pre = 3 * X / a
    C1111 = pre / 16 * (κs + μs) * ((3X4 - 4X2 - 8) * A + (8 - 2X2 + 3X4) / (X2 * u2))
    C3333 = pre / 2 * (κs + μs) * (X2 * (X2 - 4) * A + (2 + X2) / u2)
    C1133 = pre / 4 * (
        (X4 * (κs - μs) + 2X2 * (2μs - κs) + 4κs) * A + (X2 * (κs - μs) - 2 * (2κs + μs)) / u2
    )
    C1122 = pre / 16 * (
        (X4 * (μs + κs) + 4X2 * (κs - 3μs) + 8 * (μs - κs)) * A +
            (X4 * (μs + κs) + 2X2 * (5μs - 3κs) + 8 * (κs - μs)) / (X2 * u2)
    )
    C1313 = pre / 4 * (
        (X4 * μs + X2 * (κs - 2μs) + 2 * (κs + 2μs)) * A +
            (X2 * μs - (3κs + 4μs)) / u2
    )
    return C1111, C3333, C1133, C1122, C1313
end

# `arctan(√u)/u^{5/2}`, continued to the oblate branch u < 0.
_atan_branch(u) = u > 0 ? atan(sqrt(u)) / u^2 / sqrt(u) :
    atanh(sqrt(-u)) / (-u)^2 / sqrt(-u)

# Near X = 1 the two terms of each component diverge as u⁻² and cancel, so the
# closed form loses every significant digit.  The Taylor series in ε = X - 1 is
# used instead; three terms hold the relative error below 1e-12 over the
# switching window |X² - 1| < 1e-3, and it makes the spherical case exact.
function _surface_components_near_sphere(X, a, κs, μs)
    ε = X - 1
    s = κs + μs
    C1111 = (8s / 5 + ε * (-8s / 7) + ε^2 * (48s / 35) + ε^3 * (-1712s / 1155)) / a
    C3333 = (8s / 5 + ε * (24s / 35) + ε^2 * (-16s / 35) + ε^3 * (304s / 1155)) / a
    C1133 = (
        (6κs / 5 - 4μs / 5) + ε * (2κs / 35 - 12μs / 35) +
            ε^2 * (-4κs / 35 + 8μs / 35) + ε^3 * (52κs / 385 - 152μs / 1155)
    ) / a
    C1122 = (
        (6κs / 5 - 4μs / 5) + ε * (-46κs / 35 + 52μs / 35) +
            ε^2 * (52κs / 35 - 8μs / 5) + ε^3 * (-596κs / 385 + 1864μs / 1155)
    ) / a
    C1313 = (
        (κs / 5 + 6μs / 5) + ε * (-κs / 7 + 2μs / 35) +
            ε^2 * (2κs / 35 - 4μs / 35) + ε^3 * (2κs / 1155 + 52μs / 385)
    ) / a
    return C1111, C3333, C1133, C1122, C1313
end

# In-plane semi-axis, polar semi-axis and symmetry axis of a spheroid.  A
# triaxial ellipsoid has no transversely isotropic interface tensor, so it is
# refused rather than silently approximated by one of its axes.
function _spheroid_axes(ell::Ellipsoid{3, Spherical})
    a = ell.semi_axes[1]
    return a, a, MFH_Core._basis_col(ell.basis, 3)
end

# Prolate: semi-axes sorted descending, so the long (polar) axis is the first.
_spheroid_axes(ell::Ellipsoid{3, Prolate}) =
    (ell.semi_axes[2], ell.semi_axes[1], MFH_Core._basis_col(ell.basis, 1))

# Oblate: the short (polar) axis is the last.
_spheroid_axes(ell::Ellipsoid{3, Oblate}) =
    (ell.semi_axes[1], ell.semi_axes[3], MFH_Core._basis_col(ell.basis, 3))

_spheroid_axes(ell::Ellipsoid{3}) = throw(
    ArgumentError(
        "surface_stiffness: the interface stiffness of Dormieux et al. (2016) is " *
            "derived for spheroids; got semi-axes $(ell.semi_axes), which are " *
            "triaxial and have no transversely isotropic interface tensor."
    )
)
