# =============================================================================
#  hill_3d_ti_coaxial.jl — Analytical Hill polarization tensor for a
#  spheroidal inclusion coaxial with a transversely isotropic matrix.
#
#  Implements the closed-form formula derived in
#  [Barthélémy 2020](@cite barthelemyIJES2020_hilltrans) (eqs. 49–58 of the post-print)
#  for the Walpole-basis components `(P₁, P₂, P₃, P₅, P₆)` of the Hill
#  tensor of a spheroid (prolate or oblate) whose axis is parallel to
#  the symmetry axis of a TI elastic matrix.
#
#  The five elastic constants (C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃) of
#  the TI matrix and the spheroid aspect ratio `ω = (axial)/(transverse)`
#  enter through six elementary elliptic integrals
#  (I₀, I₂, I₄, J₀, J₂, J₄) that admit closed forms in terms of `acosh`
#  and complex square roots.
#
#  The output is a `TensTI{4, T, 5}` (major-symmetric Walpole tensor)
#  with axis equal to the common symmetry axis of `C₀` and the
#  spheroid.  When the matrix is in fact isotropic, the formula
#  recovers the classical Eshelby–Mura result
#  ([Mura 1987](@cite mura1987)).
# =============================================================================

# ── Elliptic-integral helpers — single argument η ----------------------------
#
#   I₀(η) = ∫₋₁¹ dz / [z² + η² (1−z²)]
#   I₂(η) = ∫₋₁¹ z² dz / [z² + η² (1−z²)]
#   I₄(η) = ∫₋₁¹ z⁴ dz / [z² + η² (1−z²)]
#
# Closed forms when η ≠ 1, with limit values 2, 2/3, 2/5 at η → 1.

# Tolerances chosen to absorb floating-point noise from acoustic-polynomial
# discriminants that should vanish exactly (e.g. when the TI matrix collapses
# to an isotropic one).  `_HILL_TI_EPS` is used both for `η ≈ 1` (single
# argument) and `η₁ ≈ η₂` (two arguments).
const _HILL_TI_EPS = 1.0e-6

@inline function _I0(η)
    k2 = η^2 - 1
    abs(k2) < _HILL_TI_EPS && return 2 * one(η)
    return 2 * acosh(η) / (η * sqrt(k2))
end

@inline function _I2(η)
    k2 = η^2 - 1
    abs(k2) < _HILL_TI_EPS && return 2 * one(η) / 3
    return 2 * (η * acosh(η) - sqrt(k2)) / k2^(3 // 2)
end

@inline function _I4(η)
    k2 = η^2 - 1
    abs(k2) < _HILL_TI_EPS && return 2 * one(η) / 5
    return 2 * (η^3 * acosh(η) - (1 + 4 * k2 / 3) * sqrt(k2)) / k2^(5 // 2)
end

# ── Elliptic-integral helpers — two arguments η₁, η₂ -------------------------
#
#   J_n(η₁,η₂) = ∫₋₁¹ z^n dz / { [z² + η₁²(1−z²)] [z² + η₂²(1−z²)] }
#
# Coincidence rule J_n(η,η) handled separately to avoid the 0/0 form.
# `η₁ ≈ η₂` is detected with a relative tolerance to absorb round-off
# noise from a near-degenerate acoustic polynomial (e.g. iso matrix
# expressed as a TI tensor).

@inline _close_eta(η1, η2) = abs(η1^2 - η2^2) < _HILL_TI_EPS * max(abs(η1^2), abs(η2^2), 1.0)

@inline function _J0(η1, η2)
    if _close_eta(η1, η2)
        η = (η1 + η2) / 2
        k2 = η^2 - 1; sk2 = sqrt(k2)
        abs(k2) < _HILL_TI_EPS && return 2 * one(η)
        return (acosh(η) + η * sk2) / (sk2 * η^3)
    end
    e12 = η1^2; e22 = η2^2
    return ((1 - e12) * _I0(η1) - (1 - e22) * _I0(η2)) / (e22 - e12)
end

@inline function _J2(η1, η2)
    if _close_eta(η1, η2)
        η = (η1 + η2) / 2
        k2 = η^2 - 1; sk2 = sqrt(k2)
        abs(k2) < _HILL_TI_EPS && return 2 * one(η) / 3
        return (-acosh(η) + η * sk2) / (sk2^3 * η)
    end
    e12 = η1^2; e22 = η2^2
    return ((1 - e12) * _I2(η1) - (1 - e22) * _I2(η2)) / (e22 - e12)
end

@inline function _J4(η1, η2)
    if _close_eta(η1, η2)
        η = (η1 + η2) / 2
        k2 = η^2 - 1; sk2 = sqrt(k2)
        abs(k2) < _HILL_TI_EPS && return 2 * one(η) / 5
        return (-3 * η * acosh(η) + (2 + η^2) * sk2) / sk2^5
    end
    e12 = η1^2; e22 = η2^2
    return ((1 - e12) * _I4(η1) - (1 - e22) * _I4(η2)) / (e22 - e12)
end

# ── Core routine --------------------------------------------------------------

# For real T the analytical formula has complex intermediates (square roots
# of negative discriminants when the matrix is iso/near-iso, `acosh` of
# η<1 in the oblate case, etc.) but the final P_i are real by construction.
# For complex T (frequency-domain viscoelasticity, harmonic problems) the
# intermediates and the P_i are genuinely complex.  We thus pick:
#   * `TC = Complex{T}`   if `T <: Real`  — wrap to keep complex roots safe
#   * `TC = T`            otherwise       — already complex, no double-wrap
# and only strip the imaginary part on output when the inputs were real.
@inline _strip_im(x::Complex, ::Type{T}) where {T <: Real} = real(x)
@inline _strip_im(x, ::Type) = x

"""
    _hill_ti_walpole(ω, C1111, C1122, C1133, C3333, C2323)
        -> (P1, P2, P3, P5, P6)

Closed-form Walpole-basis coefficients of the Hill tensor for a spheroid
of aspect ratio `ω = (axial)/(transverse)` coaxial with a transversely
isotropic matrix specified by the five independent elastic constants.

Element type policy: with
`T = promote_type(typeof(ω), typeof(C1111), …, typeof(C2323))`,

- if `T <: Real` (standard real-modulus case), the arithmetic is carried
  out in `Complex{T}` to keep the complex roots of the acoustic
  polynomial well-defined; the imaginary parts cancel by construction
  and only the real parts are returned. Compatible with
  `ForwardDiff.Dual` numbers (the analytical Hill tensor is
  differentiable through the elastic constants and the spheroid aspect
  ratio).
- if `T` is itself complex (frequency-domain viscoelasticity, harmonic
  problems), the formula is evaluated directly in `T` and the genuinely
  complex `P_i` are returned unchanged.

When the matrix is in fact isotropic (`C1111 = C3333 = λ + 2μ`,
`C1122 = C1133 = λ`, `C2323 = μ`) the returned coefficients reduce to
the classical Mura formula.

!!! note "Symbolic numbers (SymPy Sym)"
    The function is **not** compatible with SymPy `Sym` inputs because
    Julia's `Complex{T}` parametric type requires `T <: Real`, which
    `Sym` does not satisfy. For symbolic exploration of the analytical
    formula, derive the components by hand from eqs. 49–58 of the
    paper, or use the residue/DECUHR backends with substituted
    numerical values.

Reference: [barthelemyIJES2020_hilltrans](@cite), eqs. 49–58.
"""
function _hill_ti_walpole(ω, C1111, C1122, C1133, C3333, C2323)
    T = promote_type(
        typeof(ω), typeof(C1111), typeof(C1122),
        typeof(C1133), typeof(C3333), typeof(C2323)
    )
    TC = T <: Real ? Complex{T} : T
    ωc = TC(ω)
    C11 = TC(C1111)
    C12 = TC(C1122)
    C13 = TC(C1133)
    C33 = TC(C3333)
    C44 = TC(C2323)

    om2 = ωc^2
    om4 = om2^2

    # Roots γ₁, γ₂ of  a γ⁴ + b γ² + c = 0  (acoustic polynomial)
    a = C44 * C33
    b = C13^2 + 2 * C13 * C44 - C11 * C33
    c = C11 * C44
    # Detect iso (or near-iso) values: b² − 4ac vanishes exactly when the
    # matrix is isotropic. Using `sqrt` on the discriminant in Dual
    # arithmetic at that point produces NaN partials (the analytic
    # derivative of √x at x = 0 is +∞). Branch on the *real* value of
    # the discriminant so the comparison is a plain Bool and the Dual
    # partials are propagated through the appropriate iso-coaxial limit.
    disc = b^2 - 4 * a * c
    γ1, γ2 = if abs(disc) < _HILL_TI_EPS *
            max(abs(b^2), abs(4 * a * c), one(real(TC)))
        γ_iso = sqrt(-b / (2 * a))
        (γ_iso, γ_iso)
    else
        sqd = sqrt(disc)
        (sqrt((-b + sqd) / (2 * a)), sqrt((-b - sqd) / (2 * a)))
    end

    η1 = ωc * γ1
    η2 = ωc * γ2
    η3 = ωc * sqrt((C11 - C12) / (2 * C44))

    coef = C44 * C33
    j0 = _J0(η1, η2) / coef
    j2 = _J2(η1, η2) / coef
    j4 = _J4(η1, η2) / coef
    i0 = _I0(η3) / C44
    i2 = _I2(η3) / C44

    # Walpole components (eqs. 53–58 of Barthélémy 2020)
    P1_c = ((C44 - om2 * C11) * j4 + om2 * C11 * j2) / 2
    P2_c = om2 * ((om2 * C44 - C33) * j4 + (C33 - 2 * om2 * C44) * j2 + om2 * C44 * j0) / 4
    P3_c = om2 / (2 * sqrt(TC(2))) * (C44 + C13) * (j4 - j2)
    P5_c = P2_c / 2 + om2 * (i0 - i2) / 8
    P6_c = (
        (om4 * C11 + C33 + 2 * om2 * C13) * j4
            - 2 * om2 * (om2 * C11 + C13) * j2
            + om4 * C11 * j0 + i2
    ) / 8

    return (
        _strip_im(P1_c, T),
        _strip_im(P2_c, T),
        _strip_im(P3_c, T),
        _strip_im(P5_c, T),
        _strip_im(P6_c, T),
    )
end

# ── Public builders — coaxial spheroid in TI matrix --------------------------

"""
    _hill_3d_ti_coaxial(ell, C₀) -> TensTI{4, Float64, 5}

Hill polarization tensor for a spheroidal inclusion `ell` coaxial with a
transversely isotropic matrix `C₀::TensTI{4, T, 5}`.  Both must share
the same symmetry axis; coaxiality is checked by the dispatcher
([`_ti_coaxial`](@ref)) and the `Analytical` algorithm is selected when
the test succeeds.  Otherwise the user is silently routed to the
generic residue/DECUHR backend.

# Arguments
- `ell::Ellipsoid{3, Oblate}`: oblate spheroid `a = b ≥ c`, axis `e₃`,
  aspect ratio `ω = c/a`.
- `ell::Ellipsoid{3, Prolate}`: prolate spheroid `a ≥ b = c`, axis `e₁`,
  aspect ratio `ω = a/b`.

# Returns
`TensTI{4, Float64, 5}` (major-symmetric Walpole tensor) with axis equal
to the matrix's TI axis.

Reference: [Barthélémy 2020](@cite barthelemyIJES2020_hilltrans).
"""
function _hill_3d_ti_coaxial(ell::Ellipsoid{3, Oblate}, C₀::TensND.TensTI{4, T, 5}) where {T}
    a, _, c = ell.semi_axes
    ω = c / a   # generic division — preserves Dual derivatives
    C1111, C1122, C1133, C3333, C2323 = TensND.arg_TI(C₀)
    P1, P2, P3, P5, P6 = _hill_ti_walpole(ω, C1111, C1122, C1133, C3333, C2323)
    return TensND.TensTI{4}(P1, P2, P3, P5, P6, TensND.axis(C₀))
end

function _hill_3d_ti_coaxial(ell::Ellipsoid{3, Prolate}, C₀::TensND.TensTI{4, T, 5}) where {T}
    a, b, _ = ell.semi_axes
    ω = a / b   # generic division — preserves Dual derivatives
    C1111, C1122, C1133, C3333, C2323 = TensND.arg_TI(C₀)
    P1, P2, P3, P5, P6 = _hill_ti_walpole(ω, C1111, C1122, C1133, C3333, C2323)
    return TensND.TensTI{4}(P1, P2, P3, P5, P6, TensND.axis(C₀))
end

function _hill_3d_ti_coaxial(::Ellipsoid{3, Spherical}, C₀::TensND.TensTI{4, T, 5}) where {T}
    # Spheroid degenerates to sphere — ω = 1, formula limit values
    C1111, C1122, C1133, C3333, C2323 = TensND.arg_TI(C₀)
    P1, P2, P3, P5, P6 = _hill_ti_walpole(one(T), C1111, C1122, C1133, C3333, C2323)
    return TensND.TensTI{4}(P1, P2, P3, P5, P6, TensND.axis(C₀))
end

# ── TI(N=6) inputs: collapse to the N=5 (major-symmetric) form via the ──────
# (ℓ₃, ℓ₄) → (ℓ₃ + ℓ₄)/2 average. Elasticity stiffnesses (and Hill tensors
# computed from them) are major-symmetric, so this is the right thing in
# practice; if the input has a non-trivial ℓ₃ ≠ ℓ₄ asymmetry it is silently
# averaged out here. The N=6 form arises naturally when a TI(axis) tensor
# is built as a `dcontract` of two N=5 tensors (the result is structurally
# allowed to be non-major-symmetric even when the operands are not).

function _ti6_to_ti5(C::TensND.TensTI{4, T, 6}) where {T}
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = TensND.get_data(C)
    ℓ34 = (ℓ₃ + ℓ₄) / 2
    return TensND.TensTI{4, T, 5}((ℓ₁, ℓ₂, ℓ34, ℓ₅, ℓ₆), TensND.axis(C))
end

_hill_3d_ti_coaxial(ell::Ellipsoid{3, Oblate}, C₀::TensND.TensTI{4, T, 6}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti6_to_ti5(C₀))
_hill_3d_ti_coaxial(ell::Ellipsoid{3, Prolate}, C₀::TensND.TensTI{4, T, 6}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti6_to_ti5(C₀))
_hill_3d_ti_coaxial(ell::Ellipsoid{3, Spherical}, C₀::TensND.TensTI{4, T, 6}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti6_to_ti5(C₀))

# ── TI(N=8) inputs ──────────────────────────────────────────────────────────
#
# `TensTI{4,T,8}` spans the commutant of SO(2) — rotations about the axis
# ONLY. That space is 8-dimensional, not 6: in Kelvin-Mandel the axial part
# is a full 2×2 block (ℓ₁..ℓ₄) and each of the m=1 and m=2 doublets carries a
# commutant ≅ ℂ, i.e. two real parameters instead of one (4 + 2 + 2 = 8).
# The two extra generators W₇, W₈ are built from the in-plane rotation
# generator `w·p = n × p`, so they are ODD in `n` and MAJOR-ANTISYMMETRIC.
#
# Requiring invariance under the *full* transverse-isotropy point group (i.e.
# adding the reflection n → −n) kills them, leaving the classical 6; adding
# major symmetry leaves the classical 5. `transverse_isotropify` returns N=8
# uniformly because the exact azimuthal average of a strain-CONCENTRATION
# tensor genuinely populates ℓ₇, ℓ₈ (such a tensor is neither major-symmetric
# nor reflection-invariant).
#
# A STIFFNESS, however, is major-symmetric, so its azimuthal average has
# ℓ₇ = ℓ₈ = 0 exactly — verified on the self-consistent running estimate.
# Dropping them here is therefore exact for every reference medium the
# analytical builder is meant to accept, and is the same convention as the
# N=6 → N=5 collapse just above (which averages a non-trivial ℓ₃ ≠ ℓ₄).
function _ti8_to_ti6(C::TensND.TensTI{4, T, 8}) where {T}
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, _, _ = TensND.get_ℓ8(C)
    return TensND.TensTI{4, T, 6}((ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆), TensND.axis(C))
end

_hill_3d_ti_coaxial(ell::Ellipsoid{3, Oblate}, C₀::TensND.TensTI{4, T, 8}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti8_to_ti6(C₀))
_hill_3d_ti_coaxial(ell::Ellipsoid{3, Prolate}, C₀::TensND.TensTI{4, T, 8}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti8_to_ti6(C₀))
_hill_3d_ti_coaxial(ell::Ellipsoid{3, Spherical}, C₀::TensND.TensTI{4, T, 8}) where {T} =
    _hill_3d_ti_coaxial(ell, _ti8_to_ti6(C₀))

# ── Coaxiality test ----------------------------------------------------------

"""
    _ti_coaxial(C₀::TensTI{4}, ell::Ellipsoid{3}) -> Bool

`true` when the TI symmetry axis of `C₀` is parallel to the (unique)
spheroid axis of `ell` (column 3 for `Oblate`/`Spherical`, column 1 for
`Prolate`).  Used by the dispatcher to route coaxial-spheroid problems
to the analytical TI builder.
"""
function _ti_coaxial(C₀::TensND.TensTI{4}, ell::Ellipsoid{3, Oblate})
    axis_C = collect(TensND.axis(C₀))
    axis_e = collect(TensND.components_canon(TensND.tens_basis(ell.basis, 3)))
    return isapprox(abs(dot(axis_C, axis_e)), 1.0; atol = 1.0e-10)
end

function _ti_coaxial(C₀::TensND.TensTI{4}, ell::Ellipsoid{3, Prolate})
    axis_C = collect(TensND.axis(C₀))
    axis_e = collect(TensND.components_canon(TensND.tens_basis(ell.basis, 1)))
    return isapprox(abs(dot(axis_C, axis_e)), 1.0; atol = 1.0e-10)
end

# Sphere is coaxial with anything (no preferred axis on the inclusion side).
_ti_coaxial(::TensND.TensTI{4}, ::Ellipsoid{3, Spherical}) = true

# Triaxial ellipsoids cannot be coaxial with a TI matrix in general.
_ti_coaxial(::TensND.TensTI{4}, ::Ellipsoid{3, Triaxial}) = false
