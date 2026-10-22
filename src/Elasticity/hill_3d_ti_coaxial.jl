# =============================================================================
#  hill_3d_ti_coaxial.jl — Analytical Hill polarization tensor for a
#  spheroidal inclusion coaxial with a transversely isotropic matrix.
#
#  Implements the closed-form formula derived in
#  [barthelemyIJES2020_hilltrans](@citet) (eqs. 49–58 of the post-print)
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
#  [mura1987](@cite).
# =============================================================================

# The auxiliary integrals I_n, J_n of the closed form, with the regimes that
# keep them exact near the sphere, near coincident roots and for flat shapes,
# live in `src/Core/spheroid_integrals.jl`: the Newton potentials of the
# isotropic case share them.

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
of aspect ratio ``\\omega`` (axial / transverse semi-axis) coaxial with a transversely
isotropic matrix specified by the five independent elastic constants.

Element type policy: with
`T = promote_type(typeof(ω), typeof(C1111), …, typeof(C2323))`,

- if `T <: Real` (standard real-modulus case), the arithmetic is carried
  out in `Complex{T}` to keep the complex roots of the acoustic
  polynomial well-defined; the imaginary parts cancel by construction
  and only the real parts are returned. Compatible with
  `ForwardDiff.Dual` numbers: the analytical Hill tensor is
  differentiable through the elastic constants and the spheroid aspect
  ratio, at an isotropic point too, where the two acoustic roots
  coincide (see `src/Core/spheroid_integrals.jl`).
- if `T` is itself complex (frequency-domain viscoelasticity, harmonic
  problems), the formula is evaluated directly in `T` and the genuinely
  complex ``P_i`` are returned unchanged.

When the matrix is in fact isotropic (``C_{1111} = C_{3333} = \\lambda + 2\\mu``,
``C_{1122} = C_{1133} = \\lambda``, ``C_{2323} = \\mu``) the returned coefficients reduce to
the classical Mura formula.

!!! note "Symbolic numbers (SymPy Sym)"
    The function is **not** compatible with SymPy `Sym` inputs because
    Julia's `Complex{T}` parametric type requires `T <: Real`, which
    `Sym` does not satisfy. For symbolic exploration of the analytical
    formula, derive the components by hand from eqs. 49–58 of the
    paper, or use the residue/DECUHR backends with substituted
    numerical values.

Reference: [barthelemyIJES2020_hilltrans](@citet), eqs. 49–58.
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

    # The acoustic polynomial a γ⁴ + b γ² + c has roots γ₁², γ₂² of sum −b/a
    # and product c/a; aᵢ = ω² γᵢ² enter through their mean and squared
    # half-difference, rational in the moduli and free of their scale.
    a = C44 * C33
    b = C13^2 + 2 * C13 * C44 - C11 * C33
    c = C11 * C44
    m = -om2 * b / (2 * a)
    d2 = om4 * (b^2 - 4 * a * c) / (4 * a^2)
    a3 = om2 * (C11 - C12) / (2 * C44)

    coef = C44 * C33
    J4, L21, L02 = MFH_Core._spheroid_JL(m, d2)
    j4 = J4 / coef
    l21 = L21 / coef
    l02 = L02 / coef
    i2 = MFH_Core._spheroid_I2(a3) / C44
    iw = MFH_Core._spheroid_Iw(a3) / C44

    # Walpole components (eqs. 53–58 of Barthélémy 2020), written with
    # j₂ − j₄ = l₂₁ and j₀ − 2j₂ + j₄ = l₀₂: in j₀, j₂, j₄ they are differences of
    # terms ω² or ω⁴ larger than the result for a needle.
    P1_c = (C44 * j4 + om2 * C11 * l21) / 2
    P2_c = om2 * (C33 * l21 + om2 * C44 * l02) / 4
    P3_c = -om2 / (2 * sqrt(TC(2))) * (C44 + C13) * l21
    P5_c = P2_c / 2 + om2 * iw / 8
    P6_c = (C33 * j4 - 2 * om2 * C13 * l21 + om4 * C11 * l02 + i2) / 8

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
- `ell::Ellipsoid{3, Oblate}`: oblate spheroid ``a = b \\ge c``, axis ``\\underline{e}_3``,
  aspect ratio ``\\omega = c/a``.
- `ell::Ellipsoid{3, Prolate}`: prolate spheroid ``a \\ge b = c``, axis ``\\underline{e}_1``,
  aspect ratio ``\\omega = a/b``.

# Returns
`TensTI{4, Float64, 5}` (major-symmetric Walpole tensor) with axis equal
to the matrix's TI axis.

Reference: [barthelemyIJES2020_hilltrans](@citet).
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

_ti6_to_ti5(C::TensND.TensTI{4, T, 6}) where {T} = TensND.Tensors.majorsymmetric(C)

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
_ti8_to_ti6(C::TensND.TensTI{4, T, 8}) where {T} = TensND.rebuild(C, TensND.get_ℓ8(C)[1:6])

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
to the analytical TI builder. The test is TensND's `isparallel`, safe on symbolic
axes, where `isapprox(abs(dot(…)), 1)` threw on a `Num` and conjugated a `Sym`.
"""
_ti_coaxial(C₀::TensND.TensTI{4}, ell::Ellipsoid{3, Oblate}) =
    TensND.isparallel(TensND.axis(C₀), TensND.tens_basis(ell.basis, 3))
_ti_coaxial(C₀::TensND.TensTI{4}, ell::Ellipsoid{3, Prolate}) =
    TensND.isparallel(TensND.axis(C₀), TensND.tens_basis(ell.basis, 1))

# Sphere is coaxial with anything (no preferred axis on the inclusion side).
_ti_coaxial(::TensND.TensTI{4}, ::Ellipsoid{3, Spherical}) = true

# Triaxial ellipsoids cannot be coaxial with a TI matrix in general.
_ti_coaxial(::TensND.TensTI{4}, ::Ellipsoid{3, Triaxial}) = false
