# =============================================================================
#  newton_potential.jl
#
#  Newton potentials `Iᵢ` (2D / 3D) for an ellipsoid given its semi-axes.
#  Kept in Core because both the `Elasticity` and `Conductivity`
#  sub-modules need it. Element-type genericity covers `Float64`,
#  `ForwardDiff.Dual`, `SymPyPythonCall.Sym`, `Symbolics.Num`.
#
#  The `newton_potential_3d(ell::Ellipsoid)` and
#  `newton_potential_2d(ell::Ellipsoid)` convenience methods are deliberately
#  added in the `Elasticity` sub-module (where `Ellipsoid` is defined), not
#  here, to avoid a circular dependency.
# =============================================================================

"""
    newton_potential_3d(a, b, c) -> (Iv, IIv)

Newton potential integrals for a 3-D ellipsoid with semi-axes ``a \\ge b \\ge c > 0``.

Returns:
- `Iv` ``= (I_a, I_b, I_c)`` with ``I_a + I_b + I_c = 4\\pi``
- `IIv` ``= (I_{aa}, I_{bb}, I_{cc}, I_{bc}, I_{ca}, I_{ab})``

The route depends on whether comparisons on `T` return a `Bool`
(`is_hard_numeric`), not on `T <: Real`, which `Symbolics.Num` also satisfies:
- numbers (`Float64`, `ForwardDiff.Dual`, …): a spheroid, sphere
  included, goes through the integrals of `src/Core/spheroid_integrals.jl`; a
  triaxial ellipsoid whose two closer semi-axes differ by less than 2 %
  (relative, in squares) through their expansion around that spheroid; any
  other through elliptic integrals. All three are exact to rounding, and the
  first two stay differentiable where two semi-axes meet: a dual semi-axis is
  merged with another one only when their derivatives coincide too.
- symbolic types (`Symbolics.Num`, `SymPyPythonCall.Sym`): structural equality via `isequal`
  selects the same four cases, in closed form.
"""
newton_potential_3d(a::T, b::T, c::T) where {T <: Number} =
    _newton_potential_3d(Val(is_hard_numeric(T)), a, b, c)
newton_potential_3d(a::Number, b::Number, c::Number) = newton_potential_3d(promote(a, b, c)...)

function _newton_potential_3d(::Val{true}, a::T, b::T, c::T) where {T}
    # The closer pair of semi-axes: its relative gap in squares, and the
    # spheroid it sits next to (axis c for a ≈ b, axis a for b ≈ c).
    a², b², c² = a * a, b * b, c * c
    r_ab = abs(a² - b²) / (a² + b²)
    r_bc = abs(b² - c²) / (b² + c²)
    if r_ab ≤ r_bc
        if _same_semi_axis(a, b)            # oblate, or a sphere: axis c
            Ic, Ia, Ica, Icc, Iaa = _newton_spheroid(c, a)
            return (Ia, Ia, Ic), (Iaa, Iaa, Icc, Ica, Ica, Iaa)
        elseif r_ab ≤ _NEAR_SPHEROID_SPLIT  # close to that spheroid
            Ic, Ia, Ib, Icc, Iaa, Ibb, Ica, Icb, Iab = _newton_near_spheroid(c, a, b)
            return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Icb, Ica, Iab)
        end
    else
        if _same_semi_axis(b, c)            # prolate: axis a
            Ia, Ic, Ica, Iaa, Icc = _newton_spheroid(a, c)
            return (Ia, Ic, Ic), (Iaa, Icc, Icc, Icc, Ica, Ica)
        elseif r_bc ≤ _NEAR_SPHEROID_SPLIT
            Ia, Ib, Ic, Iaa, Ibb, Icc, Iab, Iac, Ibc = _newton_near_spheroid(a, b, c)
            return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Iac, Iab)
        end
    end
    return _newton_potential_3d_triaxial(a, b, c)
end

# Symbolic types: no comparison returns a `Bool`, the semi-axes are ordered by
# the caller and the cases are told apart structurally.
function _newton_potential_3d(::Val{false}, a::T, b::T, c::T) where {T}
    AeqB = isequal(a, b)
    BeqC = isequal(b, c)

    if AeqB && BeqC
        A2 = a * a
        Ia = Ib = Ic = 4 * T(π) / 3
        Iaa = Ibb = Icc = Ibc = Ica = Iab = 4 * T(π) / (5 * A2)
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)

    elseif AeqB && !BeqC
        A2 = a * a;  C2 = c * c
        Ia = Ib = 2 * T(π) * c * (A2 * acos(c / a) - c * sqrt(A2 - C2)) /
            (A2 - C2)^(3 // 2)
        Ic = 4 * T(π) - 2 * Ia
        Ica = Ibc = (Ia - Ic) / (C2 - A2)
        Icc = 4 * T(π) / (3 * C2) - 2 * Ica / 3
        Iaa = Ibb = Iab = T(π) / A2 - Ica / 4
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)

    elseif !AeqB && BeqC
        A2 = a * a;  C2 = c * c
        Ic = Ib = 2 * T(π) * a * (a * sqrt(A2 - C2) - C2 * acosh(a / c)) /
            (A2 - C2)^(3 // 2)
        Ia = 4 * T(π) - 2 * Ic
        Ica = Iab = (Ia - Ic) / (C2 - A2)
        Iaa = 4 * T(π) / (3 * A2) - 2 * Ica / 3
        Ibb = Icc = Ibc = T(π) / C2 - Ica / 4
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)

    else
        return _newton_potential_3d_triaxial(a, b, c)
    end
end

# ── Internal: general triaxial formula ───────────────────────────────────────
function _newton_potential_3d_triaxial(a::T, b::T, c::T) where {T <: Number}
    A2 = a * a;  B2 = b * b;  C2 = c * c
    theta = asin(sqrt(one(T) - C2 / A2))
    k2 = (A2 - B2) / (A2 - C2)
    Fell = ell_F(theta, k2)
    Eell = ell_E(theta, k2)
    fac = 4 * T(π) * a * b * c / sqrt(A2 - C2)
    Ia = fac / (A2 - B2) * (Fell - Eell)
    Ic = fac / (B2 - C2) * (b * sqrt(A2 - C2) / (a * c) - Eell)
    Ib = 4 * T(π) - Ia - Ic
    Ibc = (Ic - Ib) / (B2 - C2)
    Ica = (Ic - Ia) / (A2 - C2)
    Iab = (Ib - Ia) / (A2 - B2)
    Iaa = (4 * T(π) / A2 - Iab - Ica) / 3
    Ibb = (4 * T(π) / B2 - Iab - Ibc) / 3
    Icc = (4 * T(π) / C2 - Ica - Ibc) / 3
    return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
end

# ── 3-D Newton potentials — infinite cylinder (a → ∞, b ≥ c > 0) ────────────

"""
    newton_potential_3d_cylinder(b, c) -> (Iv, IIv)

Newton potential integrals for an infinite cylinder of elliptic cross-section
with transverse semi-axes ``b \\ge c > 0`` (cylinder axis ``\\underline{e}_1``, transverse plane
``(\\underline{e}_2, \\underline{e}_3)``).

Obtained as the limit ``a \\to \\infty`` of [`newton_potential_3d`](@ref) — the
cylinder axis contributes no finite Newton mass (``I_a = I_{aa} = I_{ab} = I_{ac} = 0``)
and the transverse potentials collapse to simple rational expressions in
``(b, c)``.

Returns:
- `Iv` ``= (I_a, I_b, I_c)`` with ``I_a + I_b + I_c = 4\\pi``, ``I_a = 0``.
- `IIv` ``= (I_{aa}, I_{bb}, I_{cc}, I_{bc}, I_{ca}, I_{ab})`` with ``I_{aa} = I_{ab} = I_{ac} = 0``.

The elliptic closed forms hold down to ``b = c``, where they reduce to the
circular ones (``I_b = I_c = 2\\pi``, ``I_{bb} = I_{cc} = I_{bc} = \\pi/b^2``): there is
no ``1/(b^2 - c^2)`` denominator, so no tolerance is needed, and the circular forms
are taken only for semi-axes that are one — the same expression, or the same
number with the same derivatives (`_one_semi_axis`). Generic in the scalar type:
`Float64`, `ForwardDiff.Dual`, `Symbolics.Num`, `SymPyPythonCall.Sym`.
"""
function newton_potential_3d_cylinder(b, c)
    T = promote_type(typeof(b), typeof(c))
    return newton_potential_3d_cylinder(T(b), T(c))
end

function newton_potential_3d_cylinder(b::T, c::T) where {T <: Number}
    Ia = Iaa = Iab = Ica = zero(T)
    if _one_semi_axis(b, c)
        Ib = Ic = 2 * T(π)
        Ibb = Icc = Ibc = T(π) / (b * b)
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
    end
    s = b + c
    s2 = s * s
    Ib = 4 * T(π) * c / s
    Ic = 4 * T(π) * b / s
    Ibc = 4 * T(π) / s2
    Ibb = 4 * T(π) / 3 * (one(T) / (b * b) - one(T) / s2)
    Icc = 4 * T(π) / 3 * (one(T) / (c * c) - one(T) / s2)
    return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
end

"""
    newton_potential_2d(a, b) -> (Ia, Ib)

Newton potential integrals for a 2-D ellipse with semi-axes ``a \\ge b > 0``.
Returns `(Ia, Ib)` ``= (I_a, I_b)`` with ``I_a + I_b = 2\\pi``.

Works for any `T<:Number` including `ForwardDiff.Dual`, `SymPyPythonCall.Sym`, `Symbolics.Num`.

Formulas:
- Circle: ``I_a = I_b = \\pi``
- General ellipse: ``I_a = 2\\pi b/(a+b)``, ``I_b = 2\\pi a/(a+b)``
"""
function newton_potential_2d(a::T, b::T) where {T <: Number}
    _one_semi_axis(a, b) && return (T(π), T(π))
    s = a + b
    return (2 * T(π) * b / s, 2 * T(π) * a / s)
end
newton_potential_2d(a::Number, b::Number) = newton_potential_2d(promote(a, b)...)
