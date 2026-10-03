# =============================================================================
#  newton_potential.jl
#
#  Newton potentials `Iᵢ` (2D / 3D) for an ellipsoid given its semi-axes.
#  Kept in Core because both the `Elasticity` and `Conductivity`
#  sub-modules need it. Element-type genericity covers `Float64`,
#  `ForwardDiff.Dual`, `SymPy.Sym`, `Symbolics.Num`.
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

Two methods are provided:
- `T<:Real` (includes `Float64`, `ForwardDiff.Dual`): numerically stable case-split
  using tolerance comparisons (sphere, oblate, prolate, triaxial with elliptic integrals).
- `T<:Number` (e.g. `SymPy.Sym`, `Symbolics.Num`): structural equality via `isequal`
  selects the same four cases.
"""
function newton_potential_3d(a::T, b::T, c::T) where {T <: Real}
    tol = 1.0e-6 * one(T)
    AeqB = (a - b) ≤ a * tol
    BeqC = (b - c) ≤ b * tol

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

# Symbolic types (T<:Number but not T<:Real)
function newton_potential_3d(a::T, b::T, c::T) where {T <: Number}
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

Two methods are provided:
- `T<:Real` (includes `Float64`, `ForwardDiff.Dual`): numerically stable
  case-split via tolerance comparison to pick the circular (``b = c``) or the
  elliptic (``b > c``) branch.
- `T<:Number` (`SymPy.Sym`, `Symbolics.Num`, …): structural equality via
  `isequal` selects the two branches.

Both branches are written as closed-form limits — no ``1/(b^2 - c^2)`` style
denominators, so the routine is free of ``0/0`` indeterminacies at ``b = c``
and differentiable through `ForwardDiff`.
"""
function newton_potential_3d_cylinder(b, c)
    T = promote_type(typeof(b), typeof(c))
    return newton_potential_3d_cylinder(T(b), T(c))
end

function newton_potential_3d_cylinder(b::T, c::T) where {T <: Real}
    tol = 1.0e-6 * one(T)
    BeqC = (b - c) ≤ b * tol

    if BeqC
        # Circular base (b = c): I_b = I_c = 2π, I_bb = I_cc = I_bc = π/b²
        B2 = b * b
        Ia = zero(T)
        Ib = Ic = 2 * T(π)
        Iaa = Iab = Ica = zero(T)
        Ibb = Icc = Ibc = T(π) / B2
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
    else
        # Elliptic base (b > c)
        s = b + c
        s2 = s * s
        Ia = zero(T)
        Ib = 4 * T(π) * c / s
        Ic = 4 * T(π) * b / s
        Iaa = Iab = Ica = zero(T)
        Ibc = 4 * T(π) / s2
        Ibb = 4 * T(π) / 3 * (one(T) / (b * b) - one(T) / s2)
        Icc = 4 * T(π) / 3 * (one(T) / (c * c) - one(T) / s2)
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
    end
end

# Symbolic types (T<:Number but not T<:Real) — structural equality
function newton_potential_3d_cylinder(b::T, c::T) where {T <: Number}
    if isequal(b, c)
        B2 = b * b
        Ia = zero(T)
        Ib = Ic = 2 * T(π)
        Iaa = Iab = Ica = zero(T)
        Ibb = Icc = Ibc = T(π) / B2
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
    else
        s = b + c
        s2 = s * s
        Ia = zero(T)
        Ib = 4 * T(π) * c / s
        Ic = 4 * T(π) * b / s
        Iaa = Iab = Ica = zero(T)
        Ibc = 4 * T(π) / s2
        Ibb = 4 * T(π) / 3 * (one(T) / (b * b) - one(T) / s2)
        Icc = 4 * T(π) / 3 * (one(T) / (c * c) - one(T) / s2)
        return (Ia, Ib, Ic), (Iaa, Ibb, Icc, Ibc, Ica, Iab)
    end
end

# ── 2-D Newton potentials ─────────────────────────────────────────────────────

"""
    newton_potential_2d(a, b) -> (Ia, Ib)

Newton potential integrals for a 2-D ellipse with semi-axes ``a \\ge b > 0``.
Returns `(Ia, Ib)` ``= (I_a, I_b)`` with ``I_a + I_b = 2\\pi``.

Works for any `T<:Number` including `ForwardDiff.Dual`, `SymPy.Sym`, `Symbolics.Num`.

Formulas:
- Circle: ``I_a = I_b = \\pi``
- General ellipse: ``I_a = 2\\pi b/(a+b)``, ``I_b = 2\\pi a/(a+b)``
"""
function newton_potential_2d(a::T, b::T) where {T <: Real}
    tol = 1.0e-6 * one(T)
    if (a - b) ≤ a * tol
        return (T(π), T(π))
    else
        s = a + b
        return (2 * T(π) * b / s, 2 * T(π) * a / s)
    end
end

function newton_potential_2d(a::T, b::T) where {T <: Number}
    if isequal(a, b)
        return (T(π), T(π))
    else
        s = a + b
        return (2 * T(π) * b / s, 2 * T(π) * a / s)
    end
end
