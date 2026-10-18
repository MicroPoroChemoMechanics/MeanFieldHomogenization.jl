# ═════════════════════════════════════════════════════════════════════════════
#  Complete integrals via AGM (Abramowitz & Stegun 17.6, NIST DLMF 19.8)
# ═════════════════════════════════════════════════════════════════════════════

function _ell_K_agm(m::T) where {T <: Number}
    a = one(T)
    b = sqrt(one(T) - m)
    tol = _agm_tol(T)
    for _ in 1:60
        a_new = (a + b) / 2
        b_new = sqrt(a * b)
        a, b = a_new, b_new
        _agm_converged(T, a, b, tol) && break
    end
    return T(π) / (2 * a)
end

function _ell_E_agm(m::T) where {T <: Number}
    a = one(T)
    b = sqrt(one(T) - m)
    c = sqrt(m)
    s = c^2 / 2                              # (1/2) · 2⁰ · c₀²
    p = one(T)                               # running 2ⁿ
    tol = _agm_tol(T)
    for _ in 1:60
        a_new = (a + b) / 2
        b_new = sqrt(a * b)
        c_new = (a - b) / 2
        p *= 2
        s += p * c_new^2 / 2
        a, b, c = a_new, b_new, c_new
        _agm_converged(T, a, b, tol) && break
    end
    K_val = T(π) / (2 * a)
    return K_val * (one(T) - s)
end

# ─── Which scalar types can be *compared* ────────────────────────────────────
#
# `is_hard_numeric` used to be defined here. It is a predicate on scalar TYPES
# with nothing elliptic about it, and TensND — which owns `ApproxType`, its
# tolerance-side companion — is its natural home, so it now lives in
# `TensND/src/array_utils.jl` and is re-exported by this module for the eight
# sub-modules that reach it through `using ..Elliptic`.
#
# Known consumers: `_agm_converged` below, the stopping tests of `ell_RF` and
# `ell_RD`, `Core._sort_axes_and_basis`, `Core._classify_shape_2d/3d`,
# `Core._one_semi_axis` (Newton potentials), the `Ellipsoid` constructors,
# `Cracks._classify_crack`,
# `Cracks._is_unit_alignment`, `Cracks._elliptic_CS`, `Laminates._parallel`,
# `Laminates.add_layer!` / `validate_laminate`, `Schemes.validate_rve`,
# `Viscoelasticity._checkable`.

# ─── Tolerances — tuned per scalar type ──────────────────────────────────────

# A tolerance at the working precision where comparisons return a `Bool`; none
# for a complex or a symbolic scalar, which runs the fixed 60 iterations.
_agm_tol(::Type{T}) where {T <: AbstractFloat} = 10 * eps(T)
_agm_tol(::Type{T}) where {T <: Number} = is_hard_numeric(T) ? 10 * eps(Float64) : 0.0

"""
    _agm_converged(::Type{T}, a, b, tol) -> Bool

Whether the AGM iteration has converged. Always returns a *hard* `Bool`, so it
can drive `&&`.

`T <: Real` is **not** a sufficient test for "this scalar compares to a
tolerance": `Symbolics.Num <: Real`, yet `<` on a `Num` yields a symbolic
inequality rather than a `Bool`, and using it in a boolean context throws. (The
same wrong predicate once made `Cracks._elliptic_CS` return `NaN` on a symbolic
penny and `Cracks._ti_aligned` throw on a symbolic TI reference.)
[`is_hard_numeric`](@ref) is the predicate: a `ForwardDiff.Dual` compares, a
`Num`, a `SymPy.Sym` or a complex number does not, and never signals
convergence: it runs the fixed 60 iterations.
"""
function _agm_converged(::Type{T}, a, b, tol) where {T <: Number}
    is_hard_numeric(T) || return false
    denom = abs(a)
    return iszero(denom) ? abs(a - b) < tol : abs(a - b) < tol * denom
end
