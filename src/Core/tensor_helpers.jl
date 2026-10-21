# =============================================================================
#  tensor_helpers.jl
#
#  Low-level helpers for 4th-order tensor assembly and simple utilities
#  (Kronecker delta, 3×3×3×3 array extraction, symmetrization …).  Shared
#  by every sub-module.
# =============================================================================

"""
    _δ(i, j, ::Type{T})

Kronecker delta ``\\delta_{ij}`` in element type `T` — returns `one(T)` when
`i == j` and `zero(T)` otherwise.
"""
@inline _δ(i, j, ::Type{T}) where {T} = i == j ? one(T) : zero(T)

"""
    _primal(x)

`x` stripped of every level of `ForwardDiff.Dual`: the number a choice that
does not depend on derivatives — a quadrature exponent, a scaling period, a
bracket — is made on, so that the derivatives stay where the mathematics puts
them.
"""
_primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))
_primal(x) = x

"""
    _fill_sym4!(A, i, j, k, l, v)

Fill the eight entries of a 4-rank array `A` linked to `(i,j,k,l)` by
the full minor + major symmetries with the same value `v`.
"""
@inline function _fill_sym4!(A::AbstractArray{<:Any, 4}, i, j, k, l, v)
    A[i, j, k, l] = A[j, i, k, l] = A[i, j, l, k] = A[j, i, l, k] = v
    A[k, l, i, j] = A[l, k, i, j] = A[k, l, j, i] = A[l, k, j, i] = v
    return A
end

"""
    _C_array(C₀) -> Array{T,4}

**Canonical** components of a 4th-order TensND tensor, as a dense `3×3×3×3`
array (element type inherited from `C₀`). Every caller combines them with
canonical data (a position, a direction, another canonical array), and
`C₀[i, j, k, l]` or `get_array(C₀)` would read a tensor written in a rotated
basis in that basis instead. Free for a tensor already in the canonical basis,
which `components_canon` returns as it is.
"""
_C_array(C₀::TensND.AbstractTens{4, 3}) = Array(TensND.components_canon(C₀))

"""
    _components_in(t, basis) -> Array

Components of a TensND tensor in `basis`, as a plain array. No change of basis
when `t` is written in `basis` already: `change_tens` returns it as it is.
"""
_components_in(t::TensND.AbstractTens, basis::TensND.AbstractBasis) =
    Array(TensND.get_array(TensND.change_tens(t, basis)))

# ─── Orthotropic assembly (3D, 4th order) ────────────────────────────────────

"""
    _make_ortho(::Type{T}, C11, C22, C33, C12, C13, C23, C44, C55, C66,
                _, basis) -> AbstractTens{4,3}

Build a 4th-order orthotropic tensor from its nine Voigt-Mandel
independent components.  Returns the most specific TensND type the
element types allow:

* `CanonicalBasis{3}`          → `TensOrtho` in a fresh
                                 `CanonicalBasis{3,T}`
* `RotatedBasis{3,T}`          → `TensOrtho` in that rotated basis
* otherwise (e.g. `T` ≠ basis) → generic `Tens` in the canonical frame.

The ignored positional argument (here a leading `_`) corresponds to an
optional pre-allocated buffer slot that is currently unused.
"""
function _make_ortho(
        ::Type{T},
        C11, C22, C33, C12, C13, C23, C44, C55, C66,
        _, basis::TensND.CanonicalBasis{3}
    ) where {T}
    return TensND.TensOrtho(
        C11, C22, C33, C12, C13, C23, C44, C55, C66,
        TensND.CanonicalBasis{3, T}()
    )
end

function _make_ortho(
        ::Type{T},
        C11, C22, C33, C12, C13, C23, C44, C55, C66,
        _, basis::TensND.RotatedBasis{3, T}
    ) where {T}
    return TensND.TensOrtho(C11, C22, C33, C12, C13, C23, C44, C55, C66, basis)
end

# Fallback: type mismatch (e.g. `T = ForwardDiff.Dual` + `Float64` basis).
# Reconstruct the 81-component array manually and wrap as a generic `Tens`
# in the given basis.
function _make_ortho(
        ::Type{T},
        C11, C22, C33, C12, C13, C23, C44, C55, C66,
        _, basis
    ) where {T}
    P_arr = zeros(T, 3, 3, 3, 3)
    P_arr[1, 1, 1, 1] = C11; P_arr[2, 2, 2, 2] = C22; P_arr[3, 3, 3, 3] = C33
    _fill_sym4!(P_arr, 1, 1, 2, 2, C12)
    _fill_sym4!(P_arr, 1, 1, 3, 3, C13)
    _fill_sym4!(P_arr, 2, 2, 3, 3, C23)
    _fill_sym4!(P_arr, 2, 3, 2, 3, C44)
    _fill_sym4!(P_arr, 1, 3, 1, 3, C55)
    _fill_sym4!(P_arr, 1, 2, 1, 2, C66)
    return TensND.Tens(P_arr, basis)
end
