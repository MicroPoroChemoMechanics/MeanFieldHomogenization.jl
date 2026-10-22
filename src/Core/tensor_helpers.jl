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

# `primal(x)`, the value under every level of dual numbers, is TensND's.

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
