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

# Static array of size `dims` whose entry at `(i, j, …)` is `f(i, j, …)`, built
# entry by entry without an intermediate heap array.
@inline _sfill(f, ::Val{dims}) where {dims} = SArray{Tuple{dims...}}(
    ntuple(q -> f(Tuple(CartesianIndices(dims)[q])...), Val(prod(dims)))
)

# `primal(x)`, the value under every level of dual numbers, is TensND's.

# The tensor of components `arr` in `basis`, of order and dimension known from
# the types: built from a plain array, they are read from its size at run time,
# and inference loses them, with the type of every Hill tensor built so.
@inline _tens_in(arr::AbstractArray{T, order}, basis::TensND.AbstractBasis{dim}) where {T, order, dim} =
    TensND.Tens(TensND._static_tensor(arr, Val(dim), Val(order)), basis)

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
