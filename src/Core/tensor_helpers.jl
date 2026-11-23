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

# The same for an array symmetric by construction (the transpose for order 2, the
# minor symmetries for order 4), stored as a `SymmetricTensor`. `_tens_in` decides
# the storage on the values, symmetric up to rounding or not, so that its type is
# a `Union` of the two storages; here the storage, hence the type, is known before
# the values. The components are those `_tens_in` gives a symmetric array, the
# mean over the symmetries, without the comparison that decided the storage.
@inline _sym_tens_in(arr::AbstractArray{T, order}, basis::TensND.AbstractBasis{dim}) where {T, order, dim} =
    TensND.Tens(TensND.Tensors.symmetric(TensND.Tensors.Tensor{order, dim, T}(arr)), basis)

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

# ── Properties of a two-dimensional cell ─────────────────────────────────────

# Relative distance to isotropy under which a general tensor of the plane (the
# running estimate of a scheme) is read as isotropic.
const _PLANE_ISO_RTOL = 1.0e-8

"""
    _plane_isotropic(P) -> TensISO

A property of the plane that is isotropic: a `TensISO`, or a general tensor of
the plane that is one, as the estimate of a scheme is. Numbers only for the
second: the distance to isotropy is compared.
"""
_plane_isotropic(P::TensND.TensISO{order, 2}) where {order} = P
function _plane_isotropic(P::TensND.AbstractTens{order, 2}) where {order}
    is_hard_numeric(eltype(P)) || throw(
        ArgumentError(
            "an inclusion of the plane takes a symbolic property as a TensISO; a " *
                "$(nameof(typeof(P))) would have to be compared with its isotropic part"
        )
    )
    Piso, _, drel = TensND.project(P, Val(:ISO))
    drel ≤ _PLANE_ISO_RTOL || throw(
        ArgumentError(
            "an inclusion of the plane takes a property that is isotropic in the plane; " *
                "the $(nameof(typeof(P))) it received is at a relative distance $drel from isotropy"
        )
    )
    return Piso
end
_plane_isotropic(P) = throw(
    ArgumentError(
        "in a two-dimensional cell the properties are tensors of the plane, those of the " *
            "layers of an inclusion included (`plane_strain_stiffness`, `plane_stress_stiffness` " *
            "of TensND build them from a three-dimensional stiffness); got a $(nameof(typeof(P)))"
    )
)
