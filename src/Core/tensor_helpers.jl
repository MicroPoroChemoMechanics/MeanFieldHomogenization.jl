# =============================================================================
#  tensor_helpers.jl
#
#  Low-level helpers for 4th-order tensor assembly and simple utilities
#  (Kronecker delta, static arrays filled entry by entry, the fill of the
#  eight entries a symmetric tensor shares), and the reading of a property of
#  the plane as an isotropic tensor.  Shared by every sub-module.
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

# `primal(x)`, the value under every level of dual numbers, is TensND's, and so
# are the inferred constructors from a component array (`static_tens`,
# `symmetric_tens`) that every Hill tensor is built with.

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
    Piso = TensND.project_exact(P, Val(:ISO); rtol = _PLANE_ISO_RTOL)
    Piso === nothing || return Piso
    # Refused: the distance is computed again only to be named in the message.
    _, _, drel = TensND.project(P, Val(:ISO))
    throw(
        ArgumentError(
            "an inclusion of the plane takes a property that is isotropic in the plane; " *
                "the $(nameof(typeof(P))) it received is at a relative distance $drel from isotropy"
        )
    )
end
_plane_isotropic(P) = throw(
    ArgumentError(
        "in a two-dimensional cell the properties are tensors of the plane, those of the " *
            "layers of an inclusion included (`plane_strain_stiffness`, `plane_stress_stiffness` " *
            "of TensND build them from a three-dimensional stiffness); got a $(nameof(typeof(P)))"
    )
)
