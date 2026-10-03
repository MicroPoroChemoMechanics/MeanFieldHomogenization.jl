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

Dense `3×3×3×3` component array of a TensND stiffness tensor (element
type inherited from `eltype(C₀)`).  Used by the numerical residue and
DECUHR algorithms.
"""
function _C_array(C₀::TensND.AbstractTens{4, 3})
    T = eltype(C₀)
    C = Array{T}(undef, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C[i, j, k, l] = C₀[i, j, k, l]
    end
    return C
end

"""
    _C_array(C₀, ::Type{Tdst}) -> Array{Tdst,4}

Variant of [`_C_array`](@ref) that casts the component array to a
different element type `Tdst`.  Used by the residue algorithm to obtain
a `ComplexF64` array even when the TensND tensor stores `Float64`.
"""
function _C_array(C₀::TensND.AbstractTens{4, 3}, ::Type{Tdst}) where {Tdst}
    C_src = _C_array(C₀)
    C = Array{Tdst}(undef, 3, 3, 3, 3)
    @inbounds for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        C[i, j, k, l] = Tdst(C_src[i, j, k, l])
    end
    return C
end

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
