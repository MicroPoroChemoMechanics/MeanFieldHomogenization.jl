# =============================================================================
#  bases.jl
#
#  Low-level helpers operating on TensND `AbstractBasis` objects or on
#  the Julia scalar types themselves — shared by every sub-module.
# =============================================================================

"""
    _floatlike(::Type{T})

Promote integer element types to their floating-point counterpart (so that
`Integer`-valued semi-axes become `Float64` or `BigFloat`) while leaving
every `AbstractFloat`, `ForwardDiff.Dual` or symbolic type untouched.
"""
_floatlike(::Type{T}) where {T <: Integer} = float(T)
_floatlike(::Type{T}) where {T <: Number} = T

"""
    _basis_eltype(::Type{T})

Element type of the default `CanonicalBasis` associated with a given
scalar element type `T`.  Real types always use `Float64` (to match the
historical TensND default); non-real scalars (SymPy, Symbolics, …) keep
their own element type so that the `CanonicalBasis` and the tensor data
share the same element type.
"""
_basis_eltype(::Type{T}) where {T <: Real} = Float64
_basis_eltype(::Type{T}) where {T <: Number} = T

"""
    _basis_col(basis, m::Int) -> NTuple

Return the `m`-th column of the TensND basis as an `NTuple{d}` — the
coordinates of the `m`-th basis vector in the canonical frame.
"""
function _basis_col(basis::TensND.AbstractBasis, m::Int)
    E = TensND.vecbasis(basis, :cov)
    d = size(E, 1)
    return ntuple(i -> E[i, m], d)
end

"""
    _frame_columns(basis) -> (l̂, m̂, n̂)

Return the three axis vectors of a 3D basis as plain `Vector`s — a
convenience form used by the numerical (residue / DECUHR) algorithms.
The returned element type follows the basis element type.
"""
@inline function _frame_columns(basis::TensND.AbstractBasis)
    l̂ = Vector(TensND.components_canon(TensND.tens_basis(basis, 1)))
    m̂ = Vector(TensND.components_canon(TensND.tens_basis(basis, 2)))
    n̂ = Vector(TensND.components_canon(TensND.tens_basis(basis, 3)))
    return l̂, m̂, n̂
end

"""
    _normalize_euler(angles) -> NTuple{3}

Pad a ZYZ Euler-angle tuple of length 0–3 to a full triple, promoting every
entry to one common type.  Missing trailing angles default to `0`.  Accepts
heterogeneous tuples (`Int`, `Irrational`, `Float32/64`, `Dual`, …) thanks to
`promote_type` + `float`, and **symbolic** ones (`Sym`, `Num`), which keep their
own type rather than being forced through `Float64`.

```
_normalize_euler(())                 == (0.0, 0.0, 0.0)
_normalize_euler((π/2,))             == (π/2, 0.0, 0.0)       # Float64
_normalize_euler((π, 0, 0))          == (π, 0.0, 0.0)          # Irrational → Float
_normalize_euler((0, 1, 2))          == (0.0, 1.0, 2.0)
_normalize_euler((θ, 0, 0))          == (θ, Sym(0), Sym(0))    # symbolic θ
```

A tuple of length > 3 raises `ArgumentError`.
"""
function _normalize_euler(angles::Tuple{Vararg{Number}})
    n = length(angles)
    n > 3 && throw(
        ArgumentError(
            "euler_angles accepts at most 3 ZYZ angles; got $n values ($(angles))."
        )
    )
    if n == 0
        return (0.0, 0.0, 0.0)
    end
    T_promoted = promote_type(map(typeof, angles)...)
    # `float` is what turns an `Int` or an `Irrational` angle into a usable one;
    # it is meaningless on a symbolic type, and `Symbolics.Num` subtypes `Real`
    # without being convertible from `Float64`, so `is_hard_numeric` — not
    # `<: Real` — is what decides here.
    T = if is_hard_numeric(T_promoted)
        Tf = float(T_promoted)
        Tf <: AbstractFloat ? Tf : Float64
    else
        T_promoted
    end
    return ntuple(i -> i ≤ n ? convert(T, angles[i]) : zero(T), 3)
end

# Convenience forwarding methods for common concrete-type patterns.
_normalize_euler(::Tuple{}) = (0.0, 0.0, 0.0)

"""
    _frame_from_normal(n; T = nothing, in_plane = nothing) -> AbstractBasis

Complete a normal vector `n` into an orthonormal frame `(ℓ, m, n̂)` whose
**third** axis is `n̂ = n/‖n‖`, by Gram-Schmidt against a reference axis.
Returns a `CanonicalBasis` when `n` is exactly the third canonical axis, so the
common laminate needs no rotation at all.

`T` is the element-type floor of the cell the frame belongs to, and it decides
the element type of that `CanonicalBasis` short-circuit alone: a symbolic cell
keeps a symbolic canonical frame, whose axes are the exact `0` and `1` rather
than `0.0` and `1.0`. That distinction is not cosmetic — a `TensTI` converts its
axis to the element type of its *data*, so a `Float64` `1.0` reappears as a
symbolic `1.0` multiplying every coefficient of the result.

`in_plane` is the reference axis Gram-Schmidt orthogonalizes against, i.e. what
fixes `ℓ` in the plane of the layers; it must not be parallel to `n̂`. Left
unset, a **numeric** normal picks whichever canonical axis is least aligned with
`n̂`, which can never degenerate; a **symbolic** normal cannot answer that
comparison, so it falls back to `e₁` and the caller must pass another reference
when `n̂ ∥ e₁`.

The effective property of a laminate is invariant under rotation about its
normal, so this choice never changes the physics and never leaks into a
gradient — but pass an explicit basis if the frame itself has to be
differentiable.
"""
function _frame_from_normal(nvec; T = nothing, in_plane = nothing)
    length(nvec) == 3 ||
        throw(ArgumentError("a laminate normal has 3 components; got $(length(nvec))"))
    in_plane === nothing || length(in_plane) == 3 || throw(
        ArgumentError("`in_plane` is a 3-component axis; got $(length(in_plane))")
    )
    Tn = _floatlike(promote_type(typeof(nvec[1]), typeof(nvec[2]), typeof(nvec[3])))
    n = SVector{3, Tn}(ntuple(i -> convert(Tn, nvec[i]), 3))
    # The split is `is_hard_numeric`, NOT `Tn <: Real`: `Symbolics.Num` subtypes
    # `Real` yet answers no comparison, so a `Real` guard would let it through
    # and fail deep inside `argmin`.
    if !is_hard_numeric(Tn)
        n̂ = n / sqrt(n ⋅ n)
        ref = in_plane === nothing ? (1, 0, 0) : in_plane
        return _gram_schmidt_frame(n̂, SVector{3, Tn}(ntuple(i -> convert(Tn, ref[i]), 3)))
    end
    nrm = sqrt(n ⋅ n)
    nrm > 0 || throw(ArgumentError("a laminate normal must be non-zero"))
    n̂ = n / nrm
    # The canonical laminate (n = e₃) keeps the canonical basis: exact, and it
    # lets the kernel skip the Bond rotation entirely.
    if in_plane === nothing && n̂ == SVector{3, Tn}(0, 0, 1)
        return TensND.CanonicalBasis{3, _basis_eltype(T === nothing ? Tn : T)}()
    end
    a = if in_plane === nothing
        k = argmin(abs.(n̂))
        SVector{3, Tn}(ntuple(i -> i == k ? one(Tn) : zero(Tn), 3))
    else
        SVector{3, Tn}(ntuple(i -> convert(Tn, in_plane[i]), 3))
    end
    return _gram_schmidt_frame(n̂, a)
end

# Shared completion, once `n̂` is normalized and the in-plane reference chosen.
# `(ℓ̂, m̂, n̂)` is right-handed: ℓ̂ × m̂ = n̂.
function _gram_schmidt_frame(n̂::SVector{3, T}, a::SVector{3, T}) where {T}
    ℓ = a - (a ⋅ n̂) * n̂
    ℓ̂ = ℓ / sqrt(ℓ ⋅ ℓ)
    m̂ = n̂ × ℓ̂
    R = [ifelse(j == 1, ℓ̂[i], ifelse(j == 2, m̂[i], n̂[i])) for i in 1:3, j in 1:3]
    # `tsimplify` is the identity on every numeric element type, so this costs
    # the numeric path nothing. On a symbolic one it is what turns
    # `sin(θ)/√(sin²θ + cos²θ)` back into `sin(θ)` — worth doing ONCE here
    # rather than leaving the normalization to resurface in every component of
    # every result, and in `show`.
    return TensND.RotatedBasis(TensND.tsimplify(R))
end

"""
    _default_basis(::Type{T}, euler_angles)

Build the default TensND basis associated with a set of ZYZ Euler
angles: `CanonicalBasis{3,_basis_eltype(T)}` when all (normalized)
angles are zero, `RotatedBasis(normalized...)` otherwise.  Accepts any
tuple of length 0–3 with heterogeneous `Number` entries, symbolic ones
included — see [`_normalize_euler`](@ref).
"""
function _default_basis(::Type{T}, euler_angles::Tuple{Vararg{Number}}) where {T}
    normalized = _normalize_euler(euler_angles)
    # A symbolic angle set carries its own element type; forcing the declared
    # floor would put a `Float64` frame under symbolic data, which is exactly
    # what makes a `1.0` reappear in an otherwise exact result.
    Tbasis = _basis_eltype(promote_type(T, eltype(normalized)))
    return all(iszero, normalized) ? TensND.CanonicalBasis{3, Tbasis}() :
        TensND.RotatedBasis(normalized...)
end

"""
    _frame_matrix(basis) -> Matrix

The 3×3 (or 2×2) rotation matrix of a TensND basis, **keeping the basis element
type** — columns are the local axes expressed in the canonical frame.

The counterpart of [`_basis_matrix`](@ref), which pins the result to `Float64`
because its callers feed purely numerical kernels (cubature, multipole sums,
axis permutations). Anything that has to survive a symbolic frame — the laminate
does — must go through this one instead: `Float64(::Sym)` throws.
"""
function _frame_matrix(basis::TensND.AbstractBasis)
    E = TensND.vecbasis(basis, :cov)
    d = size(E, 1)
    return [E[i, j] for i in 1:d, j in 1:d]
end

# ─── Basis-permutation helpers ────────────────────────────────────────────────
#
# When the user enters semi-axes in a non-descending order, the
# constructors sort them internally and permute the associated basis
# columns to preserve the physical geometry (shape_tensor in the
# canonical frame).  The helpers below build a new RotatedBasis from an
# existing basis with its columns permuted and, if needed, one column
# flipped in sign to keep det = +1.

"""
    _basis_matrix(basis) -> Matrix{Float64}

Extract the 3×3 (or 2×2) rotation matrix associated with a TensND basis
as a `Matrix{Float64}` — columns are the local axes expressed in the
canonical frame.  See [`_frame_matrix`](@ref) for the element-type-preserving
variant.
"""
function _basis_matrix(basis::TensND.AbstractBasis)
    E = TensND.vecbasis(basis, :cov)
    d = size(E, 1)
    M = zeros(Float64, d, d)
    @inbounds for j in 1:d, i in 1:d
        M[i, j] = Float64(E[i, j])
    end
    return M
end

"""
    _permute_basis_3d(basis, σ::NTuple{3,Int}) -> RotatedBasis

Return a new 3D basis whose column `k` is the `σ[k]`-th column of
`basis`.  If `σ` is an odd permutation, the 3rd column is negated to
preserve a right-handed (det = +1) frame — physically equivalent since
flipping an axis direction leaves an ellipsoid invariant.
"""
function _permute_basis_3d(basis::TensND.AbstractBasis, σ::NTuple{3, Int})
    M = _basis_matrix(basis)
    M′ = similar(M)
    @inbounds for k in 1:3
        M′[:, k] = M[:, σ[k]]
    end
    # Sign of permutation (1,2,3) → even; one transposition → odd; …
    _permutation_sign(σ) == -1 && (M′[:, 3] .*= -1)
    return TensND.RotatedBasis(M′)
end

"""
    _permute_basis_2d(basis, swap::Bool) -> RotatedBasis

Return a new 2D basis: the original basis unchanged when `swap=false`,
or the basis with columns `(1, 2)` swapped and column 2 negated when
`swap=true` (preserves det = +1).
"""
function _permute_basis_2d(basis::TensND.AbstractBasis, swap::Bool)
    M = _basis_matrix(basis)
    swap || return TensND.RotatedBasis(M)
    M′ = similar(M)
    @inbounds begin
        M′[:, 1] = M[:, 2]
        M′[:, 2] = -M[:, 1]
    end
    return TensND.RotatedBasis(M′)
end

# Sign of a permutation of {1,2,3} (σ as NTuple{3,Int}).
@inline function _permutation_sign(σ::NTuple{3, Int})
    inv = 0
    @inbounds for i in 1:3, j in (i + 1):3
        σ[i] > σ[j] && (inv += 1)
    end
    return isodd(inv) ? -1 : 1
end

"""
    _sort_axes_and_basis(axes, basis, layout::Symbol) -> (sorted_axes, new_basis)

Sort `axes` into descending order and permute the columns of `basis`
accordingly to preserve the physical geometry of the inclusion.

Supported layouts:
- `:ellipsoid_3d` — full 3D permutation of `(a, b, c)` over columns 1,2,3.
- `:ellipsoid_2d` — 2D permutation of `(a, b)` over columns 1,2.
- `:cylinder`    — swap columns 2,3 if the two transverse axes are in
  ascending order (column 1 = cylinder axis, fixed).
- `:crack`       — swap columns 1,2 if `b > a` (column 3 = crack
  normal, fixed).

Symbolic or non-Real element types are returned untouched (no
comparison is performed).
"""
function _sort_axes_and_basis(
        axes::NTuple{3, T}, basis::TensND.AbstractBasis, layout::Symbol
    ) where {T}
    is_hard_numeric(T) || return (axes, basis)
    if layout === :ellipsoid_3d
        σ = Tuple(sortperm(collect(axes); rev = true))::NTuple{3, Int}
        σ == (1, 2, 3) && return (axes, basis)
        sorted = (axes[σ[1]], axes[σ[2]], axes[σ[3]])
        return (sorted, _permute_basis_3d(basis, σ))
    elseif layout === :cylinder
        # axes = (axis, b, c) with axes[1] ≡ Inf placeholder — ignore
        # here; caller passes (b, c).  This 3-tuple branch is unused.
        throw(ArgumentError("layout `:cylinder` expects a 2-tuple (b, c)"))
    end
    throw(ArgumentError("unknown layout $(layout) for 3-tuple axes"))
end

function _sort_axes_and_basis(
        axes::NTuple{2, T}, basis::TensND.AbstractBasis, layout::Symbol
    ) where {T}
    is_hard_numeric(T) || return (axes, basis)
    a, b = axes
    swap = b > a
    if layout === :ellipsoid_2d
        swap || return (axes, basis)
        return ((b, a), _permute_basis_2d(basis, true))
    elseif layout === :cylinder
        # axes = (b, c); cylinder axis is column 1 and stays fixed.
        swap || return (axes, basis)
        return ((b, a), _permute_basis_cols23(basis))
    elseif layout === :crack
        # axes = (a, b); crack normal is column 3 and stays fixed.
        swap || return (axes, basis)
        return ((b, a), _permute_basis_cols12(basis))
    end
    throw(ArgumentError("unknown layout $(layout) for 2-tuple axes"))
end

"""
    _permute_basis_cols23(basis) -> RotatedBasis

Swap the 2nd and 3rd columns of a 3D basis (column 1 fixed) and negate
column 3 to keep det = +1.  Used by `Cylinder`.
"""
function _permute_basis_cols23(basis::TensND.AbstractBasis)
    M = _basis_matrix(basis)
    M′ = copy(M)
    @inbounds begin
        M′[:, 2] = M[:, 3]
        M′[:, 3] = -M[:, 2]
    end
    return TensND.RotatedBasis(M′)
end

"""
    _permute_basis_cols12(basis) -> RotatedBasis

Swap the 1st and 2nd columns of a 3D basis (column 3 fixed) and negate
column 2 to keep det = +1.  Used by `EllipticCrack`.
"""
function _permute_basis_cols12(basis::TensND.AbstractBasis)
    M = _basis_matrix(basis)
    M′ = copy(M)
    @inbounds begin
        M′[:, 1] = M[:, 2]
        M′[:, 2] = -M[:, 1]
    end
    return TensND.RotatedBasis(M′)
end

# ─── Shape classification helpers ─────────────────────────────────────────────

"""
    _classify_shape_3d(::Type{T}, a, b, c)

Classify a 3D ellipsoid from its (already sorted) semi-axes into one of
the shape traits `Spherical`, `Prolate`, `Oblate`, `Triaxial`.  The
actual shape types are defined in the `Elasticity` sub-module; the
classification logic is delegated to a caller-supplied tuple of type
tags — this keeps `Core` dependency-free from `Elasticity`.

Returns an integer code instead of a type to avoid circular references:
    1 → equivalent of `Spherical`   (a == b == c)
    2 → equivalent of `Prolate`     (a > b == c)
    3 → equivalent of `Oblate`      (a == b > c)
    4 → equivalent of `Triaxial`    (a > b > c)
"""
function _classify_shape_3d(::Type{T}, a, b, c) where {T}
    if is_hard_numeric(T)
        tol = max(a, b, c) * (1.0e-10 * one(T))
        AeqB = (a - b) ≤ tol
        BeqC = (b - c) ≤ tol
    else
        AeqB = isequal(a, b)
        BeqC = isequal(b, c)
    end
    if AeqB &&  BeqC
        return 1   # Spherical
    elseif !AeqB &&  BeqC
        return 2   # Prolate
    elseif AeqB && !BeqC
        return 3   # Oblate
    else
        return 4  # Triaxial
    end
end

"""
    _classify_shape_2d(::Type{T}, a, b)

Classify a 2D ellipse from its (already sorted) semi-axes:
    1 → equivalent of `Circular`  (a == b)
    2 → equivalent of `Elliptic`  (a > b)
"""
function _classify_shape_2d(::Type{T}, a, b) where {T}
    is_equal = is_hard_numeric(T) ? (a - b) ≤ max(a, b) * (1.0e-10 * one(T)) : isequal(a, b)
    return is_equal ? 1 : 2
end
