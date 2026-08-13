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
    _normalize_euler(angles) -> NTuple{3, <:AbstractFloat}

Pad a ZYZ Euler-angle tuple of length 0–3 to a full triple, promoting
every entry to a common floating-point type.  Missing trailing angles
default to `0`.  Accepts heterogeneous tuples (`Int`, `Irrational`,
`Float32/64`, `Dual`, …) thanks to `promote_type` + `float`.

```
_normalize_euler(())                 == (0.0, 0.0, 0.0)
_normalize_euler((π/2,))             == (π/2, 0.0, 0.0)       # Float64
_normalize_euler((π, 0, 0))          == (π, 0.0, 0.0)          # Irrational → Float
_normalize_euler((0, 1, 2))          == (0.0, 1.0, 2.0)
```

A tuple of length > 3 raises `ArgumentError`.
"""
function _normalize_euler(angles::Tuple{Vararg{Real}})
    n = length(angles)
    n > 3 && throw(
        ArgumentError(
            "euler_angles accepts at most 3 ZYZ angles; got $n values ($(angles))."
        )
    )
    if n == 0
        return (0.0, 0.0, 0.0)
    end
    T_promoted = float(promote_type(map(typeof, angles)...))
    T = T_promoted <: AbstractFloat ? T_promoted : Float64
    padded = ntuple(i -> i ≤ n ? T(angles[i]) : zero(T), 3)
    return padded
end

# Convenience forwarding methods for common concrete-type patterns.
_normalize_euler(::Tuple{}) = (0.0, 0.0, 0.0)

"""
    _frame_from_normal(n) -> AbstractBasis

Complete a normal vector `n` into an orthonormal frame `(ℓ, m, n̂)` whose
**third** axis is `n̂ = n/‖n‖`, by Gram-Schmidt against whichever canonical
axis is least aligned with `n̂`. Returns a `CanonicalBasis` when `n` is
exactly the third canonical axis, so the common laminate needs no rotation at
all.

The in-plane pair `(ℓ, m)` is chosen discretely and the effective property of
a laminate is invariant under rotation about its normal, so this choice never
leaks into a gradient — but pass an explicit basis if the frame itself has to
be differentiable.

Requires a `Real` element type: the axis selection is a comparison, which a
symbolic normal cannot answer. Build the basis explicitly in that case.
"""
function _frame_from_normal(nvec)
    length(nvec) == 3 ||
        throw(ArgumentError("a laminate normal has 3 components; got $(length(nvec))"))
    T0 = promote_type(typeof(nvec[1]), typeof(nvec[2]), typeof(nvec[3]))
    T0 <: Real || throw(
        ArgumentError(
            "_frame_from_normal needs a Real normal (the in-plane axis choice is a " *
                "comparison); pass an explicit `basis = …` for a symbolic frame"
        )
    )
    T = _floatlike(T0)
    n = SVector{3, T}(T(nvec[1]), T(nvec[2]), T(nvec[3]))
    nrm = sqrt(n ⋅ n)
    nrm > 0 || throw(ArgumentError("a laminate normal must be non-zero"))
    n̂ = n / nrm
    # The canonical laminate (n = e₃) keeps the canonical basis: exact, and it
    # lets the kernel skip the Bond rotation entirely.
    n̂ == SVector{3, T}(0, 0, 1) && return TensND.CanonicalBasis{3, _basis_eltype(T)}()
    k = argmin(abs.(n̂))
    a = SVector{3, T}(ntuple(i -> i == k ? one(T) : zero(T), 3))
    ℓ = a - (a ⋅ n̂) * n̂
    ℓ̂ = ℓ / sqrt(ℓ ⋅ ℓ)
    m̂ = n̂ × ℓ̂                       # (ℓ̂, m̂, n̂) right-handed: ℓ̂ × m̂ = n̂
    R = zeros(T, 3, 3)
    @inbounds for i in 1:3
        R[i, 1] = ℓ̂[i]
        R[i, 2] = m̂[i]
        R[i, 3] = n̂[i]
    end
    return TensND.RotatedBasis(R)
end

"""
    _default_basis(::Type{T}, euler_angles)

Build the default TensND basis associated with a set of ZYZ Euler
angles: `CanonicalBasis{3,_basis_eltype(T)}` when all (normalized)
angles are zero, `RotatedBasis(normalized...)` otherwise.  Accepts any
tuple of length 0–3 with heterogeneous `Real` entries — see
[`_normalize_euler`](@ref).
"""
function _default_basis(::Type{T}, euler_angles::Tuple{Vararg{Real}}) where {T}
    Tbasis = _basis_eltype(T)
    normalized = _normalize_euler(euler_angles)
    return all(iszero, normalized) ? TensND.CanonicalBasis{3, Tbasis}() :
        TensND.RotatedBasis(normalized...)
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
canonical frame.
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
