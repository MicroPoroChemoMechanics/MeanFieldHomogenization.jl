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
    # The columns of `vecbasis(ℬ, :cov)` are the basis vectors in canonical
    # components, whatever the basis: no tensor to build and read back.
    E = TensND.vecbasis(basis, :cov)
    return E[:, 1], E[:, 2], E[:, 3]
end

"""
    _normalize_euler(angles) -> NTuple{3}

Pad a ZYZ Euler-angle tuple of length 0–3 to a full triple, promoting every
entry to one common type.  Missing trailing angles default to `0`.  Accepts
heterogeneous tuples (`Int`, `Irrational`, `Float32/64`, `Dual`, …) thanks to
`promote_type` + `float`, and **symbolic** ones (`Sym`, `Num`), which keep their
own type rather than being forced through `Float64`.

```julia
_normalize_euler(())               == (0.0, 0.0, 0.0)
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
    T_promoted = promote_type(map(typeof, angles)...)
    # `float` is what turns an `Int` or an `Irrational` angle into a usable one,
    # and leaves a `ForwardDiff.Dual` a dual (which is not an `AbstractFloat`);
    # it is meaningless on a symbolic type, and `Symbolics.Num` subtypes `Real`
    # without being convertible from `Float64`, so `is_hard_numeric` — not
    # `<: Real` — is what decides here.
    T = is_hard_numeric(T_promoted) ? float(T_promoted) : T_promoted
    return ntuple(i -> i ≤ n ? convert(T, angles[i]) : zero(T), 3)
end

# No angle: the identity rotation (`promote_type()` of no type is `Union{}`).
_normalize_euler(::Tuple{}) = (0.0, 0.0, 0.0)

"""
    _frame_from_normal(n; T = nothing, in_plane = nothing) -> AbstractBasis

Complete a normal vector `n` into an orthonormal frame
``(\\underline{\\ell}, \\underline{m}, \\underline{n})`` whose **third** axis is `n`
normalized, by Gram-Schmidt against a reference axis: TensND's
`frame_from_axis`, after the checks a laminate normal needs.
Returns a `CanonicalBasis` when `n` is exactly the third canonical axis, so the
common laminate needs no rotation at all.

`T` is the element-type floor of the cell the frame belongs to, and it decides
the element type of that `CanonicalBasis` short-circuit alone: a symbolic cell
keeps a symbolic canonical frame, whose axes are the exact `0` and `1` rather
than `0.0` and `1.0`. That distinction is not cosmetic — a `TensTI` converts its
axis to the element type of its *data*, so a `Float64` `1.0` reappears as a
symbolic `1.0` multiplying every coefficient of the result.

`in_plane` is the reference axis Gram-Schmidt orthogonalizes against, i.e. what
fixes ``\\underline{\\ell}`` in the plane of the layers; it must not be parallel to ``\\underline{n}``. Left
unset, a **numeric** normal picks whichever canonical axis is least aligned with
``\\underline{n}``, which can never degenerate; a **symbolic** normal cannot answer that
comparison, so it takes ``\\underline{e}_1``, or ``\\underline{e}_2`` when ``\\underline{n}`` lies along ``\\underline{e}_1``
identically.

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
    # `Real` yet answers no comparison.
    if is_hard_numeric(Tn)
        iszero(n) && throw(ArgumentError("a laminate normal must be non-zero"))
        # The canonical laminate (n = e₃) keeps the canonical basis in the
        # element type of the cell, so that a symbolic cell keeps exact 0 and 1.
        in_plane === nothing && n / sqrt(sum(abs2, n)) == SVector{3, Tn}(0, 0, 1) &&
            return TensND.CanonicalBasis{3, _basis_eltype(T === nothing ? Tn : T)}()
    end
    return TensND.frame_from_axis(n; in_plane)
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
_frame_matrix(basis::TensND.AbstractBasis) = Matrix(TensND.vecbasis(basis, :cov))

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
canonical frame — for the code that is Float64 by construction (the
quadrature oracle of the pair tensors). Anything that may meet a dual or
symbolic frame reads [`_frame_matrix`](@ref), which keeps the element type.
"""
function _basis_matrix(basis::TensND.AbstractBasis)
    return Matrix{Float64}(TensND.vecbasis(basis, :cov))
end

"""
    _sort_axes_and_basis(axes, basis, layout::Val) -> (sorted_axes, new_basis)

Sort `axes` into descending order and permute the columns of `basis`
accordingly to preserve the physical geometry of the inclusion.

Supported layouts:
- `Val(:ellipsoid_3d)` — full 3D permutation of ``(a, b, c)`` over columns 1,2,3.
- `Val(:ellipsoid_2d)` — 2D permutation of ``(a, b)`` over columns 1,2.
- `Val(:cylinder)`    — swap columns 2,3 if the two transverse axes are in
  ascending order (column 1 = cylinder axis, fixed).
- `Val(:crack)`       — swap columns 1,2 if ``b > a`` (column 3 = crack
  normal, fixed).

Symbolic or non-Real element types are returned untouched (no
comparison is performed).
"""
function _sort_axes_and_basis(
        axes::NTuple{3, T}, basis::TensND.AbstractBasis, ::Val{:ellipsoid_3d}
    ) where {T}
    is_hard_numeric(T) || return (axes, basis)
    σ = Tuple(sortperm(collect(axes); rev = true))::NTuple{3, Int}
    σ == (1, 2, 3) && return (axes, basis)
    sorted = (axes[σ[1]], axes[σ[2]], axes[σ[3]])
    return (sorted, TensND.permute_basis(basis, σ))
end

function _sort_axes_and_basis(
        axes::NTuple{2, T}, basis::TensND.AbstractBasis, layout::Val
    ) where {T}
    is_hard_numeric(T) || return (axes, basis)
    a, b = axes
    b > a || return (axes, basis)
    return ((b, a), _swap_axes_basis(basis, layout))
end

# The two columns a 2-axis layout swaps, the frame kept right-handed by
# reversing one vector (`TensND.permute_basis`): the second in 2-D; for a
# cylinder (axes = (b, c), its axis in column 1, fixed) the third; for a crack
# (axes = (a, b), its normal in column 3, fixed) the second, so that the normal
# keeps its sign.
_swap_axes_basis(basis, ::Val{:ellipsoid_2d}) = TensND.permute_basis(basis, (2, 1))
_swap_axes_basis(basis, ::Val{:cylinder}) = TensND.permute_basis(basis, (1, 3, 2))
_swap_axes_basis(basis, ::Val{:crack}) = TensND.permute_basis(basis, (2, 1, 3); negate = 2)

# ─── Shape classification helpers ─────────────────────────────────────────────

"""
    _classify_shape_3d(::Type{T}, a, b, c)

Classify a 3D ellipsoid from its (already sorted) semi-axes into one of
the shape traits `Spherical`, `Prolate`, `Oblate`, `Triaxial`.  The
actual shape types are defined in the `Elasticity` sub-module; the
classification logic is delegated to a caller-supplied tuple of type
tags — this keeps `Core` dependency-free from `Elasticity`.

Returns an integer code instead of a type to avoid circular references:

- 1 → equivalent of `Spherical` (``a = b = c``)
- 2 → equivalent of `Prolate` (``a > b = c``)
- 3 → equivalent of `Oblate` (``a = b > c``)
- 4 → equivalent of `Triaxial` (``a > b > c``)
"""
function _classify_shape_3d(::Type{T}, a, b, c) where {T}
    if is_hard_numeric(T)
        AeqB = _coincident(a, b, 1.0e-10)
        BeqC = _coincident(b, c, 1.0e-10)
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

- 1 → equivalent of `Circular` (``a = b``)
- 2 → equivalent of `Elliptic` (``a > b``)
"""
function _classify_shape_2d(::Type{T}, a, b) where {T}
    is_equal = is_hard_numeric(T) ? _coincident(a, b, 1.0e-10) : isequal(a, b)
    return is_equal ? 1 : 2
end

"""
    _coincident(a, b, rtol) -> Bool

Whether two semi-axes are one: equal to within `rtol` (relative) and, when they
carry derivatives, with equal derivatives. Two dual semi-axes merged while their
derivatives differ would build a sphere, a spheroid or a circle that no longer
depends on their gap, and the derivative along it would be lost without a word.
"""
@inline _coincident(a, b, rtol) =
    abs(a - b) ≤ max(abs(a), abs(b)) * rtol && !carries_partials(a - b)

"""
    _same_semi_axis(a, b) -> Bool

Whether two semi-axes are exactly equal, derivatives included: the condition for
the closed forms of a spheroid to be exact in value and in derivative.
"""
@inline _same_semi_axis(a, b) = a == b && !carries_partials(a - b)

"""
    _one_semi_axis(a, b) -> Bool

Whether two semi-axes are one, in any scalar type: [`_same_semi_axis`](@ref) where
comparisons return a `Bool`, the same expression for a symbolic type, where they
do not (`Symbolics.Num <: Real` notwithstanding).
"""
@inline _one_semi_axis(a::T, b::T) where {T} = _one_semi_axis(Val(is_hard_numeric(T)), a, b)
@inline _one_semi_axis(::Val{true}, a, b) = _same_semi_axis(a, b)
@inline _one_semi_axis(::Val{false}, a, b) = isequal(a, b)

# Whether `x` carries a derivative at any nesting level: TensND's `carries_partials`.
