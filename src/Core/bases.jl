# =============================================================================
#  bases.jl
#
#  The axis conventions of the inclusions (sorting their semi-axes, permuting
#  their frame with them, classifying their shape) and the checks a laminate
#  normal needs. Bases themselves — from Euler angles (`euler_basis`), about an
#  axis (`frame_from_axis`), permuted (`permute_basis`), and the element type of
#  a canonical one (`canonical_eltype`) — are TensND's.
# =============================================================================

"""
    _frame_from_normal(n; T = nothing, in_plane = nothing) -> AbstractBasis

The frame of a laminate of normal `n`: TensND's `frame_from_axis(n; in_plane, T)`,
whose third axis is `n` normalized, after the checks a laminate normal needs.
The canonical laminate (`n` along `e₃`, no `in_plane`) gets the canonical basis
in the element type of its cell `T`, so that a symbolic cell keeps exact `0` and
`1`. The effective property of a laminate is invariant under rotation about its
normal, so the in-plane axes never change the physics.
"""
function _frame_from_normal(nvec; T = nothing, in_plane = nothing)
    length(nvec) == 3 ||
        throw(ArgumentError("a laminate normal has 3 components; got $(length(nvec))"))
    in_plane === nothing || length(in_plane) == 3 || throw(
        ArgumentError("`in_plane` is a 3-component axis; got $(length(in_plane))")
    )
    return TensND.frame_from_axis(nvec; in_plane, T)
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
