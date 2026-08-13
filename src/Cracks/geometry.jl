# =============================================================================
#  geometry.jl
#
#  Crack geometry types subtyping `Core.AbstractCrack{T}`.
# =============================================================================

"""
    CrackShape

Abstract supertype used as the second type parameter of
[`EllipticCrack`](@ref).  Concrete subtypes: [`Penny`](@ref) (a == b) and
[`EllipticShape`](@ref) (a > b).
"""
abstract type CrackShape end

"Circular (penny-shaped) flat crack: a == b."
struct Penny <: CrackShape end

"General flat elliptical crack: a > b."
struct EllipticShape <: CrackShape end

"""
    Ribbon

Placeholder type tag for ribbon-like cracks.  Mirrors the role of
[`CrackShape`](@ref) for elliptical cracks.
"""
struct Ribbon end

# =============================================================================
#  EllipticCrack
# =============================================================================

"""
    EllipticCrack{T, S<:CrackShape, B<:AbstractBasis}

Flat elliptical crack with semi-axes `a ≥ b` oriented by a local basis
`ℬ = (l̂, m̂, n̂)`.
"""
struct EllipticCrack{T <: Number, S <: CrackShape, B <: TensND.AbstractBasis} <:
    MFH_Core.AbstractCrack{T}
    a::T
    b::T
    basis::B
end

# =============================================================================
#  RibbonCrack
# =============================================================================

"""
    RibbonCrack{T, B<:AbstractBasis}

Ribbon-like (tunnel) crack of half-width `b` along `m̂`, unbounded along
`l̂`, with normal `n̂`.
"""
struct RibbonCrack{T <: Number, B <: TensND.AbstractBasis} <: MFH_Core.AbstractCrack{T}
    b::T
    basis::B
end

# =============================================================================
#  Internal helpers
# =============================================================================

function _classify_crack(::Type{T}, a, b) where {T}
    if is_hard_numeric(T)
        tol = max(a, b) * (1.0e-10 * one(T))
        return (a - b) ≤ tol ? Penny : EllipticShape
    else
        return isequal(a, b) ? Penny : EllipticShape
    end
end

# =============================================================================
#  Constructors
# =============================================================================

"""
    EllipticCrack(a, b; euler_angles=(0,0,0))

Elliptical flat crack with semi-axes `a` and `b` oriented by ZYZ Euler
angles.  `euler_angles` accepts tuples of length 0–3 with heterogeneous
`Real` entries (trailing zeros are implicit).

**Input-order convention** (`T <: Real`): columns 1 and 2 of the local
basis carry the user's `a` and `b` respectively; column 3 is the crack
normal.  The stored semi-axes are sorted so that `a ≥ b` and columns 1
and 2 are permuted when needed to preserve the physical geometry.
Column 3 (normal) is never touched.
"""
function EllipticCrack(
        a::Ta, b::Tb;
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {Ta, Tb}
    T = MFH_Core._floatlike(promote_type(Ta, Tb))
    basis_in = MFH_Core._default_basis(T, euler_angles)
    axes_sorted, basis = MFH_Core._sort_axes_and_basis((T(a), T(b)), basis_in, :crack)
    a_, b_ = axes_sorted
    S = _classify_crack(T, a_, b_)
    return EllipticCrack{T, S, typeof(basis)}(a_, b_, basis)
end

"""
    EllipticCrack(a, b, R::AbstractMatrix)

Elliptical flat crack whose local frame columns are the columns of `R`.
"""
function EllipticCrack(a::Ta, b::Tb, R::AbstractMatrix) where {Ta, Tb}
    T = MFH_Core._floatlike(promote_type(Ta, Tb))
    basis_in = TensND.RotatedBasis(Matrix{Float64}(R))
    axes_sorted, basis = MFH_Core._sort_axes_and_basis((T(a), T(b)), basis_in, :crack)
    a_, b_ = axes_sorted
    S = _classify_crack(T, a_, b_)
    return EllipticCrack{T, S, typeof(basis)}(a_, b_, basis)
end

"""
    EllipticCrack(a, b, basis::AbstractBasis)

Elliptical flat crack with an already-constructed TensND basis.
"""
function EllipticCrack(a::Ta, b::Tb, basis::TensND.AbstractBasis) where {Ta, Tb}
    T = MFH_Core._floatlike(promote_type(Ta, Tb))
    axes_sorted, basis_out = MFH_Core._sort_axes_and_basis((T(a), T(b)), basis, :crack)
    a_, b_ = axes_sorted
    S = _classify_crack(T, a_, b_)
    return EllipticCrack{T, S, typeof(basis_out)}(a_, b_, basis_out)
end

"""
    PennyCrack(a; euler_angles=(0,0,0))

Convenience constructor for a circular (penny-shaped) flat crack.
"""
PennyCrack(a; euler_angles::Tuple{Vararg{Real}} = ()) =
    EllipticCrack(a, a; euler_angles = euler_angles)

"""
    RibbonCrack(b; euler_angles=(0,0,0))

Ribbon-like (tunnel) crack of half-width `b`.
"""
function RibbonCrack(
        b::Tb;
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {Tb}
    T = MFH_Core._floatlike(Tb)
    basis = MFH_Core._default_basis(T, euler_angles)
    return RibbonCrack{T, typeof(basis)}(T(b), basis)
end

"""
    RibbonCrack(b, R::AbstractMatrix)

Ribbon-like crack with local frame columns `R = [l̂ | m̂ | n̂]`.
"""
function RibbonCrack(b::Tb, R::AbstractMatrix) where {Tb}
    T = MFH_Core._floatlike(Tb)
    basis = TensND.RotatedBasis(Matrix{Float64}(R))
    return RibbonCrack{T, typeof(basis)}(T(b), basis)
end

"""
    RibbonCrack(b, basis::AbstractBasis)

Ribbon-like crack with an already-constructed TensND basis.
"""
function RibbonCrack(b::Tb, basis::TensND.AbstractBasis) where {Tb}
    T = MFH_Core._floatlike(Tb)
    return RibbonCrack{T, typeof(basis)}(T(b), basis)
end

# =============================================================================
#  Equality and hashing (field-wise)
# =============================================================================

Base.:(==)(x::T, y::T) where {T <: EllipticCrack} =
    x.a == y.a && x.b == y.b && x.basis == y.basis

function Base.hash(x::EllipticCrack, h::UInt)
    h = hash(typeof(x), h)
    h = hash(x.a, h)
    h = hash(x.b, h)
    return hash(x.basis, h)
end

Base.:(==)(x::T, y::T) where {T <: RibbonCrack} =
    x.b == y.b && x.basis == y.basis

function Base.hash(x::RibbonCrack, h::UInt)
    h = hash(typeof(x), h)
    h = hash(x.b, h)
    return hash(x.basis, h)
end

# =============================================================================
#  Interface implementations (Core abstractions)
# =============================================================================

MFH_Core.dimension(::MFH_Core.AbstractCrack) = 3
MFH_Core.inclusion_basis(c::MFH_Core.AbstractCrack) = c.basis
MFH_Core.shape_trait(::EllipticCrack{T, S}) where {T, S} = S
MFH_Core.shape_trait(::RibbonCrack) = Ribbon

"""
    shape_tensor(c::EllipticCrack) -> AbstractTens{2,3}

Return the symmetric 2nd-order shape tensor of a flat elliptic (or
penny-shaped) crack: the diagonal in the local frame `(l̂, m̂, n̂)` is
`(a, b, 0)` — the zero entry reflects the vanishing extent along the
crack normal `n̂` (column 3 of the basis).
"""
function MFH_Core.shape_tensor(c::EllipticCrack{T}) where {T}
    D = zeros(T, 3, 3)
    D[1, 1] = c.a
    D[2, 2] = c.b
    return TensND.Tens(D, c.basis)
end

"""
    shape_tensor(c::RibbonCrack) -> AbstractTens{2,3}

Return the symmetric 2nd-order shape tensor of a ribbon (tunnel) crack:
the diagonal in the local frame `(l̂, m̂, n̂)` is `(Inf, b, 0)` — `Inf`
along the tunnel axis `l̂`, zero along the crack normal `n̂`.
"""
function MFH_Core.shape_tensor(c::RibbonCrack{T}) where {T}
    D = zeros(T, 3, 3)
    D[1, 1] = T(Inf)
    D[2, 2] = c.b
    return TensND.Tens(D, c.basis)
end

# =============================================================================
#  Accessors
# =============================================================================

"Return the local basis ``(\\hat{\\mathbf l}, \\hat{\\mathbf m}, \\hat{\\mathbf n})`` of the crack."
crack_basis(c::MFH_Core.AbstractCrack) = c.basis

"Return the unit normal ``\\hat{\\mathbf n}`` of the crack plane."
crack_normal(c::MFH_Core.AbstractCrack) = TensND.tens_basis(c.basis, 3)

"""
    aspect_ratio(c)

Aspect ratio ``η = b/a`` for an elliptical crack, or `zero(T)` for a
[`RibbonCrack`](@ref) (limit case).
"""
aspect_ratio(c::EllipticCrack) = c.b / c.a
aspect_ratio(c::RibbonCrack{T}) where {T} = zero(T)

"Semi-major axis of an elliptical crack."
semi_major(c::EllipticCrack) = c.a
semi_major(c::RibbonCrack) = error("`semi_major` is undefined for a ribbon crack (infinite).  Use `semi_minor` for the half-width.")

"Semi-minor axis of an elliptical crack, or half-width of a ribbon crack."
semi_minor(c::EllipticCrack) = c.b
semi_minor(c::RibbonCrack) = c.b

"""
    _crack_local_frame(c, C₀) -> (Carr, l̂, m̂, n̂, ℬ)

Everything the anisotropic COD kernels need, expressed **in one single frame**:
the components of `C₀` in the crack basis `ℬ = crack_basis(c)`, together with
the crack's own frame vectors *in that same basis* — where they are, by
construction, the canonical triad ``(\\mathbf e_1, \\mathbf e_2, \\mathbf e_3)``
with ``\\mathbf e_3 = \\hat{\\mathbf n}``.

Frame consistency is the whole point. `MFH_Core._frame_columns(ℬ)` returns the
crack frame in *global* coordinates; combining those with components expressed
in `ℬ` mixes two frames and yields a wrong COD for every crack that is not
aligned with the canonical axes. The aligned case coincides, which is exactly
why such a mix can survive a test suite built on aligned cracks.

The COD returned by the kernels is then in the crack basis too, so wrapping it
as `Tens(…, ℬ)` is consistent.
"""
function _crack_local_frame(
        c::MFH_Core.AbstractCrack, C₀::TensND.AbstractTens{4, 3}
    )
    basis = crack_basis(c)
    Carr = MFH_Core._C_array(TensND.change_tens(C₀, basis))
    T = eltype(Carr)
    l̂ = T[one(T), zero(T), zero(T)]
    m̂ = T[zero(T), one(T), zero(T)]
    n̂ = T[zero(T), zero(T), one(T)]
    return Carr, l̂, m̂, n̂, basis
end

"""
    crack_chi(c)

Dimensionless coefficient ``χ`` linking the average crack opening to the
maximum opening:  ``χ^{\\mathcal E} = 2/3`` for an [`EllipticCrack`](@ref),
``χ^{\\mathcal R} = π/4`` for a [`RibbonCrack`](@ref).
"""
crack_chi(::EllipticCrack{T}) where {T <: Number} = T(2) / T(3)
crack_chi(::RibbonCrack{T}) where {T <: Number} = T(π) / T(4)

# =============================================================================
#  Pretty printing
# =============================================================================

function Base.show(io::IO, c::EllipticCrack{T, S}) where {T, S}
    shape = S === Penny ? "penny-shaped" : "elliptic"
    return print(io, "EllipticCrack{", T, "} (", shape, ", a=", c.a, ", b=", c.b, ")")
end

function Base.show(io::IO, c::RibbonCrack{T}) where {T}
    return print(io, "RibbonCrack{", T, "} (half-width b=", c.b, ")")
end
