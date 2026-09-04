# =============================================================================
#  cylinder.jl
#
#  Infinite-cylinder inclusion (`a → ∞`, transverse semi-axes `b ≥ c > 0`).
#  Subtypes `Core.AbstractEllipsoidalInclusion{3, T}`.  Shape classification
#  (`CircularCylindrical` vs `EllipticCylindrical`) happens at construction
#  time and is encoded in the type parameter `S` — no runtime branching in
#  downstream `_kernel` methods.
# =============================================================================

"""
    CylindricalShape

Abstract supertype for the shape classification of a [`Cylinder`](@ref).
Sub-traits:
- [`CircularCylindrical`](@ref) — circular base (`b = c`).
- [`EllipticCylindrical`](@ref) — elliptic base (`b > c`).

All cylinders are infinite along `e₁` in the inclusion's local basis.
"""
abstract type CylindricalShape <: EllipsoidShape end

"3-D infinite cylinder with circular base: `b = c`, `a → ∞`, axis = `e₁`."
struct CircularCylindrical <: CylindricalShape end

"3-D infinite cylinder with elliptic base: `b > c`, `a → ∞`, axis = `e₁`."
struct EllipticCylindrical <: CylindricalShape end

const _SHAPE_CYLINDER = (CircularCylindrical, EllipticCylindrical)

# =============================================================================
#  Cylinder struct
# =============================================================================

"""
    Cylinder{S<:CylindricalShape, T<:Number, B<:AbstractBasis} <:
        AbstractEllipsoidalInclusion{3, T}

Infinite cylindrical inclusion with transverse semi-axes `(b, c)` (with
`b ≥ c > 0` when `T <: Real`, or in the caller-provided order when `T` is
symbolic).  The cylinder axis is the first column of the local `basis`
— consistent with the [`Prolate`](@ref) convention where the axis of
revolution is also `e₁`.

`S` encodes the cross-section shape:
- [`CircularCylindrical`](@ref) when `b = c` (transversely isotropic result).
- [`EllipticCylindrical`](@ref) when `b > c` (orthotropic result).

`T` can be any `Number` subtype (`Float64`, `ForwardDiff.Dual`, `SymPy.Sym`,
`Symbolics.Num`, …) — all analytical paths are type-generic.
"""
struct Cylinder{S <: CylindricalShape, T <: Number, B <: TensND.AbstractBasis} <:
    MFH_Core.AbstractEllipsoidalInclusion{3, T}
    semi_axes::NTuple{2, T}
    basis::B
end

# ── Shape classification (internal) ──────────────────────────────────────────

function _classify_cylinder_shape(::Type{T}, b, c) where {T}
    if T <: Real
        is_equal = (b - c) ≤ max(b, c) * (1.0e-10 * one(T))
    else
        is_equal = isequal(b, c)
    end
    return is_equal ? 1 : 2  # 1 → CircularCylindrical, 2 → EllipticCylindrical
end

# ── Constructors ─────────────────────────────────────────────────────────────

"""
    Cylinder(b, c; euler_angles=(0,0,0))

Infinite cylinder with transverse semi-axes `b` and `c`.  The cylinder
axis is oriented by ZYZ Euler angles `(θ, ϕ, ψ)` — the default aligns
it with the global `e₁`.  `euler_angles` accepts tuples of length 0–3
with heterogeneous `Real` entries (trailing zeros are implicit).

**Input-order convention** (`T <: Real`): columns 2 and 3 of the local
basis carry the user's `b` and `c` respectively.  The stored
`semi_axes` are sorted descending and columns 2 and 3 are permuted when
needed to preserve the physical geometry.  Column 1 (cylinder axis) is
never touched.
"""
function Cylinder(
        b::Tb, c::Tc;
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Tb, Tc))
    basis_in = MFH_Core._default_basis(T, euler_angles)
    axes_sorted, basis = MFH_Core._sort_axes_and_basis((T(b), T(c)), basis_in, :cylinder)
    code = _classify_cylinder_shape(T, axes_sorted...)
    S = _SHAPE_CYLINDER[code]
    return Cylinder{S, T, typeof(basis)}(axes_sorted, basis)
end

"""
    Cylinder(b, c, R::AbstractMatrix)

Infinite cylinder whose local axes are the columns of the rotation matrix
`R` — column 1 is the cylinder axis, columns 2 and 3 the transverse
axes associated with `b` and `c` respectively.
"""
function Cylinder(b::Tb, c::Tc, R::AbstractMatrix) where {Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Tb, Tc))
    basis_in = TensND.RotatedBasis(Matrix{Float64}(R))
    axes_sorted, basis = MFH_Core._sort_axes_and_basis((T(b), T(c)), basis_in, :cylinder)
    code = _classify_cylinder_shape(T, axes_sorted...)
    S = _SHAPE_CYLINDER[code]
    return Cylinder{S, T, typeof(basis)}(axes_sorted, basis)
end

"""
    Cylinder(b, c, basis::TensND.AbstractBasis)

Infinite cylinder sharing an already-constructed TensND basis (no matrix
round-trip, no new basis allocated). Column 1 of `basis` is the
cylinder axis; columns 2 and 3 are the transverse axes associated with
`b` and `c` respectively.
"""
function Cylinder(b::Tb, c::Tc, basis::TensND.AbstractBasis) where {Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Tb, Tc))
    axes_sorted, basis_out = MFH_Core._sort_axes_and_basis((T(b), T(c)), basis, :cylinder)
    code = _classify_cylinder_shape(T, axes_sorted...)
    S = _SHAPE_CYLINDER[code]
    return Cylinder{S, T, typeof(basis_out)}(axes_sorted, basis_out)
end

"""
    Cylinder(b; euler_angles=(0,0,0))

Infinite circular cylinder with transverse radius `b` (shortcut for
`Cylinder(b, b; …)` that forces the [`CircularCylindrical`](@ref) shape
trait at compile time — especially useful for symbolic element types where
structural equality `isequal(b, b)` is the only reliable test).
"""
function Cylinder(
        b::T;
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {T <: Number}
    Tf = MFH_Core._floatlike(T)
    basis = MFH_Core._default_basis(Tf, euler_angles)
    return Cylinder{CircularCylindrical, Tf, typeof(basis)}((Tf(b), Tf(b)), basis)
end

# ── Equality and hashing (field-wise) ────────────────────────────────────────

Base.:(==)(x::T, y::T) where {T <: Cylinder} =
    x.semi_axes == y.semi_axes && x.basis == y.basis

function Base.hash(x::Cylinder, h::UInt)
    h = hash(typeof(x), h)
    h = hash(x.semi_axes, h)
    return hash(x.basis, h)
end

# ── Interface implementations ────────────────────────────────────────────────

MFH_Core.dimension(::Cylinder) = 3
MFH_Core.inclusion_basis(cyl::Cylinder) = cyl.basis
MFH_Core.shape_trait(::Cylinder{S}) where {S} = S

"""
    shape_tensor(cyl::Cylinder) -> AbstractTens{2,3}

Return the symmetric 2nd-order shape tensor of an infinite cylinder:
the diagonal in the local frame is `(Inf, b, c)` — the `Inf` entry
reflects the unbounded extent along the cylinder axis (column 1 of
`cyl.basis`).
"""
function MFH_Core.shape_tensor(cyl::Cylinder)
    T = eltype(cyl.semi_axes)
    D = zeros(T, 3, 3)
    D[1, 1] = T(Inf)
    D[2, 2] = cyl.semi_axes[1]
    D[3, 3] = cyl.semi_axes[2]
    return TensND.Tens(D, cyl.basis)
end

# ── Shape helpers ────────────────────────────────────────────────────────────

"Return the spatial dimension of the cylinder (always 3)."
get_dim(::Cylinder) = 3

"Return the `i`-th transverse semi-axis (1-indexed: 1 → `b`, 2 → `c`)."
semi_axis(cyl::Cylinder, i::Int) = cyl.semi_axes[i]

"Transverse aspect ratio `ρ = c / b` (≤ 1 when `T<:Real`)."
aspect_ratio_rho(cyl::Cylinder) = cyl.semi_axes[2] / cyl.semi_axes[1]

# ── Cylinder convenience wrapper around the Core Newton potentials ──────────

"""
    newton_potential_3d_cylinder(cyl::Cylinder)

Cylinder-level convenience wrapper that forwards to
[`Core.newton_potential_3d_cylinder`](@ref) using the transverse semi-axes
of `cyl`.
"""
MFH_Core.newton_potential_3d_cylinder(cyl::Cylinder) =
    MFH_Core.newton_potential_3d_cylinder(cyl.semi_axes[1], cyl.semi_axes[2])

# ── Dispatch rules (specialized for Cylinder) ────────────────────────────────
# Isotropic matrix → closed-form `Analytical` (Mura 1987 §11.22).  These
# explicit rules disambiguate against the generic
# `AbstractEllipsoidalInclusion + TensISO` rule in Core/dispatch.jl.
MFH_Core._resolve_algo(::Val, ::Cylinder, ::TensND.TensISO) = MFH_Core.Analytical()
MFH_Core._resolve_algo(::Val{:auto}, ::Cylinder, ::TensND.TensISO) = MFH_Core.Analytical()
MFH_Core._resolve_algo(::Val{:residues}, ::Cylinder, ::TensND.TensISO) = MFH_Core.Analytical()
MFH_Core._resolve_algo(::Val{:decuhr}, ::Cylinder, ::TensND.TensISO) = MFH_Core.Analytical()
MFH_Core._resolve_algo(::Val{:nestedquadgk}, ::Cylinder, ::TensND.TensISO) =
    MFH_Core.Analytical()

# Anisotropic matrix → dedicated 1D `CylinderQuadrature`.  The acoustic
# polynomial used by the residue algorithm degenerates for a cylinder
# (one root at infinity), so `:residues` is transparently remapped to the
# quadrature.  `:auto` picks the same 1D quadrature — cheaper than the
# 2D `DECUHR` route and ForwardDiff-safe.
MFH_Core._resolve_algo(::Val{:auto}, ::Cylinder, ::TensND.AbstractTens{4, 3}) =
    MFH_Core.CylinderQuadrature()
MFH_Core._resolve_algo(::Val{:residues}, ::Cylinder, ::TensND.AbstractTens{4, 3}) =
    MFH_Core.CylinderQuadrature()
MFH_Core._resolve_algo(::Val{:decuhr}, ::Cylinder, ::TensND.AbstractTens{4, 3}) =
    MFH_Core.CylinderQuadrature()
