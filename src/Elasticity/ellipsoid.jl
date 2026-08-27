# =============================================================================
#  ellipsoid.jl
#
#  Ellipsoidal-inclusion geometry type subtyping
#  `Core.AbstractEllipsoidalInclusion{dim,T}`. Shape classification
#  happens at construction time and is encoded in the type parameter `S`
#  — no runtime branching in downstream `_kernel` methods.
# =============================================================================

"""
    EllipsoidShape

Abstract supertype for the shape classification of an [`Ellipsoid`](@ref).

Concrete subtypes (3-D): [`Spherical`](@ref), [`Prolate`](@ref),
[`Oblate`](@ref), [`Triaxial`](@ref).

Concrete subtypes (2-D): [`Circular`](@ref), [`Elliptic`](@ref).
"""
abstract type EllipsoidShape end

"3-D sphere: a = b = c."
struct Spherical <: EllipsoidShape end

"3-D prolate spheroid: a > b = c (axis of revolution = e₁)."
struct Prolate <: EllipsoidShape end

"3-D oblate spheroid: a = b > c (axis of revolution = e₃)."
struct Oblate <: EllipsoidShape end

"3-D triaxial ellipsoid: a > b > c."
struct Triaxial <: EllipsoidShape end

"2-D circle: a = b."
struct Circular <: EllipsoidShape end

"2-D ellipse: a > b."
struct Elliptic <: EllipsoidShape end

# Map `_classify_shape_3d` / `_classify_shape_2d` integer codes to shape
# types (`Core` does not depend on `EllipsoidShape`, so we do the mapping
# here, in `Elasticity`).
const _SHAPE_3D = (Spherical, Prolate, Oblate, Triaxial)
const _SHAPE_2D = (Circular, Elliptic)

# =============================================================================
#  Ellipsoid struct
# =============================================================================

"""
    Ellipsoid{dim, S<:EllipsoidShape, T<:Number, B<:AbstractBasis}

Ellipsoidal inclusion of the Eshelby problem
([eshelby1957](@cite)). In the Echoes convention, an
ellipsoid ``\\mathcal E_{\\mathbf A}`` is described by an invertible
second-order shape tensor ``\\mathbf A`` through

```
x ∈ E_A  ⇔  x·(Aᵀ·A)⁻¹·x ≤ 1 ,
Aᵀ·A = Σᵢ ρᵢ² êᵢ^A ⊗ êᵢ^A ,   ρ₁=a ≥ ρ₂=b ≥ ρ₃=c .
```

`semi_axes` stores the eigenvalues ``\\rho_i`` (sorted in decreasing
order for real-valued types) and `basis` stores the orthonormal frame
``(\\hat{\\mathbf e}_i^{\\mathbf A})`` relative to the canonical frame.

The shape `S` is determined at construction time:
- 3-D: `Spherical`, `Prolate`, `Oblate`, or `Triaxial`
- 2-D: `Circular` or `Elliptic`

`T` can be any `Number` subtype (`Float64`, `ForwardDiff.Dual`,
`SymPy.Sym`, `Symbolics.Num`, …).
"""
struct Ellipsoid{dim, S <: EllipsoidShape, T <: Number, B <: TensND.AbstractBasis} <:
    MFH_Core.AbstractEllipsoidalInclusion{dim, T}
    semi_axes::NTuple{dim, T}
    basis::B
end

# ── 3-D constructors ──────────────────────────────────────────────────────────

# Internal helper: resolve degenerate real-valued semi-axes (one or two
# zeros / infs after sorting) to the dedicated inclusion type.  Returns
# `nothing` when the triple is a regular ellipsoid, else the pre-built
# inclusion (Cylinder / EllipticCrack / RibbonCrack).  Raises
# `ArgumentError` for unsupported combinations (slab, needle, …).
#
# The reference to the `Cracks` sub-module is resolved at *runtime* via
# `getfield` to avoid the circular dependency (`Elasticity` is loaded
# before `Cracks`).
function _resolve_degenerate_ellipsoid(axes_sorted::NTuple{3, T}, basis) where {T <: Real}
    n_inf = count(isinf, axes_sorted)
    n_zero = count(iszero, axes_sorted)

    n_inf ≥ 2 && throw(
        ArgumentError(
            "Ellipsoid with two or more infinite semi-axes (infinite slab / plane) is not supported."
        )
    )
    n_zero == 3 && throw(
        ArgumentError(
            "Ellipsoid with all semi-axes zero is not a physical inclusion."
        )
    )
    n_zero == 2 && throw(
        ArgumentError(
            "Ellipsoid with two zero semi-axes (needle / line) is not supported."
        )
    )

    # Regular case — no degeneracy
    (n_inf == 0 && n_zero == 0) && return nothing

    a1, a2, a3 = axes_sorted

    # Cylinder: one axis Inf, the other two finite > 0
    if n_inf == 1 && n_zero == 0
        return Cylinder{_SHAPE_CYLINDER[_classify_cylinder_shape(T, a2, a3)], T, typeof(basis)}(
            (a2, a3), basis
        )
    end

    # Cracks live in the sibling sub-module — resolve at runtime
    Cracks = getfield(parentmodule(@__MODULE__), :Cracks)

    # Ribbon (tunnel) crack: one axis Inf, one axis zero
    if n_inf == 1 && n_zero == 1
        return Base.invokelatest(getfield(Cracks, :RibbonCrack), a2, basis)
    end

    # Flat elliptic crack (penny or elliptic): one axis zero, two finite > 0
    if n_inf == 0 && n_zero == 1
        # EllipticCrack takes a ≥ b and a basis built from Euler angles.
        # We pass the axes in descending order and the pre-built basis.
        return Base.invokelatest(
            getfield(Cracks, :EllipticCrack), a1, a2, basis
        )
    end
    return nothing
end

_resolve_degenerate_ellipsoid(::NTuple{3, T}, _) where {T} = nothing

"""
    Ellipsoid(a, b, c; euler_angles=(θ,ϕ,ψ))

3-D ellipsoid with semi-axes `a`, `b`, `c` oriented by ZYZ Euler angles
`(θ, ϕ, ψ)`.

**Input-order convention** (`T <: Real`).  The three semi-axis values
are interpreted as the lengths along columns 1, 2, 3 of the local
basis defined by `euler_angles`.  For internal consistency the stored
`semi_axes` are sorted descending and the basis columns are permuted
accordingly, so the physical geometry in the canonical frame
(`change_tens_canon(shape_tensor(ell))`) matches the user's input
regardless of their order.

**`euler_angles`**.  Any tuple of length 0–3 with heterogeneous `Real`
element types is accepted; missing trailing angles default to `0`.

Degenerate limits (`T <: Real` only) are routed automatically to the
dedicated inclusion type:
- one infinite semi-axis → [`Cylinder`](@ref);
- one zero semi-axis → `EllipticCrack`;
- one infinite *and* one zero semi-axis → `RibbonCrack`.
Unsupported combinations (two or more infinite axes, two zero axes) raise
an `ArgumentError`.  Symbolic element types (`Sym`, `Num`) skip this
detection — call the dedicated constructor explicitly.
"""
function Ellipsoid(
        a::Ta, b::Tb, c::Tc;
        euler_angles::Tuple{Vararg{Real}} = ()
    ) where {Ta, Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Ta, Tb, Tc))
    axes_in = (T(a), T(b), T(c))
    basis_in = MFH_Core._default_basis(T, euler_angles)
    axes_sorted, basis = MFH_Core._sort_axes_and_basis(axes_in, basis_in, :ellipsoid_3d)
    if T <: Real
        redirected = _resolve_degenerate_ellipsoid(axes_sorted, basis)
        redirected === nothing || return redirected
    end
    code = MFH_Core._classify_shape_3d(T, axes_sorted...)
    S = _SHAPE_3D[code]
    return Ellipsoid{3, S, T, typeof(basis)}(axes_sorted, basis)
end

"""
    Ellipsoid(a, b, c, R::AbstractMatrix)

3-D ellipsoid whose principal axes are the columns of the rotation
matrix `R`; column `i` carries semi-axis `(a, b, c)[i]`.  When
`T <: Real`, the stored `semi_axes` are sorted descending and the
columns of `R` are permuted to preserve the physical geometry.  Same
degenerate-limit redirection rules as
[`Ellipsoid(a, b, c; euler_angles)`](@ref).
"""
function Ellipsoid(a::Ta, b::Tb, c::Tc, R::AbstractMatrix) where {Ta, Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Ta, Tb, Tc))
    axes_in = (T(a), T(b), T(c))
    basis_in = TensND.RotatedBasis(Matrix{Float64}(R))
    axes_sorted, basis = MFH_Core._sort_axes_and_basis(axes_in, basis_in, :ellipsoid_3d)
    if T <: Real
        redirected = _resolve_degenerate_ellipsoid(axes_sorted, basis)
        redirected === nothing || return redirected
    end
    code = MFH_Core._classify_shape_3d(T, axes_sorted...)
    S = _SHAPE_3D[code]
    return Ellipsoid{3, S, T, typeof(basis)}(axes_sorted, basis)
end

"""
    Ellipsoid(a, b, c, basis::TensND.AbstractBasis)

3-D ellipsoid sharing an already-constructed TensND basis.  Column `i`
of `basis` carries semi-axis `(a, b, c)[i]`; when `T <: Real` the
stored `semi_axes` are sorted descending and the basis columns are
permuted accordingly (the returned basis may therefore differ from the
one passed in).  Same degenerate-limit redirection rules as
[`Ellipsoid(a, b, c; euler_angles)`](@ref).
"""
function Ellipsoid(
        a::Ta, b::Tb, c::Tc, basis::TensND.AbstractBasis
    ) where {Ta, Tb, Tc}
    T = MFH_Core._floatlike(promote_type(Ta, Tb, Tc))
    axes_in = (T(a), T(b), T(c))
    axes_sorted, basis_out = MFH_Core._sort_axes_and_basis(axes_in, basis, :ellipsoid_3d)
    if T <: Real
        redirected = _resolve_degenerate_ellipsoid(axes_sorted, basis_out)
        redirected === nothing || return redirected
    end
    code = MFH_Core._classify_shape_3d(T, axes_sorted...)
    S = _SHAPE_3D[code]
    return Ellipsoid{3, S, T, typeof(basis_out)}(axes_sorted, basis_out)
end

# ── 2-D constructors ──────────────────────────────────────────────────────────

"""
    Ellipsoid(a, b; angle=0.0)

2-D ellipse with semi-axes `a`, `b` and orientation angle `θ` (radians)
of the local frame w.r.t. the first global axis.  The user's input
order defines which local axis carries each length; when `T <: Real`
the stored `semi_axes` are sorted descending and the orientation is
adjusted to preserve the physical geometry in the canonical frame.
"""
function Ellipsoid(a::Ta, b::Tb; angle::Real = 0.0) where {Ta, Tb}
    T = MFH_Core._floatlike(promote_type(Ta, Tb))
    Tbasis = MFH_Core._basis_eltype(T)
    basis_in = iszero(angle) ? TensND.CanonicalBasis{2, Tbasis}() : TensND.RotatedBasis(float(angle))
    axes_sorted, basis = MFH_Core._sort_axes_and_basis((T(a), T(b)), basis_in, :ellipsoid_2d)
    code = MFH_Core._classify_shape_2d(T, axes_sorted...)
    S = _SHAPE_2D[code]
    return Ellipsoid{2, S, T, typeof(basis)}(axes_sorted, basis)
end

# ── Sphere / circle ───────────────────────────────────────────────────────────

"""
    Ellipsoid(r; dim=3)

Sphere (3-D) or circle (2-D) of radius `r`.
"""
function Ellipsoid(r::T; dim::Int = 3) where {T <: Number}
    Tf = MFH_Core._floatlike(T)
    axes = ntuple(_ -> Tf(r), dim)
    basis = TensND.CanonicalBasis{dim, MFH_Core._basis_eltype(Tf)}()
    S = dim == 3 ? Spherical : Circular
    return Ellipsoid{dim, S, Tf, typeof(basis)}(axes, basis)
end

# ── Spheroid (axisymmetric) convenience constructors ─────────────────────────

"""
    Spheroid(ω; euler_angles = ())

Convenience constructor for an axisymmetric ellipsoid (spheroid). The
aspect ratio `ω` is the ratio of the polar semi-axis (along the axis
of revolution) to the equatorial semi-axis:

  * `ω < 1` ⇒ oblate (disc-like, axis of revolution is the *short* axis)
  * `ω > 1` ⇒ prolate (needle-like, axis of revolution is the *long* axis)
  * `ω = 1` ⇒ sphere

The two equatorial semi-axes are fixed to `1`. Eshelby/Hill computations
are scale-invariant, so the absolute size of the inclusion has no effect
on the localization tensor and only `ω` matters.

Optional `euler_angles` (tuple of length 0–3, ZYZ convention, radians)
rotate the symmetry axis from `ez`. Without angles, the axis of
revolution is `ez` (oblate) or `ex` (prolate) — i.e. the result is
strictly equivalent to `Ellipsoid(1, 1, ω; euler_angles)`, with internal
sorting and basis permutation handled by the regular `Ellipsoid`
constructor.

# Examples
```julia
Spheroid(0.2)                                      # oblate (axis ‖ ez)
Spheroid(5.0)                                      # prolate (axis ‖ ez after sort)
Spheroid(0.2; euler_angles = (π/4, π/3, 0))        # tilted axis
```
"""
function Spheroid(ω::T; euler_angles::Tuple{Vararg{Real}} = ()) where {T <: Number}
    return Ellipsoid(one(T), one(T), ω; euler_angles = euler_angles)
end

# ── Equality and hashing (field-wise) ────────────────────────────────────────
# Two `Ellipsoid`s are equal when their semi-axes and local basis compare
# equal (via `==`, i.e. by value — *not* by reference, which is the default
# Julia behavior for fields whose type is not `isbits`, such as
# `RotatedBasis`).

Base.:(==)(x::T, y::T) where {T <: Ellipsoid} =
    x.semi_axes == y.semi_axes && x.basis == y.basis

function Base.hash(x::Ellipsoid, h::UInt)
    h = hash(typeof(x), h)
    h = hash(x.semi_axes, h)
    return hash(x.basis, h)
end

# ── Interface implementations ────────────────────────────────────────────────

MFH_Core.dimension(::Ellipsoid{dim}) where {dim} = dim
MFH_Core.inclusion_basis(ell::Ellipsoid) = ell.basis
MFH_Core.shape_trait(::Ellipsoid{dim, S}) where {dim, S} = S

"""
    shape_tensor(ell::Ellipsoid) -> AbstractTens{2}

Return the symmetric representative of the 2nd-order shape tensor
``\\mathbf A = \\mathbf R\\,\\mathrm{diag}(\\rho_i)\\,\\mathbf R^{\\!T}``
of `ell`, expressed in the canonical frame.

Note: the Echoes convention
([eshelby_hill.qmd](https://jfbarthelemy.github.io/echoes/))
allows ``\\mathbf A`` to be any invertible 2nd-order tensor — only the
symmetric product ``\\mathbf A^{\\!T}\\!\\cdot\\mathbf A`` enters any
Hill expression. MFH stores the symmetric representative for
convenience.  See the generic [`shape_tensor`](@ref) docstring for
conventions.
"""
function MFH_Core.shape_tensor(ell::Ellipsoid{dim}) where {dim}
    T = eltype(ell.semi_axes)
    D = zeros(T, dim, dim)
    @inbounds for i in 1:dim
        D[i, i] = ell.semi_axes[i]
    end
    return TensND.Tens(D, ell.basis)
end

# ── Shape helpers used by scripts and downstream code ──────────────────────

"Return the spatial dimension of the ellipsoid."
get_dim(::Ellipsoid{dim}) where {dim} = dim

"Return the i-th semi-axis (1-indexed)."
semi_axis(ell::Ellipsoid, i::Int) = ell.semi_axes[i]

"Aspect ratio η = a₂/a₁  (≤ 1)."
aspect_ratio_eta(ell::Ellipsoid{3}) = ell.semi_axes[2] / ell.semi_axes[1]

"Aspect ratio ω = a₃/a₁  (≤ η ≤ 1)."
aspect_ratio_omega(ell::Ellipsoid{3}) = ell.semi_axes[3] / ell.semi_axes[1]

"Aspect ratio ρ = a₂/a₁  (≤ 1)."
aspect_ratio_rho(ell::Ellipsoid{2}) = ell.semi_axes[2] / ell.semi_axes[1]

# ── Ellipsoid convenience wrappers around the Core Newton potentials ────────

"""
    newton_potential_3d(ell::Ellipsoid{3})

Ellipsoid-level convenience wrapper that forwards to
[`Core.newton_potential_3d`](@ref).
"""
MFH_Core.newton_potential_3d(ell::Ellipsoid{3}) =
    MFH_Core.newton_potential_3d(ell.semi_axes[1], ell.semi_axes[2], ell.semi_axes[3])

"""
    newton_potential_2d(ell::Ellipsoid{2})

Ellipsoid-level convenience wrapper that forwards to
[`Core.newton_potential_2d`](@ref).
"""
MFH_Core.newton_potential_2d(ell::Ellipsoid{2}) =
    MFH_Core.newton_potential_2d(ell.semi_axes[1], ell.semi_axes[2])
