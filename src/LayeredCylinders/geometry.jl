# =============================================================================
#  geometry.jl — concrete type `LayeredCylinder` (infinite n-layer cylinder of
#  circular section, transversely isotropic layers about the cylinder axis).
#
#  Same convention as `LayeredSphere`: radii ascending from the axis,
#      r₀ = 0 (implicit),  r₁ < r₂ < … < r_N,
#  layer `k` occupying `r_{k-1} ≤ r < r_k` (layer 1 is the core), interface `k`
#  at `r_k` (interface `N` is the boundary with the matrix). The matrix is not
#  stored: it is the reference passed to every call.
# =============================================================================

"""
    LayeredCylinder{T, N, Cs, Is, A} <: AbstractLayeredInclusion{3, T}

Infinite cylinder of circular section made of `N` coaxial layers (core and
coatings), embedded in an infinite matrix. Each layer is isotropic or
transversely isotropic about the cylinder axis, and so is the matrix: the
problem then splits into four radial problems solved in closed form
(see the theory page).

- `radii`: ``(r_1, \\dots, r_N)``, ascending; layer ``k`` lies between
  ``r_{k-1}`` and ``r_k``, with ``r_0 = 0``.
- `moduli`: one property per layer, a stiffness (`TensISO{4,3}` or a
  `TensTI{4}` whose axis is the cylinder axis) or a conductivity
  (`TensISO{2,3}` or `TensTI{2}`).
- `interfaces`: the condition at each ``r_k``
  ([`PerfectInterface`](@ref) by default); interface ``N`` is the boundary
  with the matrix.
- `axis`: unit vector of the cylinder axis, ``\\underline{e}_3`` by default.

Type parameters: `T` is the element type of the radii (`Float64`, `BigFloat`,
`ForwardDiff.Dual`, `Symbolics.Num`, `SymPyPythonCall.Sym`), `N` the number of
layers, `A` the element type of the axis.
"""
struct LayeredCylinder{T <: Number, N, Cs, Is, A <: Number} <:
    MFH_Core.AbstractLayeredInclusion{3, T}
    radii::NTuple{N, T}
    moduli::Cs
    interfaces::Is
    axis::NTuple{3, A}
    LayeredCylinder{T, N, Cs, Is, A}(
        radii, moduli, interfaces, axis
    ) where {T <: Number, N, Cs, Is, A <: Number} =
        new{T, N, Cs, Is, A}(radii, moduli, interfaces, axis)
end

# Positional form, for `Schemes._replace_geom_field`.
LayeredCylinder(
    radii::Tuple{T, Vararg{T, M}}, moduli::Cs, interfaces::Is, axis::NTuple{3, A}
) where {T <: Number, M, Cs, Is, A <: Number} =
    LayeredCylinder{T, M + 1, Cs, Is, A}(radii, moduli, interfaces, axis)

"""
    LayeredCylinder(radii, moduli; interfaces, axis = (0.0, 0.0, 1.0))

Build an `N`-layer cylinder from its ascending `radii`, the per-layer `moduli`
and, optionally, the `interfaces` at each radius and the `axis` of the
cylinder. A numeric axis is normalized; a symbolic one is taken as given and
must be a unit vector.
"""
function LayeredCylinder(
        radii::Tuple{T, Vararg{T, M}},
        moduli::Cs;
        interfaces::Is = ntuple(_ -> PerfectInterface{TensND.float_like(T)}(), Val(M + 1)),
        axis::Tuple{Number, Number, Number} = (0.0, 0.0, 1.0),
    ) where {T <: Number, M, Cs, Is}
    N = M + 1
    length(moduli) == N || throw(
        DimensionMismatch("LayeredCylinder: $N radii but $(length(moduli)) layer moduli")
    )
    length(interfaces) == N || throw(
        DimensionMismatch("LayeredCylinder: $N radii but $(length(interfaces)) interfaces")
    )
    # Radii are compared only when they compare (`Num <: Real` does not).
    if is_hard_numeric(T)
        all(>(0), radii) ||
            throw(ArgumentError("LayeredCylinder radii must be strictly positive; got $(radii)"))
        issorted(radii; lt = ≤) || throw(
            ArgumentError("LayeredCylinder radii must be strictly ascending; got $(radii)")
        )
    end
    Tf = TensND.float_like(T)
    ax = _unit_axis(promote(axis...))
    return LayeredCylinder{Tf, N, Cs, Is, eltype(ax)}(
        NTuple{N, Tf}(Tf.(radii)), moduli, interfaces, ax
    )
end

LayeredCylinder(::Tuple{}, moduli; kwargs...) = throw(
    ArgumentError("LayeredCylinder requires at least one layer; got no radius")
)

# Mixed-eltype radii (one radius a `Dual`, the others `Float64`): promote.
function LayeredCylinder(radii::Tuple{Number, Vararg{Number}}, moduli; kwargs...)
    T = promote_type(map(typeof, radii)...)
    return LayeredCylinder(map(T, radii), moduli; kwargs...)
end

_unit_axis(ax::NTuple{3, A}) where {A} = _unit_axis(Val(is_hard_numeric(A)), ax)
function _unit_axis(::Val{true}, ax::NTuple{3})
    nrm = sqrt(sum(abs2, ax))
    iszero(nrm) && throw(ArgumentError("LayeredCylinder: the axis must be a nonzero vector"))
    return ax ./ nrm
end
# A symbolic axis is not comparable: taken as given.
_unit_axis(::Val{false}, ax::NTuple{3}) = ax

# ── Accessors (generics declared in `LayeredSpheres`) ───────────────────────

layer_count(::LayeredCylinder{T, N}) where {T, N} = N
layer_radius(cyl::LayeredCylinder, k::Int) = cyl.radii[k]
layer_modulus(cyl::LayeredCylinder, k::Int) = cyl.moduli[k]
layer_interface(cyl::LayeredCylinder, k::Int) = cyl.interfaces[k]
outer_radius(cyl::LayeredCylinder) = cyl.radii[end]

_inner_radius(cyl::LayeredCylinder{T}, k::Int) where {T} = k == 1 ? zero(T) : cyl.radii[k - 1]

"""
    layer_volume_fraction(cyl::LayeredCylinder, k) -> T

Area fraction ``(r_k^2 - r_{k-1}^2)/r_N^2`` of layer `k` in the section of the
cylinder, which is its volume fraction.
"""
layer_volume_fraction(cyl::LayeredCylinder, k::Int) =
    (cyl.radii[k]^2 - _inner_radius(cyl, k)^2) / outer_radius(cyl)^2

_layer_fractions(cyl::LayeredCylinder{T, N}) where {T, N} =
    ntuple(k -> layer_volume_fraction(cyl, k), Val(N))

# ── AbstractInclusion interface ──────────────────────────────────────────────

MFH_Core.dimension(::LayeredCylinder) = 3

MFH_Core.inclusion_basis(::LayeredCylinder{T}) where {T} =
    TensND.CanonicalBasis{3, TensND.canonical_eltype(T)}()

struct CylindricalLayered end
MFH_Core.shape_trait(::LayeredCylinder) = CylindricalLayered

MFH_Core.is_homogeneous_inclusion(::LayeredCylinder) = false

# ── Equality / hashing / display ────────────────────────────────────────────

Base.:(==)(a::T, b::T) where {T <: LayeredCylinder} =
    a.radii == b.radii && a.moduli == b.moduli && a.interfaces == b.interfaces &&
    a.axis == b.axis

Base.hash(c::LayeredCylinder, h::UInt) =
    hash(c.axis, hash(c.interfaces, hash(c.moduli, hash(c.radii, hash(typeof(c), h)))))

Base.show(io::IO, c::LayeredCylinder{T, N}) where {T, N} =
    print(io, "LayeredCylinder{", T, "} (", N, " layer(s), radii = ", c.radii, ", axis = ", c.axis, ")")
