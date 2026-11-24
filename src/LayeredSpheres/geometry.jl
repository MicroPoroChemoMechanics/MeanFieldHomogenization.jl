# =============================================================================
#  geometry.jl — concrete type `LayeredSphere` (n-layer spherical
#  composite inclusion, isotropic elasticity / conductivity).
#
#  Convention: radii are listed in ASCENDING order from the center,
#      r₀ = 0  (implicit),  r₁ < r₂ < … < r_N,
#  with layer `k` occupying `r_{k-1} ≤ r < r_k` (layer 1 is the core).
#  The composite sphere is embedded in an infinite matrix outside r_N
#  (the matrix is NOT stored in the `LayeredSphere` — it is passed to
#  every API call as the second argument `C₀` / `K₀`).
#
#  Storage: `radii::NTuple{N,T}` holds `(r₁, …, r_N)` — `r₀=0` is
#  always implicit.  `moduli::Cs` is a `NTuple{N, …}` of per-layer
#  stiffness (resp. conductivity) tensors — generally `TensISO{4,3}`
#  (resp. `TensISO{2,3}`) for the isotropic case; anisotropic
#  per-layer moduli are admitted by the type but trigger an error in
#  the iso kernels.  `interfaces::NTuple{N,<:AbstractInterface}`
#  stores the condition at each radius `r_k`; by default all are
#  `PerfectInterface()`.
# =============================================================================

"""
    LayeredSphere{T, N, Cs, Is} <: AbstractLayeredInclusion{3, T}

Isotropic `N`-layer spherical composite inclusion (core + concentric
shells) embedded in an infinite matrix.  Type parameters:

- `T` — element type of the radii (`Float64`, `BigFloat`,
  `ForwardDiff.Dual`, `SymPyPythonCall.Sym`, `Symbolics.Num`, …).
- `N` — number of layers (``N \\ge 1``).
- `Cs` — concrete type of the `moduli` NTuple.
- `Is` — concrete type of the `interfaces` NTuple.

Use the keyword-argument constructor
[`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere)`(radii, moduli;
interfaces)` for most cases.

## Convention
- Radii ``(r_1, \\dots, r_N)`` are ascending, ``r_0 = 0`` implicit.
- Moduli ``(\\mathbb{C}_1, \\dots, \\mathbb{C}_N)`` per layer, layer ``k`` between ``r_{k-1}`` and ``r_k``.
- Interfaces ``(I_1, \\dots, I_N)`` at each radius ``r_k``; interface ``N`` is the
  outer boundary with the matrix.
"""
struct LayeredSphere{T <: Number, N, Cs, Is} <: MFH_Core.AbstractLayeredInclusion{3, T}
    radii::NTuple{N, T}
    moduli::Cs
    interfaces::Is
    LayeredSphere{T, N, Cs, Is}(radii, moduli, interfaces) where {T <: Number, N, Cs, Is} =
        new{T, N, Cs, Is}(radii, moduli, interfaces)
end

# Positional form, for `Schemes._replace_geom_field`: written out because the
# generated one leaves `T` unbound for the empty tuple (Aqua, Julia 1.14).
LayeredSphere(
    radii::Tuple{T, Vararg{T, M}}, moduli::Cs, interfaces::Is
) where {T <: Number, M, Cs, Is} =
    LayeredSphere{T, M + 1, Cs, Is}(radii, moduli, interfaces)

# `Tuple{T, Vararg{T, M}}` rather than `NTuple{N, T}`: the latter also matches
# the empty tuple, where `T` is bound by nothing at all — a signature Julia
# accepts and can never dispatch on. One layer at least is required anyway, so
# the type now says so, and the empty case keeps its message through the method
# just below.
function LayeredSphere(
        radii::Tuple{T, Vararg{T, M}},
        moduli::Cs;
        interfaces::Is = ntuple(_ -> PerfectInterface{TensND.float_like(T)}(), Val(M + 1)),
    ) where {T <: Number, M, Cs, Is}
    N = M + 1
    # Validate ascending radii when they compare (`Num <: Real` does not).
    if is_hard_numeric(T)
        any(radii[k] ≤ 0 for k in 1:N) &&
            throw(ArgumentError("LayeredSphere radii must be strictly positive"))
        issorted(radii; lt = ≤) ||
            throw(ArgumentError("LayeredSphere radii must be strictly ascending; got $(radii)"))
    end
    Tf = TensND.float_like(T)
    radii_f = NTuple{N, Tf}(Tf.(radii))
    layers = _as_layers(moduli)
    return LayeredSphere{Tf, N, typeof(layers), typeof(interfaces)}(
        radii_f, layers, interfaces
    )
end

LayeredSphere(::Tuple{}, moduli; kwargs...) = throw(
    ArgumentError("LayeredSphere requires at least one layer; got no radius")
)

# Mixed-eltype radii — promote, then delegate to the homogeneous method above.
# The signature above binds a single `T` across every radius, so
# `LayeredSphere((r₁, 2.0), …)` with `r₁::ForwardDiff.Dual` used to be a
# `MethodError`: differentiating with respect to ONE interface radius, the most
# natural sensitivity to ask of a coated inclusion, was out of reach while
# differentiating with respect to all of them at once worked. Promoting here
# costs nothing and keeps that door open for `Dual`, `BigFloat` and symbolic
# radii alike.
function LayeredSphere(radii::Tuple{Number, Vararg{Number}}, moduli; kwargs...)
    T = promote_type(map(typeof, radii)...)
    return LayeredSphere(map(T, radii), moduli; kwargs...)
end

# ── Accessors ────────────────────────────────────────────────────────────────

"""
    layer_count(sphere) -> Int

Number of layers (excluding the matrix).
"""
layer_count(::LayeredSphere{T, N}) where {T, N} = N

"""
    layer_radius(sphere, k) -> T

Outer radius of layer `k`.  Layer 1 extends from `0` to `layer_radius(1)`,
layer `k` from `layer_radius(k-1)` to `layer_radius(k)`.
"""
layer_radius(sphere::LayeredSphere, k::Int) = sphere.radii[k]

"""
    layer_modulus(sphere, k)

Stiffness (or conductivity) tensor of layer `k`.
"""
layer_modulus(sphere::LayeredSphere, k::Int) = sphere.moduli[k]

"""
    layer_interface(sphere, k) -> AbstractInterface

Interface condition at `layer_radius(k)` (between layer `k` and layer
`k+1` if `k < N`, or with the matrix if `k = N`).
"""
layer_interface(sphere::LayeredSphere, k::Int) = sphere.interfaces[k]

"""
    layer_volume_fraction(sphere, k) -> T

Volume fraction of layer `k` inside the outer sphere of radius
`layer_radius(N)`.
"""
function layer_volume_fraction(sphere::LayeredSphere{T, N}, k::Int) where {T, N}
    r_outer = sphere.radii[N]
    r_k = sphere.radii[k]
    r_km1 = k == 1 ? zero(T) : sphere.radii[k - 1]
    return (r_k^3 - r_km1^3) / r_outer^3
end

"""
    outer_radius(sphere) -> T

Outermost radius of the composite sphere.
"""
outer_radius(sphere::LayeredSphere) = sphere.radii[end]

# ── Layers that follow a radial profile ─────────────────────────────────────

"""
    graded_layers(profile, edges) -> (radii, moduli)
    graded_layers(profile, r_in, r_out, n) -> (radii, moduli)

Uniform layers that stand for a zone whose moduli vary with the radius.

`profile(r)` returns the modulus at the radius `r`, a stiffness or a
conductivity tensor. The zone between `edges[1]` and `edges[end]` is cut at the
radii `edges`, ascending, and each layer takes the modulus of the profile at
its mid-radius. The second form cuts `[r_in, r_out]` into `n` layers of equal
thickness. The result holds the outer radius and the modulus of each layer,
ready to be appended to those of a core:

```jldoctest
julia> using MeanFieldHomogenization, TensND

julia> radii, moduli = graded_layers(r -> TensISO{3}(2 - 1 / r), 1.0, 2.0, 4);

julia> radii
(1.25, 1.5, 1.75, 2.0)

julia> sphere = LayeredSphere((1.0, radii...), (TensISO{3}(5.0), moduli...));

julia> layer_count(sphere)
5
```

The response of the layered inclusion tends to that of the graded one as the
square of the thickness of the layers. The same radii and moduli serve a
`LayeredCylinder`.

The number of layers is part of the type of the result: with `edges` given as
a tuple the call is inferred, with a vector or with `n` it is not, and the
inclusion built from it is the function barrier.
"""
function graded_layers(profile::F, edges::Tuple{Number, Number, Vararg{Number}}) where {F}
    inner, outer = Base.front(edges), Base.tail(edges)
    return outer, map((a, b) -> profile((a + b) / 2), inner, outer)
end
graded_layers(profile::F, edges::AbstractVector{<:Number}) where {F} = graded_layers(profile, Tuple(edges))
function graded_layers(profile::F, r_in::Number, r_out::Number, n::Integer) where {F}
    n ≥ 1 || throw(ArgumentError("graded_layers needs at least one layer; got n = $n"))
    return graded_layers(profile, ntuple(j -> r_in + (r_out - r_in) * (j - 1) / n, n + 1))
end
graded_layers(::Any, edges::Union{Tuple{}, Tuple{Number}}) = throw(
    ArgumentError("graded_layers needs two radii at least, the two ends of the zone; got $(edges)")
)

# ── AbstractInclusion interface ──────────────────────────────────────────────

MFH_Core.dimension(::LayeredSphere) = 3

MFH_Core.inclusion_basis(::LayeredSphere{T}) where {T} =
    TensND.CanonicalBasis{3, TensND.canonical_eltype(T)}()

struct SphericalLayered end
MFH_Core.shape_trait(::LayeredSphere) = SphericalLayered

"""
    shape_tensor(sphere::LayeredSphere) -> AbstractTens{2,3}

Symmetric 2nd-order shape tensor of the composite sphere, i.e.
``r_N\\,\\boldsymbol{1}`` (a ball of radius ``r_N``, isotropic).
"""
function MFH_Core.shape_tensor(sphere::LayeredSphere{T}) where {T}
    r = outer_radius(sphere)
    return r * TensISO{3}(one(T))
end

# ── Equality / hashing ──────────────────────────────────────────────────────

Base.:(==)(a::T, b::T) where {T <: LayeredSphere} =
    a.radii == b.radii && a.moduli == b.moduli && a.interfaces == b.interfaces

function Base.hash(s::LayeredSphere, h::UInt)
    h = hash(typeof(s), h)
    h = hash(s.radii, h)
    h = hash(s.moduli, h)
    return hash(s.interfaces, h)
end

function Base.show(io::IO, s::LayeredSphere{T, N}) where {T, N}
    return print(io, "LayeredSphere{", T, "} (", N, " layer(s), radii = ", s.radii, ")")
end
