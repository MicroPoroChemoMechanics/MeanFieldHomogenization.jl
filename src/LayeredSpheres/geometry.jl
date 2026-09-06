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
  `ForwardDiff.Dual`, `SymPy.Sym`, `Symbolics.Num`, …).
- `N` — number of layers (≥ 1).
- `Cs` — concrete type of the `moduli` NTuple.
- `Is` — concrete type of the `interfaces` NTuple.

Use the keyword-argument constructor
[`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere)`(radii, moduli;
interfaces)` for most cases.

## Convention
- Radii `(r₁, …, r_N)` are ascending, `r₀ = 0` implicit.
- Moduli `(C₁, …, C_N)` per layer, layer `k` between `r_{k-1}` and `r_k`.
- Interfaces `(I_1, …, I_N)` at each radius `r_k`; interface `N` is the
  outer boundary with the matrix.
"""
struct LayeredSphere{T <: Number, N, Cs, Is} <: MFH_Core.AbstractLayeredInclusion{3, T}
    radii::NTuple{N, T}
    moduli::Cs
    interfaces::Is
end

# `Tuple{T, Vararg{T, M}}` rather than `NTuple{N, T}`: the latter also matches
# the empty tuple, where `T` is bound by nothing at all — a signature Julia
# accepts and can never dispatch on. One layer at least is required anyway, so
# the type now says so, and the empty case keeps its message through the method
# just below.
function LayeredSphere(
        radii::Tuple{T, Vararg{T, M}},
        moduli::Cs;
        interfaces::Is = ntuple(_ -> PerfectInterface{MFH_Core._floatlike(T)}(), Val(M + 1)),
    ) where {T <: Number, M, Cs, Is}
    N = M + 1
    # Validate ascending radii for Real types (skip for symbolic).
    if T <: Real
        any(radii[k] ≤ 0 for k in 1:N) &&
            throw(ArgumentError("LayeredSphere radii must be strictly positive"))
        for k in 1:(N - 1)
            radii[k] ≥ radii[k + 1] &&
                throw(
                ArgumentError(
                    "LayeredSphere radii must be strictly ascending; got $(radii)"
                )
            )
        end
    end
    Tf = MFH_Core._floatlike(T)
    radii_f = NTuple{N, Tf}(Tf.(radii))
    return LayeredSphere{Tf, N, typeof(moduli), typeof(interfaces)}(
        radii_f, moduli, interfaces
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

# ── AbstractInclusion interface ──────────────────────────────────────────────

MFH_Core.dimension(::LayeredSphere) = 3

MFH_Core.inclusion_basis(::LayeredSphere{T}) where {T} =
    TensND.CanonicalBasis{3, MFH_Core._basis_eltype(T)}()

struct SphericalLayered end
MFH_Core.shape_trait(::LayeredSphere) = SphericalLayered

"""
    shape_tensor(sphere::LayeredSphere) -> AbstractTens{2,3}

Symmetric 2nd-order shape tensor of the composite sphere, i.e.
`r_N² · 𝟙` (a ball of radius `r_N`, isotropic).
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
