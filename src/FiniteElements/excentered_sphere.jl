# =============================================================================
#  excentered_sphere.jl — the sphere with an off-center spherical core, solved
#  by *axisymmetric Fourier* finite elements.
#
#  This is the morphology of Adessina, Barthélémy, Lavergne & Ben Fraj,
#  *Int. J. Eng. Sci.* 119 (2017) 1-15 — a recycled-concrete aggregate: an old
#  natural aggregate (the core) sitting off-center inside a shell of adhered
#  old mortar, the whole embedded in fresh cement paste.  The paper solves it
#  in full 3-D; the geometry is a solid of revolution, so we solve it instead
#  on the *meridian half-plane*, expanding the fields in Fourier series in the
#  azimuth.  Four two-dimensional solves replace six three-dimensional ones and
#  the cost collapses by two orders of magnitude at equal accuracy.
#
#  The type, the gmsh geometry, the Fourier operators and the algebra live in
#  this sub-module; only the discretization comes from a backend extension.
#  See `docs/src/applications/recycled_aggregate.md`.
# =============================================================================

"""
    FEAxiMeshOptions(; radius_ratio = 4.0, nradial = 24, order = 2,
                     coarsening = 6.0)

Discretization settings of an axisymmetric finite-element inclusion.

| Field | Default | Meaning |
|---|---|---|
| `radius_ratio` | `4.0` | radius `R` of the surrounding ball of matrix, in units of the inclusion radius. `4` suffices *because* the boundary condition is corrected — the uncorrected problem needs 10 or more (Adessina et al. 2017, Table 2). |
| `nradial` | `24` | element size inside the inclusion, as `a / nradial`. |
| `coarsening` | `6.0` | ratio of the element size at the outer boundary to the size inside the inclusion. |
| `order` | `2` | polynomial order of the displacement / temperature interpolation (1 or 2). |

The mesh is two-dimensional (the meridian half-plane), so refining is cheap:
`nradial = 40` on a triangle mesh still solves in a fraction of a second.
"""
struct FEAxiMeshOptions
    radius_ratio::Float64
    nradial::Float64
    coarsening::Float64
    order::Int
    function FEAxiMeshOptions(;
            radius_ratio = 4.0, nradial = 24, coarsening = 6.0, order = 2
        )
        order in (1, 2) || throw(
            ArgumentError(
                "interpolation `order` must be 1 or 2 — the common ground " *
                    "of the finite-element backends — got $order"
            )
        )
        radius_ratio > 1 ||
            throw(ArgumentError("`radius_ratio` must exceed 1, got $radius_ratio"))
        nradial > 0 || throw(ArgumentError("`nradial` must be positive, got $nradial"))
        coarsening ≥ 1 ||
            throw(ArgumentError("`coarsening` must be at least 1, got $coarsening"))
        return new(
            Float64(radius_ratio), Float64(nradial), Float64(coarsening), Int(order)
        )
    end
end

"""
    FEExcenteredSphere(a, (P_core, P_shell); core_fraction, eccentricity = 0.0,
                       euler_angles = (), radius_ratio = 4.0, nradial = 24,
                       coarsening = 6.0, order = 2)

Spherical inclusion of radius `a` containing a **spherical core placed off the
center**, resolved by axisymmetric Fourier finite elements.

# Geometry

| Symbol | Definition |
|---|---|
| `a` | radius of the whole inclusion |
| `w = core_fraction` | volume fraction of the core *within the inclusion* |
| `a_core = a·w^(1/3)` | core radius, fixed by `w` |
| `α = eccentricity` | offset of the core center, as a fraction of the largest offset that keeps the core inside: `d = α·(a − a_core)` |

`α = 0` is the concentric two-layer sphere, for which
[`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere) gives the
exact Hervé-Zaoui answer — the reference this type is validated against.
`α → 1` brings the core tangent to the outer surface. The symmetry axis is
`axis`; the response is transversely isotropic about it (and isotropic at
`α = 0`).

# What it provides

The inclusion is **heterogeneous**, so it enters through gate B of the
inclusion contract with *both* localization tensors — the strain-side
`A_εε` and the stress-side `A_σε` (resp. `A_∇∇` and `A_q∇` in transport). All
schemes that consume those two follow: `Dilute`, `MoriTanaka`, `Maxwell`,
`PonteCastanedaWillis`, `SelfConsistent`, `DifferentialScheme`.

`Voigt` and `Reuss` work too, which is not automatic for a heterogeneous
inclusion: a bound averages the *constituent* properties over the inclusion and
therefore needs its internal volume fractions, which the RVE does not carry.
Here the geometry fixes them exactly — `w` for the core, `1 - w` for the shell,
whatever the eccentricity — so the type implements `Schemes._layer_voigt` /
`Schemes._layer_reuss` and the bounds follow.

# Properties

Like [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere), the
constituent properties live **in the geometry object**, core first then shell,
and the `Dict` handed to `add_phase!` is a placeholder that the kernel ignores.
Build one object per physics: a pair of `Tens{4,3}` for elasticity, a pair of
`Tens{2,3}` for conduction.

Requires a finite-element backend to be loaded — `Ferrite`, `FerriteGmsh` and
`Gmsh`, or `Gridap` and `GridapGmsh`; pass `backend = GridapBackend()` to pick
the second when both are available, and see [`FEBackend`](@ref). The reference
medium must be isotropic (the corrected boundary condition uses the closed-form
Kelvin dipole field); the constituents may be isotropic or transversely
isotropic about the symmetry axis.

# Example

```julia
using MeanFieldHomogenization
import Ferrite, FerriteGmsh, Gmsh        # or: import Gridap, GridapGmsh

C_core, C_shell = iso_stiffness(20.0, 12.0), iso_stiffness(6.0, 4.0)
C₀ = iso_stiffness(10.0, 6.0)
incl = FEExcenteredSphere(1.0, (C_core, C_shell);
                          core_fraction = 0.5, eccentricity = 0.4)

A, B = fe_axi_localization(incl, C₀)          # both tensors, one solve

rve = RVE()
add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :rca, incl, Dict(:C => C₀); fraction = 0.3)
homogenize(rve, MoriTanaka(), :C)
```

See also [`FEAxiMeshOptions`](@ref), [`fe_axi_breakdown`](@ref),
[`fe_axi_mesh_report`](@ref).
"""
struct FEExcenteredSphere{T <: Number, P, B <: TensND.AbstractBasis} <:
    Core.AbstractCustomInclusion{T}
    a::T
    core_fraction::T
    eccentricity::T
    props::P
    basis::B
    mesh::FEAxiMeshOptions
    cache::FECache
    backend::FEBackend
end

function FEExcenteredSphere(
        a::Real,
        props::Tuple{TensND.AbstractTens{O, 3}, TensND.AbstractTens{O, 3}};
        core_fraction::Real,
        eccentricity::Real = 0.0,
        basis::Union{Nothing, TensND.AbstractBasis} = nothing,
        euler_angles::Tuple{Vararg{Real}} = (),
        radius_ratio = 4.0, nradial = 24, coarsening = 6.0, order = 2,
        backend::FEBackend = AutoBackend()
    ) where {O}
    T = Core._floatlike(promote_type(typeof(a), typeof(core_fraction), typeof(eccentricity)))
    a > 0 || throw(ArgumentError("the inclusion radius must be positive, got $a"))
    0 < core_fraction < 1 || throw(
        ArgumentError(
            "`core_fraction` is the volume fraction of the core within the " *
                "inclusion and must lie strictly between 0 and 1, got $core_fraction"
        )
    )
    0 ≤ eccentricity < 1 || throw(
        ArgumentError(
            "`eccentricity` is normalized by the largest offset that keeps the " *
                "core inside, so it must lie in [0, 1), got $eccentricity"
        )
    )
    bas = basis === nothing ? Core._default_basis(T, euler_angles) : basis
    opts = FEAxiMeshOptions(; radius_ratio, nradial, coarsening, order)
    return FEExcenteredSphere{T, typeof(props), typeof(bas)}(
        T(a), T(core_fraction), T(eccentricity), props, bas, opts, FECache(), backend
    )
end

"""
    tensor_order(incl) -> Int

`4` for an elasticity object (constituents are stiffness tensors), `2` for a
transport one (conductivity tensors). Fixed at construction by the pair of
properties handed in.
"""
tensor_order(s::FEExcenteredSphere) = _tens_order(s.props[1])
_tens_order(::TensND.AbstractTens{O, 3}) where {O} = O

"""
    ExcenteredSphereShape

[`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) of
[`FEExcenteredSphere`](@ref). No kernel dispatches on it — the inclusion
supplies its own localization tensors.
"""
struct ExcenteredSphereShape end

Core.dimension(::FEExcenteredSphere) = 3
Core.inclusion_basis(s::FEExcenteredSphere) = s.basis
Core.shape_trait(::FEExcenteredSphere) = ExcenteredSphereShape
Core.is_homogeneous_inclusion(::FEExcenteredSphere) = false
_fe_cache(s::FEExcenteredSphere) = s.cache

function Core.shape_tensor(s::FEExcenteredSphere{T}) where {T}
    D = zeros(T, 3, 3)
    for i in 1:3
        D[i, i] = s.a
    end
    return TensND.Tens(D, s.basis)
end

"""
    core_radius(incl) -> Real

Radius of the off-center core, `a·w^(1/3)`.
"""
core_radius(s::FEExcenteredSphere) = s.a * cbrt(s.core_fraction)

"""
    core_offset(incl) -> Real

Distance from the inclusion center to the core center, `α·(a − a_core)`.
"""
core_offset(s::FEExcenteredSphere) = s.eccentricity * (s.a - core_radius(s))

# ─── Layer averages — what the bounds need ───────────────────────────────────
#
#  A heterogeneous inclusion has no single phase property, so `Voigt` and
#  `Reuss` cannot read one off the RVE.  They can, however, be given the
#  volume average over the constituents — and here that average is exact and
#  free, because the geometry fixes the internal fractions: `w` for the core,
#  `1 - w` for the shell, whatever the eccentricity.
#
#  (`AsymmetricSelfConsistent` does *not* depend on this: its iteration needs
#  only the two localization tensors, and its branch-selection heuristic falls
#  back to the dilute estimate when no bound is available.  Supplying the
#  averages here simply lets it use the Voigt bound, as it does everywhere
#  else.)

function _layer_average(s::FEExcenteredSphere, ::TensND.AbstractTens{O, 3}, f) where {O}
    tensor_order(s) == O || throw(
        ArgumentError(
            "this `FEExcenteredSphere` carries order-$(tensor_order(s)) " *
                "constituents, so it cannot serve an order-$O bound. Build one " *
                "object per physics: a pair of `Tens{4,3}` for elasticity, a pair " *
                "of `Tens{2,3}` for transport."
        )
    )
    w = s.core_fraction
    return w * f(s.props[1]) + (1 - w) * f(s.props[2])
end

Schemes.has_layer_average(::FEExcenteredSphere) = true
Schemes._layer_voigt(s::FEExcenteredSphere, ref::TensND.AbstractTens) =
    _layer_average(s, ref, identity)
Schemes._layer_reuss(s::FEExcenteredSphere, ref::TensND.AbstractTens) =
    _layer_average(s, ref, inv)

"""
    fe_axi_localization(incl, P₀) -> (A, B)

The pair of localization tensors of an axisymmetric finite-element inclusion,
computed in one solve — the strain-side `A_εε` and stress-side `A_σε` in
elasticity, `A_∇∇` and `A_q∇` in transport.

Calling the two generics separately returns exactly the same tensors at no
extra cost: they share the memoized solve.
"""
fe_axi_localization(s::FEExcenteredSphere, P₀::TensND.AbstractTens; kw...) =
    _fe_axi_localization(s, P₀; kw...)

# ─── Gate B — both localization tensors ──────────────────────────────────────
#
#  The inclusion is heterogeneous, so the stress-side tensor is *not*
#  `C₁ : A_εε` for any single `C₁`; it is measured on the finite-element
#  solution alongside the strain-side one, in the same pass.
#
#  As for `LayeredSphere`, the three-argument scheme signature carries a phase
#  property that this inclusion has no use for — its constituents live in the
#  geometry object — so the middle argument is accepted and ignored.

for (gen, order, idx) in (
        (:(Core.strain_strain_loc), 4, 1),
        (:(Core.stress_strain_loc), 4, 2),
        (:(Core.gradient_gradient_loc), 2, 1),
        (:(Core.flux_gradient_loc), 2, 2),
    )
    @eval $gen(
        s::FEExcenteredSphere,
        ::TensND.AbstractTens{$order, 3},
        P₀::TensND.AbstractTens{$order, 3};
        kw...
    ) = _fe_axi_localization(s, P₀; kw...)[$idx]
end
