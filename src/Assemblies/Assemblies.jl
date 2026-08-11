"""
    MeanFieldHomogenization.Assemblies

Assemblies of individually located inclusions, and the two N-body schemes that
act on them.

Every other cell of the package describes a microstructure statistically — an
`RVE` by volume fractions, a `Laminate` by a stacking order. That is all a
one-site scheme can use, and it is not enough here: the field one inclusion
induces in another depends on the vector joining their centers. Hence
[`ParticleAssembly`](@ref), a cell that carries positions.

Two schemes consume it, and they share both the cell and the interaction
kernel of `MeanFieldHomogenization.Interactions`:

- [`ClusterModel`](@ref) — Molinari & El Mouden (1996), solving for the mean
  strain of every inclusion inside a cluster of neighbors;
- [`EquivalentInclusion`](@ref) — Brisard, Dormieux & Sab (2014), a Galerkin
  discretization of the weak Lippmann-Schwinger equation, which additionally
  delivers a rigorous bound on the apparent stiffness.

Contents
--------
- `assembly.jl`      : `Particle`, the boundary treatments (`MixedBC`,
                        `PeriodicBox`), the `ParticleAssembly` cell and its
                        `AbstractHomogenizationCell` contract
- `generators.jl`    : cubic lattices and hard-particle random microstructures
- `block_solve.jl`   : linear systems whose unknowns are tensors, flattened
                        onto the Kelvin-Mandel basis
- `cluster_model.jl` : the Molinari & El Mouden kernel
- `eim.jl`           : the Brisard, Dormieux & Sab kernel
- `as_rve.jl`        : `RVE(asm)` — forget the positions, keep the fractions,
                        which is what makes every one-site scheme of the
                        package available on an assembly
- `parameters.jl`    : parameter lenses (positions, radii) for the
                        differentiation entry points
"""
module Assemblies

using LinearAlgebra
using StaticArrays
using TensND
using Random

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity
import ..Interactions
import ..Schemes
using ..Schemes: HomogenizationScheme, ClusterModel, EquivalentInclusion,
    AbstractParameter, MoriTanaka

# Extend rather than shadow: `add_matrix!`, `matrix_property` and
# `matrix_volume_fraction` are `Schemes` generics, and an assembly is just
# another cell answering them.
import ..Schemes: add_matrix!, matrix_property, matrix_volume_fraction,
    get_param, set_param

include("assembly.jl")
include("generators.jl")
include("block_solve.jl")
include("cluster_model.jl")
include("eim.jl")
include("as_rve.jl")
include("parameters.jl")

# ── The cell and its parts ──────────────────────────────────────────────────
export ParticleAssembly, Particle
export AbstractAssemblyBoundary, MixedBC, PeriodicBox
export add_particle!
export particle_names, particle, particle_center, particle_geometry
export particle_property, particle_family, family_labels
export particle_volume, particle_volume_fraction, inclusion_volume_fraction
export assembly_volume, validate_assembly

# ── Generators ──────────────────────────────────────────────────────────────
export cubic_lattice, random_assembly, max_packing_fraction

# ── Scheme-specific queries ─────────────────────────────────────────────────
export eim_bound_type, eim_polarizations, cluster_localizations

# ── Parameter lenses ────────────────────────────────────────────────────────
export CenterParameter, RadiusParameter, center_param, radius_param

end # module
