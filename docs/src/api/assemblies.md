# [API — Particle assemblies](@id api-assemblies)

```@docs
MeanFieldHomogenization.Assemblies
```

## The cell

```@docs
ParticleAssembly
Particle
add_particle!
validate_assembly
```

## Boundary treatments

```@docs
AbstractAssemblyBoundary
MixedBC
PeriodicBox
```

## Accessors

```@docs
particle_names
particle
particle_center
particle_geometry
particle_property
particle_family
family_labels
particle_volume
particle_volume_fraction
inclusion_volume_fraction
assembly_volume
```

## Generators

```@docs
cubic_lattice
random_assembly
max_packing_fraction
```

## The N-body schemes

```@docs
ClusterModel
EquivalentInclusion
eim_bound_type
eim_polarizations
cluster_localizations
```

## Parameter lenses

```@docs
CenterParameter
RadiusParameter
center_param
radius_param
```
