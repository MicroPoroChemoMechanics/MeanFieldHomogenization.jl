# [API — Constitutive](@id api-constitutive)

```@docs
MeanFieldHomogenization.Constitutive
```

## The contract

```@docs
AbstractMFHMaterial
AbstractMaterialState
NoState
MaterialResponse
initial_state
material_response
stress
tangent
state
```

## Self-description

```@docs
gradient_names
flux_names
tangent_blocks
transport_property
check_material_interface
probe_gradients
```

## Materials

```@docs
HomogenizedElastic
stiffness
MicrocrackedMaterial
CrackedState
open_set
apertures
FracturedPoroelasticRock
PoroFracturedState
conductivities
fluid_content
```

## Memoization

```@docs
MaterialCache
cached!
cache_stats
reset_cache!
```

## `Tensors.jl` bridge

[`to_tensors`](@extref TensND TensND.to_tensors), [`from_tensors`](@extref TensND TensND.from_tensors), [`voigt_stress`](@extref TensND TensND.voigt_stress), [`voigt_strain`](@extref TensND TensND.voigt_strain), [`stress_from_voigt`](@extref TensND TensND.stress_from_voigt)
and [`strain_from_voigt`](@extref TensND TensND.strain_from_voigt) are TensND's, re-exported unchanged: components in the
canonical basis, whatever basis a tensor is written in
([TensND, Tensors.jl and the Voigt forms](@extref TensND Tensors.jl-and-the-Voigt-forms)).

```@docs
plane_strain_response
```
