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
```

## Materials

```@docs
HomogenizedElastic
stiffness
MicrocrackedMaterial
CrackedState
open_set
apertures
```

## Memoization

```@docs
MaterialCache
cached!
cache_stats
reset_cache!
```

## `Tensors.jl` bridge

```@docs
to_tensors
from_tensors
plane_strain_response
voigt_stress
voigt_strain
stress_from_voigt
strain_from_voigt
```
