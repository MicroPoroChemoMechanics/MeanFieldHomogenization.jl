# [API — LayeredCylinder](@id api-layered-cylinder)

The layered cylinder shares its interfaces, its layer accessors and its
pointwise entry points with the layered sphere: the functions listed on
[API — LayeredSphere](@ref api-layered-sphere) have a method for a
[`LayeredCylinder`](@ref). This page lists what is its own.

```@docs
MeanFieldHomogenization.LayeredCylinders
MeanFieldHomogenization.LayeredCylinders.LayeredCylinder
MeanFieldHomogenization.LayeredCylinders.LayeredCylinderFields
MeanFieldHomogenization.LayeredCylinders.LayeredCylinderTransportFields
MeanFieldHomogenization.LayeredCylinders.cylinder_gradient_average
MeanFieldHomogenization.LayeredCylinders.cylinder_flux_average
MeanFieldHomogenization.LayeredCylinders.cylinder_strain_average
MeanFieldHomogenization.LayeredCylinders.cylinder_stress_average
```

| Generic function | Elasticity |
|:-----------------|:-----------|
| `cylinder_gradient_average` | `cylinder_strain_average` |
| `cylinder_flux_average` | `cylinder_stress_average` |

The other names of elasticity (`layer_strain_average`, `local_strain`,
`local_stress`, …) are those of the layered sphere and accept a cylinder.
