# [API — LayeredSphere](@id api-layered-sphere)

```@docs
MeanFieldHomogenization.LayeredSpheres
MeanFieldHomogenization.LayeredSpheres.LayeredSphere
MeanFieldHomogenization.LayeredSpheres.AbstractInterface
MeanFieldHomogenization.LayeredSpheres.PerfectInterface
MeanFieldHomogenization.LayeredSpheres.SpringInterface
MeanFieldHomogenization.LayeredSpheres.MembraneInterface
MeanFieldHomogenization.LayeredSpheres.KapitzaInterface
MeanFieldHomogenization.LayeredSpheres.SurfaceConductiveInterface
MeanFieldHomogenization.LayeredSpheres.layer_count
MeanFieldHomogenization.LayeredSpheres.layer_radius
MeanFieldHomogenization.LayeredSpheres.layer_modulus
MeanFieldHomogenization.LayeredSpheres.layer_property_average
MeanFieldHomogenization.LayeredSpheres.layer_inverse_property_average
MeanFieldHomogenization.LayeredSpheres.layer_interface
MeanFieldHomogenization.LayeredSpheres.layer_volume_fraction
MeanFieldHomogenization.LayeredSpheres.outer_radius
MeanFieldHomogenization.LayeredSpheres.layer_gradient_average
MeanFieldHomogenization.LayeredSpheres.layer_flux_average
MeanFieldHomogenization.LayeredSpheres.sphere_gradient_average
MeanFieldHomogenization.LayeredSpheres.sphere_flux_average
MeanFieldHomogenization.LayeredSpheres.cumulative_strain_average
MeanFieldHomogenization.LayeredSpheres.spring_compliances
MeanFieldHomogenization.LayeredSpheres.spring_stiffnesses
MeanFieldHomogenization.LayeredSpheres.LayeredSphereFields
MeanFieldHomogenization.LayeredSpheres.LayeredSphereTransportFields
MeanFieldHomogenization.LayeredSpheres.get_layer
MeanFieldHomogenization.LayeredSpheres.region_stiffness
MeanFieldHomogenization.LayeredSpheres.shell_concentration
MeanFieldHomogenization.LayeredSpheres.local_gradient_gradient_concentration
MeanFieldHomogenization.LayeredSpheres.local_flux_gradient_concentration
MeanFieldHomogenization.LayeredSpheres.local_gradient_flux_concentration
MeanFieldHomogenization.LayeredSpheres.local_flux_flux_concentration
MeanFieldHomogenization.LayeredSpheres.local_gradient
MeanFieldHomogenization.LayeredSpheres.local_flux
MeanFieldHomogenization.LayeredSpheres.local_displacement
MeanFieldHomogenization.LayeredSpheres.local_temperature
```

## The names of elasticity and of conduction

The functions above serve both physics, the order of the matrix property
selecting the method. Each also answers to the name it has in elasticity (or,
for the layer averages of a property, in conduction); such a name forwards to
the generic function, accepts one order only, and refuses the other with an
error that names the generic function.

| Generic function | Elasticity | Conduction |
|:-----------------|:-----------|:-----------|
| `layer_property_average` | `layer_stiffness_average` | `layer_conductivity_average` |
| `layer_inverse_property_average` | `layer_compliance_average` | `layer_resistivity_average` |
| `layer_gradient_average` | `layer_strain_average` | |
| `layer_flux_average` | `layer_stress_average` | |
| `sphere_gradient_average` | `sphere_strain_average` | |
| `sphere_flux_average` | `sphere_stress_average` | |
| `local_gradient_gradient_concentration` | `local_strain_strain_concentration` | |
| `local_flux_gradient_concentration` | `local_stress_strain_concentration` | |
| `local_gradient_flux_concentration` | `local_strain_stress_concentration` | |
| `local_flux_flux_concentration` | `local_stress_stress_concentration` | |
| `local_gradient` | `local_strain` | |
| `local_flux` | `local_stress` | |

```@docs
MeanFieldHomogenization.LayeredSpheres.layer_stiffness_average
MeanFieldHomogenization.LayeredSpheres.layer_compliance_average
MeanFieldHomogenization.LayeredSpheres.layer_conductivity_average
MeanFieldHomogenization.LayeredSpheres.layer_resistivity_average
MeanFieldHomogenization.LayeredSpheres.layer_strain_average
MeanFieldHomogenization.LayeredSpheres.layer_stress_average
MeanFieldHomogenization.LayeredSpheres.sphere_strain_average
MeanFieldHomogenization.LayeredSpheres.sphere_stress_average
MeanFieldHomogenization.LayeredSpheres.local_strain_strain_concentration
MeanFieldHomogenization.LayeredSpheres.local_stress_strain_concentration
MeanFieldHomogenization.LayeredSpheres.local_strain_stress_concentration
MeanFieldHomogenization.LayeredSpheres.local_stress_stress_concentration
MeanFieldHomogenization.LayeredSpheres.local_strain
MeanFieldHomogenization.LayeredSpheres.local_stress
```
