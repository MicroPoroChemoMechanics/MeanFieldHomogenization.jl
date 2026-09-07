# [API — Superspheres](@id api-superspheres)

The superspherical and superspheroidal shape family: closed-form geometry, with
no mesh and no solver involved. What turns one of these shapes into a phase of
an [`RVE`](@ref) is a finite-element solve of its Eshelby problem, or a
surrogate trained on those solves.

The exponent is ``2p``, not ``p`` — see the warning on
[`Supersphere`](@ref MeanFieldHomogenization.Superspheres.Supersphere).

```@docs
MeanFieldHomogenization.Superspheres
MeanFieldHomogenization.Superspheres.AbstractSuperShape
MeanFieldHomogenization.Superspheres.Supersphere
MeanFieldHomogenization.Superspheres.Superspheroid
MeanFieldHomogenization.Superspheres.shape_exponent
MeanFieldHomogenization.Superspheres.is_concave
MeanFieldHomogenization.Superspheres.level_set
MeanFieldHomogenization.Superspheres.radial_distance
MeanFieldHomogenization.Superspheres.surface_point
MeanFieldHomogenization.Superspheres.outward_normal
MeanFieldHomogenization.Superspheres.diagonal_radius
MeanFieldHomogenization.Superspheres.edge_radius
MeanFieldHomogenization.Superspheres.bounding_radius
MeanFieldHomogenization.Superspheres.inner_radius
MeanFieldHomogenization.Superspheres.shape_volume
MeanFieldHomogenization.Superspheres.projected_area
MeanFieldHomogenization.Superspheres.equivalent_sphere_radius
```
