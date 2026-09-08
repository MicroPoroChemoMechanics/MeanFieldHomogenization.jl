# [API — Superspheres](@id api-superspheres)

The superspherical and superspheroidal shape family: closed-form geometry, with
no mesh and no solver involved. What turns one of these shapes into a phase of
an [`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) is a finite-element solve of its Eshelby problem, or a
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
MeanFieldHomogenization.Superspheres.has_coordinate_mirrors
MeanFieldHomogenization.Superspheres.check_coordinate_mirrors
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

## The triangulated surface

No CAD kernel can represent `|x|^{2p} + |y|^{2p} + |z|^{2p} = a^{2p}`, so the
surface is discretized here and handed to a mesher as a *discrete* entity. What
it is discretized from is a subdivided **octahedron**, for three reasons at
once: its face edges lie exactly in the coordinate planes, so an octant is
native and cubic symmetry is exploitable with no tolerance offset; for a concave
``p < 1/2`` those same planes are where the surface creases, so mesh edges land
on the creases rather than straddling them; and there is no polar degeneracy.
And the octahedron *is* the ``p = 1/2`` supersphere, which makes that value a
bit-exact oracle rather than a converging one.

Unlike the geometry above, this part is deliberately **not** type-generic: a
mesh exists to be handed to a mesher, which wants floating point.

```@docs
MeanFieldHomogenization.Superspheres.TriSurface
MeanFieldHomogenization.Superspheres.node_count
MeanFieldHomogenization.Superspheres.triangle_count
MeanFieldHomogenization.Superspheres.octant_patch
MeanFieldHomogenization.Superspheres.unit_octahedron
MeanFieldHomogenization.Superspheres.project_to_shape!
MeanFieldHomogenization.Superspheres.shape_surface
MeanFieldHomogenization.Superspheres.relax_surface!
MeanFieldHomogenization.Superspheres.mesh_area
MeanFieldHomogenization.Superspheres.mesh_volume
MeanFieldHomogenization.Superspheres.edge_lengths
MeanFieldHomogenization.Superspheres.mesh_quality
MeanFieldHomogenization.Superspheres.boundary_chains
MeanFieldHomogenization.Superspheres.patch_corners
```
