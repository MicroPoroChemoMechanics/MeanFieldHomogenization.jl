"""
    MeanFieldHomogenization.Superspheres

Superspherical and superspheroidal morphologies: shapes of the family

```math
|x/a|^{2p} + |y/a|^{2p} + |z/a|^{2p} \\le 1
\\qquad\\text{and}\\qquad
(\\rho/a)^{2p} + |z/c|^{2p} \\le 1 ,
```

which interpolate between the sphere (``p = 1``), the octahedron or double cone
(``p = 1/2``), the concave range (``p < 1/2``) and the cube or cylinder
(``p \\to \\infty``).

**These are not ellipsoids and have no closed-form Eshelby solution**, which is
the whole reason they are here: they are the family the package's
non-ellipsoidal routes were built for. A shape becomes a phase of an
[`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) either through a finite-element solve of its Eshelby problem or
through a surrogate trained on those solves.

This file holds the geometry alone — closed-form level set, radial map, normals,
characteristic radii, and the exact volume and shadow area that any mesh is
measured against. Nothing here meshes or solves.

Convention: the exponent is ``2p``, after the Sevostianov–Giraud–Chen–Grgic
literature. See the warning on [`Supersphere`](@ref).
"""
module Superspheres

using TensND

import SpecialFunctions: loggamma

import ..Core

# `is_hard_numeric` — the numeric/symbolic split used throughout the package.
using ..Elliptic: is_hard_numeric

export AbstractSuperShape, Supersphere, Superspheroid
export shape_exponent, is_concave, is_convex, is_sphere
export level_set, radial_distance, surface_point, outward_normal
export diagonal_radius, edge_radius, bounding_radius, inner_radius
export shape_volume, projected_area, equivalent_sphere_radius
export TriSurface, node_count, triangle_count
export octant_patch, unit_octahedron, project_to_shape!, shape_surface, relax_surface!
export mesh_area, mesh_volume, edge_lengths, mesh_quality

include("shapes.jl")
include("surface_mesh.jl")

end # module
