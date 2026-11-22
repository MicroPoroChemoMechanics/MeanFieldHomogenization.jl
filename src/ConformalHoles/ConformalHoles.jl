"""
    MeanFieldHomogenization.ConformalHoles

**Cylindrical cavities whose cross-section is given by a conformal map**: the
outside of the section is the image of the outside of the unit disk by a map
with a finite number of terms,

```math
z = \\omega(\\zeta) = R\\,\\Bigl(\\zeta + \\sum_{k=1}^{N} m_k\\,\\zeta^{-k}\\Bigr) .
```

For such a map the complex potentials of the plane-strain problem are finite
sums, whose coefficients solve a linear system of size ``2N``; the antiplane
shear and the conduction around an insulating hole are known in closed form.
The matrix is isotropic, or transversely isotropic about the axis of the
cavity: it is then isotropic in the plane of the section. In a two-dimensional
cell the same hole is its section, with tensors of the plane.
The field in the cavity is not uniform and there is no Hill tensor: the hole
supplies its
concentration tensors (level B of the
[custom-inclusion interface](@ref man-custom-inclusions)).

Exports [`ConformalHole`](@ref), its shape tag [`ConformalShape`](@ref) and
[`polygon_map`](@ref), the terms of the map of a regular polygon.
"""
module ConformalHoles

using StaticArrays: SMatrix, SVector
using TensND
using TensND: is_hard_numeric

import ..Core
# The moduli of a medium that is isotropic, or transversely isotropic about a
# given axis: read as a layered cylinder reads those of its layers.
import ..LayeredCylinders: _hill_moduli, _conductivities

export ConformalHole, ConformalShape, polygon_map

include("map.jl")
include("hole.jl")

end # module
