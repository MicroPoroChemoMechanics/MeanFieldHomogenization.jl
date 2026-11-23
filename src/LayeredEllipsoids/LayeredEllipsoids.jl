"""
    MeanFieldHomogenization.LayeredEllipsoids

Isotropic `n`-layer inclusion bounded by **confocal ellipsoids** of three
unequal axes (a core and its confocal coatings), in **conduction**, in an
isotropic medium, with perfect interfaces. Public entry points:
[`LayeredEllipsoid`](@ref), [`layered_ellipsoid_from_fractions`](@ref).

Like the other layered inclusions it has no Hill tensor and enters the schemes
through its concentration tensors. They need no series: the mean gradients of
the layers satisfy as many linear relations as there are layers, which are
exact for confocal boundaries (`conductivity.jl`), and whose coefficients are
the Hill tensors of the ellipsoids, taken from `Elasticity.hill_tensor`.

- `geometry.jl`     : the type, its constructors and its accessors.
- `conductivity.jl` : the relations between the mean gradients, the
  concentration tensors and the entries of the schemes.
"""
module LayeredEllipsoids

using LinearAlgebra
using StaticArrays: SMatrix
using TensND
import Tensors

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity
import ..Elasticity: Ellipsoid
import ..LayeredSpheres: layer_count, layer_modulus, layer_volume_fraction,
    layer_property_average, layer_inverse_property_average
import ..LayeredSpheroids: layer_semiaxes, outer_semiaxes
import ..Core: gradient_gradient_concentration, flux_gradient_concentration,
    property_contribution, is_homogeneous_inclusion

using ..Elliptic: is_hard_numeric

include("geometry.jl")
include("conductivity.jl")

export LayeredEllipsoid, layered_ellipsoid_from_fractions

end # module
