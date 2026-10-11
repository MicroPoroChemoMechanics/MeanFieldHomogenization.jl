"""
    MeanFieldHomogenization.Conductivity

2nd-order Hill tensor for conductivity / diffusivity problems.
Extends the `_kernel` method table of [`MeanFieldHomogenization.Elasticity`] with
additional 2nd-order specializations.
"""
module Conductivity

using LinearAlgebra
using TensND
using QuadGK
using StaticArrays: SMatrix

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity
using ..Elasticity: Ellipsoid, Spherical, Prolate, Oblate, Triaxial,
    Circular, Elliptic, tens_IA,
    Cylinder, CylindricalShape, CircularCylindrical, EllipticCylindrical

include("hill_order2_3d.jl")
include("hill_order2_2d.jl")
include("api.jl")

end # module
