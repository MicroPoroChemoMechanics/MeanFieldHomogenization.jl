"""
    MeanFieldHomogenization.ToroidalInclusions

**Toroidal inhomogeneities**: a ring-shaped heterogeneity bounded by a torus of
circular cross-section, in an isotropic matrix.

A torus is a coordinate surface of the toroidal system, in which the harmonic
functions separate once a factor is taken out. The field around the torus is a
series of these functions, whose coefficients satisfy a three-term relation; the
far field of the series gives the concentration tensor of the inclusion. The
field in a torus is not uniform, so that there is no Hill tensor: the inclusions
below supply their concentration tensors (level B of the
[custom-inclusion interface](@ref man-custom-inclusions)).

Exports [`InsulatingTorus`](@ref), the toroidal pore of the conduction
problem, and its shape tag [`ToroidalShape`](@ref).
"""
module ToroidalInclusions

using LinearAlgebra: Tridiagonal, dot
using TensND

import ..Core

export InsulatingTorus, ToroidalShape

include("series.jl")
include("insulating.jl")

end # module
