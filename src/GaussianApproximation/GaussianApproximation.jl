"""
    GaussianApproximation

Cracks and heterogeneities of arbitrary shape, by the approximation of Maz'ya
[mazya1996](@cite): the unknown field of an integral equation is expanded on
Gaussians centered at the nodes of a regular grid. The action of the integral
operator on a Gaussian is known in closed form, the matrix is a convolution
evaluated by FFT, and a shape enters only through the nodes it contains — no
mesh.

- [`AbstractRegion`](@ref): describing a shape (catalog, set operations,
  function, image);
- [`PlanarCrack`](@ref): a flat crack of any contour, accepted by every scheme;
- [`quasi_interpolate`](@ref), [`saturation_error`](@ref): the approximation
  itself.
"""
module GaussianApproximation

using LinearAlgebra
using StaticArrays
using TensND
import ForwardDiff
import Krylov
import FFTA
using SpecialFunctions: besselix, gamma, dawson

import ..Core as MFH_Core
import ..Cracks
import ..Schemes
import ..FiniteElements: FECache, _fe_memo, _fe_cache
using ..Elliptic: is_hard_numeric

include("grids.jl")
include("quasi_interpolation.jl")
include("regions.jl")
include("convolution.jl")
include("solve.jl")
include("kernels_crack.jl")
include("edge_rule.jl")
include("planar_crack.jl")

export GaussianGrid, gaussian_generator, saturation_error, QuasiInterpolant, quasi_interpolate
export AbstractRegion, bounding_box, measure, inside_nodes, coverage
export Ellipse, Disk, Superellipse, Rectangle, StarShape, Polygon, Annulus
export Region, PixelRegion, UnionRegion, IntersectionRegion, DifferenceRegion
export edge_rule, ContourShape, PlanarCrack, crack_area, crack_opening

end # module
