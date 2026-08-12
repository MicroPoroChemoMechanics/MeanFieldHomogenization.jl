"""
    MeanFieldHomogenizationFerriteMaterialExt

Ferrite glue for **MeanFieldHomogenization as a Gauss-point constitutive law** —
`mfh_states`, `mfh_element!`, `annulus_grid`.

Activated by `import Ferrite` alone. That is the point of keeping it separate
from [`MeanFieldHomogenizationFerriteExt`](@ref), which serves the opposite
coupling (finite elements *inside* the package, solving one inclusion's Eshelby
problem) and needs `FerriteGmsh` and `Gmsh` as well: a structural computation
that merely wants a homogenized material law has no reason to pull a ~100 MB
gmsh artifact into its environment, nor into a documentation build.

The file is deliberately thin — the contract in `src/Constitutive/` is
finite-element-agnostic, so all that is needed here is the per-quadrature-point
state bookkeeping Ferrite leaves to the user, and an element routine.
"""
module MeanFieldHomogenizationFerriteMaterialExt

using MeanFieldHomogenization
using TensND
using Ferrite
using Tensors
using LinearAlgebra

include("material_backend.jl")

end # module
