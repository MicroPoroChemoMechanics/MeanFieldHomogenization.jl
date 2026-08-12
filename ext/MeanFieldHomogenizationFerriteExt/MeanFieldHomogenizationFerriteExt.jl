"""
    MeanFieldHomogenizationFerriteExt

Finite-element resolution of the Eshelby problem, activated by
`import Ferrite, FerriteGmsh, Gmsh`.

Two things live here. [`MeanFieldHomogenization.FEEllipticCrack`](@ref) is implemented in
full — mesh, solver and the 3+3 corrected scheme — because the gmsh `Crack`
plugin and the front welding it requires are Ferrite-grid surgery. For
[`MeanFieldHomogenization.FEExcenteredSphere`](@ref) only the nine methods of the
[`MeanFieldHomogenization.FEBackend`](@ref) contract are supplied, in `axi_backend.jl`;
the physics of that solve lives in `src/FiniteElements/`.

Both use the finite Eshelby cell with a first-order corrected boundary
condition of Adessina, Barthélémy, Lavergne & Ben Fraj, *Int. J. Eng. Sci.*
**119** (2017) 1-15.

The opposite coupling — a Ferrite computation using a whole microstructure as
its constitutive law — lives in `MeanFieldHomogenizationFerriteMaterialExt`,
which `import Ferrite` alone activates, so it never pulls gmsh in.

The point of the exercise is that nothing downstream knows: because the type
subtypes `AbstractCrack` and declares a standard `shape_trait`, supplying
`cod_tensor` is enough for ℍ, ℕ, 𝐑, 𝐍_K, the bundled pair and the four
`delta_*` to be inherited, and the crack drops into every scheme —
`symmetrize` included.
"""
module MeanFieldHomogenizationFerriteExt

using MeanFieldHomogenization
using TensND
using Ferrite
using FerriteGmsh
using Gmsh: gmsh
using Tensors
using LinearAlgebra

const FE = MeanFieldHomogenization.FiniteElements


include("crack_backend.jl")
include("axi_backend.jl")

end # module
