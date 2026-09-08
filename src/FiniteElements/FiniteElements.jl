"""
    MeanFieldHomogenization.FiniteElements

Inclusions whose response is obtained from a **finite-element resolution of the
Eshelby problem** rather than from a closed form — the package's own
demonstration that the [`CustomInclusions`](@ref MeanFieldHomogenization.CustomInclusions)
contract is enough to reach every scheme.

| Type | Morphology | Discretization | Entry gate |
|---|---|---|---|
| [`FEEllipticCrack`](@ref) | flat elliptical crack | 3-D tetrahedra | COD tensor → crack algebra |
| [`FEExcenteredSphere`](@ref) | sphere with an off-center spherical core | axisymmetric Fourier | B — the two localization tensors |
| [`FESupershapePore`](@ref) | superspherical cavity, cube-symmetric | 3-D tetrahedra, whole cell or one octant | B — the strain side alone, the stress side being zero |
| [`FEAxiSupershapePore`](@ref) | superspheroidal cavity, a solid of revolution | axisymmetric Fourier | B — the strain side alone |
| [`FEAxiLayeredSpheroid`](@ref) | `N` nested coaxial spheroids, free semi-axes | axisymmetric Fourier | B — both localization tensors |

Both use the *finite Eshelby cell with a first-order corrected boundary
condition* of Adessina, Barthélémy, Lavergne & Ben Fraj, *Int. J. Eng. Sci.*
**119** (2017) 1-15: the infinite matrix is truncated to a ball of finite
radius and the truncation bias is removed by adding the inclusion's own dipole
far field to the imposed boundary displacement.

The types, the Fourier operators, the boundary data and the algebra of the
corrected boundary condition all live here. What a package extension supplies
is only the **discretization** — a mesh, scalar Lagrange spaces, an assembly
and a quadrature — through the generics of [`FEBackend`](@ref), which come in
three groups: the crack, the axisymmetric contract (ten methods, the tenth being
`fe_axi_pore_boundary`, needed by a cavity alone since it has no interior to
average over) and the three-dimensional cell.

Two backends exist: [`FerriteBackend`](@ref)
(`import Ferrite, FerriteGmsh, Gmsh`), which implements all three groups, and
[`GridapBackend`](@ref) (`import Gridap, GridapGmsh`), which implements the crack
and the axisymmetric ones. An inclusion built without naming one takes
[`AutoBackend`](@ref) and picks at its first solve; with neither loaded, that
solve errors informatively.

See `docs/src/manual/fe_inclusions.md`,
`docs/src/applications/recycled_aggregate.md` and
`docs/src/applications/concave_pores.md`.
"""
module FiniteElements

using TensND

using ..Superspheres

import LinearAlgebra
import Tensors

import ..Core
import ..Cracks
import ..Schemes
# The interface types, so a layered inclusion can name the ones it does not
# implement rather than accepting them silently.
import ..LayeredSpheres

export FECache, fe_assembly_count, fe_reset!, fe_available_gb
export FEBackend, AutoBackend, FerriteBackend, GridapBackend
export FEMeshOptions, FEEllipticCrack, fe_cod_breakdown, fe_mesh_report
export FEAxiMeshOptions, FEExcenteredSphere
export FECellMeshOptions, fe_cell_size_estimate
export FESupershapePore, SupershapePoreShape, has_surrogate, pore_shape_params
export check_nested_spheroids, axi_layer_set
export FEAxiLayeredSpheroid, LayeredSpheroidShape, layer_volumes, layer_fractions
export fe_cell_localization, fe_cell_mesh_report
export fe_cell_curved_volume, fe_cell_meshed_volume
export fe_axi_breakdown, fe_axi_mesh_report, fe_axi_localization
export fe_axi_pore_boundary
export FEAxiSupershapePore, AxiSupershapePoreShape
export fe_axi_pore_localization, fe_axi_pore_breakdown, fe_axi_pore_mesh_report

include("common.jl")
include("backends.jl")
include("crack.jl")
include("excentered_sphere.jl")

# The flat crack: shared gmsh geometry and the 3 + 3 driver.
include("crack_gmsh_geometry.jl")
include("crack_driver.jl")

# The axisymmetric solver: geometry, Fourier operators, algebra, driver.
include("axi_gmsh_geometry.jl")
include("axi_fourier.jl")
include("axi_algebra.jl")
include("axi_driver.jl")
include("axi_pore_gmsh_geometry.jl")
include("axi_layered_gmsh_geometry.jl")
include("axi_pore_driver.jl")
include("axi_supershape_pore.jl")
include("axi_layered_spheroid.jl")

# The three-dimensional cell around a non-ellipsoidal shape: geometry first.
include("cell_gmsh_geometry.jl")
include("cell_octant.jl")
include("cell_driver.jl")
include("supershape_pore.jl")

# ─── Sensitivity is not available through a finite-element geometry ──────────
#
#  `Schemes._replace_geom_field` rebuilds a geometry by copying every
#  non-`<:Number` field by reference — which for these types includes the
#  `FECache`.  The perturbed inclusion would therefore share the original's
#  memoized tensors, whose key is the reference medium alone, and hand back the
#  *unperturbed* answer; the derivative would come out as exactly zero, quietly.
#  The grid stored in that same cache was built for the original geometry too.
#
#  Even without the cache the answer would be zero: the solve converts its
#  geometry to `Float64` on entry, so a `ForwardDiff.Dual` loses its
#  perturbation at the door.  Refusing is the only honest option.

# `FESupershapePore` and `FEAxiSupershapePore` are deliberately absent: they
# refuse sensitivity only when answering from a mesh, and decide per object in
# their own files, because a surrogate-backed one *is* differentiable.
# `FEAxiLayeredSpheroid` has no surrogate route yet, so it refuses
# unconditionally — and it has to be named here, or the request falls through to
# a generic that would hand back a zero.
const _FEGeometry = Union{
    FEEllipticCrack, FEExcenteredSphere, FEAxiLayeredSpheroid,
}

_no_fe_sensitivity(geom, name) = error(
    "analytic sensitivity is not available through `$(nameof(typeof(geom)))`: " *
        "the finite-element solve runs in `Float64` and memoizes on the reference " *
        "medium alone, so differentiating `:$name` would silently return zero. " *
        "Use a finite difference over freshly constructed inclusions instead."
)

Schemes._replace_geom_field(geom::_FEGeometry, ::Val{name}, ::Nothing, value) where {name} =
    _no_fe_sensitivity(geom, name)
Schemes._replace_geom_field(geom::_FEGeometry, ::Val{name}, ::Int, value) where {name} =
    _no_fe_sensitivity(geom, name)

end # module
