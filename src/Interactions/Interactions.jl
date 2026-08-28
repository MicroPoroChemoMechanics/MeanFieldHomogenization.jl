"""
    MeanFieldHomogenization.Interactions

Two-inclusion interaction tensors — the shared numerical ingredient of the
N-body homogenization models of the package.

Where every one-site scheme (Dilute, Mori-Tanaka, Maxwell, PCW,
self-consistent, differential) needs only the Hill tensor `ℙ` of a single
inclusion in a reference medium, the equivalent inclusion method
([`EquivalentInclusion`](@ref MeanFieldHomogenization.Schemes.EquivalentInclusion)) and the cluster model ([`ClusterModel`](@ref MeanFieldHomogenization.Schemes.ClusterModel))
need one object more: the tensor `𝕋^{ab}` measuring (minus) the field induced
in one inclusion by a uniform polarization of another, whose self term is
`𝕋^{aa} = +ℙ`. [brisard2014](@cite)
(2014), §3.1, observe that their influence pseudotensors of order zero
coincide with the interaction tensors of Berveiller et al. (1987) and
Molinari & El Mouden (1996) — so the two families share this module.

Contents
--------
- `api.jl`             : `interaction_tensor`, `self_interaction_tensor` and
                          the `_pair_kernel` table (plus the sign convention,
                          which is worth reading before transcribing formulas)
- `pair_ball_iso.jl`   : closed forms for two balls (3D) or two disks (2D) in
                          an isotropic reference — exact at any separation
- `pair_multipole.jl`  : truncated multipole expansion for general ellipsoids
- `pair_quadrature.jl` : brute-force product quadrature, the validation oracle
- `lattice_sums.jl`    : sums over the periodic images of a source inclusion
"""
module Interactions

using LinearAlgebra
using StaticArrays
using TensND
using ForwardDiff

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity
using ..Elasticity: Ellipsoid

include("pair_ball_iso.jl")
include("pair_multipole.jl")
include("pair_quadrature.jl")
include("api.jl")
include("lattice_sums.jl")

export interaction_tensor, self_interaction_tensor
export lattice_interaction_tensor, periodic_images

end # module
