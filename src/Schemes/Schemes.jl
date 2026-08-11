"""
    MeanFieldHomogenization.Schemes

Mean-field homogenization schemes.  Provides the [`RVE`](@ref) container
(matrix + named phases with their volume fractions or crack densities, plus
an optional distribution shape) and the suite of homogenization
[`HomogenizationScheme`](@ref) types: bounds (`Voigt`, `Reuss`), one-shot
schemes with a matrix (`Dilute`, `DiluteDual`, `MoriTanaka`, `Maxwell`,
`PonteCastanedaWillis`), iterative self-consistent schemes
(`SelfConsistent`, `AsymmetricSelfConsistent`) and the differential
scheme (`Differential`) with user-selectable trajectory.

Public entry point: [`homogenize(rve, scheme; property=:C)`](@ref).  The
scheme can also be passed as a `Symbol` shortcut (`:MT`, `:SC`, …).
"""
module Schemes

using LinearAlgebra
using TensND
using Tensors
using ForwardDiff
using OrdinaryDiffEq

import ..Core
using ..Core
const MFH_Core = Core

# The homogenization *cell* contract is declared in `Core` (`Core/cells.jl`)
# so that `Schemes`, `Laminates` and user code all attach their methods to one
# canonical function per generic. `RVE` is one implementation of it; the
# laminate unit cell is another.
import ..Core: AbstractHomogenizationCell, AbstractParameter,
    homogenize, validate_cell, _evaluate, get_param, set_param,
    cell_member_names, cell_container_property, cell_set_property,
    Homogenized, NestedParameter, nested, resolve_property

# Forward declarations of inclusion types we touch from the other sub-modules
# (loaded earlier than Schemes by `MeanFieldHomogenization.jl`).
import ..Elasticity: Ellipsoid, hill_tensor
import ..Core: compliance_contribution, delta_compliance, delta_resistivity,
    compliance_and_stiffness_contribution
import ..LayeredSpheres
import ..LayeredSpheroids
import ..LayeredSpheres: layer_stiffness_average, layer_compliance_average,
    layer_conductivity_average, layer_resistivity_average

include("rve.jl")
include("symmetrize.jl")
include("orientation.jl")
include("scheme_types.jl")
include("homogenize.jl")
include("contribution_helpers.jl")
include("voigt.jl")
include("reuss.jl")
include("dilute.jl")
include("dilute_dual.jl")
include("mori_tanaka.jl")
include("maxwell.jl")
include("pcw.jl")
include("self_consistent.jl")
# Needs `_sc_solid_averages` (self_consistent.jl) and `_frob_sq`, hence its
# position after every scheme body it decomposes.
include("crack_families.jl")
include("trajectory.jl")
include("differential.jl")
include("parameters.jl")
include("sensitivities.jl")

# ── Exports ────────────────────────────────────────────────────────────────
# Data model
export AbstractAmount, VolumeFraction, CrackDensity
export AbstractDistributionShape, UniformDistribution
export AbstractSymmetrize, NoSymmetrize, IsoSymmetrize, TISymmetrize
export AbstractHomogenizationCell, Homogenized, NestedParameter, nested
export Phase, RVE
export add_matrix!, add_phase!
export matrix_phase, inclusion_phase_names
export phase_property, phase_property_raw, matrix_property
export validate_cell
export volume_fraction, crack_density, matrix_volume_fraction
export phase_symmetrize
export validate_rve, promote_rve
export best_fit_ti, best_fit_iso, best_fit_ortho
export polar_orientation_bins

# Schemes
export HomogenizationScheme
export Voigt, Reuss, Laminated, Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis
export SelfConsistent, AsymmetricSelfConsistent
export ClusterModel, EquivalentInclusion
export AndersonDefault, NewtonDefault, AutoNonlinear
export DifferentialTrajectory, Proportional, Sequential, CustomPath, Path, DifferentialScheme

# Entry point
export homogenize, differential_path
export crack_family_compliances, crack_family_residual

# Sensitivities — lentilles paramétriques + wrappers ForwardDiff (extension)
export AbstractParameter, AmountParameter, PropertyParameter,
    GeometryParameter, DistributionShapeParameter
export amount, property, geometry, shape_param
export get_param, set_param
export derivative, gradient, jacobian, sensitivity

end # module
