"""
    MeanFieldHomogenization.Laminates

Periodic **multilayer** homogenization: a unit cell of parallel layers of
common normal `n`, with no matrix, no auxiliary Eshelby problem and no
reference medium — and an exact analytical solution rather than an estimate.

This is a different kind of microstructure from the one the rest of the
package addresses. The schemes of `Schemes` (bounds, dilute, Mori-Tanaka,
Maxwell, Ponte-Castañeda-Willis, self-consistent, differential) all describe
**random** morphologies through the Eshelby auxiliary problem; some of them
single out a matrix phase and some do not, but all of them estimate. A
laminate is **periodic and deterministic**, and its effective behavior
follows in closed form from two continuity conditions:

- the traction `σ·n` is continuous — the *out-of-plane* stress components;
- the in-plane strain is continuous and equal to the macroscopic one, so
  `ε_i = E + a_i ⊗ˢ n`.

The exported [`Laminate`](@ref) is therefore an
[`AbstractHomogenizationCell`](@ref) alongside `RVE`, solved by the
[`Laminated`](@ref MeanFieldHomogenization.Schemes.Laminated) scheme, and it plugs into the multiscale chain like any
other cell — including declaratively, as the value of a phase property (see
[`Homogenized`](@ref)).

Provides:

- [`Laminate`](@ref), [`Layer`](@ref), [`add_layer!`](@ref) and the accessors;
- elasticity **and** transport, dispatched on the order of the stored
  property, exactly as the mean-field schemes are;
- the four imperfect-interface models of `LayeredSpheres`, reused unchanged:
  [`SpringInterface`](@ref) / [`MembraneInterface`](@ref) in elasticity,
  [`KapitzaInterface`](@ref) / [`SurfaceConductiveInterface`](@ref) in
  transport, each entering with an interface *density* `1/L`;
- per-layer localization ([`layer_strain_localization`](@ref), …), the two
  Hill tensors ([`laminate_hill`](@ref)) and the interface jumps
  ([`interface_jump`](@ref));
- parameter lenses [`ThicknessParameter`](@ref) and
  [`InterfaceParameter`](@ref), so `derivative` / `gradient` / `jacobian`
  reach a thickness or an interface compliance.

The block algebra itself lives in `Core/laminate_algebra.jl`, so that the
ageing-viscoelastic laminate reuses the very same kernel with the Volterra
inversion substituted.
"""
module Laminates

using LinearAlgebra
using StaticArrays
using TensND

import ..Core
using ..Core
const MFH_Core = Core

# `is_hard_numeric` — the numeric/symbolic split used throughout the package.
# NOT `T <: Real`: `Symbolics.Num` subtypes `Real` and answers no comparison.
using ..Elliptic: is_hard_numeric

import ..Core: AbstractHomogenizationCell, AbstractParameter,
    homogenize, validate_cell, _evaluate, get_param, set_param,
    cell_member_names, cell_container_property, cell_set_property,
    Homogenized, NestedParameter, nested, resolve_property

# The package-wide localization generics, declared bodyless in
# `Core/abstractions.jl`. A laminate is a cell, not an inclusion, so the
# methods added here are keyed on a layer NAME rather than on a `(C₁, C₀)`
# pair — the same shape as `strain_strain_loc(::LayeredSphere, C₀; layer)`.
import ..Core: strain_strain_loc, stress_strain_loc, strain_stress_loc,
    stress_stress_loc, gradient_gradient_loc, flux_gradient_loc,
    gradient_flux_loc, flux_flux_loc

# The five interface models are shared with the layered sphere and spheroid:
# a planar interface is the curvature-free case of a spherical one, so the
# types, their fields and their conventions carry over unchanged.
#
# The layered-morphology accessors are likewise EXTENDED, not redefined —
# `layer_count`, `layer_interface` and `layer_volume_fraction` mean the same
# thing for a sphere, a spheroid and a laminate, and `LayeredSpheroids`
# already imports them from `LayeredSpheres` for exactly this reason.
# Declaring rival bindings here would make the names ambiguous at `using
# MeanFieldHomogenization`.
import ..LayeredSpheres: spring_compliances, spring_stiffnesses,
    _spring_from_compliances
import ..LayeredSpheres: AbstractInterface, PerfectInterface, SpringInterface,
    MembraneInterface, KapitzaInterface, SurfaceConductiveInterface,
    layer_count, layer_interface, layer_volume_fraction

import ..Schemes
using ..Schemes: HomogenizationScheme, Laminated, Voigt, Reuss,
    AmountParameter, PropertyParameter

include("laminate.jl")
include("interfaces_laminate.jl")
include("evaluate.jl")
include("parameters.jl")

# ── Exports ────────────────────────────────────────────────────────────────
# Cell
export Laminate, Layer, add_layer!
export layer_names, layer_count, layer_property, layer_property_raw
export layer_thickness, layer_volume_fraction, layer_interface
export laminate_period, laminate_basis, laminate_normal
export validate_laminate

# Anisotropic interfaces (laminate-specific: a plane imposes no symmetry on
# the interface, unlike the spherical recurrence)
export AnisotropicSpringInterface, AnisotropicMembraneInterface
export AnisotropicSurfaceConductiveInterface

# Fields
export laminate_hill
export layer_strain_localization, layer_stress_localization
export layer_gradient_localization, layer_flux_localization
export interface_jump

# Lenses
export ThicknessParameter, InterfaceParameter, thickness, interface_param

end # module
