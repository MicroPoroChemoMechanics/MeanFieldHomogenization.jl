"""
    MeanFieldHomogenization.LayeredSpheroids

Isotropic `n`-layer confocal spheroidal composite inclusion (core +
concentric confocal shells), in **conduction** (thermal / electric /
Darcy) and in **axisymmetric elasticity**
([`spheroid_elastic_coefficients`](@ref), case I of Duan et al. 2005, with
perfect interfaces).

The two do not share a solver shape. Confocal surfaces are not homothetic, so
the harmonic degrees couple — in conduction only at an imperfect interface,
which is why its blocks are diagonal in the perfect case, but in elasticity at
every interface. There is no per-interface transfer matrix to chain on the
elastic side; `elasticity.jl` assembles one global system instead. Public
entry points: [`LayeredSpheroid`](@ref),
[`layered_spheroid_from_fractions`](@ref).

Like [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere),
a composite spheroid has **no Hill tensor** — it plugs into the
mean-field schemes through its volume-averaged concentration
(gradient/flux) tensors, assembled layer by layer via a confocal
spheroidal-harmonic transfer-matrix recurrence (`conductivity.jl`)
instead of the sphere's simple 2×2 state-vector propagation. Perfect
(`PerfectInterface`), Kapitza (`KapitzaInterface`, resistance) and
surface-conductive (`SurfaceConductiveInterface`, conductance)
interfaces — reused from [`LayeredSpheres`](@ref
MeanFieldHomogenization.LayeredSpheres) — couple different harmonic degrees, unlike
the sphere, requiring the truncated series machinery of
`legendre.jl` / `coupling.jl`.

That path into the schemes now works in **elasticity** too, prolate and oblate
alike: [`spheroid_strain_concentration`](@ref) assembles the full transversely
isotropic concentration tensor from the three elementary problems, and
`strain_strain_loc` / `stiffness_contribution` hand it to the schemes. Only
perfect interfaces, and only the default axis `ê₃`.
"""
module LayeredSpheroids

using LinearAlgebra
using QuadGK
using TensND

import ..Core
using ..Core
const MFH_Core = Core

import ..LayeredSpheres: _iso_bulk_shear
import ..LayeredSpheres: PerfectInterface, KapitzaInterface, SurfaceConductiveInterface,
    AbstractInterface, interfaces_eltype,
    layer_conductivity_average, layer_resistivity_average,
    layer_count, layer_modulus, layer_interface, layer_volume_fraction
# Pointwise-field generics are declared by `LayeredSpheres` (included first) and
# extended here, so a single binding carries both inclusion families.
import ..LayeredSpheres: get_layer, local_temperature, local_gradient, local_flux,
    local_gradient_gradient_loc, local_flux_gradient_loc,
    local_gradient_flux_loc, local_flux_flux_loc

import ..Core: strain_strain_loc, stiffness_contribution
import ..Core: gradient_gradient_loc, flux_gradient_loc, gradient_flux_loc, flux_flux_loc,
    conductivity_contribution, resistivity_contribution, is_homogeneous_inclusion

# `is_hard_numeric` — the numeric/symbolic split used throughout the package.
using ..Elliptic: is_hard_numeric

include("legendre.jl")
include("coupling.jl")
include("geometry.jl")
include("conductivity.jl")     # confocal-harmonic transfer-matrix recurrence
include("pn_modes.jl")           # one PN harmonic mode -> u and traction
include("elasticity.jl")         # elastic case I closed forms (cross-check reference)
include("elastic_cases.jl")     # the three elementary problems -> concentration
include("localfields.jl")        # pointwise T, ∇T, flux reconstruction
include("scheme_integration.jl") # concentration tensors → mean-field schemes

# ── Exports ─────────────────────────────────────────────────────────────────
export LayeredSpheroid, layered_spheroid_from_fractions, confocal_layer_radii
export layer_count, layer_q, layer_modulus, layer_interface, layer_semiaxes,
    layer_volume_fraction, outer_semiaxes
export local_temperature, local_gradient, local_flux
export spheroid_state_sequence, spheroid_ba_ratios, get_layer
export LayeredSpheroidTransportFields
export AxisymmetricCase, spheroid_elastic_coefficients, spheroid_core_strain
export TransverseShearCase, LongitudinalShearCase, spheroid_strain_concentration

end # module
