"""
    MeanFieldHomogenization.Viscoelasticity

Ageing linear viscoelastic (ALV) homogenization.  Provides:

  * [`ViscoLaw`](@ref) — relaxation `R(t,t')` or creep `J(t,t')` kernel,
    scalar- or 4-tensor-valued, with built-in Maxwell / Kelvin
    constructors.
  * [`trapezoidal_matrix`](@ref) — discretization of the Stieltjes
    integral on a time grid into a lower-block-triangular matrix
    (`n×n` for scalar, `6n×6n` for 4-tensor in Mandel form).
  * [`volterra_inverse`](@ref) — block forward-substitution that takes
    a discrete relaxation kernel to the corresponding creep kernel
    (and vice versa).
  * [`iso_params_from_blocks`](@ref) / [`iso_blocks_from_params`](@ref)
    and their `ti_` / `ortho_` counterparts — conversions between
    symmetry-structured per-component scalar matrices and the full
    `6n×6n` block matrix.
  * `hill_kernel` — discrete ALV Hill polarization tensor for an
    ellipsoidal inclusion, isotropic-matrix branch using the
    time-space decoupling formula
    ([barthelemyIJSS2016](@cite), App. *ALV Hill kernel*).
  * Time-domain viscoelastic homogenization schemes (Voigt, Reuss,
    Dilute, DiluteDual, Mori-Tanaka, Maxwell, Self-Consistent),
    plugged into the existing [`homogenize`](@ref MeanFieldHomogenization.Core.homogenize)
    dispatcher whenever a phase carries a `ViscoLaw` property.

All ALV operators are stored as dense `Matrix{T}` of size `(B·n)×(B·n)`
(`B = 6` for 4-tensor, `B = 1` for scalar) with explicit zeros above
the block diagonal — this is the convention of
[sanahuja2013](@cite) and the C++ ECHOES reference.
"""
module Viscoelasticity

using LinearAlgebra
using TensND

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity
import ..Elasticity: tens_UA, tens_VA, tens_IA, Ellipsoid, Spheroid
import ..Cracks
using ..Elliptic: is_hard_numeric

using ..Cracks: EllipticCrack, RibbonCrack, PennyCrack,
    crack_basis, crack_normal, aspect_ratio,
    semi_minor, semi_major
import ..LayeredSpheres
using ..LayeredSpheres: LayeredSphere, layer_radius, layer_modulus,
    layer_interface, AbstractInterface, PerfectInterface,
    SpringInterface, MembraneInterface,
    layer_count, layer_volume_fraction, outer_radius
import ..Schemes
import ..Laminates
using ..Schemes: RVE, HomogenizationScheme, Laminated, Voigt, Reuss, Dilute, DiluteDual,
    MoriTanaka, Maxwell, SelfConsistent, AsymmetricSelfConsistent,
    PonteCastanedaWillis, DifferentialScheme,
    Proportional, Sequential, CustomPath, Path,
    UniformDistribution,
    AndersonDefault, NewtonDefault,
    matrix_phase,
    inclusion_phase_names, matrix_property, phase_property,
    volume_fraction, matrix_volume_fraction,
    AbstractSymmetrize, NoSymmetrize, IsoSymmetrize, TISymmetrize,
    phase_symmetrize,
    VolumeFraction, CrackDensity, amount_value
using ForwardDiff
using OrdinaryDiffEq
using SpecialFunctions: gamma, expintx

# ── The Laplace-Carson half (non-ageing) ────────────────────────────────────
# `laplace_inversion.jl` depends on nothing from this package, so it goes
# first; the rheology catalog then needs it for its fallbacks, and
# `rheology_iso.jl` needs `ViscoLaw` from `visco_law.jl` for the bridge.
include("laplace_inversion.jl")
include("mittag_leffler.jl")
include("rheology_interface.jl")
include("prony.jl")
include("rheology_models.jl")

# ── The ageing half (time domain) ───────────────────────────────────────────
include("visco_law.jl")
include("rheology_iso.jl")
include("trapezoidal.jl")
include("volterra_inverse.jl")
include("conversions.jl")
include("hill_alv.jl")
include("schemes_alv.jl")
include("iso_schemes_alv.jl")
include("ti_schemes_alv.jl")
include("ortho_schemes_alv.jl")
include("alv_kernel_types.jl")
include("schemes_alv_sc.jl")
include("schemes_alv_sc_newton.jl")
include("schemes_alv_extra.jl")
include("layered_alv.jl")
include("laminate_alv.jl")
include("homogenize_alv.jl")
include("order2_alv.jl")
include("cracks_alv.jl")

# ── Exports ─────────────────────────────────────────────────────────────────
export AbstractViscoLaw, ViscoLaw, VALID_VISCO_MODES
export visco_mode, visco_eval
export maxwell_relaxation, kelvin_creep, maxwell_iso, kelvin_iso, heaviside_law
export trapezoidal_matrix
export volterra_inverse, volterra_product, volterra_divide, volterra_left_divide
export iso_params_from_blocks, iso_blocks_from_params
export ti_params_from_blocks, ti_blocks_from_params
export ortho_params_from_blocks, ortho_blocks_from_params
export AbstractALVKernel, ALVKernelISO, ALVKernelTI, ALVKernelOrtho
export hill_kernel
export dilute_concentration_alv, dilute_contribution_alv
export voigt_alv, reuss_alv, dilute_alv, dilute_dual_alv
export mori_tanaka_alv, maxwell_alv
export voigt_alv_iso, reuss_alv_iso, dilute_alv_iso, dilute_dual_alv_iso
export mori_tanaka_alv_iso, maxwell_alv_iso
export dilute_concentration_alv_iso, dilute_contribution_alv_iso
export voigt_alv_ti, reuss_alv_ti, dilute_alv_ti, dilute_dual_alv_ti
export mori_tanaka_alv_ti, maxwell_alv_ti
export dilute_concentration_alv_ti, dilute_contribution_alv_ti
export voigt_alv_ortho, reuss_alv_ortho, dilute_alv_ortho, dilute_dual_alv_ortho
export mori_tanaka_alv_ortho, maxwell_alv_ortho
export dilute_concentration_alv_ortho, dilute_contribution_alv_ortho
export self_consistent_alv, asymmetric_self_consistent_alv,
    pcw_alv, differential_alv
export bulk_localization_alv, bulk_state_seq_alv, shear_localization_alv
export strain_strain_loc_alv, stiffness_contribution_alv
export homogenize_alv, has_visco_property
export iso_order2_params_from_blocks, iso_order2_blocks_from_params
export hill_kernel_order2
export voigt_alv_order2, reuss_alv_order2, dilute_alv_order2,
    dilute_dual_alv_order2, mori_tanaka_alv_order2, maxwell_alv_order2
export dilute_concentration_alv_order2, dilute_contribution_alv_order2
export homogenize_alv_order2
export laminate_alv
export cod_kernel_alv, compliance_contribution_alv, delta_compliance_alv
export stiffness_contribution_alv, stiffness_contribution_alv_at, delta_stiffness_alv

# ── Laplace-Carson (non-ageing) ─────────────────────────────────────────────
export AbstractLaplaceInversion, GaverStehfest, FixedTalbot, TalbotTrefethen, DeHoog
export DEFAULT_INVERSION
export inverse_laplace, inverse_carson, inverse_carson_rate
export AbstractRheology, AbstractTensorRheology
export carson_relaxation, carson_creep, relaxation, creep, complex_modulus
export storage_modulus, loss_modulus, loss_factor
export glassy_modulus, equilibrium_modulus, is_fluid, default_inversion
export PronyRelaxation, PronyCreep, PRONY_MERGE_TOL, PRONY_FLUID_TOL
export maxwell_to_kelvin, kelvin_to_maxwell
export prony_fit_relaxation, prony_fit_creep
export Spring, Dashpot, MaxwellUnit, KelvinUnit
export relaxation_time, retardation_time
export zener_maxwell, zener_kelvin, burgers
export ScottBlair, FractionalMaxwell, FractionalKelvin, FractionalZener, Rabotnov
export HuetSayegh, Model2S2P1D, creep_kernel, carson_creep_kernel, creep_kernel_law
export LogarithmicCreep
export AbstractIsoPairing, BulkShear, YoungPoisson, IsoRheology
export iso_rheology, iso_rheology_E_nu
export homogenize_lc

end # module
