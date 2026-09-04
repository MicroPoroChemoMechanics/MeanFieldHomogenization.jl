"""
    MeanFieldHomogenization

Julia package for mean-field homogenization of heterogeneous materials.

`MeanFieldHomogenization` unifies the computation of Hill polarization tensors for
ellipsoidal inhomogeneities, crack opening displacement (COD) tensors, stress
and displacement intensity factors, homogenization schemes over representative
volume elements (RVEs), and ageing viscoelastic constitutive laws, sharing a
common abstraction for inclusions, algorithms, and material symmetry classes.

# Sub-modules

- `MeanFieldHomogenization.Elliptic`     — type-generic Legendre and Carlson elliptic
  integrals (`ForwardDiff`- and `Sym`-compatible).
- `MeanFieldHomogenization.Core`         — abstractions (`AbstractInclusion`,
  `AbstractAlgorithm`, `MaterialSymmetry`), shared numerics
  (Green / Newton kernels, Masson-style residue, DECUHR integrand), modulus
  extractors, and central dispatch.
- `MeanFieldHomogenization.Elasticity`   — Hill polarization for ellipsoidal inclusions
  and infinite cylinders (2D / 3D, isotropic and anisotropic matrix).
- `MeanFieldHomogenization.Cracks`       — COD tensors, compliance contributions, SIF
  and DIF for elliptic and ribbon cracks.
- `MeanFieldHomogenization.Conductivity` — 2nd-order Hill tensor for conductivity /
  diffusion problems.
- `MeanFieldHomogenization.Interactions` — two-inclusion interaction tensors
  `Γ^{ab}`, the shared ingredient of the N-body schemes (equivalent inclusion
  method, cluster model): closed forms for ball and disk pairs, multipole
  expansion for general ellipsoids, and periodic lattice sums.
- `MeanFieldHomogenization.LayeredSpheres`   — `n`-layer composite spheres with five
  interface types, volume-average and pointwise localization.
- `MeanFieldHomogenization.LayeredSpheroids` — `n`-layer confocal spheroids in
  conduction, with imperfect interfaces.
- `MeanFieldHomogenization.Schemes`      — RVEs, amounts, symmetrization and the
  homogenization schemes themselves (dilute, Mori–Tanaka, self-consistent,
  PCW, Maxwell, differential).
- `MeanFieldHomogenization.Poromechanics`  — poroelastic upscaling of a saturated
  medium with a homogeneous solid phase: Biot tensor and modulus, drained ↔
  undrained conversion, Skempton tensor, effective stresses. A post-processor
  of a homogenized stiffness, not a scheme.
- `MeanFieldHomogenization.Viscoelasticity`  — linear viscoelasticity by two routes:
  the ageing time-domain one through Volterra operators (`homogenize_alv`), and
  the non-ageing Laplace-Carson one (`homogenize_lc`), with a catalog of
  rheological models, numerical Laplace inversion, and the exact Kelvin ↔
  Maxwell conversion joining them.
- `MeanFieldHomogenization.CustomInclusions` — the user-defined inclusion contract:
  `CustomInclusion` and `check_inclusion_interface`.
- `MeanFieldHomogenization.FiniteElements`   — inclusions whose response comes out of a
  finite-element resolution of the Eshelby problem (`FEEllipticCrack`,
  `FEExcenteredSphere`); the discretization comes from a backend extension,
  `MeanFieldHomogenizationFerriteExt` or `MeanFieldHomogenizationGridapExt`.
- `MeanFieldHomogenization.NeuralInclusions` — inclusions whose response comes out of a
  trained neural network (`NeuralHillInclusion`, `NeuralLocalizationInclusion`),
  together with the sampling and fitting machinery; the optimizer comes from
  `MeanFieldHomogenizationLuxExt`, evaluation needs no extra dependency.
- `MeanFieldHomogenization.Constitutive` — the package **as a constitutive law at
  each Gauss point** of a structural finite-element computation, the role an
  MFront behavior or an Abaqus UMAT plays: `material_response`, per-point
  internal state, consistent tangent, and the `Tensors.jl` bridge. The mirror
  image of `FiniteElements` — there the finite elements are inside MFH, here MFH
  is inside the finite-element code.

# Shared generic interface

```julia
hill_tensor(ell::AbstractEllipsoidalInclusion, C₀; method=:auto, ...)
cod_tensor(crack::AbstractCrack, C₀; method=:auto, ...)
compliance_contribution(crack, C₀; method=:auto, ...)    # returns H (or R)
delta_compliance(crack, H, ε)                             # ΔS = factor · ε · H
delta_resistivity(crack, R, ε)                            # ΔR = factor · ε · R
sif(crack, C₀, Σ; method=:auto, ...)
dif(crack, C₀, Σ; method=:auto, ...)
```

All high-level entry points share the same algorithmic traits
(`Analytical`, `Residue`, `DECUHR`) and the same material-symmetry dispatch
rules. See the developer documentation (`docs/src/developer/`) for guidance
on extending the package with new inclusions, algorithms or schemes.
"""
module MeanFieldHomogenization

using TensND

include("Elliptic/Elliptic.jl")
include("Core/Core.jl")
include("Elasticity/Elasticity.jl")
include("Cracks/Cracks.jl")
include("Conductivity/Conductivity.jl")
# `Interactions` needs the ellipsoid types and `hill_tensor` from `Elasticity`
# and the 2nd-order kernels registered by `Conductivity`, and is needed in turn
# by the N-body schemes — so it sits between the two.
include("Interactions/Interactions.jl")
include("LayeredSpheres/LayeredSpheres.jl")
include("LayeredSpheroids/LayeredSpheroids.jl")
include("Schemes/Schemes.jl")
# `Laminates` sits between `Schemes` and `Viscoelasticity`: it extends
# `_evaluate` and uses `HomogenizationScheme`/`Laminated`/`Voigt`/`Reuss` from
# the former, and the latter needs `Laminate` for the ageing-viscoelastic
# multilayer.
include("Laminates/Laminates.jl")
# `Assemblies` follows the same pattern as `Laminates`: the scheme *types* are
# declared in `Schemes`, their kernels live here with the cell they act on.
include("Assemblies/Assemblies.jl")
# `Poromechanics` is a *post-processor* of a homogenized stiffness, not a
# scheme: it only needs `Schemes` for the RVE convenience layer, and adds no
# `_evaluate` method. Placed here so the poroelastic parameters are available
# to the Gauss-point constitutive laws further down.
include("Poromechanics/Poromechanics.jl")
include("Viscoelasticity/Viscoelasticity.jl")

using .Elliptic
using .Core
using .Elasticity
using .Cracks
using .Conductivity
using .Interactions
using .LayeredSpheres
using .LayeredSpheroids
using .Schemes
using .Laminates
using .Assemblies
using .Poromechanics
using .Viscoelasticity

# ─── Localization + contribution (top-level: need all sub-module APIs) ──────
# Every generic is declared in `Core`; sub-modules (and user code) extend them
# via qualified imports so that all methods attach to the same canonical
# function.
import .Core: strain_strain_loc, stress_strain_loc, strain_stress_loc,
    stress_stress_loc, gradient_gradient_loc, flux_gradient_loc,
    gradient_flux_loc, flux_flux_loc,
    stiffness_contribution, conductivity_contribution,
    resistivity_contribution, compliance_contribution,
    delta_stiffness, delta_conductivity, delta_compliance, delta_resistivity,
    loc_and_stiffness, loc_and_stress_average,
    compliance_and_stiffness_contribution, is_homogeneous_inclusion
import .Elasticity: hill_tensor

include("localization.jl")
include("contribution.jl")

# ─── Sub-modules that build on the generic algebra above ────────────────────
include("CustomInclusions/CustomInclusions.jl")
include("FiniteElements/FiniteElements.jl")
include("NeuralInclusions/NeuralInclusions.jl")
# `Constitutive` turns a whole cell + scheme into a Gauss-point material law, so
# it comes after every inclusion family a microstructure may hold. It is the
# mirror image of `FiniteElements`: there the FE code is inside MFH, here MFH is
# inside the FE code.
include("Constitutive/Constitutive.jl")

using .CustomInclusions
using .FiniteElements
using .NeuralInclusions
using .Constitutive

# ─── MFH Studio launcher ─────────────────────────────────────────────────────
include("Studio.jl")

# ── Abstractions ─────────────────────────────────────────────────────────────
export AbstractInclusion, AbstractEllipsoidalInclusion, AbstractCrack
export AbstractLayeredInclusion, AbstractCustomInclusion
export AbstractAlgorithm, Analytical, Residue, DECUHR, NestedQuadGK,
    CylinderQuadrature, Multipole, Auto
export MaterialSymmetry, IsotropicSym, TransverselyIsotropicSym,
    OrthotropicSym, GeneralAnisotropicSym
export material_symmetry, dimension, inclusion_basis, shape_trait, shape_tensor
export eshelby_tensor
export green_gradient_iso, dipole_displacement_iso, green_operator_iso
export green_operator_aniso, green_operator, green_function_aniso
export gauss_legendre_nodes

# ── Two-inclusion interaction tensors (EIM / cluster model ingredient) ───────
export interaction_tensor, self_interaction_tensor
export lattice_interaction_tensor, periodic_images

# ── Elasticity ───────────────────────────────────────────────────────────────
export Ellipsoid, Spheroid
export EllipsoidShape, Spherical, Prolate, Oblate, Triaxial, Circular, Elliptic
export Cylinder, CylindricalShape, CircularCylindrical, EllipticCylindrical
export newton_potential_3d_cylinder
export tens_IA, tens_UA, tens_VA
export hill_tensor
export surface_stiffness, equivalent_particle
export k_mu, iso_stiffness, E_nu, iso_stiffness_E_nu
export hoenig_params, hoenig_stiffness

# ── Cracks ───────────────────────────────────────────────────────────────────
export CrackShape, Penny, EllipticShape, Ribbon
export EllipticCrack, RibbonCrack, PennyCrack
export ConductiveCrack, fracture_conductivity, with_conductivity
export crack_basis, aspect_ratio, semi_major, semi_minor, crack_normal
export cod_tensor, B_tensor
export cod_from_compliance, compliance_from_cod
export crack_density_factor
export sif, dif

# ── Custom (user-defined) inclusions ─────────────────────────────────────────
export CustomInclusion, CustomShape, check_inclusion_interface

# ── Finite-element inclusions (need a backend extension) ─────────────────────
export FECache, fe_assembly_count, fe_reset!
export FEBackend, AutoBackend, FerriteBackend, GridapBackend
export FEEllipticCrack, FEMeshOptions, fe_cod_breakdown, fe_mesh_report
export FEExcenteredSphere, FEAxiMeshOptions
export fe_axi_localization, fe_axi_breakdown, fe_axi_mesh_report

# ── Neural-network (surrogate) inclusions ────────────────────────────────────
export NeuralHillInclusion, NeuralLocalizationInclusion, NeuralShape
export NeuralSurrogate, Provenance, worst_error
export save_surrogate, load_surrogate, model_path, shipped_models
export HillISO, HillTI, HillOrtho, HillISO2, HillTI2
export StrainLocTI, StressLocTI
export DimensionlessHill, AffineHill
export SampleBox, Dataset, generate_dataset, fit_scaling
export TrainingOptions, train_surrogate, assemble_surrogate
export validate_surrogate, report_surrogate, component_labels

# ── Localization & contribution (Eshelby dilute, Kachanov-Sevostianov) ───────
export strain_strain_loc, stress_strain_loc, strain_stress_loc, stress_stress_loc
export gradient_gradient_loc, flux_gradient_loc, gradient_flux_loc, flux_flux_loc
export stiffness_contribution, conductivity_contribution, resistivity_contribution
export compliance_contribution
export is_homogeneous_inclusion
export delta_stiffness, delta_conductivity, delta_compliance, delta_resistivity

# ── LayeredSphere (Hervé-Zaoui / Hervé-Luanco / Gurtin-Murdoch / Kapitza) ────
export LayeredSphere, AbstractInterface, PerfectInterface
export SpringInterface, MembraneInterface
export KapitzaInterface, SurfaceConductiveInterface
export layer_count, layer_radius, layer_modulus, layer_interface,
    layer_volume_fraction, outer_radius
export layer_strain_average, sphere_strain_average, cumulative_strain_average

# ── LayeredSpheroid (Barthélémy-Bignonnet confocal spheroid, conduction) ─────
export LayeredSpheroid, layered_spheroid_from_fractions
export layer_q, layer_semiaxes, outer_semiaxes
export local_temperature, local_gradient, local_flux
export spheroid_state_sequence, spheroid_ba_ratios, get_layer

# ── Elliptic integrals (type-generic) ────────────────────────────────────────
export ell_K, ell_E, ell_F, ell_RF, ell_RD

# ── Schemes : RVE + amounts + distribution shape + symmetrize ───────────────
export AbstractAmount, VolumeFraction, CrackDensity, Remainder
export AbstractFractionClosure, StrictFractions, ComplementFraction, RescaledFractions
export remainder_phase_name, remainder_volume_fraction, phase_names
export AbstractDistributionShape, UniformDistribution
export AbstractSymmetrize, NoSymmetrize, IsoSymmetrize, TISymmetrize
export isotropify, transverse_isotropify
export ti_average_mandel66, iso_average_mandel66
export best_fit_ti, best_fit_iso, best_fit_ortho
export polar_orientation_bins
export Phase, RVE
export add_phase!
export inclusion_phase_names
export phase_property, phase_property_raw
export volume_fraction, crack_density
export phase_symmetrize
export validate_rve, promote_rve

# ── The homogenization cell contract + declarative multiscale chaining ──────
export AbstractHomogenizationCell, validate_cell
export Homogenized, NestedParameter, nested

# ── Laminates : the periodic multilayer cell ────────────────────────────────
export Laminate, Layer, add_layer!
export layer_names, layer_property, layer_property_raw
export layer_thickness
export laminate_period, laminate_basis, laminate_normal
export validate_laminate
# `layer_count`, `layer_interface` and `layer_volume_fraction` are NOT re-listed
# here: `Laminates` extends the `LayeredSpheres` generics rather than declaring
# rival bindings, so the exports above the LayeredSphere block already cover a
# laminate. The volume fraction of a layer is `layer_volume_fraction` and the
# interface on top of it is `layer_interface`; `layer_fraction` and
# `laminate_interface` were exported here for a while and are defined nowhere.
export AnisotropicSpringInterface, AnisotropicMembraneInterface
export AnisotropicSurfaceConductiveInterface
export laminate_hill
export layer_strain_localization, layer_stress_localization
export layer_gradient_localization, layer_flux_localization
export interface_jump
export ThicknessParameter, InterfaceParameter, thickness, interface_param

# ── Assemblies : the positional cell of the N-body schemes ──────────────────
export ParticleAssembly, Particle
export AbstractAssemblyBoundary, MixedBC, PeriodicBox
export add_matrix!, add_particle!
export matrix_property, matrix_volume_fraction
export particle_names, particle, particle_center, particle_geometry
export particle_property, particle_family, family_labels
export particle_volume, particle_volume_fraction, inclusion_volume_fraction
export assembly_volume, validate_assembly
export cubic_lattice, random_assembly, max_packing_fraction
export eim_bound_type, eim_polarizations, cluster_localizations
export CenterParameter, RadiusParameter, center_param, radius_param

# ── Schemes : scheme types + entry point ─────────────────────────────────────
export HomogenizationScheme
export Voigt, Reuss, Laminated, Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis
export SelfConsistent, AsymmetricSelfConsistent
export ClusterModel, EquivalentInclusion
export AndersonDefault, NewtonDefault, AutoNonlinear
export DifferentialTrajectory, Proportional, Sequential, CustomPath, Path, DifferentialScheme
export homogenize, differential_path
export crack_family_compliances, crack_family_residual
export fracture_permeability

# ── Schemes : sensitivities (autodiff via ForwardDiff strong dependency) ────
export AbstractParameter, AmountParameter, PropertyParameter,
    GeometryParameter, DistributionShapeParameter
export amount, property, geometry, shape_param
export get_param, set_param
export derivative, gradient, jacobian, sensitivity

# ── Poromechanics : Biot tensor / modulus, drained ↔ undrained ──────────────
export terzaghi_stress, biot_effective_stress
export biot_tensor, inverse_biot_modulus, biot_modulus, poroelastic_parameters
export undrained_stiffness, drained_stiffness, skempton_tensor
export pore_volume_fraction

# ── Viscoelasticity (ALV) ────────────────────────────────────────────────────
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
export cod_kernel_alv, compliance_contribution_alv, delta_compliance_alv
export stiffness_contribution_alv, stiffness_contribution_alv_at, delta_stiffness_alv

# ── Viscoelasticity (Laplace-Carson, non-ageing) ────────────────────────────
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

# ── Constitutive : MFH as a Gauss-point law inside an FE code ───────────────
export AbstractMFHMaterial, AbstractMaterialState, NoState
export MaterialResponse, stress, tangent, state
export initial_state, material_response
export gradient_names, flux_names, tangent_blocks, transport_property
export check_material_interface
export HomogenizedElastic, stiffness
export MicrocrackedMaterial, CrackedState, open_set, apertures
export FracturedPoroelasticRock, PoroFracturedState, conductivities, fluid_content
export MaterialCache, cached!, cache_stats, reset_cache!
export plane_strain_response
export to_tensors, from_tensors
export voigt_stress, voigt_strain, stress_from_voigt, strain_from_voigt

# ── MFH Studio launcher ──────────────────────────────────────────────────────
export mfhstudio

# ── Backwards-compat aliases ─────────────────────────────────────────────────
const HillAlgorithm = AbstractAlgorithm
const CrackAlgorithm = AbstractAlgorithm
export HillAlgorithm, CrackAlgorithm

end # module
