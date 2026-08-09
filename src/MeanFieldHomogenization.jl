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
- `MeanFieldHomogenization.LayeredSpheres`   — `n`-layer composite spheres with five
  interface types, volume-average and pointwise localization.
- `MeanFieldHomogenization.LayeredSpheroids` — `n`-layer confocal spheroids in
  conduction, with imperfect interfaces.
- `MeanFieldHomogenization.Schemes`      — RVEs, amounts, symmetrization and the
  homogenization schemes themselves (dilute, Mori–Tanaka, self-consistent,
  PCW, Maxwell, differential).
- `MeanFieldHomogenization.Viscoelasticity`  — ageing linear viscoelasticity through
  Volterra operators, available to every scheme.
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
include("LayeredSpheres/LayeredSpheres.jl")
include("LayeredSpheroids/LayeredSpheroids.jl")
include("Schemes/Schemes.jl")
# `Laminates` sits between `Schemes` and `Viscoelasticity`: it extends
# `_evaluate` and uses `HomogenizationScheme`/`Laminated`/`Voigt`/`Reuss` from
# the former, and the latter needs `Laminate` for the ageing-viscoelastic
# multilayer.
include("Laminates/Laminates.jl")
include("Viscoelasticity/Viscoelasticity.jl")

using .Elliptic
using .Core
using .Elasticity
using .Cracks
using .Conductivity
using .LayeredSpheres
using .LayeredSpheroids
using .Schemes
using .Laminates
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

using .CustomInclusions
using .FiniteElements
using .NeuralInclusions

# ─── MFH Studio launcher ─────────────────────────────────────────────────────
include("Studio.jl")

# ── Abstractions ─────────────────────────────────────────────────────────────
export AbstractInclusion, AbstractEllipsoidalInclusion, AbstractCrack
export AbstractLayeredInclusion, AbstractCustomInclusion
export AbstractAlgorithm, Analytical, Residue, DECUHR, NestedQuadGK,
    CylinderQuadrature, Auto
export MaterialSymmetry, IsotropicSym, TransverselyIsotropicSym,
    OrthotropicSym, GeneralAnisotropicSym
export material_symmetry, dimension, inclusion_basis, shape_trait, shape_tensor
export eshelby_tensor
export green_gradient_iso, dipole_displacement_iso

# ── Elasticity ───────────────────────────────────────────────────────────────
export Ellipsoid, Spheroid
export EllipsoidShape, Spherical, Prolate, Oblate, Triaxial, Circular, Elliptic
export Cylinder, CylindricalShape, CircularCylindrical, EllipticCylindrical
export newton_potential_3d_cylinder
export tens_IA, tens_UA, tens_VA
export hill_tensor
export k_mu, iso_stiffness, E_nu, iso_stiffness_E_nu
export hoenig_params, hoenig_stiffness

# ── Cracks ───────────────────────────────────────────────────────────────────
export CrackShape, Penny, EllipticShape, Ribbon
export EllipticCrack, RibbonCrack, PennyCrack
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
export AbstractAmount, VolumeFraction, CrackDensity
export AbstractDistributionShape, UniformDistribution
export AbstractSymmetrize, NoSymmetrize, IsoSymmetrize, TISymmetrize
export isotropify, transverse_isotropify
export ti_average_mandel66, iso_average_mandel66
export best_fit_ti, best_fit_iso, best_fit_ortho
export polar_orientation_bins
export Phase, RVE
export add_matrix!, add_phase!
export matrix_phase, inclusion_phase_names
export phase_property, phase_property_raw, matrix_property
export volume_fraction, crack_density, matrix_volume_fraction
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

# ── Schemes : scheme types + entry point ─────────────────────────────────────
export HomogenizationScheme
export Voigt, Reuss, Laminated, Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis
export SelfConsistent, AsymmetricSelfConsistent
export AndersonDefault, NewtonDefault, AutoNonlinear
export DifferentialTrajectory, Proportional, Sequential, CustomPath, Path, DifferentialScheme
export homogenize, differential_path

# ── Schemes : sensitivities (autodiff via ForwardDiff strong dependency) ────
export AbstractParameter, AmountParameter, PropertyParameter,
    GeometryParameter, DistributionShapeParameter
export amount, property, geometry, shape_param
export get_param, set_param
export derivative, gradient, jacobian, sensitivity

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

# ── MFH Studio launcher ──────────────────────────────────────────────────────
export mfhstudio

# ── Backwards-compat aliases ─────────────────────────────────────────────────
const HillAlgorithm = AbstractAlgorithm
const CrackAlgorithm = AbstractAlgorithm
export HillAlgorithm, CrackAlgorithm

end # module
