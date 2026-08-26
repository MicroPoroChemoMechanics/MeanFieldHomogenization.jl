"""
    MeanFieldHomogenization.Core

Shared abstractions and numerical kernels used throughout `MeanFieldHomogenization`.

Contents
--------
- `counters.jl`           : opt-in work counters for the benchmark harness
- `abstractions.jl`       : inclusion hierarchy (`AbstractInclusion` …)
- `cells.jl`              : the homogenization *cell* contract
                             (`AbstractHomogenizationCell`, `AbstractParameter`)
                             and the declarative multiscale seam
                             (`Homogenized`, `NestedParameter`)
- `traits.jl`             : algorithm and material-symmetry traits
- `bases.jl`              : helpers around `TensND` bases
- `tensor_helpers.jl`     : low-level utilities (`_δ`, `_C_array`, Voigt)
- `moduli.jl`             : modulus extractors for the common symmetry classes
- `newton_potential.jl`   : Newton potentials (2D / 3D)
- `green_kernel.jl`       : closed-form 3×3 inverse (`_inv3`)
- `laminate_algebra.jl`   : in-plane / out-of-plane Kelvin-Mandel block
                             algebra of a periodic laminate (`plane_pinv`,
                             `flat_hill`, `laminate_stiffness`, …)
- `green_residue.jl`      : Masson / Cauchy residue summation
- `green_helpers.jl`      : quadrature-agnostic Green-function helpers
- `green_dipole.jl`       : real-space Kelvin Green gradient and the dipole
                             far field of a polarized inclusion (isotropic)
- `green_operator.jl`     : real-space Green operator 𝔾⁰(x) — one derivative
                             beyond `green_dipole.jl`, the regular kernel of
                             the Lippmann-Schwinger equation (isotropic)
- `green_aniso.jl`        : its anisotropic counterpart — Barnett line integral
                             for elasticity, closed form for conduction — and
                             the `green_operator` dispatcher over both
- `quadrature.jl`         : DECUHR cubature backend seam
- `dispatch.jl`           : central `_resolve_algo` mechanism
- `dispatch_pair.jl`      : `_resolve_pair_algo` — its two-inclusion counterpart
"""
module Core

using LinearAlgebra
using StaticArrays
using TensND
using QuadGK
using ForwardDiff
using ..Elliptic
using Polynomials
using PolynomialRoots

include("counters.jl")
include("abstractions.jl")
include("cells.jl")
include("traits.jl")
include("bases.jl")
include("tensor_helpers.jl")
include("moduli.jl")
include("newton_potential.jl")
include("green_kernel.jl")
include("laminate_algebra.jl")
include("green_residue.jl")
include("green_helpers.jl")
include("green_dipole.jl")
include("green_operator.jl")
include("green_aniso.jl")
include("quadrature.jl")
include("dispatch.jl")
include("dispatch_pair.jl")

# Abstractions
export AbstractInclusion, AbstractEllipsoidalInclusion,
    AbstractCrack, AbstractLayeredInclusion, AbstractCustomInclusion

# The homogenization cell contract + declarative multiscale nesting
export AbstractHomogenizationCell, AbstractParameter
export homogenize, validate_cell, get_param, set_param
export cell_member_names, cell_container_property, cell_set_property
export Homogenized, NestedParameter, nested, resolve_property, has_nested_property
export dimension, element_type, inclusion_basis, shape_trait, shape_tensor
export eshelby_tensor
export is_homogeneous_inclusion

# Traits — algorithms
export AbstractAlgorithm, Analytical, Residue, DECUHR, NestedQuadGK,
    CylinderQuadrature, Multipole, Auto

# Traits — material symmetry
export MaterialSymmetry, IsotropicSym, TransverselyIsotropicSym,
    OrthotropicSym, GeneralAnisotropicSym, material_symmetry

# Modulus extractors (public — consumed by sub-modules and users)
export extract_iso_moduli, extract_ti_moduli, extract_iso_conductivity

# Exact rotation-group averages (public — used by Schemes, ALV and users)
# Exact SO(3) / azimuthal averages of minor-symmetric tensors, and their
# Kelvin-Mandel block forms. They used to live in `Core/rotational_average.jl`;
# being pure tensor algebra with nothing homogenization-specific about them
# they now belong to TensND (`src/tens_rotational_average.jl`), together with
# the non-major-symmetric Walpole read-off `ti8_params_from_KM` that the
# laminate localization needs. Re-exported here so that every call site of the
# form `Core.isotropify(...)` keeps working unchanged.
export isotropify, transverse_isotropify
export ti_average_mandel66, iso_average_mandel66
export mandel66_minor, array_from_mandel66
export ti8_params_from_KM, KM_from_ti8_params

# Newton potentials (public — used downstream and in tests)
export newton_potential_3d, newton_potential_2d, newton_potential_3d_cylinder

# Real-space Kelvin Green gradient / dipole far field (isotropic matrix)
export green_gradient_iso, dipole_displacement_iso
export green_operator_iso, green_operator_aniso, green_operator
export green_function_aniso, gauss_legendre_nodes

# Laminate block algebra (public — used by Laminates, the ALV twin and users)
export KM_IP, KM_OP
export plane_pinv, plane_pinv2, flat_hill, acoustic_tensor, compliance_op_block
export laminate_stiffness, laminate_conductivity
export laminate_strain_localization, laminate_stress_localization
export laminate_stress_strain_localization, laminate_strain_stress_localization
export laminate_gradient_localization, laminate_flux_localization
export laminate_flux_gradient_localization, laminate_gradient_flux_localization

# Localization & contribution (generics; methods added at top level and in Cracks)
export strain_strain_loc, stress_strain_loc, strain_stress_loc, stress_stress_loc
export gradient_gradient_loc, flux_gradient_loc, gradient_flux_loc, flux_flux_loc
export stiffness_contribution, conductivity_contribution, resistivity_contribution
export compliance_contribution
export delta_stiffness, delta_conductivity, delta_compliance, delta_resistivity

end # module
