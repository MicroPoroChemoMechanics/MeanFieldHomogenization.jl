"""
    MeanFieldHomogenization.LayeredSpheres

Isotropic `n`-layer spherical composite inclusion (core + concentric
shells) embedded in an infinite matrix, with perfect or imperfect
interfaces (spring / surface-elastic / Kapitza / surface-conductive).
Public entry points: [`LayeredSphere`](@ref), the bulk / shear
localization and layer-average utilities.
"""
module LayeredSpheres

using LinearAlgebra
using TensND

import ..Core
using ..Core
const MFH_Core = Core

import ..Elasticity   # for single-layer shear delegation to `Ellipsoid`

# `is_hard_numeric` — the numeric/symbolic split used throughout the package.
using ..Elliptic: is_hard_numeric

# Core-level generics we extend (strain_strain_loc etc. are declared in
# Core/abstractions.jl so every sub-module can attach methods uniformly).
import ..Core: strain_strain_loc, stress_strain_loc, strain_stress_loc,
    stress_stress_loc, gradient_gradient_loc, flux_gradient_loc,
    gradient_flux_loc, flux_flux_loc,
    stiffness_contribution, conductivity_contribution,
    resistivity_contribution, is_homogeneous_inclusion

include("interfaces.jl")
include("geometry.jl")
include("interface_transfer.jl") # interface jump matrices (bulk + shear)
include("bulk_recurrence.jl")    # bulk state-vector transfer + localization
include("shear_recurrence.jl")   # multi-layer shear stub
include("conductivity.jl")       # Y₁-harmonic conductivity state-vector
include("averages.jl")
include("localfields.jl")     # pointwise u, ε, σ, T, ∇T, flux reconstruction
include("scheme_integration.jl") # concentration tensors → mean-field schemes

# ── Localization / contribution overrides for LayeredSphere ─────────────────

"""
    strain_strain_loc(sphere::LayeredSphere, C₀::TensISO{4,3}; layer::Int) -> Tens{4,3}

Per-layer strain-strain localization tensor in an ISO `LayeredSphere`.
Returns the isotropic 4-tensor ``\\mathbb{A}_k = \\alpha_k\\,\\mathbb{J} + \\beta_k\\,\\mathbb{K}`` for the requested
layer.  `layer` must be in `1..N`.
"""
function strain_strain_loc(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3};
        layer::Int,
        kw...,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    α_k = _bulk_localization(sphere, κ₀, μ₀)[layer]
    β_k = _shear_localization(sphere, C₀)[layer]
    return TensISO{3}(α_k, β_k)
end

# =============================================================================
#  Contribution tensors (iso composite sphere in iso matrix)
# =============================================================================

"""
    stiffness_contribution(sphere, C₀) -> Tens{4,3}

Size-independent stiffness contribution tensor of the composite sphere
relative to the matrix `C₀`.  The dilute-scheme effective stiffness
is ``\\mathbb{C}^{\\mathrm{hom}} = \\mathbb{C}_0 + f\\,\\mathbb{N}`` where ``f`` is the volume fraction of the
composite sphere.  For ISO materials this reduces to two scalar
contributions (bulk + shear), ``f_k`` being the volume fraction of layer
``k`` within the sphere:

```math
\\mathbb{N} = 3N_{\\mathrm{bulk}}\\,\\mathbb{J} + 2N_{\\mathrm{shear}}\\,\\mathbb{K},
\\qquad
N_{\\mathrm{bulk}} = \\sum_k f_k\\,(k_k - k_0)\\,\\alpha_k,
\\qquad
N_{\\mathrm{shear}} = \\sum_k f_k\\,(\\mu_k - \\mu_0)\\,\\beta_k,
```

plus the Gurtin–Murdoch surface stress of any membrane interface.
"""
function Core.stiffness_contribution(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    κμ = _bulk_layer_moduli(sphere)
    α = _bulk_localization(sphere, κ₀, μ₀)
    β = _shear_localization(sphere, C₀)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    N_bulk = sum(f[k] * (κμ[k][1] - κ₀) * α[k] for k in 1:N)
    N_shear = sum(f[k] * (κμ[k][2] - μ₀) * β[k] for k in 1:N)
    # Gurtin–Murdoch surface stress of any dual (membrane) interface.
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀)
    return TensISO{3}(3 * N_bulk + a_surf, 2 * N_shear + b_surf)
end

# =============================================================================
#  Conductivity LayeredSphere overrides
# =============================================================================

"""
    gradient_gradient_loc(sphere::LayeredSphere, K₀; layer)

Per-layer gradient-gradient localization tensor for an isotropic
`LayeredSphere` embedded in an isotropic matrix of conductivity `K₀`.
Returns the scalar ``\\alpha_k`` packed as `TensISO{3}(α_k)` (isotropic
2-tensor), satisfying ``\\langle\\nabla T\\rangle_k = \\alpha_k\\,\\nabla T^{\\infty}``.
"""
function gradient_gradient_loc(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3};
        layer::Int,
        kw...,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    k₀ = _iso_scalar(K₀)
    α_k = _cond_localization(sphere, k₀)[layer]
    return TensISO{3}(α_k)
end

"""
    conductivity_contribution(sphere::LayeredSphere, K₀) -> Tens{2,3}

Size-independent conductivity contribution tensor of the composite
sphere:  ``\\boldsymbol{N} = \\sum_k f_k\\,(k_k - k_0)\\,\\alpha_k\\,\\boldsymbol{1}``, plus the surface-conduction
flux [`_cond_surface_flux`](@ref) of any dual (surface-conductive)
interface (Echoes' `DUALDISC`).
"""
function Core.conductivity_contribution(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    k_layers = _cond_layer_moduli(sphere)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    N_K = sum(f[k] * (k_layers[k] - k₀) * α[k] for k in 1:N) +
        _cond_surface_flux(sphere, k₀)
    return TensISO{3}(N_K)
end

# ── Exports ─────────────────────────────────────────────────────────────────
export LayeredSphere, AbstractInterface, PerfectInterface
export SpringInterface, MembraneInterface
export spring_compliances, spring_stiffnesses
export KapitzaInterface, SurfaceConductiveInterface
export layer_count, layer_radius, layer_modulus, layer_interface,
    layer_volume_fraction, outer_radius
export layer_strain_average, sphere_strain_average, cumulative_strain_average
export layer_stress_average, sphere_stress_average
export layer_gradient_average, sphere_gradient_average, layer_flux_average
# Pointwise fields.  `get_layer` and the `local_*` names below are declared
# HERE and extended by `LayeredSpheroids`, which imports them rather than
# declaring rival bindings — same arrangement as `layer_count` & friends.
export get_layer
export LayeredSphereFields, LayeredSphereTransportFields
export region_stiffness, shell_localization
export local_strain_strain_loc, local_stress_strain_loc,
    local_strain_stress_loc, local_stress_stress_loc
export local_strain, local_stress, local_displacement
export local_gradient_gradient_loc, local_flux_gradient_loc,
    local_gradient_flux_loc, local_flux_flux_loc
export local_temperature, local_gradient, local_flux
export layer_stiffness_average, layer_compliance_average,
    layer_conductivity_average, layer_resistivity_average

end # module
