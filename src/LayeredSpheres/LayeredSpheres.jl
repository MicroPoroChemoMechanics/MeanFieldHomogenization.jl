"""
    MeanFieldHomogenization.LayeredSpheres

Isotropic `n`-layer spherical composite inclusion (core + concentric
shells) embedded in an infinite matrix, with perfect or imperfect
interfaces (spring / surface-elastic / Kapitza / surface-conductive).
Public entry points: [`LayeredSphere`](@ref), the bulk / shear
concentration and layer-average utilities.
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

# Core-level generics we extend (gradient_gradient_concentration etc. are declared in
# Core/abstractions.jl so every sub-module can attach methods uniformly).
import ..Core: gradient_gradient_concentration, flux_gradient_concentration,
    gradient_flux_concentration, flux_flux_concentration, property_contribution,
    inverse_property_contribution, is_homogeneous_inclusion

include("interfaces.jl")
include("geometry.jl")
include("interface_transfer.jl") # interface jump matrices (bulk + shear)
include("bulk_recurrence.jl")    # bulk state-vector transfer + concentration
include("shear_recurrence.jl")   # multi-layer shear stub
include("conductivity.jl")       # Y₁-harmonic conductivity state-vector
include("averages.jl")
include("localfields.jl")     # pointwise u, ε, σ, T, ∇T, flux reconstruction
include("scheme_integration.jl") # concentration tensors → mean-field schemes
include("order_aliases.jl")       # names of elasticity, restricted to order 4

# ── Concentration / contribution overrides for LayeredSphere ─────────────────

"""
    gradient_gradient_concentration(sphere::LayeredSphere, C₀::TensISO{4,3}; layer::Int,
                      external = false, internal = false) -> Tens{4,3}

Per-layer strain-strain concentration tensor in an ISO `LayeredSphere`.
Returns the isotropic 4-tensor ``\\mathbb{A}_k = \\alpha_k\\,\\mathbb{J} + \\beta_k\\,\\mathbb{K}`` for the requested
layer.  `layer` must be in `1..N`.

By default the average is over the material of the layer. `external = true`
adds the displacement jump across its outer boundary ``r_k``, `internal = true`
the jump across its inner boundary ``r_{k-1}`` (both zero unless that interface is
a [`SpringInterface`](@ref)), as Echoes' `layer_eE(k - 1, external, internal)`
does — whose default is `external = True`. With `external = true` on every
layer the shells partition the sphere,
``\\sum_k f_k\\,\\mathbb{A}_k = \\mathbb{A}_\\Omega``, the whole-sphere tensor
[`gradient_gradient_concentration`](@ref)`(sphere, C₀, C₀)`.
"""
function gradient_gradient_concentration(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3};
        layer::Int,
        external::Bool = false,
        internal::Bool = false,
        kw...,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    α_k = _bulk_concentration(sphere, κ₀, μ₀)[layer]
    β_k = _shear_concentration(sphere, C₀)[layer]
    Δα, Δβ = _layer_strain_jump_terms(sphere, C₀, layer; external, internal)
    return TensISO{3}(α_k + Δα, β_k + Δβ)
end

# =============================================================================
#  Contribution tensors (iso composite sphere in iso matrix)
# =============================================================================

"""
    property_contribution(sphere, C₀; external = true) -> Tens{4,3}

Size-independent stiffness contribution tensor of the composite sphere
relative to the matrix `C₀`.  The dilute-scheme effective stiffness
is ``\\mathbb{C}^{\\mathrm{hom}} = \\mathbb{C}_0 + f\\,\\mathbb{N}`` where ``f`` is the volume fraction of the
composite sphere.  For ISO materials this reduces to two scalar
contributions (bulk + shear), ``f_k`` being the volume fraction of layer
``k`` within the sphere:

```math
\\begin{aligned}
\\mathbb{N} &= 3N_{\\mathrm{bulk}}\\,\\mathbb{J} + 2N_{\\mathrm{shear}}\\,\\mathbb{K},
\\\\
N_{\\mathrm{bulk}} &= \\sum_k f_k\\,(k_k - k_0)\\,\\alpha_k - k_0\\,\\Delta\\alpha,
\\\\
N_{\\mathrm{shear}} &= \\sum_k f_k\\,(\\mu_k - \\mu_0)\\,\\beta_k - \\mu_0\\,\\Delta\\beta,
\\end{aligned}
```

plus the Gurtin–Murdoch surface stress of any membrane interface.
``\\Delta\\alpha``, ``\\Delta\\beta`` are the displacement jumps of the spring interfaces, and
`external` decides whether the outer interface belongs to the sphere, as in
[`gradient_gradient_concentration`](@ref). For an incompressible layer (``k_k = \\infty``,
``\\alpha_k = 0``) the product ``k_k\\,\\alpha_k`` is its finite mean pressure, read from the
traction. The same tensor as the three-argument form
`property_contribution(sphere, C₀, C₀)`.
"""
function Core.property_contribution(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    return property_contribution(sphere, C₀, C₀; kw...)
end

# =============================================================================
#  Conductivity LayeredSphere overrides
# =============================================================================

"""
    gradient_gradient_concentration(sphere::LayeredSphere, K₀; layer, external = false, internal = false)

Per-layer gradient-gradient concentration tensor for an isotropic
`LayeredSphere` embedded in an isotropic matrix of conductivity `K₀`.
Returns the scalar ``\\alpha_k`` packed as `TensISO{3}(α_k)` (isotropic
2-tensor), satisfying ``\\langle\\nabla T\\rangle_k = \\alpha_k\\,\\nabla T^{\\infty}``.

`external` and `internal` add the temperature jumps across the outer and inner
boundaries of the layer (nonzero across a [`KapitzaInterface`](@ref) only), as
the elastic method `(sphere, C₀; layer)` does.
"""
function gradient_gradient_concentration(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3};
        layer::Int,
        external::Bool = false,
        internal::Bool = false,
        kw...,
    ) where {T, N}
    1 ≤ layer ≤ N || throw(BoundsError(sphere, layer))
    k₀ = _iso_scalar(K₀)
    α_k = _cond_concentration(sphere, k₀)[layer]
    return TensISO{3}(α_k + _layer_gradient_jump_term(sphere, k₀, layer; external, internal))
end

"""
    property_contribution(sphere::LayeredSphere, K₀; external = true) -> Tens{2,3}

Size-independent conductivity contribution tensor of the composite
sphere:  ``\\boldsymbol{N} = \\big(\\sum_k f_k\\,(k_k - k_0)\\,\\alpha_k - k_0\\,\\Delta\\alpha\\big)\\,\\boldsymbol{1}``, plus the
surface-conduction flux [`_cond_surface_flux`](@ref) of any dual
(surface-conductive) interface (Echoes' `DUALDISC`). ``\\Delta\\alpha`` is the
temperature jump of the Kapitza interfaces, and `external` decides whether the
outer interface belongs to the sphere (see [`gradient_gradient_concentration`](@ref)).
"""
function Core.property_contribution(
        sphere::LayeredSphere{T, N},
        K₀::TensND.TensISO{2, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_concentration(sphere, k₀)
    k_layers = _cond_layer_moduli(sphere)
    f = ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
    N_K = sum(f[k] * (k_layers[k] - k₀) * α[k] for k in 1:N) +
        _cond_surface_flux(sphere, k₀; external) - k₀ * _gradient_jump_term(sphere, k₀; external)
    return TensISO{3}(N_K)
end

# ── Exports ─────────────────────────────────────────────────────────────────
export LayeredSphere, AbstractInterface, PerfectInterface
export SpringInterface, MembraneInterface
export spring_compliances, spring_stiffnesses
export KapitzaInterface, SurfaceConductiveInterface
export layer_count, layer_radius, layer_modulus, layer_interface,
    layer_volume_fraction, outer_radius
export layer_gradient_average, sphere_gradient_average, cumulative_strain_average
export layer_flux_average, sphere_flux_average
export layer_strain_average, sphere_strain_average
export layer_stress_average, sphere_stress_average
# Pointwise fields.  `get_layer` and the `local_*` names below are declared
# HERE and extended by `LayeredSpheroids`, which imports them rather than
# declaring rival bindings — same arrangement as `layer_count` & friends.
export get_layer
export LayeredSphereFields, LayeredSphereTransportFields
export region_stiffness, shell_concentration
export local_gradient_gradient_concentration, local_flux_gradient_concentration,
    local_gradient_flux_concentration, local_flux_flux_concentration
export local_gradient, local_flux, local_displacement, local_temperature
export local_strain_strain_concentration, local_stress_strain_concentration,
    local_strain_stress_concentration, local_stress_stress_concentration
export local_strain, local_stress
export layer_property_average, layer_inverse_property_average
export layer_stiffness_average, layer_compliance_average,
    layer_conductivity_average, layer_resistivity_average

end # module
