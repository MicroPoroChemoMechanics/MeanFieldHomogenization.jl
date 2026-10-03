# =============================================================================
#  scheme_integration.jl — plug a `LayeredSphere` into the mean-field schemes.
#
#  A composite sphere has NO Hill tensor: it is not an ellipsoidal
#  inhomogeneity with a uniform eigenstrain.  What it does have — and what the
#  schemes actually need — is a **concentration (localization) tensor**, which
#  the layered recurrences already provide per layer:
#
#      <ε>_k = α_k · ε∞_sph + β_k · ε∞_dev
#
#  The generic `strain_strain_loc(::AbstractInclusion, …)` in `localization.jl`
#  builds `A` from `hill_tensor`, so without the specializations below a
#  `LayeredSphere` phase would fall into it and fail.  The methods here
#  short-circuit that path, exactly as the conductivity side already does.
#
#  Two quantities are needed by the schemes:
#
#    * the whole-inclusion concentration tensor, a volume average over layers
#          A_Ω = (Σ_k f_k α_k) 𝕁 + (Σ_k f_k β_k) 𝕂 ;
#    * the stiffness contribution, which must be assembled **layer by layer**
#          N = Σ_k f_k (C_k − C₀) : A_k ,
#      and is *not* `(C₁ − C₀) : A_Ω` — the inclusion is heterogeneous, so no
#      single `C₁` represents it.
# =============================================================================

"""
    is_homogeneous_inclusion(::LayeredSphere) -> false

A composite sphere has no single representative property: its average stress
must be summed over layers. See [`stress_strain_loc`](@ref).
"""
Core.is_homogeneous_inclusion(::LayeredSphere) = false

"""
    _layer_iso_pairs(sphere) -> NTuple{N, Tuple}

Per-layer ``(3k_k, 2\\mu_k)`` pairs of the isotropic stiffnesses, i.e. ``\\mathbb{C}_k = 3k_k\\,\\mathbb{J} + 2\\mu_k\\,\\mathbb{K}``.
"""
@inline function _layer_iso_pairs(sphere::LayeredSphere{T, N}) where {T, N}
    return ntuple(k -> TensND.get_data(layer_modulus(sphere, k)), Val(N))
end

@inline function _layer_fractions(sphere::LayeredSphere{T, N}) where {T, N}
    return ntuple(k -> layer_volume_fraction(sphere, k), Val(N))
end

"""
    _layer_localizations(sphere, C₀) -> (α, β, f)

Per-layer bulk (``\\alpha_k``) and deviatoric (``\\beta_k``) localization scalars together
with the layer volume fractions.
"""
function _layer_localizations(
        sphere::LayeredSphere{T, N},
        C₀::TensND.TensISO{4, 3},
    ) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    α = _bulk_localization(sphere, κ₀, μ₀)
    β = _shear_localization(sphere, C₀)
    return α, β, _layer_fractions(sphere)
end

"""
    _membrane_surface_stress(sphere, C₀) -> (a_surf, b_surf)

Contribution of Gurtin–Murdoch surface stress on the dual
([`MembraneInterface`](@ref)) interfaces to the volume-averaged stress of
the composite sphere, per unit remote strain, split into bulk (``\\mathbb{J}``) and
shear (``\\mathbb{K}``) scalars.  From the average-stress theorem with a coherent
surface, ``\\langle\\boldsymbol{\\sigma}\\rangle_\\Omega = \\sum_k f_k\\,\\mathbb{C}_k:\\mathbb{A}_k + \\frac{1}{V}\\sum_\\Gamma\\oint_\\Gamma\\boldsymbol{\\sigma}^{\\mathrm s}\\,\\mathrm{d}S``; the surface
integrals give, for each membrane interface (with ``\\kappa^{\\mathrm s} = \\lambda^{\\mathrm s} + \\mu^{\\mathrm s}``, ``r`` the interface radius, ``R`` the
outer radius):

```math
\\begin{aligned}
a_{\\mathrm{surf}} &= 4\\kappa^{\\mathrm s}\\,u_r(r)\\,r/R^3,\\\\
b_{\\mathrm{surf}} &= \\tfrac{1}{3}\\,(-6\\kappa^{\\mathrm s} U + 18\\kappa^{\\mathrm s} W + 36\\mu^{\\mathrm s} W)\\,r/(5R^3),
\\end{aligned}
```

the factor ``1/3`` being the ``\\mathbb{K}``-projection of the
``\\sigma_{zz} - \\sigma_{xx}`` surface integral.
``u_r(r)`` is the bulk radial amplitude (normalized by the far-field ``A_\\infty``);
``U(r)``, ``W(r)`` are the deviatoric displacement amplitudes at the interface
(already normalized to a unit remote deviatoric far field).
"""
function _membrane_surface_stress(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}
    ) where {T, N}
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    radii = sphere.radii
    R³ = radii[N]^3

    # Any membrane interface present?  (cheap short-circuit)
    has_membrane = any(k -> layer_interface(sphere, k) isa MembraneInterface, 1:N)
    has_membrane || return (zero(T), zero(T))

    # Bulk amplitudes u_r(r_k), normalized by the far-field A∞.
    inside_b, s_b = _bulk_state_seq(sphere, κ₀, μ₀)
    A_inf, _ = _bulk_extract_AB(radii[N], κ₀, μ₀, s_b[1], s_b[2])
    # Deviatoric state amplitudes (U, W)(r_k), already at unit remote far field.
    states_s, _ = _shear_state_seq(sphere, C₀)

    a_surf = zero(promote_type(T, typeof(κ₀), typeof(A_inf)))
    b_surf = zero(a_surf)
    for k in 1:N
        intf = layer_interface(sphere, k)
        intf isa MembraneInterface || continue
        κs = intf.κs; μs = intf.μs
        r = radii[k]
        u_r = inside_b[k][1] / A_inf
        a_surf += 4 * κs * u_r * r / R³
        U = states_s[k][1]; W = states_s[k][2]
        # (σzz−σxx)/V = C·r/R³;  𝕂-amplitude b = (σzz−σxx)/3.
        C = (-6 * κs * U + 18 * κs * W + 36 * μs * W) / 5
        b_surf += (C * r / R³) / 3
    end
    return a_surf, b_surf
end

"""
    strain_strain_loc(sphere::LayeredSphere, C₁, C₀; kw...) -> TensISO{4,3}

Whole-inclusion **strain concentration tensor** of a composite sphere embedded
in the isotropic reference `C₀`:

```math
\\mathbb{A}_\\Omega = \\Big(\\sum_k f_k\\,\\alpha_k\\Big)\\,\\mathbb{J} + \\Big(\\sum_k f_k\\,\\beta_k\\Big)\\,\\mathbb{K},
\\qquad
\\langle\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\mathbb{A}_\\Omega:\\boldsymbol{\\varepsilon}^{\\infty}.
```

`C₁` is accepted for signature compatibility with the generic
`strain_strain_loc(::AbstractInclusion, C₁, C₀)` used by the scheme dispatch,
but is **ignored**: the moduli of a composite sphere live in its layers
(`layer_modulus`), not in a single phase tensor.

For a single layer this reduces exactly to the Eshelby result for a sphere.
"""
function strain_strain_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    return TensISO{3}(sum(f[k] * α[k] for k in 1:N), sum(f[k] * β[k] for k in 1:N))
end

"""
    stiffness_contribution(sphere::LayeredSphere, C₁, C₀; kw...) -> TensISO{4,3}

Size-independent **stiffness contribution tensor** of a composite sphere,

```math
\\mathbb{N} = \\sum_k f_k\\,(\\mathbb{C}_k - \\mathbb{C}_0):\\mathbb{A}_k,
```

plus the Gurtin–Murdoch surface stress of any membrane interface
(`_membrane_surface_stress`).

Assembled layer by layer: a composite sphere is heterogeneous, so the usual
``(\\mathbb{C}_1 - \\mathbb{C}_0):\\mathbb{A}`` of a homogeneous inhomogeneity does not apply. `C₁` is ignored,
as in [`strain_strain_loc`](@ref).
"""
function stiffness_contribution(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    C_k = _layer_iso_pairs(sphere)
    α₀, β₀ = TensND.get_data(C₀)
    a = sum(f[k] * (C_k[k][1] - α₀) * α[k] for k in 1:N)
    b = sum(f[k] * (C_k[k][2] - β₀) * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀)
    return TensISO{3}(a + a_surf, b + b_surf)
end

"""
    stress_strain_loc(sphere::LayeredSphere, C₁, C₀; kw...) -> TensISO{4,3}

Whole-inclusion **average stress** per unit remote strain,

```math
\\langle\\mathbb{C}:\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\Big(\\sum_k f_k\\,\\mathbb{C}_k:\\mathbb{A}_k\\Big):\\boldsymbol{\\varepsilon}^{\\infty},
```

assembled layer by layer, plus the Gurtin–Murdoch surface stress of any
membrane interface (`_membrane_surface_stress`). This is what the self-consistent and Mori-Tanaka
kernels need; it is *not* ``\\mathbb{C}_1:\\mathbb{A}_\\Omega``. `C₁` is ignored (see
[`strain_strain_loc`](@ref)).

Consistency: `stiffness_contribution = stress_strain_loc - C₀ : strain_strain_loc`.
"""
function stress_strain_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    C_k = _layer_iso_pairs(sphere)
    a = sum(f[k] * C_k[k][1] * α[k] for k in 1:N)
    b = sum(f[k] * C_k[k][2] * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀)
    return TensISO{3}(a + a_surf, b + b_surf)
end

"""
    flux_gradient_loc(sphere::LayeredSphere, K₁, K₀; kw...) -> TensISO{2,3}

Conductivity counterpart of [`stress_strain_loc`](@ref):
``\\langle k\\,\\nabla T\\rangle_\\Omega = \\big(\\sum_k f_k\\,k_k\\,\\alpha_k\\big)\\,\\nabla T^{\\infty}``, plus the surface-conduction flux
[`_cond_surface_flux`](@ref) of any dual (surface-conductive) interface.
`K₁` is ignored.
"""
function flux_gradient_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    k_layers = _cond_layer_moduli(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(
        sum(f[k] * k_layers[k] * α[k] for k in 1:N) + _cond_surface_flux(sphere, k₀)
    )
end

"""
    gradient_gradient_loc(sphere::LayeredSphere, K₁, K₀; kw...) -> TensISO{2,3}

Whole-inclusion **gradient concentration tensor**
``\\alpha_\\Omega = \\sum_k f_k\\,\\alpha_k``, the conductivity counterpart of
[`strain_strain_loc`](@ref). `K₁` is ignored (see there).

The per-layer form is available as
`gradient_gradient_loc(sphere, K₀; layer = k)`.
"""
function gradient_gradient_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    α = _cond_localization(sphere, _iso_scalar(K₀))
    f = _layer_fractions(sphere)
    return TensISO{3}(sum(f[k] * α[k] for k in 1:N))
end

"""
    conductivity_contribution(sphere::LayeredSphere, K₁, K₀; kw...) -> TensISO{2,3}

Three-argument form matching the scheme dispatch; `K₁` is ignored and the
computation is delegated to the two-argument method.
"""
function Core.conductivity_contribution(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    return Core.conductivity_contribution(sphere, K₀; kw...)
end

"""
    layer_stiffness_average(sphere) -> TensISO{4,3}

Voigt (volume) average of the layer stiffnesses, ``\\sum_k f_k\\,\\mathbb{C}_k``. This is what
the Voigt bound needs for a composite sphere: the declared phase property does
not represent it.
"""
function layer_stiffness_average(sphere::LayeredSphere{T, N}) where {T, N}
    C_k = _layer_iso_pairs(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(
        sum(f[k] * C_k[k][1] for k in 1:N),
        sum(f[k] * C_k[k][2] for k in 1:N),
    )
end

"""
    layer_compliance_average(sphere) -> TensISO{4,3}

Reuss (volume) average of the layer compliances, ``\\sum_k f_k\\,\\mathbb{C}_k^{-1}``.
"""
function layer_compliance_average(sphere::LayeredSphere{T, N}) where {T, N}
    C_k = _layer_iso_pairs(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(
        sum(f[k] / C_k[k][1] for k in 1:N),
        sum(f[k] / C_k[k][2] for k in 1:N),
    )
end

"""
    layer_conductivity_average(sphere) -> TensISO{2,3}

Voigt average of the layer conductivities, ``\\sum_k f_k\\,k_k``.
"""
function layer_conductivity_average(sphere::LayeredSphere{T, N}) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(sum(f[k] * k_layers[k] for k in 1:N))
end

"""
    layer_resistivity_average(sphere) -> TensISO{2,3}

Reuss average of the layer resistivities, ``\\sum_k f_k/k_k``.
"""
function layer_resistivity_average(sphere::LayeredSphere{T, N}) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(sum(f[k] / k_layers[k] for k in 1:N))
end

# =============================================================================
#  Bundled localization + contribution for a composite sphere
#
#  A `LayeredSphere` is `is_homogeneous_inclusion == false`, so a scheme asking
#  for both a concentration tensor and a contribution tensor runs the
#  Hervé-Zaoui bulk + shear recurrences twice (once per function), on top of
#  the `_membrane_surface_stress` call that re-runs `_bulk_state_seq` /
#  `_shear_state_seq` internally.
#
#  The bundles below share `_layer_localizations` (and `_membrane_surface_stress`
#  where both members need it) while keeping every `sum(...)` expression
#  verbatim.  In particular `N` is NOT derived from the documented identity
#  `N = B − C₀:A` (see `stress_strain_loc` above): that identity is exact in
#  real arithmetic, but the surface terms enter both members additively and the
#  subtraction would reassociate the floating-point operations.  Keeping the
#  original expressions is what makes these bundles bitwise identical.
# =============================================================================

function Core.loc_and_stiffness(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N), sum(f[k] * β[k] for k in 1:N))
    C_k = _layer_iso_pairs(sphere)
    α₀, β₀ = TensND.get_data(C₀)
    a = sum(f[k] * (C_k[k][1] - α₀) * α[k] for k in 1:N)
    b = sum(f[k] * (C_k[k][2] - β₀) * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀)
    return (A, TensISO{3}(a + a_surf, b + b_surf))
end

function Core.loc_and_stress_average(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N), sum(f[k] * β[k] for k in 1:N))
    C_k = _layer_iso_pairs(sphere)
    a = sum(f[k] * C_k[k][1] * α[k] for k in 1:N)
    b = sum(f[k] * C_k[k][2] * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀)
    return (A, TensISO{3}(a + a_surf, b + b_surf))
end

function Core.loc_and_stiffness(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    f = _layer_fractions(sphere)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N))
    k_layers = _cond_layer_moduli(sphere)
    N_K = sum(f[k] * (k_layers[k] - k₀) * α[k] for k in 1:N) +
        _cond_surface_flux(sphere, k₀)
    return (A, TensISO{3}(N_K))
end

function Core.loc_and_stress_average(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    f = _layer_fractions(sphere)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N))
    k_layers = _cond_layer_moduli(sphere)
    B = TensISO{3}(
        sum(f[k] * k_layers[k] * α[k] for k in 1:N) + _cond_surface_flux(sphere, k₀)
    )
    return (A, B)
end
