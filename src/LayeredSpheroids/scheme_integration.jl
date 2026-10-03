# =============================================================================
#  scheme_integration.jl — plug a `LayeredSpheroid` into the mean-field
#  schemes, in conduction and in elasticity. The elastic entries live at the
#  bottom of this file and go through `elastic_cases.jl`, which needs all three
#  elementary problems: case I alone would give only the axisymmetric block of
#  the concentration tensor.
#
#  Exactly as for `LayeredSphere` (see `LayeredSpheres/scheme_integration.jl`),
#  a composite spheroid has NO Hill tensor: what the schemes need is its
#  whole-inclusion volume-averaged concentration tensors
#
#      ⟨∇T⟩_Ω = A_Ω · ∇T∞ ,       ⟨K∇T⟩_Ω = B_Ω · ∇T∞ ,
#
#  obtained here from the confocal transfer-matrix recurrence of
#  `conductivity.jl` (`spheroid_gradient_gradient` / `spheroid_flux_gradient`)
#  rather than a layer-by-layer sum — the imperfect interfaces couple
#  harmonic degrees, so there is no simple per-layer partition of the
#  average as there is for the sphere. `A_Ω`, `B_Ω` are transversely
#  isotropic about the spheroid's own axis (`TensND.TensTI{2,3}`),
#  `axis`-aligned unless the geometry was built with a tilted `axis`.
#
#  The declared phase property (`K₁` in the scheme dispatch) is
#  accepted for signature compatibility but IGNORED: the layer moduli
#  live in `sphere.moduli`, exactly as for `LayeredSphere`.
#
#  Resistivity-based (compliance-formulation) scheme kernels are, as
#  for `LayeredSphere`, not supported (no `resistivity_contribution`
#  override) — only Voigt/Reuss (via `layer_conductivity_average` /
#  `layer_resistivity_average`, volume averages of the LOCAL isotropic
#  layer conductivities, direction-independent) and the dilute/MT/SC/
#  Maxwell/differential kernels that route through
#  `gradient_gradient_loc` / `flux_gradient_loc` / `conductivity_contribution`.
# =============================================================================

"""
    is_homogeneous_inclusion(::LayeredSpheroid) -> false

A composite spheroid has no single representative conductivity: its
average flux must be obtained from the transfer-matrix recurrence, not
from ``(\\boldsymbol{K}_1 - \\boldsymbol{K}_0)\\cdot\\boldsymbol{A}``. See [`flux_gradient_loc`](@ref).
"""
Core.is_homogeneous_inclusion(::LayeredSpheroid) = false

"""
    _spheroid_concentration(s, k₀) -> (αt, αa, βt, βa)

The four real scalars parametrizing the transversely isotropic
gradient (``\\alpha``) and flux (``\\beta``) whole-inclusion concentration tensors,
``t``/``a`` for transverse/axial, computed once from the shared
``(b/a)`` ratios ([`spheroid_ba_ratios`](@ref)).
"""
function _spheroid_concentration(s::LayeredSpheroid{T, N}, k₀) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    qN = s.q[N]
    ba_a, ba_t = spheroid_ba_ratios(s, k₀)
    αa = real(1 + ba_a * _shape_Ta(qN))
    αt = real(1 + ba_t * _shape_Tt(qN))
    βa = real(1 + ba_a * _shape_Ua(qN))
    βt = real(1 + ba_t * _shape_Ut(qN))
    return αt, αa, βt, βa
end

"""
    gradient_gradient_loc(s::LayeredSpheroid, K₁, K₀; kw...) -> TensTI{2,3}

Whole-inclusion **gradient concentration tensor**
``\\boldsymbol{A}_\\Omega = \\alpha_t\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n}) + \\alpha_a\\,\\underline{n}\\otimes\\underline{n}`` (``\\underline{n}`` the spheroid's axis), such that
``\\langle\\nabla T\\rangle_\\Omega = \\boldsymbol{A}_\\Omega\\cdot\\nabla T^{\\infty}``. `K₁` is ignored (see the module docstring).
"""
function gradient_gradient_loc(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = MFH_Core.extract_iso_conductivity(K₀)
    αt, αa, _, _ = _spheroid_concentration(s, k₀)
    return TensND.TensTI{2}(αt, αa, s.axis)
end

"""
    flux_gradient_loc(s::LayeredSpheroid, K₁, K₀; kw...) -> TensTI{2,3}

Whole-inclusion **average flux** per unit remote gradient,
``\\boldsymbol{B}_\\Omega = k_0\\,\\bigl(\\beta_t\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n}) + \\beta_a\\,\\underline{n}\\otimes\\underline{n}\\bigr)``,
``\\langle\\boldsymbol{K}\\cdot\\nabla T\\rangle_\\Omega = \\boldsymbol{B}_\\Omega\\cdot\\nabla T^{\\infty}``, `K₁` ignored.

Consistency: `conductivity_contribution = flux_gradient_loc - K₀ ⋅
gradient_gradient_loc`, verified in the test suite
(``\\boldsymbol{N}_K = \\boldsymbol{B}_\\Omega - \\boldsymbol{K}_0\\cdot\\boldsymbol{A}_\\Omega``,
the same invariant as `LayeredSphere`'s).
"""
function flux_gradient_loc(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = MFH_Core.extract_iso_conductivity(K₀)
    _, _, βt, βa = _spheroid_concentration(s, k₀)
    return k₀ * TensND.TensTI{2}(βt, βa, s.axis)
end

"""
    conductivity_contribution(s::LayeredSpheroid, K₁, K₀; kw...) -> TensTI{2,3}

Size-independent **conductivity contribution tensor**
``\\boldsymbol{N}_K = \\boldsymbol{B}_\\Omega - \\boldsymbol{K}_0\\cdot\\boldsymbol{A}_\\Omega``, i.e.
``\\langle\\boldsymbol{K}\\cdot\\nabla T\\rangle_\\Omega - \\boldsymbol{K}_0\\cdot\\langle\\nabla T\\rangle_\\Omega`` per unit remote
gradient, assembled from the whole-inclusion
average (the spheroid is heterogeneous, so ``(\\boldsymbol{K}_1 - \\boldsymbol{K}_0)\\cdot\\boldsymbol{A}`` does not
apply). `K₁` is ignored (see [`gradient_gradient_loc`](@ref)).
"""
function Core.conductivity_contribution(
        s::LayeredSpheroid{T, N},
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    B = flux_gradient_loc(s, K₁, K₀; kw...)
    A = gradient_gradient_loc(s, K₁, K₀; kw...)
    return B - K₀ ⋅ A
end

"""
    layer_conductivity_average(spheroid) -> TensISO{2,3}

Voigt (volume) average of the layer conductivities, ``\\sum_k f_k\\,k_k``
(scalar, direction-independent — the Voigt/Reuss bounds only ever need
the phase's volume-averaged property, not its shape).
"""
function layer_conductivity_average(s::LayeredSpheroid{T, N}) where {T, N}
    k_layers = _spheroid_layer_moduli(s)
    f = ntuple(k -> layer_volume_fraction(s, k), Val(N))
    return TensND.TensISO{3}(sum(f[k] * k_layers[k] for k in 1:N))
end

"""
    layer_resistivity_average(spheroid) -> TensISO{2,3}

Reuss (volume) average of the layer resistivities, ``\\sum_k f_k/k_k``.
"""
function layer_resistivity_average(s::LayeredSpheroid{T, N}) where {T, N}
    k_layers = _spheroid_layer_moduli(s)
    f = ntuple(k -> layer_volume_fraction(s, k), Val(N))
    return TensND.TensISO{3}(sum(f[k] / k_layers[k] for k in 1:N))
end

# =============================================================================
#  Bundled localization + contribution for a confocal layered spheroid
#
#  `conductivity_contribution` already calls BOTH `flux_gradient_loc` and
#  `gradient_gradient_loc`, each of which reruns `_spheroid_concentration`
#  (i.e. the confocal transfer-matrix recurrence `spheroid_ba_ratios`).  Add
#  the scheme layer's own `gradient_gradient_loc` and a Mori-Tanaka phase costs
#  THREE recurrences where `_spheroid_concentration` already returns all four
#  scalars at once.
#
#  Bitwise identical: `A` and `B` are the verbatim expressions of the two
#  localization functions, and `N = B - K₀ ⋅ A` is the verbatim body of
#  `conductivity_contribution`.
# =============================================================================

function Core.loc_and_stiffness(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = MFH_Core.extract_iso_conductivity(K₀)
    αt, αa, βt, βa = _spheroid_concentration(s, k₀)
    A = TensND.TensTI{2}(αt, αa, s.axis)
    B = k₀ * TensND.TensTI{2}(βt, βa, s.axis)
    return (A, B - K₀ ⋅ A)
end

function Core.loc_and_stress_average(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        kw...,
    ) where {T, N}
    k₀ = MFH_Core.extract_iso_conductivity(K₀)
    αt, αa, βt, βa = _spheroid_concentration(s, k₀)
    return (
        TensND.TensTI{2}(αt, αa, s.axis),
        k₀ * TensND.TensTI{2}(βt, βa, s.axis),
    )
end

# ── Elasticity ──────────────────────────────────────────────────────────────
#
#  Same contract as the conduction entries above and as `LayeredSphere`'s: a
#  composite spheroid has no Hill tensor, so what the schemes consume is its
#  volume-averaged concentration tensor, and `C₁` is IGNORED — the moduli of a
#  composite live in its layers (`layer_modulus`), not in a single phase tensor.
#
#  Case I alone would give only the axisymmetric block; all three elementary
#  problems together give the six coefficients, which is why these arrive with
#  `elastic_cases.jl` and not before.

"""
    strain_strain_loc(s::LayeredSpheroid, C₁, C₀; D = 6, kw...) -> Tens{4,3}

Volume-averaged strain concentration tensor ``\\mathbb{A}`` of an elastic `n`-layer
confocal spheroid, ``\\langle\\boldsymbol{\\varepsilon}\\rangle = \\mathbb{A}:\\boldsymbol{E}``, ready for a mean-field scheme.

`C₁` is accepted for signature compatibility and **ignored**. `D` is the number
of degrees kept per harmonic family; a single homogeneous spheroid is exact at
any `D` and reproduces Eshelby, a layered one converges geometrically.

Prolate and oblate alike, an oblate spheroid going through the complex
substitution ``q = \\mathrm{i}\\tau``. Only the default axis ``\\underline{e}_3`` and perfect interfaces are
supported — see [`spheroid_strain_concentration`](@ref).
"""
function strain_strain_loc(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        D::Int = 6, kw...,
    ) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    return spheroid_strain_concentration(s, C₀; D, kw...).A
end

"""
    stiffness_contribution(s::LayeredSpheroid, C₁, C₀; D = 6, kw...) -> Tens{4,3}

Size-independent **stiffness contribution tensor** of a composite spheroid,

```math
\\mathbb{N} = \\sum_k f_k\\,(\\mathbb{C}_k - \\mathbb{C}_0) : \\mathbb{A}_k .
```

Assembled layer by layer, because a composite spheroid is heterogeneous and the
``(\\mathbb{C}_1 - \\mathbb{C}_0):\\mathbb{A}`` of a homogeneous inhomogeneity does not apply. The per-layer
``\\mathbb{A}_k`` come from [`spheroid_layer_strain_concentration`](@ref); the total ``\\mathbb{A}``
alone cannot give this, the layers having different moduli. `C₁` is ignored.
"""
function MFH_Core.stiffness_contribution(
        s::LayeredSpheroid{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        D::Int = 6, kw...,
    ) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    r = spheroid_layer_strain_concentration(s, C₀; D, kw...)
    acc = layer_volume_fraction(s, 1) * ((layer_modulus(s, 1) - C₀) ⊡ r.layers[1])
    for k in 2:N
        acc += layer_volume_fraction(s, k) * ((layer_modulus(s, k) - C₀) ⊡ r.layers[k])
    end
    return acc
end
