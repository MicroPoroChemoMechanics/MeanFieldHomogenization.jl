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

# ── Incompressible layers ───────────────────────────────────────────────────
#
#  A layer with k = ∞ has α_k = 0 and 3k_k = ∞, so its mean pressure 3k_k α_k
#  is finite but not computable as that product (∞·0 = NaN). With
#  u_r = A r + B/r² and σ_rr = 3kA − 4μB/r³, it is σ_rr + 4μ u_r/r at any
#  radius of the layer, which the state at r_k⁻ gives directly.

@inline _is_incompressible(κ) = is_hard_numeric(κ) && isinf(κ)

_any_incompressible(sphere::LayeredSphere{T, N}) where {T, N} =
    any(k -> _is_incompressible(_iso_bulk_shear(layer_modulus(sphere, k))[1]), 1:N)

"""
    _layer_pressures(sphere, C₀, α) -> NTuple{N}

``3k_k\\,\\alpha_k`` per layer, the ``\\mathbb J`` part of the mean stress of layer ``k`` per unit
remote strain. For an incompressible layer, where it reads ``\\infty\\cdot 0``, it is taken
from the state at ``r_k^-``: ``3k_kA_k = \\sigma_{rr} + 4\\mu_k\\,u_r/r_k``.
"""
function _layer_pressures(sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}, α) where {T, N}
    C_k = _layer_iso_pairs(sphere)
    _any_incompressible(sphere) || return ntuple(k -> C_k[k][1] * α[k], Val(N))
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    κμ = _bulk_layer_moduli(sphere)
    radii = sphere.radii
    inside, s_m = _bulk_state_seq(sphere, κ₀, μ₀)
    A_inf, _ = _bulk_extract_AB(radii[N], κ₀, μ₀, s_m[1], s_m[2])
    return ntuple(Val(N)) do k
        if _is_incompressible(κμ[k][1])
            u, σ = inside[k]
            (σ + 4 * κμ[k][2] * u / radii[k]) / A_inf
        else
            C_k[k][1] * α[k]
        end
    end
end

# Σ f_k 3k_k α_k and Σ f_k (3k_k − α₀) α_k, the J parts of the mean stress and
# of the contribution. Without an incompressible layer they are the plain sums,
# operation for operation as before.
function _bulk_stress_sum(sphere::LayeredSphere{T, N}, C₀, α, f) where {T, N}
    if _any_incompressible(sphere)
        P = _layer_pressures(sphere, C₀, α)
        return sum(f[k] * P[k] for k in 1:N)
    end
    C_k = _layer_iso_pairs(sphere)
    return sum(f[k] * C_k[k][1] * α[k] for k in 1:N)
end

function _bulk_contrast_sum(sphere::LayeredSphere{T, N}, C₀, α, f, α₀) where {T, N}
    if _any_incompressible(sphere)
        P = _layer_pressures(sphere, C₀, α)
        return sum(f[k] * (P[k] - α₀ * α[k]) for k in 1:N)
    end
    C_k = _layer_iso_pairs(sphere)
    return sum(f[k] * (C_k[k][1] - α₀) * α[k] for k in 1:N)
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

With `external = false` the outer interface is left out: its surface stress is
then counted with the matrix rather than with the sphere (see
[`strain_strain_loc`](@ref)).
"""
function _membrane_surface_stress(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3};
        external::Bool = true,
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
    for k in 1:(external ? N : N - 1)
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

# ── Interface jumps in the average strain and gradient ──────────────────────
#
#  Over a ball of radius r the average strain is a surface integral,
#
#      ⟨ε⟩ = (1/V) ∮_{∂B} u ⊗ˢ n dS ,
#
#  so it is fixed by the displacement on the sphere of radius r. The sum of the
#  layer averages Σ f_k ⟨ε⟩_k telescopes to the same integral, read on the inner
#  side of r_N, MINUS (1/V) Σ_k ∮_{r_k} [u] ⊗ˢ n dS over the interfaces it
#  contains: it is the strain of the material only, without the opening of a
#  spring interface. The jump belongs to some phase, or the strain-average rule
#  E = Σ_i f_i ⟨ε⟩_i fails; Echoes, `LayeredSpheroid` and the laminates all give
#  it to the inclusion. On the two harmonics the angular integrals reduce the
#  jump term of the interface r_k, in a ball of radius R, to
#
#      bulk          r_k² [u_r]_k / R³                  (u_r per unit A∞),
#      deviatoric    r_k² ([U]_k + 3 [W]_k) / (5 R³)    (u_r = U P₂, u_θ = W dP₂/dθ),
#      conduction    r_k² [T]_k / R³                    (T per unit A∞),
#
#  the deviatoric weights (1/5, 3/5) being the ones for which mode 1 (U, W) =
#  (2r, r) averages to 1 and the singular modes 3 and 4 to 0.

# A primal interface lets the field itself jump (the displacement across a
# spring, the temperature across a Kapitza resistance); a dual one, its flux.
_is_primal(::AbstractInterface) = false
_is_primal(::SpringInterface) = true
_is_primal(::KapitzaInterface) = true

_has_primal(sphere::LayeredSphere{T, N}) where {T, N} =
    any(k -> _is_primal(layer_interface(sphere, k)), 1:N)

"""
    _strain_jump_moments(sphere, C₀) -> (jb, jd)

Per interface ``r_k``, the moments ``r_k^2\\,[\\![u_r]\\!]_k`` (bulk, per unit remote
amplitude) and ``r_k^2\\,([\\![U]\\!]_k + 3[\\![W]\\!]_k)/5`` (deviatoric, per unit remote
deviatoric strain) of the displacement jump. Divided by the volume of a ball,
or of a shell, they are what its jumps add to its average strain. Zero across a
perfect or a dual interface: the jump is ``(\\mathbf J - \\mathbb 1)\\,\\mathbf s``, whose displacement
rows vanish unless the interface is primal.
"""
function _strain_jump_moments(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}
    ) where {T, N}
    κ₀, μ₀ = _iso_bulk_shear(C₀)
    κμ = _bulk_layer_moduli(sphere)
    radii = sphere.radii
    inside_b, s_b = _bulk_state_seq(sphere, κ₀, μ₀)
    A_inf, _ = _bulk_extract_AB(radii[N], κ₀, μ₀, s_b[1], s_b[2])
    states_s, _ = _shear_state_seq(sphere, C₀)
    jb = ntuple(Val(N)) do k
        (κk, μk) = κμ[k]
        J = _bulk_interface_T(layer_interface(sphere, k), κk, μk, radii[k])
        radii[k]^2 * ((J - _eye(eltype(J), size(J, 1))) * inside_b[k])[1] / A_inf
    end
    jd = ntuple(Val(N)) do k
        (κk, μk) = κμ[k]
        J = _shear_interface_T(layer_interface(sphere, k), κk, μk, radii[k])
        ΔS = (J - _eye(eltype(J), size(J, 1))) * states_s[k]
        radii[k]^2 * (ΔS[1] + 3 * ΔS[2]) / 5
    end
    return jb, jd
end

"""
    _strain_jump_terms(sphere, C₀; external = true) -> (Δα, Δβ)

What the displacement jumps add to the average strain of the whole composite
sphere, per unit remote strain, on ``\\mathbb J`` and ``\\mathbb K``: those of every inner
interface, and that of the outer one when `external` is `true`. Exactly zero
when no interface is primal, so a sphere with perfect interfaces keeps its
digits.
"""
function _strain_jump_terms(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3};
        external::Bool = true,
    ) where {T, N}
    z = zero(promote_type(T, eltype(C₀)))
    _has_primal(sphere) || return (z, z)
    jb, jd = _strain_jump_moments(sphere, C₀)
    R³ = sphere.radii[N]^3
    last = external ? N : N - 1
    return sum((jb[k] for k in 1:last); init = z) / R³,
        sum((jd[k] for k in 1:last); init = z) / R³
end

"""
    _layer_strain_jump_terms(sphere, C₀, layer; external, internal) -> (Δα, Δβ)

Per-layer counterpart of [`_strain_jump_terms`](@ref): the jump across the outer
boundary of `layer` when `external`, and across its inner boundary when
`internal`, divided by the volume of the shell.
"""
function _layer_strain_jump_terms(
        sphere::LayeredSphere{T, N}, C₀::TensND.TensISO{4, 3}, layer::Int;
        external::Bool, internal::Bool,
    ) where {T, N}
    z = zero(promote_type(T, eltype(C₀)))
    inner = internal && layer > 1
    (external || inner) && _has_primal(sphere) || return (z, z)
    jb, jd = _strain_jump_moments(sphere, C₀)
    radii = sphere.radii
    V = radii[layer]^3 - (layer == 1 ? zero(eltype(radii)) : radii[layer - 1]^3)
    Δα = (external ? jb[layer] : z) + (inner ? jb[layer - 1] : z)
    Δβ = (external ? jd[layer] : z) + (inner ? jd[layer - 1] : z)
    return Δα / V, Δβ / V
end

"""
    _gradient_jump_moments(sphere, k₀) -> NTuple{N}

Conduction twin of [`_strain_jump_moments`](@ref): per interface, the moment
``r_k^2\\,[\\![\\hat T]\\!]_k`` of the temperature jump, per unit remote gradient.
"""
function _gradient_jump_moments(sphere::LayeredSphere{T, N}, k₀) where {T, N}
    k_layers = _cond_layer_moduli(sphere)
    radii = sphere.radii
    inside, s_matrix = _cond_state_seq(sphere, k₀)
    A_inf, _ = _cond_extract_AB(radii[N], k₀, s_matrix[1], s_matrix[2])
    return ntuple(Val(N)) do k
        J = _cond_interface_T(layer_interface(sphere, k), k_layers[k], radii[k])
        radii[k]^2 * ((J - _eye(eltype(J), size(J, 1))) * inside[k])[1] / A_inf
    end
end

"""
    _gradient_jump_term(sphere, k₀; external = true) -> Δα

Conduction twin of [`_strain_jump_terms`](@ref).
"""
function _gradient_jump_term(sphere::LayeredSphere{T, N}, k₀; external::Bool = true) where {T, N}
    z = zero(promote_type(T, typeof(k₀)))
    _has_primal(sphere) || return z
    jT = _gradient_jump_moments(sphere, k₀)
    last = external ? N : N - 1
    return sum((jT[k] for k in 1:last); init = z) / sphere.radii[N]^3
end

"""
    _layer_gradient_jump_term(sphere, k₀, layer; external, internal) -> Δα

Conduction twin of [`_layer_strain_jump_terms`](@ref).
"""
function _layer_gradient_jump_term(
        sphere::LayeredSphere{T, N}, k₀, layer::Int; external::Bool, internal::Bool,
    ) where {T, N}
    z = zero(promote_type(T, typeof(k₀)))
    inner = internal && layer > 1
    (external || inner) && _has_primal(sphere) || return z
    jT = _gradient_jump_moments(sphere, k₀)
    radii = sphere.radii
    V = radii[layer]^3 - (layer == 1 ? zero(eltype(radii)) : radii[layer - 1]^3)
    return ((external ? jT[layer] : z) + (inner ? jT[layer - 1] : z)) / V
end

"""
    strain_strain_loc(sphere::LayeredSphere, C₁, C₀; external = true, kw...) -> TensISO{4,3}

Whole-inclusion **strain concentration tensor** of a composite sphere embedded
in the isotropic reference `C₀`:

```math
\\mathbb{A}_\\Omega = \\Big(\\sum_k f_k\\,\\alpha_k + \\Delta\\alpha\\Big)\\,\\mathbb{J} + \\Big(\\sum_k f_k\\,\\beta_k + \\Delta\\beta\\Big)\\,\\mathbb{K},
\\qquad
\\langle\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\mathbb{A}_\\Omega:\\boldsymbol{\\varepsilon}^{\\infty}.
```

The sums are the strain of the material of the layers. ``\\Delta\\alpha`` and ``\\Delta\\beta`` add the
displacement jumps across its [`SpringInterface`](@ref)s,
``\\frac{1}{V}\\oint[\\![\\underline{u}]\\!]\\otimes^{\\mathrm s}\\underline{n}\\,\\mathrm{d}S``, and vanish when every interface is
perfect or a membrane. The jumps must belong to some phase for the strain
average rule ``\\boldsymbol{E} = \\sum_i f_i\\,\\langle\\boldsymbol{\\varepsilon}\\rangle_i`` to hold: those of the inner
interfaces always belong to the sphere, and `external` decides for the outer
one.

- `external = true` (default, and what the schemes use): the outer interface
  belongs to the sphere, whose strain is then read on the matrix side of
  ``r_N``. This is the convention of Echoes, of
  [`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid)
  and of the laminates.
- `external = false`: it belongs to the matrix, and the strain is read on the
  inner side of ``r_N`` — Echoes' `sphere_eE(n - 1, external = False)`. The
  surface stress of an outer [`MembraneInterface`](@ref) then leaves
  [`stress_strain_loc`](@ref) and [`stiffness_contribution`](@ref) with the same
  keyword, so that ``\\mathbb N = \\mathbb A_{\\sigma\\varepsilon} - \\mathbb C_0:\\mathbb A_\\Omega`` holds for either value.

`C₁` is accepted for signature compatibility with the generic
`strain_strain_loc(::AbstractInclusion, C₁, C₀)` used by the scheme dispatch,
but is **ignored**: the moduli of a composite sphere live in its layers
(`layer_modulus`), not in a single phase tensor.

For a single layer and a perfect interface this reduces exactly to the Eshelby
result for a sphere.
"""
function strain_strain_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    Δα, Δβ = _strain_jump_terms(sphere, C₀; external)
    return TensISO{3}(
        sum(f[k] * α[k] for k in 1:N) + Δα, sum(f[k] * β[k] for k in 1:N) + Δβ
    )
end

"""
    stiffness_contribution(sphere::LayeredSphere, C₁, C₀; external = true, kw...) -> TensISO{4,3}

Size-independent **stiffness contribution tensor** of a composite sphere,

```math
\\mathbb{N} = \\sum_k f_k\\,(\\mathbb{C}_k - \\mathbb{C}_0):\\mathbb{A}_k - \\mathbb{C}_0:\\Delta\\mathbb{A},
```

plus the Gurtin–Murdoch surface stress of any membrane interface
(`_membrane_surface_stress`). ``\\Delta\\mathbb{A} = \\Delta\\alpha\\,\\mathbb{J} + \\Delta\\beta\\,\\mathbb{K}`` is the part of
[`strain_strain_loc`](@ref) due to the displacement jumps of spring interfaces:
a jump is strain without stress, so it lowers the contribution. `external`
has the meaning it has there.

Assembled layer by layer: a composite sphere is heterogeneous, so the usual
``(\\mathbb{C}_1 - \\mathbb{C}_0):\\mathbb{A}`` of a homogeneous inhomogeneity does not apply. `C₁` is ignored,
as in [`strain_strain_loc`](@ref).
"""
function stiffness_contribution(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    C_k = _layer_iso_pairs(sphere)
    α₀, β₀ = TensND.get_data(C₀)
    a = _bulk_contrast_sum(sphere, C₀, α, f, α₀)
    b = sum(f[k] * (C_k[k][2] - β₀) * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀; external)
    Δα, Δβ = _strain_jump_terms(sphere, C₀; external)
    return TensISO{3}(a + a_surf - α₀ * Δα, b + b_surf - β₀ * Δβ)
end

"""
    stress_strain_loc(sphere::LayeredSphere, C₁, C₀; external = true, kw...) -> TensISO{4,3}

Whole-inclusion **average stress** per unit remote strain,

```math
\\langle\\mathbb{C}:\\boldsymbol{\\varepsilon}\\rangle_\\Omega = \\Big(\\sum_k f_k\\,\\mathbb{C}_k:\\mathbb{A}_k\\Big):\\boldsymbol{\\varepsilon}^{\\infty},
```

assembled layer by layer, plus the Gurtin–Murdoch surface stress of any
membrane interface (`_membrane_surface_stress`), that of the outer one only
when `external` is `true` (see [`strain_strain_loc`](@ref)). The traction is
continuous across a spring interface, so its displacement jump leaves this
average unchanged. This is what the self-consistent and Mori-Tanaka kernels
need; it is *not* ``\\mathbb{C}_1:\\mathbb{A}_\\Omega``. `C₁` is ignored (see
[`strain_strain_loc`](@ref)).

Consistency: `stiffness_contribution = stress_strain_loc - C₀ : strain_strain_loc`.
"""
function stress_strain_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    C_k = _layer_iso_pairs(sphere)
    a = _bulk_stress_sum(sphere, C₀, α, f)
    b = sum(f[k] * C_k[k][2] * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀; external)
    return TensISO{3}(a + a_surf, b + b_surf)
end

"""
    flux_gradient_loc(sphere::LayeredSphere, K₁, K₀; external = true, kw...) -> TensISO{2,3}

Conductivity counterpart of [`stress_strain_loc`](@ref):
``\\langle k\\,\\nabla T\\rangle_\\Omega = \\big(\\sum_k f_k\\,k_k\\,\\alpha_k\\big)\\,\\nabla T^{\\infty}``, plus the surface-conduction flux
[`_cond_surface_flux`](@ref) of any dual (surface-conductive) interface, that
of the outer one only when `external` is `true` (see
[`gradient_gradient_loc`](@ref)). `K₁` is ignored.
"""
function flux_gradient_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    k_layers = _cond_layer_moduli(sphere)
    f = _layer_fractions(sphere)
    return TensISO{3}(
        sum(f[k] * k_layers[k] * α[k] for k in 1:N) +
            _cond_surface_flux(sphere, k₀; external)
    )
end

"""
    gradient_gradient_loc(sphere::LayeredSphere, K₁, K₀; external = true, kw...) -> TensISO{2,3}

Whole-inclusion **gradient concentration tensor**
``\\alpha_\\Omega = \\sum_k f_k\\,\\alpha_k + \\Delta\\alpha``, the conductivity counterpart of
[`strain_strain_loc`](@ref): ``\\Delta\\alpha`` adds the temperature jumps across the
[`KapitzaInterface`](@ref)s, those of the inner interfaces always and that of the
outer one when `external` is `true` (the default, and what the schemes use).
`K₁` is ignored (see there).

The per-layer form is available as
`gradient_gradient_loc(sphere, K₀; layer = k)`.
"""
function gradient_gradient_loc(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    f = _layer_fractions(sphere)
    return TensISO{3}(sum(f[k] * α[k] for k in 1:N) + _gradient_jump_term(sphere, k₀; external))
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
        external::Bool = true,
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    Δα, Δβ = _strain_jump_terms(sphere, C₀; external)
    A = TensISO{3}(
        sum(f[k] * α[k] for k in 1:N) + Δα, sum(f[k] * β[k] for k in 1:N) + Δβ
    )
    C_k = _layer_iso_pairs(sphere)
    α₀, β₀ = TensND.get_data(C₀)
    a = _bulk_contrast_sum(sphere, C₀, α, f, α₀)
    b = sum(f[k] * (C_k[k][2] - β₀) * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀; external)
    return (A, TensISO{3}(a + a_surf - α₀ * Δα, b + b_surf - β₀ * Δβ))
end

function Core.loc_and_stress_average(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{4, 3},
        C₀::TensND.TensISO{4, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    α, β, f = _layer_localizations(sphere, C₀)
    Δα, Δβ = _strain_jump_terms(sphere, C₀; external)
    A = TensISO{3}(
        sum(f[k] * α[k] for k in 1:N) + Δα, sum(f[k] * β[k] for k in 1:N) + Δβ
    )
    C_k = _layer_iso_pairs(sphere)
    a = _bulk_stress_sum(sphere, C₀, α, f)
    b = sum(f[k] * C_k[k][2] * β[k] for k in 1:N)
    a_surf, b_surf = _membrane_surface_stress(sphere, C₀; external)
    return (A, TensISO{3}(a + a_surf, b + b_surf))
end

function Core.loc_and_stiffness(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    f = _layer_fractions(sphere)
    Δα = _gradient_jump_term(sphere, k₀; external)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N) + Δα)
    k_layers = _cond_layer_moduli(sphere)
    N_K = sum(f[k] * (k_layers[k] - k₀) * α[k] for k in 1:N) +
        _cond_surface_flux(sphere, k₀; external) - k₀ * Δα
    return (A, TensISO{3}(N_K))
end

function Core.loc_and_stress_average(
        sphere::LayeredSphere{T, N},
        ::TensND.AbstractTens{2, 3},
        K₀::TensND.TensISO{2, 3};
        external::Bool = true,
        kw...,
    ) where {T, N}
    k₀ = _iso_scalar(K₀)
    α = _cond_localization(sphere, k₀)
    f = _layer_fractions(sphere)
    A = TensISO{3}(sum(f[k] * α[k] for k in 1:N) + _gradient_jump_term(sphere, k₀; external))
    k_layers = _cond_layer_moduli(sphere)
    B = TensISO{3}(
        sum(f[k] * k_layers[k] * α[k] for k in 1:N) +
            _cond_surface_flux(sphere, k₀; external)
    )
    return (A, B)
end
