# =============================================================================
#  contribution.jl — size-independent contribution tensors of the dilute
#  Eshelby problem (Kachanov-Sevostianov convention).
#
#  Given an inclusion of stiffness `C₁` in a matrix `C₀`, the dilute
#  effective stiffness correction is
#
#      ΔC_eff = f × N,      N = (C₁ - C₀) : A_εε,
#
#  where `A_εε` is the strain-strain concentration tensor.  Dually, the
#  dilute effective compliance correction is
#
#      ΔS_eff = f × H,      H = (S₁ - S₀) : A_σσ,
#
#  with `S = C⁻¹` and `A_σσ` the stress-stress concentration tensor.
#  The helpers `delta_stiffness` and `delta_compliance` apply the
#  volume fraction `f` (analogous to Budiansky density for cracks).
#
#  Conductivity analogs are implemented with 2-tensor algebra.
# =============================================================================

"""
    stiffness_contribution(incl, C₁, C₀; kw...) -> Tens{4,3}

Size-independent **stiffness contribution tensor**
``\\mathbb{N} = (\\mathbb{C}_1 - \\mathbb{C}_0):\\mathbb{A}_{\\varepsilon\\varepsilon}`` for an `AbstractInclusion` of stiffness `C₁`
in a matrix `C₀`.  For a dilute family of inclusions of volume fraction
``f``, the effective stiffness correction is
``\\Delta\\mathbb{C}^{\\mathrm{hom}} = f\\,\\mathbb{N}`` — see [`delta_stiffness`](@ref).

!!! note "Heterogeneous inclusions"
    ``(\\mathbb{C}_1 - \\mathbb{C}_0):\\mathbb{A}_{\\varepsilon\\varepsilon}`` presupposes a *single* stiffness inside the inclusion.
    When [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) is `false` the exact identity
    ``\\mathbb{N} = \\mathbb{A}_{\\sigma\\varepsilon} - \\mathbb{C}_0:\\mathbb{A}_{\\varepsilon\\varepsilon}`` is used instead — it needs no `C₁`, only the two
    concentration tensors, and reduces to the expression above whenever a
    uniform `C₁` does exist. This is what lets a layered or finite-element
    inclusion reach the schemes through gate B alone.

See [kachanov2018](@citet).
"""
function stiffness_contribution(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    A = strain_strain_concentration(incl, C₁, C₀; kw...)
    is_homogeneous_inclusion(incl) || return stress_strain_concentration(incl, C₁, C₀; kw...) - C₀ ⊡ A
    return (C₁ - C₀) ⊡ A
end

"""
    compliance_contribution(incl, C₁, C₀; kw...) -> Tens{4,3}

Size-independent **compliance contribution tensor**
``\\mathbb{H} = (\\mathbb{S}_1 - \\mathbb{S}_0):\\mathbb{A}_{\\sigma\\sigma}`` for an `AbstractInclusion` of stiffness `C₁`
in a matrix `C₀` (``\\mathbb{S} = \\mathbb{C}^{-1}``).  For a dilute family, the effective
compliance correction is ``\\Delta\\mathbb{S}^{\\mathrm{hom}} = f\\,\\mathbb{H}`` — see [`delta_compliance`](@ref).

It is evaluated through the exact identity
``\\mathbb{H} = -\\,\\mathbb{S}_0:\\mathbb{N}:\\mathbb{S}_0``, with ``\\mathbb{N}`` the
[`stiffness_contribution`](@ref), which never inverts `C₁`. A void (`C₁ = 0`)
therefore has the finite contribution it should, where the literal product
``(\\mathbb{S}_1 - \\mathbb{S}_0):\\mathbb{A}_{\\sigma\\sigma}`` is an infinite compliance
times a zero concentration. The identity holds for any inclusion, heterogeneous
ones included, for which `C₁` is not even meaningful.

See [kachanov2018](@citet).
"""
function compliance_contribution(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    S₀ = inv(C₀)
    return -(S₀ ⊡ stiffness_contribution(incl, C₁, C₀; kw...) ⊡ S₀)
end

"""
    delta_stiffness(N, f) -> Tens{4,3}

Dilute **effective stiffness correction** ``\\Delta\\mathbb{C} = f\\,\\mathbb{N}`` from the
size-independent contribution tensor ``\\mathbb{N}`` and the volume fraction ``f``
of inclusions sharing that contribution.
"""
delta_stiffness(N::TensND.AbstractTens{4, 3}, f) = f * N

"""
    delta_compliance(H, f) -> Tens{4,3}

Dilute **effective compliance correction** ``\\Delta\\mathbb{S} = f\\,\\mathbb{H}`` from the
size-independent contribution tensor ``\\mathbb{H}`` and the volume fraction ``f``.
(See also the crack-specific methods `delta_compliance(crack, H, ε)`
which use the Budiansky density convention and apply a geometric
prefactor.)
"""
delta_compliance(H::TensND.AbstractTens{4, 3}, f) = f * H

# =============================================================================
#  Conductivity contribution (2-tensor fields)
#
#  Same sign convention as `concentration.jl`: the "flux" side of these
#  expressions carries σ ≡ -q = K·∇T, so `A_q∇` and `A_qq` are the transport
#  twins of `A_σε` and `A_σσ` symbol for symbol.  Both kernels below are
#  written in terms of those tensors, so nothing here needs a compensating
#  sign — see the block comment in `concentration.jl`.
# =============================================================================

"""
    conductivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}

Size-independent **conductivity contribution tensor**
``\\boldsymbol{N} = (\\boldsymbol{K}_1 - \\boldsymbol{K}_0)\\cdot\\boldsymbol{A}_{\\nabla\\nabla}`` for an `AbstractInclusion` of conductivity
`K₁` in a matrix `K₀`.  Dilute effective correction:
``\\Delta\\boldsymbol{K}^{\\mathrm{hom}} = f\\,\\boldsymbol{N}``.

For a heterogeneous inclusion the exact
``\\boldsymbol{N} = \\boldsymbol{A}_{q\\nabla} - \\boldsymbol{K}_0\\cdot\\boldsymbol{A}_{\\nabla\\nabla}`` is used
instead — see [`stiffness_contribution`](@ref).
"""
function conductivity_contribution(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    A = gradient_gradient_concentration(incl, K₁, K₀; kw...)
    is_homogeneous_inclusion(incl) || return flux_gradient_concentration(incl, K₁, K₀; kw...) - K₀ ⋅ A
    return (K₁ - K₀) ⋅ A
end

"""
    resistivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}

Size-independent **resistivity contribution tensor**
``\\boldsymbol{H} = (\\boldsymbol{K}_1^{-1} - \\boldsymbol{K}_0^{-1})\\cdot\\boldsymbol{A}_{qq}``
for an `AbstractInclusion` (the resistivity is ``\\boldsymbol{K}^{-1}``).
Dilute effective correction: ``\\Delta(\\boldsymbol{K}^{-1})^{\\mathrm{hom}} = f\\,\\boldsymbol{H}``.

It is evaluated through the exact identity
``\\boldsymbol{H} = -\\,\\boldsymbol{K}_0^{-1}\\cdot\\boldsymbol{N}\\cdot\\boldsymbol{K}_0^{-1}``, with
``\\boldsymbol{N}`` the [`conductivity_contribution`](@ref), which never inverts `K₁`:
an insulating inclusion (`K₁ = 0`) has a finite contribution, and so has a
heterogeneous one — see [`compliance_contribution`](@ref).
"""
function resistivity_contribution(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    R₀ = inv(K₀)
    return -(R₀ ⋅ conductivity_contribution(incl, K₁, K₀; kw...) ⋅ R₀)
end

"""
    delta_conductivity(N_K, f) -> Tens{2,3}

Dilute effective conductivity correction ``\\Delta\\boldsymbol{K} = f\\,\\boldsymbol{N}``,
with ``\\boldsymbol{N}`` the conductivity contribution `N_K`.
"""
delta_conductivity(N::TensND.AbstractTens{2, 3}, f) = f * N

"""
    delta_resistivity(H_R, f) -> Tens{2,3}

Dilute effective resistivity correction ``\\Delta(\\boldsymbol{K}^{-1}) = f\\,\\boldsymbol{H}``,
with ``\\boldsymbol{H}`` the resistivity contribution `H_R`.  Generic
2-argument method; for cracks, see the 3-argument
`delta_resistivity(crack, R, ε)` with the Budiansky density prefactor.
"""
delta_resistivity(H::TensND.AbstractTens{2, 3}, f) = f * H

# =============================================================================
#  Bundled two-argument contributions of a flat inclusion.
#
#  Generic fallback of the perf seam consumed by the density branch of
#  Mori-Tanaka.  `Cracks` specializes it to share a single `cod_tensor` solve;
#  any other flat morphology (a user-defined one, a finite-element crack, …)
#  gets the correct — if not the fastest — behavior for free.  Per the
#  contract in the `Core` docstring, the fallback *is* the pair.
# =============================================================================

compliance_and_stiffness_contribution(
    incl::AbstractInclusion, C₀::TensND.AbstractTens{4, 3}; kw...
) = (
    compliance_contribution(incl, C₀; kw...),
    stiffness_contribution(incl, C₀; kw...),
)

compliance_and_stiffness_contribution(
    incl::AbstractInclusion, K₀::TensND.AbstractTens{2, 3}; kw...
) = (
    compliance_contribution(incl, K₀; kw...),
    conductivity_contribution(incl, K₀; kw...),
)
