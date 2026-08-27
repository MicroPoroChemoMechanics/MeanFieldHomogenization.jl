# =============================================================================
#  contribution.jl — size-independent contribution tensors of the dilute
#  Eshelby problem (Kachanov-Sevostianov convention).
#
#  Given an inclusion of stiffness `C₁` in a matrix `C₀`, the dilute
#  effective stiffness correction is
#
#      ΔC_eff = f × N,      N = (C₁ - C₀) : A_εε,
#
#  where `A_εε` is the strain-strain localization tensor.  Dually, the
#  dilute effective compliance correction is
#
#      ΔS_eff = f × H,      H = (S₁ - S₀) : A_σσ,
#
#  with `S = C⁻¹` and `A_σσ` the stress-stress localization tensor.
#  The helpers `delta_stiffness` and `delta_compliance` apply the
#  volume fraction `f` (analogous to Budiansky density for cracks).
#
#  Conductivity analogs are implemented with 2-tensor algebra.
# =============================================================================

"""
    stiffness_contribution(incl, C₁, C₀; kw...) -> Tens{4,3}

Size-independent **stiffness contribution tensor**
`N = (C₁ - C₀) : A_εε` for an `AbstractInclusion` of stiffness `C₁`
in a matrix `C₀`.  For a dilute family of inclusions of volume fraction
`f`, the effective stiffness correction is
`ΔC_eff = f × N` — see [`delta_stiffness`](@ref).

!!! note "Heterogeneous inclusions"
    `(C₁ - C₀) : A_εε` presupposes a *single* stiffness inside the inclusion.
    When [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) is `false` the exact identity
    `N = A_σε - C₀ : A_εε` is used instead — it needs no `C₁`, only the two
    localization tensors, and reduces to the expression above whenever a
    uniform `C₁` does exist. This is what lets a layered or finite-element
    inclusion reach the schemes through gate B alone.

See [kachanov2018](@cite).
"""
function stiffness_contribution(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    A = strain_strain_loc(incl, C₁, C₀; kw...)
    is_homogeneous_inclusion(incl) || return stress_strain_loc(incl, C₁, C₀; kw...) - C₀ ⊡ A
    return (C₁ - C₀) ⊡ A
end

"""
    compliance_contribution(incl, C₁, C₀; kw...) -> Tens{4,3}

Size-independent **compliance contribution tensor**
`H = (S₁ - S₀) : A_σσ` for an `AbstractInclusion` of stiffness `C₁`
in a matrix `C₀` (`S = C⁻¹`).  For a dilute family, the effective
compliance correction is `ΔS_eff = f × H` — see [`delta_compliance`](@ref).

!!! note "Heterogeneous inclusions"
    As for [`stiffness_contribution`](@ref), `inv(C₁)` is meaningless when
    [`is_homogeneous_inclusion`](@ref MeanFieldHomogenization.Core.is_homogeneous_inclusion) is `false`; the exact identity
    `H = A_εσ - S₀ : A_σσ = (A_εε - S₀ : A_σε) : S₀` is used instead.

See [kachanov2018](@cite).
"""
function compliance_contribution(
        incl::AbstractInclusion,
        C₁::TensND.AbstractTens{4, 3},
        C₀::TensND.AbstractTens{4, 3};
        kw...
    )
    # `stress_strain_loc` (unlike `stress_stress_loc`) needs no `inv(C₀)`, so
    # `S₀` is computed here exactly once and reused for both `A_σσ = A_σε ⊡ S₀`
    # and the `(S₁ - S₀)` factor, instead of once inside `stress_stress_loc`
    # and once again here.
    S₀ = inv(C₀)
    A_σε = stress_strain_loc(incl, C₁, C₀; kw...)
    is_homogeneous_inclusion(incl) ||
        return (strain_strain_loc(incl, C₁, C₀; kw...) - S₀ ⊡ A_σε) ⊡ S₀
    return (inv(C₁) - S₀) ⊡ (A_σε ⊡ S₀)
end

"""
    delta_stiffness(N, f) -> Tens{4,3}

Dilute **effective stiffness correction** `ΔC = f × N` from the
size-independent contribution tensor `N` and the volume fraction `f`
of inclusions sharing that contribution.
"""
delta_stiffness(N::TensND.AbstractTens{4, 3}, f) = f * N

"""
    delta_compliance(H, f) -> Tens{4,3}

Dilute **effective compliance correction** `ΔS = f × H` from the
size-independent contribution tensor `H` and the volume fraction `f`.
(See also the crack-specific methods `delta_compliance(crack, H, ε)`
which use the Budiansky density convention and apply a geometric
prefactor.)
"""
delta_compliance(H::TensND.AbstractTens{4, 3}, f) = f * H

# =============================================================================
#  Conductivity contribution (2-tensor fields)
#
#  Same sign convention as `localization.jl`: the "flux" side of these
#  expressions carries σ ≡ -q = K·∇T, so `A_q∇` and `A_qq` are the transport
#  twins of `A_σε` and `A_σσ` symbol for symbol.  Both kernels below are
#  written in terms of those tensors, so nothing here needs a compensating
#  sign — see the block comment in `localization.jl`.
# =============================================================================

"""
    conductivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}

Size-independent **conductivity contribution tensor**
`N_K = (K₁ - K₀) · A_∇∇` for an `AbstractInclusion` of conductivity
`K₁` in a matrix `K₀`.  Dilute effective correction:
`ΔK_eff = f × N_K`.

For a heterogeneous inclusion the exact `N_K = A_q∇ - K₀ · A_∇∇` is used
instead — see [`stiffness_contribution`](@ref).
"""
function conductivity_contribution(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    A = gradient_gradient_loc(incl, K₁, K₀; kw...)
    is_homogeneous_inclusion(incl) || return flux_gradient_loc(incl, K₁, K₀; kw...) - K₀ ⋅ A
    return (K₁ - K₀) ⋅ A
end

"""
    resistivity_contribution(incl, K₁, K₀; kw...) -> Tens{2,3}

Size-independent **resistivity contribution tensor**
`H_R = (R₁ - R₀) · A_qq` for an `AbstractInclusion` (with `R = K⁻¹`).
Dilute effective correction: `ΔR_eff = f × H_R`.

For a heterogeneous inclusion the exact `H_R = (A_∇∇ - R₀ · A_q∇) · R₀` is
used instead — see [`compliance_contribution`](@ref).
"""
function resistivity_contribution(
        incl::AbstractInclusion,
        K₁::TensND.AbstractTens{2, 3},
        K₀::TensND.AbstractTens{2, 3};
        kw...
    )
    # `flux_gradient_loc` (unlike `flux_flux_loc`) needs no `inv(K₀)`, so `R₀`
    # is computed here exactly once and reused for both `A_qq = A_q∇ ⋅ R₀`
    # and the `(R₁ - R₀)` factor.
    R₀ = inv(K₀)
    A_q∇ = flux_gradient_loc(incl, K₁, K₀; kw...)
    is_homogeneous_inclusion(incl) ||
        return (gradient_gradient_loc(incl, K₁, K₀; kw...) - R₀ ⋅ A_q∇) ⋅ R₀
    return (inv(K₁) - R₀) ⋅ (A_q∇ ⋅ R₀)
end

"""
    delta_conductivity(N_K, f) -> Tens{2,3}

Dilute effective conductivity correction `ΔK = f × N_K`.
"""
delta_conductivity(N::TensND.AbstractTens{2, 3}, f) = f * N

"""
    delta_resistivity(H_R, f) -> Tens{2,3}

Dilute effective resistivity correction `ΔR = f × H_R`.  Generic
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
