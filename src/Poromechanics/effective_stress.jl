# =============================================================================
#  effective_stress.jl — the two effective-stress measures of poroelasticity.
#
#  SIGN CONVENTION.  Tension is positive throughout the package, and the pore
#  pressure `p` is positive in compression of the fluid.  Both effective
#  stresses therefore *add* a pressure term: a positive pore pressure makes the
#  effective stress less compressive, which is what opens fractures.
# =============================================================================

"""
    terzaghi_stress(Σ, p) -> Tens{2,3}

**Terzaghi effective stress** ``\\boldsymbol{\\Sigma}' = \\boldsymbol{\\Sigma} +
p\\,\\boldsymbol{1}``.

This is the loading measure that drives the *microstructure*: for a porous or
fractured medium whose solid phase has uniform elastic properties, the problem
defined by ``(\\boldsymbol{\\Sigma}, p)`` splits into

1. a dry problem (no fluid pressure) under the macroscopic stress
   ``\\boldsymbol{\\Sigma} + p\\,\\boldsymbol{1}``, during which pores and
   fractures may open or close, and
2. a superimposed loading ``(-p\\,\\boldsymbol{1}, p)`` whose solution is
   the *uniform* pair ``\\boldsymbol{\\sigma} = -p\\,\\boldsymbol{1}``,
   ``\\boldsymbol{\\varepsilon} = -p\\,\\mathbb{S}_{\\rm s} : \\boldsymbol{1}``.

Step 2 carries no strain singularity, so it cannot change the aperture of a flat
crack. All the information about the evolution of the pore space is therefore
contained in step 1, i.e. it depends on the Terzaghi effective stress alone.
This is the argument of [barthelemyARMA2011](@cite) § 1.1 and the reason why
the constitutive laws of
[`MeanFieldHomogenization.Constitutive`](@ref MeanFieldHomogenization.Constitutive)
drive their internal state with `terzaghi_stress` rather than with
`\\boldsymbol{\\Sigma}`.

Do not confuse it with [`biot_effective_stress`](@ref), which is the measure
that makes the *macroscopic* constitutive law take its drained form.

See also [`biot_tensor`](@ref).
"""
function terzaghi_stress(Σ::TensND.AbstractTens{2, 3}, p::Number)
    T = promote_type(eltype(Σ), typeof(p))
    return Σ + p * TensND.tens_Id2(Val(3), Val(T))
end

"""
    biot_effective_stress(Σ, p, B) -> Tens{2,3}

**Biot effective stress** ``\\boldsymbol{\\Sigma} + p\\,\\boldsymbol{B}``, the measure
for which the poroelastic law
``\\dot{\\boldsymbol{\\Sigma}} = \\mathbb{C}^{\\rm hom} : \\dot{\\boldsymbol{E}} -
\\dot p\\,\\boldsymbol{B}`` reduces to the drained relation
``\\boldsymbol{\\Sigma} + p\\,\\boldsymbol{B} = \\mathbb{C}^{\\rm hom} : \\boldsymbol{E}``.

It coincides with [`terzaghi_stress`](@ref) only when ``\\boldsymbol{B} =
\\boldsymbol{1}``, i.e. for an incompressible solid phase.

See also [`biot_tensor`](@ref), [`terzaghi_stress`](@ref).
"""
function biot_effective_stress(
        Σ::TensND.AbstractTens{2, 3}, p::Number, B::TensND.AbstractTens{2, 3}
    )
    return Σ + p * B
end
