# =============================================================================
#  closed_pores.jl — the pressure of a fluid sealed in a pore.
#
#  A fluid trapped in a closed pore (no flow between pores: the undrained limit
#  of each pore, or the high-frequency limit of a wave) is an inclusion of
#  stiffness 3k_f 𝕁, which every scheme homogenizes as any other phase. Its
#  pressure is the opposite of its mean stress trace, read on the stress
#  concentration tensor of that phase; the concentration tensors come from
#  `Schemes.concentration_tensors`.
# =============================================================================

"""
    pore_pressure_tensor(ct::ConcentrationTensors, pore::Symbol) -> AbstractTens{2}

Second-order tensor ``\\boldsymbol{B}^{\\mathrm{sk}}_r`` that gives the mean
pressure of phase `pore` under a macroscopic stress,

```math
p_r = -\\frac1d\\,\\operatorname{tr}\\langle\\boldsymbol{\\sigma}\\rangle_r
= -\\boldsymbol{B}^{\\mathrm{sk}}_r:\\boldsymbol{\\Sigma},
\\qquad
\\boldsymbol{B}^{\\mathrm{sk}}_r = \\frac1d\\,\\boldsymbol{1}:\\mathbb{B}_r ,
```

in dimension ``d`` (3, or 2 for a cell of the plane), with ``\\mathbb{B}_r`` the stress concentration tensor of the phase
([`stress_stress_concentration`](@ref)). When the phase is a fluid sealed in
its pores (stiffness ``d\\,k_f\\,\\mathbb{J}``), ``p_r`` is the fluid pressure:
pores of different shapes or orientations carry different pressures, the
*pressure polarization* of [shafiro1997](@citet), whose tensor ``\\boldsymbol{Q}``
(eq. 6, pressure positive in compression) is ``-\\boldsymbol{B}^{\\mathrm{sk}}_r``.
The sign follows [`skempton_tensor`](@ref): under a confining pressure
``\\boldsymbol{\\Sigma} = -P\\,\\boldsymbol{1}``, ``p_r = P\\,\\operatorname{tr}\\boldsymbol{B}^{\\mathrm{sk}}_r``,
the Skempton coefficient of the pore.

The tensor depends on the scheme through ``\\mathbb{B}_r``: in the dilute
limit it is that of an isolated pore, and with Mori–Tanaka or the
self-consistent scheme it accounts for the other pores, fluid-filled or not.
"""
function pore_pressure_tensor(ct::Schemes.ConcentrationTensors, pore::Symbol)
    𝔹 = MFH_Core.stress_stress_concentration(ct, pore)
    return TensND.tens_Id2(Val(TensND.get_dim(𝔹)), Val(eltype(𝔹))) ⊡ 𝔹 / TensND.get_dim(𝔹)
end
