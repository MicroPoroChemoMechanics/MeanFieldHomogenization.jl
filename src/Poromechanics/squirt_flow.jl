# =============================================================================
#  squirt_flow.jl — the frequency-dependent stiffness of a solid whose pores
#  exchange fluid with each other (squirt flow).
#
#  Each pore family exchanges fluid with a common reservoir at a rate set by a
#  relaxation time τ, total fluid mass being conserved (the exchange law of
#  Jakobsen & Chapman 2009, without flow at the scale of the wave). Written per
#  family, the law makes the fluid an inclusion of complex bulk modulus
#  k_f iωτ/(1 + iωτ) loaded by a pressure common to all families. The first
#  part is a cell with complex moduli, homogenized by any scheme
#  (`Schemes.concentration_tensors`); the second is a uniform prestress of the
#  fluid phases in a homogeneous solid, which Levin's uniform field solves
#  exactly.
# =============================================================================

"""
    squirt_flow(rve, scheme, ωτ; fluid, k_f, solid = nothing, property = :C)
        -> (compliance, stiffness, pressure)

Complex effective compliance and stiffness at the dimensionless frequency
`ωτ` of a cell made of one homogeneous solid (phase `solid`, by default the one
taking the volume complement) and of the pore families listed in `fluid`,
saturated by a fluid of bulk modulus `k_f` that flows between families with the
relaxation time ``\\tau``. `pressure[r]` is the tensor
``\\boldsymbol{B}^{\\mathrm{sk}}_r`` of family `r`, complex: its pressure is
``p_r = -\\boldsymbol{B}^{\\mathrm{sk}}_r:\\boldsymbol{\\Sigma}``, as in
[`pore_pressure_tensor`](@ref).

The fluid content of family ``r`` (volume fraction ``\\varphi_r``) changes as
``\\mathrm{i}\\omega\\,\\delta m_r = -\\varphi_r\\,(p_r - \\bar p)/(k_f\\tau)``, with
``\\sum_r \\delta m_r = 0`` ([jakobsen2009](@citet), eq. 18, without the flow at
the scale of the wave), for a time dependence ``e^{\\mathrm{i}\\omega t}``: the
imaginary part of the stiffness is positive (dissipation). The pressure of a
family is that of a fluid of bulk modulus ``k_c = k_f\\,\\mathrm{i}\\omega\\tau/(1 + \\mathrm{i}\\omega\\tau)``
loaded by the common pressure ``\\bar p/(1 + \\mathrm{i}\\omega\\tau)``:

- `ωτ = Inf`: no exchange, the pores are sealed (the fluid phases of
  modulus ``k_f``; [`pore_pressure_tensor`](@ref));
- `ωτ = 0`: one pressure in all the pores, the undrained (Gassmann, Brown and
  Korringa) stiffness of the drained cell ([`undrained_stiffness`](@ref) with
  `k_f`).

For thin cracks in the non-interacting approximation (`DiluteDual`), the
difference between the compliances of the two limits is eq. (10) of
[gueguen2011](@citet).

The scheme is any one whose concentration tensors accept complex moduli; the
geometry, amounts and orientations of the families are those of `rve`, whose
own stiffness of the fluid phases is ignored.
"""
function squirt_flow(
        rve::Schemes.RVE, scheme::Schemes.HomogenizationScheme, ωτ::Real;
        fluid::AbstractVector{Symbol}, k_f::Number,
        solid::Union{Nothing, Symbol} = nothing, property::Symbol = :C
    )
    ωτ ≥ 0 || throw(DomainError(ωτ, "the dimensionless frequency ωτ must be nonnegative"))
    s = Schemes.host_phase_name(rve, solid, "squirt_flow")
    _check_squirt_phases(rve, s, fluid)
    C_s = Schemes.phase_property(rve, s, property)
    # z = 1/(1 + iωτ) is 0 at ωτ = Inf, where the pores are sealed.
    z = inv(1 + im * ωτ)
    k_c = k_f * (1 - z)
    ct = Schemes.concentration_tensors(_with_fluid_modulus(rve, fluid, k_c, property), scheme, property)
    S_hom = ct.compliance
    Id2 = TensND.tens_Id2(Val(3), Val(eltype(S_hom)))
    # Uniform field of a unit prestress pressure in the fluid phases:
    # C_s : ε₀ = 3k_c 𝕁 : ε₀ − 1.
    ε₀ = -(inv(C_s - k_c * (Id2 ⊗ Id2)) ⊡ Id2)
    δ = ε₀ - S_hom ⊡ (C_s ⊡ ε₀)
    φ = Dict(r => Schemes.volume_fraction(rve, r) for r in fluid)
    g = Dict(r => Id2 ⊡ MFH_Core.gradient_flux_concentration(ct, r) for r in fluid)   # tr⟨ε⟩_r per unit Σ
    a = Dict(r => Id2 ⊡ ε₀ - g[r] ⊡ (C_s ⊡ ε₀) for r in fluid)                         # tr⟨ε⟩_r per unit pressure
    Φ = sum(values(φ))
    # The common pressure q = h : Σ that conserves the fluid mass.
    h = -k_f * z * sum(φ[r] * g[r] for r in fluid) / (Φ + k_f * z * sum(φ[r] * a[r] for r in fluid))
    S = S_hom + δ ⊗ h
    pressure = Dict(r => k_c * g[r] - (1 - k_c * a[r]) * h for r in fluid)
    return (compliance = S, stiffness = inv(S), pressure = pressure)
end

# The pore families replaced by a fluid of bulk modulus k (no shear stiffness).
_with_fluid_modulus(rve, fluid, k, property) = foldl(fluid; init = rve) do cell, r
    Schemes.cell_set_property(cell, r, property, TensISO{3}(3k, zero(k)))
end

function _check_squirt_phases(rve, solid, fluid)
    isempty(fluid) && throw(ArgumentError("squirt_flow: list at least one fluid-filled pore family in `fluid`"))
    for r in fluid
        r == solid && throw(ArgumentError("squirt_flow: the solid :$(solid) cannot be a pore family"))
        haskey(rve.phases, r) || throw(ArgumentError("squirt_flow: this RVE has no phase :$(r); its phases are $(rve.phase_names)"))
    end
    others = setdiff(rve.phase_names, fluid, (solid,))
    isempty(others) || throw(
        ArgumentError(
            "squirt_flow: the solid must be homogeneous, and the phases $(others) are " *
                "neither the solid :$(solid) nor a pore family of `fluid`"
        )
    )
    return nothing
end
