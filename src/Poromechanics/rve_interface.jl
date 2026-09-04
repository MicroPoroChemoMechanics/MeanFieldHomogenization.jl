# =============================================================================
#  rve_interface.jl — the RVE-flavored convenience layer over `biot.jl`.
#
#  `biot.jl` is deliberately RVE-free: it is tensor algebra, and stays usable
#  on a `C_hom` obtained from anywhere (a laminate, a particle assembly, an
#  experiment).  The methods here only spare the caller from writing
#  `phase_property(rve, solid, :C)` by hand and from getting the porosity wrong.
# =============================================================================

"""
    biot_tensor(rve::RVE, C_hom; solid = nothing, property = :C) -> Tens{2,3}

[`biot_tensor`](@ref) with the skeleton stiffness read from the phase that
plays the homogeneous solid of microporomechanics.

`solid` names that phase. Left unset it is the one taking up the volume
complement, which is the usual reading of a porous RVE; an RVE that designates
none has to be told, because no container can know which of its phases is the
skeleton.
"""
biot_tensor(
    rve::Schemes.RVE, C_hom::TensND.AbstractTens{4, 3};
    solid::Union{Nothing, Symbol} = nothing, property::Symbol = :C
) = biot_tensor(
    C_hom,
    Schemes.phase_property(
        rve, Schemes.host_phase_name(rve, solid, "biot_tensor"), property
    )
)

"""
    poroelastic_parameters(rve::RVE, C_hom, φ; solid = nothing, property = :C) -> NamedTuple

[`poroelastic_parameters`](@ref) with the skeleton stiffness read from the
phase that plays the homogeneous solid — see [`biot_tensor`](@ref) for `solid`.
"""
poroelastic_parameters(
    rve::Schemes.RVE, C_hom::TensND.AbstractTens{4, 3}, φ::Number;
    solid::Union{Nothing, Symbol} = nothing, property::Symbol = :C
) = poroelastic_parameters(
    C_hom,
    Schemes.phase_property(
        rve, Schemes.host_phase_name(rve, solid, "poroelastic_parameters"), property
    ),
    φ
)

"""
    pore_volume_fraction(rve::RVE, names) -> Number

Sum of the volume fractions of the phases listed in `names` — the Lagrangian
porosity of the connected pore space, for use as the `φ` argument of
[`inverse_biot_modulus`](@ref MeanFieldHomogenization.Poromechanics.inverse_biot_modulus) and [`poroelastic_parameters`](@ref).

`names` is any iterable of phase symbols; a single `Symbol` is accepted too.
Passing the pore phases explicitly is deliberate: an RVE has no way of knowing
which of its soft inclusions are fluid-filled and connected, and guessing would
silently produce a wrong Biot modulus.

!!! warning "Crack phases carry no volume"
    A [`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity) phase
    contributes **zero** here, because a flat crack has no volume: its fraction
    ``f_i = (4\\pi/3)\\,d_i\\,\\omega_i`` depends on the aspect ratio, which the
    crack geometry does not carry and which evolves during a simulation. For a
    fractured medium the porosity must therefore be assembled from the current
    apertures — that is what
    [`MeanFieldHomogenization.Constitutive`](@ref MeanFieldHomogenization.Constitutive)
    does — and passed to [`poroelastic_parameters`](@ref) directly.
"""
function pore_volume_fraction(rve::Schemes.RVE{T}, names) where {T}
    φ = zero(T)
    for name in names
        φ = φ + Schemes.volume_fraction(rve, name)
    end
    return φ
end

pore_volume_fraction(rve::Schemes.RVE, name::Symbol) =
    pore_volume_fraction(rve, (name,))
