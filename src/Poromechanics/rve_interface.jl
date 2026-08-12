# =============================================================================
#  rve_interface.jl — the RVE-flavored convenience layer over `biot.jl`.
#
#  `biot.jl` is deliberately RVE-free: it is tensor algebra, and stays usable
#  on a `C_hom` obtained from anywhere (a laminate, a particle assembly, an
#  experiment).  The methods here only spare the caller from writing
#  `matrix_property(rve, :C)` by hand and from getting the porosity wrong.
# =============================================================================

"""
    biot_tensor(rve::RVE, C_hom; property = :C) -> Tens{2,3}

[`biot_tensor`](@ref) with the solid stiffness read from the RVE's **matrix**
phase, which is what plays the role of the homogeneous solid phase in
microporomechanics.

Equivalent to `biot_tensor(C_hom, matrix_property(rve, property))`.
"""
biot_tensor(
    rve::Schemes.RVE, C_hom::TensND.AbstractTens{4, 3}; property::Symbol = :C
) = biot_tensor(C_hom, Schemes.matrix_property(rve, property))

"""
    poroelastic_parameters(rve::RVE, C_hom, φ; property = :C) -> NamedTuple

[`poroelastic_parameters`](@ref) with the solid stiffness read from the RVE's
matrix phase.
"""
poroelastic_parameters(
    rve::Schemes.RVE, C_hom::TensND.AbstractTens{4, 3}, φ::Number;
    property::Symbol = :C
) = poroelastic_parameters(C_hom, Schemes.matrix_property(rve, property), φ)

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
