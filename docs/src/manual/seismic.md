# [Seismic signatures](@id man-seismic)

!!! info "Before this page"
    [Cracks](@ref man-cracks) for the crack families, [Poromechanics](@ref manual-poromechanics)
    for the fluid in the pores, and the [Plane waves](@extref TensND man-plane-waves)
    of TensND, which turn a stiffness and a density into velocities.

A wave sees a homogenized medium through two quantities, its stiffness and its
mass density. This page gives the density of a cell, the waves of its
estimate, and the compliance of a set of fractures read on a crack family; the
derivations are in [Cracks seen as fractures](@ref th-fracture-compliance).

## Density and waves of a cell

The density is a scalar property `:ρ` of the phases, averaged over their
volumes by [`average_density`](@ref) (a crack family, of no volume, adds
nothing). The velocities, polarizations and Thomsen's parameters are those of
TensND, applied to the estimate:

```@example seis
using MeanFieldHomogenization, TensND

E, ν = 70.0, 0.25                                    # GPa
C_s = TensISO{3}(E / (1 - 2ν), E / (1 + ν))
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_s, :ρ => 2.7); fraction = :rest)   # g/cm³
add_phase!(rve, :cracks, PennyCrack(1.0), Dict(:C => C_s); density = 0.1)
ρ = average_density(rve)
C = homogenize(rve, MoriTanaka())
w = plane_waves(C, ρ, [0.0, 0.0, 1.0])               # km/s, along the normal to the cracks
(ρ = ρ, V = w.velocity)
```

Cracks of one normal leave the medium transversely isotropic about it; its
Thomsen parameters follow once the estimate is written in that class
(`best_fit_ti` about the normal, exact here). With the density of cracks a
symbol, the stress-driven dilute scheme gives them as functions of it, whose
slopes at zero are the first-order ones:

```@example seis
using Symbolics
@variables ε
sym = RVE()
add_phase!(sym, :M, Ellipsoid(1.0), Dict(:C => C_s, :ρ => 2.7); fraction = :rest)
add_phase!(sym, :cracks, PennyCrack(1.0), Dict(:C => C_s); density = ε)
t = thomsen_parameters(best_fit_ti(homogenize(sym, DiluteDual()), [0, 0, 1]), 2.7)
[k => Symbolics.substitute(Symbolics.derivative(t[k], ε), Dict(ε => 0)) for k in (:ε, :δ, :γ)]
```

The slope of ``\varepsilon`` is the ``8/3`` of eq. (14) of [gueguen2011](@citet), for
dry aligned cracks in the non-interacting approximation, and that of ``\gamma``,
``8(1-\nu)/(3(2-\nu)) = 8/7``, the value ``\simeq 1.1`` they give with ``\nu_0 = 0.27``
(``1.125``). The fraction printed in that equation reads ``3(1 - 2\nu_0)`` in
the denominator, which would give ``4.2`` at ``\nu_0 = 0.27``; the factor ``(2 - \nu_0)``
is the one consistent with their approximate value and with the computation.

## The compliance of a fracture set

[`fracture_compliance`](@ref) reads on the excess compliance a scheme gives a
crack family the compliance of linear slip ``\boldsymbol Z`` of the equivalent set of
fractures, with its normal and tangential parts:

```@example seis
f = fracture_compliance(rve, MoriTanaka(), :cracks)
(Z_N = f.Z_N, Z_T = f.Z_T, ratio = f.Z_N / f.Z_T)
```

The ratio ``Z_N/Z_T = 1 - \nu/2`` is that of dry flat cracks; a fluid in the cracks
lowers it, which is what the characterization of cracks from velocities reads.

Sets of fractures declared by their compliances add up on a background
([`fractured_stiffness`](@ref)); one set is the periodic laminate of the
background and its spring interface. Three orthogonal sets are read back on a
change of compliance by [`orthogonal_fracture_compliances`](@ref):

```@example seis
e = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
C_f = fractured_stiffness(C_s, e[1] => (0.004, 0.006), e[2] => (0.002, 0.001), e[3] => (0.010, 0.003))
g = orthogonal_fracture_compliances(inv(C_f) - inv(C_s), [1.0 0 0; 0 1 0; 0 0 1])
(Z_N = g.Z_N, Z_T = g.Z_T)
```

## Fluids and frequency

The fluid in the cracks changes the stiffness the wave sees: sealed at high
frequency, connected at low frequency, exchanging in between
([`pore_pressure_tensor`](@ref), [`poroelastic_parameters`](@ref) with `k_f`,
[`squirt_flow`](@ref) in [Poromechanics](@ref manual-poromechanics)). A complex
stiffness gives complex velocities, whose real part of the slowness is the
phase velocity and whose imaginary part is the attenuation.

## Where to go next

The applications of the section *Cracked and saturated rocks* follow these
calls on data of the literature: the dispersion of a cracked rock by squirt
flow ([Squirt flow in a cracked rock](@ref app-squirt-dispersion)), the cracks of
a gneiss read on its velocities ([Cracks of the KTB gneiss](@ref app-ktb-cracks))
and the pressure of the fluid in sealed pores
([Fluid-filled pores](@ref app-fluid-filled-pores)). The next chapter,
[Sensitivities](@ref man-sensitivities), differentiates any of these
properties with respect to an input of the cell.
