# [A cracked rock under confinement](@id mp-ex-closing-cracks)

This example runs the [nonlinear poroelastic law of a cracked rock](@ref mp-theory-closing-cracks):
a rock whose microcracks close as it is confined, so that its stiffness rises and
its Biot coefficient falls. It then shows the two roles of the pore pressure,
which reopens the cracks through the Terzaghi effective stress but deforms the
rock through the Biot one.

## The rock

A solid of bulk modulus 40 GPa and shear modulus 25 GPa holds three families of
microcracks, normal to the three axes, each of density 0.1. Three orthogonal
families are the simplest sampling of the orientations that keeps the rock
symmetric under a hydrostatic loading. Their closure stresses are distributed
exponentially, with a mean of 20 MPa.

```@example mp_ex_closing
using MeanFieldHomogenization, TensND, Plots

C_s = TensISO{3}(3 * 40e3, 2 * 25e3)                                  # MPa
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C_s); fraction = :rest)
add_phase!(rve, :F3, PennyCrack(1.0), Dict(:C => C_s); density = 0.1)                          # normal e₃
add_phase!(rve, :F1, PennyCrack(1.0; euler_angles = (π / 2, 0.0)), Dict(:C => C_s); density = 0.1)      # normal e₁
add_phase!(rve, :F2, PennyCrack(1.0; euler_angles = (π / 2, π / 2)), Dict(:C => C_s); density = 0.1)    # normal e₂
rock = ClosingCrackPoroelasticity(rve; closure = ExponentialClosure(20.0), porosity = 0.005)
nothing # hide
```

| code | theory |
|:--|:--|
| `rve` | the solid and its crack families, at their densities with every crack open |
| `closure` | the open fraction ``f(\sigma_n)`` of a family under the normal effective traction |
| `porosity` | the porosity ``\phi`` occupied by the fluid, which sets the Biot modulus |

## Drained confinement

The rock is confined to 80 MPa with the pore pressure held at zero, so that the
effective stress is the total stress:

```@example mp_ex_closing
P_max = 80.0
confine = LoadingPath(range(0, 1; length = 41); σ11 = t -> -P_max * t, σ22 = t -> -P_max * t,
    σ33 = t -> -P_max * t, σ23 = 0, σ13 = 0, σ12 = 0, p = 0)
r = point_response(rock, confine)
nothing # hide
```

The moduli at each state are read on the consistent tangent of a step that does
not move. Its blocks are the drained stiffness ``\mathbb{C}^{\mathrm{hom}}`` and
the Biot tensor ``\boldsymbol{B}``. Their isotropic parts give the drained bulk
modulus ``K`` and the Biot coefficient ``b``:

```@example mp_ex_closing
function moduli(law, state)
    _, 𝕂, _ = point_update(law, state.g, state, 1.0)
    K = sum(𝕂[1:3, 1:3]) / 9                  # (1 : C : 1) / 9
    b = sum(𝕂[7, 1:3]) / 3                    # (1 : B) / 3
    return K, b
end
P = -r[:σ11]
Kb = [moduli(rock, st) for st in r.states]
p1 = plot(P, first.(Kb) / 1e3; lw = 2, label = "", xlabel = "confining pressure (MPa)",
    ylabel = "drained bulk modulus K (GPa)")
hline!(p1, [40.0]; ls = :dash, label = "solid without cracks")
p2 = plot(P, last.(Kb); lw = 2, label = "", xlabel = "confining pressure (MPa)", ylabel = "Biot coefficient b")
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
```

At zero confinement every crack is open: the rock has the linear moduli of the
cracked solid. As the confinement grows the cracks close, the bulk modulus rises
toward that of the solid and the Biot coefficient falls. The rise is fastest at
low pressure, where the thinnest cracks close.

## The pore pressure reopens the cracks

Starting from the confined state, the total stress is now held at 80 MPa and the
pore pressure raised to 60 MPa. The effective stress falls to 20 MPa, and the
cracks that the confinement had closed reopen:

```@example mp_ex_closing
times = vcat(range(0, 1; length = 41), range(1.025, 2; length = 40))
both = LoadingPath(times; σ11 = t -> -P_max * min(t, 1), σ22 = t -> -P_max * min(t, 1),
    σ33 = t -> -P_max * min(t, 1), σ23 = 0, σ13 = 0, σ12 = 0, p = t -> 60.0 * max(t - 1, 0))
r2 = point_response(rock, both)
reopen = 41:length(r2.t)
K2 = [moduli(rock, st)[1] for st in r2.states[reopen]]
p1 = plot(P, first.(Kb) / 1e3; lw = 2, label = "drained confinement",
    xlabel = "effective pressure −Σ′ (MPa)", ylabel = "drained bulk modulus K (GPa)", legend = :bottomright)
plot!(p1, -r2[:Σ′11][reopen], K2 / 1e3; lw = 2, ls = :dash, label = "pore pressure raised")
p2 = plot(P, 3 * r[:ε11] * 1e3; lw = 2, label = "drained confinement",
    xlabel = "effective pressure −Σ′ (MPa)", ylabel = "volume change (‰)", legend = :bottomleft)
plot!(p2, -r2[:Σ′11][reopen], 3 * r2[:ε11][reopen] * 1e3; lw = 2, ls = :dash, label = "pore pressure raised")
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
```

The two plots tell the two principles of the [theory](@ref mp-theory-closing-cracks)
apart. The moduli are the same along both paths at the same effective
pressure: the open cracks, hence the stiffness and the Biot tensor, depend on
the Terzaghi effective stress alone. The deformation is not: raising the pore
pressure recovers only part of the volume that the confinement took. The strain
rate is ``\mathbb{S}^{\mathrm{hom}}:(\dot{\boldsymbol{\Sigma}} + \boldsymbol{B}\,\dot p)``,
and while most cracks are closed the Biot coefficient is small, so that the
pore pressure acts on the deformation far less than on the cracks.

## Checks

**Every crack open: the linear cracked rock.** Under a dilation no crack
closes, and the law gives the moduli of the fractured-rock material
[`FracturedPoroelasticRock`](@ref) with every family open:

```@example mp_ex_closing
fr = FracturedPoroelasticRock(rve, MoriTanaka(); ω₀ = (1e-3, 1e-3, 1e-3), porosity_ref = 0.005)
g = (ε = Mandel{Float64}(1e-4, 1e-4, 1e-4, 0, 0, 0), p = 0.0)
a = material_response(rock, g, initial_state(rock), 1.0).tangents.φp
b = material_response(fr, (ε = TensND.from_mandel(g.ε), p = g.p), initial_state(fr), 1.0).tangents.φp
a ≈ b
```

**The tangent is consistent**, every block of it:

```@example mp_ex_closing
check_material_interface(rock; verbose = false)
```

## Going further

- Families symmetrized over all orientations are refused, since their cracks do
  not see one normal traction. More oriented families sample the orientations
  more finely.
- Each family may close in its own way: `closure = (F1 = …, F2 = …, F3 = …)`,
  or any function of the normal traction.
- The law has two fields and runs in a coupled finite-element computation as the
  poroelastic laws of the [finite-element coupling](@ref fe-quickstart) do.
