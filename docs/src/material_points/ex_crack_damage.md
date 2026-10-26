# [Damage by the growth of microcracks](@id mp-ex-crack-damage)

This example runs the [damage law derived from the microcracks](@ref mp-theory-crack-damage)
of a brittle solid. It compares a progressive damage with the unstable growth
that Griffith's criterion gives, and it shows how a viscosity regularizes the
latter.

## The cracked solid

A rock-like solid (bulk modulus 30 GPa, shear modulus 20 GPa) contains
penny-shaped microcracks of all orientations, of initial density
``d_0 = 0.05``. The stiffness of the cracked solid is the Mori–Tanaka one, as for
a linear computation:

```@example mp_ex_crack_damage
using MeanFieldHomogenization, TensND, Plots

k, μ = 30e3, 20e3                                                 # MPa
C₀ = TensISO{3}(3k, 2μ)
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C₀); density = 0.05, symmetrize = :iso)
nothing # hide
```

`symmetrize = :iso` spreads the family over every orientation, so that the
solid stays isotropic as the cracks grow.

## Progressive damage

A first resistance increases with the density, as if the cracks met obstacles
while they grow:

```@example mp_ex_crack_damage
rising(d) = 0.04 * (1 + 20 * (d - 0.05))                          # MPa
progressive = CrackGrowthDamage(rve; resistance = rising)
nothing # hide
```

| code | theory |
|:--|:--|
| `rve` | the cracked solid; the density of the crack phase is the initial one |
| `resistance` | the resistance ``R(d)`` that the driving force ``Y`` must reach |
| `CrackGrowthDamage` | the law whose internal variable is the density of each family |

The bar is pulled, unloaded to zero strain, then pulled further. Its lateral
faces are free.

```@example mp_ex_crack_damage
ε_axial(t) = t < 0.5 ? 3e-3 * t / 0.5 : t < 0.7 ? 3e-3 * (0.7 - t) / 0.2 : 4e-3 * (t - 0.7) / 0.3
r = point_response(progressive, LoadingPath(range(0, 1; length = 61); ε11 = ε_axial))

p1 = plot(r[:ε11] * 1e3, r[:σ11]; lw = 2, label = "", xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)")
p2 = plot(r[:ε11] * 1e3, r[:F]; lw = 2, label = "", xlabel = "axial strain (‰)", ylabel = "crack density d")
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
```

The response is linear until the driving force reaches the resistance, near
1 ‰. The cracks then grow, and the stress bends under its linear trend. On
unloading the density is kept: the stress returns to the origin along a straight
line, whose slope is the stiffness of the more cracked solid. On reloading, the
density grows again only past the largest strain reached before. Nothing in this
response was fitted except the resistance. The loss of stiffness is what the
homogenization gives for the density reached.

## Unstable growth, regularized

Griffith's resistance, for cracks of initial radius 1 mm and a toughness of
20 J/m², decreases as the cracks grow:

```@example mp_ex_crack_damage
griffith = CrackGrowthResistance(; Gc = 0.02, a₀ = 1.0, d₀ = 0.05)   # N/mm, mm
nothing # hide
```

The growth is then unstable: at the threshold the density jumps. A
rate-independent law cannot follow the jump from the density of the last step,
so the growth is regularized by a time ``\tau``, compared here with the duration
of the loading (one unit of time):

```@example mp_ex_crack_damage
path = LoadingPath(range(0, 1; length = 41); ε11 = t -> 2e-3 * t)
p1 = plot(; xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)", legend = :bottomright)
p2 = plot(; xlabel = "axial strain (‰)", ylabel = "crack density d", legend = :topleft)
for τ in (1.0, 0.1, 0.01)
    rτ = point_response(CrackGrowthDamage(rve; resistance = griffith, τ), path)
    plot!(p1, rτ[:ε11] * 1e3, rτ[:σ11]; lw = 2, label = "τ = $τ")
    plot!(p2, rτ[:ε11] * 1e3, rτ[:F]; lw = 2, label = "τ = $τ")
end
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
```

With the largest ``\tau`` the cracks grow slowly and the stress keeps rising for
a while past the threshold. As ``\tau`` decreases, the curves approach the
rate-independent response: the density jumps at the threshold, and the stress
drops at once to the value that the more cracked solid carries. The peak stress
is the same for ``\tau = 0.1`` and ``\tau = 0.01``, the threshold of the
rate-independent law; the response after the peak still moves with ``\tau``,
and a smaller ``\tau`` needs shorter steps, which the driver takes by cutting
those that fail.

## Checks

**The tangent is consistent.** The contract checker compares the tangent from
the implicit-function theorem with the derivative of the stress, on a damaging
step:

```@example mp_ex_crack_damage
check_material_interface(progressive; verbose = false)
```

**Below the threshold, the law is the linear cracked solid.** Its tangent is
then the Mori–Tanaka stiffness at the initial density:

```@example mp_ex_crack_damage
σ, 𝕂, _ = point_update(progressive, Mandel{Float64}(2e-4, -1e-4, 0, 1e-4, 0, 0), initial_state(progressive), 1.0)
𝕂 ≈ KM(homogenize(rve, MoriTanaka(), :C))
```

## Going further

- Oriented families (cracks with a given normal, without `symmetrize`) grow
  each under the tractions on their own plane, so that the damage becomes
  anisotropic. Each family may have its own resistance:
  `resistance = (F1 = R₁, F2 = R₂)`.
- The law is a law of `MaterialPoints` and runs at the quadrature points of a
  [finite-element computation](@ref fe-quickstart). A softening law localizes
  in a structure, and the finite-element response then depends on the mesh.
