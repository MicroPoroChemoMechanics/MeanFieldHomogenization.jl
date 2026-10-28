```@meta
EditURL = "../../../../scripts/material_laws/15_crack_damage.jl"
```

# [Damage by the growth of microcracks](@id mp-theory-crack-damage)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_laws/15_crack_damage.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_laws/15_crack_damage.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_laws/15_crack_damage.jl`, runs it end to end.

In most damage models, a scalar variable degrades the stiffness through a
function chosen to fit tests. This page derives a damage law from the
microstructure instead. The variables are the densities of the families of
microcracks that the solid contains, the stiffness is the homogenized stiffness
of the cracked solid, and the evolution follows from an energy balance on the
cracks. The approach is that of [dormieuxKondo2016book](@citet), applied by
[dormieuxKondoUlm2006damage](@citet) to fluid-saturated media.
[The example below](@ref mp-ex-crack-damage) runs the law.

## State and energy

A linear elastic solid of stiffness ``\mathbb{C}_0`` contains ``n`` families of
penny-shaped cracks, family ``i`` having the density ``d_i`` and a given
orientation (or an isotropic distribution of orientations). The cracks are open
and their faces carry no traction. At fixed densities the cracked solid is
linear elastic, with the stiffness that a scheme of the package gives,
``\mathbb{C}^{\mathrm{hom}}(d_1, \ldots, d_n)``. Its free energy per unit volume is

```math
W(\boldsymbol{\varepsilon}, d_1, \ldots, d_n)
= \frac12\,\boldsymbol{\varepsilon}:\mathbb{C}^{\mathrm{hom}}(d_1, \ldots, d_n):\boldsymbol{\varepsilon}.
```

The stress and the force conjugate to each density follow from it:

```math
\boldsymbol{\sigma} = \frac{\partial W}{\partial\boldsymbol{\varepsilon}}
= \mathbb{C}^{\mathrm{hom}}:\boldsymbol{\varepsilon},
\qquad
Y_i = -\frac{\partial W}{\partial d_i}
= -\frac12\,\boldsymbol{\varepsilon}:\frac{\partial\mathbb{C}^{\mathrm{hom}}}{\partial d_i}:\boldsymbol{\varepsilon}.
```

``Y_i`` is the energy released per unit increase of the density ``d_i`` at
fixed strain: the force that drives the growth of the family. The dissipation
is ``\sum_i Y_i\,\dot d_i``, and it is nonnegative because cracks only grow
(``\dot d_i \ge 0``) and because a crack can only soften the solid
(``Y_i \ge 0``).

## What drives a family

When the compliance of the cracked solid is affine in the densities,
``\mathbb{S}^{\mathrm{hom}} = \mathbb{S}_0 + \sum_i d_i\,\mathbb{H}_i``, as with the
Mori–Tanaka scheme, the derivative of the stiffness is
``\partial\mathbb{C}^{\mathrm{hom}}/\partial d_i = -\mathbb{C}^{\mathrm{hom}}:\mathbb{H}_i:\mathbb{C}^{\mathrm{hom}}``.
The driving force then takes a form that shows what drives it:

```math
Y_i = \frac12\,\boldsymbol{\sigma}:\mathbb{H}_i:\boldsymbol{\sigma}.
```

``\mathbb{H}_i`` acts on the traction ``\boldsymbol{\sigma}\cdot\underline{n}_i``
that the stress applies to the plane of the family. A family is driven by the
normal and the shear tractions on its own plane, and a family whose plane
carries no traction does not grow, however large the stress. This is what makes
the damage anisotropic, as the families loaded differently grow differently.
The law itself does not rely on this form: it differentiates
``\mathbb{C}^{\mathrm{hom}}`` by automatic differentiation through the scheme,
whatever the scheme.

## The criterion and the resistance

A family grows when its driving force reaches a resistance ``R_i(d_i)``:

```math
\dot d_i \ge 0,
\qquad
Y_i - R_i(d_i) \le 0,
\qquad
\dot d_i\,\big(Y_i - R_i(d_i)\big) = 0 .
```

These are the conditions of a rate-independent process. Either the family
keeps its density, below the criterion, or it grows while staying on it.

The resistance follows from the fracture of the cracks themselves when the
family is ``\mathcal{N}`` cracks per unit volume that grow in size at fixed
number. With ``a`` their radius, the density is ``d = \mathcal{N} a^3``. Griffith's
criterion [griffith1921](@cite) states that a crack front advances when the
energy released per unit area of new crack surface reaches the toughness
``G_c``. When the radius of every crack grows by ``\mathrm{d}a``, the crack surface
created per unit volume is ``\mathcal{N}\,2\pi a\,\mathrm{d}a``, and it costs the
energy ``\mathcal{N}\,G_c\,2\pi a\,\mathrm{d}a``. The density grows by
``\mathrm{d}d = 3\mathcal{N} a^2\,\mathrm{d}a``, which releases
``Y\,\mathrm{d}d``. Equating the two:

```math
Y = \frac{2\pi G_c}{3a} = R(d),
\qquad
R(d) = \frac{2\pi G_c}{3a_0}\left(\frac{d_0}{d}\right)^{1/3},
```

``a_0`` and ``d_0`` being the radius and the density in a reference state, which
fix ``\mathcal{N} = d_0/a_0^3``. This is [`CrackGrowthResistance`](@ref). Any
other function of the density can be given instead, for instance a resistance
that increases with the density to represent obstacles that the cracks meet as
they grow.

## Stability

Under a fixed strain, a family on its criterion keeps growing if a small
increase of its density raises the driving force more than the resistance, that
is, if ``\partial Y_i/\partial d_i - R_i'(d_i) > 0`` at fixed strain. The growth is
then unstable. With Griffith's resistance ``R'`` is negative: once the
criterion is reached, the density jumps to a larger value, the first one where
the driving force has fallen back to the resistance, and the stress drops. A
resistance that increases fast enough makes the growth stable and the damage
progressive.

An unstable growth is also a numerical difficulty. Newton's method, started from
the densities of the last step, cannot jump to a distant root. A small viscosity
regularizes it:

```math
\dot d_i = \frac{1}{\tau}\,\Big\langle \frac{Y_i}{R_i(d_i)} - 1 \Big\rangle,
\qquad \langle x\rangle = \max(x, 0).
```

Above the criterion the density now grows at a finite rate, so a step short
against ``\tau`` changes it little. As ``\tau`` becomes small against the duration
of the loading, the response tends to the rate-independent one, jump included.
The regularization is the keyword `τ` of [`CrackGrowthDamage`](@ref).

## The law in `MaterialLaws`

[`CrackGrowthDamage`](@ref) builds a [`ResidualLaw`](@ref) whose internal
variables are the densities. Over a step from ``d_{i,n}`` it solves, for every
family, one of the two following equations:

```math
\min\Big(d_i - d_{i,n},\ 1 - \frac{Y_i}{R_i(d_i)}\Big) = 0
\quad\text{(rate independent)},
\qquad
d_i - d_{i,n} - \frac{\Delta t}{\tau}\,\Big\langle \frac{Y_i}{R_i(d_i)} - 1\Big\rangle = 0
\quad\text{(viscous)}.
```

The first is the complementarity of the growth and of the criterion: its root
either keeps the density or puts the family on its criterion. Both terms are
dimensionless, so that they compare. A step is elastic, and the densities are
kept, when every family is below its criterion at the densities of the last
step.

The consistent tangent comes, as for any law of the chapter, from the
implicit-function theorem applied to these equations. It contains the softening
due to the growth of the cracks during the step. A density must stay positive;
a Newton correction that leads to a negative density is halved until it does
not.

## Scope

| | |
|:--|:--|
| cracks | penny-shaped, open, with traction-free faces: no unilateral contact, no friction |
| families | any number, oriented or isotropically distributed; each its own resistance |
| schemes | any linear scheme of the package that handles cracks |
| softening | a softening law localizes in a structure; the response of a finite-element computation then depends on the mesh, which a point computation does not show |

## [Example: damage by the growth of microcracks](@id mp-ex-crack-damage)

This example runs the [damage law derived from the microcracks](@ref mp-theory-crack-damage)
of a brittle solid. It compares a progressive damage with the unstable growth
that Griffith's criterion gives, and it shows how a viscosity regularizes the
latter.

### The cracked solid

A rock-like solid (bulk modulus 30 GPa, shear modulus 20 GPa) contains
penny-shaped microcracks of all orientations, of initial density
``d_0 = 0.05``. The stiffness of the cracked solid is the Mori–Tanaka one, as for
a linear computation:

````@example mp_crack_damage
using MeanFieldHomogenization, TensND, Plots

k, μ = 30e3, 20e3                                                 # MPa
C₀ = TensISO{3}(3k, 2μ)
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C₀); density = 0.05, symmetrize = :iso)
nothing #hide
````

`symmetrize = :iso` spreads the family over every orientation, so that the
solid stays isotropic as the cracks grow.

### Progressive damage

A first resistance increases with the density, as if the cracks met obstacles
while they grow:

````@example mp_crack_damage
rising(d) = 0.04 * (1 + 20 * (d - 0.05))                          # MPa
progressive = CrackGrowthDamage(rve; resistance = rising)
nothing #hide
````

| code | theory |
|:--|:--|
| `rve` | the cracked solid; the density of the crack phase is the initial one |
| `resistance` | the resistance ``R(d)`` that the driving force ``Y`` must reach |
| `CrackGrowthDamage` | the law whose internal variable is the density of each family |

The bar is pulled, unloaded to zero strain, then pulled further. Its lateral
faces are free.

````@example mp_crack_damage
ε_axial(t) = t < 0.5 ? 3e-3 * t / 0.5 : t < 0.7 ? 3e-3 * (0.7 - t) / 0.2 : 4e-3 * (t - 0.7) / 0.3
r = point_response(progressive, LoadingPath(range(0, 1; length = 61); ε11 = ε_axial))

p1 = plot(r[:ε11] * 1e3, r[:σ11]; lw = 2, label = "", xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)")
p2 = plot(r[:ε11] * 1e3, r[:F]; lw = 2, label = "", xlabel = "axial strain (‰)", ylabel = "crack density d")
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
````

The response is linear until the driving force reaches the resistance, near
1 ‰. The cracks then grow, and the stress bends under its linear trend. On
unloading the density is kept: the stress returns to the origin along a straight
line, whose slope is the stiffness of the more cracked solid. On reloading, the
density grows again only past the largest strain reached before. Nothing in this
response was fitted except the resistance. The loss of stiffness is what the
homogenization gives for the density reached.

### Unstable growth, regularized

Griffith's resistance, for cracks of initial radius 1 mm and a toughness of
20 J/m², decreases as the cracks grow:

````@example mp_crack_damage
griffith = CrackGrowthResistance(; Gc = 0.02, a₀ = 1.0, d₀ = 0.05)   # N/mm, mm
nothing #hide
````

The growth is then unstable: at the threshold the density jumps. A
rate-independent law cannot follow the jump from the density of the last step,
so the growth is regularized by a time ``\tau``, compared here with the duration
of the loading (one unit of time):

````@example mp_crack_damage
path = LoadingPath(range(0, 1; length = 41); ε11 = t -> 2e-3 * t)
p1 = plot(; xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)", legend = :bottomright)
p2 = plot(; xlabel = "axial strain (‰)", ylabel = "crack density d", legend = :topleft)
for τ in (1.0, 0.1, 0.01)
    rτ = point_response(CrackGrowthDamage(rve; resistance = griffith, τ), path)
    plot!(p1, rτ[:ε11] * 1e3, rτ[:σ11]; lw = 2, label = "τ = $τ")
    plot!(p2, rτ[:ε11] * 1e3, rτ[:F]; lw = 2, label = "τ = $τ")
end
plot(p1, p2; layout = (1, 2), size = (760, 320), left_margin = 3Plots.mm, bottom_margin = 4Plots.mm)
````

With the largest ``\tau`` the cracks grow slowly and the stress keeps rising for
a while past the threshold. As ``\tau`` decreases, the curves approach the
rate-independent response: the density jumps at the threshold, and the stress
drops at once to the value that the more cracked solid carries. The peak stress
is the same for ``\tau = 0.1`` and ``\tau = 0.01``, the threshold of the
rate-independent law; the response after the peak still moves with ``\tau``,
and a smaller ``\tau`` needs shorter steps, which the driver takes by cutting
those that fail.

### Checks

**The tangent is consistent.** The contract checker compares the tangent from
the implicit-function theorem with the derivative of the stress, on a damaging
step:

````@example mp_crack_damage
check_material_interface(progressive; verbose = false)
````

**Below the threshold, the law is the linear cracked solid.** Its tangent is
then the Mori–Tanaka stiffness at the initial density:

````@example mp_crack_damage
σ, 𝕂, _ = point_update(progressive, Mandel{Float64}(2e-4, -1e-4, 0, 1e-4, 0, 0), initial_state(progressive), 1.0)
𝕂 ≈ KM(homogenize(rve, MoriTanaka(), :C))
````

### Going further

- Oriented families (cracks with a given normal, without `symmetrize`) grow
  each under the tractions on their own plane, so that the damage becomes
  anisotropic. Each family may have its own resistance:
  `resistance = (F1 = R₁, F2 = R₂)`.
- The law is a law of `MaterialLaws` and runs at the quadrature points of a
  [finite-element computation](@ref fe-quickstart). A softening law localizes
  in a structure, and the finite-element response then depends on the mesh.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

