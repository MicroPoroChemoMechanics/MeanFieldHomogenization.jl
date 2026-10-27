# [Damage by the growth of microcracks](@id mp-theory-crack-damage)

In most damage models, a scalar variable degrades the stiffness through a
function chosen to fit tests. This page derives a damage law from the
microstructure instead. The variables are the densities of the families of
microcracks that the solid contains, the stiffness is the homogenized stiffness
of the cracked solid, and the evolution follows from an energy balance on the
cracks. The approach is that of [dormieuxKondo2016book](@citet), applied by
[dormieuxKondoUlm2006damage](@citet) to fluid-saturated media.
[The example](@ref mp-ex-crack-damage) runs the law.

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

## The law in `MaterialPoints`

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
