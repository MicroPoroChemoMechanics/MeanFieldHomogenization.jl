# [Integrating a law over a step](@id mp-theory)

A nonlinear law is not a formula that turns a strain into a stress: the stress
also depends on the history, carried by internal variables that evolve. A
finite-element code, or a test machine, asks the law a discrete question: *the
gradients were* ``\mathsf{g}_n`` *at the beginning of a time step of length*
``\Delta t``*, the internal variables were* ``\mathsf{y}_n``*; the gradients are*
``\mathsf{g}`` *at its end; what are the fluxes, the internal variables, and how
do the fluxes vary with* ``\mathsf{g}``*?* This page derives the answer the code
computes, and why it is exact.

## The objects of a step

A law has three kinds of variables, which the code stores as flat vectors (in
sans-serif in this page):

| | meaning | example (poroelastoplasticity) |
|:--|:--|:--|
| ``\mathsf{g}`` | the **gradients**: the primal fields at the point | strain ``\boldsymbol{\varepsilon}`` and pore pressure ``p`` |
| ``\mathsf{F}`` | the **fluxes** conjugate to them | stress ``\boldsymbol{\sigma}`` and change of fluid content ``\phi`` |
| ``\mathsf{y}`` | the **internal variables**: what the point remembers | plastic strain ``\boldsymbol{\varepsilon}^p``, cumulated plastic strain ``p`` |

A symmetric second-order tensor enters these vectors by its six Kelvin–Mandel
components ``(v_{11}, v_{22}, v_{33}, \sqrt2 v_{23}, \sqrt2 v_{13}, \sqrt2 v_{12})``.
The factor ``\sqrt2`` makes the basis orthonormal: the double contraction
``\boldsymbol{\sigma}:\boldsymbol{\varepsilon}`` is the dot product of the two
vectors, and a fourth-order tensor with the minor symmetries acts on them as a
``6\times6`` matrix. Nothing in a law has to remember a factor 2 on the shears.

## The residual route

Integrating the evolution laws of the internal variables over the step by an
implicit scheme (backward Euler in all the laws shipped) gives as many
equations as internal variables, written as a **residual**

```math
\mathsf{R}(\mathsf{y};\ \mathsf{g},\ \mathsf{y}_n,\ \Delta t) = \mathsf{0},
\qquad
\mathsf{F} = \mathsf{F}(\mathsf{g},\ \mathsf{y}).
```

For von Mises plasticity, for instance, ``\mathsf{y} = (\boldsymbol{\varepsilon}^p, p)``
and the residual collects the discrete normality rule and the consistency
condition (see [`J2Plasticity`](@ref)):

```math
\boldsymbol{\varepsilon}^p - \boldsymbol{\varepsilon}^p_n - (p - p_n)\,\frac32\,\frac{\operatorname{dev}\boldsymbol{\xi}}{\sigma_{eq}(\boldsymbol{\xi})} = \boldsymbol{0},
\qquad
\sigma_{eq}(\boldsymbol{\xi}) - R(p) = 0,
\qquad
\boldsymbol{\xi} = \mathbb{C}:(\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^p) - \boldsymbol{X}.
```

This is the standard form of the implicit integration of inelastic laws
[simo1998](@cite). The residual is solved by Newton's method, with the
Jacobian ``\partial\mathsf{R}/\partial\mathsf{y}`` obtained by automatic
differentiation of the residual the user wrote. Convergence is judged on the
size of the Newton correction, which has the units of ``\mathsf{y}`` whatever the
units of the residual.

### The elastic predictor

A rate-independent law has two regimes. If the stress computed with the
internal variables frozen at ``\mathsf{y}_n`` (the *trial* state) is admissible,
the step is elastic and ``\mathsf{y} = \mathsf{y}_n``; only otherwise is the
residual solved. A law gives this test as its `elastic` function; a
damage law gives its threshold the same way. On an elastic step the internal
variables do not depend on ``\mathsf{g}``, which the tangent below uses.

## The consistent tangent

A global Newton method on the structure needs the derivative of the fluxes at
the end of the step with respect to the gradients at the end of the step,
**through the integration algorithm**: ``\mathsf{y}`` is itself a function of
``\mathsf{g}``, defined implicitly by ``\mathsf{R} = \mathsf{0}``. Differentiating
``\mathsf{R}(\mathsf{y}(\mathsf{g}), \mathsf{g}) = \mathsf{0}``,

```math
\frac{\partial\mathsf{R}}{\partial\mathsf{y}}\,\frac{\mathrm{d}\mathsf{y}}{\mathrm{d}\mathsf{g}}
+ \frac{\partial\mathsf{R}}{\partial\mathsf{g}} = \mathsf{0}
\quad\Longrightarrow\quad
\frac{\mathrm{d}\mathsf{y}}{\mathrm{d}\mathsf{g}}
= -\left(\frac{\partial\mathsf{R}}{\partial\mathsf{y}}\right)^{-1}\frac{\partial\mathsf{R}}{\partial\mathsf{g}},
```

and the chain rule on ``\mathsf{F}(\mathsf{g}, \mathsf{y}(\mathsf{g}))`` gives the
tangent

```math
\mathsf{K} = \frac{\mathrm{d}\mathsf{F}}{\mathrm{d}\mathsf{g}}
= \frac{\partial\mathsf{F}}{\partial\mathsf{g}}
- \frac{\partial\mathsf{F}}{\partial\mathsf{y}}
\left(\frac{\partial\mathsf{R}}{\partial\mathsf{y}}\right)^{-1}
\frac{\partial\mathsf{R}}{\partial\mathsf{g}} .
```

On an elastic step the second term vanishes. This is the tangent *consistent*
with the integration algorithm, the one that keeps the quadratic convergence
of the structural Newton method [simo1985](@cite); the tangent of the
continuous law (the "continuum" tangent) does not. The four partial
derivatives are again computed by automatic differentiation, so a law never
contains a derivative written by hand.

For a law of several fields the formula is unchanged: ``\mathsf{K}`` is the full
matrix of the derivatives of every flux with respect to every gradient. Its
blocks are the coupling terms a coupled finite-element formulation assembles,
``\partial\boldsymbol{\sigma}/\partial\boldsymbol{\varepsilon}``,
``\partial\boldsymbol{\sigma}/\partial p``, ``\partial\phi/\partial\boldsymbol{\varepsilon}``,
``\partial\phi/\partial p`` for poroelasticity.

## The potential route

Many laws derive from two potentials, in the framework of generalized standard
materials [germain1983](@cite): a free energy ``\psi(\mathsf{g}, \boldsymbol{\alpha})``
of the gradients and of the internal variables ``\boldsymbol{\alpha}``, which
gives the fluxes, ``\mathsf{F} = \partial\psi/\partial\mathsf{g}``, and a convex
dissipation potential ``\varphi(\dot{\boldsymbol{\alpha}})``, which gives the
evolution of the internal variables through
``-\partial\psi/\partial\boldsymbol{\alpha} \in \partial\varphi(\dot{\boldsymbol{\alpha}})``.

Over a step, replace the rate by the difference quotient
``(\boldsymbol{\alpha} - \boldsymbol{\alpha}_n)/\Delta t`` and consider the
**incremental energy**

```math
W(\boldsymbol{a}) = \psi(\mathsf{g}, \boldsymbol{a})
+ \Delta t\;\varphi\!\left(\frac{\boldsymbol{a} - \boldsymbol{\alpha}_n}{\Delta t}\right).
```

Its stationarity in ``\boldsymbol{a}``,
``\partial\psi/\partial\boldsymbol{a} + \partial\varphi/\partial\dot{\boldsymbol{\alpha}} = \boldsymbol{0}``,
is exactly the backward Euler discretization of the evolution law, so the
internal variables at the end of the step minimize ``W`` when the potentials
are convex. This variational form of the update is due to
[ortiz1999](@citet). In the code it becomes a residual like any other,

```math
\mathsf{R}(\mathsf{y}) = \frac{\partial W}{\partial \boldsymbol{a}}(\mathsf{y}),
```

with its Jacobian the Hessian of ``W``: the potential route reuses the whole
residual machinery, Newton's method and the consistent tangent included, and
automatic differentiation provides the first and second derivatives of the two
potentials.

Each flux may carry a sign: a poroelastic free energy written in the strain and
the pore pressure gives the stress as ``+\partial\psi/\partial\boldsymbol{\varepsilon}``
but the change of fluid content as ``-\partial\psi/\partial p`` (see
[`BiotPoroelasticity`](@ref)).

## Derivatives with respect to the parameters, exact

Identifying a law means differentiating its response with respect to its
parameters ``\mathsf{q}`` (moduli, yield stress, viscosity). The internal
variables depend on them through ``\mathsf{R}(\mathsf{y}, \mathsf{q}) = \mathsf{0}``,
and by the same argument as for the tangent,

```math
\frac{\mathrm{d}\mathsf{y}}{\mathrm{d}\mathsf{q}}
= -\left(\frac{\partial\mathsf{R}}{\partial\mathsf{y}}\right)^{-1}\frac{\partial\mathsf{R}}{\partial\mathsf{q}} .
```

The code obtains this without forming it. It first solves
``\mathsf{R} = \mathsf{0}`` on the values alone, which gives a root ``\mathsf{y}^\star``.
It then takes one Newton step in dual numbers, from that root:

```math
\mathsf{y}^{+} = \mathsf{y}^\star - \left(\frac{\partial\mathsf{R}}{\partial\mathsf{y}}\right)^{-1}\mathsf{R}(\mathsf{y}^\star, \mathsf{q}).
```

The value of ``\mathsf{R}(\mathsf{y}^\star, \mathsf{q})`` is zero, and its derivative
with respect to ``\mathsf{q}`` is ``\partial\mathsf{R}/\partial\mathsf{q}``, since
``\mathsf{y}^\star`` carries none. The derivative of ``\mathsf{y}^{+}`` is therefore
exactly the formula above. Newton's method doubles the order of agreement at
each step, so ``k`` such steps give every derivative up to order ``2^k - 1``.
This is what nested differentiation needs: one step for a gradient, two for a
Hessian.

No tolerance is ever applied to a derivative, and no finite difference is used
anywhere. The same holds for a dual strain or a dual time step, which is how
the tests check the consistent tangent: the Jacobian of the whole step,
computed with dual numbers carried through the predictor and the Newton
iterations, equals ``\mathsf{K}``.

## Driving a point: mixed control

A material test prescribes some components by their gradient (an axial strain)
and others by their flux (free lateral faces: zero lateral stress). Let
``\mathsf{c}`` mark the components driven by their gradient. At each time the
driver looks for the free gradients such that the fluxes reach their targets
``\mathsf{F}^\mathrm{d}``, by Newton's method on

```math
\mathsf{r}_i =
\begin{cases} 0 & \text{if } \mathsf{c}_i \\ \mathsf{F}_i(\mathsf{g}) - \mathsf{F}^\mathrm{d}_i & \text{otherwise,} \end{cases}
\qquad
\mathsf{A}_{ij} =
\begin{cases} \delta_{ij} & \text{if } \mathsf{c}_i \\ \mathsf{K}_{ij} & \text{otherwise,} \end{cases}
\qquad
\mathsf{g} \leftarrow \mathsf{g} - \mathsf{A}^{-1}\mathsf{r}.
```

The consistent tangent makes this iteration converge quadratically; with a
linear hardening law a single correction is exact. A step whose local or
global iteration fails is retried as two half steps, recursively, which is
also what a finite-element driver does.
