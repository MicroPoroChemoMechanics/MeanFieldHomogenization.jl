# [The Eshelby inclusion problem](@id th-eshelby-problem)

Everything in this section rests on one result. This page states it, and defines
the three tensors it produces — ``\mathbb{P}``, ``\mathbb{S}``, ``\mathbb{Q}``.
Their closed forms are the subject of the next page,
[Hill polarization tensors](hill_tensors.md).

## The problem

Fill ``\mathbb{R}^3`` with a homogeneous linear elastic medium of stiffness
``\mathbb{C}``, and single out an ellipsoid ``\mathcal{E}_{\boldsymbol{A}}``
centered at the origin (shape tensor ``\boldsymbol{A}``, semi-axes
``a\ge b\ge c``; see [Notation](notation.md#Ellipsoid-geometry)). Prescribe a
uniform **polarization stress** ``\boldsymbol{\tau}`` inside the ellipsoid and
zero outside, with no remote loading.

![The inclusion problem: a uniform polarization inside the ellipsoid, no remote loading (from the Echoes book [echoes](@cite))](../assets/geometry/eshelby_inclusion.png)

[eshelby1957](@cite) showed that the resulting strain field is **uniform inside
the ellipsoid**. This is the whole reason mean-field homogenization works, and
it is specific to the ellipsoid: no other shape has it.

```math
\forall\,\underline{x}\in\mathcal{E}_{\boldsymbol{A}},\qquad
\boldsymbol{\varepsilon}(\underline{x}) = -\,\mathbb{P}:\boldsymbol{\tau}.
```

``\mathbb{P} = \mathbb{P}(\boldsymbol{A},\mathbb{C})`` is the **Hill
polarization tensor** [hill1963](@cite), [willis1977](@cite). It depends only on
the ellipsoid's shape and orientation (through ``\boldsymbol{A}``) and on the
reference medium (through ``\mathbb{C}``) — in particular it is **independent of
the ellipsoid's size**.

## Three tensors, one contraction apart

The same solution is written three ways in the literature. Knowing which is
which avoids most confusion when comparing formulas across papers.

**Eshelby tensor ``\mathbb{S}``.** Introduce the equivalent *eigenstrain*
(stress-free strain) ``\boldsymbol{\varepsilon}^{\star} =
-\mathbb{C}^{-1}:\boldsymbol{\tau}``. Then

```math
\boldsymbol{\varepsilon}(\underline{x}) = \mathbb{S}:\boldsymbol{\varepsilon}^{\star},
\qquad
\boxed{\;\mathbb{S} = \mathbb{P}:\mathbb{C}\;}
```

which is Eshelby's original form. ``\mathbb{S}`` is dimensionless;
``\mathbb{P}`` has the dimension of a compliance.

**Second Hill tensor ``\mathbb{Q}``.** Asking for the *stress* inside the
inclusion rather than the strain gives the dual statement

```math
\boldsymbol{\sigma}(\underline{x}) = -\,\mathbb{Q}:\boldsymbol{\varepsilon}^{\star},
\qquad
\boxed{\;\mathbb{Q} = \mathbb{C} - \mathbb{C}:\mathbb{P}:\mathbb{C}\;}
```

``\mathbb{Q}`` is what degenerates in a controlled way when the inclusion
becomes flat, which is why the crack theory is built on it rather than on
``\mathbb{P}`` — see [Crack opening displacement](cod_tensors.md).

In `MeanFieldHomogenization`, ``\mathbb{P}`` and ``\mathbb{S}`` are [`hill_tensor`](@ref)
and [`eshelby_tensor`](@ref). There is no public accessor for ``\mathbb{Q}``:
assemble it from ``\mathbb{P}`` when you need it,

```julia
P = hill_tensor(inclusion, C₀)
Q = C₀ - C₀ ⊡ P ⊡ C₀
```

which is all the crack machinery does internally before taking the flat limit.

## The transport counterpart

Replace elasticity by a scalar diffusion problem — heat conduction, mass
diffusion, electric conduction, Darcy flow. The unknown is a scalar potential
``T``, and the reference property is an order-2 conductivity
``\boldsymbol{K}``. Fourier's law carries a minus sign that Hooke's does not,
so the stress analog is taken to be **minus** the flux,

```math
\boldsymbol{\sigma} \equiv -\,\underline{q} = \boldsymbol{K}\cdot\nabla T ,
```

the convention fixed in
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q) —
``\boldsymbol{\sigma}\cdot\underline{n}`` is then, in both theories, what the
exterior transmits to the interior across a surface. With that substitution
Eshelby's uniformity result holds *literally*: a uniform polarization
``\underline{\tau}_q`` inside the ellipsoid produces a **uniform gradient**
inside it, and an order-2 Hill tensor
``\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K})`` plays the role of
``\mathbb{P}`` [willis1977](@cite):

```math
\nabla T(\underline{x}) = -\,\boldsymbol{P}\cdot\underline{\tau}_q,
\qquad
\boldsymbol{s} = \boldsymbol{P}\cdot\boldsymbol{K},
\qquad
\boldsymbol{Q} = \boldsymbol{K} - \boldsymbol{K}\cdot\boldsymbol{P}\cdot\boldsymbol{K},
```

which is the elastic ``\boldsymbol{\varepsilon} = -\mathbb{P}:\boldsymbol{\tau}``
above with ``(\boldsymbol{\varepsilon},\mathbb{P},\boldsymbol{\tau})`` replaced
by ``(\nabla T,\boldsymbol{P},\underline{\tau}_q)`` — no sign changed.

The two problems are handled by the same functions in `MeanFieldHomogenization`, which
dispatch on the order of the property tensor passed in: an order-4
``\mathbb{C}`` selects the elastic path, an order-2 ``\boldsymbol{K}`` the
transport one. That single implementation is possible *because* of the
convention above.

See [Conduction and diffusion](@ref man-conductivity) for the call.

## Why this matters for a real material

A real heterogeneous material is not one ellipsoid in an infinite medium. What
comes closest is the **inhomogeneity** problem — an ellipsoid of a *different*
stiffness ``\mathbb{C}^I``, loaded remotely by ``\underline{u} = \boldsymbol{E}\cdot\underline{x}``
— and it reduces to the inclusion problem above by the equivalent-polarization
argument of [Localization](localization.md):

![The inhomogeneity problem: a different stiffness inside, remote loading outside (from the Echoes book [echoes](@cite))](../assets/geometry/eshelby_inhomogeneity.png)

The step from there to an estimate of effective properties is the subject of the
next two pages, and it has two parts:

1. each inclusion is treated as if it were alone in an infinite *reference*
   medium — this is what makes ``\mathbb{P}`` usable, and it is exactly the
   approximation that distinguishes one mean-field scheme from another
   ([Homogenization schemes](homogenization.md));
2. the choice of that reference medium is the scheme: the matrix itself
   (Mori–Tanaka), the effective medium being sought (self-consistent), or
   something in between ([Localization](localization.md)).

