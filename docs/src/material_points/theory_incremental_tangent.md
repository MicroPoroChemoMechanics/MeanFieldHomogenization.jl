# [The incremental tangent approach](@id mp-theory-incremental-tangent)

The [secant methods](@ref mp-theory-secant) need phases whose stress is a
function of the strain alone. A plastic phase has a history: its stress depends
on the path, and a secant modulus means nothing for it. The incremental approach
of [hill1965](@citet) linearizes such a phase on each step instead: over an
increment of the loading, every phase behaves as a linear material of its
tangent stiffness, and a linear scheme concentrates the increment of the
macroscopic strain in the phases. It applies to any phase law of
`MaterialPoints`, plasticity, viscosity or damage, and stands between the two
[uniform-field composites](@ref mp-theory-uniform-fields), which ignore the
morphology. [The example](@ref mp-ex-incremental-tangent) runs it.

## A step

Over a step from ``\boldsymbol{E}_n`` to ``\boldsymbol{E}``, phase ``r`` goes from the
strain ``\boldsymbol{\varepsilon}_{r,n}`` and the internal variables
``\mathsf{y}_{r,n}`` to ``\boldsymbol{\varepsilon}_r`` and ``\mathsf{y}_r``, which its own law
relates, and its consistent tangent at the end of the step,
``\mathbb{C}^{t}_r = \mathrm{d}\boldsymbol{\sigma}_r/\mathrm{d}\boldsymbol{\varepsilon}_r``,
describes how its stress responds to a change of its strain. The composite of
these tangent stiffnesses is linear, and a scheme of the package gives the
strain concentration tensors ``\mathbb{A}_r`` of its phases. The increments of
the phase strains follow from that of the macroscopic strain:

```math
\boldsymbol{\varepsilon}_r - \boldsymbol{\varepsilon}_{r,n}
= \mathbb{A}_r\big(\mathbb{C}^{t}_1, \ldots, \mathbb{C}^{t}_n\big):\big(\boldsymbol{E} - \boldsymbol{E}_n\big),
\qquad
\boldsymbol{\Sigma} = \sum_r f_r\,\boldsymbol{\sigma}_r .
```

The tangents are taken at the end of the step, where the strains are not yet
known: the concentration is a fixed point, which the package solves together
with the laws of the phases. The unknowns of a step are the strains and the
internal variables of every phase; the equations are the laws of the phases,
the concentration of the increment in every phase but one, and the mixture
rule ``\sum_r f_r\,\boldsymbol{\varepsilon}_r = \boldsymbol{E}`` for the last. The concentration
tensors satisfy ``\sum_r f_r\,\mathbb{A}_r = \mathbb{I}``, so that the mixture rule is
the concentration of the last phase written in another form.

## The isotropic projection

The tangent of a plastic phase is anisotropic, oriented by its current flow.
A scheme with an anisotropic matrix needs the Hill tensor of an anisotropic
medium at every iteration, a numerical integral. As
[doghriOuaar2003](@citet) do, the package projects every tangent on the
isotropic tensors before homogenizing:

```math
\mathbb{C}^{t}_r \;\to\; 3k^{t}_r\,\mathbb{J} + 2\mu^{t}_r\,\mathbb{K},
\qquad
k^{t}_r = \tfrac13\,\mathbb{J}::\mathbb{C}^{t}_r ,
\qquad
\mu^{t}_r = \tfrac1{10}\,\mathbb{K}::\mathbb{C}^{t}_r ,
```

which is the average of the tangent over all the rotations of space. The Hill
tensors of an isotropic matrix are in closed form, and the projection of the
tangent of an isotropic elastic phase is the phase itself.

## The composite as a law

[`IncrementalTangent`](@ref) builds a law of `MaterialPoints` whose internal
variables are the strains of the phases (`M_ε`, …) and their internal
variables (`M_εᵖ`, …). Its consistent tangent comes from the
implicit-function theorem applied to all the equations of the step, the
variation of the concentration tensors with the tangents included. With
elastic phases it is the linear estimate of the scheme, exactly. With plastic
phases its response lies between the [uniform stress and the uniform
strain](@ref mp-theory-uniform-fields), and depends on the morphology through
the scheme.

## Scope

| | |
|:--|:--|
| phases | any laws of the strain and the stress, with internal variables; one law per phase of the cell |
| schemes | any linear scheme of the package; the phases are projected on isotropic tensors |
| cost | about 1 ms per step for two phases and the Mori–Tanaka scheme |
| not covered | the anisotropic tangent (without projection), and the other linearizations of the literature, which give different estimates: the [affine formulation](@ref mp-theory-affine) for laws of the strain alone, the second-order one |
