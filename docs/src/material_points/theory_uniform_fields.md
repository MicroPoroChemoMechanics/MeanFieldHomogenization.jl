# [Uniform fields: Voigt–Taylor and Reuss–Sachs](@id mp-theory-uniform-fields)

The simplest composites of nonlinear phases assume one field to be uniform:
every phase undergoes the macroscopic strain, or every phase carries the
macroscopic stress. They ignore the shapes and the arrangement of the phases.
For linear elastic phases they are the bounds of [voigt1889](@citet) and
[reuss1929](@citet), between which every estimate of the effective stiffness
lies [hill1952](@cite). For plastic phases they are the models of Taylor and
of Sachs, the extremes of the [polycrystal models](@ref mp-theory-polycrystals).
This page states them for any law of `MaterialPoints`; the
[secant methods](@ref mp-theory-secant) and the polycrystal interactions sit in
between. [The example](@ref mp-ex-uniform-fields) runs them.

## Uniform strain

Every phase ``r``, of volume fraction ``f_r``, undergoes the macroscopic strain
and integrates its own law over the step:

```math
\boldsymbol{\varepsilon}_r = \boldsymbol{E},
\qquad
\boldsymbol{\Sigma} = \sum_r f_r\,\boldsymbol{\sigma}_r\big(\boldsymbol{E}, \mathsf{y}_r\big),
```

``\mathsf{y}_r`` being the internal variables of the phase. The phases interact
only through the strain they share: each integrates independently, and the
consistent tangent of the composite is the average of theirs. A uniform strain
is a kinematically admissible field: for phases deriving from convex
potentials, it gives an upper bound of the effective potential. For linear
elastic phases this is the Voigt stiffness ``\sum_r f_r\,\mathbb{C}_r``.

## Uniform stress

Every phase carries the macroscopic stress, and the strains of the phases,
unknown, average to the macroscopic one:

```math
\boldsymbol{\sigma}_r\big(\boldsymbol{\varepsilon}_r, \mathsf{y}_r\big) = \boldsymbol{\Sigma}
\quad\text{for every } r,
\qquad
\sum_r f_r\,\boldsymbol{\varepsilon}_r = \boldsymbol{E}.
```

Under a prescribed macroscopic strain, the unknowns of a step are the
macroscopic stress, the strain of every phase and their internal variables.
They are solved together, the equilibrium of every phase with the macroscopic
stress and the mixture rule of the strains being added to the equations of the
phase laws. A uniform stress is statically admissible: for convex potentials it
gives a lower bound. For linear elastic phases this is the Reuss stiffness
``\big(\sum_r f_r\,\mathbb{S}_r\big)^{-1}``.

## As laws of `MaterialPoints`

[`UniformStrain`](@ref) and [`UniformStress`](@ref) build laws whose internal
variables gather those of the phases, renamed after them (`M_εᵖ`, …), and for
the uniform stress also the macroscopic stress (`Σ_σ`) and the strains of the
phases (`M_ε`, …). A phase whose step is elastic keeps its internal variables,
whatever the others do. The consistent tangent of the composite comes, as for
any law, from the implicit-function theorem applied to all these equations:
for the uniform stress it is the inverse of the average of the tangent
compliances of the phases, which no formula of the code writes. Every phase law
of the package fits, with internal variables (plasticity, viscosity, damage) or
with several fields, provided the phases share their fields.

## Numerical notes

| | |
|:--|:--|
| start | the Newton iteration of a uniform-stress step starts with every phase taking the macroscopic increment of the strain, a uniform-strain predictor |
| strongly nonlinear phases | inverting a phase law (a power law of small exponent) is ill-conditioned: the driver cuts the steps that fail |
| no finite tangent | a phase without a finite tangent at the current strain (a power law at zero strain) makes the tangent of the composite undefined there (`NaN`), not the update |
