# [The affine formulation](@id mp-theory-affine)

The [secant methods](@ref mp-theory-secant) replace each nonlinear phase by a
linear one of the same stress at the same strain, without prestress: a line
through the origin. The [incremental approach](@ref mp-theory-incremental-tangent)
uses the tangent, but only for the increments of a step. Between the two, the
affine formulation of [massonAffine2000](@citet) uses the tangent on the total
strain: every phase is replaced by the line that touches its stress–strain
curve at the current strain, a linear law with a prestress. The composite of
these phases is a linear thermoelastic one, which the schemes of the package
homogenize. [The example](@ref mp-ex-affine) runs it.

## The linearization of a phase

Phase ``r``, of volume fraction ``f_r``, is nonlinear elastic (or viscous, the
strain then being a strain rate): its stress ``\boldsymbol{\sigma}_r(\boldsymbol{\varepsilon})`` is
a function of the strain alone. Near a strain ``\bar{\boldsymbol{\varepsilon}}_r`` it is
approximated by its first-order expansion,

```math
\boldsymbol{\sigma} \approx \mathbb{L}_r:\boldsymbol{\varepsilon} + \boldsymbol{\tau}_r,
\qquad
\mathbb{L}_r = \frac{\partial\boldsymbol{\sigma}_r}{\partial\boldsymbol{\varepsilon}}\big(\bar{\boldsymbol{\varepsilon}}_r\big),
\qquad
\boldsymbol{\tau}_r = \boldsymbol{\sigma}_r\big(\bar{\boldsymbol{\varepsilon}}_r\big) - \mathbb{L}_r:\bar{\boldsymbol{\varepsilon}}_r ,
```

exact at ``\bar{\boldsymbol{\varepsilon}}_r`` in value and in slope. The tangent
``\mathbb{L}_r`` of an isotropic law is anisotropic as soon as the strain has a
deviator: the stiffness along the current deviator differs from the stiffness
across it. For the power law ([`PowerLawElasticity`](@ref)) of exponent ``m``,
the shear stiffness along the deviator is ``m`` times the secant one.

The exact linearization would be taken at the local strain, different at every
point. As in the secant methods, the formulation keeps one linearization per
phase, about a strain that represents the phase: here its average strain.

## The comparison composite

With uniform tangents and prestresses, the phases form a linear composite with
eigenstresses, whose response a linear scheme gives through its strain
concentration tensors ``\mathbb{A}_r``. For two phases, Levin's theorem gives
its phase averages without any further scheme-specific tensor: the effect of
uniform eigenstresses follows from the concentration tensors of the composite
without them. Written for the strain,

```math
f_1\,\big(\bar{\boldsymbol{\varepsilon}}_1 - \mathbb{A}_1:\boldsymbol{E}\big)
= (\mathbb{L}_1 - \mathbb{L}_2)^{-1}:\big(\tilde{\boldsymbol{\tau}} - \langle\boldsymbol{\tau}\rangle\big),
\qquad
\tilde{\boldsymbol{\tau}} = \sum_r f_r\,\mathbb{A}_r^{T}:\boldsymbol{\tau}_r,
\qquad
\langle\boldsymbol{\tau}\rangle = \sum_r f_r\,\boldsymbol{\tau}_r ,
```

and the mixture rule ``f_1\bar{\boldsymbol{\varepsilon}}_1 + f_2\bar{\boldsymbol{\varepsilon}}_2 = \boldsymbol{E}``
gives the second phase. The tangents and prestresses depend on the average
strains, which the comparison composite returns: the formulation is a fixed
point on ``\bar{\boldsymbol{\varepsilon}}_1``, solved by Newton's method. The
macroscopic stress is that of the comparison composite,
``\boldsymbol{\Sigma} = \sum_r f_r\,(\mathbb{L}_r:\bar{\boldsymbol{\varepsilon}}_r + \boldsymbol{\tau}_r) = \sum_r f_r\,\boldsymbol{\sigma}_r(\bar{\boldsymbol{\varepsilon}}_r)``:
the phases carry the stress of their average strain.

## Properties

- **Linear phases.** The tangents are the stiffnesses and the prestresses
  vanish: the formulation is the linear scheme.
- **Laminates.** The exact fields of a nonlinear laminate are uniform in each
  layer, so that the linearization about the average strain of a layer is exact
  there, and the thermoelastic laminate of the tangents has the same solution.
  With the scheme [`Laminated`](@ref), the affine formulation is the exact
  nonlinear laminate. The tests of the package check it against an independent
  solution, by the continuity of the tractions across the layers.
- **Between the secant estimates.** For a matrix–inclusion composite of power
  laws, [massonAffine2000](@citet) find the affine estimate softer than the
  classical secant and incremental ones, and in general stiffer than the
  variational (modified secant) one, except in shear at very high nonlinearity;
  [the example](@ref mp-ex-affine) finds the same order with Mori–Tanaka.
- **The third invariant.** The anisotropic tangent sees the direction of the
  strain deviator: unlike the secant estimates, the affine one distinguishes a
  tension from a shear of the same equivalent strain.
- **No effective potential.** The stress is not the derivative of an effective
  energy, and [massonAffine2000](@citet) show that it can violate the
  Hashin–Shtrikman variational bound at small contrast and high nonlinearity.

## The composite as a law

[`AffineComposite`](@ref) builds a law of `MaterialPoints` whose internal
variable is the average strain of the first phase (`M_ε`, after its name). The
tangents come from automatic differentiation of the phase laws, and the cell
whose stiffnesses they replace gives the concentration tensors: any linear
scheme of the package for an [`RVE`](@ref), [`Laminated`](@ref) for a
[`Laminate`](@ref). The tangent of the composite follows from the
implicit-function theorem applied to the fixed point, the variation of the
concentration tensors with the tangents included.

At zero strain, a power law of exponent below one has an infinite shear
tangent: the linearization does not exist. The composite at rest is left
untouched; a loading that leaves a phase without deviatoric strain, such as a
pure volume change, is refused.

## Scope

| | |
|:--|:--|
| phases | two, laws of the strain alone (nonlinear elasticity, or viscosity read in strain rates) |
| schemes | any linear scheme of the package; the tangents are anisotropic, and the Hill tensors of an anisotropic matrix are integrated numerically |
| cost | about 70 ms per step for spheres and Mori–Tanaka, most of it in the derivatives of the anisotropic Hill tensors |
| not covered | more than two phases (Levin's theorem then needs the eigenstress concentration of the scheme); the extension to history-dependent laws in the same paper; the tangent model of [molinari1987](@citet) for viscoplastic polycrystals, which uses the same linearization but, as [massonAffine2000](@citet) show, combines it with a secant localization, an inconsistency that drives it toward the Reuss bound at high nonlinearity |
