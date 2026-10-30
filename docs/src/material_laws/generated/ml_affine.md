```@meta
EditURL = "../../../../scripts/material_laws/14_affine.jl"
```

# [The affine formulation](@id ml-theory-affine)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_laws/14_affine.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_laws/14_affine.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_laws/14_affine.jl`, runs it end to end.

The [secant methods](@ref ml-theory-secant) replace each nonlinear phase by a
linear one of the same stress at the same strain, without prestress: a line
through the origin. The [incremental approach](@ref ml-theory-incremental-tangent)
uses the tangent, but only for the increments of a step. Between the two, the
affine formulation of [massonAffine2000](@citet) uses the tangent on the total
strain: every phase is replaced by the line that touches its stress–strain
curve at the current strain, a linear law with a prestress. The composite of
these phases is a linear thermoelastic one, which the schemes of the package
homogenize. [The example below](@ref ml-ex-affine) runs it.

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
  [the example below](@ref ml-ex-affine) finds the same order with Mori–Tanaka.
- **The third invariant.** The anisotropic tangent sees the direction of the
  strain deviator: unlike the secant estimates, the affine one distinguishes a
  tension from a shear of the same equivalent strain.
- **No effective potential.** The stress is not the derivative of an effective
  energy, and [massonAffine2000](@citet) show that it can violate the
  Hashin–Shtrikman variational bound at small contrast and high nonlinearity.
  The [second-order estimate](@ref ml-theory-second-order) below takes the energy of
  the same comparison composite instead.

## [The second-order estimate](@id ml-theory-second-order)

[ponteCastaneda1996](@citet) builds the same comparison composite from a
different starting point: the expansion of the energy ``w_r`` of each phase to
second order about a reference strain, with the tangent at that strain as the
curvature. Taking the reference strain of each phase at its average strain, as
that paper does in its applications, the expanded energies are those of the
linearized phases above, up to constants, and the comparison composite and its
fixed point are those of the affine formulation. What differs is the estimate
drawn from it: the effective energy is approximated by that of the comparison
composite, constants included,

```math
\tilde{w}(\boldsymbol{E}) = \sum_r f_r\,\Big[w_r(\bar{\boldsymbol{\varepsilon}}_r)
  - \boldsymbol{\sigma}_r(\bar{\boldsymbol{\varepsilon}}_r):\bar{\boldsymbol{\varepsilon}}_r
  + \tfrac{1}{2}\,\bar{\boldsymbol{\varepsilon}}_r:\mathbb{L}_r:\bar{\boldsymbol{\varepsilon}}_r\Big]
  + \tfrac{1}{2}\,\boldsymbol{E}:\tilde{\mathbb{L}}:\boldsymbol{E} + \tilde{\boldsymbol{\tau}}:\boldsymbol{E}
  + \tfrac{1}{2}\,(\boldsymbol{\tau}_1 - \boldsymbol{\tau}_2):(\mathbb{L}_1 - \mathbb{L}_2)^{-1}:(\tilde{\boldsymbol{\tau}} - \langle\boldsymbol{\tau}\rangle),
```

``\tilde{\mathbb{L}} = \sum_r f_r\,\mathbb{L}_r\mathbb{A}_r`` the effective stiffness of
the comparison composite. The bracket gathers the constants of the expanded
energies; the next two terms are the energy of the comparison composite under
``\boldsymbol{E}`` without its prestresses, and the last one is the energy the
prestresses store at zero macroscopic strain, which Levin's theorem gives for
two phases. The macroscopic stress is the derivative of this energy,

```math
\boldsymbol{\Sigma} = \frac{\mathrm{d}\tilde{w}}{\mathrm{d}\boldsymbol{E}},
```

the average strains following ``\boldsymbol{E}``. They are not stationary points of
``\tilde{w}``, so that this derivative differs from the stress of the affine
formulation; it needs the derivative of the average strains, which the
implicit-function theorem gives from the fixed point. Since ``\mathrm{d}w_r`` is
``\boldsymbol{\sigma}_r:\mathrm{d}\bar{\boldsymbol{\varepsilon}}_r``, the energies of the phases
themselves are never needed, only their stresses and tangents.

For linear phases the estimate is the linear scheme, and it is exact for a
laminate, whose layers carry uniform strains, as the paper points out; the
tests of the package check both, and that the stress is the derivative of the
energy above. [`SecondOrderComposite`](@ref) builds it with the arguments of
[`AffineComposite`](@ref). The paper also mentions, as reference for the
curvature, the second moments of the strain in the phases instead of their
averages, as a possibility left for later work: it is not covered.

## The composite as a law

[`AffineComposite`](@ref) builds a law of `MaterialLaws` whose internal
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
| cost | about 70 ms per step for spheres and Mori–Tanaka, most of it in the derivatives of the anisotropic Hill tensors; the second-order estimate about 0.3 s, its stress holding the first derivatives of the Hill tensors and its tangent the second, and its first call compiles those for several minutes |
| not covered | more than two phases (Levin's theorem then needs the eigenstress concentration of the scheme); the extension to history-dependent laws in the same paper; the tangent model of [molinari1987](@citet) for viscoplastic polycrystals, which uses the same linearization but, as [massonAffine2000](@citet) show, combines it with a secant concentration rule, an inconsistency that drives it toward the Reuss bound at high nonlinearity |

## [Example: a power-law composite by the affine formulation](@id ml-ex-affine)

This example homogenizes the composite of [the secant example](@ref ml-ex-composite)
with the [affine formulation](@ref ml-theory-affine), and compares it with the
secant estimates and the [uniform-field composites](@ref ml-theory-uniform-fields).

### The composite

A soft matrix holds 30 % of stronger spherical particles, both of power-law
type ([`PowerLawElasticity`](@ref)).

````@example ml_affine
using MeanFieldHomogenization, TensND, Plots

f = 0.3                                                              # particle fraction
matrix = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 0.2)      # MPa
particles = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 0.6)
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
laws = (M = matrix, P = particles)
nothing #hide
````

| code | theory |
|:--|:--|
| `AffineComposite(rve, laws)` | each phase replaced by its tangent and prestress at its average strain, Mori–Tanaka on the comparison composite |
| `SecondOrderComposite(rve, laws)` | the same comparison composite, the stress derived from its energy |
| `NonlinearHomogenized(rve, Dict(pairs(laws)); method)` | the [secant estimates](@ref ml-theory-secant), classical or modified |
| `UniformStrain`, `UniformStress` | the [uniform-field composites](@ref ml-theory-uniform-fields) |

### Tension

An isochoric tension, of equivalent strain `e_max` at the end:

````@example ml_affine
e_max = 5e-3
tension = LoadingPath(range(0, 1; length = 26); ε11 = t -> e_max * t,
    ε22 = t -> -e_max * t / 2, ε33 = t -> -e_max * t / 2, ε23 = 0, ε13 = 0, ε12 = 0)
σ_eq(r) = r[:σ11] - r[:σ22]                       # the von Mises stress on this path
p = plot(; xlabel = "macroscopic equivalent strain (‰)", ylabel = "macroscopic equivalent stress (MPa)",
    legend = :bottomright)
for (c, label, ls) in ((UniformStrain(laws, (M = 1 - f, P = f)), "uniform strain", :dash),
                       (NonlinearHomogenized(rve, Dict(pairs(laws)); method = ClassicalSecant()), "classical secant", :solid),
                       (AffineComposite(rve, laws), "affine", :solid),
                       (SecondOrderComposite(rve, laws), "second order", :solid),
                       (NonlinearHomogenized(rve, Dict(pairs(laws))), "modified secant", :solid),
                       (UniformStress(laws, (M = 1 - f, P = f)), "uniform stress", :dash))
    r = point_response(c, tension)
    plot!(p, r[:ε11] * 1e3, σ_eq(r); lw = 2, ls, label)
end
p
````

The affine estimate lies between the two secant ones, softer than the classical
and stiffer than the modified, as [massonAffine2000](@citet) find for the
Hashin–Shtrikman estimate.
The second-order estimate, drawn from the energy of the same comparison
composite, is softer than both the affine and the modified secant estimates
on this composite.

### Tension and shear

A shear of the same equivalent strain:

````@example ml_affine
shear = LoadingPath(range(0, 1; length = 11); ε11 = 0, ε22 = 0, ε33 = 0, ε23 = 0, ε13 = 0,
    ε12 = t -> sqrt(3) / 2 * e_max * t)
final(c) = (rt = point_response(c, tension); rs = point_response(c, shear);
    (tension = σ_eq(rt)[end], shear = sqrt(3) * rs[:σ12][end]))
(affine = final(AffineComposite(rve, laws)), second_order = final(SecondOrderComposite(rve, laws)),
    modified_secant = final(NonlinearHomogenized(rve, Dict(pairs(laws)))))
````

The modified secant estimate depends on the strain through its second
invariant only, and gives the same equivalent stress in both. The affine and
second-order ones do not: the tangent of each phase is stiffer across its
strain deviator than along it, and the composite sees the direction of the
loading.

### Check

With linear phases the formulation is the linear Mori–Tanaka estimate. A
power law of exponent one is linear, of shear modulus ``\sigma_0/(3\varepsilon_0)``:

````@example ml_affine
linear = (M = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 1.0),
          P = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 1.0))
c = AffineComposite(rve, linear)
lin = RVE()
add_phase!(lin, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60e3, 2 * 80.0 / 3e-3)); fraction = :rest)
add_phase!(lin, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 150e3, 2 * 320.0 / 3e-3)); fraction = f)
E = Mandel{Float64}(1e-3, -5e-4, -5e-4, 0, 0, 0)
point_update(c, E, initial_state(c), 1.0)[2] ≈ mandel(homogenize(lin, MoriTanaka(), :C))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

