```@meta
EditURL = "../../../../scripts/material_laws/13_incremental_tangent.jl"
```

# [The incremental tangent approach](@id ml-theory-incremental-tangent)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_laws/13_incremental_tangent.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_laws/13_incremental_tangent.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_laws/13_incremental_tangent.jl`, runs it end to end.

The [secant methods](@ref ml-theory-secant) need phases whose stress is a
function of the strain alone. A plastic phase has a history: its stress depends
on the path, and a secant modulus means nothing for it. The incremental approach
of [hill1965](@citet) linearizes such a phase on each step instead: over an
increment of the loading, every phase behaves as a linear material of its
tangent stiffness, and a linear scheme concentrates the increment of the
macroscopic strain in the phases. It applies to any phase law of
`MaterialLaws`, plasticity, viscosity or damage, and stands between the two
[uniform-field composites](@ref ml-theory-uniform-fields), which ignore the
morphology. [The example below](@ref ml-ex-incremental-tangent) runs it.

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

[`IncrementalTangent`](@ref) builds a law of `MaterialLaws` whose internal
variables are the strains of the phases (`M_ε`, …) and their internal
variables (`M_εᵖ`, …). Its consistent tangent comes from the
implicit-function theorem applied to all the equations of the step, the
variation of the concentration tensors with the tangents included. With
elastic phases it is the linear estimate of the scheme, exactly. With plastic
phases its response lies between the [uniform stress and the uniform
strain](@ref ml-theory-uniform-fields), and depends on the morphology through
the scheme.

## Scope

| | |
|:--|:--|
| phases | any laws of the strain and the stress, with internal variables; one law per phase of the cell |
| schemes | any linear scheme of the package; the phases are projected on isotropic tensors |
| cost | about 1 ms per step for two phases and the Mori–Tanaka scheme |
| not covered | the anisotropic tangent (without projection), and the other linearizations of the literature, which give different estimates: the [affine formulation](@ref ml-theory-affine) for laws of the strain alone, the second-order one |

## [Example: a particle-reinforced alloy](@id ml-ex-incremental-tangent)

This example homogenizes an elastoplastic aluminum alloy reinforced by stiff
elastic particles with the [incremental tangent approach](@ref ml-theory-incremental-tangent),
and places it between the [uniform-field composites](@ref ml-theory-uniform-fields).

### The composite

````@example ml_incremental_tangent
using MeanFieldHomogenization, TensND, Plots

alloy = J2Plasticity(; E = 70e3, ν = 0.33, σ₀ = 150.0, H = 1e3)       # MPa
particle = LinearElasticity(; E = 400e3, ν = 0.2)
cell(f) = (rve = RVE();
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest);
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f);
    rve)
laws = (M = alloy, I = particle)
nothing #hide
````

| code | theory |
|:--|:--|
| `cell(f)` | the morphology: a matrix and spherical particles of fraction `f`; its stiffnesses are replaced |
| `IncrementalTangent(cell(f), laws)` | every phase linearized by its tangent at each step, Mori–Tanaka on the isotropic projections |

### Tension

````@example ml_incremental_tangent
path = LoadingPath(range(0, 1; length = 41); ε11 = t -> 1e-2 * t)
p = plot(; xlabel = "axial strain (%)", ylabel = "axial stress (MPa)", legend = :topleft)
r = point_response(IncrementalTangent(cell(0.3), laws), path)
plot!(p, r[:ε11] * 100, r[:σ11]; lw = 2, label = "incremental tangent, 30 % particles")
for (c, label) in ((UniformStrain(laws, (M = 0.7, I = 0.3)), "uniform strain"),
                   (UniformStress(laws, (M = 0.7, I = 0.3)), "uniform stress"))
    rc = point_response(c, path)
    plot!(p, rc[:ε11] * 100, rc[:σ11]; lw = 2, ls = :dash, label)
end
plot!(p, r[:ε11] * 100, point_response(alloy, path)[:σ11]; lw = 2, ls = :dot, label = "alloy alone")
p
````

The incremental tangent estimate lies between the two uniform fields. The
uniform strain loads the stiff particles with the full strain of the alloy and
overestimates the stress by far; the uniform stress lets the alloy carry all the
strain. The particles stiffen the alloy and raise its apparent yield stress, by
carrying a share of the load that grows as the alloy yields.

### The fraction of particles

````@example ml_incremental_tangent
q = plot(; xlabel = "axial strain (%)", ylabel = "axial stress (MPa)", legend = :topleft)
for f in (0.0001, 0.1, 0.2, 0.3)
    rf = point_response(IncrementalTangent(cell(f), laws), path)
    plot!(q, rf[:ε11] * 100, rf[:σ11]; lw = 2, label = "f = $f")
end
q
````

### Check

With elastic phases the approach is the linear Mori–Tanaka estimate:

````@example ml_incremental_tangent
elastic = IncrementalTangent(cell(0.3), (M = LinearElasticity(; E = 70e3, ν = 0.33), I = particle))
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70e3, 0.33)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(400e3, 0.2)); fraction = 0.3)
E = Mandel{Float64}(1e-3, 0, 0, 0, 0, 0)
point_update(elastic, E, initial_state(elastic), 1.0)[2] ≈ mandel(homogenize(rve, MoriTanaka(), :C))
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

