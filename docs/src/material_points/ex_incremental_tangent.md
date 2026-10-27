# [A particle-reinforced alloy](@id mp-ex-incremental-tangent)

This example homogenizes an elastoplastic aluminum alloy reinforced by stiff
elastic particles with the [incremental tangent approach](@ref mp-theory-incremental-tangent),
and places it between the [uniform-field composites](@ref mp-theory-uniform-fields).

## The composite

```@example mp_ex_incremental
using MeanFieldHomogenization, TensND, Plots

alloy = J2Plasticity(; E = 70e3, ν = 0.33, σ₀ = 150.0, H = 1e3)       # MPa
particle = LinearElasticity(; E = 400e3, ν = 0.2)
cell(f) = (rve = RVE();
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest);
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f);
    rve)
laws = (M = alloy, I = particle)
nothing # hide
```

| code | theory |
|:--|:--|
| `cell(f)` | the morphology: a matrix and spherical particles of fraction `f`; its stiffnesses are replaced |
| `IncrementalTangent(cell(f), laws)` | every phase linearized by its tangent at each step, Mori–Tanaka on the isotropic projections |

## Tension

```@example mp_ex_incremental
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
```

The incremental tangent estimate lies between the two uniform fields. The
uniform strain loads the stiff particles with the full strain of the alloy and
overestimates the stress by far; the uniform stress lets the alloy carry all the
strain. The particles stiffen the alloy and raise its apparent yield stress, by
carrying a share of the load that grows as the alloy yields.

## The fraction of particles

```@example mp_ex_incremental
q = plot(; xlabel = "axial strain (%)", ylabel = "axial stress (MPa)", legend = :topleft)
for f in (0.0001, 0.1, 0.2, 0.3)
    rf = point_response(IncrementalTangent(cell(f), laws), path)
    plot!(q, rf[:ε11] * 100, rf[:σ11]; lw = 2, label = "f = $f")
end
q
```

## Check

With elastic phases the approach is the linear Mori–Tanaka estimate:

```@example mp_ex_incremental
elastic = IncrementalTangent(cell(0.3), (M = LinearElasticity(; E = 70e3, ν = 0.33), I = particle))
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70e3, 0.33)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(400e3, 0.2)); fraction = 0.3)
E = Mandel{Float64}(1e-3, 0, 0, 0, 0, 0)
point_update(elastic, E, initial_state(elastic), 1.0)[2] ≈ KM(homogenize(rve, MoriTanaka(), :C))
```
