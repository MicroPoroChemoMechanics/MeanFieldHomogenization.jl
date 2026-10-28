# [A power-law composite by the affine formulation](@id mp-ex-affine)

This example homogenizes the composite of [the secant example](@ref mp-ex-composite)
with the [affine formulation](@ref mp-theory-affine), and compares it with the
secant estimates and the [uniform-field composites](@ref mp-theory-uniform-fields).

## The composite

A soft matrix holds 30 % of stronger spherical particles, both of power-law
type ([`PowerLawElasticity`](@ref)).

```@example mp_ex_affine
using MeanFieldHomogenization, TensND, Plots

f = 0.3                                                              # particle fraction
matrix = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 0.2)      # MPa
particles = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 0.6)
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
laws = (M = matrix, P = particles)
nothing # hide
```

| code | theory |
|:--|:--|
| `AffineComposite(rve, laws)` | each phase replaced by its tangent and prestress at its average strain, Mori–Tanaka on the comparison composite |
| `NonlinearHomogenized(rve, Dict(pairs(laws)); method)` | the [secant estimates](@ref mp-theory-secant), classical or modified |
| `UniformStrain`, `UniformStress` | the [uniform-field composites](@ref mp-theory-uniform-fields) |

## Tension

An isochoric tension, of equivalent strain `e_max` at the end:

```@example mp_ex_affine
e_max = 5e-3
tension = LoadingPath(range(0, 1; length = 26); ε11 = t -> e_max * t,
    ε22 = t -> -e_max * t / 2, ε33 = t -> -e_max * t / 2, ε23 = 0, ε13 = 0, ε12 = 0)
σ_eq(r) = r[:σ11] - r[:σ22]                       # the von Mises stress on this path
p = plot(; xlabel = "macroscopic equivalent strain (‰)", ylabel = "macroscopic equivalent stress (MPa)",
    legend = :bottomright)
for (c, label, ls) in ((UniformStrain(laws, (M = 1 - f, P = f)), "uniform strain", :dash),
                       (NonlinearHomogenized(rve, Dict(pairs(laws)); method = ClassicalSecant()), "classical secant", :solid),
                       (AffineComposite(rve, laws), "affine", :solid),
                       (NonlinearHomogenized(rve, Dict(pairs(laws))), "modified secant", :solid),
                       (UniformStress(laws, (M = 1 - f, P = f)), "uniform stress", :dash))
    r = point_response(c, tension)
    plot!(p, r[:ε11] * 1e3, σ_eq(r); lw = 2, ls, label)
end
p
```

The affine estimate lies between the two secant ones, softer than the classical
and stiffer than the modified, as [massonAffine2000](@citet) find for the
Hashin–Shtrikman estimate.

## Tension and shear

A shear of the same equivalent strain:

```@example mp_ex_affine
shear = LoadingPath(range(0, 1; length = 11); ε11 = 0, ε22 = 0, ε33 = 0, ε23 = 0, ε13 = 0,
    ε12 = t -> sqrt(3) / 2 * e_max * t)
final(c) = (rt = point_response(c, tension); rs = point_response(c, shear);
    (tension = σ_eq(rt)[end], shear = sqrt(3) * rs[:σ12][end]))
(affine = final(AffineComposite(rve, laws)), modified_secant = final(NonlinearHomogenized(rve, Dict(pairs(laws)))))
```

The modified secant estimate depends on the strain through its second
invariant only, and gives the same equivalent stress in both. The affine one
does not: the tangent of each phase is stiffer across its strain deviator than
along it, and the composite sees the direction of the loading.

## Check

With linear phases the formulation is the linear Mori–Tanaka estimate. A
power law of exponent one is linear, of shear modulus ``\sigma_0/(3\varepsilon_0)``:

```@example mp_ex_affine
linear = (M = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 1.0),
          P = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 1.0))
c = AffineComposite(rve, linear)
lin = RVE()
add_phase!(lin, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60e3, 2 * 80.0 / 3e-3)); fraction = :rest)
add_phase!(lin, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 150e3, 2 * 320.0 / 3e-3)); fraction = f)
E = Mandel{Float64}(1e-3, -5e-4, -5e-4, 0, 0, 0)
point_update(c, E, initial_state(c), 1.0)[2] ≈ KM(homogenize(lin, MoriTanaka(), :C))
```
