# [Two elastoplastic phases: uniform strain and uniform stress](@id mp-ex-uniform-fields)

This example builds the two [uniform-field composites](@ref mp-theory-uniform-fields)
of a steel and an aluminum alloy, and checks them against the elastic bounds
and against the closed form of linear hardening.

## The phases

```@example mp_ex_uniform
using MeanFieldHomogenization, Plots

steel = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 10e3)     # MPa
alloy = J2Plasticity(; E = 70e3, ν = 0.33, σ₀ = 150.0, H = 2e3)
laws, fractions = (S = steel, A = alloy), (S = 0.3, A = 0.7)
taylor = UniformStrain(laws, fractions)
sachs = UniformStress(laws, fractions)
nothing # hide
```

| code | theory |
|:--|:--|
| `UniformStrain(laws, fractions)` | every phase at the macroscopic strain (Voigt, Taylor) |
| `UniformStress(laws, fractions)` | every phase at the macroscopic stress (Reuss, Sachs) |
| `laws`, `fractions` | the law and the volume fraction of each phase, by name |

## A tensile test

```@example mp_ex_uniform
path = LoadingPath(range(0, 1; length = 41); ε11 = t -> 6e-3 * t)
rT, rS = point_response(taylor, path), point_response(sachs, path)
p = plot(rT[:ε11] * 1e3, rT[:σ11]; lw = 2, label = "uniform strain (Taylor)",
    xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)", legend = :bottomright)
plot!(p, rS[:ε11] * 1e3, rS[:σ11]; lw = 2, label = "uniform stress (Sachs)")
```

The uniform strain is the stiffer: the strong steel is strained as much as the
alloy and carries a large share of the stress. Under a uniform stress the alloy
yields first and takes most of the strain, which the composite shows as an
earlier and lower plastic branch. The internal variables of each phase are read
by name:

```@example mp_ex_uniform
(alloy_p_taylor = rT[:A_p][end], alloy_p_sachs = rS[:A_p][end], alloy_strain_sachs = rS[:A_ε11][end])
```

## Checks

**Linear hardening in uniaxial stress.** Under a uniform stress every phase is
in uniaxial stress ``\sigma``, and its axial strain is
``\sigma/E + \langle\sigma - \sigma_0\rangle/H``. The macroscopic strain is
their average:

```@example mp_ex_uniform
axial(σ, E, σ₀, H) = σ / E + max(σ - σ₀, 0) / H
expected = [0.3 * axial(σ, 200e3, 250.0, 10e3) + 0.7 * axial(σ, 70e3, 150.0, 2e3) for σ in rS[:σ11]]
maximum(abs, expected - rS[:ε11])
```

**The elastic bounds.** In the elastic range the tangents of the two composites
are the Voigt and Reuss stiffnesses of the package:

```@example mp_ex_uniform
using TensND, StaticArrays
rve = RVE()
add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(200e3, 0.3)); fraction = 0.3)
add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70e3, 0.33)); fraction = :rest)
E = Mandel{Float64}(1e-4, 0, 0, 0, 0, 0)
(point_update(taylor, E, initial_state(taylor), 1.0)[2] ≈ KM(homogenize(rve, Voigt(), :C)),
 point_update(sachs, E, initial_state(sachs), 1.0)[2] ≈ KM(homogenize(rve, Reuss(), :C)))
```
