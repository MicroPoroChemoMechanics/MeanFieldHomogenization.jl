# [A random FCC aggregate](@id mp-ex-polycrystal)

This example pulls an aggregate of copper-like grains of random orientations,
under the [Taylor model](@ref mp-theory-polycrystals), and recovers the Taylor
factor of a random FCC texture.

## The aggregate

```@example mp_ex_polycrystal
using MeanFieldHomogenization, Random, Plots

τ₀ = 10.0                                                   # critical resolved shear stress, MPa
grain(frame) = CrystalViscoplasticity(; C₁₁ = 168.4e3, C₁₂ = 121.4e3, C₄₄ = 75.4e3,
    K = 0.05, n = 10, τ₀, frame)
frames = random_orientations(60; rng = MersenneTwister(2))
aggregate = Polycrystal(grain.(frames))
nothing # hide
```

| code | theory |
|:--|:--|
| `CrystalViscoplasticity(…; frame)` | the law of a grain, its cube axes along `frame` |
| `random_orientations(60)` | 60 orientations drawn uniformly over the rotations |
| `Polycrystal(grains)` | every grain at the macroscopic strain (Taylor), equal volumes |

## Tension

The aggregate is pulled along ``\underline{e}_1``, its lateral faces free:

```@example mp_ex_polycrystal
r = point_response(aggregate, LoadingPath(range(0, 1; length = 26); ε11 = t -> 1e-2 * t))
plot(r[:ε11] * 100, r[:σ11] / τ₀; lw = 2, label = "Taylor aggregate, 60 grains",
    xlabel = "axial strain (%)", ylabel = "σ₁₁ / τ₀", legend = :bottomright)
hline!([3.06]; ls = :dash, label = "Taylor factor, Bishop and Hill (1951)")
```

The response is elastic until the grains reach their critical stresses, then
flows at about ``3.06\,\tau_0``: the Taylor factor of a random FCC texture. The
plateau is slightly above it, by the viscous overstress of the grains and the
sampling of 60 orientations:

```@example mp_ex_polycrystal
r[:σ11][end] / τ₀
```

## Inside the aggregate

The state of every grain is kept. The number of slip systems that slipped in
each grain over the last step:

```@example mp_ex_polycrystal
active(s_old, s_new) = count(abs.(s_new.y[1:12] - s_old.y[1:12]) .> 1e-8)
counts = active.(grain_states(r.states[end - 1]), grain_states(r.states[end]))
histogram(counts; bins = 0.5:1:12.5, label = "", xlabel = "active slip systems in a grain",
    ylabel = "number of grains")
```

Most grains slip on six or eight systems, a few on four. A rigid–plastic grain
following an imposed strain selects five independent systems; the viscous law
of the grains makes no such selection, and spreads the slip over every system
whose resolved stress reaches its critical value.
