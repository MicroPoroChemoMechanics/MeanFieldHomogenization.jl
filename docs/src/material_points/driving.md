# [Driving a point](@id mp-driving)

Before a law goes into a structural computation, it is run at a single
material point along a known path, as a specimen in a test machine. That is
what [`point_response`](@ref) does: it takes a law (or any material of the
package) and a [`LoadingPath`](@ref), and returns the whole response.

## A loading path

A path has two parts: the **times** at which the response is computed, and,
for every component of every field, **what is prescribed**: either the
gradient (a strain, a pore pressure) or the conjugate flux (a stress, a fluid
content).

```@example mp_driving
using MeanFieldHomogenization

law = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 2e3)

tension = LoadingPath(range(0, 1; length = 41); ε11 = t -> 5e-3 * t)
```

`ε11 = t -> 5e-3 * t` drives the axial strain. Every component that is not
named has a **zero flux**: here the five other stress components vanish, which
is a uniaxial tensile test with free lateral faces. A value is a number (held
constant) or a function of time. The names are

| field | by its gradient | by its flux |
|:--|:--|:--|
| strain / stress | `ε11`, `ε22`, `ε33`, `ε23`, `ε13`, `ε12` | `σ11`, …, `σ12` |
| a scalar field (a law of [coupled fields](@ref mp-writing)) | its name: `p` | its flux's name: `φ` |

Shear components are tensor components: `ε12` is ``\varepsilon_{12}``, half the
engineering shear. `ε21` and `ε12` name the same component. A component
cannot be driven by both its gradient and its flux.

## Running it

```@example mp_driving
r = point_response(law, tension)
r[:σ11][end], r[:ε22][end], r[:p][end]
```

The response is read **by name**: the components of the fields (`r[:σ11]`,
`r[:ε22]`), the internal variables of the law (`r[:p]`, `r[:εᵖ]`), a component
of a tensorial internal variable (`r[:εᵖ11]`), and the times `r[:t]`. Each is a
vector, one value per time of the path.

```@example mp_driving
using Plots
plot(r[:ε11] * 100, r[:σ11]; xlabel = "axial strain ε₁₁ (%)", ylabel = "axial stress σ₁₁ (MPa)",
    label = "J2Plasticity", lw = 2, legend = :bottomright)
```

The lateral strain ``\varepsilon_{22}`` is not prescribed: at each time the
driver finds the lateral strains that make the lateral stresses vanish, by
Newton's method with the consistent tangent (see
[mixed control](@ref mp-theory)).

## Common tests

```@example mp_driving
creep    = LoadingPath(range(0, 100; length = 51); σ11 = 150.0)                  # constant stress
relax    = LoadingPath(range(0, 10; length = 51); ε11 = 1e-3, ε22 = 0, ε33 = 0)   # fixed strain, oedometric
cycle    = LoadingPath(range(0, 1; length = 81); ε11 = t -> 4e-3 * sin(2π * t))   # one tension–compression cycle
shear    = LoadingPath(range(0, 1; length = 21); ε12 = t -> 2e-3 * t, ε11 = 0, ε22 = 0, ε33 = 0)
nothing # hide
```

For a law of coupled fields, a drained test holds the pore pressure
(`p = 0`), an undrained one leaves the fluid content unprescribed, hence zero:
see [Drained and undrained poroelasticity](@ref mp-ex-biot).

## Robustness

When the local integration of the law or the driver's own iteration fails on a
step, the step is retried as **two half steps**, recursively, down to
`max_cuts` halvings (8 by default). Only the times of the path are recorded.
If a step still fails, the error says where:

- [`LocalIntegrationError`](@ref): the local equations of the law were not
  solved;
- [`PointDriverError`](@ref): the prescribed fluxes were not reached.

## Writing as it goes, and continuing

`on_step(i, t, g, F, state)` is called after each time of the path, which is
how a long run writes its results to a file as it computes them:

```@example mp_driving
io = IOBuffer()
point_response(law, tension; on_step = (i, t, g, F, st) -> println(io, t, ",", F[1]))
first(split(String(take!(io)), '\n'), 3)
```

A run continues from where another stopped by passing its last state:

```@example mp_driving
unload = LoadingPath(range(1, 2; length = 21); ε11 = t -> 5e-3 * (2 - t))
r2 = point_response(law, unload; state = r.states[end])
r2[:σ11][end]                       # the residual stress after unloading to zero strain
```

## Any material of the package

The driver accepts every material of the Gauss-point contract, not only the
laws of this module. A homogenized elastic material, or the fractured rock of
the [ARMA 2011 model](@ref fe-fractured-rock), is driven the same way:

```@example mp_driving
using TensND                                    # TensISO, the isotropic stiffnesses
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0, 2 * 18.0)); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 90.0, 2 * 60.0)); fraction = 0.2)
homogenized = HomogenizedElastic(rve, MoriTanaka())
point_response(homogenized, LoadingPath(0:0.5:1; ε11 = t -> 1e-3 * t))[:σ11]
```
