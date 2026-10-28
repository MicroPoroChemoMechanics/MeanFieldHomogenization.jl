```@meta
EditURL = "../../../../scripts/material_points/02_damage.jl"
```

# [A damage law written from scratch](@id mp-ex-damage)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_points/02_damage.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_points/02_damage.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_points/02_damage.jl`, runs it end to end.

This example writes a law that is not shipped with the package, from its
equations to a cyclic test, in about fifteen lines. It is a scalar isotropic
damage law of the simplest kind, chosen for being easy to check by hand: the
point is the procedure, which is the same for any law with internal
variables.

## The model

The material is linear elastic, of stiffness ``\mathbb{C}``, degraded by a
scalar damage ``d \in [0, 1)``:

```math
\boldsymbol{\sigma} = (1 - d)\,\mathbb{C}:\boldsymbol{\varepsilon}.
```

The damage is driven by the elastic energy of the undamaged material,
``Y = \tfrac12\,\boldsymbol{\varepsilon}:\mathbb{C}:\boldsymbol{\varepsilon}``.
Below a threshold ``Y_0`` nothing happens; beyond it the damage takes the value
``d^\ast(Y) = 1 - \sqrt{Y_0 / Y}``, for which the stored energy
``(1 - d)\,Y`` stays equal to ``\sqrt{Y_0 Y}``. Damage never heals: when the
strain decreases, ``d`` keeps its largest value so far.

Over a step this reads: if ``d^\ast(Y) \le d_n`` the step is elastic,
``d = d_n``; otherwise ``d`` solves ``d - d^\ast(Y) = 0``.

## The law in code

````@example mp_damage
using MeanFieldHomogenization, StaticArrays, Plots

k, μ = 20e3, 12e3                                     # bulk and shear moduli, MPa
J = identity_mandel() * identity_mandel()' / 3        # the projector on volume changes
ℂ = 3k * J + 2μ * (one(SMatrix{6, 6, Float64}) - J)   # isotropic stiffness, Mandel form

Y(ε, p) = transpose(ε) * (p.ℂ * ε) / 2
d_star(ε, p) = 1 - sqrt(p.Y₀ / Y(ε, p))

damage = ResidualLaw(;
    name = :ScalarDamage,
    stress = (ε, y, p) -> (1 - y.d) * (p.ℂ * ε),
    residual = (y, g, prev, Δt, p) -> (d = y.d - d_star(g.ε, p),),
    elastic = (g, prev, Δt, p) -> d_star(g.ε, p) <= prev.y.d,
    layout = (d = 1,),
    p = (ℂ = ℂ, Y₀ = 1e-3),                            # Y₀ in MPa (energy per volume)
)
````

Each line is one piece of the model:

| code | model |
|:--|:--|
| `stress` | ``\boldsymbol{\sigma} = (1-d)\,\mathbb{C}:\boldsymbol{\varepsilon}`` |
| `residual` | ``d - d^\ast(Y) = 0`` on a damaging step |
| `elastic` | ``d^\ast(Y) \le d_n``: below the threshold, or unloading |
| `layout` | one internal variable, the scalar `d` |
| `p` | the stiffness and the threshold |

The derivative of the stress with respect to the strain, the consistent
tangent of a damaging step, is not written: it comes from the
implicit-function theorem applied to the residual. The contract checker
confirms it against a finite difference:

````@example mp_damage
check_material_interface(damage; verbose = false)
````

## A test with unloading

Uniaxial strain (``\varepsilon_{22} = \varepsilon_{33} = 0``), loaded, unloaded
to zero, then loaded further:

````@example mp_damage
ε_axial(t) = t < 0.4 ? 5e-4 * t / 0.4 : t < 0.6 ? 5e-4 * (0.6 - t) / 0.2 : 1e-3 * (t - 0.6) / 0.4
path = LoadingPath(range(0, 1; length = 101); ε11 = ε_axial, ε22 = 0, ε33 = 0)
r = point_response(damage, path)
plot(r[:ε11] * 1e3, r[:σ11]; lw = 2, label = "ScalarDamage", legend = :topleft,
    xlabel = "axial strain ε₁₁ (‰)", ylabel = "axial stress σ₁₁ (MPa)")
````

The unloading goes back to the origin along a straight line of reduced slope
``(1 - d)\,\mathbb{C}_{1111}``, and the reloading follows it up to the previous
maximum before damage resumes.

## Checking it by hand

In uniaxial strain ``Y = \tfrac12\,\mathbb{C}_{1111}\,\varepsilon_{11}^2``. The damage
is the running maximum of ``d^\ast``, and the stress follows:

````@example mp_damage
C11 = ℂ[1, 1]
d_ref = accumulate(max, [e == 0 ? 0.0 : max(0.0, 1 - sqrt(1e-3 / (C11 * e^2 / 2))) for e in r[:ε11]])
maximum(abs, r[:d] - d_ref), maximum(abs, r[:σ11] - (1 .- d_ref) .* C11 .* r[:ε11])
````

Both differences are at the rounding of the numbers.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

