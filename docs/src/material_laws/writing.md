# [Writing a law](@id mp-writing)

A law is written as **plain functions and a `NamedTuple` of parameters**.
There are two ways to write one, and both give a material that runs everywhere
in the package:

| route | you give | suits |
|:--|:--|:--|
| [`PotentialLaw`](@ref) | a free energy, and a dissipation potential if the law has internal variables | elasticity, viscoelasticity, any law of the generalized standard family |
| [`ResidualLaw`](@ref) | the fluxes, and the residual of the local equations over a step | plasticity, damage, any law written as evolution equations |

Neither asks for a derivative: the consistent tangent and the derivatives with
respect to the parameters are obtained by automatic differentiation (see
[Integrating a law over a step](@ref mp-theory)).

## What the functions receive

The functions of a law receive their arguments **by name**:

| argument | what it is | read as |
|:--|:--|:--|
| `g` | the gradients at the end of the step | `g.ε` (strain), `g.p` (pore pressure), … |
| `y` (or `α`) | the internal variables at the end of the step | `y.εᵖ`, `y.d`, … |
| `prev` | the state at the beginning of the step | `prev.y.εᵖ`, `prev.g.ε` |
| `Δt` | the length of the step | a number |
| `p` | the parameters of the law | `p.E`, `p.ℂ`, … |

A symmetric tensor is a [`Mandel`](@ref) vector of six components; a scalar is
a number. On Mandel vectors the double contraction is `sum(a .* b)`, a
fourth-order tensor is an `SMatrix{6, 6}` applied by `*`, and the helpers
[`trace_mandel`](@ref), [`deviator`](@ref), [`von_mises`](@ref) and
[`equivalent_strain`](@ref) give the usual invariants.

```@example mp_writing
using MeanFieldHomogenization, StaticArrays
ε = SVector(1e-3, -2e-4, -2e-4, 0, 0, 0)       # a Mandel vector
trace_mandel(ε), equivalent_strain(ε)
```

## A first law by the potential route

Linear isotropic elasticity is the free energy
``\psi = \tfrac{k}{2}(\operatorname{tr}\boldsymbol{\varepsilon})^2 + \mu\,\boldsymbol{e}:\boldsymbol{e}``,
``\boldsymbol{e}`` being the deviator of the strain. It has no internal
variable, so no dissipation:

```@example mp_writing
elastic = PotentialLaw(;
    name = :MyElasticity,
    energy = (g, α, p) -> p.k / 2 * trace_mandel(g.ε)^2 + p.μ * sum(abs2, deviator(g.ε)),
    p = (k = 160e3, μ = 80e3),                      # MPa
)
```

[`point_update`](@ref) integrates it over one step. It returns the stress, the
tangent and the new state:

```@example mp_writing
σ, ℂ, state = point_update(elastic, ε, initial_state(elastic), 0.0)
σ
```

The tangent is the stiffness ``3k\,\mathbb{J} + 2\mu\,\mathbb{K}``, here in Mandel
form: its `[1, 2]` entry is ``k - 2\mu/3``.

```@example mp_writing
ℂ[1, 2], 160e3 - 2 * 80e3 / 3
```

## Adding an internal variable: a Maxwell solid

A spring in series with a dashpot carries a viscous strain
``\boldsymbol{\varepsilon}^v``. The free energy stores the elastic part, the
dissipation potential the viscous rate. Both are written; the layout names the
internal variable and its size:

```@example mp_writing
ℂ_mandel = 3 * 160e3 * (identity_mandel() * identity_mandel()' / 3) +
           2 * 80e3 * (one(SMatrix{6, 6, Float64}) - identity_mandel() * identity_mandel()' / 3)
maxwell = PotentialLaw(;
    name = :MyMaxwell,
    energy = (g, α, p) -> sum((g.ε - α.εᵛ) .* (p.ℂ * (g.ε - α.εᵛ))) / 2,
    dissipation = (α̇, α, p) -> p.τ * sum(α̇.εᵛ .* (p.ℂ * α̇.εᵛ)) / 2,
    layout = (εᵛ = 6,),
    p = (ℂ = ℂ_mandel, τ = 2.0),
)
```

Over a step the viscous strain minimizes the incremental energy; the code
solves that minimization and its tangent. Held at a fixed strain, the stress
relaxes:

```@example mp_writing
r = point_response(maxwell, LoadingPath(range(0, 4; length = 41); ε11 = 1e-3, ε22 = 0, ε33 = 0))
r[:σ11][end] / r[:σ11][1], exp(-4 / 2.0)        # implicit scheme vs exact relaxation
```

The two numbers differ by the error of the implicit scheme over steps of 0.1;
finer steps bring them together. The same law ships as
[`MaxwellViscoelasticity`](@ref).

## The residual route: plasticity and damage

When a law is written as evolution equations, give the residual whose root is
the end of the step. Here is a scalar damage law: the stiffness is degraded by
a factor ``1 - d``, and ``d`` grows with the strain energy ``Y`` beyond a
threshold ``Y_0``, as ``d = 1 - \sqrt{Y_0/Y}``, never decreasing.

```@example mp_writing
Y(ε, p) = sum(ε .* (p.ℂ * ε)) / 2                    # strain energy of the undamaged material
d_of(ε, p) = 1 - sqrt(p.Y₀ / Y(ε, p))                 # the damage the threshold calls for

damage = ResidualLaw(;
    name = :MyDamage,
    stress = (ε, y, p) -> (1 - y.d) * (p.ℂ * ε),
    residual = (y, g, prev, Δt, p) -> (d = y.d - d_of(g.ε, p),),
    elastic = (g, prev, Δt, p) -> d_of(g.ε, p) <= prev.y.d,   # below the threshold: d frozen
    layout = (d = 1,),
    p = (ℂ = ℂ_mandel, Y₀ = 0.05),
)
```

Three functions:

- `stress` gives the stress from the strain and the internal variables (a law
  of the strain alone may give `stress`; a coupled law gives `fluxes`, below);
- `residual` returns, by name, the equations the internal variables satisfy at
  the end of the step: here ``d - d(\boldsymbol\varepsilon) = 0``;
- `elastic` says when the step leaves them unchanged: below the threshold, or
  when unloading. It is what makes the damage irreversible.

```@example mp_writing
path = LoadingPath(range(0, 1; length = 5); ε11 = t -> 1e-3 * sin(π * t), ε22 = 0, ε33 = 0)
rd = point_response(damage, path)
rd[:d]
```

The damage grows while the strain rises, and stays at its maximum when it
falls back: the `elastic` test kept it.

For plasticity the residual collects the flow rule and the consistency
condition; [`J2Plasticity`](@ref) is written exactly this way, in about twenty
lines of `library.jl` that can serve as a template.

## Coupled fields

A law of several primal fields declares them, with their conjugate fluxes in
the same order, and returns its fluxes by name. Biot's poroelasticity couples
the strain and the pore pressure:

```@example mp_writing
b, M = 0.8, 1e4
biot = PotentialLaw(;
    name = :MyBiot,
    energy = (g, α, p) -> sum(g.ε .* (p.ℂ * g.ε)) / 2 - p.b * g.p * trace_mandel(g.ε) - g.p^2 / (2p.M),
    gradients = (ε = 6, p = 1), flux_names = (σ = 6, φ = 1),
    signs = (φ = -1,),                         # φ = −∂ψ/∂p
    p = (ℂ = ℂ_mandel, b = b, M = M),
)
F, 𝕂, _ = point_update(biot, SVector(1e-3, 0, 0, 0, 0, 0, 2.0), initial_state(biot), 0.0)
𝕂[7, 7], 1 / M                                  # ∂φ/∂p = 1/M
```

The tangent `𝕂` is the ``7 \times 7`` matrix of all the derivatives: its blocks
are ``\partial\boldsymbol\sigma/\partial\boldsymbol\varepsilon``,
``\partial\boldsymbol\sigma/\partial p``, ``\partial\phi/\partial\boldsymbol\varepsilon``
and ``\partial\phi/\partial p``. A thermo-hydro-mechanical law adds the
temperature and the entropy the same way, `gradients = (ε = 6, p = 1, T = 1)`;
a field of another size, a temperature gradient ``\nabla T`` of size 3 with the
heat flux, is declared by its size too. The residual route takes the same
`gradients` and `flux_names`, with `fluxes = (g, y, p) -> (σ = …, φ = …)` in
place of `stress`.

## External variables: a temperature

Some variables are read by a law without being returned with a flux: a
temperature imposed by a separate heat computation, a time, a degree of
hydration. They are given by name for the step, `external = (T = …,)`, and the
law reads them in `p`, beside its parameters. A thermoelastic law, whose free
expansion is ``\alpha\,(T - T_0)\,\boldsymbol{1}``:

```@example mp_writing
thermal(g, p) = g.ε - p.α * (p.T - p.T₀) * identity_mandel()
thermo = PotentialLaw(;
    name = :Thermoelastic,
    energy = (g, a, p) -> sum(thermal(g, p) .* (p.ℂ * thermal(g, p))) / 2,
    p = (ℂ = ℂ_mandel, α = 1.2e-5, T₀ = 293.0),
)
σ_hot, _, _ = point_update(thermo, zero(ε), initial_state(thermo), 0.0; external = (T = 343.0,))
σ_hot[1], -3 * 160e3 * 1.2e-5 * 50            # blocked expansion: −3kαΔT
```

A loading path gives them as functions of time, and a value already in `p`
takes the value given, which makes a parameter vary along the path:

```@example mp_writing
heating = LoadingPath(0:0.25:1; external = (T = t -> 293.0 + 100t,))   # free faces
point_response(thermo, heating)[:ε11]           # α ΔT along the ramp
```

## Checking a law

[`check_material_interface`](@ref) exercises a law against the Gauss-point
contract, including the tangent against a central finite difference of the
stress (a check, not a computation the law ever uses):

```@example mp_writing
check_material_interface(damage; verbose = false), check_material_interface(biot; verbose = false)
```

!!! tip "Parameters belong in `p`"
    Put every number of the law in `p`: building the law again with
    `ForwardDiff.Dual` parameters then differentiates its whole response with
    respect to them (see [Plugging a law into a computation](@ref mp-plugging)).

!!! note "Symbolic numbers"
    A law without internal variable evaluates with `Symbolics.Num` entries, which
    gives its stress and tangent in closed form, as the elastic law above does.
    One restriction comes from Symbolics itself: its ForwardDiff extension
    (version 7.47) fails on `muladd` with a symbolic number among dual ones, which
    a matrix product such as `p.ℂ * g.ε` performs; write such a law with traces
    and deviators, as above, until that is fixed. A law with internal variables
    is solved iteratively and refuses symbolic numbers with a clear message.
