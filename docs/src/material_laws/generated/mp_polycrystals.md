```@meta
EditURL = "../../../../scripts/material_laws/12_polycrystals.jl"
```

# [Polycrystals](@id mp-theory-polycrystals)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_laws/12_polycrystals.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_laws/12_polycrystals.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_laws/12_polycrystals.jl`, runs it end to end.

A metal is an aggregate of grains, each a single crystal that deforms by slip
on its own slip systems, oriented at random or along a texture. Its response
follows from the [single-crystal law](@ref mp-laws) of every grain and from a
rule saying how the grains share the macroscopic strain and stress. This page
states three such rules for an aggregate of many grains: the uniform strain of
Taylor, and the interaction of each grain with a matrix standing for the
others, elastic after Kröner and elastoplastic after Berveiller and Zaoui. The
[uniform-field composites](@ref mp-theory-uniform-fields) are the uniform rules
for a few phases. [The example below](@ref mp-ex-polycrystal) runs an aggregate.

## The grains

Grain ``g`` has the volume fraction ``w_g`` and a law of
[`CrystalViscoplasticity`](@ref), whose cube axes are the vectors of its
orientation: cubic elasticity, slip on the twelve systems of the FCC or the BCC
lattice, a critical resolved shear stress ``\tau_0`` and a viscous overstress.
On system ``s`` of the grain, of normal ``\underline{n}_s`` and direction
``\underline{m}_s`` rotated with it, the resolved shear stress is
``\tau_s = \boldsymbol{\sigma}:\operatorname{sym}(\underline{m}_s\otimes\underline{n}_s)``.
As the viscous overstress becomes small (the parameter ``K`` of the law), a grain
slips only on the systems whose resolved stress reaches ``\tau_0``.

## The Taylor model

Every grain undergoes the macroscopic strain:

```math
\boldsymbol{\varepsilon}_g = \boldsymbol{E},
\qquad
\boldsymbol{\Sigma} = \sum_g w_g\,\boldsymbol{\sigma}_g ,
\qquad
\frac{\partial\boldsymbol{\Sigma}}{\partial\boldsymbol{E}} = \sum_g w_g\,\frac{\partial\boldsymbol{\sigma}_g}{\partial\boldsymbol{E}} .
```

It is the [uniform-strain composite](@ref mp-theory-uniform-fields) of the
grains. Since a grain then sees nothing of the others, each is integrated on
its own: [`Polycrystal`](@ref) keeps the state of every grain in a vector and
calls the local integration of each, which scales to hundreds of grains, where
[`UniformStrain`](@ref) solves all the phases in one system of equations.

To follow a plastic strain imposed on it, a grain has to activate enough slip
systems to produce any isochoric strain, five independent ones in general. The
macroscopic flow stress in tension is then ``M\,\tau_0``, ``M`` being the Taylor
factor of the texture. For a random texture of FCC crystals,
[bishopHill1951](@citet) found ``M \approx 3.06``. In elasticity the Taylor model
gives the Voigt average of the rotated stiffnesses; for cubic crystals and a
random texture it is isotropic, of bulk modulus ``(C_{11} + 2C_{12})/3`` and shear
modulus ``(C_{11} - C_{12} + 3C_{44})/5``.

## A random texture

[`random_orientations`](@ref) draws orientations uniformly over the rotations.
With the Euler angles of TensND, ``\theta`` the angle of the third axis from
``\underline{e}_3`` and ``\phi``, ``\psi`` the two azimuths, the uniform measure is
``\sin\theta\,\mathrm{d}\theta\,\mathrm{d}\phi\,\mathrm{d}\psi``: ``\cos\theta``
uniform on ``[-1, 1]``, ``\phi`` and ``\psi`` uniform on ``[0, 2\pi)``. Drawing
``\theta`` itself uniformly would crowd the orientations near the poles.

## Grains in a matrix: Kröner

Under the Taylor model a grain cannot strain differently from its
neighbors, whatever their resistance. [kroner1961](@citet) lets each grain be a
spherical inclusion in a homogeneous matrix that stands for all the others.
Grains and matrix share the isotropic elasticity ``(k, \mu)``; the grain carries
its plastic strain ``\boldsymbol{\varepsilon}^p_g`` and the matrix the average
one, ``\boldsymbol{E}^p = \sum_g w_g\,\boldsymbol{\varepsilon}^p_g``. The difference
of the two is an eigenstrain in an elastic medium, and Eshelby's solution for
a sphere gives the strain of the grain, ``\boldsymbol{\varepsilon}_g - \boldsymbol{E} = \mathbb{S}^{E}:(\boldsymbol{\varepsilon}^p_g - \boldsymbol{E}^p)``.
For deviatoric plastic strains, it is the interaction law

```math
\boldsymbol{\sigma}_g = \boldsymbol{\Sigma} + 2\mu\,(1 - \beta)\,\big(\boldsymbol{E}^p - \boldsymbol{\varepsilon}^p_g\big),
\qquad
\beta = \frac{2(4 - 5\nu)}{15(1 - \nu)} .
```

A grain that slips more than the average is unloaded by the matrix, at the
rate of the elastic shear modulus. Plastic strains soon exceed the elastic
ones by far, and so the internal stresses grow until every grain slips almost
as the average: past the elastoplastic transition, Kröner's model comes back
close to Taylor's.

## A matrix that flows: Berveiller and Zaoui

The matrix stands for grains that flow too, and accommodates the misfit of a
grain plastically as well. [berveillerZaoui1978](@citet) give it an isotropic
secant elastoplastic behavior: its plastic strain proportional to its stress
deviator, ``\boldsymbol{E}^p = h\,\boldsymbol{S}``, so that its shear modulus
falls to ``\mu^* = \mu/(1 + 2\mu h)`` while its bulk modulus stays ``k``. The same
Eshelby problem, in this softer matrix, tempers the interaction by the plastic
accommodation factor ``\alpha``:

```math
\boldsymbol{\sigma}_g = \boldsymbol{\Sigma} + 2\alpha\mu\,(1 - \beta)\,\big(\boldsymbol{E}^p - \boldsymbol{\varepsilon}^p_g\big),
\qquad
\alpha = \frac{1 + 6\mu h\,(1 + \nu)/(7 - 5\nu)}
  {1 + 2\mu h\,(13 - 5\nu)/(15(1 - \nu)) + 8\mu^2h^2(1 + \nu)/(15(1 - \nu))},
```

equation (3.15) of the paper: ``\alpha = 1`` at the onset of plastic flow, where
the model is Kröner's, and ``\alpha \approx 1/(1 + \mu h)`` once the matrix
flows, small as soon as the plastic strain exceeds the elastic one.

## The interaction in the package

[`KronerModel`](@ref) and [`BerveillerZaouiModel`](@ref) write both rules in
strains, for grains of any law. With the shared elasticity ``\mathbb{C}``, the
plastic strain of a grain is what its stress leaves of its strain,
``\boldsymbol{\varepsilon}^p_g = \boldsymbol{\varepsilon}_g - \mathbb{C}^{-1}:\boldsymbol{\sigma}_g``,
and the grain strains as

```math
\boldsymbol{\varepsilon}_g = \boldsymbol{E} + \mathcal{C}:\big(\boldsymbol{\varepsilon}^p_g - \boldsymbol{E}^p\big),
\qquad
\mathcal{C} = \frac{3k}{3k + 4\mu^*}\,\mathbb{J} + \frac{\mu\beta^*}{\mu\beta^* + \mu^*(1 - \beta^*)}\,\mathbb{K},
```

``\beta^*`` being ``\beta`` for the moduli ``(k, \mu^*)``: ``\mu^* = \mu`` for
Kröner, the secant modulus for Berveiller and Zaoui. Its deviatoric part is
``1 - \alpha(1 - \beta)``, which gives back the two interaction laws above. The
grains are coupled only through ``\boldsymbol{E}^p``: [`Polycrystal`](@ref)
solves its six components by the trust-region method of SimpleNonlinearSolve,
each iterate integrating every grain on its own at the strain the interaction
gives it, itself the root of a trust-region iteration. Since
``\boldsymbol{\Sigma} = \mathbb{C}:(\boldsymbol{E} - \boldsymbol{E}^p)`` at the root,
the consistent tangent follows from the derivative of ``\boldsymbol{E}^p`` with
respect to ``\boldsymbol{E}``, by the implicit-function theorem.

Both papers assume grains of the isotropic elasticity of the matrix. Grains of
cubic elasticity are accepted, their elastic mismatch then entering the
eigenstrain, an approximation outside the papers. The secant matrix of
Berveiller and Zaoui is meant for proportional loading; otherwise ``h`` is read
as the ratio of the norms of ``\boldsymbol{E}^p`` and ``\boldsymbol{S}``.

## Steady viscoplastic flow: the tangent model

At large strains the elasticity of the grains is negligible, and the grains
can be taken rigid–viscoplastic: [`PowerLawCrystal`](@ref) slips on each system
at the rate ``\dot\gamma_s = \dot\gamma_0\,|\tau_s/\tau_0|^n\operatorname{sign}\tau_s``,
so that its strain rate ``\boldsymbol{d}(\boldsymbol{\sigma})`` is deviatoric and
homogeneous of degree ``n`` in its stress. Linearized at its current stress,
a grain has the tangent compliance ``\partial\boldsymbol{d}/\partial\boldsymbol{\sigma}``,
``n`` times its secant one; the aggregate, homogeneous of degree ``n`` as well,
has a tangent viscosity ``\eta_t = \eta_s/n``.

The tangent model of [molinari1987](@citet) lets every grain, a sphere, interact
with a homogeneous matrix that behaves as the aggregate linearized at its
current stress: through its tangent viscosity, since the stress around the
grain departs from the average. In their computations, as here, the matrix is
isotropic and incompressible, of viscosity ``\eta_t``, and Eshelby's solution
for a sphere gives the interaction law

```math
\boldsymbol{s}_g - \boldsymbol{S} = -3\,\eta_t\,\big(\boldsymbol{d}_g - \boldsymbol{D}\big),
\qquad
\eta_t = \frac{1}{n}\,\frac{\Sigma_{eq}}{3D_{eq}},
\qquad
\boldsymbol{S} = \sum_g w_g\,\boldsymbol{s}_g ,
```

``\boldsymbol{s}`` the stress deviators, ``\boldsymbol{D}`` the macroscopic strain
rate. Averaged over the grains the law gives ``\langle\boldsymbol{d}_g\rangle = \boldsymbol{D}``:
the closure on the stress is the whole self-consistency condition.
[`flow_stress`](@ref) solves it by the trust-region method of
SimpleNonlinearSolve on the six components of ``\boldsymbol{S}``, each grain
stress being the minimum of a strictly convex function (the dual potential of
the grain plus the quadratic term of the interaction), found by Newton's
method with Armijo's line search.
[lebensohnTome1993](@citet) keep the anisotropy of the matrix; they show that,
with the tangent viscosity in the interaction, the estimate tends to the
uniform stress (Sachs) as ``n`` grows, and [massonAffine2000](@citet) trace
this to the secant relation the model keeps between the stress and the strain
rate of each grain. The example below shows the trend.

## What is not covered

- The **Sachs** model, every grain at the macroscopic stress, is the
  [uniform-stress composite](@ref mp-theory-uniform-fields) of the grains, for a
  few of them. For an aggregate of nearly rigid–plastic grains it needs the
  stress of the step to stay below the flow stress of every grain while it is
  sought, which a strain-driven iteration does not ensure; it is not offered
  for large aggregates.
- The β rule of [cailletaudPilvin1994](@citet), Hill's self-consistent
  elastoplastic scheme with the anisotropic tangent of the aggregate, and the
  anisotropic matrix of [lebensohnTome1993](@citet) in the tangent model.

## [Example: a random FCC aggregate](@id mp-ex-polycrystal)

This example pulls an aggregate of copper-like grains of random orientations,
under the [Taylor model](@ref mp-theory-polycrystals), and recovers the Taylor
factor of a random FCC texture.

### The aggregate

````@example mp_polycrystals
using MeanFieldHomogenization, Random, Plots, ForwardDiff

τ₀ = 10.0                                                   # critical resolved shear stress, MPa
grain(frame) = CrystalViscoplasticity(; C₁₁ = 168.4e3, C₁₂ = 121.4e3, C₄₄ = 75.4e3,
    K = 0.05, n = 10, τ₀, frame)
frames = random_orientations(60; rng = MersenneTwister(2))
aggregate = Polycrystal(grain.(frames))
nothing #hide
````

| code | theory |
|:--|:--|
| `CrystalViscoplasticity(…; frame)` | the law of a grain, its cube axes along `frame` |
| `random_orientations(60)` | 60 orientations drawn uniformly over the rotations |
| `Polycrystal(grains)` | every grain at the macroscopic strain (Taylor), equal volumes |

### Tension

The aggregate is pulled along ``\underline{e}_1``, its lateral faces free:

````@example mp_polycrystals
r = point_response(aggregate, LoadingPath(range(0, 1; length = 26); ε11 = t -> 1e-2 * t))
plot(r[:ε11] * 100, r[:σ11] / τ₀; lw = 2, label = "Taylor aggregate, 60 grains",
    xlabel = "axial strain (%)", ylabel = "σ₁₁ / τ₀", legend = :bottomright)
hline!([3.06]; ls = :dash, label = "Taylor factor, Bishop and Hill (1951)")
````

The response is elastic until the grains reach their critical stresses, then
flows at about ``3.06\,\tau_0``: the Taylor factor of a random FCC texture. The
plateau is slightly above it, by the viscous overstress of the grains and the
sampling of 60 orientations:

````@example mp_polycrystals
r[:σ11][end] / τ₀
````

### Inside the aggregate

The state of every grain is kept. The number of slip systems that slipped in
each grain over the last step:

````@example mp_polycrystals
active(s_old, s_new) = count(abs.(s_new.y[1:12] - s_old.y[1:12]) .> 1e-8)
counts = active.(grain_states(r.states[end - 1]), grain_states(r.states[end]))
histogram(counts; bins = 0.5:1:12.5, label = "", xlabel = "active slip systems in a grain",
    ylabel = "number of grains")
````

Most grains slip on six or eight systems, a few on four. A rigid–plastic grain
following an imposed strain selects five independent systems; the viscous law
of the grains makes no such selection, and spreads the slip over every system
whose resolved stress reaches its critical value.

### Grains in a matrix: Taylor, Kröner, Berveiller–Zaoui

Both interaction models assume grains of the isotropic elasticity of the
matrix; the grains below have it (``C_{44} = (C_{11} - C_{12})/2``), with a
slower viscous law than above. Eight grains keep the page light:

````@example mp_polycrystals
E_y, ν = 120e3, 0.3                                                     # MPa
C₁₁, C₁₂ = E_y * (1 - ν) / ((1 + ν) * (1 - 2ν)), E_y * ν / ((1 + ν) * (1 - 2ν))
iso_grain(frame) = CrystalViscoplasticity(; C₁₁, C₁₂, C₄₄ = (C₁₁ - C₁₂) / 2, K = 1.0, n = 5, τ₀, frame)
few = random_orientations(8; rng = MersenneTwister(2))
pull = LoadingPath(range(0, 1; length = 7); ε11 = t -> 3e-3 * t)
q = plot(; xlabel = "axial strain (%)", ylabel = "σ₁₁ / τ₀", legend = :bottomright)
for (model, label) in ((TaylorModel(), "Taylor"), (KronerModel(; E = E_y, ν), "Kröner"),
                       (BerveillerZaouiModel(; E = E_y, ν), "Berveiller–Zaoui"))
    rm = point_response(Polycrystal(iso_grain.(few); model), pull)
    plot!(q, rm[:ε11] * 100, rm[:σ11] / τ₀; lw = 2, marker = :circle, label)
end
q
````

Kröner's matrix, elastic, opposes any misfit of a grain with the full elastic
shear modulus: past the onset of slip it holds the grains almost to the
average strain, and the response stays close to Taylor's. The elastoplastic
matrix of Berveiller and Zaoui yields with the grains, lets them deviate,
and lowers the flow stress of the aggregate.

### Steady flow: Taylor, tangent and Sachs

Rigid–viscoplastic grains of the same random texture, pulled at a constant
strain rate, for exponents from linear viscosity to nearly rate-independent
slip:

````@example mp_polycrystals
crystals(n) = ViscoplasticPolycrystal([PowerLawCrystal(; γ̇₀ = 1e-3, τ₀, n, frame) for frame in frames])
D = Mandel{Float64}(1e-3, -5e-4, -5e-4, 0, 0, 0)                  # 1/s, isochoric tension
# Sachs: every grain at the uniaxial stress s whose average axial rate is D₁₁.
function sachs(p)
    rate(s) = sum(w * crystal_strain_rate(c, Mandel(2s / 3, -s / 3, -s / 3, zero(s), zero(s), zero(s)))[1] for (c, w) in zip(p.grains, p.weights)) - D[1]
    s = 10τ₀
    for _ in 1:100
        s -= rate(s) / ForwardDiff.derivative(rate, s)
    end
    return s
end
exponents = [1, 2, 3, 5, 8, 12, 20]
taylor = [von_mises(flow_stress(crystals(n), D; model = TaylorModel()).σ) for n in exponents]
tangent = [von_mises(flow_stress(crystals(n), D).σ) for n in exponents]
lower = [sachs(crystals(n)) for n in exponents]
plot(exponents, [taylor tangent lower] / τ₀; lw = 2, marker = :circle, xscale = :log10,
    label = ["Taylor" "tangent (Molinari et al.)" "Sachs"], xlabel = "exponent n",
    ylabel = "Σ_eq / τ₀ at D₁₁ = γ̇₀", legend = :topleft)
````

Under linear viscosity the three differ by the heterogeneity of the grains
alone. As ``n`` grows, the Taylor estimate rises toward the Taylor factor, while
the tangent one bends back toward the uniform stress, as
[lebensohnTome1993](@citet) found: the tangent viscosity of the matrix, ``n``
times smaller than its secant one, lets the grains deviate more and more from
the average strain rate.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

