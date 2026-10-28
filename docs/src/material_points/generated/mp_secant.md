```@meta
EditURL = "../../../../scripts/material_points/10_secant.jl"
```

# [Homogenizing nonlinear phases: secant methods](@id mp-theory-secant)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/material_points/10_secant.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/material_points/10_secant.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/material_points/10_secant.jl`, runs it end to end.

The laws of this chapter so far describe one material. This page turns to a
composite whose phases are each described by such a law, and asks for the law
of the composite. When the phases are linear, the schemes of the package
answer exactly within their assumptions: the effective stiffness
``\mathbb{C}^{\mathrm{hom}}`` follows from the stiffnesses of the phases, their
fractions and their shapes. When a phase is nonlinear, its stiffness changes
from point to point with the local strain, and no scheme applies as it stands.
The secant methods bring the problem back to a sequence of linear ones. This
page derives them; [the example below](@ref mp-ex-composite) runs them.

## The phases

Each phase ``i``, of volume fraction ``f_i``, is a nonlinear elastic material
with no internal variable: its stress depends on the current strain only. It is
isotropic, so that its energy depends on the strain through two invariants
only, the volume change ``\operatorname{tr}\boldsymbol{\varepsilon}`` and the
equivalent strain
``\varepsilon_{\mathrm{eq}} = \sqrt{\tfrac23\,\boldsymbol{e}:\boldsymbol{e}}``,
``\boldsymbol{e}`` being the deviator of ``\boldsymbol{\varepsilon}``. The
power-law material ([`PowerLawElasticity`](@ref)) is the standard example, a
model of the deformation theory of plasticity:

```math
\boldsymbol{\sigma} = k_i\,(\operatorname{tr}\boldsymbol{\varepsilon})\,\boldsymbol{1}
+ \frac{2}{3}\,\frac{\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}})}{\varepsilon_{\mathrm{eq}}}\,\boldsymbol{e},
\qquad
\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}}) = \sigma_0\left(\frac{\varepsilon_{\mathrm{eq}}}{\varepsilon_0}\right)^{m}.
```

## Every such law is linear at a frozen strain

The stress above is a linear function of the strain whose coefficients are
evaluated at the strain itself. Writing the isotropic stiffness with the
projectors ``\mathbb{J}`` on volume changes and ``\mathbb{K}`` on deviators,

```math
\boldsymbol{\sigma} = \mathbb{C}^{s}_i(\varepsilon_{\mathrm{eq}}):\boldsymbol{\varepsilon},
\qquad
\mathbb{C}^{s}_i(\varepsilon_{\mathrm{eq}}) = 3k_i\,\mathbb{J} + 2\mu^{s}_i(\varepsilon_{\mathrm{eq}})\,\mathbb{K},
\qquad
\mu^{s}_i(\varepsilon_{\mathrm{eq}}) = \frac{\sigma_{\mathrm{eq}}(\varepsilon_{\mathrm{eq}})}{3\,\varepsilon_{\mathrm{eq}}}.
```

``\mu^{s}_i`` is the **secant shear modulus**: the slope of the straight line
from the origin to the current point of the shear curve, not the slope of the
curve (the tangent modulus). For a law that is linear in the volume change,
``k_i`` is a constant. [`secant_moduli`](@ref) reads both moduli on any law of
the strain alone, from its response to a deviatoric strain and to a volume
change. It needs no formula from the law.

This rewriting is exact, but it does not yet help: in a composite the strain
varies inside each phase, so the secant stiffness varies too, and the phase is
not a homogeneous linear material.

## The linear comparison composite

The approximation of every secant method is to give the whole phase ``i`` a
single secant stiffness, evaluated at one **effective strain**
``\bar\varepsilon_i`` that represents the strain field of the phase:

```math
\mathbb{C}_i = \mathbb{C}^{s}_i(\bar\varepsilon_i).
```

The phases are then homogeneous and linear: they form a *linear comparison
composite*, which any linear scheme homogenizes into
``\mathbb{C}^{\mathrm{hom}}(\bar\varepsilon_1, \ldots, \bar\varepsilon_n)``. The
macroscopic stress at the macroscopic strain ``\boldsymbol{E}`` is

```math
\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}(\bar\varepsilon_1, \ldots, \bar\varepsilon_n):\boldsymbol{E}.
```

The effective strains are not known beforehand. They follow from the strain
field of the comparison composite, which depends on them. They are therefore
the solution of a **fixed point**:

```math
\bar\varepsilon_i = \mathcal{M}_i\big(\boldsymbol{E};\ \bar\varepsilon_1, \ldots, \bar\varepsilon_n\big),
\qquad i = 1, \ldots, n,
```

where ``\mathcal{M}_i`` is the measure of the strain of phase ``i`` in the
comparison composite. The choice of that measure is what distinguishes one
secant method from another.

## The classical secant method: the average strain

The simplest measure takes the equivalent strain of the **average** strain of
the phase. A linear scheme gives that average through the strain
concentration tensor ``\mathbb{A}_i`` of the phase,
``\langle\boldsymbol{\varepsilon}\rangle_i = \mathbb{A}_i:\boldsymbol{E}``, so that

```math
\bar\varepsilon_i = \varepsilon_{\mathrm{eq}}\big(\mathbb{A}_i:\boldsymbol{E}\big).
```

This is [`ClassicalSecant`](@ref). The average hides the fluctuations of the
strain inside the phase: a strain that is large in some regions and small in
others can average to a small value. In a matrix that wraps around stiff
particles, for example, the strain concentrates near the particles, and the
classical measure underestimates the strain that sets the secant modulus.

## The modified secant method: the second moment

A measure that sees the fluctuations is the **second moment** of the
equivalent strain over the phase,
``\bar\varepsilon_i^2 = \langle\varepsilon_{\mathrm{eq}}^2\rangle_i``. A linear
scheme does not give second moments directly, but they follow from the energy
of the comparison composite, as the following argument shows.

The energy stored per unit volume in the comparison composite under the
macroscopic strain ``\boldsymbol{E}`` is, by the lemma of Hill, both the
macroscopic energy and the average of the local energies:

```math
W = \frac12\,\boldsymbol{E}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{E}
= \sum_j f_j\,\Big\langle \frac12\,\boldsymbol{\varepsilon}:\mathbb{C}_j:\boldsymbol{\varepsilon} \Big\rangle_j .
```

The local strain field minimizes the energy among the kinematically admissible
fields. Change the shear modulus ``\mu_i`` of phase ``i`` alone. The energy
changes for two reasons: the explicit change of ``\mathbb{C}_i``, and the change
of the strain field. The second contributes nothing to first order, because
the field is a minimum. Only the explicit dependence remains, and
``\partial\mathbb{C}_i/\partial(2\mu_i) = \mathbb{K}`` acts only on phase ``i``:

```math
\frac{\partial W}{\partial(2\mu_i)}
= f_i\,\Big\langle \frac12\,\boldsymbol{\varepsilon}:\mathbb{K}:\boldsymbol{\varepsilon} \Big\rangle_i
= \frac{f_i}{2}\,\langle\boldsymbol{e}:\boldsymbol{e}\rangle_i
= \frac{3 f_i}{4}\,\langle\varepsilon_{\mathrm{eq}}^2\rangle_i .
```

With ``W`` written as the macroscopic quadratic form, this gives the second
moment from the derivative of the effective stiffness alone:

```math
\bar\varepsilon_i^2 = \langle\varepsilon_{\mathrm{eq}}^2\rangle_i
= \frac{2}{3 f_i}\;\boldsymbol{E}:\frac{\partial\mathbb{C}^{\mathrm{hom}}}{\partial(2\mu_i)}:\boldsymbol{E}.
```

This identity between the energy stored in a phase and the derivative of the
effective moduli is due to [kreher1990](@citet). It is exact for a scheme that
is the exact solution of some microstructure, and it serves as the definition
of the second moment for a scheme used as an estimate. This is
[`ModifiedSecant`](@ref), the default. The package computes the derivative of
``\mathbb{C}^{\mathrm{hom}}`` by automatic differentiation through the scheme:
nothing specific to a scheme is written.

!!! note "The second moment is never below the first"
    The squared equivalent strain is a convex quadratic function of the strain,
    so by Jensen's inequality
    ``\langle\varepsilon_{\mathrm{eq}}^2\rangle_i \ge \varepsilon_{\mathrm{eq}}(\langle\boldsymbol{\varepsilon}\rangle_i)^2``,
    with equality only for a strain uniform in the phase. For the same
    comparison composite the modified effective strain is therefore at least
    the classical one. A phase whose secant modulus decreases with the strain
    (``m < 1``) is then softer under the modified method, and the composite is
    expected to be softer as well, which [the example below](@ref mp-ex-composite)
    shows.

The modified method has a further standing. For phases whose energy is a
concave function of ``\varepsilon_{\mathrm{eq}}^2``, as for the power law with
``m \le 1``, the variational procedure of [ponteCastaneda1991](@citet) reaches
the same equations. That procedure expresses the effective energy of the
nonlinear composite through an optimization over linear comparison composites,
and gives a bound when the linear estimate it uses is itself a bound. The review of
[suquet1997](@citet) sets out this connection and the other secant and
variational estimates.

## The composite as a law of `MaterialPoints`

The fixed point has one unknown per nonlinear phase. Written as a residual,

```math
R_i\big(\bar\varepsilon_1, \ldots, \bar\varepsilon_n;\ \boldsymbol{E}\big)
= \bar\varepsilon_i - \mathcal{M}_i\big(\boldsymbol{E};\ \bar\varepsilon_1, \ldots, \bar\varepsilon_n\big) = 0,
```

it has exactly the form of the [local problem of a law](@ref mp-theory): the
effective strains play the role of the internal variables, the macroscopic
strain that of the gradient, and the macroscopic stress that of the flux.
[`NonlinearHomogenized`](@ref) builds that law. Everything the chapter says
about a law then applies to the composite:

- the fixed point is solved by Newton's method, with the Jacobian
  ``\partial R_i/\partial\bar\varepsilon_j`` obtained by differentiating the
  scheme itself;
- the **consistent tangent** of the composite,
  ``\mathrm{d}\boldsymbol{\Sigma}/\mathrm{d}\boldsymbol{E}``, comes from the
  implicit-function theorem. It contains the variation of the effective
  strains with ``\boldsymbol{E}``, which a tangent read on the frozen
  comparison composite (``\mathbb{C}^{\mathrm{hom}}`` alone) would miss;
- the composite can be driven along any path by [`point_response`](@ref), and it
  can be used at the quadrature points of a [finite-element computation](@ref fe-quickstart).

Because the phases have no internal variable, the composite has no history
either. The effective strains are recomputed at every step, the previous ones
serving only as the starting point of the Newton iteration.

## Scope

| | |
|:--|:--|
| phases | isotropic, laws of the strain alone (nonlinear elasticity, deformation theory); a phase without a law keeps its linear stiffness |
| schemes | any linear scheme of the package, with its own assumptions on the microstructure |
| effective strain | one per nonlinear phase: the deviatoric response is secant, the bulk response is read at the same strain size |
| not covered | phases with internal variables (plasticity by increments, viscosity), for which the [incremental approach](@ref mp-theory-incremental-tangent) is needed; for phases without, the [affine formulation](@ref mp-theory-affine) linearizes by the tangent instead of the secant |

## [Example: a composite of nonlinear phases](@id mp-ex-composite)

This example computes the response of a composite whose two phases follow
nonlinear laws. It compares the [two secant methods](@ref mp-theory-secant)
with each other and with a bound. The theory page explains each step. This page
shows how each one is done, and checks the result where a check is available.

### The composite

A soft metallic matrix holds 30 % of stiffer spherical particles. Both follow a
power law ([`PowerLawElasticity`](@ref)). The matrix hardens little
(``m = 0.2``, close to perfect plasticity). The particles are four times
stronger and harden more (``m = 0.6``).

````@example mp_secant
using MeanFieldHomogenization, TensND, Plots

f = 0.3                                                              # particle fraction
matrix = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 0.2)      # MPa
particles = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 0.6)
nothing #hide
````

The microstructure is a cell of the package, as for a linear composite. The
stiffness given to each phase is only a placeholder: the method replaces it by
the secant stiffness of the phase.

````@example mp_secant
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
laws = Dict(:M => matrix, :P => particles)

modified = NonlinearHomogenized(rve, laws)                                   # Mori–Tanaka, second moments
classical = NonlinearHomogenized(rve, laws; method = ClassicalSecant())      # Mori–Tanaka, averages
nothing #hide
````

| code | theory |
|:--|:--|
| `rve` | the microstructure of the linear comparison composite: fractions and shapes |
| `laws` | the law of each nonlinear phase, keyed by the phase name |
| `NonlinearHomogenized(rve, laws)` | the fixed point on the effective strains ``\bar\varepsilon_i``, as a law of `MaterialPoints` |
| `method` | the measure of the phase strain: second moment (default) or average |
| `scheme` (not given here) | the linear scheme, Mori–Tanaka by default |

### A shear test

The macroscopic strain is an isochoric extension
``\boldsymbol{E} = e\,(1, -\tfrac12, -\tfrac12)``, whose equivalent strain is ``e``.
Every component of the strain is prescribed:

````@example mp_secant
e_max = 5e-3
path = LoadingPath(range(0, 1; length = 26); ε11 = t -> e_max * t,
    ε22 = t -> -e_max * t / 2, ε33 = t -> -e_max * t / 2, ε23 = 0, ε13 = 0, ε12 = 0)
r_mod = point_response(modified, path)
r_cla = point_response(classical, path)
nothing #hide
````

Uniform strain in every phase gives an upper bound, the Voigt bound: each phase
is driven along the same path, and the stresses are averaged.

````@example mp_secant
r_M = point_response(matrix, path)
r_P = point_response(particles, path)
σ_eq(r) = r[:σ11] - r[:σ22]                       # the von Mises stress on this path
voigt = (1 - f) * σ_eq(r_M) + f * σ_eq(r_P)

plot(r_mod[:ε11] * 1e3, σ_eq(r_mod); lw = 2, label = "modified secant",
    xlabel = "macroscopic equivalent strain (‰)", ylabel = "macroscopic equivalent stress (MPa)",
    legend = :topleft)
plot!(r_cla[:ε11] * 1e3, σ_eq(r_cla); lw = 2, label = "classical secant")
plot!(r_M[:ε11] * 1e3, voigt; lw = 2, ls = :dash, label = "uniform strain (Voigt bound)")
plot!(r_M[:ε11] * 1e3, σ_eq(r_M); lw = 2, ls = :dot, label = "matrix alone")
````

The curves are in the order the theory predicts:

- both estimates lie between the matrix alone and the Voigt bound;
- the modified secant estimate lies below the classical one, because the
  second moment of the strain is never below the strain of the average, and the
  matrix softens as its strain grows.

The bound is far above both estimates. Uniform strain forces the strong
particles to deform as much as the matrix, whereas in the composite the
particles strain much less.

### The strains of the phases

The effective strains are the internal variables of the law, named after the
phases:

````@example mp_secant
plot(r_mod[:ε11] * 1e3, [r_mod[:M] r_mod[:P]] * 1e3; lw = 2,
    label = ["matrix, modified" "particles, modified"],
    xlabel = "macroscopic equivalent strain (‰)", ylabel = "effective strain of the phase (‰)",
    legend = :topleft)
plot!(r_cla[:ε11] * 1e3, [r_cla[:M] r_cla[:P]] * 1e3; lw = 2, ls = :dash,
    label = ["matrix, classical" "particles, classical"])
````

At the end of the path, in ‰:

````@example mp_secant
map(x -> round(1e3 * x; digits = 2), (matrix_modified = r_mod[:M][end], matrix_classical = r_cla[:M][end], macroscopic = e_max))
````

The matrix takes most of the deformation, more than the macroscopic strain
itself. Its second moment exceeds its average, which is the fluctuation the
classical method does not see.

### Checks

**Linear phases give back the linear scheme.** With ``m = 1`` a power law is
linear elasticity of shear modulus ``\sigma_0/(3\varepsilon_0)``. The secant
moduli are then constant, the fixed point is trivial, and the tangent of the
composite must be the Mori–Tanaka stiffness itself:

````@example mp_secant
linear(μ, k) = PowerLawElasticity(; k, σ₀ = 3μ * 1e-3, ε₀ = 1e-3, m = 1.0)
rve_lin = RVE()
add_phase!(rve_lin, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60e3, 2 * 20e3)); fraction = :rest)
add_phase!(rve_lin, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 150e3, 2 * 80e3)); fraction = f)
h = NonlinearHomogenized(rve_lin, Dict(:M => linear(20e3, 60e3), :P => linear(80e3, 150e3)))
E = Mandel{Float64}(1e-3, 2e-4, -3e-4, 1e-4, 0, 2e-4)
σ, 𝕂, _ = point_update(h, E, initial_state(h), 0.0)
𝕂 ≈ KM(homogenize(rve_lin, MoriTanaka(), :C))
````

**The tangent is consistent.** The contract checker compares the tangent from
the implicit-function theorem with the derivative of the stress:

````@example mp_secant
check_material_interface(modified; verbose = false)
````

### Going further

- Any linear scheme can replace Mori–Tanaka, for example
  `NonlinearHomogenized(rve, laws; scheme = SelfConsistent())`, with the
  assumptions on the microstructure that the scheme carries.
- The composite is a law like any other. It runs at the quadrature points of a
  structure in the same way as the law of the
  [finite-element quickstart](@ref fe-quickstart).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

