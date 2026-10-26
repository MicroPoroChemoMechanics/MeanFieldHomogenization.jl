# [A composite of nonlinear phases](@id mp-ex-composite)

This example computes the response of a composite whose two phases follow
nonlinear laws. It compares the [two secant methods](@ref mp-theory-secant)
with each other and with a bound. The theory page explains each step. This page
shows how each one is done, and checks the result where a check is available.

## The composite

A soft metallic matrix holds 30 % of stiffer spherical particles. Both follow a
power law ([`PowerLawElasticity`](@ref)). The matrix hardens little
(``m = 0.2``, close to perfect plasticity). The particles are four times
stronger and harden more (``m = 0.6``).

```@example mp_ex_composite
using MeanFieldHomogenization, TensND, Plots

f = 0.3                                                              # particle fraction
matrix = PowerLawElasticity(; k = 60e3, σ₀ = 80.0, ε₀ = 1e-3, m = 0.2)      # MPa
particles = PowerLawElasticity(; k = 150e3, σ₀ = 320.0, ε₀ = 1e-3, m = 0.6)
nothing # hide
```

The microstructure is a cell of the package, as for a linear composite. The
stiffness given to each phase is only a placeholder: the method replaces it by
the secant stiffness of the phase.

```@example mp_ex_composite
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
laws = Dict(:M => matrix, :P => particles)

modified = NonlinearHomogenized(rve, laws)                                   # Mori–Tanaka, second moments
classical = NonlinearHomogenized(rve, laws; method = ClassicalSecant())      # Mori–Tanaka, averages
nothing # hide
```

| code | theory |
|:--|:--|
| `rve` | the microstructure of the linear comparison composite: fractions and shapes |
| `laws` | the law of each nonlinear phase, keyed by the phase name |
| `NonlinearHomogenized(rve, laws)` | the fixed point on the effective strains ``\bar\varepsilon_i``, as a law of `MaterialPoints` |
| `method` | the measure of the phase strain: second moment (default) or average |
| `scheme` (not given here) | the linear scheme, Mori–Tanaka by default |

## A shear test

The macroscopic strain is an isochoric extension
``\boldsymbol{E} = e\,(1, -\tfrac12, -\tfrac12)``, whose equivalent strain is ``e``.
Every component of the strain is prescribed:

```@example mp_ex_composite
e_max = 5e-3
path = LoadingPath(range(0, 1; length = 26); ε11 = t -> e_max * t,
    ε22 = t -> -e_max * t / 2, ε33 = t -> -e_max * t / 2, ε23 = 0, ε13 = 0, ε12 = 0)
r_mod = point_response(modified, path)
r_cla = point_response(classical, path)
nothing # hide
```

Uniform strain in every phase gives an upper bound, the Voigt bound: each phase
is driven along the same path, and the stresses are averaged.

```@example mp_ex_composite
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
```

The curves are in the order the theory predicts:

- both estimates lie between the matrix alone and the Voigt bound;
- the modified secant estimate lies below the classical one, because the
  second moment of the strain is never below the strain of the average, and the
  matrix softens as its strain grows.

The bound is far above both estimates. Uniform strain forces the strong
particles to deform as much as the matrix, whereas in the composite the
particles strain much less.

## The strains of the phases

The effective strains are the internal variables of the law, named after the
phases:

```@example mp_ex_composite
plot(r_mod[:ε11] * 1e3, [r_mod[:M] r_mod[:P]] * 1e3; lw = 2,
    label = ["matrix, modified" "particles, modified"],
    xlabel = "macroscopic equivalent strain (‰)", ylabel = "effective strain of the phase (‰)",
    legend = :topleft)
plot!(r_cla[:ε11] * 1e3, [r_cla[:M] r_cla[:P]] * 1e3; lw = 2, ls = :dash,
    label = ["matrix, classical" "particles, classical"])
```

At the end of the path:

```@example mp_ex_composite
(matrix_modified = r_mod[:M][end], matrix_classical = r_cla[:M][end], macroscopic = e_max)
```

The matrix takes most of the deformation, more than the macroscopic strain
itself. Its second moment exceeds its average, which is the fluctuation the
classical method does not see.

## Checks

**Linear phases give back the linear scheme.** With ``m = 1`` a power law is
linear elasticity of shear modulus ``\sigma_0/(3\varepsilon_0)``. The secant
moduli are then constant, the fixed point is trivial, and the tangent of the
composite must be the Mori–Tanaka stiffness itself:

```@example mp_ex_composite
linear(μ, k) = PowerLawElasticity(; k, σ₀ = 3μ * 1e-3, ε₀ = 1e-3, m = 1.0)
rve_lin = RVE()
add_phase!(rve_lin, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 60e3, 2 * 20e3)); fraction = :rest)
add_phase!(rve_lin, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 150e3, 2 * 80e3)); fraction = f)
h = NonlinearHomogenized(rve_lin, Dict(:M => linear(20e3, 60e3), :P => linear(80e3, 150e3)))
E = Mandel{Float64}(1e-3, 2e-4, -3e-4, 1e-4, 0, 2e-4)
σ, 𝕂, _ = point_update(h, E, initial_state(h), 0.0)
𝕂 ≈ KM(homogenize(rve_lin, MoriTanaka(), :C))
```

**The tangent is consistent.** The contract checker compares the tangent from
the implicit-function theorem with the derivative of the stress:

```@example mp_ex_composite
check_material_interface(modified; verbose = false)
```

## Going further

- Any linear scheme can replace Mori–Tanaka, for example
  `NonlinearHomogenized(rve, laws; scheme = SelfConsistent())`, with the
  assumptions on the microstructure that the scheme carries.
- The composite is a law like any other. It runs at the quadrature points of a
  structure in the same way as the law of the
  [finite-element quickstart](@ref fe-quickstart).
