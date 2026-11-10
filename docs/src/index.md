```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "MeanFieldHomogenization.jl"
  text: "Effective properties of heterogeneous materials"
  tagline: One Eshelby chain — Hill tensor, concentration, scheme — generalized one direction at a time, in pure Julia and differentiable throughout.
  image:
    src: /logo.png
    alt: A representative volume element of ellipsoidal inclusions in a matrix
  actions:
    - theme: brand
      text: Get started
      link: /quickstart
    - theme: alt
      text: Theory
      link: /theory/
    - theme: alt
      text: API
      link: /api/elliptic
    - theme: alt
      text: View on GitHub
      link: https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl

features:
  - icon: 📐
    title: Theory
    details: One heterogeneity first — the Eshelby problem, cracks, layered patterns — then every homogenization scheme, one family per page, and the extensions to conductivity and viscoelasticity.
    link: /theory/
  - icon: 🧰
    title: Manual
    details: Every inclusion family, cell and scheme, with the call that produces it — from an ellipsoid to a neural surrogate.
    link: /manual/
  - icon: 🎓
    title: Tutorials
    details: The API by worked example, topic by topic, each page runnable in full.
    link: /tutorials/
  - icon: 🧪
    title: Applications
    details: Complete micromechanical models of real materials, each after a published one — cement paste and concrete, clays, sands and sandstones, bituminous mixtures.
    link: /applications/
  - icon: 🔁
    title: Tools and migration
    details: Build a model in the browser with MFH Studio, or port one from the Echoes C++/Python codebase.
    link: /tools/from_echoes
  - icon: 🏗️
    title: Finite-element coupling
    details: One microstructure per Gauss point, handing a structural code a stress, a consistent tangent and Biot coefficients.
    link: /fe_coupling/
  - icon: 📖
    title: API reference
    details: Every exported function, grouped by sub-module.
    link: /api/elliptic
  - icon: 🛠️
    title: Developer guide
    details: How the package is put together, and how to add an inclusion, an algorithm or a scheme without touching the rest.
    link: /developer/architecture
---
```

## What it does

Given a reference medium and an inclusion shape, `MeanFieldHomogenization` builds
the Hill polarization tensor ``\mathbb{P}`` (Eshelby's result), derives the
concentration tensor for each phase, and assembles a **homogenization scheme** —
dilute, Mori–Tanaka, self-consistent, differential, PCW, or the classical bounds
— into an effective stiffness or conductivity.

That chain is the whole library. The [reading path](@ref th-index) follows it:
first the problem of a single heterogeneity, an ellipsoid, a crack or a layered
pattern; then the schemes that assemble those solutions, one family per page,
from the bounds to the N-body schemes that resolve the interaction inclusion by
inclusion; and finally the chain **generalized in one direction at a time**, a
different physics for transport and a different time dependence for
viscoelasticity.

## [Lineage: Echoes](@id index-lineage-echoes)

`MeanFieldHomogenization` is the Julia successor of **Echoes** [echoes](@cite), the
mean-field homogenization library written in C++ with a Python interface by the same
author at Cerema, documented in its own
[manual](https://jfbarthelemy.github.io/echoes/). The Eshelby and Hill machinery,
the crack tensors, the ``n``-layer sphere, the schemes and the aging viscoelastic approach
were ported from it, with the methods of the articles it implements: the compliance and
the Hill polarization tensor of a crack in an anisotropic matrix [barthelemyIJSS2009](@cite),
the Hill polarization tensor of an ellipsoid and of a crack in an anisotropic conductor
[barthelemyTIPM2009](@cite), and the Eshelby problem and the schemes of aging linear
viscoelasticity [barthelemyIJSS2016, barthelemyIJES2019](@cite), discretized in time as
proposed by [sanahuja2013](@citet).

Echoes remains the reference of the port: the same problems are run by both libraries
in [Cross-validation](@ref dev-validation), and their timings are compared in
[Performance vs Echoes](@ref dev-benchmarks). The port extends it in six directions:

- **numbers of any type**: every computation accepts `ForwardDiff` dual numbers, so
  that the derivative of an effective property with respect to any parameter of the
  model (a fraction, an aspect ratio, a modulus at any scale, a time of a law) comes
  out of the same call, and symbolic numbers (`Symbolics`, `SymPy`), which give closed
  forms where the algorithm has no branch on a value;
- **closed forms**: the Hill tensor of a spheroid coaxial with a transversely isotropic
  matrix, derived by transformation [barthelemyIJES2020_hilltrans](@cite), the opening
  displacement tensor of an elliptic crack in a matrix transversely isotropic about its
  normal [barthelemyIJES2021](@cite), and the displacement and stress intensity factors
  of elliptic and ribbon cracks [barthelemyIJSS2026](@cite);
- **morphologies**: confocal ``n``-layer spheroids with imperfect interfaces
  [barthelemyBignonnetIJES2020](@cite), periodic laminates, and inclusions whose
  response comes from a finite-element solution or a trained network;
- **interactions**: the cluster model and the equivalent inclusion method, which act on
  the positions of the inclusions instead of a one-site average;
- **time**: beside the aging approach, the non-aging approach in the Laplace–Carson domain,
  with numerical inversion and a catalog of rheological models;
- **use**: homogenization as a constitutive law called by a finite-element code, the
  graphical interface MFH Studio, and [`echoes2mfh`](@ref tools-echoes2mfh), which
  translates an Echoes script.

[From Echoes to MeanFieldHomogenization](@ref tools-from-echoes) gives the
correspondence between the two libraries, class by class.

## Installation

```julia-repl
pkg> add MeanFieldHomogenization
```

Nothing else has to be installed by hand for numerical work, derivatives
included.

!!! tip "What to load for symbolic computations"
    | to work with | add | load |
    | :--- | :--- | :--- |
    | numbers and derivatives (`ForwardDiff`) | nothing | `using MeanFieldHomogenization` |
    | SymPy symbols (`Sym`: `symbols`, `@syms`) | `pkg> add SymPyPythonCall` | `using MeanFieldHomogenization, SymPyPythonCall` |
    | Symbolics variables (`Num`: `@variables`) | `pkg> add Symbolics` | `using MeanFieldHomogenization, Symbolics` |

    SymPyPythonCall is optional: loading it activates the SymPy methods of
    MeanFieldHomogenization and of TensND. The first time it loads, CondaPkg
    installs a private Python with `sympy` (internet needed, once).

## A first calculation

A porous solid, and every scheme the package ships, against the bounds that
frame them. The void is given a small but non-zero stiffness, which is what
lets the self-consistent branch stay on the physical solution:

```@example home
using MeanFieldHomogenization, TensND, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

C_solid = iso_stiffness(90.0, 30.0)     # GPa
C_void  = iso_stiffness(0.01, 0.005)

function bulk(f, scheme)
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C_solid); fraction = :rest)
    f > 0 && add_phase!(r, :V, Ellipsoid(1.0), Dict(:C => C_void); fraction = f)
    return first(k_mu(homogenize(r, scheme, :C)))
end

fs = range(0.0, 0.5; length = 61)
schemes = ("Voigt" => Voigt(),
           "Reuss" => Reuss(),
           "Mori-Tanaka" => MoriTanaka(),
           "Dilute (dual)" => DiluteDual(),
           "Self-consistent" => AsymmetricSelfConsistent(; abstol = 1.0e-10,
                                                           maxiters = 200,
                                                           select_best = true),
           "Differential" => DifferentialScheme(; nsteps = 100))

plt = plot(; xlabel = "porosity f", ylabel = "effective bulk modulus k [GPa]",
             legend = :topright, framestyle = :box, size = (760, 470))
for (name, s) in schemes
    plot!(plt, fs, [bulk(f, s) for f in fs]; label = name, lw = 2)
end
plt
```

Voigt and Reuss bracket the others; the three estimates between them differ by
how much of the load each void is assumed to see. [Porous materials](@ref tut-porous-materials) works through why the standard self-consistent scheme fails on
this problem and what replaces it, and [Getting started](@ref getting-started)
takes a single estimate more slowly, defining each object on the way.

## Reading paths

The documentation is organized by the question each chapter answers.
[Theory](@ref th-index) explains why an estimate is the right one, the
[Manual](@ref man-index) describes how each kind of object is written, the
[Tutorials](@ref tut-index) drive a calculation from start to finish, and the
Applications build complete models of real materials. Three entry points follow
from it, depending on what the reader already knows.

**New to mean-field homogenization.** The chapter [Theory](@ref th-index) is
written for this reader and is best read in the order it states, from
[The Eshelby inclusion problem](@ref th-eshelby-problem) to
[Homogenization schemes](@ref th-homogenization), which together contain the
whole chain that the rest generalizes. The tutorial
[A first homogenization](@ref tut-first-estimate) then puts it to work.

**Knowing what is to be computed.** [Getting started](@ref getting-started) and
[Schemes and RVEs](@ref man-schemes) are enough to write a first model; the
Manual then has one page per morphology, from
[Ellipsoidal inclusions](@ref man-ellipsoidal-inclusions) to
[Neural-surrogate inclusions](@ref man-neural-inclusions), and the Applications,
from [Multiscale elasticity of a hydrating cement paste](@ref app-cement-paste)
onwards, show complete models.

**Coming from Echoes.** The correspondence between the two libraries, class by
class, is in [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes),
and [`echoes2mfh`](@ref tools-echoes2mfh) translates an existing script.

**The symbols.** Every symbol of the formulas is listed in the
[Nomenclature](@ref nomenclature), with its meaning and its unit. Hovering an
equation on any page shows the symbols it holds, with their meaning on that
page.
