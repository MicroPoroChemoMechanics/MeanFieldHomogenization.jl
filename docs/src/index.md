```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "MeanFieldHomogenization.jl"
  text: "Effective properties of heterogeneous materials"
  tagline: One Eshelby chain — Hill tensor, localization, scheme — generalized one direction at a time, in pure Julia and differentiable throughout.
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
    details: The Eshelby/Hill chain in the order it is built, then the ways it is generalized — richer patterns, conductivity, viscoelasticity, periodicity, N-body.
    link: /theory/
  - icon: 🧰
    title: Manual
    details: Every inclusion family, cell and scheme, with the call that produces it — from an ellipsoid to a neural surrogate.
    link: /manual/
  - icon: 🎓
    title: Tutorials
    details: The API by worked example, topic by topic, each page runnable end to end.
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
localization tensor for each phase, and assembles a **homogenization scheme** —
dilute, Mori–Tanaka, self-consistent, differential, PCW, or the classical bounds
— into an effective stiffness or conductivity.

That chain is the whole library. Everything else is the chain **generalized in
one direction at a time**: a richer morphological pattern in place of the
ellipsoid, a degenerate one for cracks, a different physics for transport, a
different time dependence for viscoelasticity, a periodic problem for the
multilayer, and no one-site assumption at all for the N-body schemes. The
[reading path](@ref th-index) takes them in that order.

`MeanFieldHomogenization` is a pure-Julia reimplementation of the Eshelby/Hill
machinery of the [Echoes](https://jfbarthelemy.github.io/echoes/) C++/Python
codebase; see [From Echoes to MeanFieldHomogenization](@ref tools-from-echoes)
for the translation guide.

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
