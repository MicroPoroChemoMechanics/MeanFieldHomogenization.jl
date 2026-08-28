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
      link: /manual/installation
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
    link: /manual/installation
  - icon: 🎓
    title: Tutorials
    details: The API by worked example, topic by topic, each page runnable end to end.
    link: /tutorials/
  - icon: 🧪
    title: Applications
    details: Full micromechanical models of real materials — cement paste, concrete, bituminous mixtures, clays.
    link: /applications/cement_paste
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
    r = RVE(:M)
    add_matrix!(r, Ellipsoid(1.0), Dict(:C => C_solid))
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
this problem and what replaces it.
