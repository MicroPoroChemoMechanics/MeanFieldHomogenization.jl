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

## Quick example

```julia
using MeanFieldHomogenization, TensND

# Isotropic matrix, bulk/shear moduli (k, μ) = (30, 10) GPa
C₀ = iso_stiffness(30.0, 10.0)

# Hill polarization tensor for a spherical inclusion
P = hill_tensor(Ellipsoid(1.0), C₀)

# A porous material, 10 % spherical voids, homogenized by Mori-Tanaka
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C₀))
add_phase!(rve, :V, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.01, 0.005)); fraction = 0.1)
k_eff, μ_eff = k_mu(homogenize(rve, MoriTanaka(), :C))
```

Every entry point differentiates through `ForwardDiff` and accepts symbolic
(`SymPy`/`Symbolics`) coefficients out of the box — see
[Derivatives and sensitivities](tutorials/sensitivities.md) and
[Symbolic spheres](tutorials/symbolic_spheres.md).
