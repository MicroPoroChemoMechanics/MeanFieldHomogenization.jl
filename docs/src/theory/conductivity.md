# [Extension to conductivity](@id th-conductivity)

!!! info "Before this page"
    [The Eshelby inclusion problem](@ref th-eshelby-problem) §3, where the
    transport form of the inclusion problem is stated, and
    [Localization and contribution tensors](@ref th-localization) §4 for the
    order-2 routines named below.

The foundations and the schemes were written above for elasticity, where the
unknown is a displacement and the constitutive tensor has order four. The same
machinery solves a second family of problems, in which the unknown is a scalar
potential and the constitutive tensor has order two. Nothing has to be
rederived: the Eshelby problem, the Hill tensor, the localization tensors and
every scheme carry over term by term.

This chapter states the correspondence once, so that the other chapters can
give their order-2 counterparts in a line instead of arguing for them.

## 1. One problem, several physics

The order-2 problem is a linear elliptic equation for a scalar potential whose
gradient drives a flux. Four classical theories are that equation with
different names on the symbols:

| Theory       | Potential          | Flux                    | ``\boldsymbol{K}``        |
| :----------- | :----------------- | :---------------------- | :------------------------ |
| Fourier      | temperature        | heat flux               | thermal conductivity      |
| Fick         | concentration      | species flux            | diffusivity               |
| Darcy        | pressure head      | filtration velocity     | permeability              |
| Ohm          | electric potential | current density         | electrical conductivity   |

`MeanFieldHomogenization` implements the mathematics, not the physics, so a
single set of routines serves all four: what the caller supplies is a symmetric
positive-definite ``\boldsymbol{K}``, and what the caller reads back is an
effective ``\boldsymbol{K}^{\mathrm{hom}}`` in the same units. The chemical
route of a hydrating paste and the permeability of a fractured rock use the same
`conductivity_contribution` as a heat-conduction estimate.

## 2. The one asymmetry, and how it is removed

Hooke's law carries no minus sign; Fourier's and Fick's do. Left alone, that
single difference makes order-2 and order-4 formulas look as though they
disagreed on a sign. The convention fixed in
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q) takes
as the stress analog **minus** the flux,

```math
\boldsymbol{\sigma} \;\equiv\; -\,\underline{q} \;=\; \boldsymbol{K}\cdot\nabla T ,
```

so that ``\boldsymbol{\sigma}\cdot\underline{n}`` is, in both theories, what the
exterior transmits to the interior across a surface. With that substitution the
dictionary below is literal: every entry on the right is the entry on the left
with the symbols renamed, and no sign is ever flipped.

## 3. The dictionary

| Elasticity — order 4                                    | Conductivity — order 2                                |
| :------------------------------------------------------ | :---------------------------------------------------- |
| displacement ``\underline{u}`` — vector                  | potential ``T`` — scalar                              |
| strain ``\boldsymbol{\varepsilon}`` — 2-tensor           | gradient ``\nabla T`` — vector                        |
| stress ``\boldsymbol{\sigma}`` — 2-tensor                | ``\boldsymbol{\sigma} \equiv -\underline{q}`` — vector |
| stiffness ``\mathbb{C}`` — 21 components                 | conductivity ``\boldsymbol{K}`` — 6 components         |
| Hill tensor ``\mathbb{P}`` — 4-tensor                    | Hill tensor ``\boldsymbol{P}`` — 2-tensor              |
| Eshelby tensor ``\mathbb{S}^{\mathrm{E}} = \mathbb{P}:\mathbb{C}``    | ``\boldsymbol{S}^{\mathrm{E}} = \boldsymbol{P}\cdot\boldsymbol{K}`` |
| localization ``\mathbb{A}``, contribution ``\mathbb{N}`` | ``\boldsymbol{A}``, ``\boldsymbol{N}``                 |

The routine names follow the same rule, and the correspondence is listed in
full in [Localization and contribution tensors](@ref th-localization) §4: each
`strain`/`stress` becomes a `gradient`/`flux`, each `stiffness` a
`conductivity`, each `compliance` a `resistivity`.

## 4. What transposes untouched, and what does not

**The schemes transpose entirely.** Dilute, Mori-Tanaka, self-consistent,
differential, the bounds, the N-body models: all of them are written on
localization and contribution tensors, never on the order of the constitutive
tensor. Dispatching on a 2-tensor is enough — see
[Homogenization schemes](@ref th-homogenization).

**Two things genuinely differ, and both are in the order-2 problem's favor.**

The acoustic tensor of the order-4 problem is a sextic polynomial in the wave
direction, which is why an arbitrarily anisotropic ``\mathbb{P}`` needs the
residue algorithm. Its order-2 counterpart is the quadratic form
``\underline{\xi}\cdot\boldsymbol{K}\cdot\underline{\xi}``, so
``\boldsymbol{P}`` has a **closed form at any matrix anisotropy**
[willis1977](@cite) — no quadrature, no residues. The derivation is in
[Hill polarization tensors](@ref th-hill-tensors).

The order-2 problem also carries fewer modes. A crack opens in three modes in
elasticity, hence a 6-component opening tensor; in conduction only the normal
flux jumps, so a single scalar suffices, and it carries all the anisotropy of
the matrix. That reduction is worked through in
[Thermal cracks](@ref th-thermal-cracks).

## Where to go next

The next chapter changes the time dependence rather than the physics:
[The Laplace-Carson route](@ref th-laplace-carson) reduces a non-aging
viscoelastic problem to an elastic one at each value of a transform variable.
The order-2 counterparts stated on this page are developed on the following
pages.

- [Hill polarization tensors](@ref th-hill-tensors) — the closed form of
  ``\boldsymbol{P}`` and its dispatch.
- [Localization and contribution tensors](@ref th-localization) — the full
  table of order-2 routine names.
- [Thermal cracks](@ref th-thermal-cracks) — the flat-inclusion limit in
  conduction, and its intensity factor.
- [Conductivity](@ref man-conductivity) — how to call all of this.
