# [Nomenclature](@id nomenclature)

This page lists the symbols of the formulas of the documentation, grouped by
subject. A symbol that means different things in different chapters is listed
once for each meaning, with the pages where that meaning holds; a meaning listed
without pages holds everywhere else. Hovering an equation on any page shows the
symbols it holds, with their meaning on that page.

The conventions that are not a matter of symbols, namely the sign of the
analogy between elasticity and conduction, the sign of the Green operator, the
bases of isotropic and transversely isotropic tensors, the geometry of an
ellipsoid and the storage of tensors, are set out in
[Conventions](@ref th-notation).

## Typography

The order of a tensor is carried by its typeface, as in the
[Echoes manual](https://jfbarthelemy.github.io/echoes/), so that formulas can be
compared side by side with it.

| Object | Typeset as | Example |
| :----- | :--------- | :------ |
| scalar | italic | ``k``, ``\mu``, ``\nu``, ``\omega``, ``\eta``, ``\chi`` |
| vector (order 1) | underlined | ``\underline{u}``, ``\underline{n}``, ``\underline{\xi}`` |
| tensor of order 2 | bold | ``\boldsymbol{A}``, ``\boldsymbol{B}``, ``\boldsymbol{K}``, ``\boldsymbol{\sigma}``, ``\boldsymbol{\varepsilon}`` |
| tensor of order 4 | blackboard bold | ``\mathbb{C}``, ``\mathbb{P}``, ``\mathbb{Q}``, ``\mathbb{H}``, ``\mathbb{S}`` |
| set, geometry, shape function | calligraphic | ``\mathcal{E}_{\boldsymbol{A}}``, ``\mathcal{G}_i``, ``\mathcal{T}_a``, ``\mathcal{K}_\eta`` |

One class of object is deliberately outside that table: the **column arrays and
matrices of an algebraic formalism**, which are not tensors and carry no tensor
order. Such are the two-component state vector ``\mathbf s(r) = (u_r, \sigma_{rr})``
and the transfer matrices ``\mathbf T``, ``\mathbf J`` of the
[layered sphere](@ref th-layered-sphere), or the column of phase fractions
``[f]`` and the column of ones ``\mathbf U`` of the
[differential scheme](@ref th-differential-scheme). They are typeset in upright
bold, with ``\mathbb 1`` the identity matrix of the corresponding size, and each
page states their size and their entries.

An underline is therefore a vector, bold is order 2 and blackboard bold is
order 4, which is what makes an expression such as

```math
\mathbb{H} = \tfrac{3}{4}\,
\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}
```

readable at a glance: an order-4 tensor ``\mathbb{H}`` is built from an order-2
tensor ``\boldsymbol{B}`` and a unit vector ``\underline{n}``. The identity of
order 2 is ``\boldsymbol{1}`` and that of order 4 is ``\mathbb{I}``.

The decorations follow four rules, so that a quantity is written one way on
every page and in every docstring:

- the reference medium carries the index ``0`` and a phase its index as a
  subscript, ``\mathbb{C}_0``, ``\mathbb{C}_i``, ``f_i``, while an effective
  property carries the upright superscript ``\mathrm{hom}``, as in
  ``\mathbb{C}^{\mathrm{hom}}``;
- every label made of letters is upright and written with `\mathrm`, as in
  ``\mathbb{A}_i^{\mathrm{dil}}``, ``\mathbb{C}^{\mathrm{iso}}`` or the
  differential ``\mathrm{d}``;
- ``\mathbb{S}`` is a compliance, and the Eshelby tensor is written
  ``\mathbb{S}^{\mathrm{E}}`` (``\boldsymbol{S}^{\mathrm{E}}`` in conduction);
- a formula in a docstring obeys the same rules and is written in LaTeX; Unicode
  symbols such as `C₀` are kept for the names of arguments and for code.

## Operators

| Operator | Meaning |
| :------- | :------ |
| ``\underline{u}\cdot\underline{v}``, ``\boldsymbol{A}\cdot\underline{u}`` | simple contraction (one index) |
| ``\boldsymbol{A}:\boldsymbol{B}``, ``\mathbb{C}:\boldsymbol{\varepsilon}`` | double contraction (two indices) |
| ``\underline{u}\otimes\underline{v}`` | tensor product |
| ``\underline{u}\stackrel{s}{\otimes}\boldsymbol{A}\stackrel{s}{\otimes}\underline{u}`` | symmetrized tensor product (order 4, both symmetries) |
| ``\boldsymbol{A}\stackrel{s}{\boxtimes}\boldsymbol{B}`` | symmetrized box product |
| ``\boldsymbol{A}^{\!T}`` | transpose |
| ``[\![\,\underline{u}\,]\!]`` | jump across an interface, ``\underline{u}^+-\underline{u}^-`` |
| ``\langle\,\cdot\,\rangle_\Omega`` | volume average over ``\Omega`` |
| ``\mathrm{d}S_\xi`` | surface element in the ``\underline{\xi}`` parametrization |
| ``\mathbb{A}\circ\mathbb{B}``, ``\mathbb{L}^{-\circ}`` | Volterra product and Volterra inverse of two kernels |

```@eval
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "docs", "nomenclature.jl"))
nomenclature_markdown()
```
