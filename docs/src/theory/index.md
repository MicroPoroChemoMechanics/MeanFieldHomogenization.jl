# [Theory — reading path](@id th-index)

`MeanFieldHomogenization` computes effective properties of heterogeneous materials by
mean-field homogenization. This section states the theory it implements, in the
order in which it is built. Every page is self-contained on notation
([Notation and conventions](notation.md)) and every formula is either cited or
derived on the page.

The section is in four groups, and they are meant to be read in that order:
**Foundations**, then **Schemes and specializations**, then the **N-body
models**, then the **Appendices** — pages that support the rest but are written
in its language rather than the other way round.

## Foundations: the chain, in one paragraph

An **inclusion** embedded in an infinite reference medium responds to a remote
load in a way entirely captured by one object, the **Hill polarization tensor**
``\mathbb{P}`` — this is [Eshelby's result](eshelby_problem.md), and
``\mathbb{P}`` depends only on the inclusion *shape* and the reference *moduli*
([Hill polarization tensors](hill_tensors.md)). From ``\mathbb{P}`` follows the
**localization tensor**, which says how much of the remote load each phase
actually sees, and hence each phase's **contribution** to the effective
stiffness ([Localization](localization.md)). Assembling those contributions
under an assumption about how phases interact gives a **homogenization scheme**
— dilute, Mori–Tanaka, self-consistent, differential, and the bounds
([Homogenization schemes](homogenization.md)).

## Schemes and specializations

Each of these keeps the chain above and changes one of its ingredients:

| Page | Specialization |
| :--- | :------------- |
| [Differential scheme](differential_scheme.md) | incorporation as an ODE in a fictitious time, and what a crack — which has no volume to replace — does to it |
| [Crack opening displacement](cod_tensors.md) | the flat-inclusion limit: a crack has no volume, so it is described by ``\boldsymbol{B}`` and ``\mathbb{H}`` instead of a volume fraction |
| [Thermal cracks](thermal_cracks.md) | the same limit for scalar transport |
| [Layered sphere](layered_sphere.md) | a *composite* inclusion: no Hill tensor exists, the response is assembled by a radial recurrence |
| [Layered spheroid](layered_spheroid.md) | the same for confocal spheroids, where imperfect interfaces couple harmonic degrees |
| [Laminate](laminate.md) | a periodic stack: no inclusion at all, the interface algebra replaces the Hill tensor |
| [Ageing linear viscoelasticity](viscoelasticity.md) | moduli become Volterra operators; the algebra of the chain is unchanged |

## N-body models

These are the only pages that drop the one-site assumption: instead of averaging
the interaction between inclusions, they resolve it pair by pair, and therefore
need to know *where* the inclusions are.

| Page | What it adds |
| :--- | :----------- |
| [Interaction tensors](interaction_tensors.md) | the two-inclusion tensor ``\mathbb{T}^{ab}``, its Green operator, its closed forms — and the sign convention both models follow |
| [The cluster model](cluster_model.md) | Molinari & El Mouden: the mean strain of every inclusion, resolved inside a cluster |
| [The equivalent inclusion method](eim.md) | Brisard, Dormieux & Sab: the same physics as a variational Galerkin problem, with rigorous bounds |

## Appendices

| Page | Role |
| :--- | :--- |
| [The finite Eshelby cell](corrected_cell.md) | a numerical device supporting the finite-element inclusions: the inclusion is solved on a *finite* cell and the truncation bias removed by its own dipole far field. Written in the language of [localization](localization.md) and [crack opening displacement](cod_tensors.md), so it is read after both |
| [Elliptic integrals](elliptic_integrals.md) | the special functions the closed forms need |

## Where the two physics meet

The Eshelby framework applies verbatim to elasticity (order-4 tensors) and to
scalar transport — heat conduction, diffusion, electric conduction, Darcy flow
(order-2 tensors). The documentation treats them in parallel rather than in
separate sections, because with the convention
``\boldsymbol{\sigma} \equiv -\underline{q}`` fixed in
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q) the
algebra is not merely analogous but *identical, symbol for symbol*:

| | Elasticity | Transport |
| :--- | :--- | :--- |
| property | stiffness ``\mathbb{C}`` | conductivity ``\boldsymbol{K}`` |
| Hill tensor | ``\mathbb{P}(\boldsymbol{A},\mathbb{C})`` (order 4) | ``\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K})`` (order 2) |
| Eshelby tensor | ``\mathbb{S} = \mathbb{P}:\mathbb{C}`` | ``\boldsymbol{s} = \boldsymbol{P}\cdot\boldsymbol{K}`` |
| crack descriptor | COD tensor ``\boldsymbol{B}``, compliance ``\mathbb{H}`` | COD scalar ``b``, resistivity ``\boldsymbol{R}`` |

One asymmetry is worth knowing in advance: for an **arbitrarily anisotropic**
matrix the order-2 Hill tensor has a *closed form* (via a square-root
transformation), whereas the order-4 one requires numerical cubature. This is
why the conductivity paths in `MeanFieldHomogenization` are analytical far more often than
the elastic ones.

## Relation to the Echoes manual

`MeanFieldHomogenization` is a pure-Julia reimplementation of the Eshelby/Hill machinery of
the [Echoes manual](https://jfbarthelemy.github.io/echoes/), and the
Hill-tensor pages follow its appendix closely — same expressions, same
conventions, same bibliography. Two families of difference are flagged
explicitly wherever they occur:

- **extensions**: infinite cylinders and 2-D plane strain as first-class
  inclusion types, an analytical transversely isotropic path, automatic
  differentiation and symbolic number types throughout;
- **conventions that genuinely differ**: the crack opening displacement tensor
  ``\boldsymbol{B}`` and the compliance ``\mathbb{H}`` are the notable case.
  `MeanFieldHomogenization` computes ``\boldsymbol{B}`` first and derives ``\mathbb{H}``
  from it, where Echoes computes ``\mathbb{H}`` directly and never forms
  ``\boldsymbol{B}``. The competing normalizations are named and compared in
  [Crack opening displacement](cod_tensors.md).
