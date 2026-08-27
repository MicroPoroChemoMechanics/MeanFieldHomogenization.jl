# [The equivalent inclusion method](@id th-eim)

[brisard2014](@cite). A Galerkin discretization of the
weak form of the Lippmann-Schwinger equation, with the polarization taken piecewise
polynomial over the inclusions. Unlike every other scheme of the package it also
delivers a **rigorous bound** on the apparent stiffness.

Where [moschovidis1975](@cite) discretized the *strong* form by
Taylor expansion and collocation, the variational form discretizes the *weak* one — and
inherits the extremum property of the Hashin-Shtrikman principle.

Notation as on [the conventions page](@ref th-notation); the interaction tensor
``\mathbb{T}^{ab}`` and the Green operator ``\mathbb{G}^0`` are defined on
[the interaction page](@ref th-interaction).

## Variational form

The modified Lippmann-Schwinger equation of
[brisard2013bc](@cite), posed on an **ellipsoidal** statistical
volume element ``\Omega`` embedded in an infinite medium of the matrix stiffness, has
the weak form: find ``\boldsymbol{\tau} \in \mathcal{V}`` such that
``a(\boldsymbol{\tau},\boldsymbol{\varpi}) = \ell(\boldsymbol{\varpi})`` for every
``\boldsymbol{\varpi} \in \mathcal{V}``, with

```math
a(\boldsymbol{\tau},\boldsymbol{\varpi})
  = \big\langle \boldsymbol{\varpi} : (\mathbb{C}-\mathbb{C}_0)^{-1}
      : \boldsymbol{\tau} \big\rangle
  + \big\langle \boldsymbol{\varpi} : \mathbb{G}^0 * \boldsymbol{\tau} \big\rangle
  - \langle\boldsymbol{\varpi}\rangle : \mathbb{P}_\Omega
      : \langle\boldsymbol{\tau}\rangle ,
```

```math
\ell(\boldsymbol{\varpi}) = \boldsymbol{E} : \langle\boldsymbol{\varpi}\rangle .
```

The last term of ``a`` comes from Eshelby's theorem applied to the SVE *itself* —
legitimate because ``\Omega`` is an ellipsoid — and ``\mathbb{P}_\Omega`` is its Hill
tensor. It is what implements the mixed boundary conditions: a uniform strain at
infinity plus a uniform surface traction chosen so that the loading parameter coincides
with the macroscopic strain.

## Galerkin discretization

Take the polarization piecewise polynomial of degree ``p`` over each inclusion,

```math
\boldsymbol{\tau}_p(\underline{x}) = \sum_{a=1}^{N} \sum_{k \in \mathcal{I}_p}
   \chi_a(\underline{x})\,
   (\underline{x}-\underline{x}_a)^k_\bullet\, \boldsymbol{\tau}^k_a ,
```

with ``\chi_a`` the indicator function of ``\Omega_a``. This is their discrete problem. At ``p = 0`` — one constant polarization per inclusion — dividing by
``|\Omega_a|`` gives

```math
\Big[(\mathbb{C}_a-\mathbb{C}_0)^{-1} + \mathbb{P}_a
     - f_a\, \mathbb{P}_\Omega\Big] : \boldsymbol{\tau}_a
 \;+\; \sum_{b \ne a}\Big[\mathbb{T}^{ab}(\underline{r}_{ab})
     - f_b\, \mathbb{P}_\Omega\Big] : \boldsymbol{\tau}_b
 \;=\; \boldsymbol{E} ,
```

```math
\mathbb{C}^{\mathrm{app}} : \boldsymbol{E}
  = \mathbb{C}_0 : \boldsymbol{E} + \sum_a f_a\, \boldsymbol{\tau}_a .
```

Local fields come out for free:
``\boldsymbol{\varepsilon}_a = (\mathbb{C}_a-\mathbb{C}_0)^{-1}:\boldsymbol{\tau}_a``
and
``\boldsymbol{\sigma}_a = \mathbb{C}_0 : \boldsymbol{\varepsilon}_a +
\boldsymbol{\tau}_a``.

!!! note "Transcribed verbatim"
    The package shares the sign convention of [brisard2023](@cite) (see the
    [convention note](@ref th-interaction)), so the system above is transcribed with
    nothing flipped: the self term ``|\Omega_a|^{-1}S^{00}_a`` is ``+\mathbb{P}_a``,
    which is ``\mathbb{T}^{aa}``, and every block carries a plus. It is the *cluster
    model* page that has a flip to declare.

## Relation to the cluster model

Brisard et al. observe in their §3.1 that at ``k = l = 0`` their influence
pseudotensors *coincide* with the interaction tensors of
[berveiller1987](@cite) and
[molinari1996](@cite). The two schemes of this package
accordingly share [`interaction_tensor`](@ref), and differ only in how the far field is
closed:

| | far-field term | boundary treatment |
| :-- | :-- | :-- |
| [`EquivalentInclusion`](@ref) | ``\mathbb{P}_\Omega``, Hill tensor of the SVE | [`MixedBC`](@ref) — no periodization |
| [`ClusterModel`](@ref) | ``\mathbb{P}_0``, Hill tensor of the inclusion shape | [`PeriodicBox`](@ref) — cluster cutoff |

Under a `PeriodicBox` the first reduces to the second, and the two agree to machine
precision — the acceptance gate of the implementation.

Two further exact degeneracies:

* a single spherical particle concentric in a spherical SVE gives
  ``\mathbb{P}_\Omega = \mathbb{P}``, the self block becomes
  ``(\mathbb{C}_1-\mathbb{C}_0)^{-1} + (1-f)\,\mathbb{P}``, and the scheme **is**
  Mori-Tanaka — at every volume fraction, not only in the dilute limit;
* as ``f \to 0`` both collapse onto [`Dilute`](@ref).

## Bounds

Introducing
``\mathcal{H}(\boldsymbol{\varpi}) = \ell(\boldsymbol{\varpi}) -
\tfrac{1}{2}\,a(\boldsymbol{\varpi},\boldsymbol{\varpi})``, the exact solution
satisfies

```math
\mathcal{H}(\boldsymbol{\tau}) = \tfrac{1}{2}\,\boldsymbol{E} :
   \big(\mathbb{C}^{\mathrm{app}} - \mathbb{C}_0\big) : \boldsymbol{E} ,
```

and the discrete one the same identity with ``\mathbb{C}^{\mathrm{app},p}``. Since
``\mathcal{H}`` is minimum (resp. maximum) at ``\boldsymbol{\tau}`` when
``\mathbb{C}_a \le \mathbb{C}_0`` (resp. ``\ge``) for every ``a``:

```math
\mathbb{C}_a \le \mathbb{C}_0 \;\;\forall a
  \;\Longrightarrow\;
  \mathbb{C}^{\mathrm{app}} \le \mathbb{C}^{\mathrm{app},p} ,
```

an **upper** bound, and a **lower** one in the opposite case. Two consequences: the
estimate improves monotonically as ``p`` grows (the trial space only gets larger), and
mixed contrasts give no bound at all. [`eim_bound_type`](@ref) reports which case holds.

Taking the polarization constant *and equal* across all inclusions instead recovers the
classical [hashin1962](@cite) bounds; letting it vary from one
inclusion to the next is what sharpens them.

## Reference results

[brisard2014](@cite), Table 1 — plane strain, ``N = 160`` circular
pores of radius ``a`` in a circular SVE of radius ``R = 20a``, porosity ``\phi = 0.4``,
``\nu_0 = 0.3``, 1000 realizations:

| order ``p`` | bound on ``\mu^{\mathrm{app}}`` | dofs |
| :-- | :-- | :-- |
| 0 | ``0.310\,\mu_0`` | 480 |
| 1 | ``0.278\,\mu_0`` | 1440 |
| 2 | ``0.257\,\mu_0`` | 2880 |
| 3 | ``0.247\,\mu_0`` | 4800 |

with a finite-element reference of ``0.244\,\mu_0`` and a Hashin-Shtrikman upper bound
of ``0.349\,\mu_0`` — the order-zero estimate already improves on the latter by 11 %.
Their Table 2, in 3D with polydisperse spherical pores (``N = 20/40/140`` of radii
``\rho_1``, ``0.7\rho_1``, ``0.4\rho_1``, ``\phi = 0.45``, ``R = 4.56\,\rho_1``) gives
``0.381\,\mu_0`` at ``p = 0``. `scripts/93` reproduces the ``p = 0`` row of Table 1.

!!! note "Only order = 0 is implemented"
    Orders ``p \ge 1`` require the influence *pseudotensors* of the paper's Appendix C.
    These are not tensors — they need their own change-of-basis machinery — and were
    generated by the authors with a computer algebra system. Requesting a higher order
    raises an `ArgumentError` rather than silently returning the order-zero answer.

## Slender fibers in conduction

[martin2023](@cite) specialize the method to slender cylinders in
steady conduction, taking the polarization polynomial in the *axial* coordinate and
constant across the section. Their interaction coefficients reduce, after a multipole
step, to nested one-dimensional integrals along the two axes, and their self-influence
coefficients are precomputed by finite elements. The `:quadrature` back-end of
[`interaction_tensor`](@ref) evaluates the same integrals directly; the axial
polynomial enrichment is not implemented.

## API

See [API — Particle assemblies](@ref api-assemblies).
