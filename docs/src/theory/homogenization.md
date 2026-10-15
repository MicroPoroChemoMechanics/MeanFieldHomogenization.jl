# [Homogenization schemes](@id th-homogenization)

!!! info "Before this page"
    [Localization and contribution tensors](@ref th-localization), where the
    dilute concentration tensor ``\mathbb A_i^{\mathrm{dil}}`` and the
    contribution tensors ``\mathbb N_i`` and ``\mathbb H_i`` are defined,
    [Crack opening displacement and compliance](@ref th-cod-tensors) for the
    compliance contribution of a crack and its density, and
    [Layered sphere](@ref th-layered-sphere) for a pattern that supplies a
    concentration tensor without having a Hill tensor.

The previous chapters solved the problem of a single heterogeneity in an
infinite medium: an ellipsoid, a crack, a layered pattern. A homogenization
scheme turns those solutions into an estimate of the effective property of a
representative volume element, by stating which auxiliary problem each phase is
assumed to solve. This page sets the framework that all the schemes share. The
pages that follow treat them one at a time, each with its figure, its assumption
and its equations.

## 1. The representative volume element

The representative volume element (RVE) holds

- a *matrix* phase of property tensor ``\mathbb C_0`` (or ``\boldsymbol{K}_0``
  for the order-2 conductivity problem), when one phase plays that role;
- one or more *inclusion* phases of property tensors ``\mathbb C_i``,
  geometries ``\mathcal G_i``, and amounts ``f_i`` (volume fraction) or
  ``\varepsilon_i`` (crack density).

The drawings of this chapter are redrawn after the Echoes book [echoes](@cite).

![An RVE loaded by ``\underline u = \boldsymbol E\cdot\underline x`` is replaced by a sum of single-inclusion problems in the infinite matrix ``\mathbb C_0``, each loaded by ``\underline u = \boldsymbol E^0\cdot\underline x``](../assets/schemes/rve_decomposition.svg)

That picture is the mean-field approximation, and every scheme of this chapter
is one answer to the question it leaves open: what is ``\boldsymbol E^0``, and in
which medium is each inclusion embedded? Taking ``\boldsymbol E^0 = \boldsymbol E``
in the matrix gives the dilute estimate; requiring the matrix average to come out
right gives Mori–Tanaka; embedding each phase in the unknown effective medium
gives the self-consistent scheme.

## 2. What every scheme estimates

Under uniform boundary conditions the macroscopic strain and stress are the
volume averages of their local counterparts, ``\boldsymbol E = \langle
\boldsymbol\varepsilon\rangle`` and ``\boldsymbol\Sigma = \langle
\boldsymbol\sigma\rangle`` [hill1963](@cite). Linearity defines a strain
concentration tensor per phase, ``\langle\boldsymbol\varepsilon\rangle_r =
\mathbb A_r:\boldsymbol E``, and the two averages give

```math
\sum_r f_r\,\mathbb A_r = \mathbb I ,
\qquad
\mathbb C^{\mathrm{hom}} = \sum_r f_r\,\mathbb C_r:\mathbb A_r
  = \mathbb C_0 + \sum_{i} f_i\,(\mathbb C_i - \mathbb C_0):\mathbb A_i ,
```

the sum over ``r`` running over every phase and the one over ``i`` over the
phases other than the matrix, whose term vanishes in the second form. The
effective tensor is thus known as soon as the concentration tensors are, and a
scheme is precisely an estimate of them, built from the solution of an Eshelby
problem posed in some reference medium.

The schemes below are written with three objects of
[Localization](@ref th-localization), computed in the matrix ``\mathbb C_0``
unless stated otherwise. For an ellipsoidal phase ``i`` of Hill tensor
``\mathbb P_i``,

```math
\mathbb N_i = (\mathbb C_i - \mathbb C_0):\mathbb A_i^{\mathrm{dil}}
  = \big[(\mathbb C_i - \mathbb C_0)^{-1} + \mathbb P_i\big]^{-1} ,
\qquad
\mathbb A_i^{\mathrm{dil}} = \mathbb I - \mathbb P_i:\mathbb N_i ,
\qquad
\mathbb H_i = -\,\mathbb S_0:\mathbb N_i:\mathbb S_0 .
```

The middle identity follows from
``\mathbb I - \mathbb A_i^{\mathrm{dil}}
= \big((\mathbb A_i^{\mathrm{dil}})^{-1} - \mathbb I\big):\mathbb A_i^{\mathrm{dil}}
= \mathbb P_i:(\mathbb C_i - \mathbb C_0):\mathbb A_i^{\mathrm{dil}}``,
and the last one from the definition
``\mathbb H_i = (\mathbb S_i - \mathbb S_0):\mathbb C_i:\mathbb A_i^{\mathrm{dil}}:\mathbb S_0``
once ``(\mathbb S_i - \mathbb S_0):\mathbb C_i = -\,\mathbb S_0:(\mathbb C_i - \mathbb C_0)``
is noted. The second form of ``\mathbb N_i`` needs ``\mathbb C_i - \mathbb C_0``
to be invertible, the first one does not, and that is the one the package
evaluates. The total dilute contributions are written

```math
\mathbb N_\Sigma = \sum_i f_i\,\mathbb N_i ,
\qquad
\mathbb H_\Sigma = \sum_i f_i\,\mathbb H_i = -\,\mathbb S_0:\mathbb N_\Sigma:\mathbb S_0 .
```

### Cracks

A crack family ``c`` is given by its density ``\varepsilon_c`` rather than by a
volume fraction, which vanishes in the flat limit while the opening, and hence
the strain the family carries, stays finite. The two products a scheme needs keep
a finite limit:

```math
f_c\,\mathbb A_c \;\longrightarrow\; \kappa_c\,\varepsilon_c\,\mathbb H_c:\mathbb C_0 ,
\qquad
f_c\,(\mathbb C_c - \mathbb C_0):\mathbb A_c \;\longrightarrow\;
  -\,\kappa_c\,\varepsilon_c\,\mathbb C_0:\mathbb H_c:\mathbb C_0 ,
```

with ``\kappa_c = 4\pi/3`` for an elliptic family and ``\pi`` for a ribbon
family [budiansky1976](@cite), and ``\mathbb H_c`` the compliance contribution
derived in [Crack opening displacement](@ref th-cod-tensors). The first limit is
the strain carried by the openings per unit remote strain. The second follows
because a traction-free crack carries no stress, so that
``f_c\,\mathbb C_c:\mathbb A_c \to 0``. A crack family therefore enters every
scheme through these two terms, added to ``\mathbb N_\Sigma`` as
``\kappa_c\,\varepsilon_c\,\mathbb N_c`` with
``\mathbb N_c = -\,\mathbb C_0:\mathbb H_c:\mathbb C_0``, and each page says where.

## [3. The schemes at a glance](@id th-schemes-glance)

Each scheme answers the question of §1 differently. The table names the
reference medium in which each inclusion is embedded, the field it is loaded by,
and the page where the scheme is derived.

| Scheme | Reference medium | Field applied to each inclusion | Page |
| :-- | :-- | :-- | :-- |
| [`Voigt`](@ref) | none | the macroscopic strain, in every phase | [Bounds](@ref th-scheme-bounds) |
| [`Reuss`](@ref) | none | the macroscopic stress, in every phase | [Bounds](@ref th-scheme-bounds) |
| [`Dilute`](@ref) | matrix | the macroscopic strain, at infinity | [Dilute estimates](@ref th-scheme-dilute) |
| [`DiluteDual`](@ref) | matrix | the macroscopic stress, at infinity | [Dilute estimates](@ref th-scheme-dilute) |
| [`MoriTanaka`](@ref) | matrix | the mean strain of the matrix | [Mori–Tanaka](@ref th-scheme-mori-tanaka) |
| [`Maxwell`](@ref) | matrix | a remote strain applied to a domain holding every inclusion | [Maxwell](@ref th-scheme-maxwell) |
| [`PonteCastanedaWillis`](@ref) | matrix | the field inside the distribution ellipsoid of each pair of phases | [Ponte Castañeda–Willis](@ref th-scheme-pcw) |
| [`SelfConsistent`](@ref) | effective medium, every phase | a remote strain fixed by the average | [Self-consistent schemes](@ref th-scheme-self-consistent) |
| [`AsymmetricSelfConsistent`](@ref) | effective medium, inclusions only | a remote strain fixed by the average | [Self-consistent schemes](@ref th-scheme-self-consistent) |
| [`DifferentialScheme`](@ref) | current effective medium | dilute increments | [Differential scheme](@ref th-differential-scheme) |
| [`Laminated`](@ref) | none, exact | piecewise-uniform fields | [Laminate](@ref th-laminate) |
| [`ClusterModel`](@ref) | matrix | the remote strain and the field of every neighbor | [Cluster model](@ref th-cluster) |
| [`EquivalentInclusion`](@ref) | matrix | a Galerkin polarization | [Equivalent inclusion method](@ref th-eim) |

The pages of the chapter follow the order of the table. The bounds use no shape
at all. The next four pages embed the inclusions in the matrix, each with a
richer description of what surrounds them. The self-consistent and differential
schemes then replace the matrix by the effective medium, the laminate is solved
exactly, and the last two schemes resolve the interaction inclusion by inclusion.

## [4. Schemes that need positions](@id th-nbody)

Every scheme of the table but the last two sees one inclusion in a reference
medium and accounts for the others only through that reference: the interaction
is treated in an average sense. The [`ClusterModel`](@ref) and the
[`EquivalentInclusion`](@ref) method drop that one-site assumption and resolve
the interaction inclusion by inclusion, which needs strictly more information
than an `RVE` carries — the positions. They act on a [`ParticleAssembly`](@ref)
instead, and share one ingredient, the
[two-inclusion interaction tensor](@ref th-interaction) ``\mathbb{T}^{ab}``.
Both degenerate exactly onto Mori–Tanaka when the interaction is switched off,
which is the sharpest available check that their assembly is right.

## 5. Number-type compatibility

The schemes involve nothing but tensor algebra and, for the iterative and
differential ones, a solver, so that the scalar type is left open. Every scheme
is mandated to support:

- `Float64` — default;
- `ForwardDiff.Dual` — sensitivity analysis through fractions, moduli,
  geometric parameters;
- `Complex{Float64}` — frequency-domain viscoelasticity (parity with
  the C++ ECHOES library, templated on `T = double | complex<double>`);
- `SymPy.Sym`, `Symbolics.Num`, `BigFloat` — best-effort, with explicit
  documentation of any limitation (the iterative SC solvers are not
  symbolic-friendly because the linear-system Jacobian must be numeric).

## Where to go next

The schemes are now taken one at a time, starting with the two that need
neither a shape nor a reference medium:
[Voigt and Reuss bounds](@ref th-scheme-bounds). The tutorials
[A first homogenization](@ref tut-first-estimate) and
[Bounds and classical schemes](@ref tut-bounds-and-schemes) apply the schemes of
this chapter to a first RVE, and the manual page
[Homogenization schemes](@ref man-schemes) gives their syntax.
