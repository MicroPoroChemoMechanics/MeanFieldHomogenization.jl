# [Maxwell and Ponte Castañeda–Willis](@id th-scheme-maxwell-pcw)

!!! info "Before this page"
    [Mori–Tanaka](@ref th-scheme-mori-tanaka), whose second form is compared
    with the estimates below, the
    [sign of the Green operator](@ref th-notation-green-sign) fixed in the
    Conventions, and [Localization](@ref th-localization) for the contribution
    tensors.

Both schemes keep the matrix as reference medium and add a second shape, which
describes how the inclusions are distributed rather than what they look like.
They come from two different arguments: the far field of a cluster of
inhomogeneities for Maxwell, the two-point statistics of the inclusions for
Ponte Castañeda and Willis (PCW). This page derives each from the Green operator
of the matrix, then shows that they coincide when the distribution is described
by a single shape, and that Mori–Tanaka is one particular choice of that shape.

The notation is the intrinsic one of [barthelemyIJES2019](@citet), who
transpose the Maxwell scheme to aging viscoelasticity. It is written here in
elasticity, with the conventions of this documentation: ``\mathbb S`` is a
compliance, and the Green operator ``\mathbb G^0`` of the matrix maps a
polarization onto minus the strain it induces.

## 1. The field of one inhomogeneity, seen from afar

An inhomogeneity occupies a bounded domain ``\mathcal E`` of any shape, with a
stiffness ``\mathbb C_{\mathcal E}`` that need not be uniform, and is embedded in
the infinite matrix ``\mathbb C_0`` loaded at infinity by the strain
``\boldsymbol E``. Its polarization
``\boldsymbol\tau = (\mathbb C_{\mathcal E} - \mathbb C_0):\boldsymbol\varepsilon``
vanishes outside ``\mathcal E``, and the Lippmann–Schwinger equation gives the
strain everywhere:

```math
\boldsymbol\varepsilon(\underline x) = \boldsymbol E
  - \int_{\mathcal E} \mathbb G^0(\underline x - \underline y):\boldsymbol\tau(\underline y)\,
    \mathrm dV_{\underline y} .
```

At a point ``\underline x`` far from ``\mathcal E`` compared with its size, the
kernel varies little across the domain, and
``\mathbb G^0(\underline x - \underline y)`` may be replaced by
``\mathbb G^0(\underline x)``, the origin being taken inside ``\mathcal E``. The
leading term of the disturbance is then

```math
\boldsymbol\varepsilon(\underline x) - \boldsymbol E \simeq
  -\,|\mathcal E|\,\mathbb G^0(\underline x):\langle\boldsymbol\tau\rangle_{\mathcal E}
  = -\,|\mathcal E|\,\mathbb G^0(\underline x):\mathbb N_{\mathcal E}:\boldsymbol E ,
```

because the mean polarization of the inhomogeneity is, by definition of its
stiffness contribution, ``\langle\boldsymbol\tau\rangle_{\mathcal E}
= \langle\boldsymbol\sigma\rangle_{\mathcal E} - \mathbb C_0:\langle\boldsymbol\varepsilon\rangle_{\mathcal E}
= \mathbb N_{\mathcal E}:\boldsymbol E``. Since the matrix carries
``\boldsymbol\sigma - \boldsymbol\Sigma = \mathbb C_0:(\boldsymbol\varepsilon - \boldsymbol E)``
and ``\mathbb N_{\mathcal E} = -\,\mathbb C_0:\mathbb H_{\mathcal E}:\mathbb C_0``,
the same term reads, in stress,

```math
\boldsymbol\sigma(\underline x) - \boldsymbol\Sigma \simeq
  |\mathcal E|\,\mathbb C_0:\mathbb G^0(\underline x):\mathbb C_0:\mathbb H_{\mathcal E}:\boldsymbol\Sigma .
```

At leading order, the far field thus depends on the inhomogeneity through a single product,
``|\mathcal E|\,\mathbb N_{\mathcal E}`` or equivalently
``|\mathcal E|\,\mathbb H_{\mathcal E}``: its shape, its moduli and its inner
structure enter nowhere else [sevostianovIJSS2011](@cite). For an ellipsoid of
uniform stiffness and Hill tensor ``\mathbb P_{\mathcal E}``,
``\mathbb N_{\mathcal E} = \big[(\mathbb C_{\mathcal E} - \mathbb C_0)^{-1} + \mathbb P_{\mathcal E}\big]^{-1}``.

## 2. Maxwell: a cluster and its effective twin

![A cluster of inclusions is equivalent to one inclusion Ω, both loaded remotely.](../assets/schemes/rve_maxwell.svg)

Cut out of the composite an ellipsoidal domain ``\Omega`` holding many
inhomogeneities ``\mathcal E_a``, centered at ``\underline x_a``, and embed it in
the infinite matrix under the remote strain ``\boldsymbol E``. Far from
``\Omega`` its disturbance can be described in two ways.

1. As the sum of the far fields of the inhomogeneities, each computed as if it
   were alone in the matrix. For ``|\underline x|`` large compared with the size
   of ``\Omega``, ``\mathbb G^0(\underline x - \underline x_a) \simeq \mathbb G^0(\underline x)``
   for every ``a``, and
   ``\boldsymbol\varepsilon - \boldsymbol E \simeq
   -\,\mathbb G^0(\underline x):\big(\sum_a |\mathcal E_a|\,\mathbb N_a\big):\boldsymbol E``.
2. As the far field of ``\Omega`` filled with the effective medium, an
   inhomogeneity in its own right:
   ``\boldsymbol\varepsilon - \boldsymbol E \simeq
   -\,|\Omega|\,\mathbb G^0(\underline x):\mathbb N_\Omega(\mathbb C^{\mathrm{hom}}):\boldsymbol E``.

The scheme equates the two. With ``f_i`` the volume fraction of phase ``i`` in
``\Omega``, it reads

```math
\mathbb N_\Omega(\mathbb C^{\mathrm{MX}}) = \sum_i f_i\,\mathbb N_i = \mathbb N_\Sigma ,
```

and, ``\Omega`` being an ellipsoid of Hill tensor ``\mathbb P_\Omega`` filled with
a uniform medium, the contribution on the left is
``\big[(\mathbb C^{\mathrm{MX}} - \mathbb C_0)^{-1} + \mathbb P_\Omega\big]^{-1}``.
Solving for the effective stiffness [sevostianovIJES2013](@cite):

```math
\mathbb C^{\mathrm{MX}} = \mathbb C_0 + \big(\mathbb N_\Sigma^{-1} - \mathbb P_\Omega\big)^{-1}
  = \mathbb C_0 + \mathbb N_\Sigma:\big(\mathbb I - \mathbb P_\Omega:\mathbb N_\Sigma\big)^{-1} .
```

The second form needs no inverse of ``\mathbb N_\Sigma``, which is singular for a
single family of parallel cracks, and it is the one the package evaluates. A crack
family enters through ``\mathbb N_\Sigma`` only, as
``\kappa_c\,\varepsilon_c\,\mathbb N_c``. Equating the far fields in stress
instead gives
``\mathbb S^{\mathrm{MX}} = \mathbb S_0 + \big(\mathbb H_\Sigma^{-1} - \mathbb Q_\Omega\big)^{-1}``,
with ``\mathbb Q_\Omega = \mathbb C_0 - \mathbb C_0:\mathbb P_\Omega:\mathbb C_0``:
the same estimate, since ``\mathbb H = -\,\mathbb S_0:\mathbb N:\mathbb S_0``
holds for ``\Omega`` as for each inhomogeneity.

The approximation lies in the first description, where every inhomogeneity is
loaded by the remote strain as if the others were absent, and in matching the
leading term of the far field only. The estimate is nevertheless not the dilute
one: the response of ``\Omega`` as a whole, through ``\mathbb P_\Omega``, accounts
for the interactions collectively. Expanded in the fractions,

```math
\mathbb C^{\mathrm{MX}} = \mathbb C_0 + \mathbb N_\Sigma
  + \mathbb N_\Sigma:\mathbb P_\Omega:\mathbb N_\Sigma + O(f^3) :
```

the inclusion shapes act at first order, the shape of ``\Omega`` at second order
only.

Two consequences follow from the derivation. First, each inhomogeneity enters
through its contribution tensor alone, so that any morphology that supplies one
(a crack, a layered sphere, a finite-element inclusion) may be mixed with
ellipsoids [sevostianovIJES2013](@cite); only ``\Omega`` has to be an ellipsoid.
Second, ``\Omega`` describes the distribution, not the inclusions. A sphere
suits an isotropic distribution of orientations and positions, the common shape
suits identical aligned inclusions, and the general anisotropic case is a
question of its own [sevostianovMOM2014](@cite). The package therefore takes
``\Omega`` from the RVE, where it must be declared.

## 3. Ponte Castañeda–Willis: the distribution ellipsoid

![Each inclusion inside its own spherical safety domain, the domains paving the space.](../assets/schemes/rve_pcw.svg)

[ponte1995](@citet) consider ellipsoidal inclusions of several phases in a
matrix, distributed at random with two-point statistics of ellipsoidal symmetry.
Given an inclusion of phase ``i``, the probability density of finding the center
of an inclusion of phase ``j`` at a separation ``\underline r`` depends on
``\underline r`` only through an ellipsoidal norm. It vanishes inside an
ellipsoid ``\mathcal D^{ij}`` centered on the first inclusion, large enough for
two inclusions not to overlap, and the correlation it carries beyond does not
contribute to the averages in the absence of long-range order. The symmetry of
the statistics requires ``\mathcal D^{ij} = \mathcal D^{ji}``.

They obtain the estimate as the stationary point of the Hashin–Shtrikman
variational principle, with the matrix as comparison medium and a polarization
uniform in each phase. The equations at that point admit a direct reading in
terms of the Green operator, which is the one followed here.

Seen from an inclusion of phase ``i``, the other inclusions are absent from
``\mathcal D^{ij}`` and, beyond it, contribute on average their mean
polarization. Take first a single distribution shape ``\mathcal D``, of Hill
tensor ``\mathbb P_d`` in the matrix. The medium around the inclusion is then
the matrix carrying the mean polarization
``\bar{\boldsymbol\tau} = \sum_j f_j\,\boldsymbol\tau_j`` everywhere except in the
cavity ``\mathcal D``. A uniform polarization filling the whole space leaves the
strain uniform, equal to the macroscopic strain ``\boldsymbol E``. Removing it
from ``\mathcal D`` amounts to adding ``-\bar{\boldsymbol\tau}`` there, which by
Eshelby's result adds ``-\,\mathbb P_d:(-\bar{\boldsymbol\tau})`` uniformly inside
the ellipsoid. The inclusion therefore sits in the uniform field

```math
\boldsymbol E_i = \boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau} ,
```

and, being itself an ellipsoid in the matrix, responds with the dilute solution,
``\boldsymbol\tau_i = \mathbb N_i:(\boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau})``.
With one shape per pair of phases the cavity seen by phase ``i`` differs for
each phase ``j`` around it, and the same reading gives the system of
[ponte1995](@citet):

```math
\big[(\mathbb C_i - \mathbb C_0)^{-1} + \mathbb P_i\big]:\boldsymbol\tau_i
  - \sum_j f_j\,\mathbb P_d^{ij}:\boldsymbol\tau_j = \boldsymbol E ,
```

the bracket being ``\mathbb N_i^{-1}``. With a single shape, multiplying
``\boldsymbol\tau_i = \mathbb N_i:(\boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau})``
by ``f_i`` and summing closes the system on the mean polarization,

```math
\bar{\boldsymbol\tau} = \mathbb N_\Sigma:(\boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau})
\quad\Longrightarrow\quad
\bar{\boldsymbol\tau} = (\mathbb I - \mathbb N_\Sigma:\mathbb P_d)^{-1}:\mathbb N_\Sigma:\boldsymbol E ,
```

and, the polarization vanishing in the matrix,
``\boldsymbol\Sigma = \mathbb C_0:\boldsymbol E + \bar{\boldsymbol\tau}``:

```math
\mathbb C^{\mathrm{PCW}} = \mathbb C_0
  + (\mathbb I - \mathbb N_\Sigma:\mathbb P_d)^{-1}:\mathbb N_\Sigma .
```

A crack family enters through ``\mathbb N_\Sigma`` as in the Maxwell scheme. The
requirement that ``\mathcal D`` prevent overlaps restricts the fractions that a
given pair of inclusion and distribution shapes can describe
[ponte1995](@cite).

## 4. One distribution shape: PCW is Maxwell

When every pair of phases shares the distribution shape and ``\Omega`` has that
shape, ``\mathbb P_\Omega = \mathbb P_d``, and the two estimates coincide. Two
arguments show it.

The first is algebraic. Since
``(\mathbb I - \mathbb N_\Sigma:\mathbb P_d):\mathbb N_\Sigma
= \mathbb N_\Sigma - \mathbb N_\Sigma:\mathbb P_d:\mathbb N_\Sigma
= \mathbb N_\Sigma:(\mathbb I - \mathbb P_d:\mathbb N_\Sigma)``, multiplying by
``(\mathbb I - \mathbb N_\Sigma:\mathbb P_d)^{-1}`` on the left and by
``(\mathbb I - \mathbb P_d:\mathbb N_\Sigma)^{-1}`` on the right gives

```math
(\mathbb I - \mathbb N_\Sigma:\mathbb P_d)^{-1}:\mathbb N_\Sigma
  = \mathbb N_\Sigma:(\mathbb I - \mathbb P_d:\mathbb N_\Sigma)^{-1} ,
```

so that ``\mathbb C^{\mathrm{PCW}}`` is the second form of ``\mathbb C^{\mathrm{MX}}``,
without any inverse of ``\mathbb N_\Sigma``.

The second shows that the two constructions are one. In the Maxwell
construction, write ``\boldsymbol E^0`` the remote strain applied to the cluster.
The domain ``\Omega``, filled with ``\mathbb C^{\mathrm{MX}}``, undergoes a uniform
strain ``\boldsymbol\varepsilon_\Omega`` and carries the polarization
``\boldsymbol\tau_\Omega = (\mathbb C^{\mathrm{MX}} - \mathbb C_0):\boldsymbol\varepsilon_\Omega``,
and the Eshelby solution relates them by

```math
\boldsymbol E^0 = \big[\mathbb I + \mathbb P_\Omega:(\mathbb C^{\mathrm{MX}} - \mathbb C_0)\big]:\boldsymbol\varepsilon_\Omega
  = \boldsymbol\varepsilon_\Omega + \mathbb P_\Omega:\boldsymbol\tau_\Omega .
```

Read ``\boldsymbol\varepsilon_\Omega`` as the macroscopic strain and
``\boldsymbol\tau_\Omega`` as the mean polarization of the medium inside
``\Omega``, whose mean stress is indeed
``\mathbb C_0:\boldsymbol\varepsilon_\Omega + \boldsymbol\tau_\Omega``. The
remote strain is then exactly the field ``\boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau}``
that PCW apply to every inclusion. The Maxwell condition, applied to
``\boldsymbol E^0``, has ``\mathbb N_\Omega:\boldsymbol E^0 = \boldsymbol\tau_\Omega``
on its left, by definition of ``\mathbb N_\Omega``, and therefore reads
``\bar{\boldsymbol\tau} = \mathbb N_\Sigma:(\boldsymbol E + \mathbb P_d:\bar{\boldsymbol\tau})``:
the closure of §3. The inclusions that Maxwell loads by the remote strain of the
cluster are those that PCW place in the field of their cavity, and both balance
the same polarization.

The package evaluates the two schemes with one kernel, and
[`Maxwell`](@ref) and [`PonteCastanedaWillis`](@ref) return the same tensor. They
would differ with one distribution shape per pair of phases, which only the PCW
derivation describes and which is not implemented.

## 5. Mori–Tanaka as a distribution

The second form of the [Mori–Tanaka estimate](@ref th-scheme-mori-tanaka) and the
Maxwell estimate differ by a single term,

```math
\mathbb C^{\mathrm{MT}} = \mathbb C_0 + \mathbb N_\Sigma:\Big(\mathbb I - \sum_j f_j\,\mathbb P_j:\mathbb N_j\Big)^{-1} ,
\qquad
\mathbb C^{\mathrm{MX}} = \mathbb C_0 + \mathbb N_\Sigma:\Big(\mathbb I - \mathbb P_\Omega:\sum_j f_j\,\mathbb N_j\Big)^{-1} ,
```

and they coincide when every inclusion phase has the Hill tensor of the
distribution shape, whatever the moduli. A spherical distribution therefore
reproduces Mori–Tanaka for spherical inclusions only, and departs from it as soon
as the inclusions are spheroids.

In the system of §3, Mori–Tanaka is the choice ``\mathbb P_d^{ij} = \mathbb P_j``.
The field applied to phase ``i`` is then
``\boldsymbol E + \sum_j f_j\,\mathbb P_j:\boldsymbol\tau_j``, the same for every
phase, and it is the mean strain of the matrix, since
``\boldsymbol E = f_0\,\langle\boldsymbol\varepsilon\rangle_0 + \sum_j f_j\,\mathbb A_j^{\mathrm{dil}}:\langle\boldsymbol\varepsilon\rangle_0
= \langle\boldsymbol\varepsilon\rangle_0 - \sum_j f_j\,\mathbb P_j:\boldsymbol\tau_j``.
That choice is not symmetric in ``(i, j)`` as soon as two phases differ in shape:
it violates the symmetry of the pair statistics, which [ponte1995](@citet)
identify as the origin of the asymmetry of Mori–Tanaka with several shapes.

## 6. In practice

The distribution shape is a property of the microstructure, declared on the RVE
and without default; the manual explains why
([Distribution shape](@ref man-distribution-shape)). Both schemes are a closed
form with no iteration, so that they accept dual numbers and symbolic moduli
like the dilute estimates, and conduction follows with
``(\boldsymbol K, \boldsymbol P, \boldsymbol N)`` in place of
``(\mathbb C, \mathbb P, \mathbb N)``. In aging viscoelasticity the Maxwell
estimate keeps its second form, with Volterra products and the Hill kernel of
the distribution shape [barthelemyIJES2019](@cite); see
[Aging linear Volterra behavior](@ref th-viscoelasticity).

## Where to go next

Every scheme so far embeds the inclusions in the matrix. The next page,
[Self-consistent schemes](@ref th-scheme-self-consistent), embeds each phase in
the effective medium itself, which suits composites where no phase plays the
role of a matrix.
