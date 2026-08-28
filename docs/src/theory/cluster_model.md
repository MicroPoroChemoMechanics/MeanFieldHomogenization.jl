# [The cluster model](@id th-cluster)

[molinari1996](@cite). An N-body scheme: the mean strain of
every inclusion is solved for, accounting for the pairwise interaction with each
neighbor inside a cluster, on top of the interaction with the matrix.

Notation as on [the conventions page](@ref th-notation); the interaction tensor
``\mathbb{T}^{IJ}`` and the Green operator are defined on
[the interaction page](@ref th-interaction).

## From the integral equation to a linear system

Take the matrix as reference, ``\mathbb{C}_0 = \mathbb{C}_m``, and assume the strain
uniform in each inclusion. Averaging the Lippmann-Schwinger equation over inclusion
``I`` gives

```math
\boldsymbol{\varepsilon}^I = \boldsymbol{\varepsilon}^0
  - \sum_J \mathbb{T}^{IJ} : \delta\mathbb{C}^J : \boldsymbol{\varepsilon}^J ,
\qquad \delta\mathbb{C}^J = \mathbb{C}^J - \mathbb{C}_m ,
```

and relating ``\boldsymbol{\varepsilon}^0`` to the macroscopic strain
``\boldsymbol{E}`` through the average of the same equation yields

```math
\boldsymbol{\varepsilon}^I = \boldsymbol{E}
  - \sum_J \mathbb{T}^{IJ} : \delta\mathbb{C}^J : \boldsymbol{\varepsilon}^J
  + \mathbb{E}^0 : \sum_K f_K\, \delta\mathbb{C}^K : \boldsymbol{\varepsilon}^K .
```

!!! warning "The paper's own sign is the opposite one"
    [molinari1996](@cite) writes these two equations with a ``+`` in front of the
    interaction sum, because their ``\Gamma^{IJ}`` is the opposite of this package's
    ``\mathbb{T}^{IJ}`` (their self term is ``\Gamma^{II} = -\mathbb{P}_0``). The
    equations above are theirs with that single flip applied — see the
    [convention note](@ref th-interaction). Nothing else in this page or in the
    implementation carries a compensating sign.

Two identities connect this to the rest of the package:

```math
\mathbb{T}^{II} = +\,\mathbb{P}_0 , \qquad \mathbb{E}^0 = +\,\mathbb{P}_0 ,
```

the self term of the interaction family **being** the Hill tensor, and the far-field
operator being the Hill tensor of the inclusion shape. For an isotropic matrix and
spherical inclusions,

```math
\mathbb{P}_0 = \frac{1}{3k_m+4\mu_m}\,\mathbb{J}
  + \frac{3\,(k_m+2\mu_m)}{5\,\mu_m\,(3k_m+4\mu_m)}\,\mathbb{K} .
```

## Families and the cluster

An infinite array is reduced to ``N`` unknowns by taking a periodic elementary
representative volume containing ``N`` inclusions. Each inclusion of the cell carries a
**family** of periodic images, all subjected to the same fields, and the sums are
truncated to the images lying within a cluster radius ``R_c`` of the receiver:

```math
\bar{\mathbb{T}}_{IK} = \sum_{J \in \mathcal{C}_I \cap \mathcal{F}_K,\; J \ne I}
   \mathbb{T}^{IJ} ,
```

``\mathcal{C}_I`` being the cluster attached to ``I`` and ``\mathcal{F}_K`` the family
``K``. Convergence as ``R_c \to \infty`` is proved in their Appendix B — see the
[cutoff discussion](@ref th-interaction).

## The block system

Splitting ``\mathbb{T}^{II}`` out of the sum and writing
``\boldsymbol{\varepsilon}^K = \mathbb{A}^K : \boldsymbol{E}`` turns the above into a
linear system whose unknowns are order-4 tensors:

```math
\sum_K \mathbb{M}_{IK} : \mathbb{A}^K = \mathbb{I} ,
\qquad
\boxed{\;\mathbb{M}_{IK} = \delta_{IK}\,\mathbb{I}
  + \big[\bar{\mathbb{T}}_{IK} + (\delta_{IK} - f_K)\,\mathbb{P}_0\big]
    : \delta\mathbb{C}_K \;}
```

A **single** expression covers both the diagonal and the off-diagonal block, which is
the practical dividend of the convention adopted here: in the opposite one the
two cases carry opposite signs and have to be written separately.

The matrix localization follows from the strain average rule and the effective
stiffness from the stress one:

```math
f_m\, \mathbb{A}_m = \mathbb{I} - \sum_I f_I\, \mathbb{A}_I ,
\qquad
\mathbb{C}^{\mathrm{hom}} = f_m\, \mathbb{C}_m : \mathbb{A}_m
  + \sum_I f_I\, \mathbb{C}_I : \mathbb{A}_I .
```

The implementation flattens the system onto the Kelvin-Mandel basis — where ``:``
becomes an ordinary matrix product — and solves it with one dense factorization, rather
than reproducing the tensorial Gauss elimination of the paper.

## The Mori-Tanaka limit

Reduce the cluster to its own receiver. Every ``\bar{\mathbb{T}}`` vanishes and

```math
\mathbb{M}_{II} = \mathbb{I} + (1-f_I)\,\mathbb{P}_0 : \delta\mathbb{C}_I ,
\qquad
\mathbb{M}_{IK} = -f_K\, \mathbb{P}_0 : \delta\mathbb{C}_K ,
```

which is the Mori-Tanaka system term by term.

!!! note "An exact degeneracy, not an approximation"
    With `cluster_radius = 0` the cluster model **is** [`MoriTanaka`](@ref), as an
    algebraic identity — Appendix C of [molinari1996](@cite).
    The test suite checks it to machine precision, for one family and for two, in
    elasticity and in conduction. It is the sharpest available statement that the
    assembly of ``\mathbb{M}`` is right.

## What the cluster buys

Since ``\bar{\mathbb{T}}`` has [no isotropic part](@ref th-interaction), the correction
is purely deviatoric:

* the effective **bulk** modulus of a cubic array equals the Mori-Tanaka one exactly;
* the effective **shear** moduli do not, and the array is *cubic*, not isotropic — the
  two shear constants differ;
* for a statistically isotropic arrangement the orientation average of
  ``\bar{\mathbb{T}}`` vanishes and the model collapses back onto Mori-Tanaka. The
  cluster model is therefore a statement about *anisotropic or specific* arrangements,
  which is why it needs a [`ParticleAssembly`](@ref) rather than an `RVE`.

Their own results, reproduced in `scripts/91`: the estimate is flat beyond
``R_c \approx 2`` periods; Mori-Tanaka *overestimates* the shear modulus of a
simple-cubic array of stiff spheres, so it is not a bound for that distribution; and at
equal volume fraction the SC, BCC and FCC arrangements differ, the simple-cubic one
being the softest.

## API

See [API — Particle assemblies](@ref api-assemblies).
