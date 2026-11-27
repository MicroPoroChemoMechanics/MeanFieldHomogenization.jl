# [Cross-property connections](@id th-cross-property)

!!! info "Before this page"
    [Extension to conductivity](@ref th-conductivity), where the conductive
    counterparts of the Hill and contribution tensors are defined, and
    [Crack opening displacement and compliance](@ref th-cod-tensors) for the
    contribution of a crack.

The same pores and cracks lower the stiffness and the conductivity of a
material. When both changes are written with the contributions of the
inhomogeneities, a measurement of one property tells something of the other.
This page states when such a connection is exact, when it is an approximation,
and what the approximation drops; the subject is reviewed by
[sevostianov2009](@citet) and [kachanov2018](@citet).

## 1. Two sums over the same inhomogeneities

Without interactions between inhomogeneities, the compliance and the
resistivity ``\boldsymbol r = \boldsymbol K^{-1}`` of a material are those of the
matrix plus one contribution per inhomogeneity,

```math
\mathbb S = \mathbb S_0 + \sum_i \mathbb H_i ,
\qquad
\boldsymbol r = \boldsymbol r_0 + \sum_i \boldsymbol R_i ,
```

``\mathbb H_i`` and ``\boldsymbol R_i`` (per unit volume of the material) being the
compliance and resistivity contributions of [Extension to conductivity](@ref th-conductivity)
and of the crack pages. A connection between the two properties exists when
each ``\mathbb H_i`` is a function of the corresponding ``\boldsymbol R_i``, the same
for all ``i``, and linear: the sums then go through it.

**Parallel inhomogeneities of one shape.** All the ``\mathbb H_i`` are
proportional to one tensor, all the ``\boldsymbol R_i`` to another, with the same
weight (the volume, or the crack density): every component of
``\mathbb S - \mathbb S_0`` is proportional to every nonzero component of
``\boldsymbol r - \boldsymbol r_0``, and the connection is exact whatever the
concentration in that approximation.

**Other distributions.** Inhomogeneities of several orientations add tensors
of different directions, and a linear map from ``\boldsymbol R_i`` to ``\mathbb H_i``
common to all orientations exists only if ``\mathbb H_i`` depends on the
orientation through ``\boldsymbol R_i`` alone. Cracks show what fails and how
little it weighs.

## 2. Cracks

A dry circular crack of radius ``a`` and normal ``\underline n``, in an isotropic
matrix of Young modulus ``E_0``, Poisson ratio ``\nu_0`` and conductivity ``k_0``,
contributes, per unit volume ``V``,

```math
\boldsymbol R = \frac{8}{3k_0}\frac{a^3}{V}\,\underline n\otimes\underline n ,
\qquad
H_{ijkl} = h\,\frac{a^3}{V}\Big[\tfrac14\big(\delta_{ik} n_j n_l + \delta_{il} n_j n_k
  + \delta_{jk} n_i n_l + \delta_{jl} n_i n_k\big) - \frac{\nu_0}{2}\,n_i n_j n_k n_l\Big],
```

with ``h = 32(1 - \nu_0^2)/(3(2 - \nu_0)E_0)`` (the package computes both; their
forms are those of [Crack opening displacement and compliance](@ref th-cod-tensors)).
Summed over the cracks, the resistivity depends on the second-rank crack
density ``\boldsymbol\alpha = \frac1V\sum a^3\,\underline n\otimes\underline n`` only, the
compliance on ``\boldsymbol\alpha`` and on the fourth-rank
``\boldsymbol\beta = \frac1V\sum a^3\,\underline n\otimes\underline n\otimes\underline n\otimes\underline n``.
Dropping the ``\boldsymbol\beta`` term, of relative weight ``\nu_0/2`` against the
``\boldsymbol\alpha`` term, ``\boldsymbol\alpha = \frac{3k_0}{8}(\boldsymbol r - \boldsymbol r_0)``
gives the compliance as a function of the resistivity, for any orientation
distribution:

```math
S_{ijkl} - S^0_{ijkl} \approx \frac{3hk_0}{32}\big(\delta_{ik}\Delta r_{jl} + \delta_{il}\Delta r_{jk}
  + \delta_{jk}\Delta r_{il} + \delta_{jl}\Delta r_{ik}\big),
\qquad \Delta\boldsymbol r = \boldsymbol r - \boldsymbol r_0 .
```

For cracks of random orientations both sides are isotropic, and the exact
changes of the Young modulus and of the conductivity are in a fixed ratio;
the approximation changes that ratio by the dropped ``\boldsymbol\beta`` term: by
the factor ``1 - 0.3\,\nu_0`` (the exact dilute ratio of the changes of ``E`` and
``k`` is ``2(1-\nu_0^2)(10-3\nu_0)/(5(2-\nu_0))``, the approximate one
``4(1-\nu_0^2)/(2-\nu_0)``). The application
[Elastic and conductive properties of porous and cracked materials](@ref app-cross-property)
computes both ratios from the contributions.

## 3. Pores

For spheroidal pores the contributions are not proportional to one another
across orientations, and the connection is an approximation of the same kind:
``\mathbb H`` is replaced by its closest tensor built on the second-rank
``\underline n\otimes\underline n`` that ``\boldsymbol R`` holds, and the error is the size of
what that leaves out. For spheres the two contributions are isotropic and the
connection is exact: ``(E_0 - E)/E`` is a fixed multiple of ``(k_0 - k)/k``.

## 4. Beyond the non-interacting approximation

A scheme replaces the sums of §1 by its own estimate, and the connection
becomes a relation between two estimates of the same cell, which the package
computes by homogenizing one cell that carries both properties `:C` and `:K`.
Whether the ratio survives the interactions is then a question for each
scheme, answered numerically in the application.

## 5. In the package

[`compliance_contribution`](@ref) and [`resistivity_contribution`](@ref) give
``\mathbb H`` and ``\boldsymbol R`` of any inclusion; [`homogenize`](@ref) with `:C` and
`:K` on the same cell gives both estimates. Syntax in
[Conductivity](@ref man-conductivity).

## Where to go next

The next page carries conduction into the pores, where a fluid flows:
[Permeability from Stokes flow](@ref th-brinkman).
