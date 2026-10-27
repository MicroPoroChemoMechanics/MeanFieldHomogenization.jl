# [Strength of a frictional matrix with a non-associated flow rule](@id mp-theory-frictional-strength)

Soils, rocks, mortars and concretes have a strength that depends on the mean
stress, and their plastic flow is usually less dilatant than the normality rule
would make it. This page computes the macroscopic strength of a composite whose
matrix obeys such a law, a Drucker–Prager criterion with a non-associated flow
rule, around pores or rigid inclusions. The method is that of
[maghous2009](@citet), which extends to non-associated plasticity the
approach of [barthelemyDormieux2003](@citet) for a von Mises matrix
([the strength of a porous plastic solid](@ref mp-theory-strength)). The
equation numbers below are those of the 2009 paper.
[The example](@ref mp-ex-frictional-strength) runs it.

## The matrix

The matrix can carry the stresses whose mean part ``\sigma_m`` and deviatoric
norm ``\sigma_d = \sqrt{\boldsymbol{\sigma}^d:\boldsymbol{\sigma}^d}`` satisfy

```math
\sigma_d + T\,(\sigma_m - h) \le 0 ,
```

eq. (1): ``h`` is the isotropic tensile strength and ``T`` the friction
coefficient. Its plastic strain rate follows the potential
``\sigma_d + t\,\sigma_m``, eq. (15), so that away from the apex
``d_v = t\,d_d``, eq. (16), with ``d_v`` the volume strain rate and ``d_d`` the
norm of the deviatoric one. The dilatancy coefficient ``t`` lies between 0 and
``T``; the normality rule is ``t = T``. When ``t < T`` the theorems of limit
analysis no longer hold, and the macroscopic limit states have to be defined
otherwise.

## A viscous matrix that tends to the plastic one

The limit states are defined as limits of the states of a sequence of viscous
matrices indexed by ``a > 0`` (§3.2 of the paper). Writing
``Y = d_v - t\,d_d``, the matrix of index ``a`` has the stress

```math
\boldsymbol{\sigma} = \frac{\partial \psi_a}{\partial \boldsymbol{d}} + \sigma_0(Y)\,\boldsymbol{1},
\qquad
\psi_a = f_a(Y) + h\,d_v ,
```

eq. (17), where ``f_a`` is a convex, twice differentiable function, zero for
``Y \ge 0``, decreasing, and infinite at ``Y = -a``. Its mean stress is
``f_a'(Y) + h + \sigma_0`` and the norm of its deviator ``-t\,f_a'(Y)``. The
prestress ``\sigma_0 = f_a'(Y)\,(t/T - 1)``, eq. (19), is the one that places
every stress on the criterion: ``\sigma_d + T(\sigma_m - h) = -t f_a' + t f_a' = 0``.
Since ``f_a'(Y) \ne 0`` only for ``-a < Y < 0``, the flow rule ``d_v = t\,d_d`` is
reached as ``a \to 0``.

The same stress reads as a linear law with strain-rate dependent coefficients,
eqs. (20)–(22):

```math
\boldsymbol{\sigma} = 3k\,\mathbb{J}:\boldsymbol{d} + 2\mu\,\mathbb{K}:\boldsymbol{d} + \sigma^p\,\boldsymbol{1},
\qquad
k = \frac{f_a'(Y)}{Y},\quad
\mu = -\frac{t\,f_a'(Y)}{2\,d_d},\quad
\sigma^p = h + k\Big(\frac{t}{T}\,Y - d_v\Big),
```

with positive moduli, since ``Y < 0`` and ``f_a' < 0``. One checks that
``k\,d_v + \sigma^p = h + t f_a'/T``, the mean stress found above.

The package takes ``f_a(Y) = c\,(-Y)^3/(a + Y)`` for ``-a < Y < 0`` (and zero
beyond), with ``c = h/a``, which has the required properties; the limit states do
not depend on this choice.

## The modified secant method

The moduli and the prestress vary through the matrix with the local strain
rate. As in the [secant methods](@ref mp-theory-secant), the matrix is given
uniform values, taken at effective strain rates: the square roots of the second
moments of ``d_v`` and ``d_d`` over the matrix, eq. (25),
``d_v^e = \sqrt{\langle d_v^2\rangle_m}``, ``d_d^e = \sqrt{\langle d_d^2\rangle_m}``. The
matrix is then a linear viscous material with a uniform prestress, and the
composite a linear one, whose second moments follow from the derivatives of its
effective potential ``W(\boldsymbol{D})`` with respect to the moduli of the matrix,
at fixed prestress:

```math
\frac{1-\varphi}{2}\,\big(d_v^e\big)^2 = \frac{\partial W}{\partial k},
\qquad
(1-\varphi)\,\big(d_d^e\big)^2 = \frac{\partial W}{\partial \mu},
```

``\varphi`` being the fraction of the heterogeneities. Both equations form a
fixed point on ``(d_v^e, d_d^e)``, which the package solves by Newton's method.

The effective potential of the composite with a prestressed matrix is exact in
the two cases of the paper, for any linear scheme:

- **Pores.** A uniform prestress ``\sigma^p\boldsymbol{1}`` in the matrix is a uniform
  eigenstrain ``-\sigma^p/(3k)\,\boldsymbol{1}`` there. The same eigenstrain in the
  pores changes nothing, since they carry no stress: it is a uniform eigenstrain
  of the whole cell, which only shifts the strain rate. Hence
  ``W = \tfrac12\,\boldsymbol{D}':\mathbb{C}^{\mathrm{hom}}:\boldsymbol{D}' - (1-\varphi)\,(\sigma^p)^2/(2k)``
  with ``\boldsymbol{D}' = \boldsymbol{D} + \sigma^p/(3k)\,\boldsymbol{1}``, and the macroscopic stress is
  ``\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}:\boldsymbol{D}'``. Its derivatives are those of
  eq. (45) of the paper, which is recovered for the Mori–Tanaka scheme.
- **Rigid inclusions.** The strain rate vanishes in the inclusions, so that the
  average of ``\sigma^p d_v`` over the cell is ``\sigma^p D_v``:
  ``W = \tfrac12\,\boldsymbol{D}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{D} + \sigma^p D_v`` and
  ``\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}:\boldsymbol{D} + \sigma^p\boldsymbol{1}``, eqs. (33)–(34).

The derivatives of ``\mathbb{C}^{\mathrm{hom}}`` come from automatic differentiation
through the scheme, so that nothing specific to a scheme or to a shape is
written.

## The limit states

As ``a \to 0``, ``Y^e = d_v^e - t\,d_d^e \to 0`` and ``\mu/k \to 0``: the matrix of the
comparison composite becomes nearly incompressible about the dilatant volume
change ``t\,d_d^e``. The macroscopic stresses of the regularized composites tend to
the limit states, at first order in ``a``. For spherical heterogeneities and the
Mori–Tanaka scheme the paper finds the limit in closed form:

- **porous medium**: the elliptic domain of eq. (50),

  ```math
  \frac{1 + 2\varphi/3}{T^2}\,\Sigma_d^2 + \Big(\frac{3\varphi}{2T^2} - 1\Big)\,\Sigma_m^2 + 2(1-\varphi)\,h\,\Sigma_m - (1-\varphi)^2 h^2 = 0 ,
  ```

  which does not depend on the dilatancy ``t``, while the macroscopic flow rule
  does: it derives from the potential of eqs. (53) and (55), the same ellipse
  with ``T^2`` replaced by ``T\,t``. The flow is associated only if ``t = T``.
- **rigid inclusions**: a Drucker–Prager cone, eq. (40),
  ``\Sigma_d = T^{\mathrm{hom}}(h - \Sigma_m)``, reached by strain rates that follow the
  macroscopic flow rule ``D_v = t^{\mathrm{hom}} D_d``, eq. (39), with

  ```math
  t^{\mathrm{hom}} = t\,\sqrt{\frac{1 + \tfrac32\varphi}{1 - \tfrac23\varphi\,t^2}},
  \qquad
  T^{\mathrm{hom}} = T\,\frac{\sqrt{(1 + \tfrac32\varphi)(1 - \tfrac23\varphi\,t^2)}}{1 - \tfrac23\varphi\,t\,T}.
  ```

  Every point of the cone has the same normal, so that a strain rate fixes the
  point only through its distance, of order ``a``, to the flow rule. A strain
  rate off the flow rule has no limit state.

The tests of the package recover these closed forms from the general fixed
point, and the von Mises limit (``h \to \infty``, ``T \to 0`` with ``T h`` fixed, eq. (56))
from the [rigid–plastic secant composite](@ref mp-theory-strength).

## Numerical notes and scope

| | |
|:--|:--|
| regularization | ``a`` is taken relative to ``t\,\lvert\boldsymbol{D}\rvert``: ``d_v^e = t\,d_d^e + Y`` is nonnegative, so ``a`` must stay below ``t\,d_d^e``. A dilatancy ``t = 0`` is the limit of small ``t`` and is refused |
| convergence | first order in ``a``; two values of ``a`` and a Richardson extrapolation remove most of the error; below a relative ``a`` of about ``10^{-3}`` rounding takes over |
| heterogeneities | pores are given moduli ``10^{-8}`` times the smallest modulus of the matrix, rigid inclusions ``10^{8}`` times the largest, since ``k/\mu`` grows as ``1/a`` |
| no limit state | a strain rate without one (rigid inclusions off the flow rule) drives the fixed point to the edge ``Y = -a`` of the regularized matrix, and is refused |
| cost | about 40 µs per limit state for a two-phase Mori–Tanaka cell |
| scope | a matrix and one kind of heterogeneity, pores or rigid inclusions, any scheme and shape of the package; a single effective strain rate for the whole matrix, which, as the paper notes, underestimates the strength of a porous medium in compression near the porosity ``2T^2/3`` |
