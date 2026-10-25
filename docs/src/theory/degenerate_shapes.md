# [Degenerate shapes and extreme aspect ratios](@id th-degenerate-shapes)

!!! info "Before this page"
    [Hill polarization tensors](@ref th-hill-tensors) for the closed forms and
    the cubatures, and [COD tensors](@ref th-cod-tensors) for the cracks.

The closed forms of the Hill, Eshelby and crack opening tensors are written for
a generic shape. At the shapes where two semi-axes meet, where a spheroid
becomes a sphere, a disc or a needle, or where the two acoustic roots of a
transversely isotropic matrix coincide, they reach removable ``0/0`` limits.
Next to those shapes they are differences of nearly equal terms. Evaluated as
written, they lose every digit there. Their derivatives fare worse: the
derivative with respect to a semi-axis at the point where it meets another
one, or with respect to a modulus at an isotropic point, is a limit as well,
and a code that merges the two semi-axes or the two roots drops it.

This page lists those situations, says for each one what goes wrong, and how
the package evaluates it instead. Every figure quoted is a relative error,
measured against a reference that shares no code with the formula it checks:
the nested cubature at tolerances ``10^{-14}/10^{-13}``, the same integrals in
256-bit arithmetic, and, for derivatives, `ForwardDiff` through the cubature
and finite differences of the closed form.

| Situation | What goes wrong | Treatment | Accuracy |
| :-- | :-- | :-- | :-- |
| spheroid near the sphere, isotropic matrix | divided difference of two potentials | the potentials as the integrals they stand for | ``5\times10^{-16}`` |
| triaxial ellipsoid with two close semi-axes | elliptic integrals and their divided differences | expansion around the spheroid | ``10^{-15}`` |
| derivative where two semi-axes meet | the merged shape no longer depends on their gap | shapes merged only when their derivatives coincide | ``10^{-14}`` |
| spheroid in a TI matrix: sphere, coincident roots, disc, needle | cancellations, a square root of zero | three regimes in the mean and squared half-difference of the roots | ``10^{-14}`` |
| elliptical crack near the circle | ``(\mathcal{K}-\mathcal{E})/k^2`` | power series | ``10^{-15}`` |
| a cubature under `ForwardDiff` | `NaN` error estimate, dual integration bounds | error norm on the values, plain bounds | — |
| self-consistent estimate of TI class | estimate rebuilt as a full tensor | projection within the class | ``\times 90`` faster |

The integrals behind the first four rows live in
`src/Core/spheroid_integrals.jl`.

## [The integrals of a spheroid](@id th-degenerate-integrals)

With ``D(a,z) = a + (1-a)\,z^2`` and ``w = 1-z^2``, over ``z\in[-1,1]``, three
families of integrals carry the shape of a spheroid of axial over transverse
semi-axis ``\omega``:

```math
\begin{aligned}
I_n(a) &= \int_{-1}^{1}\frac{z^n\,\mathrm{d}z}{D(a,z)},\qquad
J_{n,q}(a_1,a_2) = \int_{-1}^{1}\frac{z^n w^q\,\mathrm{d}z}{D(a_1,z)\,D(a_2,z)},
\\
M_{n,k,p}(m) &= \int_{-1}^{1}\frac{z^n w^k\,\mathrm{d}z}{D(m,z)^p}.
\end{aligned}
```

With ``a = \omega^2`` they give the Newton potentials of the spheroid; with
``a_i = \omega^2\gamma_i^2``, ``\gamma_i^2`` a root of the acoustic polynomial
of a transversely isotropic matrix, the Hill tensor of a coaxial spheroid in
it. ``I_n`` is elementary through
``R_C(a,1) = \operatorname{acosh}\sqrt{a}/\sqrt{a-1}``, which is
``\arccos\sqrt{a}/\sqrt{1-a}`` below ``a = 1``:

```math
I_0 = \frac{2R_C}{\sqrt a},\qquad
I_2 = \frac{2\,(\sqrt a\,R_C - 1)}{a-1},\qquad
I_4 = \frac{2\,(a\sqrt a\,R_C - 1 - \tfrac43(a-1))}{(a-1)^2}.
```

These cancel as ``a\to1``, the sphere. Writing ``D = 1 + k\,w`` with
``k = a-1``, all three families are instead power series in ``k`` there, with
the coefficients ``\beta_{n,j} = \int_{-1}^{1} z^n w^j\,\mathrm{d}z =
\beta_{n,j-1}\,2j/(2j+n+1)``:

```math
I_n = \sum_{j\ge0}(-1)^j k^j\,\beta_{n,j},\qquad
M_{n,k,p} = \sum_{j\ge0}\binom{-p}{j} k^j\,\beta_{n,k+j}.
```

They are used for ``|a-1|\le 1/4``, where 32 terms leave a truncation below
``10^{-18}`` (64 for the moments of high power ``p``). Away from the sphere the
moments follow from ``U_{n,p} = M_{n,0,p}`` and
``w^k = (D-1)^k/(m-1)^k``; ``U_{n,p}`` follows from ``I_0`` by two recurrences,
the integral of ``\mathrm{d}(z/D^p)/\mathrm{d}z`` and ``z^2 = (D-m)/(1-m)``:

```math
U_{0,p+1} = \frac{1 + (p-\tfrac12)\,U_{0,p}}{p\,m},\qquad
U_{n+2,p} = \frac{U_{n,p-1} - m\,U_{n,p}}{1-m}.
```

## Spheroid near the sphere, isotropic matrix

The closed forms in an isotropic matrix — Hill, Eshelby and concentration
tensors, ``\mathbb{U}^{\boldsymbol{A}}`` and ``\mathbb{V}^{\boldsymbol{A}}``,
and conduction — are built on the Newton potentials of the ellipsoid. For a
spheroid of axial semi-axis ``L`` and transverse semi-axis ``R``, the mixed
second-order potential is classically the divided difference
``I_{LR} = (I_R - I_L)/(L^2 - R^2)`` of two first-order ones that tend to the
same value at the sphere. A Hill tensor built from it is off by ``10^{-5}``
at ``|\omega-1| = 10^{-4}`` and by more than its own size at
``|\omega-1| = 2\times10^{-6}``.

`newton_potential_3d` evaluates instead, with ``m = L^2/R^2``,

```math
I_L = 2\pi\,I_2(m),\qquad
I_{LR} = \frac{2\pi}{R^2}\int_{-1}^{1}\frac{z^4\,\mathrm{d}z}{D(m,z)^2}
= \frac{2\pi}{R^2}\,M_{4,0,2}(m),
```

then ``I_R = (4\pi - I_L)/2``, ``I_{LL} = 4\pi/(3L^2) - 2I_{LR}/3`` and
``I_{RR} = \pi/R^2 - I_{LR}/4``, none of which cancels.

```@raw html
<details><summary>Where the integral forms come from</summary>
```

The potentials are ``I_i = 2\pi\,L R^2\int_0^\infty \mathrm{d}u\,/\,\big((a_i^2+u)\,\Delta(u)\big)``
with ``\Delta(u) = \sqrt{(L^2+u)(R^2+u)^2}``. The change of variable
``L^2 + u = L^2/z^2`` maps ``u\in[0,\infty)`` onto ``z\in(0,1]`` with
``\mathrm{d}u = -2L^2 z^{-3}\mathrm{d}z`` and
``R^2 + u = (R^2/z^2)\,D(m,z)``. The axial potential becomes

```math
I_L = 2\pi L R^2\int_0^1 \frac{2L^2}{z^3}\,\frac{z^2}{L^2}\,\frac{z}{L}\,
\frac{z^2}{R^2 D(m,z)}\,\mathrm{d}z
= 4\pi\int_0^1\frac{z^2\,\mathrm{d}z}{D(m,z)} = 2\pi\,I_2(m),
```

and the mixed one, with one more factor ``(L^2+u)^{-1}(R^2+u)^{-1}``, the
moment ``M_{4,0,2}`` divided by ``R^2``.

```@raw html
</details>
```

Against the nested cubature at tight tolerances, the Hill tensor of a
spheroid in an isotropic matrix is now within ``5\times10^{-16}`` at every
distance from the sphere tried, from ``10^{-9}`` to ``10^{-1}``.

## Triaxial ellipsoid close to a spheroid

A triaxial ellipsoid goes through Legendre's elliptic integrals, and its mixed
potentials through divided differences such as
``I_{ab} = (I_b - I_a)/(a^2 - b^2)``. When two semi-axes come together the
first cancel as ``k^2\to0`` and the second as ``a^2-b^2\to0``: the error grows
as ``\epsilon/\delta^2`` with the relative gap ``\delta``, to ``5\times10^{-5}``
at ``\delta = 10^{-6}``. Taking the two semi-axes as equal below some
tolerance only trades that error for one of the order of the tolerance.

When the closer pair of semi-axes ``B_1, B_2`` differ by less than 2 % (in
squares), the potentials are expanded around the spheroid they are close to.
With ``L`` the third semi-axis, ``\ell = L^2``, ``b_i = B_i^2`` and the same
change of variable as above, every potential takes the form

```math
2\pi\,L B_1B_2\;L^{2-2e_L}\,B_1^{-2e_1}\,B_2^{-2e_2}
\int_{-1}^{1}\frac{z^n\,\mathrm{d}z}{D_1^{\,e_1}\,D_2^{\,e_2}},
\qquad D_i = D(m_i,z),\quad m_i = \ell/b_i,
```

where ``e_L, e_1, e_2\in\{\tfrac12,\tfrac32,\tfrac52\}`` are the exponents of
``\ell+u``, ``b_1+u``, ``b_2+u`` in the integrand and
``n = 2(e_L+e_1+e_2)-3``. With ``\bar m = (m_1+m_2)/2`` and
``d = (m_1-m_2)/2``, ``D_{1,2} = \bar D \pm d\,w``, and

```math
\frac{1}{D_1^{\,e_1}D_2^{\,e_2}}
= \sum_{k\ge0} c_k\,d^k\,\frac{w^k}{\bar D^{\,e_1+e_2+k}},
\qquad
\sum_k c_k\,y^k = (1+y)^{-e_1}(1-y)^{-e_2},
```

so that each potential is a combination of the moments
``M_{n,k,e_1+e_2+k}(\bar m)``. The series converges as
``(|d|/\bar m)^k = \big(|b_1-b_2|/(b_1+b_2)\big)^k``; eleven terms leave a
truncation below ``10^{-18}`` at the 2 % threshold, beyond which the elliptic
integrals are accurate to ``10^{-13}``. The expansion is exact when
``B_1 = B_2``, derivatives included, which the elliptic integrals and their
divided differences cannot be.

Against the cubature, a triaxial ellipsoid with two semi-axes ``10^{-9}`` to
``10^{-2}`` apart — next to a flat, an elongated or a near-spherical
spheroid — is within ``10^{-15}``, and within ``6\times10^{-13}`` on the
elliptic side of the threshold. Two semi-axes of plain floating-point type
within ``10^{-10}`` of each other are still taken as equal, which costs an
error of that order.

## Where two semi-axes meet: derivatives

An ellipsoid whose two semi-axes are equal is classified as a spheroid (or a
sphere), and its closed form depends on the shared semi-axis only. Differentiate
with respect to one of the two at the point where they meet, and that closed
form has nothing to give: merged, the derivative of the Hill tensor of a sphere
with respect to one of its semi-axes is zero, and the derivatives along the
gap of the shapes next to a spheroid are off by 100 % to 550 %, without a
word.

Two semi-axes carried by `ForwardDiff.Dual` numbers are therefore merged only
when their derivatives coincide too; otherwise the ellipsoid is triaxial, and
the expansion above, exact at ``d = 0``, carries the derivative along the gap.
The same rule classifies ellipses, cylinders and cracks. A spheroid
parametrized by its aspect ratio, ``\mathrm{Spheroid}(\omega)``, keeps its two
equal semi-axes constant, so it stays a spheroid under differentiation. For
conduction, the derivative of the Hill tensor of a sphere of radius 1 with
respect to its third semi-axis is

```math
\frac{\partial\boldsymbol{P}}{\partial c} = \frac{1}{15\,k}
\begin{pmatrix} 2&0&0\\0&2&0\\0&0&-4 \end{pmatrix},
```

which the package returns to rounding. Against `ForwardDiff` through the
cubature these derivatives agree to ``2\times10^{-14}``, and against central
finite differences of the closed form to the accuracy of the differences,
``10^{-10}`` to ``5\times10^{-9}``.

## [Spheroid in a transversely isotropic matrix](@id th-degenerate-ti-spheroid)

The Walpole coefficients of a spheroid coaxial with a transversely isotropic
matrix, given in [Hill polarization tensors](@ref th-hill-ti-coaxial), involve
``J_n = J_{n,0}`` of the two acoustic roots ``a_{1,2} = \omega^2\gamma_{1,2}^2``
and ``I_n`` of ``a_3 = \omega^2\gamma_3^2``. The package evaluates them with

```math
\begin{gathered}
j_4 = \frac{J_{4,0}}{C_{2323}C_{3333}},\quad
l_{21} = \frac{J_{2,1}}{C_{2323}C_{3333}},\quad
l_{02} = \frac{J_{0,2}}{C_{2323}C_{3333}},
\\
i_2 = \frac{I_2(a_3)}{C_{2323}},\quad
i_w = \frac{I_0(a_3)-I_2(a_3)}{C_{2323}},
\end{gathered}
```

as

```math
\begin{aligned}
P_1 &= \tfrac12\big(C_{2323}\,j_4 + \omega^2 C_{1111}\,l_{21}\big),\\
P_2 &= \tfrac14\,\omega^2\big(C_{3333}\,l_{21} + \omega^2 C_{2323}\,l_{02}\big),\\
P_3 &= -\frac{\omega^2}{2\sqrt2}\,(C_{2323}+C_{1133})\,l_{21},\\
P_5 &= \tfrac12 P_2 + \tfrac18\,\omega^2\,i_w,\\
P_6 &= \tfrac18\big(C_{3333}\,j_4 - 2\omega^2 C_{1133}\,l_{21}
       + \omega^4 C_{1111}\,l_{02} + i_2\big).
\end{aligned}
```

These are the coefficients of that page, in which ``J_0, J_2, J_4`` appear in
differences of terms ``\omega^2`` or ``\omega^4`` larger than the result for a
needle; the combinations ``J_{2,1} = J_2 - J_4`` and
``J_{0,2} = J_0 - 2J_2 + J_4`` are computed directly instead.

The ``J`` are symmetric in the two roots: they depend on them only through the
mean ``m`` and the squared half-difference ``d^2``,

```math
\begin{aligned}
m &= -\frac{\omega^2 b}{2\,C_{2323}C_{3333}},
\qquad b = C_{1133}^2 + 2C_{1133}C_{2323} - C_{1111}C_{3333},
\\
d^2 &= \frac{\omega^4\,\big(b^2 - 4\,C_{2323}C_{3333}\,C_{1111}C_{2323}\big)}{4\,(C_{2323}C_{3333})^2},
\end{aligned}
```

rational in the moduli and free of their scale. Four situations would
otherwise break the evaluation:

- **near the sphere**, ``a_i\to1``, the closed forms cancel;
- **coincident roots** — an isotropic matrix stored as transversely
  isotropic, or one within ``10^{-6}`` of isotropy — make the partial fractions
  divide by ``a_1-a_2``, and the square root of ``d^2`` has an infinite
  derivative; merging the roots there instead makes the derivatives along
  every modulus that breaks isotropy wrong by 24 % to a factor 14, at an
  isotropic point included;
- **a flat spheroid**, ``a_i\to0``, makes the partial fractions of ``J_4``
  cancel as ``1/a``;
- **a needle**, ``a_i\to\infty``, makes those of ``J_{2,1}`` and ``J_{0,2}``
  cancel as ``a``.

They are evaluated in three regimes:

1. **near the sphere**, ``|a_i-1|\le1/4``: the power series
   ``J_{n,q} = \sum_j (-1)^j h_j\,\beta_{n,q+j}``, where
   ``h_j = \sum_{p+q=j} k_1^p k_2^q = (k_1+k_2)\,h_{j-1} - k_1k_2\,h_{j-2}`` needs
   only ``k_1+k_2 = 2(m-1)`` and ``k_1k_2 = (m-1)^2 - d^2``;
2. **nearly coincident roots**, ``|d|\le|m|/20``: with ``\bar D = D(m,z)``,
   ``1/(D_1D_2) = \sum_j d^{2j} w^{2j}/\bar D^{2j+2}``, truncated at ``j = 4``,
   so that ``J_{n,q} = \sum_j d^{2j} M_{n,2j+q,2j+2}(m)``;
3. **elsewhere**, partial fractions,
   ``J_{n,q} = \mathrm{PF}[M_{n,q}]`` with
   ``\mathrm{PF}[f] = \big((1-a_1)f(a_1) - (1-a_2)f(a_2)\big)/(a_2-a_1)``,
   arranged so as not to cancel. For a flat spheroid (``|a_i|<1``), ``J_2``
   and ``J_4`` come from ``J_0`` by ``z^2 = (D_1-a_1)/(1-a_1)``. For a needle,
   ``w = (D - z^2)/a`` turns the single-root integrals into a part rational in
   ``a``, summed exactly, and a remainder in ``I_4``:

   ```math
   J_{2,1} = \frac{2}{3a_1a_2} - \mathrm{PF}\Big[\frac{I_4(a)}{a}\Big],\qquad
   J_{0,2} = \frac{2}{a_1a_2} - \frac{2\,(a_1+a_2)}{3\,(a_1a_2)^2}
            + \mathrm{PF}\Big[\frac{I_4(a)}{a^2}\Big],
   ```

   whose partial fractions cancel as ``1/d`` only.

!!! note "The sign of the partial fractions"
    For ``q = 0`` and ``a_1 \neq a_2``, ``\mathrm{PF}`` gives
    ``J_n = \big((1-a_1)I_n(a_1) - (1-a_2)I_n(a_2)\big)/(a_2-a_1)``, as the
    Python script appended to [barthelemyIJES2020_hilltrans](@citet) does. The
    last row of its Table D.2 prints the numerator with the opposite sign, which
    gives ``-J_n``; a quadrature of the definition settles it.

No square root of ``d^2`` is taken in the first two regimes, so derivatives are
exact where the roots coincide. In derivatives, the split ``|d| = |m|/20``
balances the ``O(d^8)`` truncation of the second regime against the
cancellation of the third, which grows as ``1/d^3``.

Against the same integrals in 256-bit arithmetic, on generic, strongly
anisotropic and nearly isotropic matrices (``10^{-2}`` to ``10^{-8}``),
isotropic values included, and on ``\omega`` from ``10^{-8}`` to ``10^{6}``,
the Walpole coefficients are within ``10^{-14}``. Their derivatives with
respect to the moduli are within ``10^{-12}`` up to ``\omega = 30``, and
``2\times10^{-10}`` at ``\omega = 10^3``.

## Elliptical crack near the circle

The crack opening tensor of an elliptical crack of aspect ratio ``\eta = b/a``
in an isotropic matrix involves
``\mathcal{S} = (\mathcal{K}-\mathcal{E})/k^2`` and
``\mathcal{C} = \mathcal{K} - \mathcal{S}``, ``k^2 = 1-\eta^2``, which cancel
as ``\eta\to1`` with a relative error near ``\epsilon/k^2``. Below
``k^2 = 0.1`` they are replaced by the series of the complete integrals,

```math
\mathcal{K} = \frac{\pi}{2}\sum_{n\ge0} a_n k^{2n},\qquad
\mathcal{S} = \frac{\pi}{2}\sum_{n\ge1}\frac{2n}{2n-1}\,a_n k^{2n-2},\qquad
a_n = \Big(\frac{(1/2)_n}{n!}\Big)^2,
```

and ``\mathcal{E} = \mathcal{K} - k^2\mathcal{S}``: within ``1.1\times10^{-15}``
of the same closed forms in 256-bit arithmetic, from the circle to
``\eta = 0.5``.

!!! note "A kink at the circle"
    ``\boldsymbol{B}`` depends on the ratio of the smaller semi-axis to the
    larger, so the roles of ``a`` and ``b`` swap at the circle, and
    ``\boldsymbol{B}`` has a kink there as a function of either: ``B_{nn}`` is
    smallest at the circle and grows on both sides. Its derivative at the
    circle is one-sided. `ForwardDiff` orders two dual numbers of equal value
    by their partials, so the semi-axis being increased is sorted as the
    larger one, and the derivative returned is the one in the direction of
    the increase. Next to the circle, it agrees with the cubature to
    ``2\times10^{-14}``.

## Cubatures under `ForwardDiff`

Two traps stop a cubature under differentiation.

- **The error estimate.** QuadGK compares the Kronrod and Gauss estimates of
  each segment through a norm. `LinearAlgebra.norm` rescales by the largest
  entry and divides by it unless that entry `iszero`, and a dual number of
  value zero with non-zero partials is not `iszero`. On a segment where the two
  rules agree to the last bit, which a smooth integrand produces, the estimate
  comes out `NaN` and the integration stops with a `DomainError` — the
  derivative of the anisotropic conduction tensor of a sphere, for one. Every
  cubature of the package hands QuadGK the norm of the values: the refinement
  makes the
  decisions an evaluation without derivatives makes, so the value of a
  differentiated evaluation is that evaluation itself, and the derivatives
  ride on the same nodes.
- **The integration bounds.** QuadGK has no rule for a dual interval: bounds
  taken in the number type of the crack make the derivative with respect to a
  crack semi-axis in an anisotropic matrix a `MethodError`. The bounds are
  plain numbers; the derivatives live in the integrand.

## Self-consistent estimates of transversely isotropic class

The self-consistent scheme projects each iterate on the major-symmetric
tensors. The products of transversely isotropic tensors that build an iterate
land in the six- or eight-parameter Walpole form. Projected by rebuilding them
as generic tensors, the class is lost, and the Hill tensor of a coaxial
spheroid in the next iterate goes through a cubature. In the Walpole
basis the major transpose swaps ``\ell_3`` and ``\ell_4`` and changes the sign
of the major-antisymmetric ``\ell_7``, ``\ell_8``, so the projection is the
five-parameter tensor

```math
(\ell_1,\ \ell_2,\ \tfrac12(\ell_3+\ell_4),\ \ell_5,\ \ell_6),
```

exact, and the estimate keeps its class. On a 3:1 spheroid the scheme then
uses the closed form throughout: 46 µs instead of 4.2 ms, and the fixed point
agrees with the one reached through the cubature at tight tolerances to
``6\times10^{-10}``.

## Where to go next

The next chapter leaves the ellipsoid for patterns whose inner strain is not
uniform: [Layered sphere](@ref th-layered-sphere) solves the generalized Eshelby
problem of a concentric multilayer. The closed forms treated here are those of
[Hill polarization tensors](@ref th-hill-tensors) and
[COD tensors](@ref th-cod-tensors), and the self-consistent iteration that
relies on them is in [Self-consistent schemes](@ref th-scheme-self-consistent).
