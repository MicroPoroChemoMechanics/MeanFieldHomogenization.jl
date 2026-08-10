# [Two-inclusion interaction tensors](@id th-interaction)

Every one-site scheme of the package needs one object: the Hill tensor ``\mathbb{P}``
of a single inclusion in a reference medium. The two N-body schemes need one more —
the tensor measuring the field one inclusion induces in another. This page defines it,
gives the closed forms, and states why they are exact.

Notation follows [the conventions page](@ref th-notation): underline for vectors, bold
for order 2, blackboard bold for order 4. Two symbols are introduced here.

| Symbol | Object | Elasticity | Conduction |
| :-- | :-- | :-- | :-- |
| ``\mathbb{G}^0`` / ``\boldsymbol{G}^0`` | Green operator of the reference | order 4 | order 2 |
| ``\mathbb{T}^{ab}`` / ``\boldsymbol{T}^{ab}`` | interaction tensor of two inclusions | order 4 | order 2 |

[Molinari & El Mouden 1996](@cite molinari1996) write the interaction tensor
``\Gamma^{IJ}`` and [Brisard et al. 2014](@cite brisard2014) write it ``T^{kl}_{ab}``;
the letter ``\mathbb{T}`` is kept here because a Greek capital carries no order in this
typeface convention.

## The Lippmann-Schwinger equation

With a homogeneous reference ``\mathbb{C}_0`` and the polarization
``\boldsymbol{\tau} = (\mathbb{C}-\mathbb{C}_0):\boldsymbol{\varepsilon}``, the local
problem is equivalent to the integral equation of
[Zeller & Dederichs 1973](@cite zeller1973)

```math
\boldsymbol{\varepsilon}(\underline{x}) = \boldsymbol{E}
  + \int_{\mathbb{R}^d} \mathbb{G}^0(\underline{x}-\underline{y})
    : \boldsymbol{\tau}(\underline{y})\, \mathrm{d}V_{\underline{y}} ,
```

whose kernel is the **Green operator** of the reference medium, built from the Green
function ``\boldsymbol{G}`` by

```math
\mathbb{G}^0_{ijkl}(\underline{x})
  = \Big[\frac{\partial^2 G_{ik}}
              {\partial x_j\, \partial x_l}(\underline{x})\Big]_{(ij)(kl)} ,
```

the brackets denoting symmetrization on ``(i,j)`` and on ``(k,l)``. In conduction the
same object is the Hessian of the scalar Green function ``G``,

```math
\boldsymbol{G}^0_{ij}(\underline{x})
  = \frac{\partial^2 G}{\partial x_i\, \partial x_j}(\underline{x}) .
```

The kernel splits into a Dirac part and a regular part; integrating it over an
inclusion containing the source returns minus the Hill tensor, which is the identity
tying this page to the rest of the package.

## Definition

For two inclusions ``\Omega_a`` (receiver) and ``\Omega_b`` (source) whose centers are
separated by ``\underline{r}``,

```math
\mathbb{T}^{ab}(\underline{r}) = \frac{1}{|\Omega_a|}
   \int_{\Omega_a}\!\int_{\Omega_b}
     \mathbb{G}^0(\underline{x}-\underline{y})\,
     \mathrm{d}V_{\underline{y}}\, \mathrm{d}V_{\underline{x}} ,
```

so that ``\mathbb{T}^{ab}:\boldsymbol{\tau}_b`` is the **average field induced in
``a``** by a uniform polarization ``\boldsymbol{\tau}_b`` carried by ``b``. Over a
single inclusion the same integral gives

```math
\mathbb{T}^{aa} = -\,\mathbb{P}_a .
```

See [`self_interaction_tensor`](@ref) and the Mori-Tanaka limit on the
[cluster-model page](@ref th-cluster).

!!! warning "Sign conventions differ across the literature"
    [Molinari & El Mouden 1996](@cite molinari1996) and
    [Berveiller et al. 1987](@cite berveiller1987) use the convention above, with
    ``\mathbb{T}^{aa} = -\mathbb{P}``. [Brisard et al. 2014](@cite brisard2014) define
    their Green operator as mapping the polarization onto *minus* the induced field, so
    their influence tensors are the opposite of these and their self term is
    ``+\mathbb{P}``. The package uses the first convention throughout, because it keeps
    the self term a direct function of `hill_tensor`. Every kernel transcribing a
    formula from Brisard flips the sign explicitly.

## Exact closed forms for balls and disks

The solid mean-value expansion of a smooth field over a ball of radius ``a`` reads

```math
\langle f \rangle_{B_a} = f + \frac{a^2}{2(d+2)}\,\Delta f
   + O\!\left(a^4 \Delta^2 f\right) ,
```

and applying it once for the source and once for the receiver gives

```math
\mathbb{T}^{ab} = |\Omega_b|\left[\mathbb{G}^0(\underline{r})
   + \frac{a^2+b^2}{2(d+2)}\,\Delta\mathbb{G}^0(\underline{r})
   + O\!\left(\Delta^2\mathbb{G}^0\right)\right] .
```

The elastic Green function is **biharmonic**, so ``\Delta^2\mathbb{G}^0 \equiv 0`` away
from the source and *the expansion terminates*: the formula is exact for two
non-overlapping balls at any separation. In conduction ``\boldsymbol{G}^0`` is
harmonic, ``\Delta\boldsymbol{G}^0 \equiv 0``, and the interaction is exactly the
point-dipole field.

### Elasticity, 3D

Let ``\underline{n} = \underline{r}/R`` be the unit vector along the line of centers,
``R = \|\underline{r}\|``, and

```math
\kappa = -\frac{b^3}{12\,R^3\,\mu\,(1-\nu)} ,
\qquad
\rho^2 = \frac{a^2+b^2}{R^2} .
```

In the frame whose third axis is ``\underline{n}``
([Molinari & El Mouden 1996](@cite molinari1996), App. A;
[Berveiller et al. 1987](@cite berveiller1987)):

```math
\begin{aligned}
T_{1111} = T_{2222} &= \kappa\left(1 - 4\nu + \tfrac{9}{5}\rho^2\right), &
T_{1122} &= \kappa\left(-1 + \tfrac{3}{5}\rho^2\right), \\
T_{1133} = T_{3311} &= \kappa\left(2 - \tfrac{12}{5}\rho^2\right), &
T_{1212} &= \kappa\left(1 - 2\nu + \tfrac{3}{5}\rho^2\right), \\
T_{1313} = T_{2323} &= \kappa\left(1 + \nu - \tfrac{12}{5}\rho^2\right), &
T_{3333} &= \kappa\left(-8 + 8\nu + \tfrac{24}{5}\rho^2\right).
\end{aligned}
```

This set is transversely isotropic about ``\underline{n}`` and satisfies
``T_{1212} = (T_{1111}-T_{1122})/2`` identically, so it is carried **exactly** by five
Walpole coefficients rather than by an 81-component array — the storage described on
[the conventions page](@ref th-notation):

```math
T_1 = T_{3333},\quad
T_2 = T_{1111}+T_{1122},\quad
T_3 = \sqrt{2}\,T_{1133},\quad
T_5 = T_{1111}-T_{1122},\quad
T_6 = 2\,T_{1313},
```

```math
\mathbb{T}^{ab} = T_1\,\mathbb{W}_1 + T_2\,\mathbb{W}_2
  + T_3\,(\mathbb{W}_3+\mathbb{W}_4) + T_5\,\mathbb{W}_5 + T_6\,\mathbb{W}_6 .
```

Equivalently, basis-free, with
``\boldsymbol{N} = \underline{n}\otimes\underline{n}`` and
``\boldsymbol{Q} = \boldsymbol{1} - \boldsymbol{N}``:

```math
\mathbb{T}^{ab} = \frac{1}{3\,\mu\,(1-\nu)\,(R/b)^3}
  \Big[\gamma_1\,\mathbb{B}_{12} + 2(\gamma_3+\gamma_4)\,\mathbb{B}_2
       - \gamma_3\,\mathbb{B}_3 - \gamma_4\,\mathbb{B}_4\Big],
```

```math
\gamma_1 = \frac{1-2\nu}{\sqrt{2}},\qquad
\gamma_3 = \frac{5-10\nu+3\rho^2}{10},\qquad
\gamma_4 = \frac{5+5\nu-12\rho^2}{10},
```

```math
\begin{aligned}
\mathbb{B}_{12} &= \tfrac{\sqrt{2}}{6}\Big[
   (3\boldsymbol{N}-\boldsymbol{1})\otimes\boldsymbol{1}
   + \boldsymbol{1}\otimes(3\boldsymbol{N}-\boldsymbol{1})\Big], \\
\mathbb{B}_2 &= \tfrac{1}{6}\,
   (3\boldsymbol{N}-\boldsymbol{1})\otimes(3\boldsymbol{N}-\boldsymbol{1}), \\
\mathbb{B}_3 &= \boldsymbol{Q}\stackrel{s}{\boxtimes}\boldsymbol{Q}
   - \tfrac{1}{2}\,\boldsymbol{Q}\otimes\boldsymbol{Q}, \\
\mathbb{B}_4 &= \boldsymbol{N}\stackrel{s}{\boxtimes}\boldsymbol{Q}
   + \boldsymbol{Q}\stackrel{s}{\boxtimes}\boldsymbol{N} ,
\end{aligned}
```

### Conduction

The kernel is harmonic, so the dipole field is exact and the *receiver* radius drops
out entirely:

```math
\boldsymbol{T}^{ab} = \frac{b^3}{3\,\sigma_0 R^3}
  \big(3\,\underline{n}\otimes\underline{n} - \boldsymbol{1}\big)
\quad (3\text{D}),
\qquad
\boldsymbol{T}^{ab} = \frac{b^2}{2\,\sigma_0 R^2}
  \big(2\,\underline{n}\otimes\underline{n} - \boldsymbol{1}\big)
\quad (2\text{D}).
```

### Elasticity, plane strain

```math
\mathbb{T}^{ab} = \pi b^2\left[\mathbb{G}^0(\underline{r})
   + \frac{a^2+b^2}{8}\,\Delta\mathbb{G}^0(\underline{r})\right],
```

```math
\Delta\mathbb{G}^0(\underline{r}) = \frac{24}{8\pi\mu(1-\nu)R^4}
  \Big[\mathbb{K}_2
    - (2\boldsymbol{N}-\boldsymbol{1})\otimes(2\boldsymbol{N}-\boldsymbol{1})\Big],
```

``\mathbb{K}_2`` being the plane deviatoric projector. Note that
``\Delta\mathbb{G}^0`` does not depend on the reference Poisson ratio.

## The vanishing isotropic part

For any two distinct inclusions,

```math
T_{iijj} = 0 \qquad\text{and}\qquad T_{ijij} = 0 ,
```

i.e. the interaction tensor has **no isotropic part** (in conduction, it is traceless).
The decomposition into isotropic and anisotropic parts being linear, any sum of
interaction tensors inherits the property. The consequence is sharp: for an arrangement
of cubic symmetry the effective **bulk** modulus is blind to the spatial distribution
and coincides exactly with the Mori-Tanaka estimate. Only the shear response sees the
arrangement.

## General ellipsoids

Beyond balls the series does not terminate, and
[Brisard et al. 2014](@cite brisard2014), §4.2, expand the regular part of the kernel
about the line of centers. In moment form, with ``\boldsymbol{M}^2`` the normalized
second moment of a region about its centroid — for an ellipsoid of semi-axes
``(\rho_1,\dots,\rho_d)``,
``\boldsymbol{M}^2 = \mathrm{diag}(\rho_1^2,\dots,\rho_d^2)/(d+2)`` in the principal
frame —

```math
\mathbb{T}^{ab} = |\Omega_b|\left[\mathbb{G}^0(\underline{r})
  + \tfrac{1}{2}\,\big(\boldsymbol{M}^2_a + \boldsymbol{M}^2_b\big)_{pq}\,
    \partial_p\partial_q\, \mathbb{G}^0(\underline{r}) + \dots\right] .
```

Truncating there is `method = :multipole`, the default for non-spherical geometries. It
is asymptotic in (inclusion size / center distance) and reduces to the exact ball
formula when both moments are isotropic. `method = :quadrature` integrates the
definition directly by a product rule — geometry-agnostic, far slower, and the oracle
the closed forms are validated against.

## Anisotropic reference media

Everything above assumes an isotropic reference, whose Green operator is a closed
form. That is not a restriction of the method, only of the kernel, and it is lifted by
the line integral of Barnett (1972) and Willis (1975): for an arbitrary anisotropic
stiffness the displacement Green function is

```math
G_{ij}(\underline{x}) = \frac{1}{8\pi^2 r}
  \oint_{\underline{\xi}\perp\underline{n},\;\|\underline{\xi}\|=1}
    \big[\boldsymbol{K}(\underline{\xi})\big]^{-1}_{ij}\,\mathrm{d}\varphi ,
\qquad
K_{ij}(\underline{\xi}) = \xi_k\, C_{kijl}\, \xi_l ,
```

``\boldsymbol{K}`` being the acoustic (Christoffel) tensor and the contour the unit
circle in the plane perpendicular to ``\underline{n} = \underline{x}/r``. The
integrand is smooth and periodic, so a Gauss-Legendre rule converges fast, and the
whole expression is homogeneous of degree ``-1`` in ``\underline{x}``.

The Green *operator* takes two more derivatives of this. They are obtained by
differentiating the quadrature itself with forward-mode AD rather than by
differentiating the line integral by hand — exact, and reusing one verified
expression instead of introducing a second. See
[`green_function_aniso`](@ref) and [`green_operator_aniso`](@ref); the dispatcher
[`green_operator`](@ref) keeps the closed form whenever the reference is isotropic,
which matters because the anisotropic route is some three orders of magnitude dearer.

In conduction no quadrature is needed at all — the anisotropic scalar Green function is
elementary,

```math
G(\underline{x}) = \frac{1}{4\pi\sqrt{\det\boldsymbol{K}_0}\;
   \sqrt{\underline{x}\cdot\boldsymbol{K}_0^{-1}\cdot\underline{x}}} ,
```

and its Hessian is written out directly.

Two consequences worth keeping in mind when reading an anisotropic result:

* the **closed forms of the previous sections no longer apply**, not even for a ball
  pair. Their exactness rested on the isotropic Green function being biharmonic, which
  a general anisotropic one is not, so the series does not terminate and every pair
  goes through the truncated multipole expansion;
* the **isotropic part no longer vanishes**. ``T_{iijj} = T_{ijij} = 0`` is a property
  of the isotropic kernel, so with an anisotropic reference the bulk response *does*
  see the spatial arrangement. The statement "a cubic array keeps the Mori-Tanaka bulk
  modulus exactly" holds for an isotropic matrix only.

The one case still open is **plane-strain elasticity with an anisotropic reference**,
whose Green function needs the Stroh formalism rather than the Barnett integral;
[`green_operator`](@ref) raises rather than returning an isotropic approximation.

### Why this matters for multiscale chaining

This is not a corner case. A cluster or equivalent-inclusion estimate on a cubic array
*is* anisotropic — its two shear constants differ — so using one N-body result as the
reference medium of another scale requires exactly this kernel. Chaining the two
schemes across scales was impossible without it; see
[the manual](@ref man-assemblies) and `scripts/92`.

## Periodic images

Under a periodic boundary treatment each source carries a family of images and the
interaction becomes a lattice sum, truncated to a cluster of radius ``R_c``:

```math
\bar{\mathbb{T}}^{ab} = \sum_{\|\underline{r}_{ab} + \underline{n}_\ell L\| \le R_c}
   \mathbb{T}^{ab}\big(\underline{r}_{ab} + \underline{n}_\ell L\big),
\qquad \underline{n}_\ell \in \mathbb{Z}^d .
```

Summed over all of ``\mathbb{Z}^d`` the series is only conditionally convergent and the
summation order has to be prescribed — the difficulty
[Brisard et al. 2023](@cite brisard2023) meet around their Eq. (28), where a naive
real-space lattice sum needs a heuristic correction. The cluster truncation above does
not have that problem, and Appendix B of
[Molinari & El Mouden 1996](@cite molinari1996) says why: the kernel integrates to zero
over the exterior of a sphere centered on the receiver,

```math
\int_{\|\underline{x}-\underline{x}_a\| > R_c}
   \mathbb{G}^0(\underline{x}-\underline{x}_a)\,\mathrm{d}V = 0 ,
```

so the neglected images contribute a vanishing amount as ``R_c`` grows. Summing over a
**sphere** of images rather than a box is therefore not a detail but the reason the
truncation is legitimate.

## API

See [API — Interactions](@ref api-interactions).
