# [Two-inclusion interaction tensors](@id th-interaction)

!!! info "Before this page"
    [Concentration and contribution tensors](@ref th-concentration), whose
    one-site picture this page extends to a pair of inclusions, and the
    [sign of the Green operator](@ref th-notation-green-sign) fixed in the
    Conventions, which this page follows.

Every one-site scheme of the package needs one object: the Hill tensor ``\mathbb{P}``
of a single inclusion in a reference medium. The two N-body schemes need one more —
the tensor measuring the field one inclusion induces in another. This page defines it,
gives the closed forms, and states why they are exact.

Notation follows the [Nomenclature](@ref nomenclature): underline for vectors, bold
for order 2, blackboard bold for order 4. Two symbols are introduced here.

| Symbol | Object | Elasticity | Conduction |
| :-- | :-- | :-- | :-- |
| ``\mathbb{G}^0`` / ``\boldsymbol{G}^0`` | Green operator of the reference | order 4 | order 2 |
| ``\mathbb{T}^{ab}`` / ``\boldsymbol{T}^{ab}`` | interaction tensor of two inclusions | order 4 | order 2 |

[molinari1996](@citet) write the interaction tensor
``\Gamma^{IJ}`` and [brisard2014](@citet) write it ``T^{kl}_{ab}``;
the letter ``\mathbb{T}`` is kept here because a Greek capital carries no order in this
typeface convention.

Everything below is written once and read twice, in elasticity and in transport, using
the dictionary of
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q):
``\boldsymbol{\varepsilon} \leftrightarrow \underline{\operatorname{grad}}\,T``,
``\mathbb{C} \leftrightarrow \boldsymbol{K}``,
``\boldsymbol{\sigma} \equiv -\underline{q}``. Order-2 objects keep the bold typeface
(``\boldsymbol{T}^{ab}``, ``\boldsymbol{G}^0``, ``\boldsymbol{P}``); no sign changes
between the two columns.

## 1. The Lippmann-Schwinger equation

With a homogeneous reference ``\mathbb{C}_0`` and the polarization
``\boldsymbol{\tau} = (\mathbb{C}-\mathbb{C}_0):\boldsymbol{\varepsilon}``, the local
problem is equivalent to the integral equation of
[zeller1973](@cite), written here in the sign convention of
[brisard2023](@cite):

```math
\boldsymbol{\varepsilon}(\underline{x}) = \boldsymbol{E}
  - \int_{\mathbb{R}^d} \mathbb{G}^0(\underline{x}-\underline{y})
    : \boldsymbol{\tau}(\underline{y})\, \mathrm{d}V_{\underline{y}} ,
```

whose kernel is the **Green operator** of the reference medium, built from the Green
function ``\boldsymbol{G}`` by

```math
\mathbb{G}^0_{ijkl}(\underline{x})
  = -\Big[\frac{\partial^2 G_{ik}}
              {\partial x_j\, \partial x_l}(\underline{x})\Big]_{(ij)(kl)} ,
```

the brackets denoting symmetrization on ``(i,j)`` and on ``(k,l)``. In conduction the
same object is minus the Hessian of the scalar Green function ``G``,

```math
\boldsymbol{G}^0_{ij}(\underline{x})
  = -\frac{\partial^2 G}{\partial x_i\, \partial x_j}(\underline{x}) .
```

The leading minus is what makes this convention the coherent one. As a distribution the
conduction kernel splits into the regular part above and a Dirac self term,

```math
\boldsymbol{G}^0(\underline{x})
  = \frac{\boldsymbol{1} - 3\,\underline{n}\otimes\underline{n}}
         {4\pi\sigma_0 r^3}
  + \frac{\boldsymbol{1}}{3\sigma_0}\,\delta(\underline{x}) ,
```

and the transform of the whole is

```math
\hat{\boldsymbol{G}}^0(\underline{k})
  = \frac{\underline{k}\otimes\underline{k}}{\sigma_0\,\|\underline{k}\|^2}
\qquad (\underline{k} \ne \underline{0}) ,
```

which is the convention of [brisard2023](@citet): a **positive
semi-definite** operator. The regular part averages to zero over any ball centered on
the source, so the whole interior average is the Dirac term
``\boldsymbol{1}/(3\sigma_0)`` — which is exactly ``\boldsymbol{P}`` of a sphere.
*Plus* the Hill tensor, positive definite as ``\mathbb{P}`` is: the identity tying this
page to the rest of the package. (The functions of the package evaluate the regular
part; the Dirac term is what [`self_interaction_tensor`](@ref) carries.)

## 2. Definition

For two inclusions ``\Omega_a`` (receiver) and ``\Omega_b`` (source) whose centers are
separated by ``\underline{r}``,

```math
\mathbb{T}^{ab}(\underline{r}) = \frac{1}{|\Omega_a|}
   \int_{\Omega_a}\!\int_{\Omega_b}
     \mathbb{G}^0(\underline{x}-\underline{y})\,
     \mathrm{d}V_{\underline{y}}\, \mathrm{d}V_{\underline{x}} ,
```

so that ``-\mathbb{T}^{ab}:\boldsymbol{\tau}_b`` is the **average field induced in
``a``** by a uniform polarization ``\boldsymbol{\tau}_b`` carried by ``b``. Over a
single inclusion the same integral gives

```math
\boxed{\;\mathbb{T}^{aa} = +\,\mathbb{P}_a\;}
```

so the one-inclusion case is exactly the
``\boldsymbol{\varepsilon} = -\mathbb{P}:\boldsymbol{\tau}`` of
[the Eshelby problem](@ref th-eshelby-problem). See
[`self_interaction_tensor`](@ref) and the Mori-Tanaka limit on the
[cluster-model page](@ref th-cluster).

!!! warning "Two sign conventions exist"
    The package follows [brisard2023](@citet):
    the Green operator maps a polarization onto **minus** the induced field, its
    Fourier symbol ``\sigma_0^{-1}\underline{k}\otimes\underline{k}/\|\underline{k}\|^2``
    is positive, and ``\mathbb{T}^{aa} = +\mathbb{P}``, coherent with the Hill tensor.

    [molinari1996](@citet) and
    [berveiller1987](@citet) write instead
    ``\boldsymbol{\varepsilon}^I = \boldsymbol{\varepsilon}^0 +
    \sum_J \Gamma^{IJ}:\delta\mathbb{C}^J:\boldsymbol{\varepsilon}^J``, so their
    ``\Gamma^{IJ}`` is the **opposite** of ``\mathbb{T}^{IJ}`` and their self term is
    ``\Gamma^{II} = -\mathbb{P}``. Any formula transcribed from them — their
    Appendix A component table in particular — must be flipped before it is compared
    with anything on this page. The closed form below is *their table with the sign
    already flipped*, which is why its ``\kappa`` is positive where theirs is negative.

## 3. Exact closed forms for balls and disks

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
\kappa = \frac{b^3}{12\,R^3\,\mu\,(1-\nu)} ,
\qquad
\rho^2 = \frac{a^2+b^2}{R^2} .
```

— note the sign of ``\kappa``: [molinari1996](@citet) write
``-b^3/(12R^3\mu(1-\nu))``, and the flip to this package's convention is applied here,
once. In the frame whose third axis is ``\underline{n}``,
after their Appendix A and [berveiller1987](@citet):

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
Walpole coefficients rather than by an 81-component array — the storage described in
[Conventions](@ref th-notation-bases):

```math
\begin{gathered}
T_1 = T_{3333},\quad
T_2 = T_{1111}+T_{1122},\quad
T_3 = \sqrt{2}\,T_{1133},
\\
T_5 = T_{1111}-T_{1122},\quad
T_6 = 2\,T_{1313},
\end{gathered}
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
  \big(\boldsymbol{1} - 3\,\underline{n}\otimes\underline{n}\big)
\quad (3\text{D}),
\qquad
\boldsymbol{T}^{ab} = \frac{b^2}{2\,\sigma_0 R^2}
  \big(\boldsymbol{1} - 2\,\underline{n}\otimes\underline{n}\big)
\quad (2\text{D}).
```

The two-dimensional form is taken **literally** from
[brisard2023](@cite) — same sign, same prefactor — and the test
suite checks it component by component. It is the sharpest available statement that
the package and its normative reference share one convention.

### Elasticity, plane strain

```math
\mathbb{T}^{ab} = \pi b^2\left[\mathbb{G}^0(\underline{r})
   + \frac{a^2+b^2}{8}\,\Delta\mathbb{G}^0(\underline{r})\right],
```

```math
\Delta\mathbb{G}^0(\underline{r}) = \frac{-24}{8\pi\mu(1-\nu)R^4}
  \Big[\mathbb{K}_2
    - (2\boldsymbol{N}-\boldsymbol{1})\otimes(2\boldsymbol{N}-\boldsymbol{1})\Big],
```

``\mathbb{K}_2`` being the plane deviatoric projector.
``\Delta\mathbb{G}^0`` does not depend on the reference Poisson ratio, and that both
terms of the bracket above carry the same overall convention: ``\mathbb{G}^0`` and
``\Delta\mathbb{G}^0`` are negated together, never one without the other.

## 4. The vanishing isotropic part

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

## 5. General ellipsoids

Beyond balls the series does not terminate, and
[brisard2014](@cite), §4.2, expand the regular part of the kernel
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
definition directly by a product rule — geometry-agnostic, far slower, and the reference
the closed forms are validated against.

## 6. Anisotropic reference media

Everything above assumes an isotropic reference, whose Green operator is a closed
form. That is not a restriction of the method, only of the kernel, and it is lifted by
the Barnett line integral [barnettPSSB1972, willis1975](@cite): for an arbitrary anisotropic
stiffness of space the displacement Green function is

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

In space, the Green *operator* takes two more derivatives of this. They are obtained by
differentiating the quadrature itself with forward-mode AD rather than by
differentiating the line integral by hand — exact, and reusing one verified
expression instead of introducing a second. See
[`green_function_aniso`](@ref) and [`green_operator_aniso`](@ref); the dispatcher
[`green_operator`](@ref) keeps the closed form whenever the reference is isotropic,
which matters because the anisotropic approach is some three orders of magnitude dearer.

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

In the plane the contour normal to ``\underline{x}`` reduces to two points, and
the Green function grows like ``\ln r`` instead of decaying: the anisotropic
plane has a formula of its own, derived in the next section.

### [The Green operator of an anisotropic plane](@id th-plane-green)

This section derives the Green operator of a stiffness of the plane of any
anisotropy, the kernel [`green_operator_aniso`](@ref) evaluates for a tensor of
order 4 of dimension 2 and the interaction tensors of a cell of the plane are
built on ([Problems of the plane](@ref th-plane-problems)).

**The Green function.** ``G_{im}(\underline{x})`` is the displacement along
``\underline{e}_i`` at ``\underline{x}`` under a unit force along
``\underline{e}_m`` at the origin:

```math
C_{aijb}\,\partial_a\partial_b\,G_{jm}(\underline{x})+\delta_{im}\,\delta(\underline{x})=0 .
```

With the transform ``\hat f(\underline{k})=\int f(\underline{x})\,\mathrm{e}^{-\mathrm{i}\,\underline{k}\cdot\underline{x}}\,\mathrm{d}S``,
this reads ``\boldsymbol{K}(\underline{k})\cdot\hat{\boldsymbol{G}}(\underline{k})=\boldsymbol{1}``,
with the acoustic tensor ``K_{ij}(\underline{k})=k_a\,C_{aijb}\,k_b``,
positive definite for every real ``\underline{k}\neq\underline{0}`` and
homogeneous of degree 2. In polar coordinates,
``\underline{k}=\rho\,\underline{\xi}(\varphi)`` with the unit vector
``\underline{\xi}(\varphi)=\cos\varphi\,\underline{e}_1+\sin\varphi\,\underline{e}_2``,
``\boldsymbol{K}(\underline{k})^{-1}=\boldsymbol{K}(\underline{\xi})^{-1}/\rho^2``
and the inverse transform is

```math
\boldsymbol{G}(\underline{x})
=\frac{1}{4\pi^2}\int_0^{2\pi}\boldsymbol{K}(\underline{\xi})^{-1}
\int_0^{\infty}\frac{\mathrm{e}^{\mathrm{i}\rho\,\underline{\xi}\cdot\underline{x}}}{\rho}\,\mathrm{d}\rho\,\mathrm{d}\varphi .
```

The radial integral diverges at ``\rho\to0``, as ``\boldsymbol{G}`` grows at
infinity; as a distribution it equals
``-\ln|\underline{\xi}\cdot\underline{x}|+\tfrac{\mathrm{i}\pi}{2}\,\mathrm{sgn}(\underline{\xi}\cdot\underline{x})``
up to a constant. The imaginary part is odd in ``\underline{\xi}`` and
``\boldsymbol{K}^{-1}`` is even, so that it vanishes over a full turn:

```math
\boldsymbol{G}(\underline{x})
=-\frac{1}{4\pi^2}\int_0^{2\pi}\boldsymbol{K}(\underline{\xi})^{-1}\,
\ln|\underline{\xi}\cdot\underline{x}|\,\mathrm{d}\varphi+\boldsymbol{G}_\infty ,
```

``\boldsymbol{G}_\infty`` a constant tensor that no gradient sees. For one
component and ``K=\sigma_0``, the integral of ``\ln|\cos\varphi|`` being a
constant, it is ``-\ln r/(2\pi\sigma_0)``: the Green function of conduction
in the plane.

**A finite part.** The gradients of ``\ln|\underline{\xi}\cdot\underline{x}|``
are ``\xi_l/(\underline{\xi}\cdot\underline{x})`` and
``-\xi_k\,\xi_l/(\underline{\xi}\cdot\underline{x})^2``. The first has a simple
pole at the two directions ``\underline{\xi}\perp\underline{x}``, and the first
gradient of ``\boldsymbol{G}`` is a Cauchy principal value. Its derivative is
the Hadamard finite part of the integral of the derivative:

```math
\partial_k\partial_l\,G_{ij}(\underline{x})
=\frac{1}{4\pi^2}\;\mathrm{f.p.}\!\int_0^{2\pi}
\frac{h_{ijkl}(\varphi)}{(\underline{\xi}\cdot\underline{x})^2}\,\mathrm{d}\varphi ,
\qquad
h_{ijkl}=K^{-1}_{ij}(\underline{\xi})\,\xi_k\,\xi_l ,
```

an integral with a double pole, on which no quadrature rule can be applied as
it stands.

**A regular integral.** ``h`` and ``(\underline{\xi}\cdot\underline{x})^2``
are even in ``\underline{\xi}``: the full turn is twice a half turn, taken
centered on the pole. With ``r=\|\underline{x}\|``,
``\underline{n}=\underline{x}/r``, ``\underline{t}`` the unit vector at
``+\pi/2`` from ``\underline{n}``, and
``\underline{\xi}(\psi)=\cos\psi\,\underline{t}-\sin\psi\,\underline{n}``,
``\underline{\xi}\cdot\underline{x}=-r\sin\psi`` and

```math
\partial_k\partial_l\,G_{ij}(\underline{x})
=\frac{1}{2\pi^2r^2}\;\mathrm{f.p.}\!\int_{-\pi/2}^{\pi/2}\frac{h_{ijkl}(\psi)}{\sin^2\psi}\,\mathrm{d}\psi .
```

Since ``1/\sin^2\psi=-(\cot\psi)'``, an integration by parts outside
``(-\varepsilon,\varepsilon)``, where ``\cot(\pm\pi/2)=0`` removes the ends,
gives

```math
\int_{\varepsilon<|\psi|<\pi/2}\frac{h}{\sin^2\psi}\,\mathrm{d}\psi
=\bigl(h(\varepsilon)+h(-\varepsilon)\bigr)\cot\varepsilon
+\int_{\varepsilon<|\psi|<\pi/2}h'(\psi)\cot\psi\,\mathrm{d}\psi .
```

The first term is ``2h(0)/\varepsilon+O(\varepsilon)``: its divergent part is
what the finite part discards, and the rest vanishes. In the second,
``\cot`` is odd, and folding ``\psi<0`` onto ``\psi>0`` leaves a regular
integral over a quarter turn:

```math
\boxed{\;
\partial_k\partial_l\,G_{ij}(\underline{x})
=\frac{1}{2\pi^2r^2}\int_0^{\pi/2}\bigl[h'_{ijkl}(\psi)-h'_{ijkl}(-\psi)\bigr]\cot\psi\,\mathrm{d}\psi
\;}
```

The bracket is odd in ``\psi``, of order ``2h''(0)\,\psi`` at the origin, so
that the integrand is bounded there; it is analytic on ``[0,\pi/2]``, since
``\boldsymbol{K}`` is positive definite on every real direction. The operator
``\mathbb{G}^0`` follows by the symmetrization of section 1, and is
homogeneous of degree ``-2`` in ``\underline{x}``.

**The derivative of the integrand.** With
``\underline{\eta}=\mathrm{d}\underline{\xi}/\mathrm{d}\psi=-\sin\psi\,\underline{t}-\cos\psi\,\underline{n}``
and the derivative of an inverse,
``(\boldsymbol{K}^{-1})'=-\boldsymbol{K}^{-1}\cdot\boldsymbol{K}'\cdot\boldsymbol{K}^{-1}``,

```math
K'_{ij}=(\eta_a\,\xi_b+\xi_a\,\eta_b)\,C_{aijb},
\qquad
h'_{ijkl}=-\bigl(\boldsymbol{K}^{-1}\cdot\boldsymbol{K}'\cdot\boldsymbol{K}^{-1}\bigr)_{ij}\,\xi_k\,\xi_l
+K^{-1}_{ij}\,(\eta_k\,\xi_l+\xi_k\,\eta_l) ,
```

in closed form: nothing is differentiated numerically.

**The scalar case.** For one component, ``K=\sigma_0``, and
``\underline{x}=r\,\underline{e}_1``: ``\underline{n}=\underline{e}_1``,
``\underline{t}=\underline{e}_2``, ``h_{11}=\sin^2\psi/\sigma_0``, and the
integrand is ``2\sin2\psi\cot\psi/\sigma_0=4\cos^2\psi/\sigma_0``, of integral
``\pi/\sigma_0``. Hence ``\partial_1\partial_1G=1/(2\pi\sigma_0r^2)``, the
second derivative along the radius of ``-\ln r/(2\pi\sigma_0)``.

**No characteristic root.** The closed forms of the anisotropic plane write
``\boldsymbol{G}`` with the roots ``p`` of
``\det\boldsymbol{K}(\underline{e}_1+p\,\underline{e}_2)=0``, two pairs of
complex conjugates. In an isotropic plane of Lamé constants ``\lambda`` and
``\mu``, ``\det\boldsymbol{K}(\underline{e}_1+p\,\underline{e}_2)=\mu\,(\lambda+2\mu)\,(1+p^2)^2``:
the roots are double, and those forms are taken to a limit there. The integral
above inverts ``\boldsymbol{K}`` on real directions only: no root is computed,
and the isotropic stiffness is an ordinary case. Typed as a general tensor, it
gives the closed form of [`green_operator_iso`](@ref), to rounding:

```@example thinteraction
using MeanFieldHomogenization, TensND, LinearAlgebra, Printf
x = [0.8, -0.3]
C = iso_stiffness(Val(2), 5.0, 2.0)                  # bulk and shear moduli of the plane
Cg = Tens(canonical_components(C))                    # the same stiffness, its class forgotten
Γ₀ = get_array(green_operator_iso(C, x))
norm(get_array(green_operator_aniso(Cg, x)) - Γ₀) / norm(Γ₀)
```

**The quadrature.** A Gauss–Legendre rule of ``N`` nodes on
``(0,\pi/2)`` (the keyword `nodes`, 32 by default) integrates an analytic
function with an error that decreases geometrically with ``N``, at a rate set
by the distance to the interval of the nearest complex ``\psi`` where
``\det\boldsymbol{K}(\underline{\xi}(\psi))=0``. A strong anisotropy brings it
closer, and asks for more nodes. On an orthotropic stiffness of contrast 40/6
between its axes, against 96 nodes:

```@example thinteraction
C₀ = TensOrtho(40.0, 6.0, 2.5, 3.0, Basis(0.37))       # C₁₁₁₁, C₂₂₂₂, C₁₁₂₂, C₁₂₁₂, frame at 0.37 rad
Γ(nodes) = get_array(green_operator_aniso(C₀, x; nodes))
Γ₉₆ = Γ(96)
for nodes in (8, 16, 32, 48)
    @printf "%2d nodes   relative error %.1e\n" nodes norm(Γ(nodes) - Γ₉₆) / norm(Γ₉₆)
end
```

The interaction tensors of section 5 take this operator, and its two
derivatives along the line of centers by automatic differentiation of the same
quadrature, in place of the isotropic closed form.

### Consequence for multiscale chaining

This is not a corner case. A cluster or equivalent-inclusion estimate on a cubic array
*is* anisotropic — its two shear constants differ — so using one N-body result as the
reference medium of another scale requires exactly this kernel. Chaining the two
schemes across scales was impossible without it; see
[the manual](@ref man-assemblies) and `scripts/92`.

## 7. Periodic images

Under a periodic boundary treatment each source carries a family of images and the
interaction becomes a lattice sum, truncated to a cluster of radius ``R_c``:

```math
\bar{\mathbb{T}}^{ab} = \sum_{\|\underline{r}_{ab} + \underline{n}_\ell L\| \le R_c}
   \mathbb{T}^{ab}\big(\underline{r}_{ab} + \underline{n}_\ell L\big),
\qquad \underline{n}_\ell \in \mathbb{Z}^d .
```

Summed over all of ``\mathbb{Z}^d`` the series is only conditionally convergent and the
summation order has to be prescribed — the difficulty
[brisard2023](@citet) meet, where a naive
real-space lattice sum needs a heuristic correction. The cluster truncation above does
not have that problem, and Appendix B of
[molinari1996](@citet) says why: the kernel integrates to zero
over the exterior of a sphere centered on the receiver,

```math
\int_{\|\underline{x}-\underline{x}_a\| > R_c}
   \mathbb{G}^0(\underline{x}-\underline{x}_a)\,\mathrm{d}V = 0 ,
```

so the neglected images contribute a vanishing amount as ``R_c`` grows. Summing over a
**sphere** of images rather than a box is therefore not a detail but the reason the
truncation is legitimate.

## Where to go next

The next chapter returns to a single inclusion, in its flat limit:
[Crack opening displacement and compliance](@ref th-cod-tensors) describes a
crack by how much it opens rather than by a volume fraction and a stiffness.
The two schemes built on ``\mathbb{T}^{ab}`` come at
the end of the chapter on schemes: [The cluster model](@ref th-cluster) solves
for the mean strain of every inclusion inside a cluster of neighbors, and
[The equivalent inclusion method](@ref th-eim) poses the same problem as a
Galerkin discretization that also yields a rigorous bound. The functions
evaluating ``\mathbb{T}^{ab}`` are documented in
[API — Interactions](@ref api-interactions).
