# [Crack opening displacement and compliance](@id th-cod-tensors)

!!! info "Before this page"
    [The Eshelby inclusion problem](@ref th-eshelby-problem) §2, where the
    tensor ``\mathbb{Q}``, which degenerates in a controlled way as the
    inclusion flattens, is introduced, and
    [Concentration and contribution tensors](@ref th-concentration) §3 for the
    compliance contribution ``\mathbb{H}`` of a crack and its density.

A crack has no volume, so it cannot be described by a volume fraction and a
stiffness. It is described instead by **how much it opens** under a given remote
stress. That is the crack opening displacement tensor ``\boldsymbol{B}``, and
everything else on this page follows from it.

```math
\boldsymbol{B}
\;\longrightarrow\;
\mathbb{H}
\;\longrightarrow\;
\Delta\mathbb{S}
\qquad\text{and}\qquad
\boldsymbol{B}
\;\longrightarrow\;
\underline{K},\ \underline{N}
```

Read left to right: the **COD tensor** ``\boldsymbol{B}`` (order 2) gives the
**compliance contribution** ``\mathbb{H}`` (order 4), which gives the dilute
correction ``\Delta\mathbb{S}`` to the effective compliance; independently,
``\boldsymbol{B}`` gives the stress and displacement intensity factors
``\underline{K}``, ``\underline{N}`` at the crack front.

`MeanFieldHomogenization` computes ``\boldsymbol{B}`` first — the intensity factors need
it, and rebuilding ``\mathbb{H}`` from it avoids inverting a rank-deficient
order-4 tensor. Its normalization is not unique in the literature, hence the
[Conventions](@ref th-cod-conventions) section below.

## 1. Geometry

### Elliptic and ribbon cracks

A crack ``\mathcal{I}`` is a plane region of unit normal ``\underline{n}``, and
two shapes of it are considered here: the ellipse ``\mathcal{E}`` and the
ribbon ``\mathcal{R}``, a strip of constant width extending without bound in one
direction. Following [barthelemyIJSS2026](@citet), both are described by one
singular order-2 tensor ``\boldsymbol{A}``, of rank 2 for the ellipse and 1 for
the ribbon, whose kernel contains ``\underline{n}``:

```math
\underline{x}\in\mathcal{I}
\iff
\underline{x}\cdot\underline{n}=0
\quad\text{and}\quad
\underline{x}\cdot\bigl(\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\bigr)^{+}\!\cdot\underline{x}\le 1,
```

where ``(\cdot)^{+}`` denotes the Moore–Penrose pseudo-inverse, the inverse
itself being undefined for a singular tensor. Only
``\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}`` enters this definition, so that
``\boldsymbol{A}`` may be taken symmetric without loss of generality. With
``(\underline{\ell},\underline{m},\underline{n})`` an orthonormal frame, the
ellipse then corresponds to

```math
\boldsymbol{A}
= a\,\underline{\ell}\otimes\underline{\ell}
+ b\,\underline{m}\otimes\underline{m},
\qquad
\bigl(\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\bigr)^{+}
= \frac{\underline{\ell}\otimes\underline{\ell}}{a^{2}}
+ \frac{\underline{m}\otimes\underline{m}}{b^{2}},
\qquad a\ge b>0,
```

and the second condition reads
``(\underline{x}\cdot\underline{\ell})^{2}/a^{2}
+(\underline{x}\cdot\underline{m})^{2}/b^{2}\le 1``: ``\underline{\ell}`` and
``\underline{m}`` are directed along the major and minor axes, ``a`` and ``b``
are the corresponding semi-axes, and the in-plane aspect ratio is
``\eta = b/a\le 1``.

```@setup codgeom
using MeanFieldHomogenization
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example codgeom
crack_figure(EllipticCrack(3.0, 1.5); uid = "th-cod-elliptic",
    title = "Elliptic crack ℰ, a = 3, b = 1.5 (η = 0.5)")
```

The ribbon is not the ellipse with ``a = 0``, which would be the segment of
length ``2b`` carried by ``\underline{m}``, but its limit as ``a`` grows without
bound while ``\underline{\ell}`` is held fixed. This limit is taken on the
pseudo-inverse, the only part of the definition that remains finite, and it
leaves

```math
\lim_{a\to+\infty}\bigl(\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\bigr)^{+}
= \frac{\underline{m}\otimes\underline{m}}{b^{2}},
\qquad\text{hence}\qquad
\underline{x}\in\mathcal{R}
\iff
\underline{x}\cdot\underline{n}=0
\quad\text{and}\quad
\lvert\underline{x}\cdot\underline{m}\rvert\le b .
```

``\mathcal{R}`` is thus invariant by translation along ``\underline{\ell}``, of
width ``2b``, and its in-plane aspect ratio ``\eta`` vanishes. Since the limit
retains ``\underline{m}`` alone, the tensor no longer says how
``\underline{\ell}`` and ``\underline{n}`` are placed in the plane orthogonal to
``\underline{m}``: one of them must be given in addition, which is why
[`RibbonCrack`](@ref) stores the whole frame along with ``b``, and why a ribbon
is better handled as the end of a family of ellipses than as a shape of its own.

```@example codgeom
crack_figure(RibbonCrack(1.5); uid = "th-cod-ribbon",
    title = "Ribbon crack ℛ, b = 1.5, unbounded along <u><i>ℓ</i></u>")
```

Both shapes are available as [`EllipticCrack`](@ref)`(a, b)`, with
[`PennyCrack`](@ref)`(a)` for the circle ``\eta = 1``, and
[`RibbonCrack`](@ref)`(b)`, oriented by Euler angles or by a given frame.

### The crack as a degenerate ellipsoid

The compliance of a crack and its opening under load are obtained from the
Eshelby problem of an ellipsoid ([Conventions](@ref th-notation-ellipsoid)),
the crack being regarded as an ellipsoid of vanishing thickness
[laws1977, kachanov1992, barthelemyIJSS2009](@cite). In the notation of
[barthelemyMMS2023](@citet) and [barthelemyIJES2021](@citet), the ellipse
``\mathcal{E}`` is the limit of the ellipsoid

```math
\underline{x}\cdot\bigl(\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\bigr)^{-1}\!\cdot\underline{x}\le 1,
\qquad
\boldsymbol{A}
= a\,\bigl(\underline{\ell}\otimes\underline{\ell}
+ \eta\,\underline{m}\otimes\underline{m}
+ \omega\,\underline{n}\otimes\underline{n}\bigr),
\qquad
\omega = \frac{c}{a},
```

whose third semi-axis ``c``, along ``\underline{n}``, is sent to zero, so that
``\lim_{\omega\to 0}\boldsymbol{A}
= a\,\underline{\ell}\otimes\underline{\ell}+b\,\underline{m}\otimes\underline{m}``
is the rank-2 tensor of ``\mathcal{E}`` above. The slider of the figure lets
``\omega`` decrease from 0.4 to 0.02:

```@example codgeom
flattening_figure(EllipticCrack(3.0, 1.5); uid = "th-cod-flat-ellipsoid",
    title = "Ellipsoid of semi-axes a, ηa, ωa around ℰ, a = 3, η = 0.5")
```

The ribbon requires both limits. Letting ``a\to+\infty`` first turns the
ellipsoid into an elliptic cylinder of axis ``\underline{\ell}`` whose cross
section has the half-axes ``b`` along ``\underline{m}`` and ``c`` along
``\underline{n}``; flattening that cross section then yields ``\mathcal{R}``
[barthelemyIJES2021](@cite). As ``a`` is no longer finite, the flatness is
measured on the cross section,

```math
\rho = \frac{c}{b}\to 0 .
```

```@example codgeom
flattening_figure(RibbonCrack(1.5); uid = "th-cod-flat-cylinder",
    title = "Elliptic cylinder of half-axes b, ρb around ℛ, b = 1.5")
```

For the ellipse the same ratio reads ``\rho = c/b = \omega/\eta``, and it is by
``\rho`` that the limits defining the compliance of either crack are normalized
in [§3](@ref th-cod-compliance). On the code side, `Ellipsoid(a, b, 0)` and
`Ellipsoid(Inf, b, 0)` return an [`EllipticCrack`](@ref) and a
[`RibbonCrack`](@ref) respectively
([degenerate limits](@ref man-ellipsoidal-inclusions)).

## 2. The COD tensor ``\boldsymbol{B}``

Under a remote stress ``\boldsymbol{\Sigma}``, the two crack faces separate by
the displacement jump ``[\![\underline{u}]\!]``. By linearity and the
superposition principle, that jump depends on the loading only through the
resolved traction ``\boldsymbol{\Sigma}\cdot\underline{n}`` on the crack plane.
Its **average over the crack surface** ``\mathcal{I}`` defines
``\boldsymbol{B}``:

```math
\boxed{\;
\frac{\bigl\langle [\![\underline{u}]\!] \bigr\rangle_{\mathcal{I}}}{b}
= \boldsymbol{B}\cdot\boldsymbol{\Sigma}\cdot\underline{n}
\;}
```

The normalization is by the **in-plane half-width ``b``** — the minor semi-axis
of the ellipse, the half-width of the ribbon. This makes ``\boldsymbol{B}``
**size-independent**: it depends on the crack *shape* (through ``\eta``) and on
its orientation, never on how big it is. This is the convention of
[barthelemyIJSS2026](@cite), following [kachanov1992, kachanov1993](@citet), and it is the one `MeanFieldHomogenization` implements
([`cod_tensor`](@ref), alias [`B_tensor`](@ref)).

### The shape coefficient ``\chi``

The opening profile is an ellipsoidal cap, so the average jump is a fixed
fraction of the **maximum** jump ``\underline{\beta}`` at the crack center:

```math
\begin{aligned}
\bigl\langle [\![\underline{u}]\!] \bigr\rangle_{\mathcal{I}} &= \chi\,\underline{\beta},
\\
\chi^{\mathcal{E}}
&= \frac{1}{\pi a b}\!\!\int_{\frac{x^2}{a^2}+\frac{y^2}{b^2}\le 1}\!\!
  \sqrt{1-\frac{x^2}{a^2}-\frac{y^2}{b^2}}\;\mathrm{d}x\,\mathrm{d}y
= \frac{2}{3},
\\
\chi^{\mathcal{R}}
&= \frac{1}{2b}\!\int_{-b}^{b}\!\sqrt{1-\frac{y^2}{b^2}}\;\mathrm{d}y
= \frac{\pi}{4}.
\end{aligned}
```

``\chi`` is worth introducing explicitly, because **every numerical factor on
this page comes from it**. It is the one quantity that distinguishes the
elliptic geometry from the ribbon one.

### The 3-D → 2-D limit: a trap

A ribbon is the limiting shape of an ellipse as ``\eta\to 0``, so one expects
``\boldsymbol{B}^{\mathcal{R}}`` to be ``\lim_{\eta\to 0}\boldsymbol{B}^{\mathcal{E}}``.
**It is not** [barthelemyIJSS2026](@cite):

```math
\boxed{\;
\boldsymbol{B}^{\mathcal{R}}(\underline{m},\underline{n})
= \frac{\chi^{\mathcal{R}}}{\chi^{\mathcal{E}}}\,
  \lim_{\eta\to 0}\boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
= \frac{3\pi}{8}\,
  \lim_{\eta\to 0}\boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta).
\;}
```

The reason is exactly ``\chi``. The *geometry* does converge, and so does the
pointwise opening profile; what changes is the relation between the **average**
opening and the maximum one, because averaging over a shrinking ellipse is not
the same operation as averaging across the width of a strip
(``2/3 \ne \pi/4``). Since ``\boldsymbol{B}`` is defined from the average, it
picks up the ratio ``\chi^{\mathcal{R}}/\chi^{\mathcal{E}} = 3\pi/8``.

This matters in practice: the intensity factors of an elliptic crack are
computed from the ribbon tensor of the *tangent* ribbon at each front point
(see *Intensity factors at the crack front* below), so both objects appear in
the same formula and must not be confused.

## [3. From ``\boldsymbol{B}`` to the compliance ``\mathbb{H}``](@id th-cod-compliance)

The compliance contribution ``\mathbb{H}`` of a crack is defined from the
ellipsoid, or the elliptic cylinder, of §1 as it flattens; ``\boldsymbol{B}`` is
defined from the opening of the crack itself. This section proves that, with
``\boldsymbol{B}`` normalized by the half-width ``b`` as in §2 and the flatness
``\rho = c/b`` of §1,

```math
\boxed{\;
\mathbb{H} \;=\; \lim_{\rho\to 0}\ \rho\,\mathbb{Q}^{-1}
\;=\; k\;\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n},
\qquad
k = \tfrac{3}{4}\ \text{(ellipse)},\quad k = \tfrac{2}{\pi}\ \text{(ribbon)},
\;}
```

with ``\mathbb{Q} = \mathbb{C}-\mathbb{C}:\mathbb{P}:\mathbb{C}`` the
[second Hill tensor](@ref th-eshelby-problem). The idea goes back to
[kachanov1992](@citet): the surface of a flat cavity becomes the two lips of the
crack [barthelemyIJSS2009, barthelemyIJES2021](@cite). It is written out here
because both sides of the identity depend on two lengths, the one
``\boldsymbol{B}`` is normalized by and the one the flatness is measured on,
which the sources choose differently ([Conventions](@ref th-cod-conventions)
below).

### Step 1: the strain a cavity adds is carried by its surface

A volume ``\Omega``, of measure ``|\Omega|``, of a linear elastic solid of
stiffness ``\mathbb{C} = \mathbb{S}^{-1}`` contains a cavity ``\Omega_c`` of
volume ``V``, whose surface is free of traction, and ``\underline{\nu}`` is the
unit normal to ``\partial\Omega_c`` pointing out of the cavity. Its outer
boundary carries the tractions ``\boldsymbol{\Sigma}\cdot\underline{n}`` of a
uniform stress ``\boldsymbol{\Sigma}``, and its overall strain is the average of
the boundary displacement:

```math
\boldsymbol{E} = \frac{1}{|\Omega|}\int_{\partial\Omega}\underline{u}\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S .
```

The boundary of the solid ``\Omega\setminus\Omega_c`` is ``\partial\Omega`` and
``\partial\Omega_c``, the latter with the normal ``-\underline{\nu}``, so the
divergence theorem gives

```math
\int_{\partial\Omega}\underline{u}\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S
= \int_{\Omega\setminus\Omega_c}\boldsymbol{\varepsilon}\,\mathrm{d}V
+ \int_{\partial\Omega_c}\underline{u}\stackrel{s}{\otimes}\underline{\nu}\,\mathrm{d}S .
```

In the solid ``\boldsymbol{\varepsilon} = \mathbb{S}:\boldsymbol{\sigma}``, and
the stress integrates to ``|\Omega|\,\boldsymbol{\Sigma}`` over the solid: by
the same theorem, ``\int\boldsymbol{\sigma}\,\mathrm{d}V`` is the first moment
of the tractions on the boundary, which are those of ``\boldsymbol{\Sigma}``
outside and vanish on the cavity. Hence

```math
\boldsymbol{E} = \mathbb{S}:\boldsymbol{\Sigma}
+ \frac{1}{|\Omega|}\int_{\partial\Omega_c}\underline{u}\stackrel{s}{\otimes}\underline{\nu}\,\mathrm{d}S .
\tag{1}
```

The strain added by the cavity is an integral over its surface alone; neither
its shape nor its size has been used.

### Step 2: an ellipsoidal cavity

Let ``\Omega_c`` be the ellipsoid of §1, alone in an infinite matrix, which is
the dilute limit of (1). It is an inhomogeneity of zero stiffness, so by
[Eshelby's theorem](@ref th-eshelby-problem) the displacement of its surface is,
up to a rigid motion, that of the uniform strain

```math
\boldsymbol{\varepsilon}_c
= \bigl(\mathbb{I}-\mathbb{P}:\mathbb{C}\bigr)^{-1}:\mathbb{S}:\boldsymbol{\Sigma}
= \mathbb{Q}^{-1}:\boldsymbol{\Sigma},
\qquad\text{since}\quad
\mathbb{Q} = \mathbb{C}:\bigl(\mathbb{I}-\mathbb{P}:\mathbb{C}\bigr).
```

A rigid motion adds nothing to the integral of (1), and
``\int_{\partial\Omega_c}\underline{x}\otimes\underline{\nu}\,\mathrm{d}S = V\,\boldsymbol{1}``,
so that

```math
\int_{\partial\Omega_c}\underline{u}\stackrel{s}{\otimes}\underline{\nu}\,\mathrm{d}S
= V\,\boldsymbol{\varepsilon}_c = V\,\mathbb{Q}^{-1}:\boldsymbol{\Sigma}.
\tag{2}
```

With (1), a cavity of volume fraction ``f = V/|\Omega|`` adds
``f\,\mathbb{Q}^{-1}`` to the compliance: ``\mathbb{Q}^{-1}`` is the compliance
contribution of a cavity ([contribution tensors](@ref th-concentration)).

### Step 3: the surface collapses onto the lips

Let the third semi-axis ``c = \rho\,b`` tend to zero at fixed ``a`` and ``b``
(for the ribbon, the elliptic cylinder of half-axes ``b`` and ``\rho b``, per
unit length along ``\underline{\ell}``). The volume ``V`` vanishes like
``\rho``, and ``\mathbb{Q}^{-1}`` diverges like ``1/\rho``, but the surface
integral (2) has a finite limit: the upper half of ``\partial\Omega_c`` becomes
the lip of ``\mathcal{I}`` facing ``+\underline{n}``, where
``\underline{\nu}\to\underline{n}`` and ``\underline{u}\to\underline{u}^{+}``,
and the lower half the lip facing ``-\underline{n}``, where
``\underline{\nu}\to-\underline{n}`` and ``\underline{u}\to\underline{u}^{-}``.
With ``S`` the area of ``\mathcal{I}`` and the definition of §2,

```math
\lim_{\rho\to 0}\int_{\partial\Omega_c}\underline{u}\stackrel{s}{\otimes}\underline{\nu}\,\mathrm{d}S
= \int_{\mathcal{I}}[\![\underline{u}]\!]\stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S
= S\,\bigl\langle[\![\underline{u}]\!]\bigr\rangle_{\mathcal{I}}\stackrel{s}{\otimes}\underline{n}
= S\,b\,\bigl(\boldsymbol{B}\cdot\boldsymbol{\Sigma}\cdot\underline{n}\bigr)\stackrel{s}{\otimes}\underline{n}.
\tag{3}
```

The last term is linear in ``\boldsymbol{\Sigma}``. For any symmetric
``\boldsymbol{\Sigma}``,
``(\boldsymbol{B}\cdot\boldsymbol{\Sigma}\cdot\underline{n})\stackrel{s}{\otimes}\underline{n}
= (\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}):\boldsymbol{\Sigma}``,
where the order-4 tensor has the components

```math
\bigl(\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}\bigr)_{ijkl}
= \tfrac{1}{4}\bigl(n_i B_{jk} n_l + n_j B_{ik} n_l + n_i B_{jl} n_k + n_j B_{il} n_k\bigr).
```

Equating (2) and (3) for every ``\boldsymbol{\Sigma}``,

```math
\lim_{\rho\to 0} V\,\mathbb{Q}^{-1}
= S\,b\;\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}.
\tag{4}
```

### Step 4: the compliance contribution

Multiplying (4) by ``\rho/V = c/(b\,V)``,

```math
\mathbb{H} = \lim_{\rho\to 0}\rho\,\mathbb{Q}^{-1}
= \frac{c\,S}{V}\;\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n},
```

and ``cS/V`` is a number set by the shape alone:

| | surface ``S`` | volume ``V`` | ``k = cS/V`` |
| :-- | :-- | :-- | :-- |
| **elliptic** (3-D) | ``\pi a b`` | ``\tfrac{4}{3}\pi a b c`` | ``\tfrac{3}{4}`` |
| **ribbon** (2-D), per unit length | ``2b`` | ``\pi b c`` | ``\tfrac{2}{\pi}`` |

Both factors are **independent of ``\eta``**, which is the point of normalizing
both ``\boldsymbol{B}`` and the flatness by ``b``: one definition covers the 2-D
and the 3-D geometry, and ``\mathbb{H}`` inherits ``\boldsymbol{B}``'s
independence of size.

### Step 5: the dilute correction, and back from ``\mathbb{H}`` to ``\boldsymbol{B}``

``N`` cracks per unit volume (per unit area for ribbons) add, by step 2,
``N V\,\mathbb{Q}^{-1}`` to the compliance, whose limit (4) is
``N S b\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}
= (N S b/k)\,\mathbb{H}``: ``\tfrac{4\pi}{3}\,N a b^{2}\,\mathbb{H}`` for the
ellipse and ``\pi N b^{2}\,\mathbb{H}`` for the ribbon, the crack densities of
[§6](@ref th-cod-dilute).

The map ``\boldsymbol{B}\mapsto\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``
loses nothing on a symmetric ``\boldsymbol{B}``, which the COD tensor is: in a
frame where ``\underline{n} = \underline{e}_3``, the components ``i3j3`` of the
image are ``B_{ij}(1+\delta_{i3})(1+\delta_{j3})/4``. ``\boldsymbol{B}`` is
therefore read back from ``\mathbb{H}``, and the bridge works both ways:
[`compliance_from_cod`](@ref) and [`cod_from_compliance`](@ref)
(`src/Cracks/cod_H_bridge.jl`), each the exact inverse of the other, apply the
factor ``k`` of the shape family of the crack.

### [Conventions: what ``\boldsymbol{B}`` and ``\mathbb{H}`` are normalized by](@id th-cod-conventions)

Steps 1 and 2 involve no convention. Step 3 uses two lengths: the one the mean
opening is divided by in the definition of ``\boldsymbol{B}``, and the one the
flatness is measured on. `MeanFieldHomogenization` takes the half-width ``b``
for both, as [barthelemyIJSS2026](@citet) does for ``\boldsymbol{B}``.
[barthelemyIJES2021](@citet) and [barthelemyMMS2023](@citet) take the semi-major
axis ``a`` for an elliptic crack: their mean opening is
``\langle[\![\underline{u}]\!]\rangle/a`` and their flatness
``\omega = c/a = \eta\,\rho``. Steps 3 and 4 with ``a`` in place of ``b`` give

```math
\boldsymbol{B}_a = \eta\,\boldsymbol{B},
\qquad
\mathbb{H}_a = \lim_{\omega\to 0}\omega\,\mathbb{Q}^{-1} = \eta\,\mathbb{H}
= \tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}_a\stackrel{s}{\otimes}\underline{n}:
```

the factor ``3/4`` holds in either convention, while the tensors of an elliptic
crack differ by ``\eta``. The ribbon has no length ``a``, and the sources
normalize it by ``b`` as this package does. Echoes follows the elliptic
normalization by ``a`` and returns ``\mathbb{H}_a``:

| convention | elliptic | ribbon |
| :--------- | :------- | :----- |
| **`MeanFieldHomogenization`**, ``\boldsymbol{B}`` of [barthelemyIJSS2026](@citet): ``b`` | ``\lim \rho\,\mathbb{Q}^{-1} = \tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` | ``\lim \rho\,\mathbb{Q}^{-1} = \tfrac{2}{\pi}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` |
| [barthelemyIJES2021](@citet), [barthelemyMMS2023](@citet), Echoes: ``a`` for the ellipse | ``\lim \omega\,\mathbb{Q}^{-1} = \tfrac{3\eta}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` | as in `MeanFieldHomogenization` |

They coincide for the **penny crack** ``\eta=1``, which is why the discrepancy is
invisible on the most common test case — and why a penny-only cross-check cannot
detect a wrong ``\eta``-dependence.

Measured, not inferred from the papers — Echoes' `crack_compliance` against
[`inverse_property_contribution`](@ref) on a flat triaxial ellipsoid
(``a = 1``, ``b = \eta``, ``c = 10^{-5}``):

| ``\eta`` | 1.0 | 0.7 | 0.5 | 0.3 | 0.1 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| ``\mathbb{H}_{\mathrm{Echoes}}/\mathbb{H}_{\mathrm{MFH}}`` | 1.0000 | 0.7000 | 0.5000 | 0.3000 | 0.1000 |

On the `MeanFieldHomogenization` side the ``3/4`` is ``\eta``-independent to machine
precision (``\mathbb{H}_{3333}/B_{33} = 0.750000`` for every ``\eta``).

!!! warning "Check the normalization before comparing across sources"
    For an elliptic crack, the ``\boldsymbol{B}`` and ``\mathbb{H}`` of
    [barthelemyIJES2021](@citet) and [barthelemyMMS2023](@citet), and the
    ``\mathbb{H}`` that Echoes returns (it never forms ``\boldsymbol{B}``), are
    ``\eta`` times those of this package and of [barthelemyIJSS2026](@citet).
    Whether you compare a ``\boldsymbol{B}`` or an ``\mathbb{H}``, check the
    normalization first.

### [Checking ``\mathbb{H}`` without going through ``\boldsymbol{B}``](@id th-cod-H-check)

The closed forms of the next section give ``\boldsymbol{B}``, and ``\mathbb{H}``
follows from it by the ``3/4``. Comparing ``\mathbb{H}`` to those closed forms
would therefore verify nothing: it is the same formula read twice.

Read from right to left the boxed definition is a **recipe**: take a flat
ellipsoid, compute ``\mathbb{P}``, assemble
``\mathbb{Q} = \mathbb{C}-\mathbb{C}:\mathbb{P}:\mathbb{C}``, invert, scale by
the flatness ``\rho = c/b``. That path never touches the crack machinery, so
agreement with [`inverse_property_contribution`](@ref) is evidence rather than
tautology:

```@example Horacle
using MeanFieldHomogenization, TensND, LinearAlgebra, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

E, ν = 210.0, 0.3
C₀ = TensISO{3}(E / (1 - 2ν), E / (1 + ν))   # (3k, 2μ)

## ρ-family of the definition, evaluated at finite flatness
function H_from_hill(a, b, ρ, C₀)
    P = hill_tensor(Ellipsoid(a, b, ρ * b), C₀)
    Q = C₀ - C₀ ⊡ P ⊡ C₀
    return ρ * inv(Q)
end

relerr(A, B) = norm(A - B) / norm(B)   # Frobenius norm, basis-independent

ρs = exp10.(range(-1, -3; length = 9))
plt = plot(; xscale = :log10, yscale = :log10, legend = :topleft,
           xlabel = "flatness ρ = c/b", ylabel = "relative error on ℍ")
for η in (1.0, 0.5, 0.3)
    crack = η == 1.0 ? PennyCrack(1.0) : EllipticCrack(1.0, η)
    H_ref = inverse_property_contribution(crack, C₀)
    plot!(plt, ρs, [relerr(H_from_hill(1.0, η, ρ, C₀), H_ref) for ρ in ρs];
          marker = :circle, label = "η = $η")
end
plot!(plt, ρs, ρs; linestyle = :dash, color = :black, label = "slope 1")
plt
```

The three curves are straight lines of **slope 1**: the error decays like
``\rho``, which is the order of the Taylor term that resolves the limit
[barthelemyIJSS2009](@cite). That slope is the real content of the check — a
single ``\rho`` would not distinguish a true limit from a coincidence.

!!! note "How flat the flattening route goes"
    The sweep stops at ``\rho = 10^{-3}``. Both numerical back-ends hold far
    below it: at ``c = 10^{-4}`` the residues agree with the cubature to
    ``10^{-15}``. None of this concerns [`cod_tensor`](@ref), which resolves the
    limit analytically instead of flattening an ellipsoid.

### [Every route, in a triclinic matrix](@id th-cod-every-route)

In a matrix without any symmetry, ``\boldsymbol{B}`` comes from the reduced
kernel ``\hat{\boldsymbol{Q}}^{\star}_{nn}`` of [section 4](@ref th-cod-green)
by three integrators, and ``\mathbb{H}`` from the flattening above with the
Hill tensor by three integrators. The flattening error being ``O(\rho)``, the
combination ``\bigl(10\,\mathbb{H}(\rho/10)-\mathbb{H}(\rho)\bigr)/9`` cancels
its first-order term and leaves ``O(\rho^{2})``. Each side is compared with the
other through [`compliance_from_cod`](@ref) and [`cod_from_compliance`](@ref):

```@example Horacle
import DECUHR, Integrals   # loads the `:decuhr` back-end
using Printf

KM = [210.0 80 75 5 4 3; 80 195 90 -2 3 -1; 75 90 220 1 -2 2;
      5 -2 1 60 2.5 1.5; 4 3 -2 2.5 65 -1; 3 -1 2 1.5 -1 55]   # Kelvin–Mandel
C_tri = TensND.from_mandel(KM, TensND.CanonicalBasis{3, Float64}())
crack = EllipticCrack(1.0, 0.5)
tol = (; abstol = 1.0e-12, reltol = 1.0e-10)

B_ref = cod_tensor(crack, C_tri; method = :nestedquadgk, tol...)
H_ref = compliance_from_cod(B_ref, crack)
function H_flat(method, ρ)   # (c/b) ℚ⁻¹ of the flattened ellipsoid
    P = hill_tensor(Ellipsoid(1.0, 0.5, 0.5ρ), C_tri; method, tol...)
    return ρ * inv(C_tri - C_tri ⊡ P ⊡ C_tri)
end

for m in (:residues, :decuhr)
    @printf "B by %-12s against :nestedquadgk   %.0e\n" m relerr(cod_tensor(crack, C_tri; method = m, tol...), B_ref)
end
for (m, ρ) in ((:nestedquadgk, 1.0e-4), (:residues, 1.0e-4), (:decuhr, 1.0e-2))
    H = (10 * H_flat(m, ρ / 10) - H_flat(m, ρ)) / 9
    @printf "H flattened, %-12s ρ = %.0e: on H %.0e, on B %.0e\n" m ρ relerr(H, H_ref) relerr(cod_from_compliance(H, crack), B_ref)
end
```

The three integrations of ``\boldsymbol{B}`` agree to rounding. The extrapolated
flattening meets them to about ``10^{-9}`` with the cubature and the residues,
and to about ``10^{-5}`` with DECUHR, taken at ``\rho = 10^{-2}`` and
``10^{-3}`` only, where it stays fast.

The tutorial [Two routes to a crack](@ref tut-crack-routes) takes the limit
otherwise: from the first-order term of ``\mathbb{P}`` in ``\rho``, the route of
[barthelemyIJSS2009](@citet), with no extrapolation, in random anisotropic
matrices and for several aspect ratios, and exactly in SymPy for an isotropic
one.

## [4. From the Green operator to ``\boldsymbol{B}``](@id th-cod-green)

The closed forms of the next section are usually quoted. They are in fact
derivable in closed form from the Fourier Green operator, and knowing *where*
each factor comes from is what tells you which anisotropies admit a closed form
at all. The script `scripts/09_cod_symbolic_green.jl` carries out the whole
derivation symbolically, with the shipped closed forms as its oracles.

### The reduced kernel ``\hat{\boldsymbol{Q}}^{\star}_{nn}``

With ``\boldsymbol{N}(\underline{\xi}) = \underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}``
the **acoustic** (Christoffel) tensor, the two Fourier kernels of the traction
integral equation on the crack plane are [kunin1983, kanaun2009](@cite):

```math
\hat{\mathbb{G}}^0(\underline{\xi})
  = \underline{\xi}\stackrel{s}{\otimes}\boldsymbol{N}^{-1}(\underline{\xi})
    \stackrel{s}{\otimes}\underline{\xi},
\qquad
\hat{\mathbb{Q}}(\underline{\xi})
  = \mathbb{C} - \mathbb{C}:\hat{\mathbb{G}}^0(\underline{\xi}):\mathbb{C},
```

and the object the crack problem actually needs is the **reduced** transform —
the Fourier transform of the restriction of ``\hat{\mathbb{Q}}`` to the crack
plane [barthelemyIJSS2026](@cite):

```math
\boxed{\;
\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\xi}^{\star})
= \frac{1}{2\pi}\int_{-\infty}^{+\infty}
  \underline{n}\cdot\hat{\mathbb{Q}}(\underline{\xi}^{\star}+\xi_n\underline{n})
  \cdot\underline{n}\;\mathrm{d}\xi_n \; }
```

Two properties do all the work. Contracting twice with ``\underline{n}``
collapses the order-4 algebra to a 3×3 one,

```math
\underline{n}\cdot\hat{\mathbb{Q}}(\underline{\xi})\cdot\underline{n}
= \boldsymbol{A} - \boldsymbol{V}(\underline{\xi})\cdot
  \boldsymbol{N}^{-1}(\underline{\xi})\cdot\boldsymbol{V}^{\!T}(\underline{\xi}),
\qquad
\boldsymbol{A} = \underline{n}\cdot\mathbb{C}\cdot\underline{n},
\qquad
\boldsymbol{V} = \underline{n}\cdot\mathbb{C}\cdot\underline{\xi},
```

which is the form every back-end evaluates (`Core/green_helpers.jl`). And
``\hat{\mathbb{Q}}`` is homogeneous of degree ``0``, so the singly contracted
kernel decays like ``1/\xi_n`` while the **doubly** contracted one decays like
``1/\xi_n^{2}``: the integral above converges as written, and
``\hat{\boldsymbol{Q}}^{\star}_{nn}`` is homogeneous of degree ``1``.

### Three reductions, then the integral

The integrand is a rational function of ``\xi_n`` whose denominator is
``\det\boldsymbol{N}``, a **sextic** whose roots are the Stroh eigenvalues.
Three structural reductions precede any integration:

1. **Equivariance.** When ``\mathbb{C}`` is invariant under rotations about
   ``\underline{n}`` — isotropic, or transversely isotropic with its axis along
   ``\underline{n}`` — so is the whole construction, since ``\underline{n}`` is
   also the integrated direction. One in-plane direction determines all of them.
2. **Homogeneity.** Degree ``1`` factors out a single
   ``\rho = \|\underline{\xi}^{\star}\|``.
3. **Parity.** The ``\underline{u}``–``\underline{n}`` block of the integrand is
   odd in ``\xi_n`` and integrates to zero.

Together, with ``\underline{u} = \underline{\xi}^{\star}/\rho`` and
``\underline{w} = \underline{n}\wedge\underline{u}``:

```math
\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\xi}^{\star})
= \rho\,\bigl[\,a_1\,\underline{u}\otimes\underline{u}
            + a_2\,\underline{w}\otimes\underline{w}
            + a_3\,\underline{n}\otimes\underline{n}\,\bigr].
```

For an **isotropic** matrix the sextic degenerates to
``\mu^{2}(\lambda+2\mu)\|\underline{\xi}\|^{6}``, leaving the single pair of
double poles ``\xi_n = \pm i\rho``, and the three coefficients are just the
plane-strain and antiplane moduli, derived step by step in
[The isotropic matrix, from the kernel to ``\boldsymbol{B}``](@ref th-cod-iso-derivation):

```math
a_1 = a_3 = \frac{\mu}{2(1-\nu)} = \frac{E}{4(1-\nu^{2})},
\qquad
a_2 = \frac{\mu}{2}.
```

For a **transversely isotropic** matrix with the crack in the plane of isotropy,
every component ``N_{12}``, ``N_{23}`` carries an odd number of ``2`` indices and
therefore vanishes: the antiplane (SH) polarization decouples, and the sextic
splits into a quadratic and a **biquadratic** — solvable by radicals,

```math
\det\boldsymbol{N} = C_{2323}\bigl(\xi_n^{2}+\gamma_3^{2}\rho^{2}\bigr)\;
  C_{3333}C_{2323}\bigl(\xi_n^{2}+\gamma_1^{2}\rho^{2}\bigr)
  \bigl(\xi_n^{2}+\gamma_2^{2}\rho^{2}\bigr),
```

```math
\gamma_3^{2} = \frac{C_{1212}}{C_{2323}},
\qquad
\gamma_1\gamma_2 = \sqrt{\frac{C_{1111}}{C_{3333}}},
\qquad
\gamma_1+\gamma_2 = \sigma_\gamma,
```

```math
a_1 = \frac{C_{1111}C_{3333}-C_{1133}^{2}}{2\,\sigma_\gamma\,C_{3333}},
\qquad
a_2 = \frac{\sqrt{C_{2323}C_{1212}}}{2},
\qquad
a_3 = \frac{a_1}{\gamma_1\gamma_2}.
```

The radical ``\sigma_\gamma`` of the published TI closed form
[hoenig1978, barthelemyIJES2021](@cite) is therefore nothing but the
sum of the two in-plane Stroh roots — it *comes out of* the factorization rather
than being postulated.

### From the reduced kernel to ``\boldsymbol{B}``, for any anisotropy

The opening of an open crack under a uniform remote stress is an ellipsoidal
cap, of maximum ``\underline{\beta}``. Carried through the traction integral
equation, that profile gives a **uniform** traction on the crack,
``\boldsymbol{\Sigma}\cdot\underline{n} = \boldsymbol{\Lambda}\cdot\underline{\beta}``,
with [barthelemyIJSS2026](@cite)

```math
\begin{aligned}
\boldsymbol{\Lambda}
&= \frac{1}{4}\int_{0}^{2\pi}
  \hat{\boldsymbol{Q}}^{\star}_{nn}\bigl(\boldsymbol{S}^{\dagger}\!\cdot\underline{\kappa}^{\star}(\varphi)\bigr)\,
  \mathrm{d}\varphi,
\\
\underline{\kappa}^{\star}(\varphi) &= \cos\varphi\,\underline{\ell}+\sin\varphi\,\underline{m},
\qquad
\boldsymbol{S}^{\dagger} = \frac{\underline{\ell}\otimes\underline{\ell}}{a}
                         + \frac{\underline{m}\otimes\underline{m}}{b}.
\end{aligned}
```

Since ``\langle[\![\underline{u}]\!]\rangle_{\mathcal{I}} = \chi\,\underline{\beta}``,
the definition of ``\boldsymbol{B}`` reads
``\boldsymbol{B} = \chi\,(b\boldsymbol{\Lambda})^{-1}``. The kernel being
homogeneous of degree ``1``, ``b\boldsymbol{\Lambda}`` depends on the shape
only, through
``b\,\boldsymbol{S}^{\dagger}\!\cdot\underline{\kappa}^{\star}(\varphi)
= \eta\cos\varphi\,\underline{\ell}+\sin\varphi\,\underline{m}``, and

```math
\boxed{\;
\boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
= \frac{8}{3}\left(\int_{0}^{2\pi}
  \hat{\boldsymbol{Q}}^{\star}_{nn}\bigl(\eta\cos\varphi\,\underline{\ell}
  +\sin\varphi\,\underline{m}\bigr)\,\mathrm{d}\varphi\right)^{-1},
\qquad
\boldsymbol{B}^{\mathcal{R}}(\underline{m},\underline{n})
= \frac{\pi}{4}\,\bigl(\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{m})\bigr)^{-1}.
\;}
```

These hold whatever the anisotropy of the matrix. The closed forms of the next
section are their evaluations when ``\hat{\boldsymbol{Q}}^{\star}_{nn}`` is
known explicitly; otherwise the contour integral is computed numerically. For
the ribbon, ``a\to\infty`` leaves
``\boldsymbol{S}^{\dagger}\to\underline{m}\otimes\underline{m}/b`` and the
contour integral reduces to ``\int_0^{2\pi}|\sin\varphi|\,\mathrm{d}\varphi/4 = 1``.

```@raw html
<details><summary>Fourier conventions, the reduced transform and the ellipsoidal profile</summary>
```

The Fourier transform of a function ``\phi`` on ``\mathbb{R}^{d}`` and its
inverse are taken as

```math
\hat{\phi}(\underline{\xi}) = \int_{\mathbb{R}^{d}}
  e^{-i\,\underline{\xi}\cdot\underline{x}}\,\phi(\underline{x})\,\mathrm{d}\Omega_{x},
\qquad
\phi(\underline{x}) = \frac{1}{(2\pi)^{d}}\int_{\mathbb{R}^{d}}
  e^{i\,\underline{\xi}\cdot\underline{x}}\,\hat{\phi}(\underline{\xi})\,\mathrm{d}\Omega_{\xi}.
```

Splitting ``\underline{\xi} = \underline{\xi}^{\star}+\xi_n\underline{n}`` in
the inverse transform and integrating over ``\xi_n`` first shows that the
reduced transform
``\hat{\phi}^{\star}(\underline{\xi}^{\star}) = \frac{1}{2\pi}\int
\hat{\phi}(\underline{\xi}^{\star}+\xi_n\underline{n})\,\mathrm{d}\xi_n`` is the
``(d-1)``-dimensional Fourier transform of the restriction of ``\phi`` to the
plane ``\underline{n}^{\perp}``. A shift of ``\xi_n`` leaves it unchanged, so it
ignores any component of its argument along ``\underline{n}``; a change of
variable ``\xi_n\to\lambda\xi_n`` shows that it raises the degree of
homogeneity by one, which is how ``\hat{\mathbb{Q}}``, of degree ``0``, gives a
kernel of degree ``1``.

The ellipsoidal opening enters through the transform of the hemispherical cap
on the unit disc,

```math
\int_{\|\underline{x}\|\le 1}
  e^{-i\,\underline{\xi}\cdot\underline{x}}\,\sqrt{1-\|\underline{x}\|^{2}}\;\mathrm{d}S_{x}
= 2\pi\,\frac{j_1(\|\underline{\xi}\|)}{\|\underline{\xi}\|},
\qquad
j_1(s) = \frac{\sin s - s\cos s}{s^{2}},
```

an elliptical crack being mapped onto the unit disc by its semi-axis tensor.
Inserted in the integral equation, this transform is what turns it into the
uniform traction ``\boldsymbol{\Lambda}\cdot\underline{\beta}`` above
[barthelemyIJSS2026](@cite).

```@raw html
</details>
```

### The crack-plane integral: where the elliptic integrals enter

On the crack contour ``\underline{\xi}^{\star}(\varphi) =
\eta\cos\varphi\,\underline{\ell} + \sin\varphi\,\underline{m}`` of
[barthelemyIJSS2026](@citet),
``\rho = \sqrt{\eta^{2}\cos^{2}\varphi+\sin^{2}\varphi}``,
``\underline{u} = (\eta\cos\varphi\,\underline{\ell}+\sin\varphi\,\underline{m})/\rho``
and
``\underline{w} = \underline{n}\wedge\underline{u}
= (\eta\cos\varphi\,\underline{m}-\sin\varphi\,\underline{\ell})/\rho``, so that

```math
\begin{aligned}
\rho\,\underline{u}\otimes\underline{u}
&= \frac{\eta^{2}\cos^{2}\varphi\,\underline{\ell}\otimes\underline{\ell}
       + \sin^{2}\varphi\,\underline{m}\otimes\underline{m}
       + \eta\cos\varphi\sin\varphi\,(\underline{\ell}\otimes\underline{m}+\underline{m}\otimes\underline{\ell})}{\rho},
\\
\rho\,\underline{w}\otimes\underline{w}
&= \frac{\sin^{2}\varphi\,\underline{\ell}\otimes\underline{\ell}
       + \eta^{2}\cos^{2}\varphi\,\underline{m}\otimes\underline{m}
       - \eta\cos\varphi\sin\varphi\,(\underline{\ell}\otimes\underline{m}+\underline{m}\otimes\underline{\ell})}{\rho}.
\end{aligned}
```

The ``\rho`` of the degree-1 homogeneity has canceled the ``1/\rho^{2}`` of the
two in-plane dyads. ``\rho`` being even in ``\varphi``, the
``\cos\varphi\sin\varphi`` cross terms are odd and integrate to zero over
``[0,2\pi]``, which is why ``\boldsymbol{B}`` is diagonal in the crack frame.
**Exactly three** angular integrals survive. The integrands being even and
``\pi``-periodic, ``\frac14\int_0^{2\pi} = \int_0^{\pi/2}``, and the change of
variable ``\phi = \pi/2-\varphi`` turns ``\rho`` into
``\sqrt{\cos^{2}\phi+\eta^{2}\sin^{2}\phi}``:

```math
\frac14\!\int_0^{2\pi}\!\rho\,\mathrm{d}\varphi = \mathcal{E}_\eta,
\qquad
\frac14\!\int_0^{2\pi}\!\frac{\sin^{2}\varphi}{\rho}\,\mathrm{d}\varphi
  = \mathcal{C}_\eta,
\qquad
\frac14\!\int_0^{2\pi}\!\frac{\eta^{2}\cos^{2}\varphi}{\rho}\,\mathrm{d}\varphi
  = \eta^{2}\mathcal{S}_\eta,
```

with the complete elliptic integrals and the two combinations of
[barthelemyIJSS2026](@citet),

```math
\mathcal{K}_\eta = \int_0^{\pi/2}\frac{\mathrm{d}\phi}{\sqrt{\cos^{2}\phi+\eta^{2}\sin^{2}\phi}},
\qquad
\mathcal{E}_\eta = \int_0^{\pi/2}\sqrt{\cos^{2}\phi+\eta^{2}\sin^{2}\phi}\;\mathrm{d}\phi,
```

```math
\begin{aligned}
\mathcal{C}_\eta &= \int_0^{\pi/2}\frac{\cos^{2}\phi\;\mathrm{d}\phi}{\sqrt{\cos^{2}\phi+\eta^{2}\sin^{2}\phi}}
= \frac{\mathcal{E}_\eta-\eta^{2}\mathcal{K}_\eta}{1-\eta^{2}},
\\
\mathcal{S}_\eta &= \int_0^{\pi/2}\frac{\sin^{2}\phi\;\mathrm{d}\phi}{\sqrt{\cos^{2}\phi+\eta^{2}\sin^{2}\phi}}
= \frac{\mathcal{K}_\eta-\mathcal{E}_\eta}{1-\eta^{2}}.
\end{aligned}
```

The second forms follow from ``D = \cos^{2}\phi+\eta^{2}\sin^{2}\phi``, which
gives ``\sin^{2}\phi = (1-D)/(1-\eta^{2})`` and
``\cos^{2}\phi = (D-\eta^{2})/(1-\eta^{2})``, with ``\int_0^{\pi/2}D^{-1/2} = \mathcal{K}_\eta``
and ``\int_0^{\pi/2}D^{1/2} = \mathcal{E}_\eta``. In terms of the Legendre
*parameter* ``m = 1-\eta^{2}`` of [Elliptic integrals](@ref th-elliptic-integrals),
``\mathcal{K}_\eta = K(1-\eta^{2})`` and ``\mathcal{E}_\eta = E(1-\eta^{2})``;
`_elliptic_CS` returns ``(\mathcal{C}_\eta, \mathcal{S}_\eta, \mathcal{E}_\eta)``.

Hence

```math
b\boldsymbol{\Lambda}
= \mathrm{diag}\bigl(
  a_1\eta^{2}\mathcal{S}_\eta + a_2\mathcal{C}_\eta,\;
  a_1\mathcal{C}_\eta + a_2\eta^{2}\mathcal{S}_\eta,\;
  a_3\mathcal{E}_\eta\bigr),
\qquad
\boldsymbol{B} = \chi\,(b\boldsymbol{\Lambda})^{-1},
```

with ``\chi`` the shape coefficient of the section above. The elliptic block is
**the same** for the isotropic and the aligned-TI matrix — only ``a_1,a_2,a_3``
change. That is why the two closed forms of the next section share one skeleton.

For the ribbon the contour integral is replaced by the single direction
``\underline{u} = \underline{m}``, so
``\underline{w} = -\underline{\ell}``, and with ``\chi^{\mathcal{R}} = \pi/4``:

```math
\boldsymbol{B}^{\mathcal{R}}(\underline{m},\underline{n})
= \frac{\pi}{4}\,\bigl(\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{m})\bigr)^{-1}.
```

This is the exact identity behind the SIF ↔ DIF exchange relation at the end of
this page: the operator that trades ``\underline{K}`` for ``\underline{N}`` *is*
``\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\nu})``.

### [The isotropic matrix, from the kernel to ``\boldsymbol{B}``](@id th-cod-iso-derivation)

The steps above, carried out for
``\mathbb{C} = \lambda\,\boldsymbol{1}\otimes\boldsymbol{1} + 2\mu\,\mathbb{I}``,
end on the closed form of ``\boldsymbol{B}``. Each line was checked by computer
algebra.

**1. The three tensors of the kernel.** With
``\underline{\xi} = \underline{\xi}^{\star}+\xi_n\underline{n}`` and
``\underline{\xi}^{\star} = \rho\,\underline{u}``,

```math
\begin{aligned}
\boldsymbol{N}(\underline{\xi}) &= \mu\,\|\underline{\xi}\|^{2}\,\boldsymbol{1}
+ (\lambda+\mu)\,\underline{\xi}\otimes\underline{\xi},
\qquad
\boldsymbol{A} = \mu\,\boldsymbol{1} + (\lambda+\mu)\,\underline{n}\otimes\underline{n},
\\
\boldsymbol{V} &= \lambda\,\underline{n}\otimes\underline{\xi}
+ \mu\,\underline{\xi}\otimes\underline{n}
+ \mu\,(\underline{\xi}\cdot\underline{n})\,\boldsymbol{1},
\end{aligned}
```

and ``\boldsymbol{N}^{-1} = \big(\boldsymbol{1} - \kappa\,\underline{\xi}\otimes\underline{\xi}/\|\underline{\xi}\|^{2}\big)/(\mu\|\underline{\xi}\|^{2})``
with ``\kappa = (\lambda+\mu)/(\lambda+2\mu) = 1/(2(1-\nu))``.

**2. The doubly contracted kernel.** In the orthonormal frame
``(\underline{u},\underline{w},\underline{n})``,
``\underline{n}\cdot\hat{\mathbb{Q}}\cdot\underline{n} = \boldsymbol{A} - \boldsymbol{V}\cdot\boldsymbol{N}^{-1}\cdot\boldsymbol{V}^{\!T}`` has the components

```math
\begin{aligned}
Q_{uu} &= \frac{2\mu}{1-\nu}\,\frac{\rho^{2}\xi_n^{2}}{(\rho^{2}+\xi_n^{2})^{2}},
&\qquad
Q_{ww} &= \mu\,\frac{\rho^{2}}{\rho^{2}+\xi_n^{2}},
\\
Q_{nn} &= \frac{2\mu}{1-\nu}\,\frac{\rho^{4}}{(\rho^{2}+\xi_n^{2})^{2}},
&\qquad
Q_{un} = Q_{nu} &= -\frac{2\mu}{1-\nu}\,\frac{\rho^{3}\xi_n}{(\rho^{2}+\xi_n^{2})^{2}},
\end{aligned}
```

and ``Q_{uw} = Q_{wn} = 0``. All decay as ``\xi_n^{-2}``, as announced.

**3. The integral over ``\xi_n``.** ``Q_{un}`` is odd in ``\xi_n`` and integrates
to zero. The three others need only

```math
\int_{-\infty}^{+\infty}\frac{\mathrm{d}\xi_n}{\rho^{2}+\xi_n^{2}} = \frac{\pi}{\rho},
\qquad
\int_{-\infty}^{+\infty}\frac{\xi_n^{2}\,\mathrm{d}\xi_n}{(\rho^{2}+\xi_n^{2})^{2}} = \frac{\pi}{2\rho},
\qquad
\int_{-\infty}^{+\infty}\frac{\mathrm{d}\xi_n}{(\rho^{2}+\xi_n^{2})^{2}} = \frac{\pi}{2\rho^{3}},
```

so that, with the factor ``1/(2\pi)`` of the reduced transform,

```math
\begin{gathered}
\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\xi}^{\star})
= \rho\,\big[a_1\,\underline{u}\otimes\underline{u}
+ a_2\,\underline{w}\otimes\underline{w}
+ a_3\,\underline{n}\otimes\underline{n}\big],
\\
a_1 = a_3 = \frac{\mu}{2(1-\nu)} = \frac{E}{4(1-\nu^{2})},
\qquad
a_2 = \frac{\mu}{2} = \frac{E}{4(1+\nu)}.
\end{gathered}
```

**4. The contour integral.** With the angular integrals of the previous section,

```math
\begin{aligned}
b\boldsymbol{\Lambda}
&= \frac{E}{4(1-\nu^{2})}\Big[
  \big(\eta^{2}\mathcal{S}_\eta + (1-\nu)\,\mathcal{C}_\eta\big)\,\underline{\ell}\otimes\underline{\ell}
\\
&\qquad {}+ \big(\mathcal{C}_\eta + (1-\nu)\,\eta^{2}\mathcal{S}_\eta\big)\,\underline{m}\otimes\underline{m}
+ \mathcal{E}_\eta\,\underline{n}\otimes\underline{n}\Big],
\end{aligned}
```

using ``a_2/a_1 = 1-\nu``.

**5. The inversion.** ``b\boldsymbol{\Lambda}`` is diagonal in the crack frame,
and ``\chi^{\mathcal{E}} = 2/3``:

```math
\begin{aligned}
\boldsymbol{B}^{\mathcal{E}}
&= \frac{2}{3}\,(b\boldsymbol{\Lambda})^{-1}
\\
&= \frac{8\,(1-\nu^{2})}{3E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{(1-\nu)\,\mathcal{C}_\eta+\eta^{2}\mathcal{S}_\eta}
+ \frac{\underline{m}\otimes\underline{m}}{(1-\nu)\,\eta^{2}\mathcal{S}_\eta+\mathcal{C}_\eta}
+ \frac{\underline{n}\otimes\underline{n}}{\mathcal{E}_\eta}\right),
\end{aligned}
```

the form of [barthelemyIJSS2026](@citet). Replacing ``\mathcal{C}_\eta`` and
``\mathcal{S}_\eta`` by their expressions in ``\mathcal{K}_\eta`` and
``\mathcal{E}_\eta``,

```math
\begin{aligned}
(1-\nu)\,\mathcal{C}_\eta+\eta^{2}\mathcal{S}_\eta
&= \frac{(1-\nu-\eta^{2})\,\mathcal{E}_\eta + \nu\,\eta^{2}\mathcal{K}_\eta}{1-\eta^{2}},
\\
(1-\nu)\,\eta^{2}\mathcal{S}_\eta+\mathcal{C}_\eta
&= \frac{\big(1-(1-\nu)\eta^{2}\big)\,\mathcal{E}_\eta - \nu\,\eta^{2}\mathcal{K}_\eta}{1-\eta^{2}},
\end{aligned}
```

gives the form of the next section.

**6. The two limits.** At the circle, ``\mathcal{C}_1 = \mathcal{S}_1 = \pi/4`` and
``\mathcal{E}_1 = \pi/2``, so
``B_{nn} = 16(1-\nu^{2})/(3\pi E)`` and
``B_{\ell\ell} = B_{mm} = B_{nn}/(1-\nu/2)``. For the ribbon,
``b\boldsymbol{\Lambda}^{\mathcal{R}} = \hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{m})``,
where ``\underline{u} = \underline{m}`` and ``\underline{w} = -\underline{\ell}``:
``b\boldsymbol{\Lambda}^{\mathcal{R}} = a_2\,\underline{\ell}\otimes\underline{\ell}
+ a_1\,\underline{m}\otimes\underline{m} + a_3\,\underline{n}\otimes\underline{n}``,
and with ``\chi^{\mathcal{R}} = \pi/4``

```math
\boldsymbol{B}^{\mathcal{R}}
= \frac{\pi}{4}\,(b\boldsymbol{\Lambda}^{\mathcal{R}})^{-1}
= \frac{\pi\,(1-\nu^{2})}{E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{1-\nu}
+ \underline{m}\otimes\underline{m}
+ \underline{n}\otimes\underline{n}\right).
```

### What breaks in the general case

Nothing above survives a general anisotropy: the acoustic tensor no longer
block-decouples, ``\det\boldsymbol{N}`` is an irreducible sextic in ``\xi_n``, and
the crack plane has no rotational symmetry left, so one in-plane direction no
longer determines the others. Both integrals become numerical — which is what
the `Residue` and cubature back-ends do, the first summing residues over the six
Stroh roots located numerically.

## 5. Closed forms of ``\boldsymbol{B}``

### Isotropic matrix

For ``\mathbb{C} = 3k\,\mathbb{J}+2\mu\,\mathbb{K}`` with Young's modulus ``E``
and Poisson ratio ``\nu``, in the crack frame
``(\underline{\ell},\underline{m},\underline{n})`` — derived step by step in
[The isotropic matrix, from the kernel to ``\boldsymbol{B}``](@ref th-cod-iso-derivation):

```math
\boldsymbol{B}
= B_{nn}\,\underline{n}\otimes\underline{n}
+ B_{mm}\,\underline{m}\otimes\underline{m}
+ B_{\ell\ell}\,\underline{\ell}\otimes\underline{\ell},
```

```math
\begin{aligned}
B_{nn} &= \frac{8\,(1-\nu^{2})}{3E}\,\frac{1}{\mathcal{E}_\eta},\\[4pt]
B_{mm} &= \frac{8\,(1-\nu^{2})}{3E}\,
          \frac{1-\eta^{2}}
               {\bigl(1-(1-\nu)\eta^{2}\bigr)\mathcal{E}_\eta - \nu\,\eta^{2}\,\mathcal{K}_\eta},\\[4pt]
B_{\ell\ell} &= \frac{8\,(1-\nu^{2})}{3E}\,
          \frac{1-\eta^{2}}
               {\bigl(1-\nu-\eta^{2}\bigr)\mathcal{E}_\eta + \nu\,\eta^{2}\,\mathcal{K}_\eta},
\end{aligned}
```

with ``\mathcal{K}_\eta = \mathcal{K}(\sqrt{1-\eta^{2}})`` and
``\mathcal{E}_\eta = \mathcal{E}(\sqrt{1-\eta^{2}})`` the complete elliptic
integrals of the first and second kind [abramowitz1972](@cite), provided by
[`ell_K`](@ref) and [`ell_E`](@ref).

!!! note "Near the circle"
    ``\mathcal{C}_\eta`` and ``\mathcal{S}_\eta``, written with ``\mathcal{K}_\eta``
    and ``\mathcal{E}_\eta``, cancel as ``\eta\to1``; they are evaluated by series
    there. And ``\boldsymbol{B}``, a function of the ratio of the smaller
    semi-axis to the larger, has a kink at the circle, where the two swap
    roles: its derivative there is one-sided.
    [Degenerate shapes](@ref th-degenerate-shapes) gives both.

In terms of the combinations ``\mathcal{C}_\eta`` and ``\mathcal{S}_\eta`` of
§4, which make both limits immediate [barthelemyIJSS2026](@cite),

```math
\boldsymbol{B}^{\mathcal{E}}
= \frac{8\,(1-\nu^{2})}{3E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{(1-\nu)\,\mathcal{C}_\eta+\eta^{2}\mathcal{S}_\eta}
+ \frac{\underline{m}\otimes\underline{m}}{(1-\nu)\,\eta^{2}\mathcal{S}_\eta+\mathcal{C}_\eta}
+ \frac{\underline{n}\otimes\underline{n}}{\mathcal{E}_\eta}\right).
```

The three components are the three fracture modes: ``B_{nn}`` opens the crack
(mode I), ``B_{mm}`` and ``B_{\ell\ell}`` shear it (modes II and III).

**Penny crack** ``\eta=1``, where ``\mathcal{K}_1 = \mathcal{E}_1 = \pi/2``:

```math
B_{nn} = \frac{16\,(1-\nu^{2})}{3\pi E},
\qquad
B_{mm} = B_{\ell\ell} = \frac{B_{nn}}{1-\nu/2}.
```

**Ribbon crack**, ``\frac{3\pi}{8}\lim_{\eta\to 0}\boldsymbol{B}^{\mathcal{E}}``
with ``\mathcal{C}_0 = \mathcal{E}_0 = 1`` and ``\eta^{2}\mathcal{S}_\eta\to 0``:

```math
\boldsymbol{B}^{\mathcal{R}}
= \frac{\pi\,(1-\nu^{2})}{E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{1-\nu}
+ \underline{m}\otimes\underline{m}
+ \underline{n}\otimes\underline{n}\right).
```

Implementation: `src/Cracks/cod_analytical.jl`, selected by `method = :auto`
when ``\mathbb{C}_0`` is a `TensISO{4,3}`.

### Transversely isotropic matrix

When the matrix is transversely isotropic with its axis **aligned with the crack
normal** ``\underline{n}``, ``\boldsymbol{B}`` is still analytical. The closed
forms are written in the engineering parameters of [hoenig1978](@citet), defined
on the compliance ``\mathbb{S} = \mathbb{C}^{-1}`` in a frame whose third axis is
``\underline{n}``:

```math
\begin{gathered}
E = \frac{1}{S_{1111}},
\qquad
\nu_1 = -E\,S_{1122},
\qquad
\nu_2 = -E\,S_{1133},
\\
H = \frac{1}{E\,S_{3333}},
\qquad
\Gamma = \frac{1+\nu_1}{2\,E\,S_{2323}}.
\end{gathered}
```

With

```math
h = \sqrt{\frac{1-H\nu_2^{2}}{H\,(1-\nu_1^{2})}},
\qquad
\sigma_\gamma = \sqrt{2}\,\sqrt{\frac{1-\Gamma\nu_2}{\Gamma\,(1-\nu_1)} + h},
\qquad
\beta = \frac{\sigma_\gamma}{2}\,\sqrt{\Gamma}\,(1-\nu_1),
```

the elliptic and ribbon COD tensors are
[hoenig1978, kanaun2009, barthelemyIJES2021, barthelemyIJSS2026](@cite)

```math
\begin{aligned}
\boldsymbol{B}^{\mathcal{E}}
&= \frac{4\,\sigma_\gamma\,(1-\nu_1^{2})}{3E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{\beta\,\mathcal{C}_\eta+\eta^{2}\mathcal{S}_\eta}
+ \frac{\underline{m}\otimes\underline{m}}{\beta\,\eta^{2}\mathcal{S}_\eta+\mathcal{C}_\eta}
+ h\,\frac{\underline{n}\otimes\underline{n}}{\mathcal{E}_\eta}\right),\\[4pt]
\boldsymbol{B}^{\mathcal{R}}
&= \frac{\pi\,(1-\nu_1^{2})}{E}\left(
  \frac{\underline{\ell}\otimes\underline{\ell}}{\sqrt{\Gamma}\,(1-\nu_1)}
+ \frac{\sigma_\gamma}{2}\,\underline{m}\otimes\underline{m}
+ \frac{\sigma_\gamma}{2}\,h\,\underline{n}\otimes\underline{n}\right).
\end{aligned}
```

They reduce to the isotropic forms above for ``\nu_1=\nu_2=\nu`` and
``H=\Gamma=1``, where ``\sigma_\gamma = 2`` and ``\beta = 1-\nu``.
``\sigma_\gamma`` is the sum ``\gamma_1+\gamma_2`` of the two in-plane Stroh roots
of §4. Implementation: `src/Cracks/cod_analytical.jl`, with the parameters
read by TensND's `ti_args_hoenig` from the compliance.

```@raw html
<details><summary>The stiffness components in the engineering parameters</summary>
```

With ``A = 1-\nu_1-2H\nu_2^{2}``, the five independent components that enter
the reduced kernel of §4 are [barthelemyIJSS2026](@cite)

```math
\begin{gathered}
C_{1111} = \frac{E\,(1-H\nu_2^{2})}{A\,(1+\nu_1)},
\qquad
C_{1122} = \frac{E\,(\nu_1+H\nu_2^{2})}{A\,(1+\nu_1)},
\qquad
C_{1133} = \frac{E\,H\,\nu_2}{A},
\\
C_{3333} = \frac{E\,H\,(1-\nu_1)}{A},
\qquad
C_{2323} = \frac{E\,\Gamma}{2\,(1+\nu_1)},
\end{gathered}
```

the isotropic case giving ``A = (1+\nu)(1-2\nu)``.

```@raw html
</details>
```

A transversely isotropic axis **not** aligned with the crack normal, and more
generally a transformed transversely isotropic matrix, are treated in
[barthelemyMMS2023](@citet) for the compliance and the average opening, and in
[barthelemyIJSS2026](@citet) for the COD tensor and the intensity factors, by a
linear transformation of the crack problem. Neither is implemented here.

### Arbitrary anisotropy — numerical

No closed form exists in general. Following [barthelemyIJSS2009](@citet), the
limit ``\omega\to 0`` is resolved by extracting the **first-order term** of the
Taylor expansion of ``\mathbb{P}`` in ``\omega``; that term has an integral
representation on the unit circle of the crack plane, evaluated by one of three
algorithm traits:

- **`NestedQuadGK`** — nested adaptive Gauss–Kronrod quadratures, type-generic
  (`src/Cracks/green_nestedquadgk.jl`);
- **`DECUHR`** — adaptive cubature [espelid1994](@cite), ForwardDiff-safe, once
  its weak dependency is loaded (`src/Cracks/green_decuhr.jl`);
- **`Residue`** — Cauchy-residue reduction to a 1-D quadrature, as in
  [masson2008](@citet) adapted to the crack kernel, `Float64` only
  (`src/Cracks/green_residue.jl`).

`method = :auto` picks `NestedQuadGK`, whatever the coefficient type and
whatever else is loaded: it is the fastest and the most accurate of the three
(see [Hill polarization tensors](@ref th-hill-tensors)). `Residue` degenerates
when the reference is anisotropic in *type* and isotropic in *value*, a case the
self-consistent and differential schemes reach at their first step, so it is
available on explicit `method = :residues` only (`src/Core/dispatch.jl`).

## [6. Dilute correction to the effective compliance](@id th-cod-dilute)

[`inverse_property_contribution`](@ref)`(crack, C₀)` returns ``\mathbb{H}`` itself —
the *size-independent* contribution, not the dilute correction. Cracks have no
volume fraction, so the amount of cracking is measured by a **Budiansky crack
density** [budiansky1976, kachanov1993](@cite), and reintroduced by
[`delta_inverse_property`](@ref):

| | density | dilute correction |
| :-- | :-- | :-- |
| **elliptic** (3-D) | ``\varepsilon^{3\mathrm{d}} = N\,a\,b^{2}`` (number per unit volume × major × minor²) | ``\Delta\mathbb{S} = \tfrac{4\pi}{3}\,\varepsilon^{3\mathrm{d}}\,\mathbb{H}^{\mathcal{E}}`` |
| **ribbon** (2-D) | ``\varepsilon^{2\mathrm{d}} = N\,b^{2}`` (number per unit area × half-width²) | ``\Delta\mathbb{S} = \pi\,\varepsilon^{2\mathrm{d}}\,\mathbb{H}^{\mathcal{R}}`` |

Implementation: `src/Cracks/compliance.jl`, dispatching on the crack shape.

## 7. Intensity factors at the crack front

At a point ``\underline{x}^{\star}_{0}`` of the crack front, with in-plane outer
normal ``\underline{\nu}`` and tangent
``\underline{\tau} = \underline{n}\wedge\underline{\nu}``, the asymptotic
expansions of the jump and the traction read [irwin1957, kassir1968, willis1968](@cite):

```math
[\![\underline{u}]\!](\underline{x}^{\star}_{0}+r\underline{\nu})
\underset{r\to 0^{-}}{\sim}
8\sqrt{\frac{-r}{2\pi}}\;\underline{N},
\qquad
\underline{t}(\underline{x}^{\star}_{0}+r\underline{\nu})
\underset{r\to 0^{+}}{\sim}
\frac{\underline{K}}{\sqrt{2\pi r}} .
```

``\underline{N}`` is the **displacement intensity factor** (DIF) and
``\underline{K}`` the **stress intensity factor** (SIF), normalized so that the
local energy release rate is simply
``G = \underline{K}\cdot\underline{N}`` [barnett1972, rice1989](@cite).

The central result of the anisotropic theory [kanaun1981, kunin1983, kanaun2009](@cite) is that SIF and DIF are **purely local**
and are exchanged by the COD tensor of the **ribbon crack tangent** to the real
crack at the observation point:

```math
\boxed{\;
\underline{K}
= \pi\,\bigl(\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\bigr)^{-1}
  \cdot\underline{N}
\;}
```

This holds whatever the matrix anisotropy and whatever the remote loading — and
it is why ``\boldsymbol{B}^{\mathcal{R}}``, with its ``3\pi/8`` factor, cannot be
dispensed with even when studying elliptic cracks. With
``\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})
= \frac{\pi}{4}\bigl(\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\nu})\bigr)^{-1}``
(§4) it also reads
``\underline{K} = 4\,\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\nu})\cdot\underline{N}``,
and the energy release rate is a quadratic form in either factor
[barthelemyIJSS2026](@cite):

```math
G = \pi\,\underline{N}\cdot
    \bigl(\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\bigr)^{-1}
    \cdot\underline{N}
  = \frac{1}{\pi}\,\underline{K}\cdot
    \boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\cdot\underline{K}.
```

### Elliptic crack

Parametrize the front by an angle ``\theta_y``:

```math
\underline{y}^{\star}_{0} = \cos\theta_y\,\underline{\ell} + \sin\theta_y\,\underline{m},
\qquad
\underline{\nu}
= \frac{\boldsymbol{S}^{\dagger}\!\cdot\underline{y}^{\star}_{0}}
       {\|\boldsymbol{S}^{\dagger}\!\cdot\underline{y}^{\star}_{0}\|},
```

with ``\boldsymbol{S}`` the in-plane semi-axis tensor and
``\boldsymbol{S}^{\dagger}`` its pseudo-inverse. Then

```math
\begin{aligned}
\underline{N}^{\mathcal{E}}
&= \tfrac{3}{8}\sqrt{\pi b}\;\sqrt{\varrho}\;
  \boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
  \cdot\boldsymbol{\Sigma}\cdot\underline{n},
\\
\underline{K}^{\mathcal{E}}
&= \tfrac{3}{8}\pi^{3/2}\sqrt{b}\;\sqrt{\varrho}\;
  \bigl(\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\bigr)^{-1}
  \cdot\boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
  \cdot\boldsymbol{\Sigma}\cdot\underline{n},
\end{aligned}
```

where the dimensionless front factor is

```math
\varrho = b\,\|\boldsymbol{S}^{\dagger}\!\cdot\underline{y}^{\star}_{0}\|
= \sqrt{\eta^{2}\cos^{2}\theta_y + \sin^{2}\theta_y}.
```

The energy release rate follows as

```math
G^{\mathcal{E}} = \frac{9\pi^{2}b}{64}\,\varrho\;
  (\boldsymbol{\Sigma}\cdot\underline{n})\cdot
  \boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)\cdot
  \bigl(\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\bigr)^{-1}\cdot
  \boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)\cdot
  (\boldsymbol{\Sigma}\cdot\underline{n}).
```

```@raw html
<details><summary>Other parametrizations of the front</summary>
```

[`sif`](@ref) and [`dif`](@ref) take the front point through
``\underline{y}^{\star}_{0}`` (keyword `y₀`, ``\underline{m}`` by default), the
point itself being
``\underline{x}^{\star}_{0} = a\cos\theta_y\,\underline{\ell} + b\sin\theta_y\,\underline{m}``.
With the polar angle ``\theta_x`` of ``\underline{x}^{\star}_{0}``, or the angle
``\theta_\nu`` of the outer normal
``\underline{\nu} = \cos\theta_\nu\,\underline{\ell}+\sin\theta_\nu\,\underline{m}``,
the front factor reads [barthelemyIJSS2026](@cite)

```math
\varrho
= \sqrt{\eta^{2}\cos^{2}\theta_y+\sin^{2}\theta_y}
= \sqrt{\frac{\eta^{4}\cos^{2}\theta_x+\sin^{2}\theta_x}
             {\eta^{2}\cos^{2}\theta_x+\sin^{2}\theta_x}}
= \frac{\eta}{\sqrt{\cos^{2}\theta_\nu+\eta^{2}\sin^{2}\theta_\nu}},
```

from ``\tan\theta_x = \eta\tan\theta_y`` and ``\eta\tan\theta_\nu = \tan\theta_y``.
In terms of the maximum opening ``\underline{\beta}`` at the crack center,
``\underline{N} = \frac{1}{4}\sqrt{\pi\varrho/b}\;\underline{\beta}`` and
``\underline{K} = \sqrt{\pi\varrho/b}\;\hat{\boldsymbol{Q}}^{\star}_{nn}(\underline{\nu})\cdot\underline{\beta}``.

```@raw html
</details>
```

### Ribbon crack

Here ``\underline{\nu} = \pm\underline{m}`` and ``\varrho = 1``:

```math
\underline{N}^{\mathcal{R}}
= \sqrt{\frac{b}{\pi}}\;
  \boldsymbol{B}^{\mathcal{R}}(\underline{m},\underline{n})
  \cdot\boldsymbol{\Sigma}\cdot\underline{n},
\qquad
\underline{K}^{\mathcal{R}}
= \sqrt{\pi b}\;\boldsymbol{\Sigma}\cdot\underline{n}.
```

The SIF of an infinite ribbon crack is **independent of the matrix stiffness** —
the ``\boldsymbol{B}^{\mathcal{R}}`` of the DIF cancels against its inverse in
the exchange relation — and the energy release rate is
``G^{\mathcal{R}} = b\,(\boldsymbol{\Sigma}\cdot\underline{n})\cdot
\boldsymbol{B}^{\mathcal{R}}(\underline{m},\underline{n})\cdot
(\boldsymbol{\Sigma}\cdot\underline{n})``.

### Modes

```math
K_{I} = |\underline{K}\cdot\underline{n}|,
\qquad
K_{II} = |\underline{K}\cdot\underline{\nu}|,
\qquad
K_{III} = |\underline{K}\cdot\underline{\tau}| .
```

Evaluation: [`sif`](@ref) and [`dif`](@ref) (`src/Cracks/sif.jl`).

## 8. Dispatch

| `(crack, C₀)` | `:auto` selects | alternatives | ForwardDiff |
| :------------ | :-------------- | :----------- | :---------: |
| `EllipticCrack` / `RibbonCrack`, `TensISO` | `Analytical` | — | ✓ |
| `EllipticCrack` / `RibbonCrack`, `TensTI` (aligned) | `Analytical` (MFH) | — | ✓ |
| `EllipticCrack` / `RibbonCrack`, `AbstractTens{4,3}` | `NestedQuadGK` | `:residues`, `:decuhr`, `:nestedquadgk` | ✓ |

Entry points: [`cod_tensor`](@ref) / [`B_tensor`](@ref) for ``\boldsymbol{B}``,
[`inverse_property_contribution`](@ref) for ``\mathbb{H}``,
[`delta_inverse_property`](@ref) for ``\Delta\mathbb{S}``, [`sif`](@ref) /
[`dif`](@ref) for the front quantities. The transport counterpart — a scalar COD
and a rank-1 resistivity contribution — is treated in
[Thermal cracks](@ref th-thermal-cracks), with the same geometric factors ``3/4`` and
``2/\pi``.

## Where to go next

The transport counterpart is the next page,
[Thermal cracks](@ref th-thermal-cracks), where a single scalar opening replaces
``\boldsymbol{B}`` and carries all the anisotropy of the matrix. The crack types
and their densities are written as described in [Cracks](@ref man-cracks), and
the tutorial [Cracks and crack density](@ref tut-cracks) places cracks in an
RVE and follows the effective modulus as the density grows.
