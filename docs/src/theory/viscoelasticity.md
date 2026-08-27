# [Ageing linear viscoelasticity (ALV)](@id th-viscoelasticity)

!!! tip "If the material does not age, there is a cheaper route"
    Everything on this page treats a kernel ``\mathbb{R}(t, t')`` in which `t`
    and `t'` enter **independently**, which is what forces the Volterra
    operators to be discretized directly. When the kernel depends on `t - t'`
    alone the constitutive law is a convolution, a transform turns it into a
    product, and the whole problem reduces to an elastic one at each Carson
    variable — see [the Laplace-Carson route](@ref th-laplace-carson).

    The two algebras are worth contrasting explicitly. Creep and relaxation are
    related by ``\int_0^t \mathbb{R}(t-s):\mathrm{d}\mathbb{J}(s) =
    \mathbb{I}`` here — a convolution, discretized below into the inverse of a
    block-triangular matrix — and by the pointwise reciprocal
    ``\mathbb{J}^{*}(p) : \mathbb{R}^{*}(p) = \mathbb{I}`` there. That single
    difference is the whole cost and the whole benefit of each route.

Everything in the elastic part of this documentation — the Eshelby problem, the
Hill polarization tensor, the concentration and contribution tensors, the
schemes built on them — carries over to **ageing linear viscoelasticity**
essentially unchanged, provided two substitutions are made systematically:

| Elastic | Ageing viscoelastic |
| :------ | :------------------ |
| tensor ``\mathbb{C}`` | two-time **kernel** ``\mathbb{C}(t,t')`` |
| double contraction ``\mathbb{A}:\mathbb{B}`` | Volterra product ``\mathbb{A}\circ\mathbb{B}`` |
| tensor inverse ``\mathbb{A}^{-1}`` | Volterra inverse ``\mathbb{A}^{-\circ}`` |
| identity ``\mathbb{I}`` | ``H\,\mathbb{I}``, ``H`` the Heaviside function |

This page follows that substitution from the constitutive law to the
homogenization schemes. The derivations are those of [barthelemyIJSS2016](@cite)
(the Eshelby problem and the Hill kernel) and [barthelemyIJES2019](@cite)
(the schemes); the time discretization is that of [sanahuja2013](@cite).

!!! note "Notation used on this page"
    A kernel is a function of two times ``(t, t')`` — observation time and
    loading time — and is written in the same typeface as its elastic
    counterpart. ``\circ`` denotes the Volterra product defined in the next
    section, ``\bullet^{-\circ}`` the Volterra inverse — never the ordinary
    tensor inverse. In `MeanFieldHomogenization` these two operations are
    [`volterra_product`](@ref) and [`volterra_inverse`](@ref).

## The ageing linear viscoelastic behavior

The strain and stress histories are related by a **Stieltjes integral**
([barthelemyIJSS2016](@cite)):

```math
\boldsymbol{\varepsilon}(t)
= \int_{t'=-\infty}^{t}\mathbb{L}(t,t'):\mathrm{d}\boldsymbol{\sigma}(t'),
\qquad
\boldsymbol{\sigma}(t)
= \int_{t'=-\infty}^{t}\mathbb{C}(t,t'):\mathrm{d}\boldsymbol{\varepsilon}(t'),
```

with ``\mathbb{L}`` the creep compliance kernel and ``\mathbb{C}`` the
relaxation kernel. Causality imposes ``\mathbb{C}(t,t') = 0`` for ``t < t'``.

The **non-ageing** case is the special one where the kernels depend on ``t`` and
``t'`` only through their difference ``t-t'``. There, and only there, the
Laplace–Carson correspondence principle applies and the problem reduces to an
elastic one with complex moduli — the route taken in
[Viscoelastic composites](../tutorials/viscoelasticity.md) and cross-checked
against the present one in
[Frequency or time?](../tutorials/generated/freq_vs_time.md). When a phase
*ages* — its properties evolve with its own maturity, as a hydrating cement
paste does — ``t`` and ``t'`` enter independently, there is no convolution to
transform, and the time domain is the only available route.

### Volterra algebra

The relation above is written compactly ``\boldsymbol{\varepsilon} =
\mathbb{L}\circ\boldsymbol{\sigma}``, extending to tensors the scalar Volterra
operator ([barthelemyIJSS2016](@cite)). Between two kernels the same
symbol denotes

```math
(\mathbb{A}\circ\mathbb{B})(t,t')
= \int_{\tau=-\infty}^{t}\mathbb{A}(t,\tau):
  \frac{\partial\mathbb{B}}{\partial\tau}(\tau,t')\,\mathrm{d}\tau .
```

This product is associative and distributive over addition, but **not
commutative**: commutativity holds only for non-ageing kernels
([barthelemyIJSS2016](@cite), citing Maghous & Creus). Every formula below
therefore keeps its factors in order, including the apparently scalar ones.

The Heaviside function acts as the identity, ``H\circ X = X``, so the identity
elements are ``H\,\mathbb{I}`` for 4-tensor kernels and ``H\,\boldsymbol{1}``
for 2-tensor kernels. The **Volterra inverse** ``\mathbb{A}^{-\circ}`` is
defined by ``\mathbb{A}\circ\mathbb{A}^{-\circ} =
\mathbb{A}^{-\circ}\circ\mathbb{A} = H\,\mathbb{I}``; relaxation and creep
kernels are Volterra inverses of one another, ``\mathbb{C} =
\mathbb{L}^{-\circ}``.

### Discretization: kernels become block matrices

On a time grid ``t_0 < t_1 < \dots < t_n`` the Stieltjes integral is
approximated by the trapezoidal rule of [sanahuja2013](@cite). Strain and stress
histories become block column vectors, and each kernel a **lower
block-triangular** matrix ``\widetilde{\mathbb{C}}`` of size
``6(n+1)\times 6(n+1)`` whose blocks are

```math
\bigl[\widetilde{\mathbb{C}}\bigr]_{ik} =
\begin{cases}
\mathbb{C}(t_0,t_0), & i = k = 0,\\[4pt]
\tfrac{1}{2}\bigl[\mathbb{C}(t_i,t_{i-1}) + \mathbb{C}(t_i,t_i)\bigr], & i = k > 0,\\[4pt]
\tfrac{1}{2}\bigl[\mathbb{C}(t_i,t_0) - \mathbb{C}(t_i,t_1)\bigr], & i > 0,\ k = 0,\\[4pt]
\tfrac{1}{2}\bigl[\mathbb{C}(t_i,t_{k-1}) - \mathbb{C}(t_i,t_{k+1})\bigr], & i > k > 0.
\end{cases}
```

!!! warning "A block is not a kernel value"
    ``[\widetilde{\mathbb{C}}]_{ik}`` is a *weighted combination* of kernel
    values at neighboring times, not ``\mathbb{C}(t_i,t_k)``. Reading a
    relaxation function off a column of the matrix therefore gives the wrong
    answer; the physical extraction is to apply the matrix to a unit strain
    step, which for a step at ``t_0`` amounts to summing each block row. The
    lower-triangular structure encodes causality.

The pay-off of this representation is that the Volterra product becomes an
ordinary matrix product and the Volterra inverse an ordinary matrix inverse
([sanahuja2013](@cite); [barthelemyIJES2019](@cite), Appendix). In
`MeanFieldHomogenization` the discretization is [`trapezoidal_matrix`](@ref) and the
inverse [`volterra_inverse`](@ref).

## The Eshelby problem in ALV

Consider an ellipsoid ``\mathcal{E}`` of shape tensor ``\boldsymbol{A}``
embedded in an infinite medium of relaxation kernel ``\mathbb{C}``, carrying a
uniform polarization history ``\boldsymbol{p}(t)``:

```math
\boldsymbol{\sigma}(\underline{x})
= \mathbb{C}\circ\boldsymbol{\varepsilon}(\underline{x})
+ \boldsymbol{p}\,\chi_{\mathcal{E}}(\underline{x}),
```

with ``\chi_{\mathcal{E}}`` the characteristic function of ``\mathcal{E}``.
Momentum balance and the decay condition at infinity give
([barthelemyIJSS2016](@cite))

```math
\mathrm{div}\bigl(\mathbb{C}\circ\boldsymbol{\varepsilon}(\underline{u})\bigr)
- \boldsymbol{p}\cdot\underline{n}\,\delta_{\partial\mathcal{E}} = \underline{0},
\qquad
\lim_{\|\underline{x}\|\to\infty}\underline{u}(\underline{x}) = \underline{0},
```

where ``\delta_{\partial\mathcal{E}}`` is the surface Dirac distribution on the
boundary of ``\mathcal{E}`` and ``\underline{n}`` its outward normal. **This is
literally the elastic problem** with ``:`` replaced by ``\circ``.

Solving it through the ALV Green kernel yields the central result: the strain is
**uniform inside the ellipsoid**, exactly as in elasticity, and

```math
\forall\,\underline{x}\in\mathcal{E}\quad
\boldsymbol{\varepsilon}(\underline{x}) = -\,\mathbb{P}\circ\boldsymbol{p},
\qquad
\boldsymbol{\sigma}(\underline{x}) = -\,\mathbb{C}\circ\mathbb{P}\circ\boldsymbol{p}.
```

## The Hill polarization kernel

The kernel ``\mathbb{P}`` appearing above is the ALV counterpart of the elastic
Hill polarization tensor ([barthelemyIJSS2016](@cite)):

```math
\mathbb{P}
= \frac{\det\boldsymbol{A}}{4\pi}
  \int_{\|\underline{\xi}\|=1}
  \frac{\underline{\xi}\stackrel{s}{\otimes}
        (\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi})^{-\circ}
        \stackrel{s}{\otimes}\underline{\xi}}
       {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,
  \mathrm{d}S_{\underline{\xi}} .
```

It differs from the [elastic Hill tensor](hill_tensors.md) in exactly one place:
the inverse of the acoustic tensor
``\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}`` is a **Volterra** inverse.
Like its elastic counterpart it depends only on the shape and orientation of the
ellipsoid and on the reference kernel.

The **Eshelby kernel** follows by the same definition as in elasticity,
``\mathbb{S} = \mathbb{P}\circ\mathbb{C}``, and relates the uniform strain inside
``\mathcal{E}`` to a uniform eigenstrain history ``\boldsymbol{\varepsilon}^*``:

```math
\forall\,\underline{x}\in\mathcal{E}\quad
\boldsymbol{\varepsilon}(\underline{x})
= \mathbb{S}\circ\boldsymbol{\varepsilon}^*
= \mathbb{P}\circ\mathbb{C}\circ\boldsymbol{\varepsilon}^* .
```

### Isotropic matrix: time and space decouple

If the reference kernel is isotropic,
``\mathbb{C}(t,t') = 3k(t,t')\,\mathbb{J} + 2\mu(t,t')\,\mathbb{K}``, the
acoustic tensor is diagonal in the
``(\underline{\xi}\otimes\underline{\xi},\
\boldsymbol{1}-\underline{\xi}\otimes\underline{\xi})`` decomposition and can be
inverted in the Volterra sense analytically. The Hill kernel then **factorizes**
([barthelemyIJSS2016](@cite), [barthelemyIJES2019](@cite)):

```math
\mathbb{P}
= \bigl(k + \tfrac{4}{3}\mu\bigr)^{-\circ}\,\mathbb{U}^{\boldsymbol{A}}
+ \mu^{-\circ}\,
  \bigl(\mathbb{V}^{\boldsymbol{A}} - \mathbb{U}^{\boldsymbol{A}}\bigr),
```

where ``\mathbb{U}^{\boldsymbol{A}}`` and ``\mathbb{V}^{\boldsymbol{A}}`` are the
purely **geometric** tensors of the elastic theory
([Hill polarization tensors](hill_tensors.md)):

```math
\mathbb{U}^{\boldsymbol{A}} = \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\otimes\underline{\xi}\otimes\underline{\xi}\otimes\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\underline{\xi}},
\qquad
\mathbb{V}^{\boldsymbol{A}} = \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\stackrel{s}{\otimes}\boldsymbol{1}
      \stackrel{s}{\otimes}\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\underline{\xi}} .
```

This is what makes ALV tractable: the geometry sits entirely in
``\mathbb{U}^{\boldsymbol{A}}`` and ``\mathbb{V}^{\boldsymbol{A}}`` — **the
elastic ones**, computed once per shape — and the time dependence in two
*scalar* Volterra inverses, so a ``6n \times 6n`` tensor-kernel inversion
becomes two ``n \times n`` scalar ones. [`hill_kernel`](@ref) discretizes the
matrix law, extracts ``(3k, 2\mu)``, inverts ``k+\tfrac{4}{3}\mu`` and ``\mu``
in the Volterra sense, and assembles against [`tens_UA`](@ref) and
[`tens_VA`](@ref). An anisotropic reference kernel has no such fast path.

For a **sphere** the geometric tensors are
``\mathbb{U}^{\boldsymbol{A}} = \tfrac{1}{3}\mathbb{J} + \tfrac{2}{15}\mathbb{K}``
and ``\mathbb{V}^{\boldsymbol{A}} = \tfrac{1}{3}\mathbb{I}``, giving the closed
form ([barthelemyIJSS2016](@cite))

```math
\mathbb{P}^{\text{sphere}}
= (3k+4\mu)^{-\circ}\circ
  \Bigl(H\,\mathbb{J}
      + \tfrac{3}{5}\,(k+2\mu)\circ\mu^{-\circ}\,\mathbb{K}\Bigr),
\qquad
\mathbb{S}^{\text{sphere}}
= 3\,(3k+4\mu)^{-\circ}\circ
  \Bigl(k\,\mathbb{J} + \tfrac{2}{5}(k+2\mu)\,\mathbb{K}\Bigr).
```

An anisotropic reference kernel does not enjoy this decoupling; as in
elasticity, that is the case where the surface integral must be evaluated
numerically.

## From the inclusion to the inhomogeneity

Replace the polarization by a genuine inhomogeneity: the ellipsoid now has its
own relaxation kernel ``\mathbb{C}^{\mathcal{E}}``, and the medium is loaded by
a remote strain history ``\boldsymbol{E}(t)``. Superposing the remote field and
the response to the fictitious polarization
``(\mathbb{C}^{\mathcal{E}}-\mathbb{C}^{0})\circ\boldsymbol{\varepsilon}``
gives ([barthelemyIJSS2016](@cite))

```math
\boldsymbol{\varepsilon}
= \boldsymbol{E}
- \mathbb{P}\circ(\mathbb{C}^{\mathcal{E}}-\mathbb{C}^{0})
  \circ\boldsymbol{\varepsilon}
\qquad\Longrightarrow\qquad
\boldsymbol{\varepsilon} = \mathbb{A}^{\text{dil}}\circ\boldsymbol{E},
```

with the **dilute strain concentration kernel**

```math
\mathbb{A}^{\text{dil}}
= \bigl(H\,\mathbb{I}
      + \mathbb{P}\circ(\mathbb{C}^{\mathcal{E}}-\mathbb{C}^{0})\bigr)^{-\circ} .
```

The strain remains uniform inside ``\mathcal{E}``: it depends on time alone. The
associated **contribution kernel** is

```math
\mathbb{N}
= (\mathbb{C}^{\mathcal{E}}-\mathbb{C}^{0})\circ\mathbb{A}^{\text{dil}}
= \bigl(\mathbb{P}
      + (\mathbb{C}^{\mathcal{E}}-\mathbb{C}^{0})^{-\circ}\bigr)^{-\circ},
```

the second form following from the identity
``\mathbb{X}\circ(H\mathbb{I}+\mathbb{Y}\circ\mathbb{X})^{-\circ}
= (H\mathbb{I}+\mathbb{X}\circ\mathbb{Y})^{-\circ}\circ\mathbb{X}``, which holds
in any associative algebra and so survives the loss of commutativity.

## Schemes

With concentration kernels in hand, every matrix-based scheme transposes
term by term ([barthelemyIJES2019](@cite)). Writing ``\varphi_r`` for the volume
fraction of phase ``r`` and ``\mathbb{C}^0`` for the reference kernel, the
general form is ``\mathbb{C}^{\hom} = \langle\mathbb{C}\circ\mathbb{A}\rangle``,
or equivalently

```math
\mathbb{C}^{\hom}
= \mathbb{C}^{0}
+ \sum_r \varphi_r\,(\mathbb{C}^{r}-\mathbb{C}^{0})\circ
  \langle\mathbb{A}\rangle_r ,
```

and the schemes differ only in how ``\langle\mathbb{A}\rangle_r`` is estimated —
which yields, after substitution:

| Scheme | Effective kernel |
| :----- | :--------------- |
| **Dilute** / NIA | ``\mathbb{C}^{\hom} = \mathbb{C}^{0} + \sum_r \varphi_r\,\mathbb{N}^{r}`` |
| **Mori-Tanaka** | ``\mathbb{C}^{\hom} = \mathbb{C}^{0} + \bigl(\sum_r \varphi_r\,\mathbb{N}^{r}\bigr)\circ\bigl((1-\sum_s\varphi_s)H\,\mathbb{I} + \sum_s\varphi_s\,\mathbb{A}^{\text{dil},s}\bigr)^{-\circ}`` |
| **Maxwell** / PCW | ``(\mathbb{C}^{\hom})^{-\circ} = (\mathbb{C}^{0})^{-\circ} + \bigl((\sum_r \varphi_r\,\mathbb{N}^{r})^{-\circ} - \mathbb{P}_{\Omega}\bigr)^{-\circ}`` |
| **Self-consistent** | the same equations with ``\mathbb{C}^{0} = \mathbb{C}^{\hom}``, solved iteratively |
| **Differential** | inclusions added in infinitesimal increments, re-homogenizing at each step |

``\mathbb{P}_{\Omega}`` in the Maxwell row is the Hill kernel of the
**distribution shape** ``\Omega``, not of an inclusion.

The one new difficulty is bookkeeping: ``\circ`` does not commute, so the
order of the factors is prescribed — even in the isotropic case, where every
factor looks scalar. All ten schemes are implemented by
[`homogenize_alv`](@ref).

## The n-layer composite sphere

The Hervé–Zaoui ``n``-layer sphere ([herve1993](@cite), see
[Layered spheres](layered_sphere.md)) transposes by the same rule. Its elastic
construction propagates a state vector across the shells by a product of
transfer matrices — ``2\times 2`` for the bulk (``Y_0``) harmonic,
``4\times 4`` for the shear (``Y_2``) one — whose entries are rational
expressions in the scalar moduli ``(\kappa_i,\mu_i)`` of each layer and in the
layer radii.

In the ALV setting every scalar modulus becomes its ``n\times n`` trapezoidal
Volterra matrix, every product a matrix product and every reciprocal a Volterra
inverse; the transfer matrices become block matrices of size ``2n\times 2n`` and
``4n\times 4n`` respectively. Both harmonics carry over, together with the
per-layer localization kernels and the imperfect-interface transfers. The
composite sphere then enters the schemes exactly as in elasticity — through its
volume-averaged concentration kernel, having no Hill tensor of its own.

## [Symmetry classes and structured storage](@id th-visco-classes)

ALV operators inherit the symmetry classes of their elastic counterparts, and
those classes are **closed** under Volterra product and inverse. That closure is
what makes compact storage possible:

| Class | Stored components | Full ``6n\times 6n`` | Storage cost | Closure operation |
|:------|:------------------|:---------------------|:-------------|:------------------|
| ISO | ``(\alpha,\beta)`` | ``36n^2`` | **``2n^2``** (18×) | scalar Volterra products / inverses |
| TI | ``(\ell_1,\dots,\ell_6)`` | ``36n^2`` | **``6n^2``** (6×) | ``(2n\times 2n)`` block-Volterra + 2 scalars |
| ORTHO | ``(o_1,\dots,o_{12})`` | ``36n^2`` | **``12n^2``** (3×) | ``(3n\times 3n)`` block-Volterra + 3 scalars |
| Generic | full ``6n\times 6n`` | ``36n^2`` | ``36n^2`` | ``(6n\times 6n)`` block-LU |

with ISO ⊂ TI ⊂ ORTHO ⊂ generic. The types [`ALVKernelISO`](@ref),
[`ALVKernelTI`](@ref) and [`ALVKernelOrtho`](@ref) wrap these compact
representations as `AbstractMatrix`, so they flow through generic Julia matrix
code while preserving both the storage saving and the algebraic closure.

