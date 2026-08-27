# [Thermal cracks — COD scalar and resistivity contribution](@id th-thermal-cracks)

Transposition of
[Crack opening displacement and compliance tensors](cod_tensors.md) to the
2nd-order (conductivity / diffusion / Darcy) problem, where the driving field
is a vector rather than a symmetric 2-tensor.

## What is specific to a crack

The general elasticity ↔ conductivity dictionary — fields, fluxes, moduli, Hill
and localization tensors — is stated once in
[Extension to conductivity](@ref th-conductivity), together with the
``\boldsymbol\sigma \equiv -\underline{q}`` convention that makes every row of
it sign-free. Only the rows below are proper to the flat-inclusion limit, and
they are the reason this chapter exists at all.

| Elasticity (4-tensor problem)                                    | Conductivity (2-tensor problem)                                         |
| :--------------------------------------------------------------- | :---------------------------------------------------------------------- |
| COD tensor ``\boldsymbol{B}`` — **2-tensor** (6 components)           | COD scalar ``b`` — **scalar** (1 component)                             |
| Kachanov factorization ``\mathbb H = k\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``, ``k=3/4`` (elliptic), ``k=2/\pi`` (ribbon) | Rank-1 factorization ``\boldsymbol{R} = k\,b\,\underline{n}\otimes\underline{n}``, same ``k`` |
| Dilute contribution ``\Delta\mathbb S = (4\pi/3)\varepsilon^{3\mathrm d}\mathbb H`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\mathbb H`` (ribbon) | Dilute contribution ``\Delta\boldsymbol{R} = (4\pi/3)\varepsilon^{3\mathrm d}\boldsymbol{R}`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\boldsymbol{R}`` (ribbon) |
| Sextic acoustic polynomial [masson2008](@cite)                   | Quadratic acoustic form ``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` → **analytical** |
| Stress intensity factors ``K_I, K_{II}, K_{III}``                | Heat-flux intensity factor ``K_T`` — scalar (mode I analog only)      |
| Displacement intensity factor ``\underline{N}``                | Temperature intensity factor — scalar ``[T]_\text{avg}``                |

A scalar ``b`` suffices because ``[T]`` is a scalar and only
``\underline{q}\cdot\underline{n}`` produces a jump — there are no sliding or
shear modes to resolve, unlike the 6-component ``\boldsymbol{B}``. The direction is
always ``\underline{n}``: the null space of
``\boldsymbol{K}_0 - \boldsymbol{K}_0\boldsymbol{P}(0)\boldsymbol{K}_0`` is spanned by
``\underline{n}`` for any ``\boldsymbol{K}_0`` (derivation below), so ``b``
carries all the matrix anisotropy.

## Crack geometry

Same geometric families as in the elasticity chapter:

- **Elliptic cracks** of aspect ratio ``\eta = b/a \in (0, 1]``
  ([`EllipticCrack`](@ref)), including the circular penny ``\eta = 1``
  ([`PennyCrack`](@ref)).
- **Ribbon cracks** (tunnel cracks) of half-width ``b``
  ([`RibbonCrack`](@ref)), infinite along ``\hat{\boldsymbol\ell}``.

## Hill tensor Taylor expansion and block-matrix limit

Mirror of the elasticity derivation of
[barthelemyIJSS2009](@cite).  For a flat ellipsoidal
inclusion of aspect ratio ``\omega\to 0``, the 2nd-order Hill tensor is
computed via the formula
``\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K}_0)
= \boldsymbol{K}_0^{-1/2}\cdot\boldsymbol{I}^{\boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}}
  \cdot\boldsymbol{K}_0^{-1/2}``
([giraudMOM2019](@cite)), where
``\boldsymbol{I}^{\boldsymbol{B}}`` is assembled in the eigenbasis of
``\boldsymbol{B}^T\boldsymbol{B}`` (right singular vectors of
``\boldsymbol{B} = \boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}``).  As ``\omega\to 0``:

- The null vector of ``\boldsymbol{B}^T\boldsymbol{B}`` is
  ``\underline{v}_3 = \boldsymbol{K}_0^{1/2}\underline{n}/\sqrt{k_{nn}}``,
  ``k_{nn}=\underline{n}\cdot\boldsymbol{K}_0\underline{n}``.
- The corresponding Newton potential ``I_3\to 4\pi`` while
  ``I_1, I_2\to 0``, so

```math
\boldsymbol{P}(0) = \boldsymbol{K}_0^{-1/2}\underline{v}_3\otimes\underline{v}_3\boldsymbol{K}_0^{-1/2}
             = \frac{\underline{n}\otimes\underline{n}}{k_{nn}}
```

which is rank-1 along ``\underline{n}``.  The acoustic block is then

```math
\boldsymbol\Lambda(\omega) = \boldsymbol{K}_0 - \boldsymbol{K}_0\boldsymbol{P}(\omega)\boldsymbol{K}_0
                           = \boldsymbol\Lambda(0) + \omega\,\boldsymbol\Lambda_1 + o(\omega),
```

with ``\boldsymbol\Lambda(0) = \boldsymbol{K}_0 - (\boldsymbol{K}_0\underline{n})\otimes(\boldsymbol{K}_0\underline{n})/k_{nn}``.
Its null space is spanned by ``\underline{n}`` (one-dimensional in the
2-tensor case — to be contrasted with the 3-dimensional null space in
the elasticity problem).  The limit

```math
\boldsymbol{R}
= \lim_{\omega\to 0}\omega\,\boldsymbol\Lambda(\omega)^{-1}
= \frac{1}{Y_{nn}}\,\underline{n}\otimes\underline{n},
```

is rank-1 along ``\underline{n}``, with
``Y_{nn} = \underline{n}\cdot\boldsymbol\Lambda_1\cdot\underline{n}``.

## From the Green operator to ``b``

The section above reached ``\boldsymbol{R}`` by flattening an ellipsoid. This one
reaches ``b`` from the Fourier kernel instead — the transport twin of
[From the Green operator to ``\boldsymbol{B}``](cod_tensors.md#From-the-Green-operator-to-B)
— and it is the route that substantiates the row *"Quadratic acoustic form →
analytical"* of the opening table: the order-2 problem has a closed form for
**every** anisotropy, where elasticity needs its sextic to factorize.
`scripts/16_cod_symbolic_thermal.jl` runs the whole chain with the six
components of ``\boldsymbol{K}_0`` as free symbols.

### The reduced kernel

```math
\hat{\boldsymbol{\Gamma}}(\underline{\xi})
  = \frac{\underline{\xi}\otimes\underline{\xi}}
         {\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}},
\qquad
\hat{\boldsymbol{Q}}(\underline{\xi})
  = \boldsymbol{K}_0 - \boldsymbol{K}_0\cdot\hat{\boldsymbol{\Gamma}}\cdot\boldsymbol{K}_0,
\qquad
\hat{Q}^{\star}_{nn}(\underline{\xi}^{\star})
  = \frac{1}{2\pi}\int_{-\infty}^{+\infty}
    \underline{n}\cdot\hat{\boldsymbol{Q}}
    (\underline{\xi}^{\star}+\xi_n\underline{n})\cdot\underline{n}\,\mathrm{d}\xi_n .
```

The acoustic object
``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` is a **scalar**, not
a 3×3 matrix — that single fact is what separates this page from its elastic
sibling.

### The ``\xi_n`` integral is elementary, at any anisotropy

Write the three contractions

```math
a = \underline{n}\cdot\boldsymbol{K}_0\underline{n},
\qquad
b_\xi = \underline{n}\cdot\boldsymbol{K}_0\underline{\xi}^{\star},
\qquad
c = \underline{\xi}^{\star}\cdot\boldsymbol{K}_0\underline{\xi}^{\star},
```

so that ``N = a\xi_n^{2}+2b_\xi\xi_n+c`` and
``\underline{n}\cdot\boldsymbol{K}_0\underline{\xi} = b_\xi + a\xi_n``. Then
**every power of ``\xi_n`` cancels in the numerator**:

```math
a\,N - (b_\xi + a\xi_n)^{2} = ac - b_\xi^{2}
\qquad\Longrightarrow\qquad
\underline{n}\cdot\hat{\boldsymbol{Q}}\cdot\underline{n} = \frac{ac-b_\xi^{2}}{N},
```

a *constant over a quadratic*. Completing the square,
``N = a[(\xi_n+b_\xi/a)^{2}+p^{2}]`` with ``p = \sqrt{ac-b_\xi^{2}}/a > 0``
because ``ac-b_\xi^{2}`` is the Gram determinant of ``\boldsymbol{K}_0``
restricted to ``\mathrm{span}\{\underline{n},\underline{\xi}^{\star}\}`` and
``\boldsymbol{K}_0`` is positive definite. One elementary integral,
``\int\mathrm{d}u/(u^{2}+p^{2}) = \pi/p``, closes it:

```math
\boxed{\;
\hat{Q}^{\star}_{nn}(\underline{\xi}^{\star})
= \tfrac12\sqrt{ac-b_\xi^{2}}
= \tfrac12\sqrt{(\underline{n}\wedge\underline{\xi}^{\star})\cdot
                \mathrm{adj}\,\boldsymbol{K}_0\cdot
                (\underline{n}\wedge\underline{\xi}^{\star})}
\;}
```

the second form following from
``(\underline{u}\cdot\boldsymbol{K}\underline{u})(\underline{v}\cdot\boldsymbol{K}\underline{v})
-(\underline{u}\cdot\boldsymbol{K}\underline{v})^{2}
=(\underline{u}\wedge\underline{v})\cdot\mathrm{adj}\boldsymbol{K}\cdot(\underline{u}\wedge\underline{v})``,
with ``\mathrm{adj}\boldsymbol{K}_0 = \det\boldsymbol{K}_0\,\boldsymbol{K}_0^{-1}``.
So ``\hat{Q}^{\star}_{nn}`` is a norm of
``\underline{n}\wedge\underline{\xi}^{\star}`` in the metric
``\mathrm{adj}\boldsymbol{K}_0``, manifestly homogeneous of degree 1. No Stroh
roots, no residues, no cubature — at any anisotropy. The elastic page's closing
section, *"what breaks in the general case"*, has no counterpart here.

### The crack-plane integral: an effective ellipse

On the contour ``\underline{\xi}^{\star}(\varphi)
= \eta\cos\varphi\,\underline{\ell}+\sin\varphi\,\underline{m}`` one has
``\underline{n}\wedge\underline{\xi}^{\star}
= \eta\cos\varphi\,\underline{m}-\sin\varphi\,\underline{\ell}``, so with
``\underline{v} = (\cos\varphi,\sin\varphi)``

```math
\hat{Q}^{\star}_{nn} = \tfrac12\sqrt{\underline{v}\cdot\boldsymbol{Q}_2\cdot\underline{v}},
\qquad
\boldsymbol{Q}_2 = \begin{pmatrix}
  \eta^{2}\,\underline{m}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{m} &
 -\eta\,\underline{m}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{\ell}\\
 -\eta\,\underline{\ell}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{m} &
       \underline{\ell}\cdot\mathrm{adj}\boldsymbol{K}_0\underline{\ell}
\end{pmatrix}.
```

Diagonalizing this **2×2** form — a quadratic, so closed form — with eigenvalues
``\lambda_1\ge\lambda_2>0`` and rotating to its principal axes turns the integral
into the isotropic one:

```math
\boxed{\;
b\Lambda = \frac14\!\int_0^{2\pi}\!\hat{Q}^{\star}_{nn}\,\mathrm{d}\varphi
= \frac{\sqrt{\lambda_1}}{2}\,\mathcal{E}_{\eta'},
\qquad
\eta' = \sqrt{\lambda_2/\lambda_1},
\qquad
b = \frac{\chi^{\mathcal{E}}}{b\Lambda} = \frac{4}{3\sqrt{\lambda_1}\,\mathcal{E}_{\eta'}}
\;}
```

An arbitrarily anisotropic conductor therefore behaves, around a flat crack, as
an **isotropic** one of conductivity ``\sqrt{\lambda_1}`` containing a crack of
**effective** aspect ratio ``\eta'``. In general ``\eta'\ne\eta``: even a
circular crack acquires an effective ellipticity from the anisotropy
(``\eta' = 0.84`` for ``\eta = 1`` on the sample ``\boldsymbol{K}_0`` of the
script). The specializations:

| ``\boldsymbol{K}_0`` (crack frame) | ``\lambda_1`` | ``\eta'`` | ``b`` |
| :-- | :-- | :-- | :-- |
| isotropic ``k_0\boldsymbol{1}`` | ``k_0^{2}`` | ``\eta`` | ``4/(3k_0\mathcal{E}_\eta)`` |
| TI aligned, ``\mathrm{diag}(k_t,k_t,k_n)`` | ``k_tk_n`` | ``\eta`` | ``4/\bigl(3\sqrt{k_tk_n}\,\mathcal{E}_\eta\bigr)`` |
| any, **ribbon** (``\underline{\xi}^{\star}=\underline{m}``, ``\chi^{\mathcal{R}}=\pi/4``) | — | — | ``\pi/\bigl(2\sqrt{\det\boldsymbol{K}_0\vert_{(\underline{m},\underline{n})}}\bigr)`` |

The aligned-TI entry is the geometric mean ``\sqrt{k_tk_n}`` announced below, and
the ribbon entry involves only the transverse 2×2 block — both structures the
existing formulas already have.

### Relation to the ``\boldsymbol{K}_0^{-1/2}`` route

The two are equivalent, but they cost differently. The square-root transform
needs the eigendecomposition of the 3×3 ``\boldsymbol{K}_0`` *and* the singular
values of ``\boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}``; the adjugate form needs
one **2×2** eigenvalue problem. That is why the adjugate route is evaluable on
symbolic scalars, where `eigen` and `svdvals` are not.

## Closed-form COD scalar ``b``

All of these are the two boxed formulas above, evaluated on a more symmetric
``\boldsymbol{K}_0``. Implementation: `src/Cracks/cod_analytical_thermal.jl`.

### Anisotropic matrix — the general case

``\boldsymbol{K}_0`` arbitrary, ``\eta = b/a``, and
``A,B,C`` the in-plane entries of ``\mathrm{adj}\boldsymbol{K}_0`` defined above:

```math
\boxed{\;
b_{\text{ell}} = \frac{4}{3\sqrt{\lambda_1}\,\mathcal{E}_{\eta'}}
\;},
\qquad
\lambda_{1,2} = \frac{\eta^{2}A+C}{2}
  \pm\sqrt{\Bigl(\frac{\eta^{2}A+C}{2}\Bigr)^{2}-\eta^{2}(AC-B^{2})},
\qquad
\eta' = \sqrt{\lambda_2/\lambda_1}.
```

Being a **2×2** eigenvalue problem, this is closed form for every anisotropy —
no symmetry assumption, and no need for the numerical
``\boldsymbol{K}_0^{-1/2}`` route below.

### Isotropic matrix

``\mathrm{adj}(k_0\boldsymbol{1}) = k_0^{2}\boldsymbol{1}`` gives
``\lambda_1 = k_0^{2}`` and ``\eta' = \eta``:

```math
\boxed{\;
b_{\text{ell}}^{\text{iso}} = \frac{4}{3\,k_0\,\mathcal E_\eta}
\;},
\qquad
\mathcal E_\eta = \mathcal E\!\bigl(\sqrt{1-\eta^{2}}\bigr).
```

Penny limit ``\eta = 1``: ``b = 8/(3\pi k_0)``, which is exactly the surface
average of the textbook jump of an insulating circular crack,
``[\![T]\!](r) = \frac{4\sigma_n a}{\pi k_0}\sqrt{1-r^{2}/a^{2}}``, divided by
``a``. Ribbon: ``b = \pi/(2k_0)``.

### Transversely isotropic matrix aligned with the crack normal

``\boldsymbol{K}_0 = \mathrm{diag}(k_t,k_t,k_n)`` in the crack frame gives
``\lambda_1 = k_tk_n`` and ``\eta' = \eta``, so the effective conductivity is the
**geometric mean**:

```math
b_{\text{ell}}^{\text{aligned TI}} = \frac{4}{3\sqrt{k_tk_n}\,\mathcal E_\eta},
\qquad
b_{\text{penny}}^{\text{aligned TI}} = \frac{8}{3\pi\sqrt{k_tk_n}} .
```

### Ribbon crack — 2D formula

Only the ``(\hat{\underline{m}}, \underline{n})`` transverse block of
``\boldsymbol{K}_0`` enters, since
``\hat{Q}^{\star}_{nn}(\underline{m}) = \tfrac12\sqrt{\det\boldsymbol{K}_0\vert_{(\underline{m},\underline{n})}}``:

```math
\boxed{\;
b_{\text{ribbon}}
= \frac{\pi}{2\,\sqrt{\det\bigl(\boldsymbol{K}_0\vert_{(\hat{\underline{m}},\underline{n})}\bigr)}}
\;}
```

which reduces to ``b = \pi/(2 k_0)`` for an isotropic matrix.

### The ``\boldsymbol{K}_0^{-1/2}`` route

Historically these formulas were obtained by the square-root change of variable
``\tilde{\underline{x}} = \boldsymbol{K}_0^{-1/2}\underline{x}``
([giraudMOM2019](@cite)), which maps the problem to an
isotropic one with a *transformed crack shape*: with
``\tilde{\boldsymbol{A}} = \boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}`` and its
singular values ``\sigma_1\ge\sigma_2\ge\sigma_3 = 0``, the transformed aspect
ratio is ``\eta_t = \sigma_2/\sigma_1``. It is equivalent to the adjugate form —
``\eta_t = \eta'`` — but it needs the eigendecomposition of the 3×3
``\boldsymbol{K}_0`` *and* a singular-value decomposition, so it is `Float64`
only. The package uses the adjugate form, which is why the anisotropic thermal
COD flows through automatic differentiation and symbolic scalars.

!!! warning "These prefactors changed in v0.4.0"
    Up to v0.3.2 the formulas above read ``\eta/(\pi k_0\mathcal E_\eta)`` and
    ``2/(\pi k_0)`` — too small by ``4\pi/(3\eta)`` and ``\pi^{2}/4``. Every
    thermal ``b``, ``\boldsymbol{R}``, ``\Delta\boldsymbol{R}`` and thermal DIF
    therefore changes. The elastic branch is unaffected, and so is
    `fracture_permeability`, whose conduction side does not go through these
    formulas.

## Resistivity contribution ``\boldsymbol{R}`` and dilute correction ``\Delta\boldsymbol{R}``

The size-independent **crack resistivity contribution tensor** is
assembled from the scalar ``b`` and the effective direction
``\hat{\underline{w}}``:

```math
\boldsymbol{R}^{\mathcal E} = \tfrac{3}{4}\,b\,\underline{n}\otimes\underline{n}
\qquad\text{(elliptic)},
\qquad
\boldsymbol{R}^{\mathcal R} = \tfrac{2}{\pi}\,b\,\underline{n}\otimes\underline{n}
\qquad\text{(ribbon)}.
```

The geometric prefactors ``3/4`` and ``2/\pi`` are the same as in the
elasticity case (they come from ``cS/V`` evaluated on the ellipsoidal
and ribbon geometries — see
[Crack compliance and COD tensor](cod_tensors.md#Crack-compliance-H-and-COD-tensor-B)).
The rank-1 direction is always the crack normal ``\underline{n}``,
for any conductivity tensor ``\boldsymbol{K}_0``.

[`compliance_contribution`](@ref)`(crack, K₀)` returns ``\boldsymbol{R}``
directly.  The dilute resistivity correction to the effective
resistivity of the cracked conductor is obtained via
[`delta_resistivity`](@ref)`(crack, R, ε)`:

```math
\Delta\boldsymbol{R} = \tfrac{4\pi}{3}\,\varepsilon^{3\mathrm d}\,\boldsymbol{R}
\qquad\text{(elliptic, }\varepsilon^{3\mathrm d} = Nab^{2}\text{)},
\qquad
\Delta\boldsymbol{R} = \pi\,\varepsilon^{2\mathrm d}\,\boldsymbol{R}
\qquad\text{(ribbon, }\varepsilon^{2\mathrm d} = Nb^{2}\text{)}.
```

These reduce to the Sevostianov–Kachanov expressions
(see [sevostianov2002](@cite),
 [kachanov2018](@cite)).

## Intensity factors

Thermal analogs of the elastic stress / displacement intensity
factors:

Both are driven by ``\boldsymbol\sigma^{\infty} \equiv -\underline{q}^{\infty}``,
the transport twin of the remote stress — the convention of the table above —
so each formula is the elastic one with the symbols substituted.

- **Heat-flux intensity factor** ``K_T`` (mode I analog, scalar):
  the singular crack-tip field scales as ``\sim K_T/\sqrt{r}``.  For a
  ribbon crack of half-width ``b``,
  ``K_T = \sqrt{\pi b}\,(\underline{n}\cdot\boldsymbol\sigma^{\infty})``.
  For an elliptic crack the formula involves the tangent-ribbon COD
  ratio exactly as in the elasticity case (``b^\mathcal E/b^\mathcal R``
  replaces ``\boldsymbol{B}^\mathcal E(\boldsymbol{B}^\mathcal R)^{-1}``).
- **Temperature intensity factor** (scalar):
  ``[T]_\text{avg} = b\,(\underline{n}\cdot\boldsymbol\sigma^{\infty})``.

See [`sif`](@ref) and [`dif`](@ref) for the full signatures (dispatched
on ``\boldsymbol{K}_0::\texttt{AbstractTens\{2,3\}}``).
