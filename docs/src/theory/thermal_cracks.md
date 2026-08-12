# [Thermal cracks — COD scalar and resistivity contribution](@id th-thermal-cracks)

Transposition of
[Crack opening displacement and compliance tensors](cod_tensors.md) to the
2nd-order (conductivity / diffusion / Darcy) problem, where the driving field
is a vector rather than a symmetric 2-tensor.

## Elasticity ↔ Conductivity — correspondence table

The table below is a *literal* dictionary: every entry on the right is the
entry on the left with the symbols substituted, no sign changed. That is only
true because the stress analog is taken to be **minus** the flux,
``\boldsymbol\sigma \equiv -\underline{q} = \boldsymbol{K}_0\cdot\nabla T``, which is
the convention fixed in
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q):
``\boldsymbol\sigma\cdot\underline{n}`` is then, in both theories, what the
exterior transmits to the interior across a surface. Read the rows with that
substitution in mind and no minus sign is ever needed.

| Elasticity (4-tensor problem)                                    | Conductivity (2-tensor problem)                                         |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Displacement ``\underline{u}`` — 1-tensor                            | Temperature ``T`` — scalar                                              |
| Stress ``\boldsymbol\sigma = \mathbb C:\boldsymbol\varepsilon``  | ``\boldsymbol\sigma \equiv -\underline{q} = \boldsymbol{K}_0\cdot\nabla T`` (see below)|
| Stiffness ``\mathbb C`` — 4-tensor, 21 independent components    | Conductivity ``\boldsymbol{K}_0`` — 2-tensor, 6 independent components       |
| Hill tensor ``\mathbb P`` — 4-tensor                             | Hill tensor ``\boldsymbol{P}`` — 2-tensor                                    |
| COD tensor ``\boldsymbol{B}`` — **2-tensor** (6 components)           | COD scalar ``b`` — **scalar** (1 component)                             |
| Kachanov factorization ``\mathbb H = k\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}``, ``k=3/4`` (elliptic), ``k=2/\pi`` (ribbon) | Rank-1 factorization ``\boldsymbol{R} = k\,b\,\underline{n}\otimes\underline{n}``, same ``k`` |
| Dilute contribution ``\Delta\mathbb S = (4\pi/3)\varepsilon^{3\mathrm d}\mathbb H`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\mathbb H`` (ribbon) | Dilute contribution ``\Delta\boldsymbol{R} = (4\pi/3)\varepsilon^{3\mathrm d}\boldsymbol{R}`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\boldsymbol{R}`` (ribbon) |
| Sextic acoustic polynomial (Masson 2008)                         | Quadratic acoustic form ``\underline{\xi}\cdot\boldsymbol{K}_0\cdot\underline{\xi}`` → **analytical** |
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
[Barthélémy (2009)](@cite barthelemyIJSS2009).  For a flat ellipsoidal
inclusion of aspect ratio ``\omega\to 0``, the 2nd-order Hill tensor is
computed via the formula
``\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K}_0)
= \boldsymbol{K}_0^{-1/2}\cdot\boldsymbol{I}^{\boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}}
  \cdot\boldsymbol{K}_0^{-1/2}``
([Giraud et al. 2019](@cite giraudMOM2019)), where
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

## Closed-form COD scalar ``b``

### Isotropic matrix

For ``\boldsymbol{K}_0 = k_0\,\boldsymbol{1}`` the square-root transform is
trivial and ``\hat{\underline{w}} = \underline{n}``:

```math
\boxed{\;
b_{\text{ell}}^{\text{iso}}
= \frac{\eta}{\pi\,k_0\,\mathcal E_\eta}
\;},
\qquad
\mathcal E_\eta = \mathcal E\!\bigl(\sqrt{1-\eta^{2}}\bigr),
```

with ``\mathcal E`` the complete elliptic integral of the second kind.
Penny limit ``\eta = 1``: ``b = 2/(\pi^{2} k_0)``.  Ribbon:
``b = 2/(\pi k_0)`` (direct 2-D computation, not the ``\eta\to 0``
limit of the elliptic formula).

### Anisotropic matrix (K⁻¹ᐟ² transform)

Applying the square-root change of variable
``\tilde{\underline{x}} = \boldsymbol{K}_0^{-1/2}\underline{x}`` reduces the problem
to the isotropic one with a **transformed crack shape**.  Let
``\tilde{\boldsymbol{A}} = \boldsymbol{A}\cdot\boldsymbol{K}_0^{-1/2}`` with
``\boldsymbol{A} = \boldsymbol{R}_c\,\mathrm{diag}(a,b,0)\,\boldsymbol{R}_c^{T}``, and
denote its singular values ``\sigma_1 \ge \sigma_2 \ge \sigma_3 = 0``.
The transformed in-plane aspect ratio is
``\eta_t = \sigma_2/\sigma_1 \in (0,1]`` and the transformed ellipse
first-kind integral parameter is ``k_t^{2} = 1 - \eta_t^{2}``.  Then

```math
\boxed{\;
b_{\text{ell}}^{\text{aniso}}
= \frac{\sigma_2}
       {\pi\,a_\text{max}\,\sqrt{\underline{n}\cdot\boldsymbol{K}_0\underline{n}}\,\mathcal E_{\eta_t}}
\;},
\qquad
a_\text{max} = \max(a,b).
```

It reduces to the isotropic formula above when
``\boldsymbol{K}_0 = k_0\,\boldsymbol{1}`` (then ``\sigma_2 = \min(a,b)/\sqrt{k_0}``
and ``\underline{n}\cdot\boldsymbol{K}_0\underline{n} = k_0``).
For a TI matrix aligned with ``\underline{n}``
(``\boldsymbol{K}_0 = \mathrm{diag}(k_t, k_t, k_n)`` in the crack frame, penny):

```math
b_{\text{penny}}^{\text{aligned TI}}
= \frac{2}{\pi^{2}\,\sqrt{k_t k_n}}
\qquad\text{(geometric mean of }k_t\text{ and }k_n\text{)}.
```

### Ribbon crack — 2D formula

Only the ``(\hat{\underline{m}}, \underline{n})`` transverse block of
``\boldsymbol{K}_0`` enters the formula:

```math
\boxed{\;
b_{\text{ribbon}}^{\text{aniso}}
= \frac{2}{\pi\,\sqrt{\det\bigl(\boldsymbol{K}_0|_{(\hat{\underline{m}},\underline{n})}\bigr)}}
\;}
```

which reduces to ``b = 2/(\pi k_0)`` for an isotropic matrix.

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
(see [Sevostianov & Kachanov (2002)](@cite sevostianov2002),
 [Kachanov (2018)](@cite kachanov2018)).

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
