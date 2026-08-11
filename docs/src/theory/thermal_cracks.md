# [Thermal cracks — COD scalar and resistivity contribution](@id th-thermal-cracks)

Transposition of
[Crack opening displacement and compliance tensors](cod_tensors.md) to the
2nd-order (conductivity / diffusion / Darcy) problem, where the driving field
is a vector rather than a symmetric 2-tensor.

## Elasticity ↔ Conductivity — correspondence table

The table below is a *literal* dictionary: every entry on the right is the
entry on the left with the symbols substituted, no sign changed. That is only
true because the stress analog is taken to be **minus** the flux,
``\boldsymbol\sigma \equiv -\mathbf q = \mathbf K_0\cdot\nabla T``, which is
the convention fixed in
[Elasticity and transport: one set of formulas](@ref th-notation-sigma-q):
``\boldsymbol\sigma\cdot\hat{\mathbf n}`` is then, in both theories, what the
exterior transmits to the interior across a surface. Read the rows with that
substitution in mind and no minus sign is ever needed.

| Elasticity (4-tensor problem)                                    | Conductivity (2-tensor problem)                                         |
| ---------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Displacement ``\mathbf u`` — 1-tensor                            | Temperature ``T`` — scalar                                              |
| Stress ``\boldsymbol\sigma = \mathbb C:\boldsymbol\varepsilon``  | ``\boldsymbol\sigma \equiv -\mathbf q = \mathbf K_0\cdot\nabla T`` (see below)|
| Stiffness ``\mathbb C`` — 4-tensor, 21 independent components    | Conductivity ``\mathbf K_0`` — 2-tensor, 6 independent components       |
| Hill tensor ``\mathbb P`` — 4-tensor                             | Hill tensor ``\mathbf P`` — 2-tensor                                    |
| COD tensor ``\mathbf B`` — **2-tensor** (6 components)           | COD scalar ``b`` — **scalar** (1 component)                             |
| Kachanov factorization ``\mathbb H = k\,\hat{\mathbf n}\stackrel{s}{\otimes}\mathbf B\stackrel{s}{\otimes}\hat{\mathbf n}``, ``k=3/4`` (elliptic), ``k=2/\pi`` (ribbon) | Rank-1 factorization ``\mathbf R = k\,b\,\hat{\mathbf n}\otimes\hat{\mathbf n}``, same ``k`` |
| Dilute contribution ``\Delta\mathbb S = (4\pi/3)\varepsilon^{3\mathrm d}\mathbb H`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\mathbb H`` (ribbon) | Dilute contribution ``\Delta\mathbf R = (4\pi/3)\varepsilon^{3\mathrm d}\mathbf R`` (elliptic), ``= \pi\varepsilon^{2\mathrm d}\mathbf R`` (ribbon) |
| Sextic acoustic polynomial (Masson 2008)                         | Quadratic acoustic form ``\xi\cdot\mathbf K_0\cdot\xi`` → **analytical** |
| Stress intensity factors ``K_I, K_{II}, K_{III}``                | Heat-flux intensity factor ``K_T`` — scalar (mode I analog only)      |
| Displacement intensity factor ``\hat{\mathbf N}``                | Temperature intensity factor — scalar ``[T]_\text{avg}``                |

A scalar ``b`` suffices because ``[T]`` is a scalar and only
``\mathbf q\cdot\hat{\mathbf n}`` produces a jump — there are no sliding or
shear modes to resolve, unlike the 6-component ``\mathbf B``. The direction is
always ``\hat{\mathbf n}``: the null space of
``\mathbf K_0 - \mathbf K_0\mathbf P(0)\mathbf K_0`` is spanned by
``\hat{\mathbf n}`` for any ``\mathbf K_0`` (derivation below), so ``b``
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
``\mathbf P(\mathbf A,\mathbf K_0)
= \mathbf K_0^{-1/2}\cdot\mathbf I^{\mathbf A\cdot\mathbf K_0^{-1/2}}
  \cdot\mathbf K_0^{-1/2}``
([Giraud et al. 2019](@cite giraudMOM2019)), where
``\mathbf I^{\mathbf B}`` is assembled in the eigenbasis of
``\mathbf B^T\mathbf B`` (right singular vectors of
``\mathbf B = \mathbf A\cdot\mathbf K_0^{-1/2}``).  As ``\omega\to 0``:

- The null vector of ``\mathbf B^T\mathbf B`` is
  ``\mathbf v_3 = \mathbf K_0^{1/2}\hat{\mathbf n}/\sqrt{k_{nn}}``,
  ``k_{nn}=\hat{\mathbf n}\cdot\mathbf K_0\hat{\mathbf n}``.
- The corresponding Newton potential ``I_3\to 4\pi`` while
  ``I_1, I_2\to 0``, so

```math
\mathbf P(0) = \mathbf K_0^{-1/2}\mathbf v_3\otimes\mathbf v_3\mathbf K_0^{-1/2}
             = \frac{\hat{\mathbf n}\otimes\hat{\mathbf n}}{k_{nn}}
```

which is rank-1 along ``\hat{\mathbf n}``.  The acoustic block is then

```math
\boldsymbol\Lambda(\omega) = \mathbf K_0 - \mathbf K_0\mathbf P(\omega)\mathbf K_0
                           = \boldsymbol\Lambda(0) + \omega\,\boldsymbol\Lambda_1 + o(\omega),
```

with ``\boldsymbol\Lambda(0) = \mathbf K_0 - (\mathbf K_0\hat{\mathbf n})\otimes(\mathbf K_0\hat{\mathbf n})/k_{nn}``.
Its null space is spanned by ``\hat{\mathbf n}`` (one-dimensional in the
2-tensor case — to be contrasted with the 3-dimensional null space in
the elasticity problem).  The limit

```math
\mathbf R
= \lim_{\omega\to 0}\omega\,\boldsymbol\Lambda(\omega)^{-1}
= \frac{1}{Y_{nn}}\,\hat{\mathbf n}\otimes\hat{\mathbf n},
```

is rank-1 along ``\hat{\mathbf n}``, with
``Y_{nn} = \hat{\mathbf n}\cdot\boldsymbol\Lambda_1\cdot\hat{\mathbf n}``.

## Closed-form COD scalar ``b``

### Isotropic matrix

For ``\mathbf K_0 = k_0\,\mathbf 1`` the square-root transform is
trivial and ``\hat{\mathbf w} = \hat{\mathbf n}``:

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
``\tilde{\mathbf x} = \mathbf K_0^{-1/2}\mathbf x`` reduces the problem
to the isotropic one with a **transformed crack shape**.  Let
``\tilde{\mathbf A} = \mathbf A\cdot\mathbf K_0^{-1/2}`` with
``\mathbf A = \mathbf R_c\,\mathrm{diag}(a,b,0)\,\mathbf R_c^{T}``, and
denote its singular values ``\sigma_1 \ge \sigma_2 \ge \sigma_3 = 0``.
The transformed in-plane aspect ratio is
``\eta_t = \sigma_2/\sigma_1 \in (0,1]`` and the transformed ellipse
first-kind integral parameter is ``k_t^{2} = 1 - \eta_t^{2}``.  Then

```math
\boxed{\;
b_{\text{ell}}^{\text{aniso}}
= \frac{\sigma_2}
       {\pi\,a_\text{max}\,\sqrt{\hat{\mathbf n}\cdot\mathbf K_0\hat{\mathbf n}}\,\mathcal E_{\eta_t}}
\;},
\qquad
a_\text{max} = \max(a,b).
```

It reduces to the isotropic formula above when
``\mathbf K_0 = k_0\,\mathbf 1`` (then ``\sigma_2 = \min(a,b)/\sqrt{k_0}``
and ``\hat{\mathbf n}\cdot\mathbf K_0\hat{\mathbf n} = k_0``).
For a TI matrix aligned with ``\hat{\mathbf n}``
(``\mathbf K_0 = \mathrm{diag}(k_t, k_t, k_n)`` in the crack frame, penny):

```math
b_{\text{penny}}^{\text{aligned TI}}
= \frac{2}{\pi^{2}\,\sqrt{k_t k_n}}
\qquad\text{(geometric mean of }k_t\text{ and }k_n\text{)}.
```

### Ribbon crack — 2D formula

Only the ``(\hat{\mathbf m}, \hat{\mathbf n})`` transverse block of
``\mathbf K_0`` enters the formula:

```math
\boxed{\;
b_{\text{ribbon}}^{\text{aniso}}
= \frac{2}{\pi\,\sqrt{\det\bigl(\mathbf K_0|_{(\hat{\mathbf m},\hat{\mathbf n})}\bigr)}}
\;}
```

which reduces to ``b = 2/(\pi k_0)`` for an isotropic matrix.

## Resistivity contribution ``\mathbf R`` and dilute correction ``\Delta\mathbf R``

The size-independent **crack resistivity contribution tensor** is
assembled from the scalar ``b`` and the effective direction
``\hat{\mathbf w}``:

```math
\mathbf R^{\mathcal E} = \tfrac{3}{4}\,b\,\hat{\mathbf n}\otimes\hat{\mathbf n}
\qquad\text{(elliptic)},
\qquad
\mathbf R^{\mathcal R} = \tfrac{2}{\pi}\,b\,\hat{\mathbf n}\otimes\hat{\mathbf n}
\qquad\text{(ribbon)}.
```

The geometric prefactors ``3/4`` and ``2/\pi`` are the same as in the
elasticity case (they come from ``cS/V`` evaluated on the ellipsoidal
and ribbon geometries — see
[Crack compliance and COD tensor](cod_tensors.md#Crack-compliance-H-and-COD-tensor-B)).
The rank-1 direction is always the crack normal ``\hat{\mathbf n}``,
for any conductivity tensor ``\mathbf K_0``.

[`compliance_contribution`](@ref)`(crack, K₀)` returns ``\mathbf R``
directly.  The dilute resistivity correction to the effective
resistivity of the cracked conductor is obtained via
[`delta_resistivity`](@ref)`(crack, R, ε)`:

```math
\Delta\mathbf R = \tfrac{4\pi}{3}\,\varepsilon^{3\mathrm d}\,\mathbf R
\qquad\text{(elliptic, }\varepsilon^{3\mathrm d} = Nab^{2}\text{)},
\qquad
\Delta\mathbf R = \pi\,\varepsilon^{2\mathrm d}\,\mathbf R
\qquad\text{(ribbon, }\varepsilon^{2\mathrm d} = Nb^{2}\text{)}.
```

These reduce to the Sevostianov–Kachanov expressions
(see [Sevostianov & Kachanov (2002)](@cite sevostianov2002),
 [Kachanov (2018)](@cite kachanov2018)).

## Intensity factors

Thermal analogs of the elastic stress / displacement intensity
factors:

Both are driven by ``\boldsymbol\sigma^{\infty} \equiv -\mathbf q^{\infty}``,
the transport twin of the remote stress — the convention of the table above —
so each formula is the elastic one with the symbols substituted.

- **Heat-flux intensity factor** ``K_T`` (mode I analog, scalar):
  the singular crack-tip field scales as ``\sim K_T/\sqrt{r}``.  For a
  ribbon crack of half-width ``b``,
  ``K_T = \sqrt{\pi b}\,(\hat{\mathbf n}\cdot\boldsymbol\sigma^{\infty})``.
  For an elliptic crack the formula involves the tangent-ribbon COD
  ratio exactly as in the elasticity case (``b^\mathcal E/b^\mathcal R``
  replaces ``\mathbf B^\mathcal E(\mathbf B^\mathcal R)^{-1}``).
- **Temperature intensity factor** (scalar):
  ``[T]_\text{avg} = b\,(\hat{\mathbf n}\cdot\boldsymbol\sigma^{\infty})``.

See [`sif`](@ref) and [`dif`](@ref) for the full signatures (dispatched
on ``\mathbf K_0::\texttt{AbstractTens\{2,3\}}``).
