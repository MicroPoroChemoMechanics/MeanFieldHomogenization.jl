# [Crack opening displacement and compliance](@id th-cod-tensors)

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
*Conventions* section below.

## Geometry

A flat crack is the limit of a flat spheroidal inclusion. Keeping the shape
tensor of [Notation](notation.md#Ellipsoid-geometry),

```math
\boldsymbol{A}
= \underline{\ell}\otimes\underline{\ell}
+ \eta\,\underline{m}\otimes\underline{m}
+ \omega\,\underline{n}\otimes\underline{n},
\qquad
\eta = \frac{b}{a},
\qquad
\omega = \frac{c}{a}\to 0,
```

with ``(\underline{\ell},\underline{m})`` the in-plane unit vectors along the
major and minor semi-axes ``a\ge b``, and ``\underline{n}`` the unit normal.

![The crack frame and its semi-axes, before the flat limit is taken (from the Echoes book [echoes](@cite))](../assets/geometry/crack_frame.svg)

Two families matter, and they are **genuinely different objects**, not two
regimes of one:

| | geometry | in-plane aspect ratio | `MeanFieldHomogenization` type |
| :-- | :-- | :-- | :-- |
| **elliptic** (3-D) | bounded ellipse, semi-axes ``a\ge b`` | ``\eta\in(0,1]`` | [`EllipticCrack`](@ref), [`PennyCrack`](@ref) for ``\eta=1`` |
| **ribbon** (2-D) | infinite tunnel along ``\underline{\ell}``, half-width ``b`` | ``a\to\infty``, so ``\eta\to 0`` | [`RibbonCrack`](@ref) |

## The COD tensor ``\boldsymbol{B}``

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
[barthelemySifAniso](@cite), following [kachanov1992](@cite),
[kachanov1993](@cite), and it is the one `MeanFieldHomogenization` implements
([`cod_tensor`](@ref), alias [`B_tensor`](@ref)).

### The shape coefficient ``\chi``

The opening profile is an ellipsoidal cap, so the average jump is a fixed
fraction of the **maximum** jump ``\underline{\beta}`` at the crack center:

```math
\bigl\langle [\![\underline{u}]\!] \bigr\rangle_{\mathcal{I}} = \chi\,\underline{\beta},
\qquad
\chi^{\mathcal{E}}
= \frac{1}{\pi a b}\!\!\int_{\frac{x^2}{a^2}+\frac{y^2}{b^2}\le 1}\!\!
  \sqrt{1-\frac{x^2}{a^2}-\frac{y^2}{b^2}}\;\mathrm{d}x\,\mathrm{d}y
= \frac{2}{3},
\qquad
\chi^{\mathcal{R}}
= \frac{1}{2b}\!\int_{-b}^{b}\!\sqrt{1-\frac{y^2}{b^2}}\;\mathrm{d}y
= \frac{\pi}{4}.
```

``\chi`` is worth introducing explicitly, because **every numerical factor on
this page comes from it**. It is the one quantity that distinguishes the
elliptic geometry from the ribbon one.

### The 3-D → 2-D limit: a trap

A ribbon is the limiting shape of an ellipse as ``\eta\to 0``, so one expects
``\boldsymbol{B}^{\mathcal{R}}`` to be ``\lim_{\eta\to 0}\boldsymbol{B}^{\mathcal{E}}``.
**It is not** [barthelemySifAniso](@cite):

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

## From ``\boldsymbol{B}`` to the compliance ``\mathbb{H}``

The extra strain a crack contributes, per unit volume of the embedding
ellipsoid, is the average of the displacement jump over the crack surface
``S``, spread over the volume ``V``:

```math
\boldsymbol{\varepsilon}^{\text{extra}}
= \frac{1}{V}\int_{S} [\![\underline{u}]\!] \stackrel{s}{\otimes}\underline{n}\,\mathrm{d}S
= \frac{S}{V}\,
  \bigl\langle [\![\underline{u}]\!] \bigr\rangle_{\mathcal{I}}
  \stackrel{s}{\otimes}\underline{n}
= \frac{S\,b}{V}\;
  \underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}
  :\boldsymbol{\Sigma},
```

using the definition of ``\boldsymbol{B}``. Identifying this with
``\mathbb{Q}^{-1}:\boldsymbol{\Sigma}``, where
``\mathbb{Q} = \mathbb{C}-\mathbb{C}:\mathbb{P}:\mathbb{C}`` is the
[second Hill tensor](eshelby_problem.md), gives
``\mathbb{Q}^{-1} = (Sb/V)\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}
\stackrel{s}{\otimes}\underline{n}``, and hence

```math
\boxed{\;
\mathbb{H}
\;=\; \lim_{c/b\,\to\,0}\ \frac{c}{b}\,\mathbb{Q}^{-1}
\;=\; \frac{c\,S}{V}\;
      \underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}.
\;}
```

The limit is finite: the components ``(\mathbb{Q}^{-1})_{nijk}`` diverge like
``b/c``, so ``(c/b)\,\mathbb{Q}^{-1}`` stays bounded. Evaluating the geometric
factor ``cS/V`` for each family gives the two numbers used in the code:

| | surface ``S`` | volume ``V`` | ``cS/V`` | ``\mathbb{H}`` |
| :-- | :-- | :-- | :-- | :-- |
| **elliptic** (3-D) | ``\pi a b`` | ``\tfrac{4}{3}\pi a b c`` | ``\tfrac{3}{4}`` | ``\mathbb{H}^{\mathcal{E}} = \tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}^{\mathcal{E}}\stackrel{s}{\otimes}\underline{n}`` |
| **ribbon** (2-D) | ``4ab`` | ``2\pi a b c`` | ``\tfrac{2}{\pi}`` | ``\mathbb{H}^{\mathcal{R}} = \tfrac{2}{\pi}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}^{\mathcal{R}}\stackrel{s}{\otimes}\underline{n}`` |

Both factors are **independent of ``\eta``**, which is the point of normalizing
the limit by ``b``: one definition covers the 2-D and the 3-D geometry, and
``\mathbb{H}`` inherits ``\boldsymbol{B}``'s size-independence.

The bridge in both directions is [`compliance_from_cod`](@ref) /
[`cod_from_compliance`](@ref) (`src/Cracks/cod_H_bridge.jl`), dispatching on the
crack type to apply the correct factor.

### Conventions: what ``\boldsymbol{B}`` and ``\mathbb{H}`` are normalized by

The ``3/4`` above looks like it disagrees with the literature. It does not — the
three sources normalize the **limit** differently, while agreeing on
``\boldsymbol{B}``. Naming them once avoids a great deal of confusion:

| convention | elliptic ``\mathbb{H}^{\mathcal{E}}`` | ribbon ``\mathbb{H}^{\mathcal{R}}`` |
| :--------- | :------------------------------------ | :---------------------------------- |
| **`MeanFieldHomogenization`** — limit normalized by ``b``, uniformly | ``\lim (c/b)\,\mathbb{Q}^{-1} = \tfrac{3}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` | ``\tfrac{2}{\pi}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` |
| **Echoes** and [barthelemyMMS2023](@cite) — elliptic normalized by ``a`` | ``\lim \omega\,\mathbb{Q}^{-1} = \tfrac{3\eta}{4}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` | ``\tfrac{2}{\pi}\,\underline{n}\stackrel{s}{\otimes}\boldsymbol{B}\stackrel{s}{\otimes}\underline{n}`` |

So the two elliptic compliances differ by exactly ``\eta``:

```math
\mathbb{H}^{\mathcal{E}}_{\texttt{MeanFieldHomogenization}}
= \frac{1}{\eta}\;\mathbb{H}^{\mathcal{E}}_{\text{Echoes}},
\qquad
\mathbb{H}^{\mathcal{R}}_{\texttt{MeanFieldHomogenization}}
= \mathbb{H}^{\mathcal{R}}_{\text{Echoes}} .
```

They coincide for the **penny crack** ``\eta=1``, which is why the discrepancy is
invisible on the most common test case — and why a penny-only cross-check cannot
detect a wrong ``\eta``-dependence.

Measured, not inferred from the papers — Echoes' `crack_compliance` against
[`compliance_contribution`](@ref) on a flat triaxial ellipsoid
(``a = 1``, ``b = \eta``, ``c = 10^{-5}``):

| ``\eta`` | 1.0 | 0.7 | 0.5 | 0.3 | 0.1 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| ``\mathbb{H}_{\text{Echoes}}/\mathbb{H}_{\texttt{MFH}}`` | 1.0000 | 0.7000 | 0.5000 | 0.3000 | 0.1000 |

On the `MeanFieldHomogenization` side the ``3/4`` is ``\eta``-independent to machine
precision (``\mathbb{H}_{3333}/B_{33} = 0.750000`` for every ``\eta``).

!!! warning "``\boldsymbol{B}`` is the same, ``\mathbb{H}`` is not"
    All three sources normalize ``\boldsymbol{B}`` by the half-width ``b``, so
    the **COD tensors agree**. Only the compliance differs, and only for the
    elliptic crack. If you compare a ``\boldsymbol{B}`` across papers, expect
    agreement; if you compare an ``\mathbb{H}``, check the normalization first.

### Checking ``\mathbb{H}`` without going through ``\boldsymbol{B}``

The closed forms of the next section give ``\boldsymbol{B}``, and ``\mathbb{H}``
follows from it by the ``3/4``. Comparing ``\mathbb{H}`` to those closed forms
would therefore verify nothing: it is the same formula read twice.

Read from right to left the boxed definition is a **recipe**: take a flat
ellipsoid, compute ``\mathbb{P}``, assemble
``\mathbb{Q} = \mathbb{C}-\mathbb{C}:\mathbb{P}:\mathbb{C}``, invert, scale by
the flatness ``\omega = c/b``. That path never touches the crack machinery, so
agreement with [`compliance_contribution`](@ref) is evidence rather than
tautology:

```@example Horacle
using MeanFieldHomogenization, TensND, Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

E, ν = 210.0, 0.3
C₀ = TensISO{3}(E / (1 - 2ν), E / (1 + ν))   # (3k, 2μ)

## ω-family of the definition, evaluated at finite flatness
function H_from_hill(a, b, ω, C₀)
    P = hill_tensor(Ellipsoid(a, b, ω * b), C₀)
    Q = C₀ - C₀ ⊡ P ⊡ C₀
    return ω * inv(Q)
end

relerr(A, B) =
    maximum(abs(A[i,j,k,l] - B[i,j,k,l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3) /
    maximum(abs(B[i,j,k,l]) for i in 1:3, j in 1:3, k in 1:3, l in 1:3)

ωs = exp10.(range(-1, -3; length = 9))
plt = plot(; xscale = :log10, yscale = :log10, legend = :topleft,
           xlabel = "flatness ω = c/b", ylabel = "relative error on ℍ")
for η in (1.0, 0.5, 0.3)
    crack = η == 1.0 ? PennyCrack(1.0) : EllipticCrack(1.0, η)
    H_ref = compliance_contribution(crack, C₀)
    plot!(plt, ωs, [relerr(H_from_hill(1.0, η, ω, C₀), H_ref) for ω in ωs];
          marker = :circle, label = "η = $η")
end
plot!(plt, ωs, ωs; linestyle = :dash, color = :black, label = "slope 1")
plt
```

The three curves are straight lines of **slope 1**: the error decays like
``\omega``, which is the order of the Taylor term that resolves the limit
([barthelemyIJSS2009](@cite)). That slope is the real content of the check — a
single ``\omega`` would not distinguish a true limit from a coincidence.

!!! note "Two limits of the flattening route"
    The Hill tensor is validated down to ``\omega = 10^{-3}``, where the sweep
    stops, and the residue backend returns `NaN` below about
    ``c \approx 10^{-3}`` — hence `method = :nestedquadgk` for the anisotropic
    cases. Neither affects [`cod_tensor`](@ref), which resolves the limit
    analytically instead of flattening an ellipsoid.

## From the Green operator to ``\boldsymbol{B}``

The closed forms of the next section are usually quoted. They are in fact
derivable in closed form from the Fourier Green operator, and knowing *where*
each factor comes from is what tells you which anisotropies admit a closed form
at all. The script `scripts/09_cod_symbolic_green.jl` carries out the whole
derivation symbolically, with the shipped closed forms as its oracles.

### The reduced kernel ``\hat{\boldsymbol{Q}}^{\star}_{nn}``

With ``\boldsymbol{N}(\underline{\xi}) = \underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}``
the **acoustic** (Christoffel) tensor, the two Fourier kernels of the traction
integral equation on the crack plane are [kunin1983](@cite),
[kanaun2009](@cite):

```math
\hat{\mathbb{\Gamma}}(\underline{\xi})
  = \underline{\xi}\stackrel{s}{\otimes}\boldsymbol{N}^{-1}(\underline{\xi})
    \stackrel{s}{\otimes}\underline{\xi},
\qquad
\hat{\mathbb{Q}}(\underline{\xi})
  = \mathbb{C} - \mathbb{C}:\hat{\mathbb{\Gamma}}(\underline{\xi}):\mathbb{C},
```

and the object the crack problem actually needs is the **reduced** transform —
the Fourier transform of the restriction of ``\hat{\mathbb{Q}}`` to the crack
plane [barthelemySifAniso](@cite):

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
  \boldsymbol{N}^{-1}(\underline{\xi})\cdot\boldsymbol{V}^{\mathsf{T}}(\underline{\xi}),
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
plane-strain and antiplane moduli:

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
([hoenig1978](@cite), [barthelemyIJES2021](@cite)) is therefore nothing but the
sum of the two in-plane Stroh roots — it *comes out of* the factorization rather
than being postulated.

### The crack-plane integral: where the elliptic integrals enter

On the crack contour ``\underline{\xi}^{\star}(\varphi) =
\eta\cos\varphi\,\underline{\ell} + \sin\varphi\,\underline{m}``, so that
``\rho = \sqrt{\eta^{2}\cos^{2}\varphi+\sin^{2}\varphi}``, and the ``\rho`` of
the degree-1 homogeneity cancels the ``1/\rho^{2}`` of the two in-plane dyads.
**Exactly three** angular integrals survive:

```math
\frac14\!\int_0^{2\pi}\!\rho\,\mathrm{d}\varphi = \mathcal{E}_\eta,
\qquad
\frac14\!\int_0^{2\pi}\!\frac{\eta^{2}\cos^{2}\varphi}{\rho}\,\mathrm{d}\varphi
  = \eta^{2}\mathcal{S}_\eta,
\qquad
\frac14\!\int_0^{2\pi}\!\frac{\sin^{2}\varphi}{\rho}\,\mathrm{d}\varphi
  = \mathcal{C}_\eta,
```

with ``\mathcal{C}_\eta = (\mathcal{E}_\eta-\eta^{2}\mathcal{K}_\eta)/(1-\eta^{2})``
and ``\mathcal{S}_\eta = (\mathcal{K}_\eta-\mathcal{E}_\eta)/(1-\eta^{2})`` — the
combinations stored by `_elliptic_CS`. The ``\cos\varphi\sin\varphi`` cross term
vanishes by parity, which is why ``\boldsymbol{B}`` is diagonal in the crack
frame. Hence

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

### What breaks in the general case

Nothing above survives a general anisotropy: the acoustic tensor no longer
block-decouples, ``\det\boldsymbol{N}`` is an irreducible sextic in ``\xi_n``, and
the crack plane has no rotational symmetry left, so one in-plane direction no
longer determines the others. Both integrals become numerical — which is what
the `Residue` and cubature back-ends do, the first summing residues over the six
Stroh roots located numerically.

## Closed forms of ``\boldsymbol{B}``

### Isotropic matrix

For ``\mathbb{C} = 3k\,\mathbb{J}+2\mu\,\mathbb{K}`` with Young's modulus ``E``
and Poisson ratio ``\nu``, in the crack frame
``(\underline{\ell},\underline{m},\underline{n})``:

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

The three components are the three fracture modes: ``B_{nn}`` opens the crack
(mode I), ``B_{mm}`` and ``B_{\ell\ell}`` shear it (modes II and III).

**Penny crack** ``\eta=1``, where ``\mathcal{K}_1 = \mathcal{E}_1 = \pi/2``:

```math
B_{nn} = \frac{16\,(1-\nu^{2})}{3\pi E},
\qquad
B_{mm} = B_{\ell\ell} = \frac{B_{nn}}{1-\nu/2}.
```

Implementation: `src/Cracks/cod_analytical.jl`, selected by `method = :auto`
when ``\mathbb{C}_0`` is a `TensISO{4,3}`.

### Transversely isotropic matrix

When the matrix is transversely isotropic with its axis **aligned with the crack
normal** ``\underline{n}``, ``\boldsymbol{B}`` is still analytical. The closed
forms use the engineering parameters ``(E,\nu_1,\nu_2,H,\Gamma)`` defined on the
compliance ``\mathbb{S} = \mathbb{C}^{-1}`` [hoenig1978](@cite),
[kanaun2009](@cite), [barthelemyIJES2021](@cite), and reduce to the isotropic
case for ``\nu_1=\nu_2=\nu``, ``H=\Gamma=1``. The auxiliary coefficients are
documented inline in `src/Cracks/cod_analytical.jl`.

The more general cases — a TI axis **not** aligned with the crack normal, or an
elliptic-orthotropic matrix — are treated in [barthelemyMMS2023](@cite) and
[barthelemySifAniso](@cite) but are not yet exposed here.

### Arbitrary anisotropy — numerical

No closed form exists in general. Following [barthelemyIJSS2009](@cite), the
limit ``\omega\to 0`` is resolved by extracting the **first-order term** of the
Taylor expansion of ``\mathbb{P}`` in ``\omega``; that term has an integral
representation on the unit circle of the crack plane, evaluated by either
algorithm trait:

- **`DECUHR`** — adaptive cubature [espelid1994](@cite), ForwardDiff-safe
  (`src/Cracks/green_decuhr.jl`);
- **`Residue`** — Cauchy-residue reduction to a 1-D quadrature, as in
  [masson2008](@cite) adapted to the crack kernel, `Float64` only
  (`src/Cracks/green_residue.jl`).

`method = :auto` always picks a **cubature**, never `Residue`: `DECUHR` when its
weak dependency is loaded, and the type-generic `NestedQuadGK` otherwise — which
is also what a `ForwardDiff.Dual` or symbolic scalar gets. `Residue` is faster
but its acoustic polynomial degenerates when the reference is anisotropic in
*type* and isotropic in *value*, a case the self-consistent and differential
schemes reach at their first step, so it is available on explicit
`method = :residues` only (`src/Core/dispatch.jl`).

## Dilute correction to the effective compliance

[`compliance_contribution`](@ref)`(crack, C₀)` returns ``\mathbb{H}`` itself —
the *size-independent* contribution, not the dilute correction. Cracks have no
volume fraction, so the amount of cracking is measured by a **Budiansky crack
density** [budiansky1976](@cite), [kachanov1993](@cite), and reintroduced by
[`delta_compliance`](@ref):

| | density | dilute correction |
| :-- | :-- | :-- |
| **elliptic** (3-D) | ``\varepsilon^{3\mathrm{d}} = N\,a\,b^{2}`` (number per unit volume × major × minor²) | ``\Delta\mathbb{S} = \tfrac{4\pi}{3}\,\varepsilon^{3\mathrm{d}}\,\mathbb{H}^{\mathcal{E}}`` |
| **ribbon** (2-D) | ``\varepsilon^{2\mathrm{d}} = N\,b^{2}`` (number per unit area × half-width²) | ``\Delta\mathbb{S} = \pi\,\varepsilon^{2\mathrm{d}}\,\mathbb{H}^{\mathcal{R}}`` |

Implementation: `src/Cracks/compliance.jl`, dispatching on the crack shape.

## Intensity factors at the crack front

At a point ``\underline{x}^{\star}_{0}`` of the crack front, with in-plane outer
normal ``\underline{\nu}`` and tangent
``\underline{\tau} = \underline{n}\wedge\underline{\nu}``, the asymptotic
expansions of the jump and the traction read [irwin1957](@cite),
[kassir1968](@cite), [willis1968](@cite):

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
``G = \underline{K}\cdot\underline{N}`` [barnett1972](@cite),
[rice1989](@cite).

The central result of the anisotropic theory [kanaun1981](@cite),
[kunin1983](@cite), [kanaun2009](@cite) is that SIF and DIF are **purely local**
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
dispensed with even when studying elliptic cracks.

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
\underline{N}^{\mathcal{E}}
= \tfrac{3}{8}\sqrt{\pi b}\;\sqrt{\varrho}\;
  \boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
  \cdot\boldsymbol{\Sigma}\cdot\underline{n},
\qquad
\underline{K}^{\mathcal{E}}
= \tfrac{3}{8}\pi^{3/2}\sqrt{b}\;\sqrt{\varrho}\;
  \bigl(\boldsymbol{B}^{\mathcal{R}}(\underline{\nu},\underline{n})\bigr)^{-1}
  \cdot\boldsymbol{B}^{\mathcal{E}}(\underline{m},\underline{n},\eta)
  \cdot\boldsymbol{\Sigma}\cdot\underline{n},
```

where the dimensionless front factor is

```math
\varrho = b\,\|\boldsymbol{S}^{\dagger}\!\cdot\underline{y}^{\star}_{0}\|
= \sqrt{\eta^{2}\cos^{2}\theta_y + \sin^{2}\theta_y}.
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
the exchange relation.

### Modes

```math
K_{I} = |\underline{K}\cdot\underline{n}|,
\qquad
K_{II} = |\underline{K}\cdot\underline{\nu}|,
\qquad
K_{III} = |\underline{K}\cdot\underline{\tau}| .
```

Evaluation: [`sif`](@ref) and [`dif`](@ref) (`src/Cracks/sif.jl`).

## Dispatch

| `(crack, C₀)` | `:auto` selects | alternatives | ForwardDiff |
| :------------ | :-------------- | :----------- | :---------: |
| `EllipticCrack` / `RibbonCrack`, `TensISO` | `Analytical` | — | ✓ |
| `EllipticCrack` / `RibbonCrack`, `TensTI` (aligned) | `Analytical` (MFH) | — | ✓ |
| `EllipticCrack` / `RibbonCrack`, `AbstractTens{4,3}` | `DECUHR` if loaded, else `NestedQuadGK` | `:residues`, `:decuhr`, `:nestedquadgk` | ✓ |

Entry points: [`cod_tensor`](@ref) / [`B_tensor`](@ref) for ``\boldsymbol{B}``,
[`compliance_contribution`](@ref) for ``\mathbb{H}``,
[`delta_compliance`](@ref) for ``\Delta\mathbb{S}``, [`sif`](@ref) /
[`dif`](@ref) for the front quantities. The transport counterpart — a scalar COD
and a rank-1 resistivity contribution — is treated in
[Thermal cracks](thermal_cracks.md), with the same geometric factors ``3/4`` and
``2/\pi``.
