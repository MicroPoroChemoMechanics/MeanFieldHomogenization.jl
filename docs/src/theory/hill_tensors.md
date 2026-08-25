# [Hill polarization tensors](@id th-hill-tensors)

The [Eshelby problem](eshelby_problem.md) reduces to computing one object, the
Hill polarization tensor ``\mathbb{P}(\boldsymbol{A},\mathbb{C})``. This page
gives its closed forms.

The structure to keep in mind: ``\mathbb{P}`` factors into a purely **geometric**
part, which depends on the ellipsoid alone, and a purely **material** part,
which depends on the reference moduli alone. The geometric part is a set of
Newton-potential integrals; every shape — triaxial ellipsoid, spheroid, sphere,
infinite cylinder — is one column of the same two tables.

This page follows the appendix *Hill polarization tensors* of the
[Echoes manual](https://jfbarthelemy.github.io/echoes/); expressions,
conventions and bibliography are aligned on it. Extensions specific to
`MeanFieldHomogenization` are flagged as such.

Those columns are the shapes below — meridian sections, drawn to scale from the
semi-axes each closed form is written in. Every one of them is a limit of the
same triaxial ellipsoid, reached by sending an aspect ratio to `1`, `0` or `∞`:

```@setup hillshapes
# Drawn in a `@setup` block: this page is about the closed forms, and forty
# lines of plotting boilerplate in front of a thumbnail strip would bury them.
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

θ = range(0, 2π; length = 240)
const LIM = 1.45      # half-width of every panel, so the shapes compare directly

function panel(title, col; shape = :ellipse, ρt = 1.0, ρa = 1.0, second = nothing)
    p = plot(; aspect_ratio = 1, framestyle = :none, legend = false,
        xlims = (-LIM, LIM), ylims = (-LIM, LIM),
        title = title, titlefontsize = 8)
    if shape === :ellipse
        plot!(p, ρt .* cos.(θ), ρa .* sin.(θ); seriestype = :shape,
            fillcolor = col, fillalpha = 0.45, lc = col, lw = 1.6)
    elseif shape === :vstrip          # runs off the top and bottom: a fiber
        plot!(p, [-ρt, ρt, ρt, -ρt], [-LIM, -LIM, LIM, LIM]; seriestype = :shape,
            fillcolor = col, fillalpha = 0.45, lc = col, lw = 1.6)
    elseif shape === :hstrip          # runs off both sides: a tunnel crack
        plot!(p, [-LIM, LIM, LIM, -LIM], [-ρa, -ρa, ρa, ρa]; seriestype = :shape,
            fillcolor = col, fillalpha = 0.9, lc = col, lw = 1.6)
    end
    # A second principal section, to show that all three semi-axes differ.
    second === nothing || plot!(p, second[1] .* cos.(θ), second[2] .* sin.(θ);
        lw = 1.3, ls = :dash, lc = col)
    return p
end

panels = [
    panel("sphere\nω = 1", :steelblue; ρt = 0.95, ρa = 0.95),
    panel("prolate\nω = 3", :steelblue; ρt = 0.42, ρa = 1.26),
    panel("oblate\nω = 0.3", :steelblue; ρt = 1.26, ρa = 0.38),
    panel("triaxial\na > b > c", :steelblue; ρt = 1.26, ρa = 0.55,
        second = (0.75, 0.55)),
    panel("cylinder\na → ∞", :seagreen; shape = :vstrip, ρt = 0.50),
    panel("penny crack\nω → 0", :sienna; ρt = 0.95, ρa = 0.035),
]

# Six panels in a row leave 150 px each, too narrow for the two-line titles.
# These sketches carry no axes (`framestyle = :none`), so the page-wide margins
# set in `make.jl` would only eat into the drawing: zero them out here.
shape_strip = plot(panels...; layout = (2, 3), size = (900, 460),
    left_margin = 0Plots.mm, bottom_margin = 0Plots.mm,
    right_margin = 0Plots.mm, top_margin = 3Plots.mm)
```

```@example hillshapes
shape_strip # hide
```

Two of the six are limits rather than bounded bodies: the cylinder leaves the
panel because it is unbounded along its axis, and the penny crack is *flat*
rather than thin. The dashed outline in the triaxial panel is the second
principal section — the case where no two semi-axes coincide, and the only one
that needs the general cubature.

The **ribbon crack** is deliberately absent: it is the ``a\to\infty`` limit of
the penny crack, so in a meridian section it draws exactly the same flat
segment. What distinguishes it lies *in* the crack plane, which this view cannot
show — hence the separate `RibbonCrack` type, and the ``\eta\to 0`` column of
every table below.

The **flat** limits are the ones that need care: ``\mathbb{P}``
stays finite there, but the object built on it, ``\mathbb{Q}``, degenerates in a
controlled way — which is why the crack theory is written on ``\mathbb{Q}``
([Crack opening displacement](cod_tensors.md)).

## Newton-potential integrals

Three integrals over the unit sphere, depending on ``\boldsymbol{A}`` only,
factor every analytical Hill formula:

```math
\boldsymbol{I}^{\boldsymbol{A}}
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\otimes\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}
= \frac{1}{4\pi}
\int_{\|\underline{\zeta}\|=1}
\frac{(\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})\otimes
      (\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})}
     {\|\boldsymbol{A}^{-1}\!\cdot\underline{\zeta}\|^{2}}\,\mathrm{d}S_{\zeta}
```

```math
\mathbb{U}^{\boldsymbol{A}}
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\otimes\underline{\xi}\otimes
      \underline{\xi}\otimes\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}
```

```math
\mathbb{V}^{\boldsymbol{A}}
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\stackrel{s}{\otimes}\boldsymbol{1}
      \stackrel{s}{\otimes}\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}
= \frac{\boldsymbol{1}\stackrel{s}{\boxtimes}\boldsymbol{I}^{\boldsymbol{A}}
      + \boldsymbol{I}^{\boldsymbol{A}}\stackrel{s}{\boxtimes}\boldsymbol{1}}{2}
```

The two parametrizations of ``\boldsymbol{I}^{\boldsymbol{A}}`` are related by
the bijection of the unit sphere onto itself
``\underline{\zeta}\mapsto\underline{\xi} =
\boldsymbol{A}^{-1}\!\cdot\underline{\zeta}/
\|\boldsymbol{A}^{-1}\!\cdot\underline{\zeta}\|``, whose surface-element
identity is

```math
\mathrm{d}S_{\zeta}
= \frac{\det\boldsymbol{A}}{\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,
  \mathrm{d}S_{\xi}.
```

An intrinsic proof is given in [barthelemyIJSS2016](@cite), as an alternative to
the component reasoning of [mura1987](@cite). Both forms are useful: the
``\underline{\xi}`` form makes the geometry explicit, the ``\underline{\zeta}``
form is the one that degenerates cleanly in the cylinder limit below.

In `MeanFieldHomogenization` these are [`tens_IA`](@ref), [`tens_UA`](@ref) and
[`tens_VA`](@ref).

### Principal coefficients

``\boldsymbol{A}`` and ``\boldsymbol{I}^{\boldsymbol{A}}`` share their
eigenvectors, so

```math
\boldsymbol{I}^{\boldsymbol{A}}
= \sum_{i=1}^{3} I_i^{\boldsymbol{A}}\,
  \underline{e}^{\boldsymbol{A}}_i\otimes\underline{e}^{\boldsymbol{A}}_i .
```

The coefficients ``I_i^{\boldsymbol{A}}``, identified with Newton-potential
integrals [kellogg1929](@cite), [eshelby1957](@cite), [parnell2016](@cite), and
the secondary coefficients ``I_{ij}^{\boldsymbol{A}}`` admit closed forms in
every symmetry class:

| | ellipsoid | prolate spheroid | oblate spheroid | sphere | cylinder |
| :-- | :-- | :-- | :-- | :-- | :-- |
| | ``a>b>c`` | ``a>b=c`` | ``a=b>c`` | ``a=b=c`` | ``a\to\infty,\ b\ge c`` |
| ``I_1^{\boldsymbol{A}}`` | ``\dfrac{a\,b\,c\,(\mathcal{F}-\mathcal{E})}{(a^2-b^2)\sqrt{a^2-c^2}}`` | ``1-2\,I_3^{\boldsymbol{A}}`` | ``c\,\dfrac{a^2\arccos(c/a)-c\sqrt{a^2-c^2}}{2(a^2-c^2)^{3/2}}`` | ``\tfrac{1}{3}`` | ``0`` |
| ``I_2^{\boldsymbol{A}}`` | ``1-I_1^{\boldsymbol{A}}-I_3^{\boldsymbol{A}}`` | ``I_3^{\boldsymbol{A}}`` | ``I_1^{\boldsymbol{A}}`` | ``\tfrac{1}{3}`` | ``\dfrac{c}{b+c}`` |
| ``I_3^{\boldsymbol{A}}`` | ``\dfrac{a\,b\,c}{(b^2-c^2)\sqrt{a^2-c^2}}\left(\dfrac{b\sqrt{a^2-c^2}}{a\,c}-\mathcal{E}\right)`` | ``a\,\dfrac{a\sqrt{a^2-c^2}-c^2\operatorname{arcosh}(a/c)}{2(a^2-c^2)^{3/2}}`` | ``1-2\,I_1^{\boldsymbol{A}}`` | ``\tfrac{1}{3}`` | ``\dfrac{b}{b+c}`` |

Here ``\mathcal{F} = \mathcal{F}(\theta,\kappa)`` and
``\mathcal{E} = \mathcal{E}(\theta,\kappa)`` are the incomplete elliptic
integrals of the first and second kind [abramowitz1972](@cite), of amplitude and
parameter

```math
\theta = \arcsin\sqrt{1-\frac{c^{2}}{a^{2}}},
\qquad
\kappa = \sqrt{\frac{a^{2}-b^{2}}{a^{2}-c^{2}}}.
```

The secondary coefficients follow from the ``I_i^{\boldsymbol{A}}`` by

```math
I_{ij}^{\boldsymbol{A}} = \frac{I_j^{\boldsymbol{A}}-I_i^{\boldsymbol{A}}}
                               {\rho_i^{2}-\rho_j^{2}}
\quad (i\ne j),
\qquad
I_{ii}^{\boldsymbol{A}} = \frac{1}{3}\left(\frac{1}{\rho_i^{2}}
                        - \sum_{j\ne i} I_{ij}^{\boldsymbol{A}}\right),
```

except where the denominator degenerates — each symmetry class then has its own
regular expression:

| | ellipsoid | prolate spheroid | oblate spheroid | sphere | cylinder |
| :-- | :-- | :-- | :-- | :-- | :-- |
| ``I_{11}^{\boldsymbol{A}}`` | ``\tfrac{1}{3}\left(\tfrac{1}{a^2}-I_{31}^{\boldsymbol{A}}-I_{12}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{3}\left(\tfrac{1}{a^2}-2I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{4}\left(\tfrac{1}{a^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{5a^2}`` | ``0`` |
| ``I_{22}^{\boldsymbol{A}}`` | ``\tfrac{1}{3}\left(\tfrac{1}{b^2}-I_{12}^{\boldsymbol{A}}-I_{23}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{4}\left(\tfrac{1}{c^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{4}\left(\tfrac{1}{a^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{5a^2}`` | ``\dfrac{c(2b+c)}{3b^{2}(b+c)^{2}}`` |
| ``I_{33}^{\boldsymbol{A}}`` | ``\tfrac{1}{3}\left(\tfrac{1}{c^2}-I_{23}^{\boldsymbol{A}}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{4}\left(\tfrac{1}{c^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{3}\left(\tfrac{1}{c^2}-2I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{5a^2}`` | ``\dfrac{b(b+2c)}{3c^{2}(b+c)^{2}}`` |
| ``I_{23}^{\boldsymbol{A}}`` | ``\dfrac{I_3^{\boldsymbol{A}}-I_2^{\boldsymbol{A}}}{b^2-c^2}`` | ``\tfrac{1}{4}\left(\tfrac{1}{c^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\dfrac{I_3^{\boldsymbol{A}}-I_2^{\boldsymbol{A}}}{b^2-c^2}`` | ``\tfrac{1}{5a^2}`` | ``\dfrac{1}{(b+c)^{2}}`` |
| ``I_{31}^{\boldsymbol{A}}`` | ``\dfrac{I_3^{\boldsymbol{A}}-I_1^{\boldsymbol{A}}}{a^2-c^2}`` | ``\dfrac{I_3^{\boldsymbol{A}}-I_1^{\boldsymbol{A}}}{a^2-c^2}`` | ``\dfrac{I_3^{\boldsymbol{A}}-I_1^{\boldsymbol{A}}}{a^2-c^2}`` | ``\tfrac{1}{5a^2}`` | ``0`` |
| ``I_{12}^{\boldsymbol{A}}`` | ``\dfrac{I_2^{\boldsymbol{A}}-I_1^{\boldsymbol{A}}}{a^2-b^2}`` | ``\dfrac{I_2^{\boldsymbol{A}}-I_1^{\boldsymbol{A}}}{a^2-b^2}`` | ``\tfrac{1}{4}\left(\tfrac{1}{a^2}-I_{31}^{\boldsymbol{A}}\right)`` | ``\tfrac{1}{5a^2}`` | ``0`` |

with ``I_{ij}^{\boldsymbol{A}} = I_{ji}^{\boldsymbol{A}}``. The circular
cylinder ``b=c`` gives ``I_2^{\boldsymbol{A}}=I_3^{\boldsymbol{A}}=\tfrac{1}{2}``
and ``I_{22}^{\boldsymbol{A}}=I_{33}^{\boldsymbol{A}}=I_{23}^{\boldsymbol{A}}
=\tfrac{1}{4c^{2}}``.

!!! warning "Normalization differs from the classical references"
    For writing convenience the coefficients tabulated above are **rescaled**
    relative to [kellogg1929](@cite) and [eshelby1957](@cite): they differ by a
    factor ``4\pi/3`` for ``I_{ij}^{\boldsymbol{A}}`` with ``i\ne j``, and by
    ``4\pi`` for all the others. The normalization used here is the one that
    makes ``\sum_i I_i^{\boldsymbol{A}} = 1``.

    Internally, `newton_potential_3d` and `newton_potential_3d_cylinder` return
    the **raw** kernel — the values above multiplied by ``4\pi`` — and the
    division is applied at the [`tens_IA`](@ref) call site.

!!! note "Why the cylinder column has zeros that still matter"
    The cylinder column is the limit ``a\to\infty`` of the triaxial one. Three
    entries vanish, but they carry *finite products* that survive:

    ```math
    a^{2}\,I_{12}^{\boldsymbol{A}} \xrightarrow[a\to\infty]{} I_2^{\boldsymbol{A}} = \frac{c}{b+c},
    \qquad
    a^{2}\,I_{31}^{\boldsymbol{A}} \xrightarrow[a\to\infty]{} I_3^{\boldsymbol{A}} = \frac{b}{b+c}.
    ```

    These products appear only through ``\rho_j^{2}I_{ij}^{\boldsymbol{A}}``
    terms (``\rho_1=a``) in ``\mathbb{U}^{\boldsymbol{A}}``, where they produce
    the vanishing first row and column of
    ``\mathrm{Mat}(\mathbb{U}^{\mathrm{cyl}})`` below and keep the third Eshelby
    identity valid at the cylinder endpoint. The circular sub-case ``b=c`` is
    evaluated on a separate branch, avoiding the ``(b^2-c^2)^{-1}``
    intermediates.

### Identities

Always satisfied, and useful as numerical checks [eshelby1957](@cite):

```math
\sum_i I_i^{\boldsymbol{A}} = 1,
\qquad
3\,I_{ii}^{\boldsymbol{A}} + \sum_{j\ne i} I_{ij}^{\boldsymbol{A}} = \frac{1}{\rho_i^{2}},
\qquad
3\,\rho_i^{2}\,I_{ii}^{\boldsymbol{A}} + \sum_{j\ne i}\rho_j^{2}\,I_{ij}^{\boldsymbol{A}} = 3\,I_i^{\boldsymbol{A}}.
```

### Components of ``\mathbb{U}^{\boldsymbol{A}}`` and ``\mathbb{V}^{\boldsymbol{A}}``

Both are orthotropic along the ellipsoid axes. In the principal frame
[barthelemyIJSS2016](@cite), [barthelemyIJES2020_hilltrans](@cite):

```math
U^{\boldsymbol{A}}_{iiii} = \tfrac{3}{2}\bigl(I_i^{\boldsymbol{A}}-\rho_i^{2}I_{ii}^{\boldsymbol{A}}\bigr),
\qquad
U^{\boldsymbol{A}}_{iijj} = U^{\boldsymbol{A}}_{ijij} = U^{\boldsymbol{A}}_{ijji}
= \tfrac{1}{2}\bigl(I_j^{\boldsymbol{A}}-\rho_i^{2}I_{ij}^{\boldsymbol{A}}\bigr)
= \tfrac{1}{2}\bigl(I_i^{\boldsymbol{A}}-\rho_j^{2}I_{ij}^{\boldsymbol{A}}\bigr)
```

```math
V^{\boldsymbol{A}}_{iiii} = I_i^{\boldsymbol{A}},
\qquad
V^{\boldsymbol{A}}_{ijij} = V^{\boldsymbol{A}}_{ijji}
= \tfrac{1}{4}\bigl(I_i^{\boldsymbol{A}}+I_j^{\boldsymbol{A}}\bigr)
\qquad (i\ne j).
```

**Sphere** ``\boldsymbol{A}=\boldsymbol{1}``:

```math
\mathbb{U}^{\boldsymbol{1}} = \tfrac{1}{3}\mathbb{J} + \tfrac{2}{15}\mathbb{K},
\qquad
\mathbb{V}^{\boldsymbol{1}} = \tfrac{1}{3}\mathbb{I}.
```

**Infinite elliptic cylinder** ``a\to\infty``, axis
``\underline{e}^{\boldsymbol{A}}_1``, transverse semi-axes ``b\ge c``
([mura1987](@cite), §11.22) — substituting the cylinder column above:

```math
\mathrm{Mat}\bigl(\mathbb{U}^{\mathrm{cyl}}\bigr) =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & \frac{c(b+2c)}{2(b+c)^{2}} & \frac{bc}{2(b+c)^{2}} & 0 & 0 & 0\\
0 & \frac{bc}{2(b+c)^{2}} & \frac{b(2b+c)}{2(b+c)^{2}} & 0 & 0 & 0\\
0 & 0 & 0 & \frac{bc}{(b+c)^{2}} & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix},
\quad
\mathrm{Mat}\bigl(\mathbb{V}^{\mathrm{cyl}}\bigr) =
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & \frac{c}{b+c} & 0 & 0 & 0 & 0\\
0 & 0 & \frac{b}{b+c} & 0 & 0 & 0\\
0 & 0 & 0 & \frac{1}{2} & 0 & 0\\
0 & 0 & 0 & 0 & \frac{b}{2(b+c)} & 0\\
0 & 0 & 0 & 0 & 0 & \frac{c}{2(b+c)}
\end{pmatrix}
```

in Kelvin–Mandel storage and in the frame
``(\underline{e}^{\boldsymbol{A}}_i)``. The vanishing first row and column is
the signature of the infinite cylinder: **no polarization is transmitted along
its axis**. For the circular cylinder ``b=c`` the non-zero entries reduce to
``U^{\mathrm{cyl}}_{2222}=U^{\mathrm{cyl}}_{3333}=\tfrac{3}{8}``,
``U^{\mathrm{cyl}}_{2233}=\tfrac{1}{8}``,
``\bigl[\mathrm{Mat}(\mathbb{U}^{\mathrm{cyl}})\bigr]_{44}=\tfrac{1}{4}``, and
``\mathrm{Mat}(\mathbb{V}^{\mathrm{cyl}})`` to the diagonal
``\bigl(0,\tfrac{1}{2},\tfrac{1}{2},\tfrac{1}{2},\tfrac{1}{4},\tfrac{1}{4}\bigr)``.

## Hill tensor in elasticity

### Arbitrary anisotropy

```math
\mathbb{P}(\boldsymbol{A},\mathbb{C})
= \frac{1}{4\pi}
\int_{\|\underline{\zeta}\|=1}
(\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})\stackrel{s}{\otimes}
\Bigl((\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})\cdot\mathbb{C}
      \cdot(\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})\Bigr)^{-1}
\stackrel{s}{\otimes}(\boldsymbol{A}^{-1}\!\cdot\underline{\zeta})
\,\mathrm{d}S_{\zeta}
```

```math
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\stackrel{s}{\otimes}
      \bigl(\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}\bigr)^{-1}
      \stackrel{s}{\otimes}\underline{\xi}}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}
```

[willis1977](@cite), see also [mura1987](@cite). Inverting the acoustic tensor
``\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}`` pointwise is the source of
all the computational work: in general no closed form exists and one resorts to
numerical cubature [ghahremani1977](@cite), [gavazzi1990](@cite),
[masson2008](@cite). `MeanFieldHomogenization` offers three algorithm traits, mirroring the
Echoes `NUMINT` / `RESIDUES` options:

- **`DECUHR`** — the surface integral is evaluated by the adaptive cubature for
  singular integrands of [espelid1994](@cite). ForwardDiff-safe. Selected by
  `:auto` when its (weak-dependency) extension is loaded.
- **`NestedQuadGK`** — nested adaptive 1-D quadrature, type-generic and always
  available; the `:auto` choice otherwise, and the one used for `Dual`,
  `Complex` and symbolic coefficients whatever else is loaded.
- **`Residue`** — the inner ``\varphi`` integral is reduced to a sum of residues
  by the Cauchy theorem, leaving a single 1-D quadrature [masson2008](@cite).
  The fastest of the three (~4 ms against ~11 ms and ~31 ms on a triaxial
  ellipsoid), but reachable on an explicit `method = :residues` only: it is
  `Float64`-only (the polynomial root finder it needs is not differentiable by
  ForwardDiff), and its acoustic polynomial degenerates when the reference is
  anisotropic in *type* while isotropic in *value* — returning `NaN` there
  instead of a number. That reference is what the differential and
  self-consistent schemes feed back at their first step, hence the choice of a
  cubature as the default.

Analytical paths exist in the literature for further anisotropy classes
([withers1989](@cite), [pouya2000](@cite), [pouya2006](@cite),
[suvorov2002](@cite)) and are not all implemented yet.

### Isotropic matrix — the shape/moduli factorization

With bulk modulus ``k``, shear modulus ``\mu`` and first Lamé parameter
``\lambda = k-2\mu/3``, so that
``\mathbb{C} = 3k\,\mathbb{J}+2\mu\,\mathbb{K} = 3\lambda\,\mathbb{I}+2\mu\,\mathbb{K}``,
the general expression collapses to [willis1977](@cite)

```math
\boxed{\;
\mathbb{P}\bigl(\boldsymbol{A},\,3\lambda\,\mathbb{I}+2\mu\,\mathbb{K}\bigr)
= \frac{1}{\lambda+2\mu}\,\mathbb{U}^{\boldsymbol{A}}
+ \frac{1}{\mu}\,\bigl(\mathbb{V}^{\boldsymbol{A}}-\mathbb{U}^{\boldsymbol{A}}\bigr).
\;}
```

This is the factorization announced at the top of the page: **shape and
orientation on one side** (``\mathbb{U}^{\boldsymbol{A}}``,
``\mathbb{V}^{\boldsymbol{A}}``), **reference moduli on the other**. Each shape
column of the tables above therefore yields a closed-form ``\mathbb{P}`` at once.

For a **sphere**, substituting ``\mathbb{U}^{\boldsymbol{1}}`` and
``\mathbb{V}^{\boldsymbol{1}}`` gives the classical Eshelby result

```math
\mathbb{P}\bigl(\boldsymbol{1},\,3k\,\mathbb{J}+2\mu\,\mathbb{K}\bigr)
= \frac{1}{3k+4\mu}\left(\mathbb{J} + \frac{3(k+2\mu)}{5\mu}\,\mathbb{K}\right).
```

For an **infinite cylinder**, substituting
``\mathrm{Mat}(\mathbb{U}^{\mathrm{cyl}})`` and
``\mathrm{Mat}(\mathbb{V}^{\mathrm{cyl}})`` gives a closed form with
``P^{\mathrm{cyl}}_{1jkl}\equiv 0``, i.e. no polarization along the axis
([mura1987](@cite), §11.22).

Implementation: `src/Elasticity/hill_3d_iso.jl` and
`src/Elasticity/hill_3d_cylinder_iso.jl`, selected by `method = :auto` when
``\mathbb{C}_0`` is a `TensISO`.

### Transversely isotropic matrix coaxial with a spheroid

When the matrix is transversely isotropic and its symmetry axis is **parallel to
the spheroid axis**, a fully analytical path exists
[barthelemyIJES2020_hilltrans](@cite). The Hill tensor is transversely isotropic
too, hence five Walpole coefficients (see
[Notation](notation.md#Isotropic-and-transversely-isotropic-bases) — there is no
``P_4`` because ``\mathbb{P}`` is major-symmetric):

```math
\mathbb{P} = P_1\,\mathbb{W}_1 + P_2\,\mathbb{W}_2
           + P_3\,(\mathbb{W}_3+\mathbb{W}_4)
           + P_5\,\mathbb{W}_5 + P_6\,\mathbb{W}_6 .
```

The five coefficients are closed-form combinations of `acosh` and complex square
roots (equations 53–58 of [barthelemyIJES2020_hilltrans](@cite)), depending on
the aspect ratio ``\omega`` (axial / transverse) and the five independent
constants ``(C_{1111}, C_{1122}, C_{1133}, C_{3333}, C_{2323})``.

The dispatcher routes a `TensTI{4}` matrix combined with a coaxial
`Ellipsoid{3, Spherical|Prolate|Oblate}` to this path by default; coaxiality is
detected by `_ti_coaxial(C₀, ell)`. Non-coaxial spheroids and triaxial
ellipsoids fall back to the anisotropic default, i.e. a cubature.
Implementation: `src/Elasticity/hill_3d_ti_coaxial.jl`.

### Anisotropic matrix, cylinder limit

`Cylinder` is a first-class inclusion type here (extension over Echoes). For
an arbitrarily anisotropic matrix the Masson residue algorithm does **not**
apply: it rests on the six complex roots of the acoustic polynomial along
``\underline{\xi}_3``, and at the cylinder limit one root escapes to infinity,
degenerating the polynomial.

The ``\underline{\zeta}`` form of the Willis integral degenerates cleanly
instead. The axial component of ``\underline{\zeta}`` vanishes identically, so
the surface integral collapses to a single quadrature over the transverse unit
circle:

```math
\mathbb{P}^{\mathrm{cyl}}
= \frac{1}{2\pi}\int_{0}^{2\pi}
\underline{\zeta}\stackrel{s}{\otimes}
\bigl(\underline{\zeta}\cdot\mathbb{C}\cdot\underline{\zeta}\bigr)^{-1}
\stackrel{s}{\otimes}\underline{\zeta}
\,\mathrm{d}\varphi,
\qquad
\underline{\zeta}(\varphi) = \Bigl(0,\ \frac{\cos\varphi}{b},\ \frac{\sin\varphi}{c}\Bigr).
```

This is a one-dimensional `QuadGK` integral, and it stays
ForwardDiff-compatible. Calling
`hill_tensor(Cylinder(…), C₀; method = :residues)` falls back to it silently.
Implementation: `src/Elasticity/hill_3d_cylinder_aniso.jl` (`CylinderQuadrature`
trait). The in-plane components coincide with the solution of the 2-D
plane-strain problem — the cylinder is the 3-D realization of the 2-D ellipse.

### 2-D plane strain

Plane strain is handled directly (extension over Echoes), integrating over
the unit circle ``\underline{\xi}\in S^{1}`` with a ``1/(2\pi)`` prefactor in
place of ``1/(4\pi)``. The isotropic case is analytical; the anisotropic one
uses the Masson residue reduction on the line integral.

## Hill tensor in conductivity

### Arbitrary anisotropy — closed form

For a conductivity tensor ``\boldsymbol{K}`` [willis1977](@cite):

```math
\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K})
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{\underline{\xi}\otimes\underline{\xi}}
     {(\underline{\xi}\cdot\boldsymbol{K}\cdot\underline{\xi})\,
      \|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}.
```

Unlike the order-4 case, **this integral has a closed form for any matrix
anisotropy**. Since ``\boldsymbol{K}`` is symmetric positive definite it has a
square root, ``\boldsymbol{K}^{1/2} = \sum_i\sqrt{K_i}\,
\underline{e}^{\boldsymbol{K}}_i\otimes\underline{e}^{\boldsymbol{K}}_i``, and the
denominator can be absorbed into the change of variable
``\underline{\zeta}\mapsto\boldsymbol{K}^{1/2}\!\cdot\boldsymbol{A}^{-1}\!\cdot
\underline{\zeta}``, which turns the anisotropic problem into an isotropic one
for a **fictitious ellipsoid** of shape tensor
``\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2}``:

```math
\boxed{\;
\boldsymbol{P}(\boldsymbol{A},\boldsymbol{K})
= \boldsymbol{K}^{-1/2}\cdot
  \boldsymbol{P}(\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2},\boldsymbol{1})\cdot
  \boldsymbol{K}^{-1/2}
= \boldsymbol{K}^{-1/2}\cdot
  \boldsymbol{I}^{\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2}}\cdot
  \boldsymbol{K}^{-1/2}.
\;}
```

This is the transformation derivation of [giraudMOM2019](@cite); an equivalent
Green's-function derivation is given in [barthelemyTIPM2009](@cite). Since
``\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2}`` need not be symmetric, the
fictitious semi-axes and principal directions are obtained by diagonalizing
``\boldsymbol{K}^{-1/2}\cdot\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\cdot
\boldsymbol{K}^{-1/2}``.

### Isotropic matrix — immediate

If ``\boldsymbol{K} = K\,\boldsymbol{1}`` the prefactor comes straight out:

```math
\boldsymbol{P}(\boldsymbol{A}, K\,\boldsymbol{1})
= \frac{\boldsymbol{I}^{\boldsymbol{A}}}{K}.
```

For a sphere, ``\boldsymbol{I}^{\boldsymbol{1}} = \tfrac{1}{3}\boldsymbol{1}``
gives ``\boldsymbol{P} = \tfrac{1}{3K}\boldsymbol{1}`` and
``\boldsymbol{s} = \tfrac{1}{3}\boldsymbol{1}`` — independent of ``K``.
Implementation: `src/Conductivity/hill_order2_3d.jl`.

## Dispatch

Entry point [`hill_tensor`](@ref); shape tensor via [`shape_tensor`](@ref);
geometric auxiliaries via [`tens_IA`](@ref), [`tens_UA`](@ref),
[`tens_VA`](@ref).

| `(inclusion, C₀)` | `:auto` selects | alternatives | ForwardDiff |
| :---------------- | :-------------- | :----------- | :---------: |
| `Ellipsoid{3}, TensISO` | `Analytical` | — | ✓ |
| `Ellipsoid{3}, TensTI` (coaxial) | `Analytical` (MFH) | `:residues`, `:decuhr` | ✓ |
| `Ellipsoid{3}, AbstractTens{4,3}` | `DECUHR` if loaded, else `NestedQuadGK` | `:residues`, `:decuhr`, `:nestedquadgk` | ✓ |
| `Cylinder, TensISO` | `Analytical` | — | ✓ |
| `Cylinder, AbstractTens{4,3}` | `CylinderQuadrature` | (residue degenerates) | ✓ |
| `Ellipsoid{2}, TensISO` | `Analytical` | — | ✓ |
| `Ellipsoid{2}, AbstractTens{4,2}` | `Analytical` (residue) | — | (Float64) |
| `Ellipsoid{3}, AbstractTens{2,3}` | `Analytical` (``\boldsymbol{K}^{-1/2}``) | — | ✓ |
| `Cylinder, AbstractTens{2,3}` | `Analytical` | — | ✓ |

Cylinder shape traits: `CircularCylindrical` when ``b=c`` (transversely
isotropic response, returned as `TensTI{4}` with axis
``\underline{e}^{\boldsymbol{A}}_1``) and `EllipticCylindrical` when ``b>c``
(orthotropic, returned as `TensOrtho`). Practical usage is covered in the manual
page [Cylindrical inclusions](../manual/cylindrical_inclusions.md).
