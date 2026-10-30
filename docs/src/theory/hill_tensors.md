# [Hill polarization tensors](@id th-hill-tensors)

!!! info "Before this page"
    [The Eshelby inclusion problem](@ref th-eshelby-problem), which defines
    ``\mathbb{P}`` and the Eshelby tensor, and the
    [Walpole basis](@ref th-notation-bases) and
    [ellipsoid geometry](@ref th-notation-ellipsoid) of the Conventions, in
    which every closed form below is written.

The [Eshelby problem](@ref th-eshelby-problem) reduces to computing one object, the
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
([Crack opening displacement](@ref th-cod-tensors)).

## 1. Newton-potential integrals

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

An intrinsic proof is given in [barthelemyIJSS2016](@citet), as an alternative to
the component reasoning of [mura1987](@citet).

```@raw html
<details><summary>Proof of the surface-element identity</summary>
```

Let ``r = \|\boldsymbol{A}^{-1}\!\cdot\underline{\zeta}\|``. Differentiating
``\underline{\xi} = \boldsymbol{A}^{-1}\!\cdot\underline{\zeta}/r`` along the
sphere, the variation of ``1/r`` only adds a component along ``\underline{\xi}``:

```math
\mathrm{d}\underline{\xi}
= \bigl(\boldsymbol{1}-\underline{\xi}\otimes\underline{\xi}\bigr)\cdot
  \boldsymbol{F}\cdot\mathrm{d}\underline{\zeta},
\qquad
\boldsymbol{F} = \frac{\boldsymbol{A}^{-1}}{r}.
```

The map is therefore the linear map ``\boldsymbol{F}`` followed by the
orthogonal projection onto the plane tangent to the sphere at
``\underline{\xi}``. A linear map carries an oriented surface element
``\mathrm{d}\underline{S}`` to ``(\det\boldsymbol{F})\,\boldsymbol{F}^{-T}\!\cdot
\mathrm{d}\underline{S}`` (Nanson's formula), and the projection onto a plane
keeps the component of a surface vector along the normal to that plane. On the
two unit spheres ``\mathrm{d}\underline{S}_{\zeta} = \mathrm{d}S_{\zeta}\,\underline{\zeta}``
and ``\mathrm{d}\underline{S}_{\xi} = \mathrm{d}S_{\xi}\,\underline{\xi}``, so that,
``\boldsymbol{A}`` being symmetric,

```math
\mathrm{d}S_{\xi}
= \underline{\xi}\cdot(\det\boldsymbol{F})\,\boldsymbol{F}^{-T}\!\cdot\underline{\zeta}\;\mathrm{d}S_{\zeta}
= \frac{\det\boldsymbol{A}^{-1}}{r^{3}}\;
  r\,\underline{\xi}\cdot\boldsymbol{A}\cdot\underline{\zeta}\;\mathrm{d}S_{\zeta}
= \frac{\det\boldsymbol{A}^{-1}}{r^{3}}\,\mathrm{d}S_{\zeta},
```

because ``\underline{\xi}\cdot\boldsymbol{A}\cdot\underline{\zeta}
= \underline{\zeta}\cdot\underline{\zeta}/r = 1/r``. The inverse map
``\underline{\zeta} = \boldsymbol{A}\cdot\underline{\xi}/\|\boldsymbol{A}\cdot\underline{\xi}\|``
gives ``r = 1/\|\boldsymbol{A}\cdot\underline{\xi}\|``, which turns the last
expression into the identity above.

```@raw html
</details>
``` Both forms are useful: the
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
integrals [kellogg1929, eshelby1957, parnell2016](@cite), and
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
integrals of the first and second kind [abramowitz1972](@cite),

```math
\mathcal{F}(\theta,\kappa) = \int_{0}^{\theta}
\frac{\mathrm{d}\varphi}{\sqrt{1-\kappa^{2}\sin^{2}\varphi}},
\qquad
\mathcal{E}(\theta,\kappa) = \int_{0}^{\theta}
\sqrt{1-\kappa^{2}\sin^{2}\varphi}\,\mathrm{d}\varphi,
```

of amplitude and modulus

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
    relative to [kellogg1929](@citet) and [eshelby1957](@citet): they differ by a
    factor ``4\pi/3`` for ``I_{ij}^{\boldsymbol{A}}`` with ``i\ne j``, and by
    ``4\pi`` for all the others. The normalization used here is the one that
    makes ``\sum_i I_i^{\boldsymbol{A}} = 1``.

    Internally, `newton_potential_3d` and `newton_potential_3d_cylinder` return
    the **raw** kernel — the values above multiplied by ``4\pi`` — and the
    division is applied at the [`tens_IA`](@ref) call site.

!!! note "Close to a sphere, close to a spheroid"
    The divided differences of the table, such as
    ``I_{31}^{\boldsymbol{A}} = (I_3^{\boldsymbol{A}}-I_1^{\boldsymbol{A}})/(a^2-c^2)``,
    cancel when two semi-axes come together. `newton_potential_3d` evaluates
    them otherwise: a spheroid, sphere included, through the integrals these
    potentials stand for, and a triaxial ellipsoid whose two semi-axes differ by
    less than 2 % through an expansion around the spheroid it is close to,
    exact and differentiable where the two meet.
    [Degenerate shapes](@ref th-degenerate-shapes) gives both.

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
[barthelemyIJSS2016, barthelemyIJES2020_hilltrans](@cite):

```math
\begin{aligned}
U^{\boldsymbol{A}}_{iiii} &= \tfrac{3}{2}\bigl(I_i^{\boldsymbol{A}}-\rho_i^{2}I_{ii}^{\boldsymbol{A}}\bigr),
\\
U^{\boldsymbol{A}}_{iijj} = U^{\boldsymbol{A}}_{ijij} = U^{\boldsymbol{A}}_{ijji}
&= \tfrac{1}{2}\bigl(I_j^{\boldsymbol{A}}-\rho_i^{2}I_{ij}^{\boldsymbol{A}}\bigr)
= \tfrac{1}{2}\bigl(I_i^{\boldsymbol{A}}-\rho_j^{2}I_{ij}^{\boldsymbol{A}}\bigr)
\end{aligned}
```

```math
V^{\boldsymbol{A}}_{iiii} = I_i^{\boldsymbol{A}},
\qquad
V^{\boldsymbol{A}}_{ijij} = V^{\boldsymbol{A}}_{ijji}
= \tfrac{1}{4}\bigl(I_i^{\boldsymbol{A}}+I_j^{\boldsymbol{A}}\bigr)
\qquad (i\ne j).
```

```@raw html
<details><summary>The two tensors as 6×6 Kelvin–Mandel matrices</summary>
```

In the frame ``(\underline{e}^{\boldsymbol{A}}_i)``, with ``(\rho_1,\rho_2,\rho_3)
= (a,b,c)`` and the components ordered ``11, 22, 33, 23, 31, 12``, the formulas
above fill

```math
\begin{aligned}
&\mathrm{Mat}\bigl(\mathbb{U}^{\boldsymbol{A}}\bigr) =
\\
&\qquad
\begin{pmatrix}
\frac{3(I_1-a^2I_{11})}{2} & \frac{I_2-a^2I_{12}}{2} & \frac{I_1-c^2I_{31}}{2} & 0 & 0 & 0\\
\frac{I_2-a^2I_{12}}{2} & \frac{3(I_2-b^2I_{22})}{2} & \frac{I_3-b^2I_{23}}{2} & 0 & 0 & 0\\
\frac{I_1-c^2I_{31}}{2} & \frac{I_3-b^2I_{23}}{2} & \frac{3(I_3-c^2I_{33})}{2} & 0 & 0 & 0\\
0 & 0 & 0 & I_3-b^2I_{23} & 0 & 0\\
0 & 0 & 0 & 0 & I_1-c^2I_{31} & 0\\
0 & 0 & 0 & 0 & 0 & I_2-a^2I_{12}
\end{pmatrix}
\end{aligned}
```

```math
\mathrm{Mat}\bigl(\mathbb{V}^{\boldsymbol{A}}\bigr) =
\operatorname{diag}\Bigl(I_1,\ I_2,\ I_3,\
\tfrac{I_2+I_3}{2},\ \tfrac{I_3+I_1}{2},\ \tfrac{I_1+I_2}{2}\Bigr),
```

the superscript ``\boldsymbol{A}`` being dropped. The shear entries are twice
the tensor components, ``2\,U^{\boldsymbol{A}}_{2323}``, ``2\,V^{\boldsymbol{A}}_{2323}``,
as Kelvin–Mandel storage requires. Substituting the cylinder column of the
tables, with the finite limits of the products ``a^2 I_{1j}`` noted above,
gives the two matrices of the cylinder below.

```@raw html
</details>
```

**Sphere** ``\boldsymbol{A}=\boldsymbol{1}``:

```math
\mathbb{U}^{\boldsymbol{1}} = \tfrac{1}{3}\mathbb{J} + \tfrac{2}{15}\mathbb{K},
\qquad
\mathbb{V}^{\boldsymbol{1}} = \tfrac{1}{3}\mathbb{I}.
```

**Infinite elliptic cylinder** ``a\to\infty``, axis
``\underline{e}^{\boldsymbol{A}}_1``, transverse semi-axes ``b\ge c``
[mura1987; §11.22](@cite) — substituting the cylinder column above:

```math
\begin{aligned}
\mathrm{Mat}\bigl(\mathbb{U}^{\mathrm{cyl}}\bigr) &=
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & \frac{c(b+2c)}{2(b+c)^{2}} & \frac{bc}{2(b+c)^{2}} & 0 & 0 & 0\\
0 & \frac{bc}{2(b+c)^{2}} & \frac{b(2b+c)}{2(b+c)^{2}} & 0 & 0 & 0\\
0 & 0 & 0 & \frac{bc}{(b+c)^{2}} & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix},
\\
\mathrm{Mat}\bigl(\mathbb{V}^{\mathrm{cyl}}\bigr) &=
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0\\
0 & \frac{c}{b+c} & 0 & 0 & 0 & 0\\
0 & 0 & \frac{b}{b+c} & 0 & 0 & 0\\
0 & 0 & 0 & \frac{1}{2} & 0 & 0\\
0 & 0 & 0 & 0 & \frac{b}{2(b+c)} & 0\\
0 & 0 & 0 & 0 & 0 & \frac{c}{2(b+c)}
\end{pmatrix}
\end{aligned}
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

## 2. Hill tensor in elasticity

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

This is the form of [willis1977](@citet); see also [mura1987](@citet).

### Properties

Scaling ``\boldsymbol{A}`` by ``\lambda>0`` multiplies ``\det\boldsymbol{A}`` and
``\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}`` by the same ``\lambda^{3}``:
``\mathbb{P}`` depends on the shape of the inclusion through two aspect ratios
and the orientation of its axes, not on its size.

``\mathbb{P}`` is major-symmetric and positive definite on symmetric
second-order tensors, for any positive definite ``\mathbb{C}``. The Eshelby
tensor ``\mathbb{S}^{\mathrm{E}} = \mathbb{P}:\mathbb{C}`` is not major-symmetric
in general. It is similar to ``\mathbb{C}^{1/2}:\mathbb{P}:\mathbb{C}^{1/2}``, so
its eigenvalues are real and positive; being non-symmetric, its quadratic form is
not controlled by them. The stress counterpart
``\mathbb{Q} = \mathbb{C}-\mathbb{C}:\mathbb{P}:\mathbb{C}`` is defined in
[The Eshelby inclusion problem](@ref th-eshelby-problem).

```@raw html
<details><summary>Why ℙ is positive definite</summary>
```

For a symmetric ``\boldsymbol{\varepsilon}`` and a symmetric ``\boldsymbol{M}``,
``\boldsymbol{\varepsilon}:\bigl(\underline{\xi}\stackrel{s}{\otimes}\boldsymbol{M}
\stackrel{s}{\otimes}\underline{\xi}\bigr):\boldsymbol{\varepsilon}
= (\boldsymbol{\varepsilon}\cdot\underline{\xi})\cdot\boldsymbol{M}\cdot
(\boldsymbol{\varepsilon}\cdot\underline{\xi})``. Hence, from the
``\underline{\xi}`` form,

```math
\boldsymbol{\varepsilon}:\mathbb{P}:\boldsymbol{\varepsilon}
= \frac{\det\boldsymbol{A}}{4\pi}
\int_{\|\underline{\xi}\|=1}
\frac{(\boldsymbol{\varepsilon}\cdot\underline{\xi})\cdot
      \bigl(\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}\bigr)^{-1}\cdot
      (\boldsymbol{\varepsilon}\cdot\underline{\xi})}
     {\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}}\,\mathrm{d}S_{\xi}.
```

The acoustic tensor is symmetric positive definite, since
``\underline{v}\cdot(\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi})\cdot\underline{v}
= (\underline{v}\stackrel{s}{\otimes}\underline{\xi}):\mathbb{C}:
(\underline{v}\stackrel{s}{\otimes}\underline{\xi}) > 0`` for ``\underline{v}\ne\underline{0}``,
and so is its inverse. The integrand is therefore non-negative, and the
integral, of a continuous integrand, vanishes only if
``\boldsymbol{\varepsilon}\cdot\underline{\xi} = \underline{0}`` for every unit
``\underline{\xi}``, that is ``\boldsymbol{\varepsilon} = \boldsymbol{0}``.

```@raw html
</details>
```

### [Reduction to a sphere](@id th-hill-reduction-sphere)

Every ellipsoid can be described by a symmetric positive definite
``\boldsymbol{A} = \boldsymbol{R}\cdot\mathrm{diag}(a,b,c)\cdot\boldsymbol{R}^{T}``,
its semi-axes and their orientation. In the ``\underline{\zeta}`` form above,
the shape then enters only through the vector
``\boldsymbol{A}^{-1}\cdot\underline{\zeta}``, and pulling
``\boldsymbol{A}^{-1}`` out of the integral moves it onto the stiffness: the
integrand becomes that of a **sphere** in the medium

```math
\tilde{\mathbb{C}} = \bigl(\boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}\bigr)
:\mathbb{C}:
\bigl(\boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}\bigr),
```

which the affine stretch mapping the ellipsoid onto the unit ball makes of
``\mathbb{C}``. The Hill tensor of the ellipsoid follows from that of the sphere
by the same tensor on each side
[barthelemyIJES2020_hilltrans; eq. 8](@cite):

```math
\boxed{\;
\mathbb{P}(\boldsymbol{A},\mathbb{C})
= \bigl(\boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}\bigr)
:\mathbb{P}(\boldsymbol{1},\tilde{\mathbb{C}}):
\bigl(\boldsymbol{A}^{-1}\stackrel{s}{\boxtimes}\boldsymbol{A}^{-1}\bigr).
\;}
```

The same reduction was reached by [pouya2000, pouya2006](@citet) from the field
equations. It has two uses. A numerical method need only treat the sphere, where
the denominator ``\|\boldsymbol{A}\cdot\underline{\xi}\|^{3}`` of the
``\underline{\xi}`` form is one. And a family of matrices that the stretch maps
onto one whose spherical Hill tensor is known inherits a closed form: an
elliptically orthotropic matrix
``\lambda\,\boldsymbol{D}\otimes\boldsymbol{D}+2\mu\,\boldsymbol{D}\stackrel{s}{\boxtimes}\boldsymbol{D}``
maps onto an isotropic one [barthelemyIJES2020_hilltrans; eq. 14](@cite), and a
transversely isotropic matrix coaxial with a spheroid stays transversely
isotropic, which gives the closed form of
[the section below](@ref th-hill-ti-coaxial). The tutorial
[Hill tensors by reduction to a sphere](@ref tut-hill-reduction) computes
several Hill tensors both ways, and the application
[Hill tensors by transformation](@ref app-hill-transformation-reduction) derives
the reduction, and its elliptically orthotropic case, step by step after the
paper.

### Algorithms for an arbitrary anisotropy

Inverting the acoustic tensor ``\underline{\xi}\cdot\mathbb{C}\cdot\underline{\xi}``
pointwise is the source of all the computational work: in general no closed
form exists and one resorts to numerical integration
[ghahremani1977, gavazzi1990, masson2008](@cite). `MeanFieldHomogenization`
offers three algorithm traits, mirroring the Echoes `NUMINT` / `RESIDUES`
options:

- **`NestedQuadGK`** — nested adaptive 1-D Gauss–Kronrod quadratures,
  type-generic and always available, the `:auto` choice whatever the
  coefficient type and whatever else is loaded.
- **`DECUHR`** — the surface integral is evaluated by the adaptive cubature for
  singular integrands of [espelid1994](@citet). ForwardDiff-safe, on an explicit
  `method = :decuhr` once its (weak-dependency) extension is loaded.
- **`Residue`** — the inner ``\varphi`` integral is reduced to a sum of residues
  at the roots of the acoustic polynomial by the Cauchy theorem, leaving a
  single 1-D quadrature [masson2008](@cite). With the substitution
  ``t = \cot(\theta/2)`` of that paper, the inner integral is that of a rational
  function of ``t`` over ``(0, \infty)``, which is minus the sum of the residues
  of its product with ``\log t``. Roots that coincide or nearly do, as for an
  isotropic or transversely isotropic reference along the ellipsoid axes or a
  reference close to one, are summed as a group (below), so that the sum keeps
  the accuracy of separated roots. The method is `Float64`-only (the roots are
  the eigenvalues of a companion matrix, which ForwardDiff does not
  differentiate) and reachable on an explicit `method = :residues` only.

The residue algorithm is the one whose accuracy depends most on how it is
evaluated; the next section derives it in full.

#### [Residue sums over groups of poles](@id th-hill-residue-groups)

**The inner integral as a rational one.** At an angle ``\varphi`` of the outer
quadrature, the directions of integration form the line
``\underline{\zeta}(\xi) = \underline{\alpha}_0 + \xi\,\underline{\alpha}_1``,
``\xi\in\mathbb{R}``, and the inner integral is a combination of the seven
moments

```math
W_n = \int_{-\infty}^{\infty} \frac{\xi^n\,\mathrm{d}\xi}{2\,(1+\xi^2)^{3/2}\,Q(\xi)},
\qquad n = 0, \dots, 6,
```

where ``Q(\xi) = \det\bigl(\underline{\zeta}(\xi)\cdot\mathbb{C}\cdot\underline{\zeta}(\xi)\bigr)``
is the acoustic polynomial, of degree 6, positive on the real line for a strongly
elliptic matrix, and the power ``3/2`` comes from the normalization of the
direction. That power puts branch points at ``\xi = \pm i``, which a residue
theorem cannot take as they are. The substitution of [masson2008](@citet),
``\xi = (t^2 - 1)/(2t)``, removes them: ``t\mapsto\xi`` maps ``(0,\infty)`` onto
``\mathbb{R}`` with ``1 + \xi^2 = \bigl((1+t^2)/(2t)\bigr)^2``, and

```math
\begin{gathered}
W_n = 2\int_0^{\infty} \frac{E_n(t)}{D(t)}\,\mathrm{d}t,
\\
E_n(t) = t^{7-n}\Bigl(\frac{t^2-1}{2}\Bigr)^{n},
\qquad
D(t) = (1+t^2)^2\,t^6\,Q\bigl(\xi(t)\bigr),
\end{gathered}
```

a rational function whose denominator, of degree 16, has the double roots
``t = \pm i``, the images of the branch points, and the two preimages
``t = z \pm \sqrt{1 + z^2}`` of each root ``z`` of ``Q``, the roots of
``t^2 - 2zt - 1 = 0``, whose product is ``-1``.

```@raw html
<details><summary>Proof of the change of variable</summary>
```

With ``\xi = (t^2-1)/(2t)``, ``\mathrm{d}\xi/\mathrm{d}t = (t^2+1)/(2t^2) > 0``,
so that ``\xi`` increases from ``-\infty`` to ``\infty`` on ``(0,\infty)``, and
``(1+\xi^2)^{1/2} = (1+t^2)/(2t)`` for ``t > 0``. Then

```math
\frac{\xi^n\,\mathrm{d}\xi}{2\,(1+\xi^2)^{3/2}\,Q(\xi)}
= \frac{\xi^n}{2\,Q(\xi)}\,\frac{8t^3}{(1+t^2)^3}\,\frac{1+t^2}{2t^2}\,\mathrm{d}t
= \frac{2\,t\,\xi^n}{(1+t^2)^2\,Q(\xi)}\,\mathrm{d}t,
```

and multiplying above and below by ``t^6`` gives ``2E_n/D``, since
``t^7\xi^n = t^{7-n}\bigl((t^2-1)/2\bigr)^n`` and ``t^6 Q(\xi)`` is a polynomial
in ``t``.

```@raw html
</details>
```

**The integral over a half line.** For a rational ``F = E/D`` with
``\deg D \ge \deg E + 2`` and no root of ``D`` on ``[0, \infty)``,

```math
\int_0^{\infty} F(t)\,\mathrm{d}t = -\sum_{w} \operatorname{Res}_{w}\bigl(\log(w)\,F(w)\bigr),
```

the sum running over all the roots of ``D``, with the logarithm cut along the
positive axis, ``\arg w \in (0, 2\pi)``. This is the argument of
[masson2008; eqs. 17–19](@citet), who integrates the product of ``F`` with a
logarithm along a keyhole contour (his Fig. 2) and writes the sum for simple
roots of ``Q``, with formulas of their own for a root at ``i`` and for roots that
coincide (his eqs. 21–22 and those after), the case of two roots at ``i`` being
left open. The sums over groups of poles below take every one of these cases,
and the roots that nearly coincide, in one formula. As ``D`` has real
coefficients its roots come in conjugate pairs, and the sum reduces to the upper
half plane:

```math
\int_0^{\infty} F(t)\,\mathrm{d}t
= -\Bigl(2\,\mathrm{Re}\sum_{\mathrm{Im}\,w>0}\operatorname{Res}_{w}(\log F)
+ 2\pi\,\mathrm{Im}\sum_{\mathrm{Im}\,w>0}\operatorname{Res}_{w}F\Bigr).
```

```@raw html
<details><summary>Proof</summary>
```

Integrate ``\log(w)F(w)`` along the keyhole contour made of the circle
``|w| = \mathcal{R}``, the circle ``|w| = r`` and both edges of the cut
``[r, \mathcal{R}]``. On the large circle the integrand is
``O(\mathcal{R}^{-2}\log\mathcal{R})`` and the circle has length
``2\pi\mathcal{R}``: its contribution vanishes as ``\mathcal{R}\to\infty``, and so
does that of the small one, ``O(r\log r)``, as ``r\to0``. On the upper edge
``\log w = \log t``, on the lower one ``\log w = \log t + 2\pi i``, run in the
opposite direction, so that the two edges give
``\int_0^\infty \bigl(\log t - \log t - 2\pi i\bigr)F(t)\,\mathrm{d}t = -2\pi i\int_0^\infty F``.
The residue theorem gives ``2\pi i\sum_w \operatorname{Res}_w(\log F)``, whence the
first formula.

For the second, ``F(\bar w) = \overline{F(w)}`` gives
``\operatorname{Res}_{\bar w}F = \overline{\operatorname{Res}_w F}``, and with the cut
along the positive axis ``\log\bar w = \overline{\log w} + 2\pi i`` for
``\mathrm{Im}\,w > 0``, an identity between functions analytic near ``\bar w``,
which holds for poles of any multiplicity:
``\operatorname{Res}_{\bar w}(\log F) = \overline{\operatorname{Res}_w(\log F)} + 2\pi i\,\overline{\operatorname{Res}_w F}``.
Adding the residues at ``w`` and ``\bar w`` leaves
``2\,\mathrm{Re}\operatorname{Res}_w(\log F) + i\,\mathrm{Im}\operatorname{Res}_w(\log F) + 2\pi i\,\overline{\operatorname{Res}_w F}``,
and the integral, real, keeps the real part,
``2\,\mathrm{Re}\operatorname{Res}_w(\log F) + 2\pi\,\mathrm{Im}\operatorname{Res}_w F``.

```@raw html
</details>
```

**The sum over a group of close poles.** A root of multiplicity ``k`` comes out
of any root finder as ``k`` roots about ``\epsilon^{1/k}`` apart, ``\epsilon``
the unit roundoff, and the residue at each is of the order of the inverse of
their distance. Taken one by one, the residues are large, of opposite signs, and
their sum keeps little of their accuracy; declared coincident below some
distance, they carry an error of the order of that distance. The residues of a
group are summed as a whole instead.

Let ``w_1, \dots, w_k`` be poles of ``fN/D`` close to one another, with ``N``
and ``D`` polynomials and ``f`` analytic around them (``f = \log`` or
``f = 1``), ``c`` their centroid, ``\delta_j = w_j - c``,
``\delta = \max_j|\delta_j|``, ``R(w) = D(w)/\prod_j(w - w_j)``, which does not
vanish near ``c``, and ``\rho`` the distance from ``c`` to the nearest
singularity of ``fN/R``: the other roots of ``D`` and, for ``f = \log``, the
branch point ``0``. For ``\delta < \rho``,

```math
\sum_{j=1}^{k}\operatorname{Res}_{w_j}\frac{fN}{D}
= \sum_{l\ge0} g_{l+k-1}\,h_l(\delta_1,\dots,\delta_k),
\qquad
\frac{fN}{R}(w) = \sum_{n\ge0} g_n\,(w-c)^n,
```

with ``h_l`` the complete homogeneous symmetric polynomial of degree ``l``, the
sum of all the monomials of degree ``l`` in the ``\delta_j``.

```@raw html
<details><summary>Proof, and the rate of the series</summary>
```

The sum of the residues in the group is ``\frac{1}{2\pi i}\oint_\Gamma fN/D``
along the circle ``\Gamma: |w - c| = r`` with ``\delta < r < \rho``. On
``\Gamma``, ``|\delta_j/(w - c)| < 1`` and

```math
\frac{1}{\prod_{j=1}^{k}(w - w_j)}
= \frac{1}{(w-c)^k}\prod_{j=1}^{k}\frac{1}{1 - \delta_j/(w-c)}
= \sum_{l\ge0}\frac{h_l(\delta_1,\dots,\delta_k)}{(w-c)^{k+l}},
```

the product of ``k`` geometric series, uniformly convergent on ``\Gamma``; the
Taylor series of ``fN/R`` converges uniformly there too, as ``r < \rho``. The
integral of the product picks the coefficient of ``(w - c)^{-1}``, that is the
terms ``n - k - l = -1``.

Coincident poles leave the term ``l = 0`` alone, ``g_{k-1}``, the residue at a
pole of order ``k``; a lone pole leaves ``g_0 = f(w_1)N(w_1)/D'(w_1)``. As
``|h_l| \le \binom{l+k-1}{k-1}\delta^l`` and, by the Cauchy estimates on a circle
of radius ``\rho' < \rho``, ``|g_n| \le M\rho'^{-n}``, the terms decrease as
``\binom{l+k-1}{k-1}(\delta/\rho')^l``: the series converges geometrically at the
rate ``\delta/\rho``, and the poles enter through symmetric functions of the
``\delta_j`` only, which, unlike the individual roots of a cluster, are well
conditioned: they are, up to sign, the coefficients of
``\prod_j(w - c - \delta_j)``, which the root finder returns accurate to
rounding.

```@raw html
</details>
```

The sum is evaluated by recurrences of fixed cost. The ``h_l`` follow from
multiplying the ``k`` geometric series one at a time, ``h_l \mathrel{+}= \delta_j h_{l-1}``.
The Taylor coefficients of ``1/R`` follow from dividing by each factor
``(c - w_m) + (w - c)`` in turn, ``g_p \leftarrow (g_p - g_{p-1})/(c - w_m)``,
those of ``\log`` from ``\log(c + u) = \log c - \sum_{p\ge1}(-u/c)^p/p``, and those
of ``N`` from its factored form. The series is cut at the first ``L`` with
``(\delta/\rho)^L`` below the unit roundoff. The groups are the largest nodes of
the single-linkage tree of the poles whose ratio ``\delta/\rho`` stays below
``1/10``, each pole going to the largest node that qualifies; a lone pole is a
group of one. A tree is needed: four roots spread around a double pole, as a
double root of ``Q`` at ``i`` gives, form a tight group of six that no pair, and
no group of three, qualifies to start.

**Conditioning, and the running error bound.** The sums above are exact, but
their evaluation is not always well conditioned. When a root ``z`` of ``Q``
approaches ``\pm i``, as it does for the directions of a plane of isotropy of
the matrix (around a sphere in an isotropic matrix, ``Q \propto (1+\xi^2)^3``),
its preimages approach the double pole ``t = i``, at the distance
``|\sqrt{1+z^2}|``. Poles then crowd around ``i`` at several scales: groups of
radius comparable to their distance to the others, or to the branch point
``0``, cannot be merged into one whose series converges, and their contributions
are larger than their sum by four or five orders of magnitude. The integral
itself stays as well conditioned as anywhere else: only its representation by
residues is not.

Each sum therefore carries a running error bound. The same recurrences are run
on the moduli of their operands, every difference turned into a sum; the result
bounds, term by term, the moduli of the quantities each step adds up, and the
rounding error of the sum is at most a small multiple of the unit roundoff
times that bound [higham2002](@cite). Against the same sums in 256-bit
arithmetic, on groups from a lone pole to an eightfold one, the error stayed
below the bound times the unit roundoff by a factor 2.8 at least; the package
takes four times that product. When the bound exceeds the tolerance asked of
the inner integral, an order and more below that of the outer quadrature, the
moments ``W_n`` are computed by an adaptive Gauss-Kronrod quadrature of their
definition in ``t`` instead, which is accurate there. A spheroid elongated along
the axis of a transversely isotropic matrix, at a relative tolerance of
``10^{-12}``, met such directions over a whole range of ``\varphi``: evaluated
anyway, the sums were ``10^{-11}`` off and kept the outer quadrature from
converging, a million evaluations and 35 s for one tensor; with the bound, it
takes 285 evaluations and 0.02 s, within ``2\times10^{-16}`` of the closed form.

For the line integral of the crack opening tensor, ``f = 1`` and the poles are
the roots of ``Q`` themselves: the same groups, the same bound, and the
quadrature of the same kernel when the bound exceeds the tolerance.

All three honor the `abstol` (default ``10^{-8}``), `reltol` (``10^{-6}``) and
`maxiters` (``10^{6}`` evaluations) keywords of [`hill_tensor`](@ref). At these
defaults `NestedQuadGK` is within about ``10^{-13}`` of the exact value — of a
closed form where one exists, of a tight reference checked by a second algorithm
otherwise — and `DECUHR` within ``10^{-11}`` to ``10^{-8}``; on a triaxial
ellipsoid ``2{:}1{:}0.5`` in a triclinic matrix `NestedQuadGK` takes about 1 ms,
the residues and `DECUHR` about 4 ms
([Performance vs Echoes](@ref dev-benchmarks)).

Flat ellipsoids, ``\omega = a_3/a_1 \ll 1``, are a case apart. The integrand is
regular there, so `DECUHR` runs as an adaptive Gauss rule; it gives the values of
the Echoes `NUMINT3D` option at equal tolerances. At the default tolerances its
error is ``10^{-5}`` to ``2\times10^{-5}`` at ``\omega = 10^{-4}`` and
``10^{-5}``. At tolerances ``10^{-13}/10^{-12}`` it is still ``1.3\times10^{-6}``
at ``\omega = 10^{-5}``, for about 0.5 s per tensor. `NestedQuadGK` stretches the
end ``z \to 1`` of its outer interval by ``z = 1-(1-u)^{\alpha}``, with
``\alpha = \max(1, -\log_{10}\omega)``. The integrand also varies over a layer
``z \sim \omega`` at the other end, so below ``\omega = 10^{-3}`` breakpoints at
``z = \omega, 10\,\omega, 100\,\omega`` give that layer intervals of its own.
The result then stays within the tolerance asked for down to
``\omega = 10^{-7}``: at most ``5\times10^{-7}`` at the defaults, and
``3\times10^{-16}`` at ``10^{-13}/10^{-12}`` in 2 to 4 ms.

Analytical paths exist in the literature for further anisotropy classes
[withers1989, pouya2000, pouya2006, suvorov2002](@cite) and are not all implemented yet.

### Isotropic matrix — the shape/moduli factorization

With bulk modulus ``k``, shear modulus ``\mu`` and first Lamé parameter
``\lambda = k-2\mu/3``, so that
``\mathbb{C} = 3k\,\mathbb{J}+2\mu\,\mathbb{K} = 3\lambda\,\mathbb{J}+2\mu\,\mathbb{I}``,
the general expression collapses to the form of [willis1977](@citet)

```math
\boxed{\;
\mathbb{P}\bigl(\boldsymbol{A},\,3\lambda\,\mathbb{J}+2\mu\,\mathbb{I}\bigr)
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
[mura1987; §11.22](@cite).

Implementation: `src/Elasticity/hill_3d_iso.jl` and
`src/Elasticity/hill_3d_cylinder_iso.jl`, selected by `method = :auto` when
``\mathbb{C}_0`` is a [`TensISO`](@extref TensND TensND.TensISO).

### [Transversely isotropic matrix coaxial with a spheroid](@id th-hill-ti-coaxial)

Let the spheroid have its axis along ``\underline{e}_3`` and the aspect ratio
``\omega`` (axial over transverse semi-axis), so that
``\boldsymbol{A} = \underline{e}_1\otimes\underline{e}_1 + \underline{e}_2\otimes\underline{e}_2
+ \omega\,\underline{e}_3\otimes\underline{e}_3`` up to a scale, and let the
matrix be transversely isotropic about the same axis, with the five constants
``C_{1111}, C_{1122}, C_{1133}, C_{3333}, C_{2323}``. The stretch of the
[reduction to a sphere](@ref th-hill-reduction-sphere) divides each component of
``\mathbb{C}`` by ``\omega`` to the number of its indices equal to 3: the
transformed matrix is still transversely isotropic, and so are the Hill tensors
of the sphere in it and of the spheroid in ``\mathbb{C}``
[barthelemyIJES2020_hilltrans; eqs. 18–20](@cite). The latter has five Walpole
coefficients (see [Conventions](@ref th-notation-bases)):

```math
\mathbb{P} = P_1\,\mathbb{W}_1 + P_2\,\mathbb{W}_2
           + P_3\,(\mathbb{W}_3+\mathbb{W}_4)
           + P_5\,\mathbb{W}_5 + P_6\,\mathbb{W}_6 .
```

The sphere in a transversely isotropic medium has a closed form
[barthelemyIJES2020_hilltrans; eqs. 24–29](@cite), and carried back through the
stretch it gives the spheroid [barthelemyIJES2020_hilltrans; eqs. 30–36](@cite).
The derivation goes in three steps, each of them carried out on free symbols in
the application [Hill tensors by transformation](@ref app-hill-transformation-ti).

```@raw html
<details><summary>Derivation: the sphere in the transformed medium, then the spheroid</summary>
```

*The transformed medium.* With
``\boldsymbol{A} = \underline{e}_1\otimes\underline{e}_1 + \underline{e}_2\otimes\underline{e}_2 + \omega\,\underline{e}_3\otimes\underline{e}_3``,
``\tilde{\mathbb{C}}`` is transversely isotropic about ``\underline{e}_3``, of
constants ``\tilde{C}_{1111} = C_{1111}``, ``\tilde{C}_{1122} = C_{1122}``,
``\tilde{C}_{1133} = C_{1133}/\omega^2``, ``\tilde{C}_{3333} = C_{3333}/\omega^4``
and ``\tilde{C}_{2323} = C_{2323}/\omega^2``, and the Walpole coefficients of
``\mathbb{P}`` are those of ``\tilde{\mathbb{P}} = \mathbb{P}(\boldsymbol{1},\tilde{\mathbb{C}})``
divided by ``\omega^4``, ``1``, ``\omega^2``, ``1`` and ``\omega^2``.

*The sphere.* On the unit sphere, in the colatitude ``\theta`` and the longitude
``\varphi`` with ``z = \cos\theta``, ``\tilde{\mathbb{P}}`` is the average of
``\underline{\zeta}\stackrel{s}{\otimes}\tilde{\boldsymbol{N}}\stackrel{s}{\otimes}\underline{\zeta}``,
``\tilde{\boldsymbol{N}} = (\underline{\zeta}\cdot\tilde{\mathbb{C}}\cdot\underline{\zeta})^{-1}``.
Medium and directions are invariant by the rotations about ``\underline{e}_3``,
so that the integrand at ``\varphi`` is that at ``\varphi = 0`` rotated, and its
average over ``\varphi`` is the projection of the integrand at ``\varphi = 0`` on
the Walpole basis: the integral over ``\varphi`` costs nothing. At ``\varphi = 0``
the acoustic tensor has the components

```math
\begin{aligned}
\tilde{N}_{11} &= \frac{z^2(\tilde{C}_{3333}-\tilde{C}_{2323}) + \tilde{C}_{2323}}{q(z)},&
\tilde{N}_{33} &= \frac{z^2(\tilde{C}_{2323}-\tilde{C}_{1111}) + \tilde{C}_{1111}}{q(z)},\\
\tilde{N}_{13} &= -\frac{z\sqrt{1-z^2}\,(\tilde{C}_{1133}+\tilde{C}_{2323})}{q(z)},&
\tilde{N}_{22} &= \frac{1}{r(z)},
\end{aligned}
```

over the two denominators
``q(z) = \tilde{C}_{2323}\tilde{C}_{3333}\bigl(z^2+\eta_1^2(1-z^2)\bigr)\bigl(z^2+\eta_2^2(1-z^2)\bigr)``
and ``r(z) = \tilde{C}_{2323}\bigl(z^2+\eta_3^2(1-z^2)\bigr)``, where
``\eta_1^2, \eta_2^2`` are the roots of the acoustic polynomial written with the
constants of ``\tilde{\mathbb{C}}`` and
``\eta_3^2 = (\tilde{C}_{1111}-\tilde{C}_{1122})/(2\tilde{C}_{2323})``. Each
projected coefficient is then an even polynomial in ``z`` over ``q`` or ``r``, and
the integral over ``z\in[-1,1]`` is a combination of

```math
\begin{gathered}
\mathcal{I}_k(\eta) = \int_{-1}^{1}\frac{z^k\,\mathrm{d}z}{z^2+\eta^2(1-z^2)},
\\
\mathcal{J}_k(\eta_1,\eta_2) = \int_{-1}^{1}\frac{z^k\,\mathrm{d}z}{\bigl(z^2+\eta_1^2(1-z^2)\bigr)\bigl(z^2+\eta_2^2(1-z^2)\bigr)},
\end{gathered}
```

``k = 0, 2, 4``, which are the integrals ``I_k(\eta^2)`` and
``J_{k,0}(\eta_1^2,\eta_2^2)`` of [Degenerate shapes](@ref th-degenerate-integrals).

*Back to the spheroid.* Substituting ``\omega^2 Y`` for the unknown divides the
acoustic polynomial of ``\tilde{\mathbb{C}}`` by ``\omega^2``, which turns it into
that of ``\mathbb{C}``: ``\eta_i = \omega\gamma_i``, and likewise
``\eta_3 = \omega\gamma_3``. Writing the constants of ``\tilde{\mathbb{C}}`` in
those of ``\mathbb{C}`` and dividing by the powers of ``\omega`` above gives the
coefficients below.

```@raw html
</details>
```

The acoustic parameters of the matrix are the square roots ``\gamma_1, \gamma_2``
of the two roots ``Z`` of

```math
\begin{gathered}
C_{2323}C_{3333}\,Z^2
+ \bigl(C_{1133}^2 + 2\,C_{1133}C_{2323} - C_{1111}C_{3333}\bigr)\,Z
+ C_{1111}C_{2323} = 0,
\\
\gamma_3^2 = \frac{C_{1111}-C_{1122}}{2\,C_{2323}},
\end{gathered}
```

taken with a positive real part. As ``\mathbb{C}`` is positive definite,
``\gamma_1`` and ``\gamma_2`` are either both real and positive or complex
conjugates, so that everything below is real. With
``J_n = \mathcal{J}_n(\omega\gamma_1, \omega\gamma_2)`` and
``I_n = \mathcal{I}_n(\omega\gamma_3)``, the Walpole coefficients are

```math
\begin{aligned}
P_1 &= \frac{(C_{2323}-\omega^2 C_{1111})\,J_4 + \omega^2 C_{1111}\,J_2}
            {2\,C_{2323}C_{3333}},\\
P_2 &= \frac{\omega^2}{4\,C_{2323}C_{3333}}\,\Bigl[(\omega^2 C_{2323}-C_{3333})\,J_4
            + (C_{3333}-2\,\omega^2 C_{2323})\,J_2\\
    &\qquad + \omega^2 C_{2323}\,J_0\Bigr],\\
P_3 &= \omega^2\,\frac{(C_{2323}+C_{1133})\,(J_4-J_2)}{2\sqrt2\,C_{2323}C_{3333}},\\
P_5 &= \frac{P_2}{2} + \omega^2\,\frac{I_0-I_2}{8\,C_{2323}},\\
P_6 &= \frac{1}{8\,C_{2323}C_{3333}}\,\Bigl[(\omega^4 C_{1111}+C_{3333}+2\,\omega^2 C_{1133})\,J_4\\
    &\qquad - 2\,\omega^2(\omega^2 C_{1111}+C_{1133})\,J_2 + \omega^4 C_{1111}\,J_0\Bigr]
       + \frac{I_2}{8\,C_{2323}}.
\end{aligned}
```

**A sphere** is coaxial with every axis: at ``\omega = 1`` these are the
coefficients of a sphere in a transversely isotropic matrix, those of the
transformed problem above with ``\tilde{\mathbb{C}} = \mathbb{C}``
[barthelemyIJES2020_hilltrans; eq. 29](@cite). A sphere in a [`TensTI`](@extref TensND TensND.TensTI) matrix
therefore takes this closed form whatever the axis of the matrix, with no
quadrature.

As written, the coefficients cancel near the sphere, at coincident roots (an
isotropic matrix stored as transversely isotropic), for a disc and for a needle,
and a square root of zero spoils the derivatives at an isotropic point.
[Degenerate shapes](@ref th-degenerate-ti-spheroid) rewrites them in a form
that stays exact to rounding everywhere, derivatives included, and that is the
form the package evaluates; the application transcribes the formulas as
written, and measures what they lose
([The script of the paper, in Julia](@ref app-hill-transformation-script)).

**Flat and needle limits.** As ``\omega\to0`` the spheroid flattens into a
penny-shaped crack and
``\mathbb{P} = \mathbb{P}^0 + \omega\,\mathbb{P}^1 + O(\omega^2)``, with
``\mathbb{P}^0 = \underline{e}_3\stackrel{s}{\otimes}(\underline{e}_3\cdot\mathbb{C}\cdot\underline{e}_3)^{-1}\stackrel{s}{\otimes}\underline{e}_3``,
of Walpole coefficients ``1/C_{3333}`` and ``1/(2C_{2323})`` on ``\mathbb{W}_1`` and
``\mathbb{W}_6``, and ``\mathbb{P}^1`` in closed form in ``\gamma_1, \gamma_2, \gamma_3``
[barthelemyIJES2020_hilltrans; eqs. 37–39](@cite). The crack compliance
``\mathbb{H} = \lim_{\omega\to0}\omega\,\mathbb{Q}^{-1}``, with
``\mathbb{Q} = \mathbb{C} - \mathbb{C}:\mathbb{P}:\mathbb{C}`` the second Hill
tensor ([`second_hill_tensor`](@ref)), depends on ``\mathbb{P}^1`` only and has
two Walpole coefficients
[barthelemyIJES2020_hilltrans; eqs. 40–41](@cite):

```math
\mathbb{H} = -\frac{\mathbb{W}_1}{C_{3333}^2 p^1_1 + 2C_{1133}^2 p^1_2 + 2\sqrt2\,C_{3333}C_{1133}\,p^1_3}
- \frac{\mathbb{W}_6}{4\,C_{2323}^2\,p^1_6},
```

the transversely isotropic case of the crack opening tensors of
[COD tensors](@ref th-cod-tensors). As ``\omega\to\infty`` the spheroid tends to
a circular cylinder and ``p_2 \to 1/(2C_{1111})``,
``p_5 \to 1/(4C_{1111}) + 1/(2(C_{1111}-C_{1122}))``,
``p_6 \to 1/(4C_{2323})``, ``p_1, p_3 \to 0``, with corrections in
``\ln\omega/\omega^2`` [barthelemyIJES2020_hilltrans; eq. 42](@cite). The
expansions are derived symbolically, and held to the package, in
[The flat spheroid and the crack](@ref app-hill-transformation-flat) and
[The needle](@ref app-hill-transformation-needle).

The dispatcher routes a `TensTI{4}` matrix combined with a coaxial
`Ellipsoid{3, Spherical|Prolate|Oblate}` to this path by default; coaxiality is
detected by `_ti_coaxial(C₀, ell)`. Non-coaxial spheroids and triaxial
ellipsoids fall back to the anisotropic default, i.e. a cubature.
Implementation: `src/Elasticity/hill_3d_ti_coaxial.jl`.

### Anisotropic matrix, cylinder limit

`Cylinder` is a first-class inclusion type here (extension over Echoes). For
an arbitrarily anisotropic matrix the residue algorithm does **not**
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
uses the residue reduction on the line integral.

## 3. Hill tensor in conductivity

The order-2 Hill tensor of the transport problem, defined in
[The Eshelby inclusion problem](@ref th-eshelby-problem) §3, is built on the
same integral ``\boldsymbol{I}^{\boldsymbol{A}}`` of §1 and, unlike its elastic
counterpart, has a closed form at any anisotropy of the reference medium.

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

This is the transformation derivation of [giraudMOM2019](@citet); an equivalent
Green's-function derivation is given in [barthelemyTIPM2009](@citet). Since
``\boldsymbol{A}\cdot\boldsymbol{K}^{-1/2}`` need not be symmetric, the
fictitious semi-axes and principal directions are obtained by diagonalizing
``\boldsymbol{K}^{-1/2}\cdot\boldsymbol{A}^{\!T}\!\cdot\boldsymbol{A}\cdot
\boldsymbol{K}^{-1/2}``.

### Isotropic matrix — immediate

If ``\boldsymbol{K} = k_0\,\boldsymbol{1}`` the prefactor comes straight out:

```math
\boldsymbol{P}(\boldsymbol{A}, k_0\,\boldsymbol{1})
= \frac{\boldsymbol{I}^{\boldsymbol{A}}}{k_0}.
```

For a sphere, ``\boldsymbol{I}^{\boldsymbol{1}} = \tfrac{1}{3}\boldsymbol{1}``
gives ``\boldsymbol{P} = \tfrac{1}{3k_0}\boldsymbol{1}`` and
``\boldsymbol{S}^{\mathrm{E}} = \tfrac{1}{3}\boldsymbol{1}`` — independent of ``k_0``.
Implementation: `src/Conductivity/hill_order2_3d.jl`.

## 4. Dispatch

The closed forms above are reached through one entry point, which selects the
algorithm from the inclusion type and from the type of the reference tensor.

Entry point [`hill_tensor`](@ref); shape tensor via [`shape_tensor`](@ref);
geometric auxiliaries via [`tens_IA`](@ref), [`tens_UA`](@ref),
[`tens_VA`](@ref).

| `(inclusion, C₀)` | `:auto` selects | alternatives | ForwardDiff |
| :---------------- | :-------------- | :----------- | :---------: |
| `Ellipsoid{3}, TensISO` | `Analytical` | — | ✓ |
| `Ellipsoid{3}, TensTI` (coaxial) | `Analytical` (MFH) | `:residues`, `:decuhr` | ✓ |
| `Ellipsoid{3}, AbstractTens{4,3}` | `NestedQuadGK` | `:residues`, `:decuhr`, `:nestedquadgk` | ✓ |
| `Cylinder, TensISO` | `Analytical` | — | ✓ |
| `Cylinder, AbstractTens{4,3}` | `CylinderQuadrature` | (residue degenerates) | ✓ |
| `Ellipsoid{2}, TensISO` | `Analytical` | — | ✓ |
| `Ellipsoid{2}, AbstractTens{4,2}` | `Analytical` (residue) | — | (Float64) |
| `Ellipsoid{3}, AbstractTens{2,3}` | `Analytical` (``\boldsymbol{K}^{-1/2}``) | — | ✓ |
| `Cylinder, AbstractTens{2,3}` | `Analytical` | — | ✓ |

Cylinder shape traits: `CircularCylindrical` when ``b=c`` (transversely
isotropic response, returned as `TensTI{4}` with axis
``\underline{e}^{\boldsymbol{A}}_1``) and `EllipticCylindrical` when ``b>c``
(orthotropic, returned as [`TensOrtho`](@extref TensND TensND.TensOrtho)). Practical usage is covered in the manual
page [Cylindrical inclusions](@ref man-cylindrical-inclusions).

## Where to go next

The Hill tensor is the response of an inclusion to a prescribed polarization.
The next page, [Concentration and contribution tensors](@ref th-concentration),
turns it into the response of an inhomogeneity to a remote load and into the
contribution of each phase to the effective stiffness. The flat limits
announced at the top of this page are taken in
[Crack opening displacement and compliance](@ref th-cod-tensors), and the
tutorial [Hill polarization tensors in practice](@ref tut-hill-tensors)
evaluates these closed forms and compares the algorithms of §4 on an
anisotropic matrix.
