# [The finite Eshelby cell with a corrected boundary condition](@id th-corrected-cell)

When a morphology has no closed-form Eshelby solution, its response can be
computed on a **finite** cell — the inclusion inside a ball of matrix of radius
``R`` — and fed to the schemes through the
[custom-inclusion contract](@ref man-custom-inclusions). The difficulty is that
Eshelby's problem is posed on an *infinite* medium. This page states the
first-order correction of [adessinaIJES2017](@cite), which removes the
truncation bias, in the general form and in the two declinations the package
implements.

## The finite-size bias

We want the response of an **infinite** medium, but we can only mesh a
**finite** ball ``\Omega`` of radius ``R``. The obvious boundary condition is
the remote field itself,

```math
\underline{u}\big|_{\partial\Omega} = \boldsymbol{E}\cdot\underline{x} ,
\qquad \boldsymbol{E} = \mathbb S_0 : \boldsymbol\Sigma ,
```

but it *clamps* the perturbation radiated by the crack: the boundary is not
allowed to move the way the infinite medium would let it. The apparent opening
therefore carries a bias of order ``O\bigl((a/R)^3\bigr)``, which is why an
uncorrected computation needs `R/a` between 10 and 40 before it can be trusted.

```@setup cell
# Drawn in a `@setup` block: the subject of this page is the correction, not the
# `Plots` calls that sketch its geometry.
using Plots
gr()  # headless backend; GKSwstype is set to "100" in make.jl

# The geometry the correction is about: an inclusion of size a at the center of a
# meshed ball of radius R, with the boundary condition applied on ∂Ω.
R, a = 1.0, 0.24
θ = range(0, 2π; length = 200)

p = plot(; aspect_ratio = 1, framestyle = :none, legend = false,
    size = (620, 400), xlims = (-1.35, 1.5), ylims = (-1.25, 1.25))
plot!(p, R .* cos.(θ), R .* sin.(θ); lw = 2, ls = :dash, c = :steelblue)
plot!(p, a .* cos.(θ), 0.09a .* sin.(θ); seriestype = :shape,
    fillcolor = :sienna, lc = :sienna, fillalpha = 0.85)

# The two radii, measured from the same center. The inclusion half-width is
# dimensioned just below the crack, which would otherwise hide the line.
plot!(p, [0, a], [-0.11, -0.11]; lw = 1.6, c = :sienna)
plot!(p, [0, 0], [-0.145, -0.075]; lw = 1.6, c = :sienna)
plot!(p, [a, a], [-0.145, -0.075]; lw = 1.6, c = :sienna)
plot!(p, [0, R * cosd(35)], [0, R * sind(35)]; lw = 2, c = :steelblue)
annotate!(p, [(0.5a, -0.24, text("a", 11, :sienna)),
              (0.52R * cosd(35) - 0.04, 0.52R * sind(35) + 0.10,
                  text("R", 11, :steelblue)),
              (0.0, -0.62, text("Ω  (meshed)", 10, :gray35)),
              (0.0, 1.10, text("∂Ω :  u = E·x  +  ∇G(x) : (V ⟨p⟩)", 11, :black)),
              (0.0, -1.12, text("the second term is what the correction adds", 9, :gray45))])
nothing # hide
```

```@example cell
p # hide
```

Without the dipole term the boundary carries the remote field alone, and the ball
has to be made large enough for the neglected term to fall below the target
accuracy. With it, the boundary already knows what the infinite medium would do,
and `R/a = 5` suffices — the default `radius_ratio` of the finite-element
backends ([FE inclusions](@ref man-fe-inclusions)) — the price being that
``\langle p\rangle`` appears on both sides.

## The general fixed point

The exact infinite-medium solution is

```math
u(x) = E\cdot x + \int_{\mathcal D}\nabla G(x - x') : p(x')\,\mathrm d\Omega',
\qquad p = \sigma - \mathbb C_0 : \varepsilon ,
```

whose far field, since ``\nabla G(x-x') \to \nabla G(x)`` when
``\|x\| \gg a``, collapses to a single **force dipole**:

```math
u(x) \;\approx\; E\cdot x + \nabla G(x) : \Bigl(V_{\mathcal D}\,
      \langle p\rangle_{\mathcal D}\Bigr),
\qquad
\frac{\bigl\|V_{\mathcal D}\,\nabla G(x):\langle p\rangle\bigr\|}
     {\|E\cdot x\|} = O\!\left(\frac{V_{\mathcal D}}{\|x\|^{3}}\right).
```

Imposing ``u = E\cdot x`` on a sphere of radius ``R`` therefore leaves an
``O\bigl((a/R)^3\bigr)`` bias. Adding the dipole term removes it, at the price of a fixed point,
because ``\langle p\rangle`` is *itself* an output of the problem.

Split by linearity into two boundary-value problems on the truncated cell:

```math
\begin{aligned}
u|_{\partial\Omega} &= E\cdot x
  &&\Longrightarrow&
  \langle\varepsilon^E\rangle_{\mathcal D} &= \mathbb A^E : E, &
  \langle\sigma^E\rangle_{\mathcal D} &= \mathbb B^E : E, \\
u|_{\partial\Omega} &= \nabla G(x) : (V_{\mathcal D}\,P)
  &&\Longrightarrow&
  \langle\varepsilon^p\rangle_{\mathcal D} &= \mathbb A^p : P, &
  \langle\sigma^p\rangle_{\mathcal D} &= \mathbb B^p : P .
\end{aligned}
```

Superposing and demanding that ``P`` be the polarization it generates,

```math
P = \langle\sigma - \mathbb C_0 : \varepsilon\rangle_{\mathcal D}
  = (\mathbb B^E - \mathbb C_0 : \mathbb A^E) : E
  + (\mathbb B^p - \mathbb C_0 : \mathbb A^p) : P ,
```

which is *linear* in ``P`` and solves in closed form:

```math
\boxed{\;
\mathbb X = \bigl[\mathbb I - (\mathbb B^p - \mathbb C_0 : \mathbb A^p)\bigr]^{-1}
            : (\mathbb B^E - \mathbb C_0 : \mathbb A^E),
\qquad
\mathbb A = \mathbb A^E + \mathbb A^p : \mathbb X,
\qquad
\mathbb B = \mathbb B^E + \mathbb B^p : \mathbb X. \;}
```

Two declinations are implemented, and they differ only in what carries the
polarization:

| | solid inclusion | crack |
| :--- | :--- | :--- |
| unknown | ``\mathbb X`` on the Kelvin basis | ``\boldsymbol{B}_\infty`` |
| solves | 6 + 6, or 2 + 2 per Fourier mode | 3 + 3 |
| closes on | ``\mathbb A = \mathbb A^E + \mathbb A^p:\mathbb X`` | ``\boldsymbol{B}_\infty = (1 - \boldsymbol{B}_u)^{-1}\boldsymbol{B}_s`` |
| used by | [`FEExcenteredSphere`](@ref app-recycled-aggregate) | [`FEEllipticCrack`](@ref man-fe-inclusions) |

In the axisymmetric case each fixed point lives *inside* one Fourier mode,
since the dipole of a modal polarization radiates in the same mode — so
``\mathbb X`` is ``2\times2`` for mode 0 and a scalar for modes 1 and 2.


## The dipole fields, in closed form

For an isotropic reference medium both Green functions are closed forms, so the
boundary data costs nothing. With ``r = \|x\|``, ``\underline{n} = x/r`` and
``M = V_{\mathcal D} P`` the polarization **moment**, the elastic field is
[`dipole_displacement_iso`](@ref MeanFieldHomogenization.Core.dipole_displacement_iso):

```math
u(x) = \frac{\partial G_{ij}}{\partial x_k}(x)\,M_{jk}
     = \frac{1}{16\pi\mu(1-\nu)r^{2}}
       \Bigl[-2(1-2\nu)\,M\!\cdot\!\underline{n} + \mathrm{tr}(M)\,\underline{n}
             - 3(\underline{n}\!\cdot\! M\!\cdot\!\underline{n})\,\underline{n}\Bigr],
```

Written out for a symmetric moment, the gradient of the Kelvin solution is

```math
G_{ij}(\underline{x}) = \frac{A}{r}\bigl[(3-4\nu)\,\delta_{ij} + n_i n_j\bigr],
\qquad
\frac{\partial G_{ij}}{\partial x_k}
  = \frac{A}{r^{2}}\bigl[-(3-4\nu)\,\delta_{ij}n_k + \delta_{ik}n_j + \delta_{jk}n_i - 3\,n_i n_j n_k\bigr],
\qquad A = \frac{1}{16\pi\mu(1-\nu)} ,
```

and the transport one, with ``G = 1/(4\pi k_0 r)``,

```math
T(\underline{x}) = \frac{\partial G}{\partial x_k}(\underline{x})\,M_k
     = -\frac{\boldsymbol M\cdot\underline{x}}{4\pi k_0 r^{3}} .
```

These are [`green_gradient_iso`](@ref MeanFieldHomogenization.Core.green_gradient_iso) and
[`dipole_displacement_iso`](@ref MeanFieldHomogenization.Core.dipole_displacement_iso).
They are also why the reference medium must be **isotropic**: for arbitrary
anisotropy ``\nabla\mathbb G`` would come from the Willis angular integral, or
from the Pan–Chou closed form in the transversely isotropic case, neither of
which is implemented.

## The crack declination (3 + 3)

### The crack radiates as an elastic dipole

A displacement discontinuity ``[\![\underline{u}]\!]`` across a surface ``S`` of
normal ``\underline{n}`` is mechanically equivalent to a distribution of
**force dipoles** of density ``\mathbb C_0 : (\underline{n}
\stackrel{s}{\otimes} [\![\underline{u}]\!])``. Seen from far away the whole crack
is therefore a single point dipole of intensity

```math
\boldsymbol\Pi = \int_S \mathbb C_0 : \bigl(\underline{n}\stackrel{s}{\otimes}[\![\underline{u}]\!]\bigr)\,\mathrm dS
   = b\,S_f\; \mathbb C_0 : \bigl(\underline{n}\stackrel{s}{\otimes}\underline{U}\bigr),
\qquad \underline{U} = \frac{\langle[\![\underline{u}]\!]\rangle}{b},
\quad S_f = \pi a b ,
```

with ``b`` the semi-minor axis — the normalization `cod_tensor` uses. The field
it generates is that dipole contracted with the gradient of the Green function,
so the *correct* far field is

```math
\underline{u}(\underline{x})\;\underset{\|\underline{x}\|\to\infty}{\approx}\;
  \boldsymbol{E}\cdot\underline{x}
  \;-\; b\,S_f\,\bigl(\nabla\mathbb G(\underline{x}):\mathbb C_0\cdot\underline{n}\bigr)\cdot\underline{U} .
```

The idea of [adessinaIJES2017](@cite) is to put that second term **into the boundary data**.

### Closing the loop

The dipole intensity ``\underline{U}`` is itself unknown — it *is* what we are
trying to compute. Linearity resolves the circularity. Writing
``\langle[\![\underline{u}]\!]\rangle/b = \boldsymbol{B}\cdot\underline{t}`` with
``\underline{t} = \boldsymbol\Sigma\cdot\underline{n}``, solve two families of
three problems on the same mesh:

| Family | Boundary condition | Yields |
|:--|:--|:--|
| **traction**, ``\boldsymbol\Sigma^{(i)}\cdot\underline{n} = \underline{e}_i`` | ``\underline{u}\big\|_{\partial\Omega} = (\mathbb S_0:\boldsymbol\Sigma^{(i)})\cdot\underline{x}`` | columns of ``\boldsymbol{B}_s`` |
| **dipole**, unit intensity ``\underline{e}_m`` | ``\underline{u}\big\|_{\partial\Omega} = -b\,S_f\bigl(\nabla\mathbb G:\mathbb C_0\cdot\underline{n}\bigr)\cdot\underline{e}_m`` | columns of ``\boldsymbol{B}_u`` |

``\boldsymbol{B}_s`` is the COD tensor of the *truncated* cell; ``\boldsymbol{B}_u`` is
its response to the crack's own far field. Superposing,

```math
\underline{U} = \boldsymbol{B}_s\cdot\underline{t} + \boldsymbol{B}_u\cdot\underline{U}
\qquad\Longrightarrow\qquad
\underline{U} = (\boldsymbol{1} - \boldsymbol{B}_u)^{-1}\,\boldsymbol{B}_s\cdot\underline{t} ,
```

so the infinite-medium COD tensor follows in **one step** — no iteration:

```math
\boxed{\;\boldsymbol{B}_\infty = (\boldsymbol{1} - \boldsymbol{B}_u)^{-1}\cdot\boldsymbol{B}_s\;}
```

### What is solved for a crack

Pure linear elasticity, ``\int_\Omega \boldsymbol\sigma(\underline{u}):
\boldsymbol\varepsilon(\underline{v})\,\mathrm d\Omega = 0``, no body force. The
crack is a **zero-thickness discontinuity** — duplicated nodes — whose lips are
traction-free *naturally*: no interface term, no multiplier, no contact
condition. Only the outer sphere carries a Dirichlet condition, and its value
is the whole method.

Per evaluation: **one** assembly and **one** Cholesky factorization of the
free-free block, reused for all six right-hand sides. The mean opening is then
measured as a surface integral of the jump over each lip, with no assumption on
the opening profile:

```math
\underline{U} = \frac{1}{S_f\,b}\left(\int_{\Gamma^+}\underline{u}\,\mathrm dS - \int_{\Gamma^-}\underline{u}\,\mathrm dS\right).
```

