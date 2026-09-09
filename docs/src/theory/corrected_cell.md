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

| | solid inclusion | crack | cavity |
| :--- | :--- | :--- | :--- |
| unknown | ``\mathbb X`` on the Kelvin basis | ``\boldsymbol{B}_\infty`` | ``\mathbb A`` itself |
| solves | 6 + 6, or 2 + 2 per Fourier mode | 3 + 3 | 6 + 6, or 2 + 2 and 1 + 1 per mode |
| closes on | ``\mathbb A = \mathbb A^E + \mathbb A^p:\mathbb X`` | ``\boldsymbol{B}_\infty = (1 - \boldsymbol{B}_u)^{-1}\boldsymbol{B}_s`` | ``\mathbb A = (\mathbb I - \mathbb A_u\mathbb F)^{-1}\mathbb A_s`` |
| used by | [`FEExcenteredSphere`](@ref app-recycled-aggregate), [`FEAxiLayeredSpheroid`](@ref tut-axi-layered-spheroid) | [`FEEllipticCrack`](@ref man-fe-inclusions) | [`FESupershapePore`](@ref man-fe-inclusions), [`FEAxiSupershapePore`](@ref app-concave-pores) |

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
anisotropy ``\nabla\mathbb G`` would come from the Willis integral, or
from the Pan–Chou closed form in the transversely isotropic case, neither of
which is implemented.

## The pore declination

### A cavity is its own polarization source

The general fixed point above needs the polarization of the inclusion, and for a
cavity that quantity is not something extra to compute: it *is* the answer.
A cavity carries no stress, so

```math
\underline{\underline{P}}
  = \langle \underline{\underline\sigma}
      - \mathbb C_0 : \underline{\underline\varepsilon} \rangle_{\mathcal D}
  = -\,\mathbb C_0 : \langle \underline{\underline\varepsilon} \rangle_{\mathcal D}
  = -\,\mathbb C_0 : \mathbb A : \underline{\underline E},
```

the very tensor being measured. So the two-stage construction — solve for
``\mathbb A^E``, solve for ``\mathbb A^p``, then invert for ``\mathbb X`` —
collapses. Writing ``\mathbb A_s`` for the response to the remote field alone
and ``\mathbb A_u`` for the response to a unit dipole, the loop closes in one
inversion:

```math
\boxed{\;\mathbb A = (\mathbb I - \mathbb A_u\,\mathbb F)^{-1}\,\mathbb A_s\;},
\qquad
\mathbb F = -\,V_{\mathcal D}\,\mathbb C_0
\quad\text{(elasticity)},\qquad
\boldsymbol F = -\,k_0 V_{\mathcal D}\,\boldsymbol 1
\quad\text{(transport)} .
```

There is no ``\mathbb C_1`` anywhere in it, which is the point: `inv(C₁)` is
meaningless for a cavity, and this is why
[`FESupershapePore`](@ref man-fe-inclusions) declares itself a *heterogeneous*
inclusion and lets the package's exact identities collapse to ``\mathbb N =
-\mathbb C_0 : \mathbb A`` and ``\mathbb H = \mathbb A : \mathbb S_0``.

### The two signs, and why they are not inconsistent

``\mathbb F`` carries a minus in both physics, and the transport one looks like
it should not. It is structural. Elasticity pairs
``\underline{\underline\sigma} = +\,\mathbb C_0 : \underline{\underline\varepsilon}``
with a polarization ``\underline{\underline P}``, while transport pairs
``\underline q = -\,k_0 \underline\nabla T``: splitting the flux as
``-\underline q = k_0\underline\nabla T + \underline{\pi}'`` gives
``\underline{\pi}' = -\underline{\pi}``, and it is ``\underline{\pi}'`` that
belongs with the temperature ``T = \underline\nabla G\cdot \underline M``.

The failure mode is worth remembering, because it does not look like a sign
error. **A wrong sign leaves exactly twice the truncation bias instead of
none** — the correction is applied backwards, so it adds what it should have
subtracted. The result still converges under refinement, just to the wrong
place, and only a radius sweep exposes it.

### What is solved, and on what volume

Zero traction on the cavity wall in elasticity, zero normal flux in transport.
Nothing is meshed inside: the cell is a shell of matrix between the shape and
the outer sphere. Two consequences the implementation leans on — the
**stress-side localization is identically zero**, so gate B is served by one
tensor rather than two; and the normalizing volume is the **curved** volume of
the cavity, `fe_cell_curved_volume`, not the volume of the flat triangulation
that bounds the mesh, which differs from it by a percent at usable refinements.

The diagnostic is ``\|\mathbb A_u \mathbb F\|``, which is
``O\!\left((a/R)^3\right)`` — measured at a log-log slope of ``-2.96`` against
``-3`` from theory.

### Or two dimensions, when the body is a solid of revolution

An octant divides the cost by eight. Separating the azimuth into Fourier modes
divides it by orders of magnitude, and for an axisymmetric cavity that is the
route [`FEAxiSupershapePore`](@ref app-concave-pores) takes. Each mode is a
problem on the **meridian half-plane**, and the modes do not couple, so a
macroscopic loading excites exactly one of them:

| loading | mode | what it yields |
|:--|:--|:--|
| ``\varepsilon = (\underline e_1\otimes\underline e_1 + \underline e_2\otimes\underline e_2)/\sqrt2``, ``\underline e_3\otimes\underline e_3`` | 0 | a ``2\times2`` block |
| ``\varepsilon = \underline e_1 \otimes^{\mathrm s} \underline e_3`` | 1 | a scalar |
| ``\varepsilon = \underline e_1\otimes\underline e_1 - \underline e_2\otimes\underline e_2`` | 2 | a scalar |
| ``\nabla T = \underline e_3`` / ``\underline e_1`` | 0 / 1 | a scalar each |

Three from the ``2\times2`` block, one from mode 1, one from mode 2: exactly the
**five** constants of a transversely isotropic compliance contribution, with
modes 0 and 1 giving the **two** of the resistivity. The count is not a
coincidence to be checked afterwards — it is the same representation theory in
both places.

The fixed point lives *inside* one mode, since the dipole of a modal
polarization radiates in the same mode, so ``\mathbb X`` is ``2\times2`` for
mode 0 and a scalar for modes 1 and 2.

**And the averaging changes.** A cavity has no interior, so ``\langle
\varepsilon\rangle_{\mathcal D}`` is a boundary integral — over the meridian
trace of the wall, with the measure ``\rho\,\mathrm dl``.

There is a tempting way round it. The divergence identity on the matrix,

```math
\int_{\mathcal M} \varepsilon(\underline u)\,\mathrm dV
  = \oint_{\partial\Omega} (\underline u \otimes \underline n)^{\mathrm s}\,\mathrm dS
  - \oint_{\partial\mathcal D} (\underline u \otimes \underline n_{\mathcal D})^{\mathrm s}\,\mathrm dS ,
```

gives the cavity average from the volume average over the matrix plus an outer
term that is *analytic*, the datum there being imposed. It reuses only machinery
that already exists. It is also **numerically hopeless**: it obtains
``V_{\mathcal D}`` as the difference of ``V_\Omega`` and ``V_{\mathcal M}``,
whose ratio is ``(R/a)^3``. Two digits go at ``R/a = 4`` and more as the cell
grows — the opposite of what a larger cell is for. Measured before it was
abandoned: the implied cavity volume is 2.9 % wrong at ``V_\Omega/V_{\mathcal D}
= 8`` and 9.2 % at 216, with the localization error tracking it, and convergence
of ``O(h)`` where the direct route gives more.

One consequence of that route is worth keeping even after abandoning it:
**normalize by the volume the meshed wall actually encloses**, not by the closed
form. Integrating over one boundary and dividing by another's volume leaves a
systematic error of the geometry's own size, and refining removes none of it
because it shrinks both together.

!!! note "The solid declination in two dimensions, and its region count"
    An `N`-layer spheroid is the *solid* declination of the same correction —
    ``𝔹ᴱ`` and ``𝔹ᵖ`` are not zero, and both outputs of the fixed point are
    used, the second being gate B's stress side. So the cavity was the easy
    case: it degenerates the correction to
    ``𝕃 = (𝕀 − 𝕃_u𝔽)^{-1}𝕃_s`` with one useful output, where a heterogeneous
    inclusion carries the pair.

    Nothing else changes. The same three modes, the same two families of
    boundary data, the same single factorization per mode — and the inclusion
    average runs over `N` regions instead of two, which is the only place the
    layer count appears at all. The dipole's moment is that of the **outer**
    boundary, the inclusion being seen from outside: on a multilayer, using an
    inner layer's volume gives an answer that is almost right.

### Two exact answers, and why they are exact

Two numbers from this construction are worth reading as statements rather than
as measurements.

**Transverse isotropy holds to round-off.** For an axisymmetric cavity,
``\mathbb A_{1212} = (\mathbb A_{1111} - \mathbb A_{1122})/2`` and the
normal-to-shear block vanishes, both to ``10^{-16}`` and at any refinement,
because the modes are decoded straight onto the Kelvin basis. It is therefore a
free check that the three modes, the azimuthal projections, the boundary
integral and the reassembly are simultaneously right — which is more than any
one of them could establish alone.

**Conduction on a sphere is exact, and does not improve with refinement.** The
exterior perturbation of a spherical cavity *is* a pure dipole, so the corrected
boundary condition has no higher multipole left to truncate: the answer sits at
``5\times10^{-6}`` of ``3/2`` and stays there. In elasticity, where higher
multipoles do exist, what remains is truncation and the radius sweep says so —
the uncorrected answer falls as ``R^{-2.85}``, the ``(a/R)^3`` signature, and
the corrected one as ``R^{-5}``.

### One eighth of the cell

Whenever the three coordinate planes are mirror planes of the shape — the
condition
[`has_coordinate_mirrors`](@ref MeanFieldHomogenization.Superspheres.has_coordinate_mirrors)
records, weaker than cubic symmetry and satisfied by both shipped families — the
cell is invariant under the group ``\{\operatorname{diag}(\pm1,\pm1,\pm1)\}`` of
order 8, and an **octant** carries the whole answer.

Each Kelvin load case is an eigenvector of that group: writing
``\mathbb R_k \underline{\underline E} \mathbb R_k = \chi_k
\underline{\underline E}``, uniqueness gives ``\underline u(\mathbb R_k
\underline x) = \chi_k \mathbb R_k \underline u(\underline x)``, so on the plane
``x_k = 0``

| ``\chi_k`` | condition | pinned |
|:--|:--|:--|
| ``+1`` | symmetry | the normal displacement ``u_k`` |
| ``-1`` | antisymmetry | the tangential ones ``u_l,\ l\neq k`` |

with the complementary tractions vanishing of their own accord. In transport,
``\chi_k = +1`` needs nothing at all and ``\chi_k = -1`` needs ``T = 0``. The
six elastic cases fall into **four** parity classes and the three transport ones
into three, so the octant assembles the stiffness once and factorizes it four
(resp. three) times on a matrix eight times smaller.

**The dipole correction obeys the same law**, and that is what makes the whole
scheme compatible with an eighth of the cell rather than only its uncorrected
part. From the closed forms above, ``\mathbb R\,\underline u(\mathbb R
\underline x; \mathbb \Pi) = \underline u(\underline x; \mathbb R \mathbb \Pi
\mathbb R)``, exactly the law obeyed by ``\underline{\underline E}\cdot
\underline x``; since the driver drives both families with the same Kelvin
tensor, remote load and dipole share a ``\chi`` for every case.

One trap, and it is not a factor of eight. Reflecting the surface integral over
the eight octants gives

```math
\int_{\partial I} (\underline u \otimes \underline n)^{\mathrm s}\,\mathrm dS
  = \sum_{g} \chi(g)\, g\, I_{\text{oct}}\, g
  = 8\,\mathbb P_\chi (I_{\text{oct}}),
```

so the components whose parity differs from the load case **cancel between
octants** rather than vanishing in each. They are not small in
``I_{\text{oct}}``; they are spurious, and multiplying by eight without
projecting keeps them at full amplitude. The Kelvin basis diagonalizes the group
action, so ``\mathbb P_\chi`` is a diagonal mask — and the couplings it removes
are exactly the ones a full cell finds only as mesh noise.

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

