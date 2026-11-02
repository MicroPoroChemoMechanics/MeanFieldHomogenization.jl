# [Aging viscoelasticity (ALV)](@id man-viscoelasticity)

!!! info "Before this page"
    [Aging linear Volterra behavior (ALV)](@ref th-viscoelasticity), the theory of
    the Volterra operators and of the schemes used below, and
    [Homogenization schemes](@ref man-schemes), whose `RVE` this page reuses. A
    material that does not age may also take the shorter route of
    [Non-aging viscoelasticity](@ref man-viscoelasticity-nonaging).

In aging linear viscoelasticity the stress at time ``t`` depends on the whole
strain history through a **relaxation kernel** ``\mathbb{R}(t, t')``, the stress
at ``t`` under a unit strain held from ``t'``:

```math
\boldsymbol{\sigma}(t) = \int_{0}^{t} \mathbb{R}(t, t') : \mathrm{d}\boldsymbol{\varepsilon}(t')
= \bigl(\mathbb{R}\stackrel{\circ}{:}\boldsymbol{\varepsilon}\bigr)(t).
```

*Aging* means that ``\mathbb{R}`` depends on ``t'``, the age at loading, and not
only on ``t - t'``. A creep compliance, an aging conductivity or diffusivity are
kernels of the same kind; conduction and diffusion (second-order kernels) are
treated in [section 6](@ref man-visco-order2). Everything below runs as is; longer
versions are in `scripts/50_visco_law_basics.jl` … `scripts/59_alv_sensitivities.jl`.

```@setup vo
using MeanFieldHomogenization, TensND, LinearAlgebra
```

## [1. A law and an operator](@id man-visco-law-operator)

Two objects describe a material:

| | [`ViscoLaw`](@ref) — the **law** | [`VolterraOperator`](@ref) — the **operator** |
|:--|:--|:--|
| what it is | the kernel ``\mathbb{R}(t, t')``, a function of two times | that kernel discretized on a time grid: the matrix of the integral above |
| time grid | none: it can be evaluated at any ``(t, t')`` | one [`TimeGrid`](@ref), fixed once for all; `R(t, t′)` reads its kernel at any instants of the grid's span |
| you build it | by writing a function of ``(t, t')``, or from the [law library](@ref man-rheological-models) | from a law and a grid, a constant tensor, or a matrix |
| you use it to | describe a phase | compute: compose, invert, apply to a history; it is what [`homogenize_alv`](@ref) returns |

You describe a material by a **law**; the package discretizes it into an
**operator** to compute. One goes from one to the other in a line:

```@example vo
law = maxwell_iso(10.0, 6.0, 3.0, 1.0)             # a law: k, μ and their relaxation times
times = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]              # the instants of the computation
grid = TimeGrid(times)
R = VolterraOperator(law, grid)                     # the operator on that grid
law_back = operator_law(R)                          # the kernel again, known on the grid
R(3.0, 0.7) ≈ law(3.0, 0.7)                         # an operator is also a function of (t, t′)
```

A phase of [`homogenize_alv`](@ref) may be given either: a law, with the instants
as `times`, or an operator, which carries its grid. The result is always an
operator.

## 2. Writing a law

A [`ViscoLaw`](@ref) wraps a kernel function of ``(t, t')`` and a mode, `:relaxation`
for ``\mathbb{R}(t, t')`` or `:creep` for the compliance ``\mathbb{J}(t, t')``
(their names in conduction are `:conductivity` and `:resistivity`). The kernel
returns:

* a scalar (`Float64`, `Complex{Float64}`), for a one-dimensional problem;
* a fourth-order tensor (`TensISO{3}(3k, 2μ)`, `TensTI{4}`,
  [`TensOrtho`](@extref TensND TensND.TensOrtho)) or its ``6\times 6`` Kelvin-Mandel matrix;
* a second-order tensor (`TensISO{3}(k)`) or a ``3\times 3`` matrix, for
  conduction, diffusion or permittivity.

An elastic tensor is accepted wherever a law is: its kernel is ``\mathbb{C}\,H(t-t')``.

### 2.1 From the law library

A material that does not age is best taken from the
[law library](@ref man-rheological-models): `ViscoLaw` builds its kernel
``(t, t') \mapsto R(t - t')``.

```@example vo
m = iso_rheology(zener_maxwell(30.0, 20.0, 1.0), zener_maxwell(10.0, 8.0, 0.7))   # bulk, shear
law_lib = ViscoLaw(m)
nothing # hide
```

The same model also drives the Laplace-Carson route of
[Non-aging viscoelasticity](@ref man-viscoelasticity-nonaging), so that the two
routes compare the same material, which is what the
[three-route check](@ref tut-freq-vs-time) relies on.

### 2.2 Ready-made isotropic laws

```@example vo
law_max = maxwell_iso(5.0, 2.0, 1.0, 0.5)     # 3k e^{-t/1} 𝕁 + 2μ e^{-t/0.5} 𝕂: k, μ, then their times
law_kel = kelvin_iso(3.0, 1.0, [5.0], [2.0], [1.0], [0.5])   # creep: k₀, μ₀, then one Kelvin branch
law_el = heaviside_law(TensISO{3}(15.0, 4.0))  # elastic: C H(t - t′)
nothing # hide
```

### 2.3 Written by hand, aging included

Any function of ``(t, t')``, ``t`` the current time and ``t'`` the time of
loading, is a law. Aging means that it depends on ``t'``, not only on
``t - t'``. Here a solidifying material, whose active fraction grows with the age
at loading:

```@example vo
solidified(t′) = 0.3 * t′^4 / (1 + t′^4)        # active fraction at the time of loading
law_M = ViscoLaw(:relaxation) do t, t′
    f = solidified(t′)
    TensISO{3}(3 * (3.0 + 2.0 * f * exp(-(t - t′))), 2 * (1.0 + f * exp(-(t - t′) / 0.5)))
end
nothing # hide
```

`TensISO{3}(3k, 2μ)` is the isotropic tensor ``3k\,\mathbb{J} + 2\mu\,\mathbb{K}``.
A phase that forms during the history is loaded only by the strains that follow its
formation, and a phase that dissolves stops carrying stress: [`precipitated`](@ref) and
[`dissolved`](@ref) switch a relaxation kernel on at ``t_s`` or off at ``t_d``, for any law,
sampled ones included. A hydrating paste is built from them, one phase per step of its
hydration ([aging basic creep of concrete](@ref app-aging-basic-creep)).

## [3. Operators and histories](@id man-volterra-operators)

An operator acts on a **history**, a function of one time, ``\boldsymbol{\varepsilon}(t)``;
on the grid both have two faces:

| object | as a function | on a time grid ``t_1 < \dots < t_n`` |
|:--|:--|:--|
| an operator, [`VolterraOperator`](@ref) | of two times, ``\mathbb{R}(t, t')`` | a matrix acting on the values of a history |
| a history, [`TimeHistory`](@ref) | of one time, ``\boldsymbol{\varepsilon}(t)`` | its values ``\boldsymbol{\varepsilon}(t_i)`` |

### The time grid

Every operator and every history of one computation lives on the same
instants. A [`TimeGrid`](@ref) holds them; it is built once, and every object
built on it carries it:

```@example vo
grid = TimeGrid([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
```

[`time_grid`](@ref) returns the grid of an operator, a history, or the result of
a computation on them. Two objects on different grids are refused rather than
interpolated onto one another.

### [Jumps: an instant given twice](@id man-visco-jumps)

A creep test applies a stress at the loading age ``t'`` and holds it: the stress
**jumps** at ``t'``. Between two instants of the grid a history is linear, so on a
grid where ``t'`` appears once the stress rises over the interval before ``t'``,
and the material creeps during that rise. Giving ``t'`` **twice** makes the jump
exact: the first node is the left limit ``t'^-``, the second the right limit
``t'^+``. A history or a law given as a function is evaluated just before ``t'`` on
the first node, so a function that jumps at ``t'`` is written right-continuous,
`t ≥ t′ ? … : …`.

Here a stress held from ``t' = 1`` on a material that creeps logarithmically. The
strain is zero up to ``1^-`` and follows the compliance ``J(t, 1)`` from ``1^+`` on:

```@example vo
jump = TimeGrid([0.0, 0.5, 1.0, 1.0, 2.0, 4.0, 8.0])      # 1.0 twice: a jump at t = 1
J_log(t, t′) = t < t′ ? 0.0 : 1 / 30 + log1p((t - t′) / 0.5) / 60
creep = VolterraOperator(J_log, jump)
σ = TimeHistory(t -> t ≥ 1.0 ? 1.0 : 0.0, jump)            # 0 at 1⁻, 1 at 1⁺
ε = creep * σ
held = [i ≤ 3 ? 0.0 : J_log(t, 1.0) for (i, t) in enumerate(jump)]   # nothing up to 1⁻
[ε[i][1] for i in 1:length(jump)] ≈ held
```

The same holds for a phase that appears at ``t_s`` ([`precipitated`](@ref)) or
disappears at ``t_d`` ([`dissolved`](@ref)): with ``t_s`` given twice, a strain applied
before ``t_s`` leaves the phase unstressed, and a strain jump at ``t_s`` loads it
whole. The exact weights are derived in [the theory](@ref th-visco-jumps).

### Building an operator

From a **law**, any function of ``(t, t')``. Here an aging Maxwell material whose
bulk and shear moduli relax, the stiffness of a phase being ``3k\,\mathbb{J} +
2\mu\,\mathbb{K}`` (`TensISO{3}(3k, 2μ)`):

```@example vo
age(t′) = 1 - 0.5 * exp(-t′)          # the material stiffens with its age at loading
R = VolterraOperator(grid) do t, t′
    age(t′) * TensISO{3}(3 * 10.0 * exp(-(t - t′) / 3), 2 * 6.0 * exp(-(t - t′)))
end
```

From a **constant tensor** (an elastic phase, whose relaxation does not depend
on time):

```@example vo
E = VolterraOperator(TensISO{3}(3 * 30.0, 2 * 20.0), grid)
nothing # hide
```

From the **matrix** of an operator already discretized on the grid
(``6n\times 6n`` at order 4, with ``6\times 6`` Kelvin-Mandel blocks), as one
scale hands it to the next or as another code writes it. `like = C` stores it in
the symmetry class and frame of the tensor `C`, after checking that it belongs
to it:

```@example vo
M = Matrix(R)                                  # 36 × 36
R2 = VolterraOperator(M, grid; like = TensISO{3}(1.0, 1.0))
R2 ≈ R
```

### Computing as with elastic tensors

An operator computes as its elastic counterpart does, with the same symbols:

| elastic | aging viscoelastic |
|:--|:--|
| `C₁ ⊡ C₂` | composition ``\mathbb{R}_1 \stackrel{\circ}{:} \mathbb{R}_2`` (it does not commute: keep the order) |
| `inv(C)` | Volterra inverse |
| `C₁ + C₂`, `C₁ - C₂`, `c * C`, `C / c` | sum, difference, product by a number |
| `I` | the identity operator |
| `C ⊡ ε` | the response to a history |

At order 2 (conduction, diffusion, permittivity) `⋅` takes the place of `⊡`.

The **creep operator** of a material is the Volterra inverse of its relaxation
operator:

```@example vo
J = inv(R)
nothing # hide
```

The stress under a strain history, and the strain recovered from it:

```@example vo
ε = TimeHistory(grid) do t
    1.0e-3 * TensND.Tens(Symmetric([1.0 0 0; 0 -0.3 0; 0 0 -0.3]))   # uniaxial, held from t = 0
end
σ = R ⊡ ε
inv(R) ⊡ σ ≈ ε
```

`σ[i]` is the stress at ``t_i``, as its six Kelvin-Mandel components. A
dilute estimate written as on paper, ``\mathbb{R} + f\,(\mathbb{E} -
\mathbb{R}) \stackrel{\circ}{:} (H\,\mathbb{I} + \mathbb{P} \stackrel{\circ}{:} (\mathbb{E} - \mathbb{R}))^{-1^{\circ}}``,
is one line of Julia; [`homogenize_alv`](@ref) computes such estimates for every scheme.

!!! note "An operator is its matrix"
    A [`VolterraOperator`](@ref) is an `AbstractMatrix`: the ``6n\times 6n``
    matrix of the discretized integral, stored compactly. Its matrix product is the
    composition, its matrix inverse the Volterra inverse, so it goes wherever such a
    matrix goes. `Matrix(R)` writes it out in full.

The matrix is that of the trapezoidal rule of [sanahuja2013](@citet): on the grid
``(t_1, \dots, t_n)`` the integral becomes a block lower-triangular matrix of
``6\times 6`` blocks at order 4, ``3\times 3`` at order 2. [`trapezoidal_matrix`](@ref)
writes it out for a law, and [`volterra_inverse`](@ref) inverts such a matrix by
forward substitution; a [`VolterraOperator`](@ref) holds that matrix.

### Symmetry classes: what is stored

When the values ``\mathbb{R}(t, t')`` are isotropic, transversely isotropic
about one axis, orthotropic or cubic in one frame, each of them is a few small
**blocks** (TensND `symmetry_blocks`): two numbers for an isotropic tensor, a
``2\times 2`` matrix and two numbers for a transversely isotropic one. The
operator then stores, for each block, one matrix over the instants, and computes
on those, block by block:

| class | blocks stored | storage, relative to ``6n\times 6n`` |
|:--|:--|:--|
| isotropic | ``(\alpha)``, ``(\beta)`` | 1/18 |
| cubic | three scalars | 1/12 |
| transversely isotropic, any axis | a ``2\times 2``, two scalars | 1/6 |
| orthotropic, any frame | a ``3\times 3``, three scalars | 1/3 |
| none (dense) | the ``6\times 6`` | 1 |

An isotropic product costs two ``n\times n`` products where the dense one costs
a ``6n\times 6n`` one: about a hundred times fewer operations.

```@example vo
R
```

The class is found from the values of the law, and an isotropic operator
combined with another joins its class. [`operator_value`](@ref)`(R, i, j)` is
the value at the pair of instants ``(t_i, t_j)`` as a tensor of the class;
[`convert_class`](@ref) stores an operator in another class or dense
(`DenseClass{4}()`).

!!! warning "A value of the operator is not a value of the kernel"
    The matrix holds the weights of the trapezoidal rule, which mix the
    kernel at neighboring instants. The kernel ``\mathbb{R}(t, t')`` is
    `R(t, t′)`, below.

### [Operators and histories as functions of time](@id man-visco-functions)

An operator is also the function ``\mathbb{R}(t, t')`` and a history the function
``\boldsymbol{\varepsilon}(t)``; both are called as such:

| call | built from a function | computed (a sum, an inverse, [`homogenize_alv`](@ref)) or given by values |
|:--|:--|:--|
| `R(t, t′)` | the law itself, at any ``(t, t')``; zero for ``t' > t`` | its kernel recovered on the grid, interpolated between the instants |
| `h(t)` | the function itself, as components | its values, interpolated linearly between the instants |

The interpolation is the one the trapezoidal rule integrates exactly
([derivation](@ref th-visco-kernel-between)): linear between the instants,
right-continuous at an instant given twice, and nothing outside
``[t_1, t_n]``. The value is a tensor of the class of `R` (a number for a scalar
operator), and `h(t)` the Kelvin-Mandel components, as `h[i]`.

```@example vo
J = inv(R)                                   # a creep operator, computed: no law
J(3.0, 0.7)                                  # its kernel, between the instants
```

Finding the cell of ``(t, t')`` is a binary search; a computed operator then
recovers the two rows of its kernel that the cell needs, a cost proportional to the
number of instants. For many evaluations, recover the whole kernel once:

```@example vo
J_law = operator_law(J)                      # every sample, once
J_law(3.0, 0.7) ≈ J(3.0, 0.7)                # then a binary search and four samples
```

Both accept ForwardDiff duals, in the instants and in the parameters of a law.
A law-backed operator also takes symbolic instants; an interpolated one refuses
them, since finding the cell compares instants.

[`homogenize_alv`](@ref) computes in these classes whenever the phases allow it, and
returns its result in the cheapest class that holds it; the costs are tabulated in
[Symmetry classes and structured storage](@ref th-visco-classes).

## [4. Homogenizing](@id man-visco-homogenizing)

A phase property may be a law, an operator on the grid, or a [`Homogenized`](@ref)
cell, homogenized on the same grid, as in elasticity. With operators, the grid is
read on them:

```@example vo
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => R); fraction = :rest)
add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => E); fraction = 0.3)
Rhom = homogenize_alv(rve, MoriTanaka(:M), :C)
```

The result is a phase of the next scale as it is, on the same grid; or, without
computing it first, the cell itself:

```@example vo
concrete = RVE()
add_phase!(concrete, :paste, Ellipsoid(1.0), Dict(:C => Homogenized(rve, MoriTanaka(:M))); fraction = :rest)
add_phase!(concrete, :sand, Ellipsoid(1.0), Dict(:C => VolterraOperator(TensISO{3}(3 * 40.0, 2 * 30.0), grid)); fraction = 0.6)
homogenize_alv(concrete, MoriTanaka(:paste), :C)
```

With laws, the instants are given as `times`; here the aging matrix of §2.3 and
aligned oblate spheroids:

```@example vo
cell = RVE()
add_phase!(cell, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
add_phase!(cell, :I, Ellipsoid(1.0, 1.0, 0.5), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2)
C_eff = homogenize_alv(cell, MoriTanaka(), :C; times)
```

[`homogenize_alv`](@ref) accepts `Voigt`, `Reuss`, `Dilute`, `DiluteDual`,
`MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`, `SelfConsistent`,
`AsymmetricSelfConsistent` and `DifferentialScheme`.

### Reading the result

The result is stored by the blocks of its class: for the transversely isotropic
result above, the ``2n\times 2n`` matrix of the ``2\times 2`` Walpole blocks and the
two ``n\times n`` shear ones. These are the weights of the trapezoidal rule, not
values of the kernel; the kernel at the instants of the grid is that of
[`operator_law`](@ref), and [`operator_value`](@ref)`(C_eff, i, j)` the value of the
operator at a pair of instants, as a tensor of its class
([section 3](@ref man-volterra-operators)):

```@example vo
law_eff = operator_law(C_eff)            # the kernel C(t, t′), known on the grid
law_eff(times[end], times[1])            # its value at (tₙ, t₁)
```

For a **uniaxial creep** test, a unit stress held from ``t_1``, the strain is the creep
operator applied to that history:

```@example vo
J_eff = inv(C_eff)                       # the creep operator
σ = TimeHistory(t -> TensND.Tens(Symmetric([1.0 0 0; 0 0 0; 0 0 0])), grid)
ε = J_eff ⊡ σ
[ε[i][1] for i in eachindex(times)]      # ε_xx(tᵢ)
```

## [5. Cracks in ALV](@id man-visco-cracks)

### 5.1 Traction-free elliptic and ribbon cracks

A crack family enters with its density, in the matrix law:

```@example vo
cracked = RVE()
add_phase!(cracked, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
add_phase!(cracked, :F, PennyCrack(1.0), Dict(:C => law_M); density = 0.05, symmetrize = :iso)
homogenize_alv(cracked, MoriTanaka(), :C; times)
```

Elliptic cracks of any aspect ratio ``\eta = b/a`` (`EllipticCrack(a, b)`, the
penny included) and of any orientation (`euler_angles`) are accepted, in an
isotropic matrix. Their COD coefficients are the elastic closed form with every
quotient a Volterra inverse, in the order of the flat limit of a void ellipsoid
([`cod_kernel_alv`](@ref)); their compliance is stored in the class of the crack,
transversely isotropic about the normal of a penny, orthotropic in the frame of an
ellipse, so that an oriented family is not a dense operator:

```@example vo
add_phase!(cracked, :E, EllipticCrack(1.0, 0.4; euler_angles = (0.4, 0.9, 0.2)),
           Dict(:C => law_M); density = 0.05)
homogenize_alv(cracked, MoriTanaka(), :C; times)
```

The density is ``\varepsilon = N\,a\,b^2``, as in elasticity ([why](@ref th-cod-density),
and how to convert the densities of other sources). Checked against Echoes on aging
matrices (tilted ellipses with dilute, Mori–Tanaka and self-consistent schemes,
two families of different orientations) to ``10^{-13}``: see
`scripts/bench_echoes/alv_harness`.

Ribbon cracks (`RibbonCrack(b)`, unbounded along their first axis, of half-width
`b`) are accepted too, with the density ``\varepsilon = N\,b^2`` per unit area; their
compliance is orthotropic in their frame. A [`ConductiveCrack`](@ref) opens as the
penny it wraps. The formulas, and the limits that check them, are in
[Cracks](@ref th-visco-cracks) of the theory:

```@example vo
add_phase!(cracked, :R, RibbonCrack(1.0; euler_angles = (0.4, 0.9, 0.2)),
           Dict(:C => law_M); density = 0.05)
homogenize_alv(cracked, MoriTanaka(), :C; times)
```

### 5.2 Cracks with finite interface stiffness

For a flat crack carrying a **spring-like interface stiffness** with
time-dependent normal `Rn(t,t')` and tangential `Rt(t,t')` aging
kernels, attach the interface laws as `:Rn` / `:Rt` properties of the
crack phase, scalar laws of stiffness per unit area:

```@example vo
law_Rn = ViscoLaw((t, t′) -> 3.0 * exp(-(t - t′) / 4))   # normal stiffness, relaxing
law_Rt = heaviside_law(1.5)                             # tangential stiffness, constant
bonded = RVE()
add_phase!(bonded, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
add_phase!(bonded, :F, PennyCrack(1.0), Dict(:C => law_M, :Rn => law_Rn, :Rt => law_Rt);
           density = 0.05, symmetrize = :iso)
homogenize_alv(bonded, MoriTanaka(), :C; times)
```

The scalar COD kernels ``\widetilde{B}_n`` (normal) and ``\widetilde{B}_\ell``, ``\widetilde{B}_m`` (tangential) are
post-corrected, `Rn` on the first and `Rt` on the other two: crack-face and
interface compliances add up, as if the springs saw the average opening, at the
cost of one extra scalar Volterra inverse per direction. This is an approximation,
exact in its limits below but not derived from the integral equation of a crack
with springs ([Interfaces](@ref th-visco-cracks)).

Limits, all exercised in the test suite:

| Interface | Behavior |
|:--|:--|
| `Rn = Rt = nothing` | traction-free crack, ``\widetilde{B}`` unchanged |
| stiffnesses ``\to 0`` | traction-free again |
| stiffnesses ``\to \infty`` (rigid) | ``\widetilde{B}^{\mathrm{int}} \to 0``: the crack is bonded |

### 5.3 How each scheme takes the cracks

A crack family has no volume, which decides how each scheme takes it
([Cracks in the schemes](@ref th-visco-cracks)):

| Scheme                         | Crack treatment                                     |
|:-------------------------------|:----------------------------------------------------|
| `Voigt`, `Reuss`               | ignored (no volume)                                 |
| `Dilute`, `DiluteDual`         | its contribution added to the stiffness or the compliance |
| `Maxwell`, `PCW`               | one more contribution, of weight one                |
| `MoriTanaka`                   | the Echoes form ``\mathbb{B}\stackrel{\circ}{:}\mathbb{A}^{-1^{\circ}}``: the crack also enters the average strain, against the matrix |
| `SelfConsistent`, `AsymmetricSelfConsistent` | the same form, against the running estimate |
| `DifferentialScheme`           | its contribution against the running medium, at the rate of its density |

The Echoes form is also that of the elastic Mori–Tanaka scheme with cracks: the
crack term in the average strain is what keeps it from the spurious percolation
of a purely additive form at moderate density. The schemes on the running
estimate need an isotropic orientation average of each family
(`symmetrize = :iso`), see [section 7](@ref man-alv-iso-reference).

Scripts: `scripts/57_aging_creep_cracks.jl` (every crack-aware ALV scheme) and
`scripts/15_cracks_iso_interface.jl` (elastic and conductive cracks with interfaces,
without aging).

## [6. Conduction and diffusion (order 2)](@id man-visco-order2)

Conduction, diffusion and permittivity work the same way, with kernels of order 2:
a conductivity ``\boldsymbol{K}(t, t')`` acting on the history of a temperature
gradient by ``\stackrel{\circ}{\cdot}``. A law returns a second-order tensor,
`TensISO{3}(k)` for ``k\,\boldsymbol{1}``, and [`homogenize_alv`](@ref) recognizes the order
from it:

```@example vo
law_κ = ViscoLaw((t, t′) -> TensISO{3}(1.0 + 0.5 * exp(-(t - t′))))
rve_κ = RVE()
add_phase!(rve_κ, :M, Ellipsoid(1.0), Dict(:K => law_κ); fraction = :rest)
add_phase!(rve_κ, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(5.0)); fraction = 0.3)
K_eff = homogenize_alv(rve_κ, MoriTanaka(), :K; times)
```

An operator of order 2 acts on vector histories (`⋅` takes the place of `⊡`):

```@example vo
g = TimeHistory(t -> [1.0, 0.0, 0.0], grid)      # a unit temperature gradient along e₁
q = K_eff ⋅ g
q[end]
```

The order-2 pipeline implements the bounds, `Dilute`, `DiluteDual`,
`MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`, `SelfConsistent`,
`AsymmetricSelfConsistent` and `DifferentialScheme`. `Maxwell` and `PonteCastanedaWillis` take the Hill kernel
of the distribution shape the RVE declares (`RVE(; distribution_shape = …)`).
`scripts/56_aging_creep_order2.jl` runs them on an aging conductor.

A [`LayeredSphere`](@ref) enters every one of them, with
`KapitzaInterface` and `SurfaceConductiveInterface` between its layers. As at
order 4, the phase law is a placeholder: the layers carry their own kernels,
elastic tensors or `ViscoLaw`s.

```@example vo
grain = LayeredSphere((0.8, 1.0), (law_κ, TensISO{3}(5.0));
                      interfaces = (KapitzaInterface(0.1), SurfaceConductiveInterface(0.3)))
A_grain = gradient_gradient_concentration_alv(grain, law_κ, times)   # whole-sphere concentration
N_grain = property_contribution_alv(grain, law_κ, times)             # its contribution
size(A_grain)
```

The jumps follow the dictionary ``\boldsymbol\sigma\equiv-\underline q``: a Kapitza
resistance gives ``[\![T]\!] = \rho\,\sigma_n = -\rho\,q_n``, outer value minus inner
one, and `external = false` leaves the outer interface to the matrix, as in
elasticity.

`symmetrize` is honored in both orders. The projection is applied to the
dilute quantities (``\mathbb{A}_\alpha``, ``\mathbb{N}_\alpha``) block by block, with the projector of the
right tensor order — 6×6 Mandel blocks for order 4, plain 2-tensor blocks
for order 2. It is exact in both cases: the SO(3) average of a 2-tensor is
its spherical part ``\tfrac{1}{3}(\operatorname{tr}\boldsymbol{B})\,\boldsymbol{1}``, and the azimuthal average about ``\underline{n}``
keeps the axial component, the transverse mean and the axial antisymmetric
part.

The bounds are shape-blind: `Voigt` and `Reuss` average the phase
relaxation matrices directly, so `symmetrize` cannot change their result.

### [6.1 Insulating cracks in conduction](@id man-visco-cracks-order2)

A crack family in a conductor is insulating: it adds the resistivity
``\widetilde{\boldsymbol{R}} = \kappa\,\widetilde{b}\;\underline{n}\otimes\underline{n}``, with
``\widetilde{b}`` the thermal COD of the elastic theory in which ``1/k_0`` is the
Volterra inverse of the conductivity kernel ([the formulas](@ref th-visco-cracks)).
Elliptic cracks of any aspect ratio, ribbons, and any orientation, in an
isotropic ALV conductor; a Kapitza-like interface of conductance kernel
`α_interface` (a scalar `ViscoLaw`) is attached as a property of the phase, the
counterpart of `:Rn` and `:Rt`:

```@example vo
add_phase!(rve_κ, :F, EllipticCrack(1.0, 0.4; euler_angles = (0.4, 0.9, 0.2)),
           Dict(:K => law_κ, :α_interface => heaviside_law(0.5)); density = 0.1)
homogenize_alv(rve_κ, MoriTanaka(), :K; times)
```

The resistivity of one crack, ``\widetilde{\boldsymbol{R}}`` per unit density, is
[`resistivity_contribution_alv`](@ref):

```@example vo
resistivity_contribution_alv(PennyCrack(1.0), law_κ, times)
```

Every order-2 scheme takes them, as at order 4 (the table of
[section 5.3](@ref man-visco-cracks)), the self-consistent and differential ones
with `symmetrize = :iso`.

## [7. The isotropic reference of the self-consistent and differential schemes](@id man-alv-iso-reference)

Every ALV Hill kernel — order 2 and order 4 alike — is built for an
**isotropic** reference medium: that is the condition under which the
time and space parts decouple and a closed form exists at all. This is
not a restriction on the *result*, which is generally anisotropic; it is
a restriction on what the kernel may be evaluated *against*.

That splits the schemes in two:

* `Voigt`, `Reuss`, `Dilute`, `DiluteDual`, `MoriTanaka`, `Maxwell`, `PCW`
  evaluate the kernel against the **matrix**, which is fixed and
  isotropic. Any inclusion shape and orientation is fine.
* `SelfConsistent` and `DifferentialScheme` evaluate it against their
  **running estimate** — the fixed point for the former, ``\mathbb{C}(\tau)`` for the
  latter. An aligned non-spherical inclusion, or a crack (whose
  contribution is transversely isotropic in its own frame), drags that
  estimate out of the isotropic class, and the kernel would no longer be
  valid there.

!!! warning "Reference-updating ALV schemes need an isotropic running medium"
    With `SelfConsistent` or `DifferentialScheme` in ALV, every inclusion
    phase must keep the running estimate isotropic. Two ways to satisfy
    that: **spherical inclusions** with an isotropic phase law, or an
    **isotropic orientation average**, `symmetrize = :iso`, which is also
    what randomly oriented inclusions or cracks mean physically. A
    `LayeredSphere`, isotropic by construction, is always accepted.

    An RVE that satisfies neither raises an explicit `ArgumentError` naming
    the offending phase, rather than silently reading iso parameters off a
    matrix that is no longer isotropic. A non-isotropic ALV *matrix* is
    refused for the same reason.

```@example vo
iso_cell = RVE()          # randomly oriented cracks: the orientation average makes this legitimate
add_phase!(iso_cell, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
add_phase!(iso_cell, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.3)
add_phase!(iso_cell, :F, PennyCrack(1.0), Dict(:C => law_M); density = 0.1, symmetrize = :iso)
homogenize_alv(iso_cell, SelfConsistent(), :C; times)
```

At order 2, aligned spheroids need the same average:

```@example vo
needles = RVE()
add_phase!(needles, :M, Ellipsoid(1.0), Dict(:K => law_κ); fraction = :rest)
add_phase!(needles, :I, Spheroid(5.0), Dict(:K => TensISO{3}(10.0)); fraction = 0.2, symmetrize = :iso)
homogenize_alv(needles, DifferentialScheme(), :K; times)
```

!!! note "The elastic schemes have no such restriction"
    This is specific to ALV. In the elastic pipeline the Hill tensor is
    available for anisotropic references, so `homogenize(rve,
    SelfConsistent())` and `homogenize(rve, DifferentialScheme())` handle a
    running medium of any symmetry class — the differential scheme even
    tracks it, growing its ODE state from the iso to the TI, ortho or fully
    anisotropic layout as the phases require.

## 8. The differential scheme in ALV

`DifferentialScheme` is available in both tensor orders, with the same
keywords as the elastic scheme — `trajectory`, `nsteps`, `abstol` /
`reltol`, `alg`, and `formulation = :stiffness | :compliance` (the dual
form integrates the creep function ``\mathbb{J} = \mathbb{C}^{-1^{\circ}}``):

```@example vo
homogenize_alv(iso_cell, DifferentialScheme(; formulation = :compliance), :C; times)
```

Supported inclusions: ellipsoids and spheroids, `LayeredSphere`, and
crack families through their density, at both orders — subject to the
isotropic-reference requirement of [section 7](@ref man-alv-iso-reference). The
self-consistent and the asymmetric self-consistent schemes accept the same
inclusions at both orders.

## [9. Solving the self-consistent equation: Picard or Newton](@id man-visco-sc-solver)

`SelfConsistent()` solves its equation by Picard iteration from the matrix kernel,
with the keywords `abstol`, `reltol`, `maxiters`, `damping`, `verbose` and
`select_best`. `SelfConsistent(algorithm = NewtonDefault())` marches through the
instants instead, solving at each one a small system by Newton's method
([how](@ref th-visco-sc-solve)); its `maxiters` is a cap per instant, and the
keywords of the Picard iteration alone are refused:

```@example vo
picard = homogenize_alv(iso_cell, SelfConsistent(reltol = 1e-12), :C; times)
newton = homogenize_alv(iso_cell, SelfConsistent(algorithm = NewtonDefault(), reltol = 1e-12), :C; times)
Matrix(newton) ≈ Matrix(picard)
```

Both reach the same estimate; Newton converges in a few iterations per instant
where the Picard iteration slows down near percolation. The march starts from the
Picard estimate, loosely converged, because the equation may then have several
roots and Newton's method from the matrix may reach a non-physical one; such a
root, with a negative instantaneous modulus, is refused, and an instant that does
not converge is reported by a warning. Another algorithm, or a seed (`init`), is
refused rather than ignored.

## 10. Sensitivities (autodiff via ForwardDiff)

The pipeline carries `ForwardDiff.Dual` numbers from end to end, so derivatives of an
effective property follow by automatic differentiation. Here the property is the
effective shear relaxation modulus ``\mu(t_n, t_1)``, read on the kernel of the result:

```@example vo
using ForwardDiff
μ_relax(R̃) = k_mu(operator_law(R̃)(times[end], times[1]))[2]   # μ(tₙ, t₁)

# with respect to a volume fraction: substitute it with `set_param`
base = RVE()
add_phase!(base, :M, Ellipsoid(1.0), Dict(:C => law_M); fraction = :rest)
add_phase!(base, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = 0.2)
dμ_df = ForwardDiff.derivative(0.2) do f
    μ_relax(homogenize_alv(set_param(base, AmountParameter(:I), f), MoriTanaka(), :C; times))
end
```

A parameter inside a law is differentiated by building the cell in a function of it,
several at once with a gradient:

```@example vo
function μ_of(p)
    f, k_M, μ_M = p
    c = RVE()
    add_phase!(c, :M, Ellipsoid(1.0), Dict(:C => maxwell_iso(k_M, μ_M, 1.0, 0.5)); fraction = :rest)
    add_phase!(c, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(60.0, 20.0)); fraction = f)
    return μ_relax(homogenize_alv(c, MoriTanaka(), :C; times))
end
ForwardDiff.gradient(μ_of, [0.2, 5.0, 2.0])
```

More patterns, each checked against an independent reference, are in
`scripts/59_alv_sensitivities.jl` and the tutorial
[Derivatives through the aging-viscoelastic pipeline](@ref tut-alv-sensitivities).

## 11. Validation against ECHOES C++

| script | benchmark | agreement |
| :--- | :--- | :--- |
| `53_aging_creep_solid.jl` | multi-phase Maxwell + solidifying Maxwell + pore (ECHOES C++ manual) | — |
| `57_aging_creep_cracks.jl` | seven crack-aware ALV schemes, penny-crack RVE | — |
| `52_rabotnov_mittag_leffler.jl` | Rabotnov / Mittag-Leffler closed form, [barthelemyIJES2019](@cite) §5 | `rtol ≤ 1.3e-3` at `n_times = 200` |

The Rabotnov kernel of that benchmark is a model of the
[law library](@ref man-rheological-models), [`Rabotnov`](@ref):

```@example vo
I_Rabotnov(t, α, β) = relaxation(Rabotnov(1.0, 1.0, α, β), t) - 1.0
I_Rabotnov(2.0, -0.5, 1.0)
```

Its Laplace-Carson transform is elementary, ``R^{*}(p) = \mu_0(1 + \lambda_0/(p^{\alpha+1}+\beta))``,
with no Mittag-Leffler function in it at all. Loading `MittagLeffler.jl` (a weak
dependency) switches on the closed-form *time* value; without it the numerical
inversion of that transform supplies the same number to about `1e-10`.

!!! note "History of the Rabotnov kernel in this package"
    The benchmark first took the Rabotnov kernel from an external Python module,
    through a Python bridge, which made the script depend on the machine. The
    kernel is now a model of the library, computed in Julia like the others.

Random-RVE cross-checks vs the reference implementation live in
`scripts/bench_echoes/benchmark.jl` (relative error `≤ 1e-8` on the
Mandel `(1, 1)` block, `≤ 1e-6` on the full matrix).

## Where to go next

* Theory: [Aging linear Volterra behavior (ALV)](@ref th-viscoelasticity) derives
  the Hill kernel, the schemes, the [cracks](@ref th-visco-cracks) and the
  [solution of the self-consistent equation](@ref th-visco-sc-solve).
* Laws: [the law library](@ref man-rheological-models) supplies non-aging laws
  for `ViscoLaw(m)`; [Non-aging viscoelasticity](@ref man-viscoelasticity-nonaging)
  takes such a material through the Laplace-Carson route instead.
* Tutorials: [Aging viscoelastic schemes side by side](@ref tut-alv-schemes),
  [Derivatives through the aging-viscoelastic pipeline](@ref tut-alv-sensitivities)
  (which extends section 10),
  [Frequency or time?](@ref tut-freq-vs-time), which sets this route against the
  non-aging ones on the same material.
* Applications: [Aging creep of solidifying cementitious materials](@ref app-aging-creep)
  and [Aging basic creep of concrete](@ref app-aging-basic-creep).
