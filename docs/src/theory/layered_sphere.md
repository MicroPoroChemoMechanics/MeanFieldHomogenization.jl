# [Layered sphere — bulk + shear recurrences and imperfect interfaces](@id th-layered-sphere)

[`LayeredSphere`](@ref) is an `n`-layer isotropic spherical composite
inclusion in an infinite isotropic matrix, with per-layer localization, global
contribution tensors and layer / sphere / cumulative averages. The bulk and
shear recurrences follow [herve1993](@cite) (generalizing the three-phase model
of [christensenLo1979](@cite)); imperfect interfaces follow
[herveLuanco2014](@cite) — `PerfectInterface` plus one primal/dual pair per
physics:

| Elasticity (primal / dual)                   | Conductivity (primal / dual)                    |
| :------------------------------------------- | :---------------------------------------------- |
| `SpringInterface(kn, kt)`                    | `KapitzaInterface(ρ)`                           |
| `MembraneInterface(κs, μs)` (surface elasticity, [dormieux2016](@cite)) | `SurfaceConductiveInterface(ks)` (Barthélémy-Bignonnet 2020) |

## Convention

Radii are stored in ascending order from the center,

```math
r_0 = 0\text{ (implicit)} < r_1 < r_2 < \cdots < r_N,
```

with **layer ``k`` occupying** ``r_{k-1} \le r < r_k``.  Layer 1 is the
core, layer ``N`` is the outermost shell, and the composite sphere is
embedded in an infinite matrix for ``r > r_N``.

| The generalized Eshelby problem | The ``N = 2`` case: three-phase model |
| :---: | :---: |
| ![Concentric layered inclusion in an infinite matrix, loaded remotely](../assets/geometry/eshelby_generalized.png) | ![Pore, shell of stiffness ℂˢ, infinite medium of stiffness ℂʰᵒᵐ](../assets/geometry/three_phase_model.png) |

The left-hand figure is what the recurrences below solve — a layered pattern in
an infinite reference medium. The right-hand one is the historical special case:
take ``N = 2``, a void core and a solid shell, and make the reference medium the
*unknown* effective one, and the fixed point is Christensen–Lo. Because the
strain is **not** uniform inside such a pattern, it has no Hill tensor at all;
what it does have is a volume-averaged concentration tensor, and that is what
every scheme consumes.

An interactive view of the layered geometry is in
[The inclusion zoo](@ref man-inclusion-gallery).

Moduli ``(\mathbb C_1, \ldots, \mathbb C_N)`` are `TensISO{4,3}`
(elasticity) or `TensISO{2,3}` (conductivity).  Interface conditions
at each radius ``r_k`` are specified in an `NTuple{N, AbstractInterface}`
(default all `PerfectInterface`).

## Bulk (spherical) recurrence — Hervé-Zaoui 1993

Under a purely hydrostatic remote strain, the displacement in layer
``k`` is `u_r^{(k)}(r) = A_k r + B_k / r^2`.  To stay regular in the
**incompressibility limit** ``κ_k → ∞``, the implementation propagates
the **field-valued state vector** ``\mathbf s(r) = (u_r, σ_{rr})``
directly, with the intra-layer transfer

```math
\mathbf T(r_\mathrm{out}, r_\mathrm{in}; κ, μ) =
\begin{pmatrix}
α\,\dfrac{r_\mathrm{out}}{r_\mathrm{in}} + β\,\left(\dfrac{r_\mathrm{in}}{r_\mathrm{out}}\right)^{\!2}
& \dfrac{r_\mathrm{out} - r_\mathrm{in}^{3}/r_\mathrm{out}^{2}}{3κ + 4μ}\\[6pt]
4μβ\left(\dfrac{1}{r_\mathrm{in}} - \dfrac{r_\mathrm{in}^{2}}{r_\mathrm{out}^{3}}\right)
& α\left(\dfrac{r_\mathrm{in}}{r_\mathrm{out}}\right)^{\!3} + β
\end{pmatrix},
```

where ``α = 4μ/(3κ+4μ) ∈ [0,1]`` and ``β = 3κ/(3κ+4μ) ∈ [0,1]`` with
``α + β = 1``.  Every entry stays finite as ``κ → ∞`` (hence also for
``ν → 1/2``), and the per-layer bulk localization ``α_k = A_k/A_∞``
degenerates smoothly (``α_k → 0`` for an incompressible core).

The entry-point at ``r = 0^+`` is written in the "pressure amplitude"
parameterization ``P_1 = 3κ_1 A_1``, so that

```math
u_r(r_1^-) = \frac{r_1}{3κ_1}\,P_1 \xrightarrow{κ_1 → ∞} 0,\qquad
σ_{rr}(r_1^-) = P_1.
```

## Interface jump matrices

Each interface type provides a 2×2 (bulk) jump matrix
``\mathbf J_\text{intf}(r)`` such that
``\mathbf s(r_k^+) = \mathbf J \cdot \mathbf s(r_k^-)``.

```math
\begin{aligned}
\text{Perfect:} &\quad \mathbf J = \mathbf I,\\
\text{Spring}(k_n, k_t): &\quad \mathbf J = \begin{pmatrix}1 & k_n \\ 0 & 1\end{pmatrix}
\quad\text{(bulk uses } k_n\text{ only)},\\
\text{Membrane}(κ_s, μ_s): &\quad \mathbf J = \begin{pmatrix}1 & 0 \\ 4κ_s/r^{2} & 1\end{pmatrix}
\quad\text{(bulk uses } κ_s\text{ only)},\\
\text{Kapitza}(ρ): &\quad \mathbf J = \begin{pmatrix}1 & ρ \\ 0 & 1\end{pmatrix},\\
\text{SurfConductive}(k_s): &\quad \mathbf J = \begin{pmatrix}1 & 0 \\ -n(n{+}1)k_s/r^{2} & 1\end{pmatrix}.
\end{aligned}
```

`SpringInterface` and `KapitzaInterface` encode a **primal
discontinuity** (displacement / temperature jump), while
`MembraneInterface` and `SurfaceConductiveInterface` encode a **dual
discontinuity** (traction / flux jump).  All limit to
`PerfectInterface` when their compliance goes to zero.

## Conductivity recurrence (Y₁ harmonic)

Under a remote uniform temperature gradient, the temperature field
has a Y₁ dependence.  The state vector ``\mathbf s(r) = (T̂, q̂_n)``
(amplitudes projected onto the remote gradient direction) propagates
through a 2×2 transfer matrix ``T_{cond} = M(r_\mathrm{out}) M(r_\mathrm{in})^{-1}``
with ``M(r) = \begin{pmatrix}r & 1/r² \\ -k & 2k/r³\end{pmatrix}``.

Interface jumps for conductivity are given above (Kapitza primal,
SurfaceConductive dual, matching the structural pattern of their
elastic analogs).  The per-layer gradient localization
``α_k = A_k/A_∞`` reduces, in the single-layer case, to the classical
`3 k_0 / (2 k_0 + k_1)` of Maxwell-type composites.

## Type genericity & incompressibility

The recurrence consists of small-size matrix arithmetic over the
element type; it is exercised with `Float64`, `BigFloat`,
`ForwardDiff.Dual`, `SymPy.Sym`, and `Symbolics.Num`.  Symbolically,

```julia
using SymPy; @syms κ₀ μ₀ κ₁ μ₁
s = LayeredSphere((Sym(1),), (TensISO{3}(3κ₁, 2μ₁),))
simplify(MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, κ₀, μ₀)[1])
# → (3κ₀ + 4μ₀) / (3κ₁ + 4μ₀)
```

and the derivative with respect to any modulus or radius is obtained
by wrapping the computation in `ForwardDiff.derivative` /
`ForwardDiff.gradient`.

## Deviatoric (shear) recurrence — `Y₂`-harmonic 4×4 state vector

Under a remote pure-deviatoric strain, the displacement field in an
isotropic layer has the axisymmetric form
``u_r = U(r)\,P_2(\cos θ)``, ``u_θ = W(r)\,P_2'(\cos θ)``, and the
four linearly-independent Navier solutions at ``ℓ = 2`` are parametrized
by the power-law exponents ``n \in \{1, 3, -4, -2\}`` with material-
dependent ``U/W`` ratios derived directly from the Navier characteristic
equation (using ``x = κ/μ``):

| Mode | Radial dependence | ``(U, W)``                                  |
| :--: | :---------------- | :------------------------------------------ |
|  1   | ``r``             | ``(2r, r)`` — uniform deviatoric strain      |
|  2   | ``r^3``           | ``(6(3x-2)\,r^3,\ (15x+11)\,r^3)``           |
|  3   | ``r^{-4}``        | ``(3/r^4,\ -1/r^4)``                         |
|  4   | ``r^{-2}``        | ``(3(x+1)/r^2,\ 1/r^2)``                     |

The corresponding traction amplitudes are obtained from Hooke's law
``σ_{ij} = λ\,δ_{ij}\,\mathrm{tr}(ε) + 2μ\,ε_{ij}``:

```math
\begin{aligned}
σ_{rr}\text{ amp} &= (λ+2μ)\,U' + \frac{2λ}{r}(U - 3W),\\[2pt]
σ_{rθ}\text{ amp} &= μ\bigl(W' + (U-W)/r\bigr).
\end{aligned}
```

The state vector ``\mathbf S(r) = (U, W, σ_{rr}, σ_{rθ})`` combines
displacement and physical traction amplitudes; this form is continuous
across every perfect interface and rational in ``(κ, μ, r)``, so the
recurrence is **type-generic** (supports `Float64`, `BigFloat`,
`ForwardDiff.Dual`, `SymPy.Sym`, `Symbolics.Num`) and remains regular
in the incompressibility limit ``κ → ∞``.

Interface jumps at ``r_k``:

- **Perfect**: identity.
- **Spring**``(k_n, k_t)``:  ``[U] = k_n σ_{rr}``, ``[W] = k_t σ_{rθ}``
  (traction continuous).
- **Membrane**``(κ_s, μ_s)``: surface-elastic 2D shell generates a
  jump in the tractions driven by the surface-stress divergence.

Seeding at ``r_1^-`` uses the two regular modes (``a_1 = 1, b_1 = 0``
and ``a_1 = 0, b_1 = 1``; the two singular amplitudes ``c_1 = d_1 = 0``
are forced by regularity at the origin).  Propagating both probes and
solving a 2×2 linear system for the matrix-side far-field ``(a_∞, b_∞)
= (1, 0)`` yields the per-layer localization ``β_k = a_k``.

For ``N = 1`` the recurrence reduces to the classical Eshelby single-
sphere result; for ``N ≥ 2`` it reproduces the [christensenLo1979](@cite)
core-shell effective shear modulus and passes the Eshelby consistency tests
(``N = 2`` with core ≡ shell ↔ single-layer of radius ``r_N``, etc.).

## Averages (Echoes-style)

Three volume-average flavors are provided:

- [`layer_strain_average`](@ref)`(sphere, C₀, ε∞, k)` — mean strain in
  layer ``k`` (bulk + deviatoric parts).
- [`sphere_strain_average`](@ref)`(sphere, C₀, ε∞)` — mean strain in
  the whole composite.
- [`cumulative_strain_average`](@ref)`(sphere, C₀, ε∞, r)` — mean
  strain inside the ball of radius ``r``.

All three cover the deviatoric part for any ``N ≥ 1`` via the shear
recurrence above.
