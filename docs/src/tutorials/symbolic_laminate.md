# [Symbolic laminates: arithmetic and harmonic averages](@id tut-symbolic-laminate)

A laminate is the one microstructure in this package whose effective behavior
is **exact**, so it is also the one whose closed forms the code can *derive*
rather than merely reproduce. `TensND` being generic in its element type, the
whole laminate cell runs on **SymPy** `Sym` unchanged — nothing has to be
declared for it, the moduli, the fractions and the frame all carry their own
type — and what comes out is
the classical result of [backus1962](@cite), in the form
[Voigt and Reuss](@ref th-homogenization) would lead one to expect: some
coefficients are **arithmetic** averages across the layers, others are
**harmonic** ones, and the rest are combinations of both.

That split is not a coincidence. In the plane of the layers the strain is
shared (a *parallel* arrangement → arithmetic mean); across them the traction
is shared (a *series* arrangement → harmonic mean). This page makes that
statement literal, on matrices.

Terser script:
[`scripts/38_laminate_symbolic.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/38_laminate_symbolic.jl).

## Setup

```@example tutsymlam
using MeanFieldHomogenization
using TensND
using SymPy

@syms λ₁::positive μ₁::positive λ₂::positive μ₂::positive f₁::positive
f₂ = 1 - f₁

iso_lame(λ, μ) = TensISO{3}(3(λ + 2μ / 3), 2μ)      # (3κ, 2μ) from (λ, μ)

lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => iso_lame(λ₁, μ₁)); fraction = f₁)
add_layer!(lam, :B, Dict(:C => iso_lame(λ₂, μ₂)); fraction = f₂)
nothing # hide
```

Nothing here is special to two layers or to isotropy — it is just what keeps
the printed expressions short enough to read.

## The effective stiffness, as a matrix

```@example tutsymlam
Chom = homogenize(lam, Laminated(), :C)
M = simplify.(KM(Chom))
```

Two things are worth reading off that matrix before any formula.

**The sparsity pattern.** All the ``13``, ``23``, … couplings vanish and
``M_{11} = M_{22}``, ``M_{13} = M_{23}``, ``M_{44} = M_{55}``: the stack of
isotropic layers is **transversely isotropic** about the normal, and the
package says so in the type it returns —

```@example tutsymlam
typeof(Chom)
```

**The Mandel weights.** In the Kelvin-Mandel convention the shear block
carries a factor 2, so ``M_{44} = 2C_{2323}`` and ``M_{66} = 2C_{1212}``.
That is why the shear coefficients below are read as `M[4,4]/2`.

## Arithmetic here, harmonic there

Introduce the layer average ``\langle\,\cdot\,\rangle = \sum_i f_i (\cdot)_i``
and collect the five quantities the answer is built from:

```@example tutsymlam
avg(g) = f₁ * g(λ₁, μ₁) + f₂ * g(λ₂, μ₂)

A = avg((λ, μ) -> 1 / (λ + 2μ))          # harmonic building block, out of plane
B = avg((λ, μ) -> λ / (λ + 2μ))          # the coupling weight
E = avg((λ, μ) -> 2μ * λ / (λ + 2μ))
F = avg((λ, μ) -> μ)                     # arithmetic mean of the shear moduli
G = avg((λ, μ) -> 1 / μ)                 # harmonic mean of the shear moduli
nothing # hide
```

Each independent coefficient is now one line, and each line is either an
arithmetic average, a harmonic one, or a product of the two:

| coefficient | closed form | nature |
| :--- | :--- | :--- |
| ``C_{1212}`` | ``\langle\mu\rangle`` | **arithmetic** (parallel) |
| ``C_{2323}`` | ``\langle 1/\mu\rangle^{-1}`` | **harmonic** (series) |
| ``C_{3333}`` | ``\big\langle \tfrac{1}{\lambda+2\mu}\big\rangle^{-1}`` | **harmonic** (series) |
| ``C_{1133}`` | ``\big\langle \tfrac{1}{\lambda+2\mu}\big\rangle^{-1}\big\langle\tfrac{\lambda}{\lambda+2\mu}\big\rangle`` | harmonic × arithmetic |
| ``C_{1122}`` | ``\big\langle\tfrac{2\mu\lambda}{\lambda+2\mu}\big\rangle + \big\langle\tfrac{\lambda}{\lambda+2\mu}\big\rangle^{2}\big\langle\tfrac{1}{\lambda+2\mu}\big\rangle^{-1}`` | mixed |
| ``C_{1111}`` | ``C_{1122} + 2\langle\mu\rangle`` | mixed + arithmetic |

Every one of these is an identity, not an approximation — `simplify` returns
exactly zero:

```@example tutsymlam
claims = [
    "C₁₂₁₂ = ⟨μ⟩                      (arithmetic)" => M[6, 6] / 2 - F,
    "C₂₃₂₃ = ⟨1/μ⟩⁻¹                  (harmonic)" => M[4, 4] / 2 - 1 / G,
    "C₃₃₃₃ = ⟨1/(λ+2μ)⟩⁻¹             (harmonic)" => M[3, 3] - 1 / A,
    "C₁₁₃₃ = ⟨1/(λ+2μ)⟩⁻¹⟨λ/(λ+2μ)⟩" => M[1, 3] - B / A,
    "C₁₁₂₂ = ⟨2μλ/(λ+2μ)⟩ + ⟨λ/(λ+2μ)⟩²/⟨1/(λ+2μ)⟩" => M[1, 2] - (E + B^2 / A),
    "C₁₁₁₁ = C₁₁₂₂ + 2⟨μ⟩" => M[1, 1] - (E + B^2 / A + 2F),
]
[k => iszero(simplify(v)) for (k, v) in claims]
```

The last line, ``C_{1111} - C_{1122} = 2\langle\mu\rangle = 2C_{1212}``, is the
transverse-isotropy identity — the in-plane behavior is *entirely* an
arithmetic average.

## The whole matrix in five averages

Substituting the table back, the effective Kelvin-Mandel matrix of an
``N``-layer stack of isotropic layers is

```math
\mathrm{Mat}(\mathbb{C}^{\hom}) =
\begin{pmatrix}
E + \tfrac{B^2}{A} + 2F & E + \tfrac{B^2}{A} & \tfrac{B}{A} & 0 & 0 & 0 \\[2pt]
E + \tfrac{B^2}{A} & E + \tfrac{B^2}{A} + 2F & \tfrac{B}{A} & 0 & 0 & 0 \\[2pt]
\tfrac{B}{A} & \tfrac{B}{A} & \tfrac{1}{A} & 0 & 0 & 0 \\[2pt]
0 & 0 & 0 & \tfrac{2}{G} & 0 & 0 \\[2pt]
0 & 0 & 0 & 0 & \tfrac{2}{G} & 0 \\[2pt]
0 & 0 & 0 & 0 & 0 & 2F
\end{pmatrix},
```

with the five averages above. `A` and `G` enter **inverted** — they are the
harmonic (series, out-of-plane) part; `E` and `F` enter directly — the
arithmetic (parallel, in-plane) part; `B` couples them. Checking it against
the code is one comparison:

```@example tutsymlam
entries = [
    (1, 1) => E + B^2 / A + 2F,
    (1, 2) => E + B^2 / A,
    (1, 3) => B / A,
    (3, 3) => 1 / A,
    (4, 4) => 2 / G,
    (6, 6) => 2F,
]
all(iszero(simplify(M[i, j] - v)) for ((i, j), v) in entries)
```

(The other entries are fixed by symmetry, by ``M_{11} = M_{22}``,
``M_{13} = M_{23}``, ``M_{44} = M_{55}``, and by the zeros of the pattern —
the six above are the independent ones.)

Nothing above used ``N = 2``: with three layers the same five averages, now
over three terms, still describe the answer exactly.

```@example tutsymlam
@syms λ₃::positive μ₃::positive f₂ₛ::positive

lam3 = Laminate(; normal = (0, 0, 1))
add_layer!(lam3, :A, Dict(:C => iso_lame(λ₁, μ₁)); fraction = f₁)
add_layer!(lam3, :B, Dict(:C => iso_lame(λ₂, μ₂)); fraction = f₂ₛ)
add_layer!(lam3, :C, Dict(:C => iso_lame(λ₃, μ₃)); fraction = 1 - f₁ - f₂ₛ)
M3 = KM(homogenize(lam3, Laminated(), :C))

avg3(g) = f₁ * g(λ₁, μ₁) + f₂ₛ * g(λ₂, μ₂) + (1 - f₁ - f₂ₛ) * g(λ₃, μ₃)
A3 = avg3((λ, μ) -> 1 / (λ + 2μ))
F3 = avg3((λ, μ) -> μ)
G3 = avg3((λ, μ) -> 1 / μ)

(C₃₃₃₃ = simplify(M3[3, 3] - 1 / A3),
 C₂₃₂₃ = simplify(M3[4, 4] / 2 - 1 / G3),
 C₁₂₁₂ = simplify(M3[6, 6] / 2 - F3))
```

## The bounds, read off the same matrix

No separate computation is needed to see that the laminate **saturates** its
bounds: the table above already says it. ``C_{1212} = \langle\mu\rangle`` is
the arithmetic mean of the layer shear moduli — the Voigt value — and
``C_{3333} = \langle 1/(\lambda+2\mu)\rangle^{-1}`` is the harmonic mean of
the layer oedometric moduli — the Reuss value. The exact answer therefore
*coincides with a bound* in each of those two directions, simultaneously, and
lies strictly between them everywhere else.

!!! note "Why this one is not checked symbolically here"
    Evaluating `homogenize(lam, Reuss(), :C)` on symbolic moduli means
    inverting a ``6\times6`` matrix that is itself a sum of symbolic inverses;
    the expression swell makes `simplify` impractical, for a statement the
    closed forms above already establish. The identity is checked numerically
    instead, in `test/Laminates/test_laminate_oracles.jl` ("bounds bracket the
    exact answer") and printed by `scripts/33_laminate_basics.jl`.

## Transport: the same two means, one line each

At order 2 the out-of-plane subspace is one-dimensional, so the whole
structure collapses to the two elementary means:

```@example tutsymlam
@syms k₁::positive k₂::positive

lamK = Laminate(; normal = (0, 0, 1))
add_layer!(lamK, :A, Dict(:K => TensISO{3}(k₁)); fraction = f₁)
add_layer!(lamK, :B, Dict(:K => TensISO{3}(k₂)); fraction = f₂)
K = simplify.(components(homogenize(lamK, Laminated(), :K)))
```

```@example tutsymlam
(k_parallel_is_arithmetic = simplify(K[1, 1] - (f₁ * k₁ + f₂ * k₂)),
 k_normal_is_harmonic     = simplify(1 / K[3, 3] - (f₁ / k₁ + f₂ / k₂)))
```

Conduction along the layers is a plain arithmetic average, conduction across
them a plain harmonic one — the two limiting cases the elastic matrix
interpolates between.

## Interfaces, still in closed form

A spring interface of normal compliance ``k_n`` over a period ``L`` adds to the
**harmonic** side and nowhere else, which is the cleanest possible statement of
what a primal interface does:

```@example tutsymlam
@syms kn::positive L::positive

lamI = Laminate(; normal = (0, 0, 1))
add_layer!(lamI, :A, Dict(:C => iso_lame(λ₁, μ₁)); thickness = f₁ * L,
           interface = SpringInterface(kn, zero(kn)))
add_layer!(lamI, :B, Dict(:C => iso_lame(λ₂, μ₂)); thickness = f₂ * L)
MI = KM(homogenize(lamI, Laminated(), :C))

(compliance_adds_to_the_series_law = simplify(1 / MI[3, 3] - (A + kn / L)),
 in_plane_shear_untouched          = simplify(MI[6, 6] / 2 - F))
```

``1/C_{3333} = \langle 1/(\lambda+2\mu)\rangle + k_n/L``: the interface
compliance simply joins the sum of the layer compliances — and the ``1/L``
shows the size effect explicitly, an interface *density*.

## A tilted laminate, still symbolic

The frame is not confined to the canonical one, and not confined to floating
point either. A **symbolic normal** is completed into an orthonormal
``(\underline{\ell}, \underline{m}, \hat{\underline{n}})`` by plain
Gram-Schmidt — no trigonometry and no `atan2`, so the frame stays as readable as
the normal it came from:

```@example tutsymlam
θ = symbols("theta", real = true)

lam_tilt = Laminate(; normal = (0, sin(θ), cos(θ)))
add_layer!(lam_tilt, :A, Dict(:C => iso_lame(λ₁, μ₁)); fraction = f₁)
add_layer!(lam_tilt, :B, Dict(:C => iso_lame(λ₂, μ₂)); fraction = f₂)

C_tilt = homogenize(lam_tilt, Laminated(), :C)
(type = typeof(C_tilt), axis = simplify.(TensND.axis(C_tilt)))
```

The type is still an exact `TensTI`, and the axis is the normal we asked for.
The physics, meanwhile, has not moved at all: the five Walpole coefficients are
**identical** to those of the canonical stack — only the axis they are attached
to has changed. That is frame covariance, stated as an identity rather than
checked to a tolerance:

```@example tutsymlam
C_flat = homogenize(lam, Laminated(), :C)
all(iszero, simplify.(collect(TensND.get_data(C_tilt)) .- collect(TensND.get_data(C_flat))))
```

The in-plane pair is fixed by Gram-Schmidt against a reference axis. Which one is
physically immaterial — the answer above is invariant under rotation about
``\underline{n}`` — but it must not be parallel to ``\underline{n}``. A numeric
normal chooses the safest reference itself, by comparing components; a symbolic
one cannot answer that comparison and falls back to ``\underline{e}_1``, so name
another when the normal may lie along it:

```@example tutsymlam
lam_e1 = Laminate(; normal = (cos(θ), 0, sin(θ)), in_plane = (0, 1, 0))
simplify.(collect(laminate_normal(lam_e1)))
```

ZYZ Euler angles work symbolically too — `Laminate(; euler_angles = (θ, 0, 0))`
describes the same stack.

## Why this works at all

Three implementation choices, invisible numerically, are what let the cell run
symbolically at all — and a page like this one is how they stay honest:

- the pseudo-inverse is the **cofactor inverse** of the ``3\times3``
  out-of-plane block, never `LinearAlgebra.pinv`: an SVD is not symbolically
  evaluable (nor `ForwardDiff`-differentiable);
- every intermediate is an `SMatrix`, never an `MMatrix` —
  `MMatrix{6,6,T}(undef)` cannot even be *constructed* for a non-`isbits`
  element type such as `Sym`;
- the **frame** is read for its axis exactly. A `TensTI` converts its axis to
  the element type of its data and rebuilds its components from the Walpole
  basis of that axis, so a canonical frame read as `(0.0, 0.0, 1.0)` would put a
  symbolic `1.0` in front of every coefficient above. It is read as `(0, 0, 1)`.

The same code therefore serves `Float64` production runs, `ForwardDiff.Dual`
sensitivities and the closed forms above, with no separate symbolic path to
keep in sync. The regression tests in
`test/Laminates/test_laminate_symbolic.jl` run the same identities under both
SymPy and Symbolics.jl.
